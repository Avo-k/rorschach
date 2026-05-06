"""Main benchmark: predictability + selector behavior (fixed Δ + adaptive profiles).

Saves a rich cache to data/bench_results.jsonl: per-position raw candidates and
the full Maia probability distribution. After this run, any new selector
experiment (different Δ, different adaptive profile) can replay from cache in
seconds without re-running Patricia or Maia.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import chess

from rorschach.bot import PROFILES as ADAPTIVE_PROFILES
from rorschach.engine import PatriciaEngine
from rorschach.maia import MaiaPredictor
from rorschach.selector import select, select_adaptive

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POS_PATH = DATA_DIR / "positions.jsonl"
OUT_PATH = DATA_DIR / "bench_results.jsonl"

DELTAS = [25, 50, 100, 200, 500, 1000]
K_CANDIDATES = 8
TIME_MS = 200
MAIA_TYPE = "blitz"


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = min(int(len(s) * q), len(s) - 1)
    return s[idx]


def phase(ply: int, total: int) -> str:
    if ply < 20: return "early"
    if ply < 40: return "middle"
    return "late"


def main() -> None:
    positions = [json.loads(line) for line in POS_PATH.read_text().splitlines() if line]
    print(f"loaded {len(positions)} positions\n")

    print(f"loading maia2 ({MAIA_TYPE}) ...")
    maia = MaiaPredictor(type=MAIA_TYPE, device="cpu")
    print("ok\n")

    t_start = time.perf_counter()
    results: list[dict] = []

    with PatriciaEngine() as eng:
        for i, pos in enumerate(positions):
            board = chess.Board(pos["fen"])

            maia_probs, _ = maia.predict(board, pos["elo_self"], pos["elo_oppo"])
            played = pos["played_uci"]
            played_prob = float(maia_probs.get(played, 0.0))
            ranked = list(maia_probs.keys())
            played_rank = ranked.index(played) + 1 if played in ranked else len(ranked) + 1

            cands = eng.multipv_search(board, k=K_CANDIDATES, time_ms=TIME_MS)
            best_cp = cands[0].cp if cands else 0

            cands_serial = [
                {
                    "uci": c.move.uci(),
                    "cp": c.cp,
                    "depth": c.depth,
                    "p_maia": float(maia_probs.get(c.move.uci(), 0.0)),
                }
                for c in cands
            ]

            per_delta = {}
            for d in DELTAS:
                window_size = sum(1 for c in cands if best_cp - c.cp <= d)
                res = select(cands, maia_probs, delta_cp=d)
                per_delta[d] = {
                    "window": window_size,
                    "chosen_uci": res.chosen.move.uci(),
                    "eval_loss": res.eval_loss_cp,
                    "p_maia_chosen": res.maia_prob,
                }

            per_profile = {}
            for name, params in ADAPTIVE_PROFILES.items():
                res, delta_used = select_adaptive(cands, maia_probs, **params)
                per_profile[name] = {
                    "delta_used": delta_used,
                    "window": sum(1 for c in cands if best_cp - c.cp <= delta_used),
                    "chosen_uci": res.chosen.move.uci(),
                    "eval_loss": res.eval_loss_cp,
                    "p_maia_chosen": res.maia_prob,
                }

            row = {
                "user": pos["user"],
                "color": pos["color"],
                "phase": phase(pos["ply"], pos["total_plies"]),
                "elo_self": pos["elo_self"],
                "elo_oppo": pos["elo_oppo"],
                "played_uci": played,
                "played_prob": played_prob,
                "played_rank": played_rank,
                "n_legal": len(maia_probs),
                "patricia_depth": cands[0].depth if cands else None,
                "best_cp": best_cp,
                "candidates": cands_serial,
                "maia_probs": {uci: float(p) for uci, p in maia_probs.items()},
                "per_delta": per_delta,
                "per_profile": per_profile,
            }
            results.append(row)

            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.perf_counter() - t_start)
                eta = (len(positions) - i - 1) / rate
                print(f"  {i+1:4d}/{len(positions)}   {rate:.1f} pos/s   eta {eta/60:.1f} min")

    elapsed = time.perf_counter() - t_start
    print(f"\nfinished in {elapsed/60:.1f} min ({len(results)/elapsed:.2f} pos/s)\n")

    OUT_PATH.write_text("\n".join(json.dumps(r) for r in results) + "\n")
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"raw results -> {OUT_PATH.relative_to(DATA_DIR.parent)} ({size_mb:.1f} MB)\n")

    print_predictability(results)
    print_fixed_delta(results)
    print_adaptive_profiles(results)
    print_adaptive_by_eval_band(results)


def print_predictability(results: list[dict]) -> None:
    print("=" * 80)
    print("PREDICTABILITY (Maia at player's actual rating)")
    print("=" * 80)
    print(f"{'user':12s} {'N':>5s} {'avg P':>7s} {'median P':>9s} {'top1':>6s} {'top3':>6s} {'top5':>6s}")
    by_user = defaultdict(list)
    for r in results:
        by_user[r["user"]].append(r)
    for user, rows in by_user.items():
        probs = [r["played_prob"] for r in rows]
        ranks = [r["played_rank"] for r in rows]
        top1 = sum(1 for r in ranks if r == 1) / len(ranks)
        top3 = sum(1 for r in ranks if r <= 3) / len(ranks)
        top5 = sum(1 for r in ranks if r <= 5) / len(ranks)
        print(f"{user:12s} {len(rows):>5d} {mean(probs):>7.3f} {median(probs):>9.3f} "
              f"{top1:>6.1%} {top3:>6.1%} {top5:>6.1%}")


def print_fixed_delta(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("SELECTOR BEHAVIOR per FIXED Δ")
    print("=" * 80)
    print(f"{'Δ':>6s}  {'avg win':>8s}  {'avg loss':>9s}  "
          f"{'avg P_maia':>11s}  {'P_maia P50':>11s}  {'P_maia P90':>11s}  {'≠ best':>7s}")
    for d in DELTAS:
        wins = [r["per_delta"][d]["window"] for r in results]
        losses = [r["per_delta"][d]["eval_loss"] for r in results]
        probs = [r["per_delta"][d]["p_maia_chosen"] for r in results]
        diff = sum(1 for r in results if r["per_delta"][d]["eval_loss"] > 0)
        print(f"{d:>6d}  {mean(wins):>8.2f}  {mean(losses):>9.1f}  "
              f"{mean(probs):>11.3f}  {percentile(probs, 0.5):>11.3f}  "
              f"{percentile(probs, 0.9):>11.3f}  {diff/len(results):>7.1%}")


def print_adaptive_profiles(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("ADAPTIVE PROFILES (Δ derived from Patricia's best eval)")
    print("=" * 80)
    print(f"{'profile':12s}  {'avg Δ':>6s}  {'avg loss':>8s}  "
          f"{'avg P_maia':>10s}  {'P50':>7s}  {'P90':>7s}  {'≠ best':>7s}")
    for name in ADAPTIVE_PROFILES:
        deltas = [r["per_profile"][name]["delta_used"] for r in results]
        losses = [r["per_profile"][name]["eval_loss"] for r in results]
        probs = [r["per_profile"][name]["p_maia_chosen"] for r in results]
        diff = sum(1 for r in results if r["per_profile"][name]["eval_loss"] > 0)
        print(f"{name:12s}  {mean(deltas):>6.0f}  {mean(losses):>8.1f}  "
              f"{mean(probs):>10.3f}  {percentile(probs, 0.5):>7.3f}  "
              f"{percentile(probs, 0.9):>7.3f}  {diff/len(results):>7.1%}")


def print_adaptive_by_eval_band(results: list[dict]) -> None:
    """How adaptive Δ behaves across eval-bands. Reveals when the bot 'opens up'."""
    print("\n" + "=" * 80)
    print("ADAPTIVE 'balanced' BREAKDOWN BY eval BAND")
    print("=" * 80)
    bands = [
        ("losing      (e<-200)",  lambda e: e < -200),
        ("balanced    (-200..200)", lambda e: -200 <= e <= 200),
        ("slight adv  (200..500)", lambda e: 200 < e <= 500),
        ("comfortable (500..1000)", lambda e: 500 < e <= 1000),
        ("winning     (>1000)",  lambda e: e > 1000),
    ]
    print(f"{'band':28s}  {'N':>5s}  {'avg Δ':>6s}  {'avg loss':>8s}  {'P_maia P50':>11s}  {'P_maia P90':>11s}")
    for label, pred in bands:
        rows = [r for r in results if pred(r["best_cp"])]
        if not rows:
            print(f"{label:28s}  {0:>5d}  -")
            continue
        deltas = [r["per_profile"]["balanced"]["delta_used"] for r in rows]
        losses = [r["per_profile"]["balanced"]["eval_loss"] for r in rows]
        probs = [r["per_profile"]["balanced"]["p_maia_chosen"] for r in rows]
        print(f"{label:28s}  {len(rows):>5d}  {mean(deltas):>6.0f}  {mean(losses):>8.1f}  "
              f"{percentile(probs, 0.5):>11.3f}  {percentile(probs, 0.9):>11.3f}")


if __name__ == "__main__":
    main()
