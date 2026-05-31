"""Calibrate the talk-offer probability so the bot *actually* speaks ~10%.

The model can answer PASS, so the effective speak rate is:

    P(speak) = offer_rate * (1 - PASS_rate)

We want P(speak) ~= 0.10. This script samples ~180 positions at random from
20 real human games, builds the same (future-production) prompt the live bot
will use — opponent-move predictability + running cumulative score + a
"what's notable" hint — asks deepseek-v4-flash on every one, and measures the
PASS rate over representative positions. From that it recommends an offer_rate.

Run:
    .venv/bin/python experiments/talk_rate_calibration.py
"""
from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chat_model_bakeoff as bo  # noqa: E402  (reuse fetch/build/call/classify)
from chat_model_bakeoff import chat  # noqa: E402

MODEL = "deepseek/deepseek-v4-flash"
N_GAMES = 20
N_SAMPLES = 180          # random positions across all games
TIME_MS = 250
TARGET = 0.10            # desired effective speak rate
AVG_OUR_MOVES = 30       # rough Rorschach moves/game, for cost extrapolation
OUT = Path(__file__).resolve().parent / "talk_rate_calibration.md"


def precompute_opp(maia, game: dict) -> tuple[list, list]:
    """For each ply, Maia's prob + rank of the opponent's move (None elsewhere)."""
    rcolor = chess.WHITE if game["_bot_white"] else chess.BLACK
    sans = game["moves"].split()
    opp_p: list = [None] * len(sans)
    opp_rank: list = [None] * len(sans)
    b = chess.Board()
    for i, san in enumerate(sans):
        try:
            mv = b.parse_san(san)
        except ValueError:
            break
        if b.turn != rcolor:
            try:
                probs, _ = maia.predict(b, bo.ELO, bo.ELO)
                p = float(probs.get(mv.uci(), 0.0))
                opp_p[i] = p
                opp_rank[i] = 1 + sum(1 for v in probs.values() if v > p)
            except Exception:
                pass
        b.push(mv)
    return opp_p, opp_rank


def main() -> None:
    import os
    if not os.environ.get("LICHESS_TOKEN") or not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("need LICHESS_TOKEN and OPENROUTER_API_KEY in .env")

    random.seed(0)
    print(f"fetching {N_GAMES} human games ...")
    games = bo.fetch_human_games(os.environ["LICHESS_TOKEN"], N_GAMES)
    print(f"  {len(games)} games")

    # Pool every Rorschach-to-move position, then sample uniformly (per-move).
    pool = []
    for gi, g in enumerate(games):
        rcolor = chess.WHITE if g["_bot_white"] else chess.BLACK
        b = chess.Board()
        for i, san in enumerate(g["moves"].split()):
            if b.turn == rcolor and i >= 6:
                pool.append((gi, i, b.copy()))
            try:
                b.push_san(san)
            except ValueError:
                break
    sample = random.sample(pool, min(N_SAMPLES, len(pool)))
    print(f"  pooled {len(pool)} positions, sampling {len(sample)}")

    print("loading Patricia + Maia (5m) + explorer ...")
    engine = bo.PatriciaEngine()
    maia = bo.make_predictor("maia3-5m", device="cpu")
    explorer = bo.OpeningExplorer()

    # Maia opponent-move arrays, once per game that has a sampled position.
    needed = {gi for gi, _, _ in sample}
    print(f"precomputing opponent Maia passes for {len(needed)} games ...")
    opp_cache = {gi: precompute_opp(maia, games[gi]) for gi in needed}

    print(f"evaluating {len(sample)} positions with {MODEL} ...")
    results = []  # (bucket, passed, cost)
    for k, (gi, ply, board) in enumerate(sample, 1):
        g = games[gi]
        opp_p, opp_rank = opp_cache[gi]
        hp = opp_p[ply - 1] if ply - 1 < len(opp_p) and opp_p[ply - 1] is not None else 0.0
        rank = opp_rank[ply - 1] if ply - 1 < len(opp_rank) and opp_rank[ply - 1] else 1
        prior = [p for p in opp_p[:ply] if p is not None]
        cum = (f"Across the game so far, your opponent's moves have averaged "
               f"{sum(prior)/len(prior)*100:.0f}% human-typical over {len(prior)} "
               f"moves (higher = more predictable)." if prior else None)

        move, info = bo.rorschach_move(
            board, engine, maia, explorer=explorer,
            profile="balanced", time_ms=TIME_MS, elo_self=bo.ELO, elo_oppo=bo.ELO,
        )
        bucket, note = bo.classify(info.p_human, hp)
        item = {
            "fullmove": board.fullmove_number,
            "color": "White" if board.turn else "Black",
            "opening": g["_opening"], "pgn": chat._pgn_so_far(board),
            "opp_san": chat._opponent_last_move_san(board) or "?",
            "hp": hp, "hp_rank": rank, "hp_label": bo.label_for(hp),
            "our_san": board.san(move), "our_p": info.p_human,
            "loss": info.eval_loss, "delta": info.delta_used,
            "oracle": info.oracle, "eval": info.best_cp / 100.0,
            "cumulative": cum, "note": note, "language": "English",
        }
        res = bo.call_model(MODEL, bo.build_prompt(item))
        results.append((bucket, bool(res.get("passed")), float(res.get("cost", 0) or 0)))
        if k % 20 == 0:
            spoke = sum(1 for _, p, _ in results if not p)
            print(f"  {k}/{len(sample)}  spoke {spoke}  pass_rate "
                  f"{1 - spoke/len(results):.2f}")
    engine.quit()

    # --- aggregate -----------------------------------------------------------
    n = len(results)
    spoke = sum(1 for _, p, _ in results if not p)
    p_speak = spoke / n
    pass_rate = 1 - p_speak
    rec_offer = min(1.0, TARGET / p_speak) if p_speak > 0 else 1.0
    avg_cost = sum(c for _, _, c in results) / n

    by_bucket = defaultdict(lambda: [0, 0])  # bucket -> [count, spoke]
    for bucket, passed, _ in results:
        by_bucket[bucket][0] += 1
        if not passed:
            by_bucket[bucket][1] += 1

    L = ["# Talk-rate calibration\n",
         f"Model **{MODEL}**, {n} random positions from {len(games)} human games. "
         f"Every position was *offered* (forced an LLM call) to measure the PASS "
         f"rate.\n",
         f"- **PASS rate over random positions:** {pass_rate*100:.0f}% "
         f"(model spoke on {p_speak*100:.0f}%).",
         f"- To hit an effective **{TARGET*100:.0f}%** speak rate with a flat "
         f"random offer: **offer_rate = {rec_offer:.2f}** "
         f"(= {TARGET:.2f} / {p_speak:.2f}).",
         f"- A flat 0.10 offer would instead yield ~{0.10*p_speak*100:.1f}% "
         f"effective speak.\n",
         "## PASS rate by position type\n",
         "| bucket | positions | spoke | speak rate |",
         "|---|--:|--:|--:|"]
    for bucket in ("alien_self", "predictable_human", "surprising_human", "quiet"):
        cnt, spk = by_bucket.get(bucket, [0, 0])
        if cnt:
            L.append(f"| {bucket} | {cnt} | {spk} | {spk/cnt*100:.0f}% |")

    calls_per_game = rec_offer * AVG_OUR_MOVES
    L += ["\n## Cost at the recommended offer rate\n",
          f"Offer {rec_offer:.2f} × ~{AVG_OUR_MOVES} of our moves/game = "
          f"~{calls_per_game:.0f} LLM calls/game (≈{calls_per_game*p_speak:.0f} "
          f"spoken). Avg ${avg_cost:.4f}/call.\n",
          f"- per game: ${avg_cost*calls_per_game:.4f}",
          f"- 10 games: ${avg_cost*calls_per_game*10:.3f}",
          f"- 100 games: ${avg_cost*calls_per_game*100:.2f}\n",
          f"_Recommendation: set `BASE_PROB = {rec_offer:.2f}` in chat.py "
          f"(VERBOSE_PROB scales similarly)._"]

    OUT.write_text("\n".join(L))
    print("\n" + "\n".join(L[2:8]))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
