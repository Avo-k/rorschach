"""Estimate Rorschach's effective Elo per profile vs Maia at several ratings.

For each (profile, maia_elo) cell, plays N_GAMES_PER_CELL games with
alternating colors. Score s = (W + 0.5*D) / N → implied Elo diff via the
standard formula:

    Elo_diff = -400 * log10(1/s - 1)

Then the implied Rorschach Elo for that cell is `maia_elo + Elo_diff`.
Average across opp Elos = our point estimate of the profile's strength.

Output: data/bench_elo.json + a Markdown-y table to stdout.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import chess
from dotenv import load_dotenv

from rorschach.bot import rorschach_move
from rorschach.engine import PatriciaEngine
from rorschach.explorer import OpeningExplorer
from rorschach.maia import MaiaPredictor

PROFILES_TO_TEST = ["balanced", "aggressive"]
MAIA_ELOS = [1500, 1700, 1900, 2100]
N_GAMES_PER_CELL = 20
SEED = 42
TIME_MS = 200
MAX_PLIES = 200

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "bench_elo.json"


def implied_elo_diff(score: float) -> float:
    if score >= 0.999:
        return 800.0  # capped — sample size too small to say more
    if score <= 0.001:
        return -800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def play_one(
    engine: PatriciaEngine,
    maia: MaiaPredictor,
    explorer: OpeningExplorer,
    *,
    profile: str,
    rorschach_white: bool,
    rng: random.Random,
    opp_elo: int,
) -> tuple[str, int]:
    board = chess.Board()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < MAX_PLIES:
        is_rorschach = (board.turn == chess.WHITE) == rorschach_white
        if is_rorschach:
            move, _ = rorschach_move(
                board, engine, maia,
                explorer=explorer, profile=profile,
                time_ms=TIME_MS, elo_self=opp_elo, elo_oppo=opp_elo,
            )
        else:
            move, _ = maia.sample(board, opp_elo, opp_elo, rng=rng)
        board.push(move)
        plies += 1
    if board.is_game_over(claim_draw=True):
        return board.result(claim_draw=True), plies
    return "*", plies


def main() -> None:
    load_dotenv()
    print("loading maia2 (blitz) ...")
    maia = MaiaPredictor(type="blitz", device="cpu")
    explorer = OpeningExplorer()
    if not explorer.token:
        print("warning: no LICHESS_TOKEN, explorer disabled")
    print("ok\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cells: dict[tuple[str, int], dict[str, int]] = {}
    t_start = time.perf_counter()

    with PatriciaEngine() as engine:
        for profile in PROFILES_TO_TEST:
            for opp_elo in MAIA_ELOS:
                rng = random.Random(SEED)
                cell = {"W": 0, "D": 0, "L": 0}
                t0 = time.perf_counter()
                for i in range(N_GAMES_PER_CELL):
                    rorschach_white = (i % 2 == 0)
                    result, plies = play_one(
                        engine, maia, explorer,
                        profile=profile, rorschach_white=rorschach_white,
                        rng=rng, opp_elo=opp_elo,
                    )
                    if result == "1-0":
                        outcome = "W" if rorschach_white else "L"
                    elif result == "0-1":
                        outcome = "L" if rorschach_white else "W"
                    else:
                        outcome = "D"
                    cell[outcome] += 1
                cells[(profile, opp_elo)] = cell
                dt = time.perf_counter() - t0
                n = sum(cell.values())
                score = (cell["W"] + 0.5 * cell["D"]) / n
                implied = opp_elo + implied_elo_diff(score)
                eta_min = (time.perf_counter() - t_start) / 60
                print(f"  {profile:11s} vs M{opp_elo}:  {cell['W']:2d}W/{cell['D']:2d}D/{cell['L']:2d}L  "
                      f"score={score:.2f}  implied≈{implied:.0f}  "
                      f"({dt/60:.1f} min, total {eta_min:.1f})")

    OUT_PATH.write_text(json.dumps(
        {f"{p}|{e}": v for (p, e), v in cells.items()}, indent=2,
    ) + "\n")

    print("\n" + "=" * 70)
    print("Estimated Rorschach Elo per profile (derived from scores)")
    print("=" * 70)
    header = f"{'profile':12s}" + "".join(f" {f'vs M{e}':>10s}" for e in MAIA_ELOS) + f" {'avg':>10s}"
    print(header)
    for profile in PROFILES_TO_TEST:
        per_cell_elos: list[float] = []
        line = f"{profile:12s}"
        for opp_elo in MAIA_ELOS:
            cell = cells[(profile, opp_elo)]
            n = sum(cell.values())
            score = (cell["W"] + 0.5 * cell["D"]) / n
            implied = opp_elo + implied_elo_diff(score)
            per_cell_elos.append(implied)
            line += f" {implied:>10.0f}"
        avg = sum(per_cell_elos) / len(per_cell_elos)
        line += f" {avg:>10.0f}"
        print(line)

    print(f"\nraw cells -> {OUT_PATH.relative_to(OUT_PATH.parent.parent.parent)}")


if __name__ == "__main__":
    main()
