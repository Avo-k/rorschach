"""Play N games of Rorschach vs Maia, with Maia sampling from its distribution.

Output: data/games_vs_maia.pgn — annotated with per-move diagnostics so the
position can be reviewed at a glance.
"""
from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path

import chess
import chess.pgn

from rorschach.bot import rorschach_move
from rorschach.engine import PatriciaEngine
from rorschach.maia import MaiaPredictor

PROFILE_PLAN = [("balanced", 5), ("aggressive", 5)]
ELO_SELF = 1900
ELO_OPPO = 1900
SEED = 42  # reset per profile so each block sees identical Maia sampling
TIME_MS = 200
MAX_PLIES = 200
MAIA_TYPE = "blitz"

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "games_vs_maia.pgn"


def play_one(
    engine: PatriciaEngine,
    maia: MaiaPredictor,
    *,
    rorschach_white: bool,
    rng: random.Random,
    profile: str,
) -> tuple[list[tuple[chess.Move, str]], str]:
    board = chess.Board()
    moves: list[tuple[chess.Move, str]] = []
    while not board.is_game_over(claim_draw=True) and len(moves) < MAX_PLIES:
        is_rorschach = (board.turn == chess.WHITE) == rorschach_white
        if is_rorschach:
            move, info = rorschach_move(
                board, engine, maia,
                profile=profile, time_ms=TIME_MS,
                elo_self=ELO_SELF, elo_oppo=ELO_OPPO,
            )
            comment = (
                f"R Δ={info.delta_used} cp={info.best_cp:+d} "
                f"loss={info.eval_loss} P={info.p_maia:.3f} "
                f"win={info.n_in_window} d{info.depth}"
            )
        else:
            move, p = maia.sample(board, ELO_SELF, ELO_OPPO, rng=rng)
            comment = f"M P={p:.3f}"
        moves.append((move, comment))
        board.push(move)

    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
    else:
        result = "*"  # hit MAX_PLIES
    return moves, result


def main() -> None:
    print(f"loading maia2 ({MAIA_TYPE}, elo={ELO_SELF}) ...")
    maia = MaiaPredictor(type=MAIA_TYPE, device="cpu")
    print("ok\n")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scores: dict[str, dict[str, int]] = {}
    t_start = time.perf_counter()
    round_no = 0

    with PatriciaEngine() as engine, OUT_PATH.open("w") as f:
        for profile, n_games in PROFILE_PLAN:
            rng = random.Random(SEED)
            score = {"R": 0, "M": 0, "D": 0}
            print(f"=== profile={profile}  ({n_games} games) ===")
            for i in range(n_games):
                round_no += 1
                rorschach_white = (i % 2 == 0)
                t0 = time.perf_counter()
                moves, result = play_one(
                    engine, maia,
                    rorschach_white=rorschach_white, rng=rng, profile=profile,
                )
                dt = time.perf_counter() - t0

                game = chess.pgn.Game()
                r_label = f"Rorschach({profile})"
                game.headers["Event"] = f"Rorschach vs Maia ({profile})"
                game.headers["Site"] = "local"
                game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
                game.headers["Round"] = str(round_no)
                game.headers["White"] = r_label if rorschach_white else f"Maia2-{ELO_SELF}"
                game.headers["Black"] = f"Maia2-{ELO_SELF}" if rorschach_white else r_label
                game.headers["Result"] = result
                game.headers["TimeControl"] = f"0+{TIME_MS/1000:.2f}"

                node = game
                for move, comment in moves:
                    node = node.add_main_variation(move, comment=comment)

                f.write(str(game) + "\n\n")
                f.flush()

                if result == "1-0":
                    outcome = "R" if rorschach_white else "M"
                elif result == "0-1":
                    outcome = "M" if rorschach_white else "R"
                else:
                    outcome = "D"
                score[outcome] += 1

                color = "W" if rorschach_white else "B"
                print(f"  game {round_no:2d} [{profile:10s}]  R={color}  plies={len(moves):3d}  "
                      f"result={result:5s}  outcome={outcome}  ({dt:.1f}s)")
            scores[profile] = score
            print()

    elapsed = time.perf_counter() - t_start
    print(f"finished in {elapsed:.1f}s\n")
    for profile, s in scores.items():
        print(f"  {profile:10s}:  {s['R']}W / {s['D']}D / {s['M']}L")
    print(f"\nPGN -> {OUT_PATH.relative_to(OUT_PATH.parent.parent.parent)}")


if __name__ == "__main__":
    main()
