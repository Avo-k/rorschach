"""Selector exploration: sweep (Δ, elo) on a few FENs and print what we'd play.

Run:  uv run python scripts/explore.py
"""
from __future__ import annotations

import chess

from rorschach.engine import PatriciaEngine
from rorschach.maia import MaiaPredictor
from rorschach.selector import select

POSITIONS = [
    ("starting", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("italian",  "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("kid_main", "r1bq1rk1/pp2ppbp/2np1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQ1RK1 b - - 0 8"),
]

K = 8
TIME_MS = 200
DELTAS = (25, 50, 100, 200)
ELOS = (1500, 1900, 2200)


def main() -> None:
    print("loading maia2 ...")
    maia = MaiaPredictor(type="rapid", device="cpu")
    print("ok\n")

    with PatriciaEngine() as eng:
        for label, fen in POSITIONS:
            board = chess.Board(fen)
            print(f"=== {label}  ({'White' if board.turn else 'Black'} to move) ===")

            cands = eng.multipv_search(board, k=K, time_ms=TIME_MS)
            best_cp = cands[0].cp

            for elo in ELOS:
                maia_probs, win = maia.predict(board, elo_self=elo, elo_oppo=elo)
                print(f"\n  elo={elo}  (maia win-prob for stm = {win:.2f})")
                print(f"  Patricia top-{K} @ {TIME_MS}ms (depth ~{cands[0].depth}):")
                for c in cands:
                    p = maia_probs.get(c.move.uci(), 0.0)
                    loss = best_cp - c.cp
                    print(f"    {c.move.uci():6s}  cp={c.cp:+5d}  loss={loss:4d}  P_maia={p:.3f}")

                print(f"  selector picks per Δ:")
                for d in DELTAS:
                    res = select(cands, maia_probs, delta_cp=d)
                    print(
                        f"    Δ={d:3d}: {res.chosen.move.uci():6s}  "
                        f"loss={res.eval_loss_cp:3d}  P_maia={res.maia_prob:.3f}"
                    )
            print()


if __name__ == "__main__":
    main()
