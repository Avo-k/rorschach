"""Patricia smoke test: spawn the UCI binary, request MultiPV on a few FENs.

Run:  uv run python scripts/smoke_patricia.py
"""
from __future__ import annotations

import time
from pathlib import Path

import chess
import chess.engine

ENGINE_PATH = Path(__file__).resolve().parent.parent / "bin" / "patricia"
MULTIPV = 5
MOVETIME_MS = 200

POSITIONS = [
    ("starting", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("italian",  "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("kid_main", "r1bq1rk1/pp2ppbp/2np1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQ1RK1 b - - 0 8"),
]


def main() -> None:
    print(f"spawning Patricia at {ENGINE_PATH}")
    engine = chess.engine.SimpleEngine.popen_uci(str(ENGINE_PATH))
    try:
        # MultiPV is auto-managed by python-chess via analyse(multipv=...)
        engine.configure({"Hash": 64, "Threads": 1})

        for label, fen in POSITIONS:
            board = chess.Board(fen)
            t0 = time.perf_counter()
            infos = engine.analyse(
                board,
                limit=chess.engine.Limit(time=MOVETIME_MS / 1000),
                multipv=MULTIPV,
            )
            dt_ms = (time.perf_counter() - t0) * 1000
            print(f"[{label}] {dt_ms:6.1f}ms  ({MULTIPV} PVs @ {MOVETIME_MS}ms target)")
            for i, info in enumerate(infos, 1):
                pv = info.get("pv", [])
                first = pv[0].uci() if pv else "?"
                score = info.get("score")
                depth = info.get("depth", "?")
                print(f"  pv{i}: {first:6s}  score={score}  depth={depth}")
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
