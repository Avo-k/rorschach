"""Maia-2 smoke test: load model, run inference on a few FENs, print latency.

Run:  uv run python scripts/smoke_maia.py
"""
from __future__ import annotations

import time

from maia2 import inference, model

POSITIONS = [
    ("starting", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("italian",  "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("kid_main", "r1bq1rk1/pp2ppbp/2np1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQ1RK1 b - - 0 8"),
]
ELO_SELF = 1900
ELO_OPPO = 1900


def main() -> None:
    print(f"loading maia2 (rapid, cpu) ...")
    t0 = time.perf_counter()
    m = model.from_pretrained(type="rapid", device="cpu")
    prepared = inference.prepare()
    print(f"  loaded in {time.perf_counter() - t0:.2f}s")

    # warmup (first call includes JIT/cudnn-equiv setup)
    inference.inference_each(m, prepared, POSITIONS[0][1], ELO_SELF, ELO_OPPO)

    for label, fen in POSITIONS:
        t0 = time.perf_counter()
        move_probs, win_prob = inference.inference_each(
            m, prepared, fen, ELO_SELF, ELO_OPPO,
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        top5 = list(move_probs.items())[:5]
        top5_str = ", ".join(f"{uci}={p:.3f}" for uci, p in top5)
        print(f"[{label}] {dt_ms:6.1f}ms  win={win_prob:.3f}  top5: {top5_str}")


if __name__ == "__main__":
    main()
