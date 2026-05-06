"""Rorschach bot entry point: Patricia + Maia + adaptive selector.

Profiles bundle the three adaptive_delta knobs (dmin, dmax, safe_thresh).
Add a new profile here, not at call sites.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

from rorschach.engine import Candidate, PatriciaEngine
from rorschach.maia import MaiaPredictor
from rorschach.selector import select_adaptive

PROFILES: dict[str, dict[str, int]] = {
    "prudent":    dict(dmin=200, dmax=500,  safe_thresh=200),
    "balanced":   dict(dmin=200, dmax=1000, safe_thresh=200),
    "aggressive": dict(dmin=200, dmax=2000, safe_thresh=200),
    "defensive":  dict(dmin=200, dmax=1000, safe_thresh=500),
}


@dataclass(frozen=True)
class MoveInfo:
    delta_used: int
    best_cp: int
    eval_loss: int
    p_maia: float
    depth: int | None
    n_in_window: int


def rorschach_move(
    board: chess.Board,
    engine: PatriciaEngine,
    maia: MaiaPredictor,
    *,
    profile: str = "balanced",
    time_ms: int = 200,
    k: int = 8,
    elo_self: int = 1900,
    elo_oppo: int = 1900,
) -> tuple[chess.Move, MoveInfo]:
    """Return the move Rorschach plays, plus diagnostics for logging."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {list(PROFILES)}")

    cands: list[Candidate] = engine.multipv_search(board, k=k, time_ms=time_ms)
    maia_probs, _ = maia.predict(board, elo_self, elo_oppo)
    result, delta_used = select_adaptive(cands, maia_probs, **PROFILES[profile])

    best_cp = cands[0].cp
    n_in_window = sum(1 for c in cands if best_cp - c.cp <= delta_used)
    info = MoveInfo(
        delta_used=delta_used,
        best_cp=best_cp,
        eval_loss=result.eval_loss_cp,
        p_maia=result.maia_prob,
        depth=cands[0].depth,
        n_in_window=n_in_window,
    )
    return result.chosen.move, info
