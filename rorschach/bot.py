"""Rorschach bot entry point: Patricia + Maia + adaptive selector.

Profiles bundle the adaptive_delta knobs (dmin, dmax, safe_thresh) and an
optional `narrative_lambda`. When `narrative_lambda > 0` and the predictor
supports it (Maia-3), the selector consumes *two* probability dicts — with
and without game history — and rewards moves that history makes rarer.

Add a new profile here, not at call sites.
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

from rorschach.engine import Candidate, PatriciaEngine
from rorschach.explorer import OpeningExplorer
from rorschach.maia import MaiaPredictor
from rorschach.selector import select_adaptive, select_adaptive_narrative

PROFILES: dict[str, dict[str, float]] = {
    "balanced":   dict(dmin=20, dmax=200, safe_thresh=200, narrative_lambda=0.0),
    "aggressive": dict(dmin=50, dmax=400, safe_thresh=200, narrative_lambda=0.0),
    # Same window as balanced; turns on the narrative-break score (Maia-3 only).
    "narrative":  dict(dmin=20, dmax=200, safe_thresh=200, narrative_lambda=1.0),
}


@dataclass(frozen=True)
class MoveInfo:
    delta_used: int
    best_cp: int
    eval_loss: int
    p_human: float          # P(chosen | with history) when Maia-3 in play, else P(chosen)
    oracle: str             # "E" = Lichess explorer, "M" = Maia, "N" = Maia + narrative break
    depth: int | None
    n_in_window: int
    narrative_break: float | None = None  # P_without − P_with for the chosen move


def rorschach_move(
    board: chess.Board,
    engine: PatriciaEngine,
    maia: MaiaPredictor,
    *,
    explorer: OpeningExplorer | None = None,
    profile: str = "balanced",
    time_ms: int = 200,
    k: int = 8,
    elo_self: int = 1900,
    elo_oppo: int = 1900,
) -> tuple[chess.Move, MoveInfo]:
    """Return the move Rorschach plays, plus diagnostics for logging.

    If `explorer` is provided and has data for the current position, real
    human-move frequencies are used as the alien-detection signal. Otherwise
    falls back to the Maia predictor. When the profile requests it *and*
    Maia-3 is in use, the selector additionally consumes the no-history
    distribution to compute a "narrative break" per candidate.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {list(PROFILES)}")

    knobs = dict(PROFILES[profile])
    narrative_lambda = float(knobs.pop("narrative_lambda", 0.0))

    cands: list[Candidate] = engine.multipv_search(board, k=k, time_ms=time_ms)

    explorer_probs = explorer.predict(board) if explorer is not None else None
    narrative_break: float | None = None

    if explorer_probs is not None:
        # Real-game frequencies dominate when available; no narrative term.
        result, delta_used = select_adaptive(cands, explorer_probs, **knobs)
        oracle = "E"
    elif narrative_lambda > 0 and hasattr(maia, "predict_pair"):
        probs_with, probs_without, _ = maia.predict_pair(board, elo_self, elo_oppo)
        result, delta_used = select_adaptive_narrative(
            cands, probs_with, probs_without, lam=narrative_lambda, **knobs,
        )
        uci = result.chosen.move.uci()
        narrative_break = float(
            probs_without.get(uci, 0.0) - probs_with.get(uci, 0.0)
        )
        oracle = "N"
    else:
        probs, _ = maia.predict(board, elo_self, elo_oppo)
        result, delta_used = select_adaptive(cands, probs, **knobs)
        oracle = "M"

    best_cp = cands[0].cp
    n_in_window = sum(1 for c in cands if best_cp - c.cp <= delta_used)
    info = MoveInfo(
        delta_used=delta_used,
        best_cp=best_cp,
        eval_loss=result.eval_loss_cp,
        p_human=result.maia_prob,
        oracle=oracle,
        depth=cands[0].depth,
        n_in_window=n_in_window,
        narrative_break=narrative_break,
    )
    return result.chosen.move, info
