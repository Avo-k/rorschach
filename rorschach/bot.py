"""Rorschach bot entry point: Patricia + Maia + adaptive selector.

Profiles bundle the adaptive_delta knobs (dmin, dmax, safe_thresh) and an
optional `narrative_lambda`. When `narrative_lambda > 0` and the predictor
supports it (Maia-3), the selector consumes *two* probability dicts — with
and without game history — and rewards moves that history makes rarer.

Add a new profile here, not at call sites.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import chess

from rorschach.engine import Candidate, PatriciaEngine
from rorschach.explorer import OpeningExplorer
from rorschach.maia import Maia3Predictor
from rorschach.selector import select_adaptive, select_adaptive_narrative

PROFILES: dict[str, dict[str, float]] = {
    "balanced":   dict(dmin=50, dmax=300, safe_thresh=200, narrative_lambda=0.0),
    "aggressive": dict(dmin=80, dmax=500, safe_thresh=200, narrative_lambda=0.0),
    # Same window as balanced; turns on the narrative-break score (Maia-3 only).
    "narrative":  dict(dmin=50, dmax=300, safe_thresh=200, narrative_lambda=1.0),
}


MAX_VERIFICATIONS = 2


def verify_chosen(
    *,
    cands: list[Candidate],
    result,
    delta_used: int,
    verify: Callable[[chess.Move], int | None],
) -> tuple[Candidate, int, int | None, int]:
    """Re-evaluate the selector's pick at deeper depth; fall back if it blunders.

    A wide MultiPV scan at low movetime sometimes under-estimates the true
    eval loss of alien candidates (tactical mirages the engine couldn't see
    deep enough). For each candidate in `result.considered` (already sorted by
    selector preference), call `verify(move)` to get a single-PV cp. Accept
    the first one whose true loss `cands[0].cp - verified_cp` is within
    `delta_used`; fall back to the engine's top-1 if none survives or the
    verification budget (MAX_VERIFICATIONS attempts) is exhausted.

    Returns: (chosen_candidate, eval_loss_cp_from_multipv, verified_loss, n_verified).
    `verified_loss` is None when no verification happened (mate bypass or the
    selector already picked top-1).
    """
    best_cp = cands[0].cp

    # Skip verification entirely when nothing can change the played move:
    # mate bypass (delta_used=0) or the selector already picked the engine's
    # top-1 (its cp is the anchor, so eval_loss is trivially 0).
    if delta_used == 0 or result.chosen.move == cands[0].move:
        return result.chosen, result.eval_loss_cp, None, 0

    n_verified = 0
    for cand, _ in result.considered:
        if cand.move == cands[0].move:
            # Fallback to engine top-1 reached the head of the considered
            # list: accept without spending a verification call.
            return cand, 0, 0, n_verified

        n_verified += 1
        v_cp = verify(cand.move)
        if v_cp is not None:
            true_loss = best_cp - v_cp
            if true_loss <= delta_used:
                # MultiPV-relative loss for logging stays comparable across
                # turns; verified_loss is the deeper-depth estimate.
                return cand, best_cp - cand.cp, true_loss, n_verified

        if n_verified >= MAX_VERIFICATIONS:
            break

    # No candidate survived (verifications failed or budget exhausted). Play
    # the engine's top-1, which by definition has eval_loss = 0.
    return cands[0], 0, 0, n_verified


@dataclass(frozen=True)
class MoveInfo:
    delta_used: int
    best_cp: int
    eval_loss: int          # eval loss reported by the MultiPV scan
    p_human: float          # P(chosen | with history) when Maia-3 in play, else P(chosen)
    oracle: str             # "E" = Lichess explorer, "M" = Maia, "N" = Maia + narrative break
    depth: int | None
    n_in_window: int
    narrative_break: float | None = None  # P_without − P_with for the chosen move
    verified_loss: int | None = None      # eval loss after single-PV verification (None = not verified)
    n_verified: int = 0                   # how many candidates needed verification (1 = chosen passed)


def rorschach_move(
    board: chess.Board,
    engine: PatriciaEngine,
    maia: Maia3Predictor,
    *,
    explorer: OpeningExplorer | None = None,
    profile: str = "balanced",
    time_ms: int = 400,
    k: int = 5,
    elo_self: int = 1900,
    elo_oppo: int = 1900,
    verify_fraction: float = 0.4,
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

    # Split the move budget: MultiPV scan dominates, leaves room for up to two
    # single-PV verifications of the chosen candidate (chosen + one fallback).
    verify_fraction = max(0.0, min(0.9, verify_fraction))
    multipv_ms = max(50, int(time_ms * (1.0 - verify_fraction)))
    verify_ms = max(40, int(time_ms * verify_fraction / 2))

    cands: list[Candidate] = engine.multipv_search(board, k=k, time_ms=multipv_ms)

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

    chosen_cand, chosen_loss, verified_loss, n_verified = verify_chosen(
        cands=cands,
        result=result,
        delta_used=delta_used,
        verify=lambda move: engine.quick_eval_after_move(board, move, verify_ms),
    )

    # `p_human` and `narrative_break` describe the *originally-chosen* move;
    # if verification forced a fallback, `n_verified > 1` flags that the
    # played move differs from the one those numbers describe.
    info = MoveInfo(
        delta_used=delta_used,
        best_cp=best_cp,
        eval_loss=chosen_loss,
        p_human=result.maia_prob,
        oracle=oracle,
        depth=cands[0].depth,
        n_in_window=n_in_window,
        narrative_break=narrative_break,
        verified_loss=verified_loss,
        n_verified=n_verified,
    )
    return chosen_cand.move, info
