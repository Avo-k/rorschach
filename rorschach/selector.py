"""The selector: pick the least human-likely move within an eval-loss budget.

Pure function — no I/O, no engine state. Easy to unit-test.
"""
from __future__ import annotations

from dataclasses import dataclass

from rorschach.engine import Candidate


@dataclass(frozen=True)
class SelectorResult:
    chosen: Candidate
    eval_loss_cp: int                       # cp lost vs the best candidate
    maia_prob: float                        # Maia probability of the chosen move
    considered: list[tuple[Candidate, float]]  # candidates inside the window, sorted by maia_prob asc


def adaptive_delta(
    eval_cp: int,
    *,
    dmin: int = 200,
    dmax: int = 1000,
    safe_thresh: int = 200,
    slope: float = 1.0,
) -> int:
    """Map the engine's best eval to a Δ budget.

    Δ stays at `dmin` while eval ≤ safe_thresh (we're not comfortably winning).
    Above safe_thresh, Δ grows linearly with the surplus, clamped to dmax.

      eval =   0   -> dmin
      eval = +200  -> dmin (still in the safe zone with default safe_thresh)
      eval = +500  -> dmin + 300 (with slope=1, safe_thresh=200)
      eval = +∞    -> dmax
      eval < 0     -> dmin (losing positions get the defensive baseline)
    """
    surplus = max(0, eval_cp - safe_thresh)
    return min(dmax, max(dmin, dmin + int(slope * surplus)))


def select_adaptive(
    candidates: list[Candidate],
    maia_probs: dict[str, float],
    *,
    dmin: int = 200,
    dmax: int = 1000,
    safe_thresh: int = 200,
    slope: float = 1.0,
) -> tuple["SelectorResult", int]:
    """Run the selector with Δ derived from the engine's best eval.

    Returns (result, delta_used) — delta_used is logged for diagnostics.
    """
    if not candidates:
        raise ValueError("select_adaptive() got no candidates")
    delta = adaptive_delta(
        candidates[0].cp, dmin=dmin, dmax=dmax, safe_thresh=safe_thresh, slope=slope,
    )
    return select(candidates, maia_probs, delta_cp=delta), delta


def select(
    candidates: list[Candidate],
    maia_probs: dict[str, float],
    delta_cp: int,
) -> SelectorResult:
    """argmin_{c ∈ candidates} maia_probs[c]   subject to   best.cp - c.cp ≤ delta_cp.

    `candidates` must be non-empty and sorted best-first (highest cp first).
    Ties on maia_prob are broken by higher cp (preferring less-bad moves).
    """
    if not candidates:
        raise ValueError("select() got no candidates")

    best_cp = candidates[0].cp
    in_window = [c for c in candidates if best_cp - c.cp <= delta_cp]

    annotated = [(c, float(maia_probs.get(c.move.uci(), 0.0))) for c in in_window]
    # primary key: maia prob asc; secondary: cp desc (prefer less-bad on ties)
    annotated_sorted = sorted(annotated, key=lambda cp_p: (cp_p[1], -cp_p[0].cp))

    chosen, prob = annotated_sorted[0]
    return SelectorResult(
        chosen=chosen,
        eval_loss_cp=best_cp - chosen.cp,
        maia_prob=prob,
        considered=annotated_sorted,
    )
