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
