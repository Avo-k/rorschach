"""Pure-logic tests for the selector. No engine, no maia."""
from __future__ import annotations

import chess

from rorschach.engine import Candidate
from rorschach.selector import select


def _c(uci: str, cp: int) -> Candidate:
    return Candidate(move=chess.Move.from_uci(uci), cp=cp)


def test_picks_least_human_within_window():
    cands = [_c("e2e4", 30), _c("d2d4", 20), _c("g1f3", 10)]
    probs = {"e2e4": 0.6, "d2d4": 0.3, "g1f3": 0.1}
    res = select(cands, probs, delta_cp=50)
    assert res.chosen.move.uci() == "g1f3"
    assert res.eval_loss_cp == 20
    assert res.maia_prob == 0.1


def test_window_excludes_too_bad_moves():
    cands = [_c("e2e4", 30), _c("a2a3", -200)]
    probs = {"e2e4": 0.5, "a2a3": 0.0}
    res = select(cands, probs, delta_cp=50)
    assert res.chosen.move.uci() == "e2e4"  # a2a3 is outside the window


def test_tie_break_prefers_higher_cp():
    cands = [_c("e2e4", 30), _c("d2d4", 20), _c("c2c4", 10)]
    probs = {"e2e4": 0.0, "d2d4": 0.0, "c2c4": 0.0}
    res = select(cands, probs, delta_cp=50)
    assert res.chosen.move.uci() == "e2e4"  # highest cp on ties


def test_missing_maia_prob_treated_as_zero():
    cands = [_c("e2e4", 30), _c("d2d4", 20)]
    probs = {"e2e4": 0.7}  # d2d4 missing
    res = select(cands, probs, delta_cp=50)
    assert res.chosen.move.uci() == "d2d4"
    assert res.maia_prob == 0.0
