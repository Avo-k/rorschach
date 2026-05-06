"""Pure-logic tests for the selector. No engine, no maia."""
from __future__ import annotations

import chess

from rorschach.engine import Candidate
from rorschach.selector import adaptive_delta, select, select_adaptive


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


def test_adaptive_delta_safe_zone():
    # eval ≤ safe_thresh -> Δ = dmin
    assert adaptive_delta(-500) == 200
    assert adaptive_delta(0) == 200
    assert adaptive_delta(200) == 200


def test_adaptive_delta_linear_zone():
    assert adaptive_delta(500) == 500   # 200 + (500-200)*1
    assert adaptive_delta(700) == 700
    assert adaptive_delta(900) == 900


def test_adaptive_delta_saturation():
    assert adaptive_delta(1000) == 1000
    assert adaptive_delta(5000) == 1000
    assert adaptive_delta(9999) == 1000  # mate-in-1 cp encoding


def test_adaptive_delta_custom_max():
    assert adaptive_delta(2200, dmax=2000) == 2000
    assert adaptive_delta(1500, dmax=2000) == 1500


def test_adaptive_delta_custom_safe_thresh():
    assert adaptive_delta(500, safe_thresh=500) == 200
    assert adaptive_delta(1000, safe_thresh=500) == 700


def test_select_adaptive_returns_delta():
    cands = [_c("e2e4", 600), _c("d2d4", 550), _c("g1f3", 100)]
    probs = {"e2e4": 0.5, "d2d4": 0.05, "g1f3": 0.0}
    # eval=600, default safe_thresh=200, slope=1 -> delta = min(1000, 200+400) = 600
    # window: all three (losses 0/50/500, all ≤ 600); picks g1f3 (lowest maia)
    res, delta = select_adaptive(cands, probs)
    assert delta == 600
    assert res.chosen.move.uci() == "g1f3"

    # With dmax=300, delta = min(300, 600) = 300; g1f3 (loss=500) is excluded
    res2, delta2 = select_adaptive(cands, probs, dmax=300)
    assert delta2 == 300
    assert res2.chosen.move.uci() == "d2d4"
