"""Pure-logic tests for the selector. No engine, no maia."""
from __future__ import annotations

import chess

from rorschach.engine import Candidate
from rorschach.selector import (
    adaptive_delta,
    select,
    select_adaptive,
    select_adaptive_narrative,
    select_narrative,
)


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


def test_mate_bypass_picks_shortest_mate():
    # Patricia returns mate-in-1 first, mate-in-5 second (cp diff = 4)
    cands = [_c("e2g4", 9999), _c("a2a4", 9995), _c("h2h3", 9990)]
    probs = {"e2g4": 0.4, "a2a4": 0.001, "h2h3": 0.001}
    res, delta = select_adaptive(cands, probs)
    assert delta == 0  # mate bypass
    assert res.chosen.move.uci() == "e2g4"
    assert res.eval_loss_cp == 0


def test_mate_bypass_picks_longest_defense_when_losing():
    # We're being mated; cp=-9990 (mate-in-10) is best, -9999 (mate-in-1) worst
    cands = [_c("h7h6", -9990), _c("h7h5", -9995), _c("g7g6", -9999)]
    probs = {"h7h6": 0.5, "h7h5": 0.001, "g7g6": 0.001}
    res, delta = select_adaptive(cands, probs)
    assert delta == 0
    assert res.chosen.move.uci() == "h7h6"  # longest defense


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


# --- narrative-aware selector ----------------------------------------------


def test_narrative_lambda_zero_reduces_to_select():
    # With λ=0, score = P_with — same as plain select(). p_without is ignored.
    cands = [_c("e2e4", 30), _c("d2d4", 20), _c("g1f3", 10)]
    p_with = {"e2e4": 0.6, "d2d4": 0.3, "g1f3": 0.1}
    p_without = {"e2e4": 0.0, "d2d4": 0.0, "g1f3": 0.9}  # would flip g1f3 if used
    res = select_narrative(cands, p_with, p_without, delta_cp=50, lam=0.0)
    assert res.chosen.move.uci() == "g1f3"  # still least likely under p_with


def test_narrative_break_flips_choice():
    # Same P_with → tie under plain select; the move with the larger positive
    # break (P_without > P_with, i.e. history made it rarer) should win.
    cands = [_c("e2e4", 30), _c("d2d4", 25)]
    p_with = {"e2e4": 0.10, "d2d4": 0.10}
    # d2d4 was a 0.50 move pre-history; e2e4 was already a 0.10 move.
    # → break(d2d4) = 0.40, break(e2e4) = 0.00 → narrative picks d2d4.
    p_without = {"e2e4": 0.10, "d2d4": 0.50}
    res = select_narrative(cands, p_with, p_without, delta_cp=50, lam=1.0)
    assert res.chosen.move.uci() == "d2d4"
    # Reported p_human is still P_with (kept comparable with other oracles).
    assert res.maia_prob == 0.10


def test_narrative_reports_p_with_not_score():
    cands = [_c("e2e4", 30)]
    p_with = {"e2e4": 0.2}
    p_without = {"e2e4": 0.7}
    res = select_narrative(cands, p_with, p_without, delta_cp=50, lam=1.0)
    # score = 2*0.2 - 0.7 = -0.3, but maia_prob field shows the actual P_with.
    assert res.maia_prob == 0.2


def test_narrative_missing_probs_treated_as_zero():
    cands = [_c("e2e4", 30), _c("d2d4", 20)]
    p_with = {"e2e4": 0.5}    # d2d4 missing → 0
    p_without = {"e2e4": 0.5}  # break(e2e4)=0, break(d2d4)=0
    res = select_narrative(cands, p_with, p_without, delta_cp=50, lam=1.0)
    assert res.chosen.move.uci() == "d2d4"  # P_with=0 wins on score and ties


def test_select_adaptive_narrative_returns_delta_and_picks_break():
    cands = [_c("e2e4", 600), _c("d2d4", 550)]
    # Both look human under the position alone; d2d4 specifically violates the
    # game's history → larger break, so narrative picks it.
    p_with = {"e2e4": 0.20, "d2d4": 0.20}
    p_without = {"e2e4": 0.20, "d2d4": 0.60}
    res, delta = select_adaptive_narrative(cands, p_with, p_without, lam=1.0)
    assert delta == 600  # same adaptive_delta math as the vanilla variant
    assert res.chosen.move.uci() == "d2d4"


def test_select_adaptive_narrative_mate_bypass():
    cands = [_c("e2g4", 9999), _c("a2a4", 9995)]
    p_with = {"e2g4": 0.4, "a2a4": 0.001}
    p_without = {"e2g4": 0.4, "a2a4": 0.9}  # would otherwise scream "pick a2a4"
    res, delta = select_adaptive_narrative(cands, p_with, p_without, lam=1.0)
    assert delta == 0  # mate bypass: ignores both probability dicts
    assert res.chosen.move.uci() == "e2g4"
