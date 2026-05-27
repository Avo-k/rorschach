"""Verification-fallback logic. Pure-function tests, no engine, no maia."""
from __future__ import annotations

import chess

from rorschach.bot import MAX_VERIFICATIONS, verify_chosen
from rorschach.engine import Candidate
from rorschach.selector import SelectorResult


def _c(uci: str, cp: int) -> Candidate:
    return Candidate(move=chess.Move.from_uci(uci), cp=cp)


def _result(chosen: Candidate, considered: list[tuple[Candidate, float]], best_cp: int) -> SelectorResult:
    return SelectorResult(
        chosen=chosen,
        eval_loss_cp=best_cp - chosen.cp,
        maia_prob=considered[0][1] if considered else 0.0,
        considered=considered,
    )


def test_skip_verification_for_mate_bypass():
    best = _c("e2g4", 9999)
    res = _result(best, [(best, 0.4)], best_cp=9999)
    chosen, loss, v_loss, n = verify_chosen(
        cands=[best],
        result=res,
        delta_used=0,
        verify=lambda m: (_ for _ in ()).throw(AssertionError("must not verify")),
    )
    assert chosen.move == best.move
    assert v_loss is None
    assert n == 0


def test_skip_verification_when_chosen_is_top1():
    best = _c("e2e4", 30)
    res = _result(best, [(best, 0.1)], best_cp=30)
    chosen, loss, v_loss, n = verify_chosen(
        cands=[best, _c("d2d4", 20)],
        result=res,
        delta_used=200,
        verify=lambda m: (_ for _ in ()).throw(AssertionError("must not verify")),
    )
    assert chosen.move == best.move
    assert loss == 0
    assert v_loss is None
    assert n == 0


def test_chosen_passes_verification():
    best = _c("e2e4", 30)
    alien = _c("h2h4", -20)
    # Selector picked the alien move; verification confirms it's only 50cp worse.
    res = _result(alien, [(alien, 0.05), (best, 0.6)], best_cp=30)
    chosen, loss, v_loss, n = verify_chosen(
        cands=[best, alien],
        result=res,
        delta_used=200,
        verify=lambda m: -20,  # verified cp matches MultiPV → true_loss = 50
    )
    assert chosen.move == alien.move
    assert loss == 50               # MultiPV-relative loss
    assert v_loss == 50             # verified loss
    assert n == 1                   # one verification done


def test_fallback_when_chosen_fails_verification():
    best = _c("e2e4", 30)
    mid = _c("d2d4", 20)
    blunder = _c("h2h4", 10)        # MultiPV says ~ok, verification reveals -500
    # Selector ordered candidates by maia_prob asc: blunder first, mid next, best last.
    res = _result(blunder, [(blunder, 0.01), (mid, 0.2), (best, 0.6)], best_cp=30)
    # blunder verifies at -500 (true_loss=530, way over Δ); mid verifies at 15 (true_loss=15, OK).
    eval_map = {chess.Move.from_uci("h2h4"): -500, chess.Move.from_uci("d2d4"): 15}
    chosen, loss, v_loss, n = verify_chosen(
        cands=[best, mid, blunder],
        result=res,
        delta_used=100,
        verify=lambda m: eval_map[m],
    )
    assert chosen.move == mid.move  # fell back to the 2nd-most-alien
    assert v_loss == 15
    assert n == 2                    # blunder + mid


def test_top1_in_considered_acts_as_safe_fallback():
    # blunder fails verification; the next entry in `considered` is the engine's
    # top-1, which is accepted without spending another verification call.
    best = _c("e2e4", 30)
    blunder = _c("h2h4", 10)
    res = _result(blunder, [(blunder, 0.01), (best, 0.6)], best_cp=30)
    chosen, loss, v_loss, n = verify_chosen(
        cands=[best, blunder],
        result=res,
        delta_used=100,
        verify=lambda m: -500,
    )
    assert chosen.move == best.move
    assert loss == 0
    assert v_loss == 0
    assert n == 1                    # only blunder consumed a verify call


def test_budget_exhaustion_falls_back_to_top1():
    # All candidates ahead of top-1 in `considered` fail verification, and we
    # hit MAX_VERIFICATIONS before reaching top-1.
    assert MAX_VERIFICATIONS == 2  # if this changes, this test needs updating
    best = _c("e2e4", 30)
    bad1 = _c("h2h4", 25)
    bad2 = _c("a2a3", 20)
    bad3 = _c("g2g4", 15)
    res = _result(
        bad1,
        [(bad1, 0.01), (bad2, 0.02), (bad3, 0.03), (best, 0.6)],
        best_cp=30,
    )
    chosen, loss, v_loss, n = verify_chosen(
        cands=[best, bad1, bad2, bad3],
        result=res,
        delta_used=50,
        verify=lambda m: -500,  # every verification reveals a blunder
    )
    assert chosen.move == best.move
    assert n == MAX_VERIFICATIONS


def test_verify_returning_none_does_not_count_as_pass():
    # Engine analyse failed (returned None) — must keep falling back, not accept.
    best = _c("e2e4", 30)
    alien = _c("h2h4", 25)
    res = _result(alien, [(alien, 0.01), (best, 0.6)], best_cp=30)
    chosen, loss, v_loss, n = verify_chosen(
        cands=[best, alien],
        result=res,
        delta_used=50,
        verify=lambda m: None,
    )
    # alien's verification returned None → can't accept it → falls through to top-1.
    assert chosen.move == best.move
    assert n == 1
