"""UCI shim parsing helpers."""
from __future__ import annotations

from rorschach.uci import Options, _parse_uci_opponent_elo, _set_option


# --- UCI_Opponent value parsing --------------------------------------------


def test_parse_opponent_value_with_rating():
    assert _parse_uci_opponent_elo("none 1450 human SomeName") == 1450


def test_parse_opponent_value_with_title_and_rating():
    assert _parse_uci_opponent_elo("GM 2750 human MagnusCarlsen") == 2600  # clamped to Maia max


def test_parse_opponent_value_rating_none():
    assert _parse_uci_opponent_elo("none none human Anonymous") is None


def test_parse_opponent_value_too_short():
    assert _parse_uci_opponent_elo("") is None
    assert _parse_uci_opponent_elo("none") is None


def test_parse_opponent_value_clamped_to_maia_range():
    # Below the Maia training floor.
    assert _parse_uci_opponent_elo("none 300 human Beginner") == 600
    # Above the training ceiling.
    assert _parse_uci_opponent_elo("none 3200 computer Stockfish") == 2600


def test_parse_opponent_value_bad_rating():
    assert _parse_uci_opponent_elo("none not-a-number human Bob") is None


# --- _set_option wiring ----------------------------------------------------


def _setopt(opts: Options, raw: str) -> None:
    # Mirror what main() does: split the raw "setoption name X value Y..." line.
    _set_option(opts, raw.split())


def test_uci_opponent_sets_elo_oppo():
    opts = Options()
    _setopt(opts, "setoption name UCI_Opponent value none 1200 human Alice")
    assert opts.elo_oppo == 1200


def test_uci_opponent_none_falls_back_to_self_elo():
    opts = Options(elo=1700)
    opts.elo_oppo = 9999  # any prior value
    _setopt(opts, "setoption name UCI_Opponent value none none human Anonymous")
    assert opts.elo_oppo == 1700  # fell back to opts.elo, NOT to 1900 default


def test_uci_opponent_doesnt_touch_self_elo():
    opts = Options()
    _setopt(opts, "setoption name UCI_Opponent value none 2200 human Strong")
    assert opts.elo_oppo == 2200
    assert opts.elo == 1900  # unchanged
