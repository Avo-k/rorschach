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


def test_default_elo_is_auto_tracks_opponent():
    # No Elo override + opponent known → both self and oppo follow the opponent.
    opts = Options()
    _setopt(opts, "setoption name UCI_Opponent value none 1200 human Alice")
    assert opts.elo_oppo == 1200
    assert opts.elo_self == 1200  # the cohort we look weird to = the opponent


def test_default_elo_unknown_opponent_falls_back_to_default():
    opts = Options()
    _setopt(opts, "setoption name UCI_Opponent value none none human Anonymous")
    assert opts.elo_self == 1900  # DEFAULT_ELO
    assert opts.elo_oppo == 1900


def test_explicit_elo_override_pins_self_regardless_of_opponent():
    opts = Options()
    _setopt(opts, "setoption name Elo value 1700")
    _setopt(opts, "setoption name UCI_Opponent value none 2200 human Strong")
    assert opts.elo_self == 1700   # pinned bucket
    assert opts.elo_oppo == 2200   # opponent conditioning still tracks the foe


def test_explicit_elo_override_unknown_opponent_mirrors_self():
    opts = Options()
    _setopt(opts, "setoption name Elo value 1700")
    _setopt(opts, "setoption name UCI_Opponent value none none human Anonymous")
    assert opts.elo_self == 1700
    assert opts.elo_oppo == 1700  # mirrors self when opponent unknown


def test_elo_zero_means_auto():
    opts = Options()
    _setopt(opts, "setoption name Elo value 1500")
    _setopt(opts, "setoption name Elo value 0")  # back to auto
    assert opts.elo_override is None
    assert opts.elo_self == 1900  # auto with no opponent → DEFAULT_ELO
