"""Lichess Opening Explorer wrapper.

For positions with enough games in the Lichess DB, real human-move
frequencies are a better "what would a human play?" signal than Maia's
prediction. This module returns those frequencies as a {uci: prob} dict
when the position has at least `min_games` games, else None — caller
falls back to Maia in that case.

Beautiful side effect: a legal, engine-sound move that has been played
zero times in the DB gets P = 0 in our dict (it's just absent), making
it maximally attractive to the selector. Maia would always assign such
moves a small but non-zero probability.
"""
from __future__ import annotations

import os

import chess
import requests

URL = "https://explorer.lichess.ovh/{db}"

# As of 2026 the explorer subdomain returns 401 for unauthenticated requests
# from many IPs. Set LICHESS_TOKEN (free OAuth personal token from
# https://lichess.org/account/oauth/token) to authenticate.


class OpeningExplorer:
    def __init__(
        self,
        db: str = "lichess",
        speeds: tuple[str, ...] = ("blitz", "rapid"),
        ratings: tuple[int, ...] = (1800, 2000, 2200),
        min_games: int = 50,
        timeout: float = 10.0,
        token: str | None = None,
    ) -> None:
        self.db = db
        self.speeds = ",".join(speeds)
        self.ratings = ",".join(str(r) for r in ratings)
        self.min_games = min_games
        self.timeout = timeout
        self.token = token if token is not None else os.environ.get("LICHESS_TOKEN")
        self._cache: dict[str, dict[str, float] | None] = {}
        self._headers = {"User-Agent": "rorschach/0.1"}
        if self.token:
            self._headers["Authorization"] = f"Bearer {self.token}"
        self.n_hits = 0
        self.n_misses = 0
        self.n_401 = 0

    def predict(self, board: chess.Board) -> dict[str, float] | None:
        """Return {uci: prob} for moves in the DB, or None if too few games.

        Probabilities are based on raw game counts (white+draws+black) and
        sum to ≤ 1 — moves never played at this position are simply absent.
        """
        fen = board.fen()
        if fen in self._cache:
            return self._cache[fen]

        params = {
            "fen": fen,
            "variant": "standard",
            "speeds": self.speeds,
            "ratings": self.ratings,
            "moves": 30,
        }
        try:
            r = requests.get(
                URL.format(db=self.db), params=params,
                headers=self._headers, timeout=self.timeout,
            )
            if r.status_code == 401:
                self.n_401 += 1
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError):
            self._cache[fen] = None
            self.n_misses += 1
            return None

        total = data.get("white", 0) + data.get("draws", 0) + data.get("black", 0)
        if total < self.min_games:
            self._cache[fen] = None
            self.n_misses += 1
            return None

        probs = {
            m["uci"]: (m["white"] + m["draws"] + m["black"]) / total
            for m in data.get("moves", [])
        }
        self._cache[fen] = probs
        self.n_hits += 1
        return probs
