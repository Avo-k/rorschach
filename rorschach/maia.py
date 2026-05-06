"""Maia-2 wrapper. Returns P(move | position, elos) over all legal moves."""
from __future__ import annotations

import random as _random

import chess
from maia2 import inference, model


class MaiaPredictor:
    """Loads a Maia-2 model once, then queries per position."""

    def __init__(self, type: str = "rapid", device: str = "cpu") -> None:
        self._model = model.from_pretrained(type=type, device=device)
        self._prepared = inference.prepare()

    def predict(
        self,
        board: chess.Board,
        elo_self: int = 1900,
        elo_oppo: int = 1900,
    ) -> tuple[dict[str, float], float]:
        """Return (move_probs: {uci: prob}, predicted_win_prob_for_stm)."""
        move_probs, win_prob = inference.inference_each(
            self._model,
            self._prepared,
            board.fen(),
            elo_self,
            elo_oppo,
        )
        return move_probs, float(win_prob)

    def sample(
        self,
        board: chess.Board,
        elo_self: int = 1900,
        elo_oppo: int = 1900,
        rng: _random.Random | None = None,
    ) -> tuple[chess.Move, float]:
        """Sample a move from Maia's probability distribution.

        Returns (move, prob_of_chosen_move). Falls back to argmax if probabilities
        are degenerate (all zero or NaN).
        """
        move_probs, _ = self.predict(board, elo_self, elo_oppo)
        if not move_probs:
            raise ValueError("Maia returned no legal-move probabilities")
        items = list(move_probs.items())
        total = sum(p for _, p in items)
        rng = rng or _random
        if total <= 0:
            uci = items[0][0]  # fallback to top
        else:
            weights = [p / total for _, p in items]
            uci = rng.choices([u for u, _ in items], weights=weights, k=1)[0]
        return chess.Move.from_uci(uci), float(move_probs[uci])
