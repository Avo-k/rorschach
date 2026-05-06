"""Maia-2 wrapper. Returns P(move | position, elos) over all legal moves."""
from __future__ import annotations

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
