"""Human-move predictors. Each returns P(move | position, elos) over all legal moves.

Two backends:

- :class:`MaiaPredictor` wraps Maia-2 (the ``maia2`` PyPI package).
- :class:`Maia3Predictor` wraps Maia-3 / Chessformer (the ``maia3`` package from
  ``github.com/CSSLab/maia3``). Default. Stronger predictor, wider Elo range
  (600..2600), supports board history.

Both expose the same shape:
    predict(board, elo_self, elo_oppo) -> (move_probs: {uci: float}, win_prob: float)
"""
from __future__ import annotations

import random as _random
from collections import deque
from types import SimpleNamespace

import chess


class MaiaPredictor:
    """Maia-2 backend. Loads a model once, then queries per position."""

    def __init__(self, type: str = "rapid", device: str = "cpu") -> None:
        from maia2 import inference, model

        self._inference = inference
        self._model = model.from_pretrained(type=type, device=device)
        self._prepared = inference.prepare()
        self._name = type

    def predict(
        self,
        board: chess.Board,
        elo_self: int = 1900,
        elo_oppo: int = 1900,
    ) -> tuple[dict[str, float], float]:
        """Return (move_probs: {uci: prob}, predicted_win_prob_for_stm)."""
        move_probs, win_prob = self._inference.inference_each(
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
            uci = items[0][0]
        else:
            weights = [p / total for _, p in items]
            uci = rng.choices([u for u, _ in items], weights=weights, k=1)[0]
        return chess.Move.from_uci(uci), float(move_probs[uci])


class Maia3Predictor:
    """Maia-3 backend.

    Reconstructs board history from ``board.move_stack`` so the model sees the
    last ``cfg.history`` (default 8) positions, which is how it was trained.
    Falls back to single-position padding when no stack is available
    (e.g. board built from a bare FEN).
    """

    def __init__(
        self,
        alias: str = "maia3-5m",
        device: str = "cpu",
        use_amp: bool = False,
    ) -> None:
        import torch
        from maia3.dataset import (
            get_historical_tokens,
            get_legal_moves_mask,
            tokenize_board,
        )
        from maia3.model_registry import (
            apply_model_config,
            resolve_checkpoint_path,
            resolve_model_spec,
        )
        from maia3.models import MAIA3Model
        from maia3.utils import get_all_possible_moves, mirror_move

        self._torch = torch
        self._tokenize_board = tokenize_board
        self._get_historical_tokens = get_historical_tokens
        self._get_legal_moves_mask = get_legal_moves_mask
        self._mirror_move = mirror_move

        spec = resolve_model_spec(alias)
        cfg = SimpleNamespace(
            device=device,
            use_amp=use_amp and device.startswith("cuda"),
            trust_checkpoint=False,
            # Default arch — overwritten by apply_model_config from the spec.
            history=8, use_padding=True, include_time_info=False,
            dim_emb=128, dim_vit=192, num_blocks=8, num_heads=6, mlp_ratio=2.0,
            dropout=0.0, head_hid_dim=192,
            use_gab=True, gab_gen_size=64, gab_per_square_dim=0,
            gab_intermediate_dim=64, use_rms_norm=True, omit_qkv_biases=True,
            activation="gelu",
            use_relative_bias=False, use_absolute_pe=False,
        )
        apply_model_config(cfg, spec)
        cfg.checkpoint_path = resolve_checkpoint_path(spec)
        self.cfg = cfg

        model = MAIA3Model(cfg).to(device)
        ckpt = torch.load(
            cfg.checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        # Older checkpoints used "smolgen"; the current model uses "gab".
        renamed = {k.replace("smolgen", "gab"): v for k, v in state_dict.items()}
        model.load_state_dict(renamed, strict=False)
        model.eval()
        self._model = model

        self._all_moves = get_all_possible_moves()
        self._all_moves_dict = {m: i for i, m in enumerate(self._all_moves)}
        self._name = alias

    def _build_history(self, board: chess.Board) -> deque:
        """Rebuild the last ``cfg.history`` positions from ``board.move_stack``.

        Padding to ``cfg.history`` is done downstream by ``get_historical_tokens``.
        """
        moves = list(board.move_stack)
        history: deque = deque(maxlen=self.cfg.history)
        if not moves:
            history.append(self._tokenize_board(board))
            return history

        replay = board.copy()  # full copy preserves move_stack
        for _ in range(len(moves)):
            replay.pop()
        history.append(self._tokenize_board(replay))
        for mv in moves:
            replay.push(mv)
            history.append(self._tokenize_board(replay))
        return history

    def _forward(
        self,
        board: chess.Board,
        elo_self: int,
        elo_oppo: int,
        history: deque,
    ) -> tuple[dict[str, float], float]:
        """One model forward pass with the given history deque."""
        torch = self._torch
        cfg = self.cfg

        with torch.no_grad():
            tokens = self._get_historical_tokens(
                history, cfg,
                base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0,
            )
            tokens = tokens.unsqueeze(0).to(cfg.device)
            self_elos = torch.tensor([elo_self], dtype=torch.long, device=cfg.device)
            oppo_elos = torch.tensor([elo_oppo], dtype=torch.long, device=cfg.device)

            legal_mask = self._get_legal_moves_mask(board, self._all_moves_dict).to(cfg.device)

            logits_move, logits_value, _ = self._model(tokens, self_elos, oppo_elos)
            logits = logits_move[0].float().masked_fill(~legal_mask, float("-inf"))
            probs = torch.softmax(logits, dim=-1).cpu()

            out: dict[str, float] = {}
            for move in board.legal_moves:
                uci = move.uci()
                key = uci if board.turn == chess.WHITE else self._mirror_move(uci)
                idx = self._all_moves_dict.get(key)
                if idx is not None:
                    out[uci] = float(probs[idx])

            # Value head logits are [loss, draw, win] for the side to move.
            value_probs = torch.softmax(logits_value[0].float(), dim=-1).cpu()
            return out, float(value_probs[2])

    def predict(
        self,
        board: chess.Board,
        elo_self: int = 1900,
        elo_oppo: int = 1900,
    ) -> tuple[dict[str, float], float]:
        if board.is_game_over():
            return {}, 0.5
        return self._forward(board, elo_self, elo_oppo, self._build_history(board))

    def predict_pair(
        self,
        board: chess.Board,
        elo_self: int = 1900,
        elo_oppo: int = 1900,
    ) -> tuple[dict[str, float], dict[str, float], float]:
        """Two forward passes: with full history, and with the current position only.

        Returns ``(probs_with, probs_without, win_prob_with)``. The narrative-break
        score for a move is ``probs_without[uci] - probs_with[uci]``: positive means
        the game's recent history made the move less likely than the position alone
        would suggest. Doubles inference cost; only use it when the selector
        consumes both signals.
        """
        if board.is_game_over():
            return {}, {}, 0.5
        history_full = self._build_history(board)
        history_none: deque = deque(
            [self._tokenize_board(board)], maxlen=self.cfg.history,
        )
        probs_with, win = self._forward(board, elo_self, elo_oppo, history_full)
        probs_without, _ = self._forward(board, elo_self, elo_oppo, history_none)
        return probs_with, probs_without, win


def make_predictor(name: str, device: str = "cpu"):
    """Factory returning the right backend for ``name``.

    Names:
      - ``blitz`` / ``rapid``                  — Maia-2 (kept for A/B calibration)
      - ``maia3-5m`` / ``maia3-23m`` / ``maia3-79m`` — Maia-3 / Chessformer
    """
    if name in ("blitz", "rapid"):
        return MaiaPredictor(type=name, device=device)
    if name.startswith("maia3"):
        return Maia3Predictor(alias=name, device=device)
    raise ValueError(f"unknown maia predictor {name!r}")
