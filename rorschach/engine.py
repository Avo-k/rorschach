"""Patricia UCI wrapper. Returns top-K candidates with side-to-move cp scores."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "bin" / "patricia"
MATE_SCORE = 10000


@dataclass(frozen=True)
class Candidate:
    move: chess.Move
    cp: int           # side-to-move perspective; mates encoded as ±MATE_SCORE - mate_distance
    depth: int | None = None


class PatriciaEngine:
    """Thin wrapper over `chess.engine.SimpleEngine`. Use as context manager."""

    def __init__(
        self,
        path: Path | str = DEFAULT_PATH,
        hash_mb: int = 64,
        threads: int = 1,
    ) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(str(path))
        self._engine.configure({"Hash": hash_mb, "Threads": threads})

    def multipv_search(
        self,
        board: chess.Board,
        k: int = 5,
        time_ms: int = 200,
    ) -> list[Candidate]:
        """Return up to `k` distinct first-moves, sorted best-first (highest cp first).

        Patricia occasionally emits duplicate first-moves across PVs (different
        continuations of the same starting move). We dedupe by first move, keeping
        the best score seen for each.
        """
        infos = self._engine.analyse(
            board,
            limit=chess.engine.Limit(time=time_ms / 1000),
            multipv=k,
        )
        best_per_move: dict[str, Candidate] = {}
        for info in infos:
            pv = info.get("pv") or []
            if not pv:
                continue
            move = pv[0]
            score = info["score"].pov(board.turn).score(mate_score=MATE_SCORE)
            if score is None:
                continue
            cand = Candidate(move=move, cp=score, depth=info.get("depth"))
            existing = best_per_move.get(move.uci())
            if existing is None or cand.cp > existing.cp:
                best_per_move[move.uci()] = cand
        return sorted(best_per_move.values(), key=lambda c: -c.cp)

    def quick_eval_after_move(
        self,
        board: chess.Board,
        move: chess.Move,
        time_ms: int,
    ) -> int | None:
        """Push `move`, single-PV search, return cp from the moving player's POV.

        Used to verify a candidate's true eval at deeper depth than the wider
        MultiPV scan saw it. Returns None if the engine produced no usable score.
        """
        moving_player = board.turn
        board.push(move)
        try:
            info = self._engine.analyse(
                board, limit=chess.engine.Limit(time=time_ms / 1000),
            )
            score = info["score"].pov(moving_player).score(mate_score=MATE_SCORE)
            return score
        finally:
            board.pop()

    def quit(self) -> None:
        self._engine.quit()

    def __enter__(self) -> "PatriciaEngine":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.quit()
