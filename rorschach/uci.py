"""UCI shim for Rorschach.

Speaks the UCI protocol on stdin/stdout so any UCI host (lichess-bot,
Arena, Cute Chess, …) can drive Rorschach.

Resources are lazy: Patricia and Maia are loaded on the first `isready`,
not at process start, so the host can probe `uci` cheaply.
"""
from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass

import chess
from dotenv import load_dotenv

from rorschach.bot import PROFILES, rorschach_move
from rorschach.engine import PatriciaEngine
from rorschach.explorer import OpeningExplorer
from rorschach.maia import make_predictor

MAIA_TYPES = (
    # Maia-3 / Chessformer. 5M is the CPU-friendly pick.
    "maia3-5m", "maia3-23m", "maia3-79m",
)


# Maia self/opponent bucket used when the opponent's actual rating is unknown.
DEFAULT_ELO = 1900


@dataclass
class Options:
    profile: str = "balanced"
    time_ms: int = 400
    maia_type: str = "maia3-5m"
    # Maia "self" Elo override — the rating cohort we try to look weird to.
    # None (the default) means *auto*: follow the opponent's actual rating so
    # the move looks alien to that opponent's level specifically. Set the Elo
    # UCI option to a value >= 600 to pin a fixed bucket instead.
    elo_override: int | None = None
    # Opponent rating from the standard UCI_Opponent option; None when the host
    # doesn't send one. Maia consumes it as `elo_oppo` — its move distribution
    # shifts based on who it thinks it's playing against.
    oppo_elo: int | None = None

    @property
    def elo_self(self) -> int:
        """Effective Maia self-Elo: explicit override, else opponent, else default."""
        if self.elo_override is not None:
            return self.elo_override
        return self.oppo_elo if self.oppo_elo is not None else DEFAULT_ELO

    @property
    def elo_oppo(self) -> int:
        """Effective Maia opponent-Elo: opponent's rating, else mirror self."""
        return self.oppo_elo if self.oppo_elo is not None else self.elo_self


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _log(msg: str) -> None:
    sys.stderr.write(f"[rorschach] {msg}\n")
    sys.stderr.flush()


def _parse_position(args: list[str]) -> chess.Board:
    if not args:
        return chess.Board()
    if args[0] == "startpos":
        board = chess.Board()
        moves_at = 1
    elif args[0] == "fen":
        # FEN is 6 space-separated fields.
        if len(args) < 7:
            raise ValueError("bad position fen")
        fen = " ".join(args[1:7])
        board = chess.Board(fen)
        moves_at = 7
    else:
        raise ValueError(f"bad position prefix {args[0]!r}")

    if moves_at < len(args):
        if args[moves_at] != "moves":
            raise ValueError(f"expected 'moves', got {args[moves_at]!r}")
        for uci in args[moves_at + 1:]:
            board.push(chess.Move.from_uci(uci))
    return board


def _parse_go(args: list[str], turn_white: bool) -> int:
    """Return time_ms budget for this move based on UCI go args."""
    movetime = wtime = btime = winc = binc = None
    i = 0
    while i < len(args):
        tok = args[i]
        if tok == "movetime":
            movetime = int(args[i + 1]); i += 2
        elif tok == "wtime":
            wtime = int(args[i + 1]); i += 2
        elif tok == "btime":
            btime = int(args[i + 1]); i += 2
        elif tok == "winc":
            winc = int(args[i + 1]); i += 2
        elif tok == "binc":
            binc = int(args[i + 1]); i += 2
        elif tok in ("infinite", "ponder"):
            i += 1
        elif tok in ("depth", "nodes", "mate", "movestogo"):
            i += 2  # ignore, we always do time-budget search
        else:
            i += 1
    if movetime is not None:
        return max(50, movetime)

    my_time = wtime if turn_white else btime
    my_inc = (winc if turn_white else binc) or 0
    if my_time is None:
        return 400  # no clock info — fixed default

    # Crude TM: ~3.5% of remaining time + 90% of increment, clamped.
    # The 3.5% (up from 2.5%) gives the verification pass enough budget that
    # the wider MultiPV scan still hits a useful depth.
    budget = int(my_time * 0.035 + my_inc * 0.9)
    return max(100, min(3000, budget))


def _emit_options() -> None:
    profiles = " ".join(f"var {p}" for p in PROFILES)
    _emit(f"option name Profile type combo default balanced {profiles}")
    _emit("option name TimeMs type spin default 0 min 0 max 5000")  # 0 = use go's clock info
    # Maia-3 covers Lichess blitz 600..2600. 0 (default) = auto: track the
    # opponent's actual rating (UCI_Opponent); 600..2600 pins a fixed bucket.
    _emit("option name Elo type spin default 0 min 0 max 2600")
    maia_vars = " ".join(f"var {t}" for t in MAIA_TYPES)
    _emit(f"option name MaiaType type combo default maia3-5m {maia_vars}")
    # Standard UCI option; lichess-bot sends this with the opponent's rating.
    # Format: "<title> <elo|none> <computer|human> <name>".
    _emit("option name UCI_Opponent type string default")


# Maia's Elo conditioning is clamped to its training range (600..2600). We
# clip rather than reject so very high-rated or unknown opponents still work.
_MAIA_ELO_MIN, _MAIA_ELO_MAX = 600, 2600


def _parse_uci_opponent_elo(value: str) -> int | None:
    """Extract the opponent's Elo from a UCI_Opponent value, or None.

    Spec: "<title> <elo|none> <computer|human> <name>". We need the 2nd token.
    """
    tokens = value.split()
    if len(tokens) < 2:
        return None
    raw = tokens[1]
    if raw.lower() == "none":
        return None
    try:
        elo = int(raw)
    except ValueError:
        return None
    return max(_MAIA_ELO_MIN, min(_MAIA_ELO_MAX, elo))


def _set_option(opts: Options, words: list[str]) -> None:
    # Format: setoption name <Name> [value <Value>]
    if "name" not in words:
        return
    name_idx = words.index("name") + 1
    val_idx = words.index("value") + 1 if "value" in words else None
    name = words[name_idx]
    value = " ".join(words[val_idx:]) if val_idx is not None else ""
    if name == "Profile" and value in PROFILES:
        opts.profile = value
    elif name == "TimeMs":
        opts.time_ms = int(value)
    elif name == "Elo":
        # >= Maia floor pins a fixed bucket; anything lower (incl. 0) = auto.
        v = int(value)
        opts.elo_override = v if v >= _MAIA_ELO_MIN else None
    elif name == "MaiaType" and value in MAIA_TYPES:
        opts.maia_type = value
    elif name == "UCI_Opponent":
        # None when unknown; the elo_self/elo_oppo properties handle the fallback.
        opts.oppo_elo = _parse_uci_opponent_elo(value)


class _Resources:
    """Lazy holder for Patricia, Maia, Explorer."""

    def __init__(self) -> None:
        self.engine: PatriciaEngine | None = None
        self.maia = None  # Maia3Predictor
        self.explorer: OpeningExplorer | None = None

    def ensure(self, opts: Options) -> None:
        # Maia / huggingface_hub / tqdm chatter goes to stdout by
        # default; that pollutes the UCI channel. Redirect any side-effect
        # prints to stderr while loading. The UCI host parses stdout strictly.
        if self.engine is None:
            _log("loading Patricia ...")
            with contextlib.redirect_stdout(sys.stderr):
                self.engine = PatriciaEngine()
        if self.maia is None or getattr(self.maia, "_name", None) != opts.maia_type:
            _log(f"loading Maia ({opts.maia_type}) ...")
            with contextlib.redirect_stdout(sys.stderr):
                self.maia = make_predictor(opts.maia_type, device="cpu")
        if self.explorer is None:
            self.explorer = OpeningExplorer()
            if not self.explorer.token:
                _log("warning: LICHESS_TOKEN not set, explorer disabled")

    def shutdown(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None


def main() -> None:
    load_dotenv()
    opts = Options()
    resources = _Resources()
    board = chess.Board()

    try:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            words = line.split()
            cmd = words[0]

            if cmd == "uci":
                _emit("id name Rorschach")
                _emit("id author Avo-k")
                _emit_options()
                _emit("uciok")
            elif cmd == "isready":
                resources.ensure(opts)
                _emit("readyok")
            elif cmd == "setoption":
                _set_option(opts, words)
            elif cmd == "ucinewgame":
                board = chess.Board()
            elif cmd == "position":
                try:
                    board = _parse_position(words[1:])
                except (ValueError, chess.InvalidMoveError) as exc:
                    _log(f"bad position: {exc}")
            elif cmd == "go":
                resources.ensure(opts)
                if opts.time_ms > 0:
                    time_ms = opts.time_ms  # operator override
                else:
                    time_ms = _parse_go(words[1:], board.turn == chess.WHITE)
                try:
                    move, info = rorschach_move(
                        board,
                        resources.engine,                  # type: ignore[arg-type]
                        resources.maia,                    # type: ignore[arg-type]
                        explorer=resources.explorer,
                        profile=opts.profile,
                        time_ms=time_ms,
                        elo_self=opts.elo_self,
                        elo_oppo=opts.elo_oppo,
                    )
                    break_str = (
                        f" break={info.narrative_break:+.3f}"
                        if info.narrative_break is not None
                        else ""
                    )
                    verify_str = (
                        f" v_loss={info.verified_loss} v={info.n_verified}"
                        if info.verified_loss is not None
                        else ""
                    )
                    _emit(
                        f"info depth {info.depth or 0} score cp {info.best_cp} "
                        f"string Δ={info.delta_used} loss={info.eval_loss} "
                        f"P={info.p_human:.3f}{break_str}{verify_str} "
                        f"oracle={info.oracle}"
                    )
                    _emit(f"bestmove {move.uci()}")
                except Exception as exc:  # last-resort fallback so the bot never hangs
                    _log(f"search error: {exc!r}")
                    legal = list(board.legal_moves)
                    if legal:
                        _emit(f"bestmove {legal[0].uci()}")
                    else:
                        _emit("bestmove 0000")
            elif cmd in ("stop", "ponderhit"):
                pass  # we don't ponder; nothing to interrupt
            elif cmd == "quit":
                break
    finally:
        resources.shutdown()


if __name__ == "__main__":
    main()
