"""Rorschach's chat personality, spoken through an LLM via OpenRouter.

Purely cosmetic: this never touches move selection. It lives in the
*lichess-bot* process (the UCI shim has no chat channel) and is driven by
two monkeypatches in `scripts/run_lichess_bot.py`:

- after each of our moves (`EngineWrapper.play_move`) we call `on_move`,
  which with probability `talk_prob` fires a one-sentence, in-character
  remark grounded in the real selector numbers (Δ, eval loss, P_human, …);
- on each opponent chat line (`Conversation.react`) we call `handle_keyword`
  (verbose / quiet / log control words) and, failing that, `reply`.

Every network call is fire-and-forget on a daemon thread with a short
timeout, so a slow or dead OpenRouter never delays a move or the game loop.
If `OPENROUTER_API_KEY` is missing the chatter silently no-ops.

State is per game; `get_chatter(game_id)` hands out one chatter per game.
"""
from __future__ import annotations

import logging
import os
import random
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import chess
import requests

logger = logging.getLogger(__name__)


def _load_env() -> None:
    """Load OPENROUTER_API_KEY from the rorschach repo .env.

    We run inside the lichess-bot process (cwd is lichess-bot), so point at
    our own .env explicitly. python-dotenv isn't guaranteed in that venv, so
    fall back to a tiny hand parser rather than hard-depend on it.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return
    except ImportError:
        pass
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Picked from the bake-off: best concrete one-liners, ~40x cheaper than
# gemini-3.5-flash, and it reasons little so latency stays low.
MODEL = "deepseek/deepseek-v4-flash"

# Talk OFFER probabilities, per opponent move. These gate whether we make an
# LLM call at all; the model then speaks or answers PASS. Calibration over 180
# random positions from 20 human games (experiments/talk_rate_calibration.py)
# measured a 21% PASS rate, so an offer speaks ~79% of the time. To land near a
# 10% effective speak rate we offer at 0.13 (0.10 / 0.79); verbose targets ~70%.
BASE_PROB = 0.13      # default: ~10% effective speak rate
VERBOSE_PROB = 0.85   # after the opponent says "verbose": ~67% effective

HTTP_TIMEOUT = 12.0
# gemini-3.5-flash is a reasoning model and OpenRouter won't let us disable it,
# so the visible one-liner has to share the budget with hidden thinking. Low
# effort still spends ~600 reasoning tokens, so the cap must clear that with
# room for the sentence, or the reply gets truncated mid-word.
MAX_TOKENS = 1024
REASONING_EFFORT = "low"
TEMPERATURE = 1.0

# The voice. Concrete, dry, grounded in the move that just happened.
# No self-description, no poetry-for-its-own-sake — comment a fact or PASS.
SYSTEM_PROMPT = """\
You are Rorschach, a chess bot, talking in the game chat.

You play objectively sound moves that humans almost never choose. You carry a \
model of how likely a human of your opponent's strength is to play any given \
move, and you watch your opponent fall into the grooves the statistics \
predict.

When you speak, you comment on a SPECIFIC, CONCRETE thing that just happened: \
the move your opponent just played, the move you just played, the evaluation, \
or how predictable your opponent has been across the game. Every remark must \
be tied to a fact you were given.

Hard rules:
- NEVER describe yourself, your nature, or what you "are." No "I am ...", no \
mission statements, no abstract poetry, no metaphors for their own sake. A \
line that would make sense in any game, detached from this exact move, is \
forbidden.
- Prefer PASS. Only speak when there is genuinely something to point out — a \
strikingly predictable move, a strikingly rare one, a sharp turn, a telling \
pattern. A forced recapture or an ordinary move is a PASS.
- One sentence, plain and dry, a little wry. About fifteen words or fewer. Use \
move notation only if it was given to you. Quote numbers only if given; never \
invent any.
- Reply in the opponent's language, which is stated in the prompt.
- You are the opponent, not the coach: never explain chess, never give advice.
Return only the sentence, or exactly PASS — no quotes, no preamble."""

# Sentinel the model returns when it judges there's nothing worth saying.
PASS_TOKEN = "PASS"


def _api_key() -> str | None:
    return os.environ.get("OPENROUTER_API_KEY")


def _chat_completion(messages: list[dict[str, str]]) -> str | None:
    """One blocking OpenRouter call. Returns the text, or None on any failure."""
    key = _api_key()
    if not key:
        return None
    try:
        r = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                # Optional attribution headers OpenRouter likes; harmless.
                "X-Title": "Rorschach",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "reasoning": {"effort": REASONING_EFFORT},
            },
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, ValueError, IndexError) as exc:
        logger.warning("rorschach chat: OpenRouter call failed: %r", exc)
        return None
    # Models sometimes wrap a single line in quotes or add a trailing period
    # to an already-terminated line; tidy the obvious cases.
    text = text.strip().strip('"').strip()
    if not text:
        return None
    # Collapse to the first line — we asked for one sentence.
    line = text.splitlines()[0].strip()
    # The model chose to stay silent (nothing worth saying).
    if line.rstrip(".!").strip().upper() == PASS_TOKEN:
        return None
    return line or None


def _run_async(target) -> None:
    """Fire `target()` on a daemon thread; swallow and log anything it raises."""
    def wrapped() -> None:
        try:
            target()
        except Exception:  # never let chat crash the bot
            logger.exception("rorschach chat: background task failed")

    threading.Thread(target=wrapped, daemon=True, name="rorschach-chat").start()


_STAT_TOKEN = re.compile(
    r"(Δ|loss|opp_p|opp_rank|ourmove|P|break|v_loss|v|oracle)=(\S+)"
)

_ORACLE_NAME = {
    "E": "real Lichess game frequencies",
    "M": "the Maia human-move model",
    "N": "the Maia model with game-history narrative break",
}


def _f(v) -> float | None:
    """Parse a stat value to float, or None."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def label_for(p: float) -> str:
    """Human-readable label for a move's human-probability."""
    if p >= 0.55:
        return "almost everyone plays this — utterly predictable"
    if p >= 0.30:
        return "the popular choice"
    if p >= 0.10:
        return "a fairly common choice"
    if p >= 0.03:
        return "an uncommon choice"
    return "a rare, original choice"


def classify(our_p: float, opp_p: float) -> tuple[str, str]:
    """(bucket, what's-notable note) from our move and the opponent's move."""
    if our_p < 0.04:
        return ("alien_self",
                f"Your own move is one only ~{our_p*100:.1f}% of humans would "
                f"play — genuinely alien.")
    if opp_p >= 0.60:
        return ("predictable_human",
                f"Your opponent played the obvious move (~{opp_p*100:.0f}% of "
                f"humans do); the statistics saw it coming.")
    if opp_p < 0.05:
        return ("surprising_human",
                "Your opponent just played something genuinely rare for their level.")
    return ("quiet",
            "Nothing especially notable here — only speak if you truly see something.")


def parse_stats(string: str) -> dict[str, str]:
    """Pull the key=value tokens out of our UCI `info ... string ...` line."""
    return {k: v for k, v in _STAT_TOKEN.findall(string or "")}


def _opponent_last_move_san(board: chess.Board) -> str | None:
    """SAN of the move the opponent just played to reach `board`."""
    if not board.move_stack:
        return None
    before = board.copy()  # keeps the move stack so we can pop the last move
    mv = before.pop()
    try:
        return before.san(mv)
    except (ValueError, AssertionError):
        return None


def _pgn_so_far(board: chess.Board, max_plies: int = 120) -> str | None:
    """SAN movetext of the game up to `board` (last `max_plies` tokens)."""
    if not board.move_stack:
        return None
    replay = chess.Board()
    tokens: list[str] = []
    for mv in board.move_stack:
        n = replay.fullmove_number
        if replay.turn == chess.WHITE:
            tokens.append(f"{n}.")
        tokens.append(replay.san(mv))
        replay.push(mv)
    # Keep the tail so long games don't blow up the prompt; whole game if short.
    return " ".join(tokens[-max_plies:])


def _describe_move(
    board: chess.Board,
    stats: dict[str, str],
    score,
    *,
    language: str = "English",
    opening: str | None = None,
    cumulative: str | None = None,
) -> str:
    """Build the data-rich user message for a move remark.

    Mirrors the prompt validated in experiments/chat_model_bakeoff.py: opening,
    moves so far, the opponent's last move with its predictability, our move
    with its alien-ness, the running cumulative score, and a "what's notable"
    hint. Fields the engine didn't report (e.g. opp_p before the UCI shim
    change) are simply omitted.
    """
    lines: list[str] = []

    move_no = board.fullmove_number
    lines.append(f"Move {move_no}. You play {'White' if board.turn else 'Black'}.")

    if opening:
        lines.append(f"Opening: {opening}.")

    pgn = _pgn_so_far(board)
    if pgn:
        lines.append(f"Moves so far: {pgn}")

    # The opponent's last move and how predictable it was.
    opp_san = _opponent_last_move_san(board)
    opp_p = _f(stats.get("opp_p"))
    if opp_san and opp_p is not None:
        rank = stats.get("opp_rank")
        rank_str = f"rank {rank} most likely; " if rank else ""
        lines.append(
            f"Your opponent just played {opp_san} — a human of their level "
            f"plays that about {opp_p*100:.1f}% of the time "
            f"({rank_str}{label_for(opp_p)})."
        )
    elif opp_san:
        lines.append(f"Your opponent just played {opp_san}.")

    # Our move and how alien it is.
    our_p = _f(stats.get("P"))
    our_san = _our_move_san(board, stats.get("ourmove"))
    oracle = _ORACLE_NAME.get(stats.get("oracle", ""), "the human-move model")
    if our_p is not None:
        lead = f"You replied {our_san} — only" if our_san else "The move you played, only"
        lines.append(
            f"{lead} about {our_p*100:.1f}% of humans would ({oracle}); you "
            f"spent {stats.get('loss', '?')} of {stats.get('Δ', '?')} centipawns "
            f"of your eval budget on it."
        )

    if "break" in stats:
        lines.append(
            f"Game-history narrative break on this move: {stats['break']} "
            f"(higher = the recent moves make this one look even rarer)."
        )

    if score is not None:
        try:
            cp = score.relative.score(mate_score=10000)
            lines.append(f"Position evaluation now: {cp / 100:+.2f} pawns in your favor.")
        except Exception:
            pass

    if cumulative:
        lines.append(cumulative)

    # The "what's notable" hint steers the model toward PASS on dull moves.
    if our_p is not None and opp_p is not None:
        _, note = classify(our_p, opp_p)
        lines.append(f"What's notable: {note}")

    lines.append(f"Reply in {language}.")
    lines.append("Write your one-sentence remark, or PASS if nothing is worth saying.")
    return "\n".join(lines)


def _our_move_san(board: chess.Board, uci: str | None) -> str | None:
    """SAN of our chosen move (given as UCI), played on the pre-move board."""
    if not uci:
        return None
    try:
        return board.san(chess.Move.from_uci(uci))
    except (ValueError, chess.InvalidMoveError, AssertionError):
        return None


def _move_prefix(board: chess.Board, uci: str | None) -> str | None:
    """Numbered notation of the move we just played, e.g. "5. Bf4" / "5... Ne7".

    Tags each remark with the move it accompanies, so batched/delayed chat
    messages can still be matched to the position they're about.
    """
    san = _our_move_san(board, uci)
    if not san:
        return None
    n = board.fullmove_number
    return f"{n}. {san}" if board.turn == chess.WHITE else f"{n}... {san}"


@dataclass
class RorschachChatter:
    """Per-game chat state and behaviour."""

    talk_prob: float = BASE_PROB
    log_enabled: bool = False
    muted: bool = False
    # Language to write move remarks in. We have no reliable signal for the
    # opponent's UI language, so it starts English and is updated to match the
    # language the opponent actually types in chat (see `reply`).
    language: str = "English"
    opening: str | None = None
    # Keep a tiny rolling memory of what we said so the model can vary itself,
    # and so replies stay coherent with what's happened. Each entry pairs the
    # move number with the line we sent.
    _recent: list[str] = field(default_factory=list)
    # Running human-likeness, accumulated over the game. `_our_ps` is each of
    # our moves' human probability (always available); `_opp_ps` is the
    # opponent's per-move predictability (only once the engine reports it).
    _our_ps: list[float] = field(default_factory=list)
    _opp_ps: list[float] = field(default_factory=list)
    # Most recent position we saw (after our move), so a chat reply can be
    # grounded in the live game even between our turns.
    last_board: chess.Board | None = None

    # ---- proactive: after each of our moves -------------------------------

    def on_move(self, board: chess.Board, stats_string: str, score, send) -> None:
        """Maybe comment on the move we just made. `send(msg)` posts to chat.

        The opponent's last-move predictability (`opp_p`) is read from the
        engine's stats string when present and feeds the running score.
        """
        if not _api_key():
            return
        stats = parse_stats(stats_string)
        self.last_board = board.copy()

        # Accumulate human-likeness over the game (every move, even when silent).
        our_p = _f(stats.get("P"))
        if our_p is not None:
            self._our_ps.append(our_p)
        opp_p = _f(stats.get("opp_p"))
        if opp_p is not None:
            self._opp_ps.append(opp_p)

        # The hidden operator channel: when `log` is on, dump the raw numbers.
        if self.log_enabled and stats_string:
            send(self._format_log(board, stats, score))

        if self.muted or random.random() >= self.talk_prob:
            return

        user_msg = _describe_move(
            board, stats, score, language=self.language, opening=self.opening,
            cumulative=self._cumulative_line(),
        )

        prefix = _move_prefix(board, stats.get("ourmove"))

        def task() -> None:
            line = _chat_completion([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])
            if line:
                self._remember(line, board)
                send(f"{prefix} — {line}" if prefix else line)

        _run_async(task)

    @staticmethod
    def _format_log(board: chess.Board, stats: dict[str, str], score) -> str:
        opp = _opponent_last_move_san(board)
        cp = ""
        if score is not None:
            try:
                cp = f" eval={score.relative.score(mate_score=10000) / 100:+.2f}"
            except Exception:
                cp = ""
        parts = [f"#{board.fullmove_number}"]
        if opp:
            parts.append(f"opp={opp}")
        for k in ("P", "opp_p", "loss", "Δ", "break", "oracle"):
            if k in stats:
                parts.append(f"{k}={stats[k]}")
        return "log: " + " ".join(parts) + cp

    # ---- reactive: opponent said something --------------------------------

    def handle_keyword(self, text: str) -> str | None:
        """Handle control words. Returns an ack to send, or None if not a keyword."""
        word = text.strip().lower()
        if word in ("verbose", "chatty", "talk"):
            self.talk_prob = VERBOSE_PROB
            self.muted = False
            return "Verbose. I will narrate more of what I see."
        if word in ("quiet", "normal", "less"):
            self.talk_prob = BASE_PROB
            self.muted = False
            return "Quieter."
        if word in ("silence", "mute", "stop", "stfu", "shush"):
            self.muted = True
            return "Silent."
        # Hidden operator toggle — not advertised to opponents.
        if word == "log":
            self.log_enabled = not self.log_enabled
            return f"log {'on' if self.log_enabled else 'off'}"
        return None

    def reply(self, text: str, send, board: chess.Board | None = None) -> None:
        """Reply to a direct opponent message, grounded in the live game.

        `board` is the most recent position (reconstructed from the game in the
        react hook); falls back to the last one we saw. The model gets that
        position plus the lines it spoke earlier, so its reply stays coherent
        with what is actually happening on the board.
        """
        if self.muted or not _api_key():
            return
        board = board if board is not None else self.last_board

        ctx: list[str] = []
        if self.opening:
            ctx.append(f"Opening: {self.opening}.")
        if board is not None:
            pgn = _pgn_so_far(board)
            if pgn:
                ctx.append(f"Moves so far: {pgn}")
            last = _opponent_last_move_san(board)
            if last:
                ctx.append(f"Last move played on the board: {last}.")
        cum = self._cumulative_line()
        if cum:
            ctx.append(cum)
        if self._recent:
            ctx.append("Lines you spoke earlier this game: " + " | ".join(self._recent[-4:]))

        user_msg = (
            "You are in the middle of a game. Current context:\n"
            + "\n".join(ctx)
            + f"\n\nYour opponent just wrote to you in the chat: \"{text}\"\n"
            "Reply directly to them, in character, one sentence, in the language "
            "of their message. Stay consistent with the game so far. Do not "
            "invent numbers and never give chess advice. If the message needs no "
            "reply, answer PASS."
        )

        def task() -> None:
            line = _chat_completion([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ])
            if line:
                self._remember(line, board)
                send(line)

        _run_async(task)

    # ---- internals --------------------------------------------------------

    def _cumulative_line(self) -> str | None:
        """One line summarising how human-typical the opponent has been."""
        if not self._opp_ps:
            return None
        m = sum(self._opp_ps) / len(self._opp_ps)
        return (f"Across the game so far, your opponent's moves have averaged "
                f"{m * 100:.0f}% human-typical over {len(self._opp_ps)} moves "
                f"(higher = more predictable).")

    def _remember(self, line: str, board: chess.Board | None = None) -> None:
        tag = f"m{board.fullmove_number}: " if board is not None else ""
        self._recent.append(tag + line)
        if len(self._recent) > 8:
            self._recent.pop(0)


def board_from_moves(moves: str, initial_fen: str | None = None) -> chess.Board:
    """Reconstruct a board from a space-separated UCI move list (best-effort)."""
    board = (chess.Board(initial_fen)
             if initial_fen and initial_fen not in ("startpos", "") else chess.Board())
    for uci in (moves or "").split():
        try:
            board.push_uci(uci)
        except (ValueError, chess.IllegalMoveError):
            break
    return board


_chatters: dict[str, RorschachChatter] = {}


def get_chatter(game_id: str) -> RorschachChatter:
    """One chatter per game id, created on first use."""
    chatter = _chatters.get(game_id)
    if chatter is None:
        chatter = RorschachChatter()
        _chatters[game_id] = chatter
    return chatter
