"""Launch lichess-bot with Rorschach-specific runtime patches.

What this adds vs vanilla lichess-bot:
- Conversation.PROFILE_ALIASES (a class attribute) so chat commands
  !bal / !agg switch the running engine's profile via UCI setoption.
- An LLM chat personality (rorschach.chat) wired through two method
  patches: EngineWrapper.play_move (proactive in-character remarks after
  our moves, grounded in the real selector stats) and Conversation.react
  (verbose/quiet/log control words + in-character replies). Cosmetic only;
  it never touches move selection, and every OpenRouter call is async with
  errors swallowed, so a dead LLM can't delay a move or break a game.

The conversation.command branch that consumes PROFILE_ALIASES already
exists in our local lichess-bot/lib/conversation.py.

Where lichess-bot lives is read from $LICHESS_BOT_DIR (the Docker image
sets it to /opt/lichess-bot); it defaults to the local dev clone.

Usage:
    cd /home/avok/code/lichess-bot
    /home/avok/code/lichess-bot/.venv/bin/python \
        /home/avok/code/rorschach/scripts/run_lichess_bot.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

LICHESS_BOT_DIR = Path(os.environ.get("LICHESS_BOT_DIR", "/home/avok/code/lichess-bot"))
RORSCHACH_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(LICHESS_BOT_DIR))
sys.path.insert(0, str(RORSCHACH_DIR))  # so `import rorschach.chat` resolves here
os.chdir(LICHESS_BOT_DIR)  # config.yml + relative paths assume cwd here

from lib.conversation import ChatLine, Conversation  # noqa: E402
from lib.engine_wrapper import EngineWrapper  # noqa: E402
from lib.timer import seconds  # noqa: E402

from rorschach.chat import get_chatter  # noqa: E402

Conversation.PROFILE_ALIASES = {
    "agg": "aggressive", "aggressive": "aggressive",
    "bal": "balanced",   "balanced": "balanced",
}


def _rorschach_command(self: Conversation, line: ChatLine, cmd: str) -> None:  # noqa: D401
    """Drop-in replacement that knows about !bal / !agg."""
    from_self = line.username == self.game.username
    is_spectator = line.room == "spectator"
    is_eval = cmd.startswith("eval")
    if cmd in ("commands", "help"):
        if is_spectator:
            # Spectators can't switch modes (that's player-room only), so don't
            # advertise !bal/!agg to them — only what actually works here.
            self.send_reply(line, "Spectator commands: !eval (how alien the last move was) !name !queue")
        else:
            self.send_reply(line,
                            "Modes: !bal (balanced) !agg (aggressive). "
                            "Other: !wait !name !eval !queue")
    elif cmd in ("mode", "modes"):
        if is_spectator:
            self.send_reply(line, "My opponent picks the mode. Type !eval to see how alien the last move was.")
        else:
            self.send_reply(line, "!bal balanced (default)  |  !agg aggressive")
    elif cmd in self.PROFILE_ALIASES and line.room == "player" and not from_self:
        profile = self.PROFILE_ALIASES[cmd]
        try:
            self.engine.engine.configure({"Profile": profile})
            self.send_reply(line, f"mode → {profile}")
        except Exception as exc:
            self.send_reply(line, f"switch failed: {exc}")
    elif cmd == "wait" and self.game.is_abortable():
        self.game.ping(seconds(60), seconds(120), seconds(120))
        self.send_reply(line, "Waiting 60 seconds...")
    elif cmd == "name":
        name = self.game.me.name
        self.send_reply(line, f"{name} running {self.engine.name()} (lichess-bot v{self.version})")
    elif is_eval and (from_self or line.room == "spectator"):
        stats = self.engine.get_stats(for_chat=True)
        self.send_reply(line, ", ".join(stats))
    elif is_eval:
        self.send_reply(line, "I don't tell that to my opponent, sorry.")
    elif cmd == "queue":
        if self.challengers:
            names = ", ".join(f"@{c.challenger.name}" for c in reversed(self.challengers))
            self.send_reply(line, f"Challenge queue: {names}")
        else:
            self.send_reply(line, "No challenges queued.")


Conversation.command = _rorschach_command  # type: ignore[assignment]


# --- LLM chat personality (cosmetic; never touches move selection) ---------
#
# Two hooks share one chatter per game (keyed by game id in rorschach.chat):
#   1. after each of our moves, maybe drop an in-character remark grounded in
#      the real selector numbers (Δ / eval loss / P_human / oracle);
#   2. on each opponent chat line, handle the verbose/quiet/log control words,
#      else reply in character.
# Both are fail-safe: any error is logged and swallowed, so chat never breaks
# the game. The OpenRouter calls are fired on daemon threads inside chat.py.

_orig_play_move = EngineWrapper.play_move


def _play_move_with_chat(self: EngineWrapper, board, game, li, *args, **kwargs):
    # play_move ends by sending the move to Lichess, so our move is already
    # on the wire before this hook runs — the chat can never delay it.
    _orig_play_move(self, board, game, li, *args, **kwargs)
    # Rorschach only chats with humans; bots flood the challenge queue and the
    # whole point (looking alien to a human) is moot against them.
    if game.opponent.is_bot:
        return
    try:
        last = self.move_commentary[-1] if self.move_commentary else {}
        get_chatter(game.id).on_move(
            board,
            last.get("string", "") or "",
            last.get("score"),
            send=lambda msg: li.chat(game.id, "player", msg),
        )
    except Exception:  # never let chat break a move
        import logging
        logging.getLogger(__name__).exception("rorschach chat: on_move hook failed")


EngineWrapper.play_move = _play_move_with_chat  # type: ignore[assignment]


_orig_react = Conversation.react


def _react_with_chat(self: Conversation, line: ChatLine) -> None:
    text = (line.text or "").strip()
    from_self = line.username == self.game.username
    # Only engage with real *human* opponent messages in the player room, and
    # never with our own bang-commands (those go through Conversation.command).
    if (text and not from_self and line.room == "player"
            and not text.startswith("!") and not self.game.opponent.is_bot):
        try:
            chatter = get_chatter(self.game.id)
            ack = chatter.handle_keyword(text)
            if ack is not None:
                self.send_reply(line, ack)
                self.messages.append(line)
                return
            # Reconstruct the live position so the reply is grounded in the
            # game's current state, not just the last move we commented on.
            from rorschach.chat import board_from_moves
            board = board_from_moves((self.game.state or {}).get("moves", ""),
                                     getattr(self.game, "initial_fen", None))
            chatter.reply(text, send=lambda msg: self.send_reply(line, msg), board=board)
        except Exception:
            import logging
            logging.getLogger(__name__).exception("rorschach chat: react hook failed")
    _orig_react(self, line)


Conversation.react = _react_with_chat  # type: ignore[assignment]


from lib.lichess_bot import start_program  # noqa: E402

if __name__ == "__main__":
    start_program()
