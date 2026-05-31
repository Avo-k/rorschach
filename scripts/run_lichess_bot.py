"""Launch lichess-bot with Rorschach-specific runtime patches.

What this adds vs vanilla lichess-bot:
- Conversation.PROFILE_ALIASES (a class attribute) so chat commands
  !bal / !agg / !pru / !def switch the running engine's profile via
  UCI setoption.

The patch is a single class-attribute injection — no monkey-patching of
methods. The conversation.command branch that consumes PROFILE_ALIASES
already exists in our local lichess-bot/lib/conversation.py.

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

sys.path.insert(0, str(LICHESS_BOT_DIR))
os.chdir(LICHESS_BOT_DIR)  # config.yml + relative paths assume cwd here

from lib.conversation import ChatLine, Conversation  # noqa: E402
from lib.timer import seconds  # noqa: E402

Conversation.PROFILE_ALIASES = {
    "agg": "aggressive", "aggressive": "aggressive",
    "bal": "balanced",   "balanced": "balanced",
}


def _rorschach_command(self: Conversation, line: ChatLine, cmd: str) -> None:  # noqa: D401
    """Drop-in replacement that knows about !bal / !agg."""
    from_self = line.username == self.game.username
    is_eval = cmd.startswith("eval")
    if cmd in ("commands", "help"):
        self.send_reply(line,
                        "Modes: !bal (balanced) !agg (aggressive). "
                        "Other: !wait !name !eval !queue")
    elif cmd in ("mode", "modes"):
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

from lib.lichess_bot import start_program  # noqa: E402

if __name__ == "__main__":
    start_program()
