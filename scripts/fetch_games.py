"""Fetch recent rated rapid games from Lichess for a fixed list of users.

Caches PGN files to data/games/<user>_<color>.pgn. Idempotent — re-run is safe.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

USERS = ["Avo-k", "julesder", "poisso"]
COLORS = ("white", "black")
MAX_PER_COLOR = 10
PERF_TYPES = "rapid,blitz"  # comma-separated; we'll dispatch Maia by game type
LOOKBACK_DAYS = 90
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "games"
LICHESS_URL = "https://lichess.org/api/games/user/{user}"


def fetch_one(user: str, color: str, since_ms: int) -> str:
    params = {
        "max": MAX_PER_COLOR,
        "color": color,
        "rated": "true",
        "perfType": PERF_TYPES,
        "since": since_ms,
        "clocks": "false",
        "evals": "false",
        "opening": "false",
    }
    headers = {"Accept": "application/x-chess-pgn", "User-Agent": "rorschach-bench/0.1"}
    r = requests.get(LICHESS_URL.format(user=user), params=params, headers=headers, timeout=60)
    r.raise_for_status()
    return r.text


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    since = datetime.now(tz=timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    since_ms = int(since.timestamp() * 1000)
    print(f"fetching games since {since.date()} ({PERF_TYPES}, rated only)\n")

    for user in USERS:
        for color in COLORS:
            out = DATA_DIR / f"{user}_{color}.pgn"
            pgn = fetch_one(user, color, since_ms)
            out.write_text(pgn)
            n_games = pgn.count("[Event ")
            print(f"  {user:10s} {color:5s}  {n_games:2d} games  ->  {out.relative_to(DATA_DIR.parent.parent)}")
            time.sleep(1)  # be polite to Lichess

    print("\ndone.")


if __name__ == "__main__":
    main()
