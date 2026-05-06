"""Parse cached PGNs, extract every position where the target user is to move.

Output: data/positions.jsonl  with one JSON object per position, fields:
  fen, played_uci, elo_self, elo_oppo, user, color, perf_type, game_id, ply, total_plies
"""
from __future__ import annotations

import json
from pathlib import Path

import chess
import chess.pgn

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GAMES_DIR = DATA_DIR / "games"
OUT = DATA_DIR / "positions.jsonl"

USERS = ["Avo-k", "julesder", "poisso"]
COLORS = ("white", "black")
SKIP_OPENING_PLIES = 6  # 3 moves each side


def parse_perf_type(headers: dict[str, str]) -> str:
    """Lichess PGN puts time control in 'Event' (e.g. 'Rated rapid game').
    Fall back to 'Variant' / TimeControl heuristic if needed.
    """
    event = headers.get("Event", "").lower()
    for kind in ("rapid", "blitz", "bullet", "classical"):
        if kind in event:
            return kind
    # Heuristic from TimeControl seconds (e.g. "300+3" = blitz, "600+0" = rapid)
    tc = headers.get("TimeControl", "")
    if "+" in tc:
        try:
            base = int(tc.split("+")[0])
            if base < 180: return "bullet"
            if base < 480: return "blitz"
            if base < 1500: return "rapid"
            return "classical"
        except ValueError:
            pass
    return "unknown"


def extract_from_pgn(pgn_path: Path, target_user: str, target_color: str) -> list[dict]:
    out = []
    target_is_white = target_color == "white"
    with pgn_path.open() as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            h = game.headers
            try:
                we, be = int(h["WhiteElo"]), int(h["BlackElo"])
            except (KeyError, ValueError):
                continue

            # sanity: target user is on the expected color
            white = h.get("White", "").lower()
            black = h.get("Black", "").lower()
            target_lc = target_user.lower()
            if target_is_white and white != target_lc:
                continue
            if not target_is_white and black != target_lc:
                continue

            game_id = h.get("Site", "").rsplit("/", 1)[-1]
            perf = parse_perf_type(dict(h))

            elo_self, elo_oppo = (we, be) if target_is_white else (be, we)

            board = game.board()
            mainline = list(game.mainline_moves())
            total_plies = len(mainline)
            for ply, move in enumerate(mainline):
                if ply >= SKIP_OPENING_PLIES and (board.turn == chess.WHITE) == target_is_white:
                    out.append({
                        "fen": board.fen(),
                        "played_uci": move.uci(),
                        "elo_self": elo_self,
                        "elo_oppo": elo_oppo,
                        "user": target_user,
                        "color": target_color,
                        "perf_type": perf,
                        "game_id": game_id,
                        "ply": ply,
                        "total_plies": total_plies,
                    })
                board.push(move)
    return out


def main() -> None:
    all_positions: list[dict] = []
    perf_counts: dict[str, int] = {}
    for user in USERS:
        for color in COLORS:
            pgn = GAMES_DIR / f"{user}_{color}.pgn"
            positions = extract_from_pgn(pgn, user, color)
            print(f"  {user:10s} {color:5s}: {len(positions):4d} positions")
            for p in positions:
                perf_counts[p["perf_type"]] = perf_counts.get(p["perf_type"], 0) + 1
            all_positions.extend(positions)

    OUT.write_text("\n".join(json.dumps(p) for p in all_positions) + "\n")
    print(f"\ntotal: {len(all_positions)} positions  ->  {OUT.relative_to(DATA_DIR.parent)}")
    print(f"by perf_type: {perf_counts}")


if __name__ == "__main__":
    main()
