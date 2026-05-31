"""Retroactive chat-model bake-off (v2).

Pulls real games rorschach-bot played against *humans* on Lichess, replays a
diverse set of positions through the *actual* selector, runs one extra Maia
pass to score how predictable/surprising the opponent's move was, builds an
enriched prompt (opening + PGN + our move's alien-ness + the opponent-move
signal + reply language + a trigger reason), and asks each candidate model for
its one-sentence remark — letting it answer PASS when nothing is worth saying.

It records token usage and OpenRouter cost per call and prints a cost table
extrapolating to 1 / 10 / 100 games.

Output: experiments/chat_bakeoff_output.md (+ stdout).

Run:
    .venv/bin/python experiments/chat_model_bakeoff.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import chess
import chess.engine
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rorschach import chat  # noqa: E402  (also loads .env)
from rorschach.bot import rorschach_move  # noqa: E402
from rorschach.engine import PatriciaEngine  # noqa: E402
from rorschach.explorer import OpeningExplorer  # noqa: E402
from rorschach.maia import make_predictor  # noqa: E402

MODELS = [
    "google/gemini-3.5-flash",
    "google/gemini-3.1-flash-lite",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "qwen/qwen3.6-35b-a3b",
]

BOT_USER = "rorschach-bot"
N_GAMES = 8                 # human games to scan
PLIES_PER_GAME = 6          # candidate positions sampled per game
POOL_TARGET = 22            # evaluate at most this many candidates (selector cost)
PER_BUCKET = 2              # positions to keep per trigger bucket for the LLM
TIME_MS = 400               # selector budget per position
ELO = 1700                  # Maia bucket for the human-move pass
LANGUAGE = "English"        # reply language put in the prompt (prod learns it live)
MAX_TOKENS = 4096           # high cap so no reasoning model gets starved
COMMENTS_PER_GAME = 6       # assumption for the cost extrapolation table
OUT = Path(__file__).resolve().parent / "chat_bakeoff_output.md"


# --- data --------------------------------------------------------------------

def fetch_human_games(token: str, n: int) -> list[dict]:
    h = {"Authorization": f"Bearer {token}", "Accept": "application/x-ndjson"}
    r = requests.get(
        f"https://lichess.org/api/games/user/{BOT_USER}",
        headers=h, params={"max": 40, "moves": "true", "opening": "true"}, timeout=30,
    )
    r.raise_for_status()
    games = []
    for line in r.text.splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        ps = g.get("players", {})

        def is_bot(side: str) -> bool:
            p = ps.get(side, {})
            return p.get("aiLevel") is not None or p.get("user", {}).get("title") == "BOT"

        w_name = ps.get("white", {}).get("user", {}).get("name", "")
        bot_is_white = w_name.lower() == BOT_USER.lower()
        opp_side = "black" if bot_is_white else "white"
        if is_bot(opp_side):
            continue
        if len(g.get("moves", "").split()) < 24:
            continue
        g["_bot_white"] = bot_is_white
        g["_opp_name"] = ps.get(opp_side, {}).get("user", {}).get("name", "?")
        g["_opening"] = (g.get("opening") or {}).get("name")
        games.append(g)
        if len(games) >= n:
            break
    return games


def candidate_positions(game: dict) -> list[chess.Board]:
    """Boards where it is Rorschach's turn, sampled across the game."""
    rorschach_color = chess.WHITE if game["_bot_white"] else chess.BLACK
    sans = game["moves"].split()
    boards: list[chess.Board] = []
    board = chess.Board()
    for i, san in enumerate(sans):
        if board.turn == rorschach_color and i >= 6:
            boards.append(board.copy())
        try:
            board.push_san(san)
        except ValueError:
            break
    if not boards:
        return []
    out, seen = [], set()
    for k in range(PLIES_PER_GAME):
        idx = min(len(boards) - 1, int((k + 0.5) / PLIES_PER_GAME * len(boards)))
        if idx not in seen:
            seen.add(idx)
            out.append(boards[idx])
    return out


# --- signals -----------------------------------------------------------------

def human_move_signal(maia, board: chess.Board) -> dict | None:
    """Maia's read on the move the opponent just played to reach `board`."""
    if not board.move_stack:
        return None
    before = board.copy()
    hm = before.pop()
    try:
        probs, _ = maia.predict(before, ELO, ELO)
    except Exception:
        return None
    hp = float(probs.get(hm.uci(), 0.0))
    rank = 1 + sum(1 for v in probs.values() if v > hp)
    return {"san": before.san(hm), "p": hp, "rank": rank}


def label_for(p: float) -> str:
    if p >= 0.55:
        return "almost everyone plays this — utterly predictable"
    if p >= 0.30:
        return "the popular choice"
    if p >= 0.10:
        return "a fairly common choice"
    if p >= 0.03:
        return "an uncommon choice"
    return "a rare, original choice"


def cumulative_opp_predictability(maia, board: chess.Board,
                                  rorschach_color: bool) -> tuple[float, int] | None:
    """Mean Maia probability of the opponent's moves over the whole game so far."""
    replay = chess.Board()
    ps: list[float] = []
    for mv in board.move_stack:
        if replay.turn != rorschach_color:  # an opponent move
            try:
                probs, _ = maia.predict(replay, ELO, ELO)
                ps.append(float(probs.get(mv.uci(), 0.0)))
            except Exception:
                pass
        replay.push(mv)
    if not ps:
        return None
    return sum(ps) / len(ps), len(ps)


def classify(our_p: float, hp: float) -> tuple[str, str]:
    """(bucket, note) — what, if anything, is notable about this position."""
    if our_p < 0.04:
        return ("alien_self",
                f"Your own move is one only ~{our_p*100:.1f}% of humans would "
                f"play — genuinely alien.")
    if hp >= 0.60:
        return ("predictable_human",
                f"Your opponent played the obvious move (~{hp*100:.0f}% of "
                f"humans do); the statistics saw it coming.")
    if hp < 0.05:
        return ("surprising_human",
                "Your opponent just played something genuinely rare for their level.")
    return ("quiet",
            "Nothing especially notable here — only speak if you truly see something.")


# --- prompt ------------------------------------------------------------------

def build_prompt(item: dict) -> str:
    oracle = chat._ORACLE_NAME.get(item["oracle"], "the human-move model")
    L = [f"Move {item['fullmove']}. You play {item['color']}."]
    if item["opening"]:
        L.append(f"Opening: {item['opening']}.")
    if item["pgn"]:
        L.append(f"Moves so far: {item['pgn']}")
    L.append(
        f"Your opponent just played {item['opp_san']} — a human of their level "
        f"plays that about {item['hp']*100:.1f}% of the time "
        f"(rank {item['hp_rank']} most likely; {item['hp_label']})."
    )
    L.append(
        f"You replied {item['our_san']} — only about {item['our_p']*100:.1f}% "
        f"of humans would ({oracle}); you spent {item['loss']} of {item['delta']} "
        f"centipawns of your eval budget on it."
    )
    L.append(f"Position evaluation now: {item['eval']:+.2f} pawns in your favor.")
    if item.get("cumulative"):
        L.append(item["cumulative"])
    L.append(f"What's notable: {item['note']}")
    L.append(f"Reply in {item['language']}.")
    L.append("Write your one-sentence remark, or PASS if nothing is worth saying.")
    return "\n".join(L)


# --- model call --------------------------------------------------------------

def call_model(model: str, user_msg: str) -> dict:
    key = os.environ["OPENROUTER_API_KEY"]
    try:
        r = requests.post(
            chat.OPENROUTER_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": chat.SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": chat.TEMPERATURE,
                "reasoning": {"effort": chat.REASONING_EFFORT},
            },
            timeout=60,
        )
        j = r.json()
        if "choices" not in j:
            return {"text": f"[ERR {r.status_code}: {str(j.get('error', j))[:140]}]"}
        raw = (j["choices"][0]["message"]["content"] or "").strip().strip('"').strip()
        line = raw.splitlines()[0].strip() if raw else ""
        passed = line.rstrip(".!").strip().upper() == chat.PASS_TOKEN
        u = j.get("usage", {})
        ctd = u.get("completion_tokens_details", {})
        ptd = u.get("prompt_tokens_details", {})
        return {
            "text": "PASS" if passed else (line or "[empty]"),
            "passed": passed,
            "prompt_tokens": u.get("prompt_tokens", 0),
            "cached_tokens": ptd.get("cached_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0),
            "reasoning_tokens": ctd.get("reasoning_tokens", 0),
            "cost": float(u.get("cost", 0.0) or 0.0),
        }
    except Exception as exc:  # noqa: BLE001
        return {"text": f"[EXC {exc!r}]"}


# --- main --------------------------------------------------------------------

def main() -> None:
    if not os.environ.get("LICHESS_TOKEN") or not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("need LICHESS_TOKEN and OPENROUTER_API_KEY in .env")

    print("fetching human games ...")
    games = fetch_human_games(os.environ["LICHESS_TOKEN"], N_GAMES)
    print(f"  {len(games)} games")

    print("loading Patricia + Maia (5m) + explorer ...")
    engine = PatriciaEngine()
    maia = make_predictor("maia3-5m", device="cpu")
    explorer = OpeningExplorer()

    print("evaluating candidate positions (selector + human-move Maia pass) ...")
    pool: list[dict] = []
    for g in games:
        for board in candidate_positions(g):
            if len(pool) >= POOL_TARGET:
                break
            hs = human_move_signal(maia, board)
            if hs is None:
                continue
            move, info = rorschach_move(
                board, engine, maia, explorer=explorer,
                profile="balanced", time_ms=TIME_MS, elo_self=ELO, elo_oppo=ELO,
            )
            bucket, note = classify(info.p_human, hs["p"])
            pool.append({
                "game": g["id"], "opp": g["_opp_name"], "opening": g["_opening"],
                "fullmove": board.fullmove_number,
                "color": "White" if board.turn else "Black",
                "pgn": chat._pgn_so_far(board),
                "opp_san": hs["san"], "hp": hs["p"], "hp_rank": hs["rank"],
                "hp_label": label_for(hs["p"]),
                "our_san": board.san(move), "our_p": info.p_human,
                "loss": info.eval_loss, "delta": info.delta_used,
                "oracle": info.oracle, "eval": info.best_cp / 100.0,
                "language": LANGUAGE, "bucket": bucket, "note": note,
                "board": board, "rorschach_white": board.turn,
            })
            print(f"  [{bucket:18}] {g['id']} mv{board.fullmove_number} "
                  f"opp={hs['san']}({hs['p']*100:.0f}%) ours={board.san(move)}"
                  f"({info.p_human*100:.1f}%)")
        if len(pool) >= POOL_TARGET:
            break
    engine.quit()

    # Keep a diverse subset across buckets.
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for it in pool:
        by_bucket[it["bucket"]].append(it)
    selected: list[dict] = []
    for bucket in ("alien_self", "predictable_human", "surprising_human", "quiet"):
        selected.extend(by_bucket.get(bucket, [])[:PER_BUCKET])
    print(f"\nselected {len(selected)} positions across "
          f"{len([b for b in by_bucket if by_bucket[b]])} buckets")

    # Running opponent-predictability score over each game (extra Maia passes).
    for it in selected:
        cum = cumulative_opp_predictability(maia, it["board"], it["rorschach_white"])
        if cum:
            mean, n = cum
            it["cumulative"] = (f"Across the game so far, your opponent's moves "
                                f"have averaged {mean*100:.0f}% human-typical over "
                                f"{n} moves (higher = more predictable).")

    # Query every model on every selected position.
    usage: dict[str, list[dict]] = defaultdict(list)
    lines = ["# Chat model bake-off (v2)\n",
             f"_{len(selected)} real positions from rorschach-bot vs human games. "
             f"Enriched prompt (opening + PGN + opponent-move predictability + "
             f"trigger), models may answer **PASS**._\n"]
    for i, p in enumerate(selected, 1):
        prompt = build_prompt(p)
        print(f"\n=== position {i}/{len(selected)} [{p['bucket']}] ===")
        lines += [
            f"## Position {i} — `{p['bucket']}` — game {p['game']} vs {p['opp']}, "
            f"move {p['fullmove']}\n",
            f"Opening: {p['opening']}. Opponent played **{p['opp_san']}** "
            f"({p['hp']*100:.0f}% human, {p['hp_label']}); we replied "
            f"**{p['our_san']}** ({p['our_p']*100:.1f}% human, {p['loss']}cp lost), "
            f"eval {p['eval']:+.2f}.\n",
            "<details><summary>full prompt</summary>\n\n```\n" + prompt + "\n```\n</details>\n",
        ]
        for model in MODELS:
            res = call_model(model, prompt)
            usage[model].append(res)
            cost = res.get("cost")
            extra = (f"  _(r={res.get('reasoning_tokens')}t, ${cost:.4f})_"
                     if cost is not None and "cost" in res else "")
            print(f"  {model:32} {res['text']}{extra}")
            lines.append(f"- **{model}** — {res['text']}{extra}")
        lines.append("")

    # Cost table.
    lines += ["---\n", "## Cost & tokens per model\n",
              f"Averaged over {len(selected)} calls each. Reasoning runs even when "
              f"the model answers PASS, so PASS still costs. Extrapolation assumes "
              f"**{COMMENTS_PER_GAME} spoken comments/game**.\n",
              "| model | avg prompt | cached | avg reason | avg out | $/comment | "
              "$/game | $/10 games | $/100 games | PASS rate |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    print("\n=== cost summary ===")
    for model in MODELS:
        rows = [u for u in usage[model] if "cost" in u]
        if not rows:
            lines.append(f"| {model} | — errors only — |||||||||")
            continue
        n = len(rows)
        avg = lambda k: sum(r.get(k, 0) for r in rows) / n  # noqa: E731
        c = avg("cost")
        npass = sum(1 for r in usage[model] if r.get("passed"))
        row = (f"| {model} | {avg('prompt_tokens'):.0f} | {avg('cached_tokens'):.0f} "
               f"| {avg('reasoning_tokens'):.0f} | {avg('completion_tokens'):.0f} "
               f"| ${c:.4f} | ${c*COMMENTS_PER_GAME:.3f} | ${c*COMMENTS_PER_GAME*10:.2f} "
               f"| ${c*COMMENTS_PER_GAME*100:.2f} | {npass}/{len(usage[model])} |")
        lines.append(row)
        print(f"  {model:32} ${c:.4f}/comment  -> ${c*COMMENTS_PER_GAME*100:.2f}/100 games")

    OUT.write_text("\n".join(lines))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
