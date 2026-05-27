# Rorschach

A Lichess bot that plays moves which look **alien to humans** but are still
**objectively sound**. It pipes a strong chess engine through a human-move
predictor and uses the predictor as a *negative* filter: among the engine's top
candidates within an eval-loss budget, pick the move a human is least likely to
play.

```
move = argmin_{m ∈ engine_topK(pos)}  P_Maia(m | pos, elo)
       subject to  eval(best) − eval(m) ≤ Δ
```

- `engine_topK` — top-K candidates from Patricia (MultiPV, cp-scored).
- `P_Maia` — probability under the human-move predictor (Maia-3) at a chosen Elo.
- `Δ` — eval-loss budget in centipawns. The dominant calibration knob.

## What's interesting about this

The math is one line; the calibration is the project. Two metrics fight each
other:

1. **Soundness** — average eval loss vs engine-best, blunder rate, win rate vs a
   fixed-Elo opponent.
2. **Alien-ness** — average `−log P_Maia(played | elo)` over a game; fraction of
   moves with `P_Maia < threshold`.

Δ trades them off. Profiles bundle (Δ-shape, K, target Elo) per time control.

## Components

| Layer | Tool | Role |
|---|---|---|
| Strong engine | **[Patricia 5+](https://github.com/Adam-Kulju/Patricia)** | MultiPV candidate generation with cp evals. Single-threaded NNUE, MIT, ~3500 CCRL. Biased toward sharp lines — compounds with the anti-human filter. |
| Human predictor | **[Maia-3 / Chessformer](https://github.com/CSSLab/maia3)** | Returns `{uci → P(move)}` over all legal moves, conditioned on `(elo_self, elo_oppo)` and the last 8 board positions. |
| Lichess oracle | **[Lichess Opening Explorer](https://lichess.org/api#tag/Opening-Explorer)** | When the position has ≥10 games in the DB, real human frequencies replace Maia's prediction (zero-played moves get P=0 — maximally alien and never made up). |
| Orchestrator | This repo | Spawns Patricia as a UCI subprocess, calls Maia in-process, runs the selector, exposes a UCI shim for any host. |
| Lichess client | **[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot)** | Manages the Bot API, challenges, clocks. We plug in via UCI. |

## The narrative-break selector

Maia-3 conditions on game history, which lets us score *two* probabilities per
candidate:

- `P_with(m)`   — likelihood given the position **and** the last 8 plies.
- `P_without(m)` — likelihood from the current position alone.

`break(m) = P_without(m) − P_with(m)` is positive when the way we got here
made `m` less likely than the static position would suggest — the move
**breaks the game's narrative**. The narrative-aware selector picks:

```
argmin (1+λ)·P_with(m) − λ·P_without(m)   subject to the same Δ budget
```

λ=0 is plain Maia-3 selection. λ=1 (the `narrative` profile) rewards
history-induced rarity equally to absolute rarity. Costs 2× Maia inference
per move (~+50–100 ms on CPU at 5M params); fits the blitz/rapid budget.

## Repo layout

```
rorschach/
  rorschach/                   # Python package
    engine.py                  # Patricia UCI wrapper (subprocess + UCI protocol)
    maia.py                    # Maia-3 inference wrapper
    explorer.py                # Lichess Opening Explorer client
    selector.py                # adaptive + narrative selection rules
    bot.py                     # composition root: engine + predictor + selector
    uci.py                     # UCI shim — `rorschach-uci` entry point
  bin/
    patricia                   # vendored Patricia binary (Linux x86-64, AVX2)
  configs/
    lichess-bot.yml.example    # example lichess-bot config
  tests/
    test_selector.py           # pure-logic tests for the selection rule
  CLAUDE.md                    # internal research notes / agent guidance
```

## Quick start

```bash
# 1. install deps (Maia-3 ships from git; first run downloads the checkpoint)
uv sync

# 2. sanity check
uv run pytest -q

# 3. drive the engine via UCI for one move
printf 'uci\nsetoption name MaiaType value maia3-5m\nsetoption name Profile value narrative\nisready\nposition startpos moves e2e4 c7c5\ngo movetime 200\nquit\n' \
  | uv run rorschach-uci
```

## Lichess deployment

`rorschach-uci` is a normal UCI engine. Any host that speaks UCI can drive it.
To run as a Lichess bot:

```bash
# 1. one-time bot account upgrade (account must have zero rated games first)
curl -d '' https://lichess.org/api/bot/account/upgrade \
  -H "Authorization: Bearer YOUR_BOT_TOKEN"

# 2. clone lichess-bot next to this repo and install its deps
git clone https://github.com/lichess-bot-devs/lichess-bot.git
cd lichess-bot && pip install -r requirements.txt

# 3. copy our example config and edit the token
cp ../rorschach/configs/lichess-bot.yml.example config.yml

# 4. put LICHESS_TOKEN in rorschach/.env (for the explorer; bot token is separate)
# 5. run
python lichess-bot.py
```

The example config points lichess-bot at `rorschach/.venv/bin/rorschach-uci`.
On the first `isready`, the shim loads Patricia + Maia + the explorer.

### UCI options

| Option | Type | Values | Default |
|---|---|---|---|
| `Profile` | combo | `balanced`, `aggressive`, `narrative` | `balanced` |
| `MaiaType` | combo | `maia3-5m`, `maia3-23m`, `maia3-79m` | `maia3-5m` |
| `Elo` | spin | 600..2600 | 1900 |
| `TimeMs` | spin | 0..5000; 0 = derive from clock | 0 |

Per-move info line includes `Δ`, eval `loss`, `P` (Maia of chosen), optional
`break` (narrative score for the chosen move), and `oracle` ∈ {`E`=explorer,
`M`=Maia, `N`=Maia + narrative break}.

## Deployment target

CPU-only Proxmox VM, 2–4 vCPU, 2–4 GB RAM. The 5M Maia-3 model is the
CPU-friendly default; 23M / 79M variants exist (more accurate, slower).

## Important constraints

- **Bot accounts are explicitly whitelisted** by Lichess's anti-cheat
  (Irwin/Kaladin). This bot's policy by design maximises the exact signal
  those classifiers flag. That's fine for a registered BOT account; it would
  be an instant ban on a human account. Never run this code under a non-BOT
  login.
- **Patricia is a black box.** If we want different behavior, change selector
  parameters, not the engine.

## Prior art

- **Maia-3 / Chessformer** — [github](https://github.com/CSSLab/maia3) · [models](https://huggingface.co/collections/UofTCSSLab/maia3) · [paper](https://arxiv.org/abs/2605.19091) · [blog](https://lichess.org/@/ashtonanderson/blog/introducing-maia-3-free-and-open-source/vCPPRtX3)
- **Patricia** — [github](https://github.com/Adam-Kulju/Patricia)
- **lichess-bot** — [github](https://github.com/lichess-bot-devs/lichess-bot)
- **Detecting Fair Play Violations in Chess Using Neural Networks** (CEUR 2024) — [pdf](https://ceur-ws.org/Vol-3885/paper13.pdf). The adversarial dual of this project's objective: their positive class is exactly our policy.
- **Behavioral Stylometry in Chess** (CSSLab, NeurIPS 2021) — [arXiv](https://arxiv.org/abs/2208.01366)
- **Ken Regan's engine-correlation methodology** — [pdf](https://cse.buffalo.edu/~regan/personal/JuneCLarticleKWR.pdf)

No public Lichess bot doing exactly this composition was found at project
start. The novelty is the deployment, not the math.
