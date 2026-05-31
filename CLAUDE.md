# Rorschach

A Lichess bot that plays moves which look **alien to humans** but are still **objectively sound**. It composes a strong chess engine with a human-move predictor and uses the predictor as a *negative* filter: among the engine's top candidates within an eval-loss budget, pick the move a human is least likely to play.

## Core idea (one line of math)

```
move = argmin_{m ∈ engine_topK(pos)}  P_Maia(m | pos, elo)
       subject to  eval(best) - eval(m) ≤ Δ
```

- `engine_topK`: top-K candidates from the strong engine (Patricia, MultiPV).
- `P_Maia`: probability under the human-move predictor (Maia-3) at a chosen Elo.
- `Δ`: eval-loss budget in centipawns. The single most important calibration knob.
- `K`: candidate count from the engine.
- `elo`: which human rating Maia models — the rating we want to look "weird" to.

## Components

| Layer | Tool | Role |
|---|---|---|
| Strong engine | **Patricia 5+** (UCI) | MultiPV candidate generation with cp evals. Single-threaded NNUE, MIT, ~3500 CCRL. Biased toward sharp lines, which compounds nicely with the anti-human filter. |
| Human predictor | **Maia-3 / Chessformer** (Python, PyTorch) | Returns `move_probs: dict[uci, float]` over all legal moves, conditioned on `(elo_self, elo_oppo)` and the last 8 board positions (history reconstructed from `board.move_stack`). |
| Orchestrator | Python (`rorschach/`) | Spawns Patricia as a subprocess, calls Maia in-process, runs the selection rule, exposes a UCI shim to lichess-bot. |
| Lichess client | **lichess-bot** | Manages Lichess Bot API, challenges, clocks. Our orchestrator plugs in via `homemade.py`. |

## Repo layout (planned)

```
rorschach/
  rorschach/                 # Python package
    engine.py                # Patricia UCI wrapper (subprocess + UCI protocol)
    maia.py                  # Maia-3 inference wrapper
    selector.py              # the argmin/Δ-budget selection rule
    bot.py                   # lichess-bot MinimalEngine subclass
  bin/
    patricia                 # vendored Patricia binary (Linux x86-64, AVX2)
  configs/
    profiles.yaml            # tuned (Δ, K, elo) presets per time control
  experiments/
    eval_runner.py           # batch self-play / position-set evaluation
    notebooks/
  tests/
  pyproject.toml
  CLAUDE.md
```

## Calibration plan

The whole project is a calibration problem. Two metrics matter and they're in tension:

1. **Soundness**: average eval loss vs. engine-best, blunder rate, win rate vs. fixed-Elo opponent.
2. **Alien-ness**: average `-log P_Maia(played_move | elo)` over a game; or fraction of moves with `P_Maia < threshold`.

Calibration loop:

- Build a position set (mix of openings + tactics + quiet middlegame + endgame).
- Sweep `(Δ, K, target_elo)` and for each setting compute soundness + alien-ness.
- Plot the Pareto frontier. Pick a profile per time control.
- Validate via self-play and later against humans on Lichess.

Patricia's strength can be lowered aggressively (`UCI_Elo`, `Skill_Level`, low `depth`/`movetime`) — that is **fine and expected**. The point is not to win; it is to play *weird-but-not-losing* moves.

## Lichess deployment

`rorschach-uci` is registered as a console script (see `pyproject.toml`).
After `uv sync`, the entrypoint is at `.venv/bin/rorschach-uci` — a normal
UCI engine that any host can drive.

Wiring into [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot):

```bash
# 1. one-time bot account upgrade (zero rated games on the account first)
curl -d '' https://lichess.org/api/bot/account/upgrade \
  -H "Authorization: Bearer YOUR_BOT_TOKEN"

# 2. clone and install lichess-bot beside this repo
git clone https://github.com/lichess-bot-devs/lichess-bot.git
cd lichess-bot
pip install -r requirements.txt

# 3. cp our example config and edit the token
cp ../rorschach/configs/lichess-bot.yml.example config.yml

# 4. ensure LICHESS_TOKEN is in rorschach/.env (for the opening explorer)

# 5. run
python lichess-bot.py
```

The config points lichess-bot at `rorschach/.venv/bin/rorschach-uci`, which
loads Patricia + Maia + the explorer on the first `isready`. UCI options
expose profile, fixed-time override, target Elo, and `MaiaType` (one of
`maia3-5m` (default), `maia3-23m`, `maia3-79m`). The Maia-3 checkpoint is
downloaded from Hugging Face on first use and cached under
`~/.cache/huggingface/`.

## Deployment target

- **Proxmox VM, CPU-only.** Probably a small LXC or VM, 2–4 vCPU, 2–4 GB RAM.
- **lichess-bot** framework, our package wrapped as a `MinimalEngine` subclass (no UCI re-implementation needed).
- BOT account upgrade via `POST /api/bot/account/upgrade` (irreversible; account must have zero rated games beforehand).

### CPU latency budget (the real risk)

- **Patricia**: single-threaded NN, fast; modest depth at low movetime is fine.
- **Maia-3 (default 5M)**: encoder-only transformer (Chessformer) with Geometric Attention Bias. Single forward per move; sized for CPU at 5M params. 23M / 79M variants exist (better accuracy, much slower on CPU). Budget unknown until benchmarked — measure before committing to a profile.
- `torch==2.4.0` is pinned to match Maia-3's tested baseline.

## Important constraints

- **Bot accounts are explicitly whitelisted** by Lichess's anti-cheat (Irwin/Kaladin). Our policy by design maximizes the exact signal those classifiers flag. That is fine for a registered BOT account; it would be an instant ban on a human account. Never run this code under a non-BOT login.
- **Do not amend Patricia's source.** Treat it as a black-box UCI binary. If we want different behavior, change selector parameters, not the engine.
- **No premature abstraction.** This is a research bot. Resist building plugin systems for "future engines" before we ship one working profile.

## Open questions to resolve as we go

- Patricia vs. Stockfish as candidate generator: Patricia's sharpness may compound with Maia anti-filtering (good), or it may cause Maia to assign sharp moves *higher* probability than expected at high Elo (bad). Run a head-to-head.
- ~~Should `target_elo` for Maia be the opponent's actual rating, or fixed?~~ **Resolved: opponent's actual rating by default.** The `Elo` UCI option defaults to `0` = auto, which sets Maia's `elo_self` (and `elo_oppo`) to the opponent's rating from `UCI_Opponent`, falling back to `DEFAULT_ELO` (1900) when unknown. A non-zero `Elo` pins a fixed bucket.
- Endgame handling: tablebase moves should bypass the selector. Add Syzygy probing later.
- Opening book: skip entirely (we want weird moves from move 1) or use a tiny sharp book?
- Time management: per-move budget, not per-game — keep it dumb at first.

## Prior art (for orientation, not duplication)

- **Maia-3 / Chessformer** — [GitHub](https://github.com/CSSLab/maia3), [models](https://huggingface.co/collections/UofTCSSLab/maia3), [paper](https://arxiv.org/abs/2605.19091), [blog](https://lichess.org/@/ashtonanderson/blog/introducing-maia-3-free-and-open-source/vCPPRtX3). The discriminator. 57.1% move-match accuracy across Lichess blitz 600–2600.
- **Patricia** — [GitHub](https://github.com/Adam-Kulju/Patricia), [releases](https://github.com/Adam-Kulju/Patricia/releases). The candidate generator.
- **lichess-bot** — [GitHub](https://github.com/lichess-bot-devs/lichess-bot), [homemade-engine wiki](https://github.com/lichess-bot-devs/lichess-bot/wiki/Create-a-homemade-engine).
- **Detecting Fair Play Violations in Chess Using Neural Networks** (CEUR 2024) — [PDF](https://ceur-ws.org/Vol-3885/paper13.pdf). Defines the *adversarial dual* of our objective for cheat detection. Read this; their detector's positive class is exactly our policy.
- **Behavioral Stylometry in Chess** (CSSLab, NeurIPS 2021) — [arXiv](https://arxiv.org/abs/2208.01366). Background on Maia-style fingerprinting.
- **Ken Regan's engine-correlation methodology** — [PDF](https://cse.buffalo.edu/~regan/personal/JuneCLarticleKWR.pdf). The classical cheat-detection framework — useful reading for understanding what "looks like an engine" formally means.

No public Lichess bot doing exactly this composition was found at project start. The novelty is the deployment, not the math.

## Conventions for this codebase

- Python 3.11+. `uv` or `pip` + venv.
- One module per role (engine wrapper / Maia wrapper / selector / bot adapter). No abstract base classes until there's a second concrete implementation.
- Tests against fixed FEN sets are the primary correctness signal. Self-play and Lichess play are the primary calibration signals.
- Keep CLAUDE.md current as soon as a fact changes (e.g. when latency numbers come in, when calibration profiles are picked).
