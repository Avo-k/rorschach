# Rorschach

A Lichess bot that plays moves which look **alien to humans** but are still **objectively sound**. It composes a strong chess engine with a human-move predictor and uses the predictor as a *negative* filter: among the engine's top candidates within an eval-loss budget, pick the move a human is least likely to play.

## Core idea (one line of math)

```
move = argmin_{m ∈ engine_topK(pos)}  P_Maia(m | pos, elo)
       subject to  eval(best) - eval(m) ≤ Δ
```

- `engine_topK`: top-K candidates from the strong engine (Patricia, MultiPV).
- `P_Maia`: probability under the human-move predictor (Maia-2) at a chosen Elo.
- `Δ`: eval-loss budget in centipawns. The single most important calibration knob.
- `K`: candidate count from the engine.
- `elo`: which human rating Maia models — the rating we want to look "weird" to.

## Components

| Layer | Tool | Role |
|---|---|---|
| Strong engine | **Patricia 5+** (UCI) | MultiPV candidate generation with cp evals. Single-threaded NNUE, MIT, ~3500 CCRL. Biased toward sharp lines, which compounds nicely with the anti-human filter. |
| Human predictor | **Maia-2** (Python, PyTorch) | Returns `move_probs: dict[uci, float]` over all legal moves, conditioned on `(elo_self, elo_oppo)`. |
| Orchestrator | Python (`rorschach/`) | Spawns Patricia as a subprocess, calls Maia-2 in-process, runs the selection rule, exposes a `MinimalEngine` to lichess-bot. |
| Lichess client | **lichess-bot** | Manages Lichess Bot API, challenges, clocks. Our orchestrator plugs in via `homemade.py`. |

## Repo layout (planned)

```
rorschach/
  rorschach/                 # Python package
    engine.py                # Patricia UCI wrapper (subprocess + UCI protocol)
    maia.py                  # Maia-2 inference wrapper
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

## Deployment target

- **Proxmox VM, CPU-only.** Probably a small LXC or VM, 2–4 vCPU, 2–4 GB RAM.
- **lichess-bot** framework, our package wrapped as a `MinimalEngine` subclass (no UCI re-implementation needed).
- BOT account upgrade via `POST /api/bot/account/upgrade` (irreversible; account must have zero rated games beforehand).

### CPU latency budget (the real risk)

- **Patricia**: single-threaded NN, fast; modest depth at low movetime is fine.
- **Maia-2**: a CNN+ViT hybrid. CPU inference per position is plausibly **tens to low-hundreds of ms** single-threaded. We only call it once per move on `K` candidates (or 1 batched call), so budget ~100–500 ms per move. Acceptable for blitz/rapid; ultrabullet is not allowed for bots anyway.
- Pin Maia-2 deps in a separate venv: `torch==2.4.0`, `numpy==2.1.3` are exact-pinned and break easily.

## Important constraints

- **Bot accounts are explicitly whitelisted** by Lichess's anti-cheat (Irwin/Kaladin). Our policy by design maximizes the exact signal those classifiers flag. That is fine for a registered BOT account; it would be an instant ban on a human account. Never run this code under a non-BOT login.
- **Do not amend Patricia's source.** Treat it as a black-box UCI binary. If we want different behavior, change selector parameters, not the engine.
- **No premature abstraction.** This is a research bot. Resist building plugin systems for "future engines" before we ship one working profile.

## Open questions to resolve as we go

- Patricia vs. Stockfish as candidate generator: Patricia's sharpness may compound with Maia anti-filtering (good), or it may cause Maia to assign sharp moves *higher* probability than expected at high Elo (bad). Run a head-to-head.
- Should `target_elo` for Maia be the opponent's actual rating, or fixed (e.g. 1900) regardless? The latter is simpler and probably good enough.
- Endgame handling: tablebase moves should bypass the selector. Add Syzygy probing later.
- Opening book: skip entirely (we want weird moves from move 1) or use a tiny sharp book?
- Time management: per-move budget, not per-game — keep it dumb at first.

## Prior art (for orientation, not duplication)

- **Maia-2** — [GitHub](https://github.com/CSSLab/maia2), [paper](https://arxiv.org/html/2409.20553v1). The discriminator.
- **Patricia** — [GitHub](https://github.com/Adam-Kulju/Patricia), [releases](https://github.com/Adam-Kulju/Patricia/releases). The candidate generator.
- **lichess-bot** — [GitHub](https://github.com/lichess-bot-devs/lichess-bot), [homemade-engine wiki](https://github.com/lichess-bot-devs/lichess-bot/wiki/Create-a-homemade-engine).
- **Detecting Fair Play Violations in Chess Using Neural Networks** (CEUR 2024) — [PDF](https://ceur-ws.org/Vol-3885/paper13.pdf). Defines the *adversarial dual* of our objective for cheat detection. Read this; their detector's positive class is exactly our policy.
- **Behavioral Stylometry in Chess** (CSSLab, NeurIPS 2021) — [arXiv](https://arxiv.org/abs/2208.01366). Background on Maia-style fingerprinting.
- **Ken Regan's engine-correlation methodology** — [PDF](https://cse.buffalo.edu/~regan/personal/JuneCLarticleKWR.pdf). The classical cheat-detection framework — useful reading for understanding what "looks like an engine" formally means.

No public Lichess bot doing exactly this composition was found at project start. The novelty is the deployment, not the math.

## Conventions for this codebase

- Python 3.11+. `uv` or `pip` + venv; pin Maia-2's deps tightly.
- One module per role (engine wrapper / Maia wrapper / selector / bot adapter). No abstract base classes until there's a second concrete implementation.
- Tests against fixed FEN sets are the primary correctness signal. Self-play and Lichess play are the primary calibration signals.
- Keep CLAUDE.md current as soon as a fact changes (e.g. when latency numbers come in, when calibration profiles are picked).
