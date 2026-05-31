# Chat model bake-off (v2)

_6 real positions from rorschach-bot vs human games. Enriched prompt (opening + PGN + opponent-move predictability + trigger), models may answer **PASS**._

## Position 1 — `alien_self` — game OUDCDszV vs Avo-k, move 5

Opening: Caro-Kann Defense. Opponent played **e5** (72% human, almost everyone plays this — utterly predictable); we replied **h5** (0.0% human, 0cp lost), eval -0.26.

<details><summary>full prompt</summary>

```
Move 5. You play Black.
Opening: Caro-Kann Defense.
Moves so far: 1. e4 c6 2. d4 h6 3. Nc3 Qc7 4. f4 d5 5. e5
Your opponent just played e5 — a human of their level plays that about 71.9% of the time (rank 1 most likely; almost everyone plays this — utterly predictable).
You replied h5 — only about 0.0% of humans would (real Lichess game frequencies); you spent 0 of 50 centipawns of your eval budget on it.
Position evaluation now: -0.26 pawns in your favor.
Across the game so far, your opponent's moves have averaged 40% human-typical over 5 moves (higher = more predictable).
What's notable: Your own move is one only ~0.0% of humans would play — genuinely alien.
Reply in English.
Write your one-sentence remark, or PASS if nothing is worth saying.
```
</details>

- **google/gemini-3.5-flash** — Seventy percent of players choose e5, but virtually zero percent reply with h5.  _(r=526t, $0.0057)_
- **google/gemini-3.1-flash-lite** — You play the book moves as if you were following a map, step by step.  _(r=138t, $0.0004)_
- **deepseek/deepseek-v4-pro** — e5 was the move everyone makes; h5 is the move almost no one does.  _(r=271t, $0.0019)_
- **deepseek/deepseek-v4-flash** — Your e5, 71.9% predictable; my h5, 0.0% human.  _(r=237t, $0.0002)_
- **qwen/qwen3.6-35b-a3b** — 5. e5 takes the 71.9% line, exactly as predicted.  _(r=2555t, $0.0031)_

## Position 2 — `alien_self` — game OUDCDszV vs Avo-k, move 8

Opening: Caro-Kann Defense. Opponent played **O-O** (19% human, a fairly common choice); we replied **Bf5** (2.3% human, 5cp lost), eval -0.62.

<details><summary>full prompt</summary>

```
Move 8. You play Black.
Opening: Caro-Kann Defense.
Moves so far: 1. e4 c6 2. d4 h6 3. Nc3 Qc7 4. f4 d5 5. e5 h5 6. Nf3 g6 7. Bd3 Na6 8. O-O
Your opponent just played O-O — a human of their level plays that about 19.0% of the time (rank 2 most likely; a fairly common choice).
You replied Bf5 — only about 2.3% of humans would (the Maia human-move model); you spent 5 of 50 centipawns of your eval budget on it.
Position evaluation now: -0.62 pawns in your favor.
Across the game so far, your opponent's moves have averaged 43% human-typical over 8 moves (higher = more predictable).
What's notable: Your own move is one only ~2.3% of humans would play — genuinely alien.
Reply in English.
Write your one-sentence remark, or PASS if nothing is worth saying.
```
</details>

- **google/gemini-3.5-flash** — Only 2.3% of humans would play Bf5, yet the board favors me at -0.62.  _(r=479t, $0.0054)_
- **google/gemini-3.1-flash-lite** — You chose the standard castle, ignoring the Bishop's obscure path to f5 entirely.  _(r=123t, $0.0004)_
- **deepseek/deepseek-v4-pro** — Just 2.3% of humans would consider Bf5 here.  _(r=490t, $0.0025)_
- **deepseek/deepseek-v4-flash** — Only 2.3% of humans would find Bf5 here — yet here we are.  _(r=132t, $0.0001)_
- **qwen/qwen3.6-35b-a3b** — A 19.0 percent typical O-O, met with a 2.3 percent rarity I calculated for five.  _(r=3099t, $0.0040)_

## Position 3 — `predictable_human` — game OUDCDszV vs Avo-k, move 11

Opening: Caro-Kann Defense. Opponent played **exf7+** (92% human, almost everyone plays this — utterly predictable); we replied **Kxf7** (98.7% human, 0cp lost), eval -1.03.

<details><summary>full prompt</summary>

```
Move 11. You play Black.
Opening: Caro-Kann Defense.
Moves so far: 1. e4 c6 2. d4 h6 3. Nc3 Qc7 4. f4 d5 5. e5 h5 6. Nf3 g6 7. Bd3 Na6 8. O-O Bf5 9. Bxf5 gxf5 10. e6 Qc8 11. exf7+
Your opponent just played exf7+ — a human of their level plays that about 91.6% of the time (rank 1 most likely; almost everyone plays this — utterly predictable).
You replied Kxf7 — only about 98.7% of humans would (the Maia human-move model); you spent 0 of 50 centipawns of your eval budget on it.
Position evaluation now: -1.03 pawns in your favor.
Across the game so far, your opponent's moves have averaged 46% human-typical over 11 moves (higher = more predictable).
What's notable: Your opponent played the obvious move (~92% of humans do); the statistics saw it coming.
Reply in English.
Write your one-sentence remark, or PASS if nothing is worth saying.
```
</details>

- **google/gemini-3.5-flash** — You chose the ninety-one percent move, dropping your evaluation to minus 1.03.  _(r=778t, $0.0081)_
- **google/gemini-3.1-flash-lite** — You played the move 91.6% of your peers choose, precisely on cue.  _(r=603t, $0.0011)_
- **deepseek/deepseek-v4-pro** — That check came straight out of the statistics.  _(r=179t, $0.0005)_
- **deepseek/deepseek-v4-flash** — Ninety-one percent of humans play exf7+, and you followed the script.  _(r=62t, $0.0001)_
- **qwen/qwen3.6-35b-a3b** — exf7+ completed the 91.6% prediction exactly as forecast.  _(r=3319t, $0.0034)_

## Position 4 — `predictable_human` — game 4RX5NaTI vs julesder, move 13

Opening: Caro-Kann Defense. Opponent played **Bxh5+** (82% human, almost everyone plays this — utterly predictable); we replied **Kf8** (84.5% human, 0cp lost), eval +1.34.

<details><summary>full prompt</summary>

```
Move 13. You play Black.
Opening: Caro-Kann Defense.
Moves so far: 1. e4 c6 2. d4 h6 3. f4 h5 4. e5 Na6 5. Bd3 g6 6. Nf3 d6 7. O-O Nb4 8. Be2 Bg7 9. c3 Na6 10. Ng5 Nc7 11. Nxf7 Kxf7 12. f5 gxf5 13. Bxh5+
Your opponent just played Bxh5+ — a human of their level plays that about 81.5% of the time (rank 1 most likely; almost everyone plays this — utterly predictable).
You replied Kf8 — only about 84.5% of humans would (the Maia human-move model); you spent 0 of 50 centipawns of your eval budget on it.
Position evaluation now: +1.34 pawns in your favor.
Across the game so far, your opponent's moves have averaged 33% human-typical over 13 moves (higher = more predictable).
What's notable: Your opponent played the obvious move (~82% of humans do); the statistics saw it coming.
Reply in English.
Write your one-sentence remark, or PASS if nothing is worth saying.
```
</details>

- **google/gemini-3.5-flash** — Eighty-two percent of humans play Bxh5+, yet the evaluation now favors me by 1.34.  _(r=728t, $0.0077)_
- **google/gemini-3.1-flash-lite** — Eighty-one percent expected that check, and you delivered it right on cue.  _(r=98t, $0.0003)_
- **deepseek/deepseek-v4-pro** — You played 12 moves like a 33% outlier; move 13's 81.5% move is utterly predictable.  _(r=1835t, $0.0096)_
- **deepseek/deepseek-v4-flash** — Bxh5+ — the one move 82% of humans make, yet you've been unpredictable all game.  _(r=107t, $0.0001)_
- **qwen/qwen3.6-35b-a3b** — Bxh5+ was the exact move the 81.5% model predicted.  _(r=2754t, $0.0034)_

## Position 5 — `quiet` — game OUDCDszV vs Avo-k, move 15

Opening: Caro-Kann Defense. Opponent played **Qxe6+** (41% human, the popular choice); we replied **Qxe6** (72.3% human, 0cp lost), eval -0.69.

<details><summary>full prompt</summary>

```
Move 15. You play Black.
Opening: Caro-Kann Defense.
Moves so far: 1. e4 c6 2. d4 h6 3. Nc3 Qc7 4. f4 d5 5. e5 h5 6. Nf3 g6 7. Bd3 Na6 8. O-O Bf5 9. Bxf5 gxf5 10. e6 Qc8 11. exf7+ Kxf7 12. Ng5+ Kg6 13. Re1 Rh6 14. Qe2 e6 15. Qxe6+
Your opponent just played Qxe6+ — a human of their level plays that about 41.0% of the time (rank 2 most likely; the popular choice).
You replied Qxe6 — only about 72.3% of humans would (the Maia human-move model); you spent 0 of 50 centipawns of your eval budget on it.
Position evaluation now: -0.69 pawns in your favor.
Across the game so far, your opponent's moves have averaged 40% human-typical over 15 moves (higher = more predictable).
What's notable: Nothing especially notable here — only speak if you truly see something.
Reply in English.
Write your one-sentence remark, or PASS if nothing is worth saying.
```
</details>

- **google/gemini-3.5-flash** — PASS  _(r=115t, $0.0020)_
- **google/gemini-3.1-flash-lite** — PASS  _(r=111t, $0.0003)_
- **deepseek/deepseek-v4-pro** — PASS  _(r=77t, $0.0012)_
- **deepseek/deepseek-v4-flash** — PASS  _(r=135t, $0.0001)_
- **qwen/qwen3.6-35b-a3b** — PASS  _(r=1259t, $0.0017)_

## Position 6 — `quiet` — game OUDCDszV vs Avo-k, move 21

Opening: Caro-Kann Defense. Opponent played **Bd2** (14% human, a fairly common choice); we replied **Nc7** (23.7% human, 0cp lost), eval -0.24.

<details><summary>full prompt</summary>

```
Move 21. You play Black.
Opening: Caro-Kann Defense.
Moves so far: 1. e4 c6 2. d4 h6 3. Nc3 Qc7 4. f4 d5 5. e5 h5 6. Nf3 g6 7. Bd3 Na6 8. O-O Bf5 9. Bxf5 gxf5 10. e6 Qc8 11. exf7+ Kxf7 12. Ng5+ Kg6 13. Re1 Rh6 14. Qe2 e6 15. Qxe6+ Qxe6 16. Rxe6+ Nf6 17. Nb1 h4 18. Nd2 Rh8 19. Ndf3 Rh6 20. Ne5+ Kh5 21. Bd2
Your opponent just played Bd2 — a human of their level plays that about 14.4% of the time (rank 3 most likely; a fairly common choice).
You replied Nc7 — only about 23.7% of humans would (the Maia human-move model); you spent 0 of 50 centipawns of your eval budget on it.
Position evaluation now: -0.24 pawns in your favor.
Across the game so far, your opponent's moves have averaged 45% human-typical over 21 moves (higher = more predictable).
What's notable: Nothing especially notable here — only speak if you truly see something.
Reply in English.
Write your one-sentence remark, or PASS if nothing is worth saying.
```
</details>

- **google/gemini-3.5-flash** — PASS  _(r=264t, $0.0034)_
- **google/gemini-3.1-flash-lite** — PASS  _(r=127t, $0.0004)_
- **deepseek/deepseek-v4-pro** — PASS  _(r=182t, $0.0017)_
- **deepseek/deepseek-v4-flash** — PASS  _(r=112t, $0.0001)_
- **qwen/qwen3.6-35b-a3b** — PASS  _(r=938t, $0.0012)_

---

## Cost & tokens per model

Averaged over 6 calls each. Reasoning runs even when the model answers PASS, so PASS still costs. Extrapolation assumes **6 spoken comments/game**.

| model | avg prompt | cached | avg reason | avg out | $/comment | $/game | $/10 games | $/100 games | PASS rate |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| google/gemini-3.5-flash | 603 | 0 | 482 | 496 | $0.0054 | $0.032 | $0.32 | $3.22 | 2/6 |
| google/gemini-3.1-flash-lite | 604 | 0 | 200 | 212 | $0.0005 | $0.003 | $0.03 | $0.28 | 2/6 |
| deepseek/deepseek-v4-pro | 603 | 49 | 506 | 520 | $0.0029 | $0.018 | $0.18 | $1.75 | 2/6 |
| deepseek/deepseek-v4-flash | 590 | 86 | 131 | 151 | $0.0001 | $0.001 | $0.01 | $0.06 | 2/6 |
| qwen/qwen3.6-35b-a3b | 628 | 0 | 2321 | 2671 | $0.0028 | $0.017 | $0.17 | $1.69 | 2/6 |