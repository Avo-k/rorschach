# Talk-rate calibration

Model **deepseek/deepseek-v4-flash**, 180 random positions from 20 human games. Every position was *offered* (forced an LLM call) to measure the PASS rate.

- **PASS rate over random positions:** 21% (model spoke on 79%).
- To hit an effective **10%** speak rate with a flat random offer: **offer_rate = 0.13** (= 0.10 / 0.79).
- A flat 0.10 offer would instead yield ~7.9% effective speak.

## PASS rate by position type

| bucket | positions | spoke | speak rate |
|---|--:|--:|--:|
| alien_self | 97 | 97 | 100% |
| predictable_human | 25 | 25 | 100% |
| surprising_human | 7 | 7 | 100% |
| quiet | 51 | 13 | 25% |

## Cost at the recommended offer rate

Offer 0.13 × ~30 of our moves/game = ~4 LLM calls/game (≈3 spoken). Avg $0.0001/call.

- per game: $0.0005
- 10 games: $0.005
- 100 games: $0.05

_Recommendation: set `BASE_PROB = 0.13` in chat.py (VERBOSE_PROB scales similarly)._