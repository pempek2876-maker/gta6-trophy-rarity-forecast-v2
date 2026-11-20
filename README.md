# gta6-trophy-rarity-forecast-v2

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/pempek2876-maker/gta6-trophy-rarity-forecast-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/pempek2876-maker/gta6-trophy-rarity-forecast-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Trophy Rarity Forecaster v2 — per-band calibrated ensemble forecasting GTA VI week-one trophy earn-rate quantiles. Trophies are grouped into earn-rate bands (COMMON ≥5%, MID 1–5%, RARE 0.2–1%, ULTRA <0.2%); each band learns its own blend of four median candidates (persistence, logit-linear all-history, logit-linear last-24h, counts E/N) from a multi-horizon rolling-origin pinball backtest, and each band's quantile spread is calibrated by extrapolating its blend residual sd to the target horizon with a power law (floor 0.75, cap 3.0 logit). A trophy at 100% in every snapshot gets [97, 99, 100, 100, 100]. Node 22, stdlib only, no network, deterministic.

## Run it

```sh
MAKEFUN_INPUT_DIR=<input-dir> MAKEFUN_OUTPUT_DIR=<output-dir> node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=7wmama53&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
