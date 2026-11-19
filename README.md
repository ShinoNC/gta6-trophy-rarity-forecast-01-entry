# gta6-trophy-rarity-forecast-01 — entry

[![make.fun](https://make.fun/badge.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

Forecast, for each of the 55 GTA VI trophies, the 5 quantiles (5%, 25%, 50%, 75%,
95%) of the share of tracked PS5 players who will have earned it at
2026-11-26 00:00 UTC (one week after launch). Scored by pinball loss on the
logit scale against the first valid Exophase read in the hour from 00:00 UTC
on 2026-11-26.

## Method (E/P decomposition)

Exophase's page is a step function: rates only change when Exophase refreshes,
and each refresh dilutes every rate because the tracked-player base grows
faster than earners. So instead of extrapolating rates directly, the model
decomposes each trophy into earners and players:

- **Players tracked P(t):** the launch-night page is a fixed reviewer base P0;
  the general population P(t) − P0 grows as a power law, so `ln(P − P0)` is
  fitted linear in `ln(t)` over the case's distinct snapshots and extrapolated.
  With fewer than 2 usable points the slope falls back to a prior (0.9).
- **Earners E(t):** the first snapshot's earners are the launch-night cohort;
  what grows afterwards is the general population, whose earn rate
  r(t) = (E(t) − E0) / (P(t) − P0) follows a saturating adoption curve
  (Weibull form: A · (1 − exp(−((t+6)/tau)^k))). A log-linear fit cannot
  saturate, so it runs away for fast trophies and under-shoots slow ones;
  the Weibull form captures both the early acceleration (long-tau grinds)
  and the late saturation (short-tau story/easy trophies). The shape
  exponent k is selected per role group from the case itself (a fixed k
  would mislead if the real adoption curve is more or less S-shaped);
  tau is fitted per trophy by grid search and shrunk toward its role-group
  median; the level A is fitted per trophy by weighted least squares.
  Roles are classified from the case's own descriptions + tiers — no trophy
  id is hard-coded.
- **Refresh timing:** the answer is the page state at the LAST refresh before
  the target, not at the target itself. The refresh rate is estimated from
  the case's own snapshot times (with a prior toward ~1/day and a floor of
  0.5/day), and the median is evaluated at the median last-refresh time
  (target minus ln2/lambda). There are deliberately NO median caps: the
  Weibull fit already saturates (A ≤ 1), so a soft cap toward a "mature
  level" would double-count the saturation and bias fast trophies down.
  The 100% trophy (Gone Fishing) is forecast at ~99.99%.
- **Spread:** normal quantiles on the logit scale,
  `s = sqrt(resid² + P-resid² + (0.15·√days)² + 0.5² + gap²)`, capped at 2.5,
  where the gap term is the local log-slope of the rate times the expected
  refresh gap (1/lambda days). The 0.5 floor covers refresh risk: a single
  Exophase refresh can move a rate by ~1 logit unit. No per-role spread
  multipliers — with the Weibull median the per-role biases are gone, so
  the multipliers only added noise.

Deterministic, stdlib only, runs in ~0.3 s. Falls back to the persistence
baseline if a case has no usable history, so a run never crashes.

## Validation (offline simulator, `tools/`)

- Scorer reproduces the RULES worked examples exactly (0.403404; invalid-case
  0.46023). Baselines reimplemented from the RULES reproduce the stated
  practice numbers exactly (1.508366 on p00/p01).
- Practice cases (answers ≈ the 04:30Z refresh, snapshot B): **loss 1.245**
  vs persistence 1.508 and curve 1.508. Coverage is low (0.02) — the practice
  history is perfectly flat, so the median is persistence and the single
  refresh dilutes rates by 1–2.4 logit; the final case has 2.6 days of real
  growth history, which is what the simulator models. Practice is a validity
  check here, not the selection criterion.
- Simulator tournament (two-population generative model: power-law P growth,
  Weibull earn curves, Poisson refreshes; 60 worlds, seeds 7+42): **mean loss
  0.18, median 0.15, worst 0.66, coverage 0.94**; beats both house baselines
  in 58 of 60 worlds (persistence 1.08, curve 1.32). Fresh-seed validation
  (seeds 11+99): mean 0.20, wins 59/60. Per-role median logit error is within
  ±0.13 for every role (grind 0.00, story1 +0.06, easy +0.21 — the largest
  remaining bias).
- Stress + diversity (20 worlds each): slow/fast progression 0.18/0.19,
  refresh 0.5/day 0.38 and 2/day 0.15, P exponent 0.7/1.15 0.17/0.17,
  adoption shape 1.0/2.0 0.31/0.41, saturating P 0.26, front/back-loaded
  refresh 0.33/0.17, tau/maturity jitter 0.26/0.26 — the entry beats both
  baselines in 287 of 300 worlds across all 15 sets. The weakest regimes are
  slow refresh (0.5/day) and front-loaded refresh, where the last-refresh
  time is hardest to pin down; both are dominated by irreducible Poisson
  timing noise rather than model bias.

## Layout

- `main.py` — the entry (Python 3.13, stdlib only).
- `../tools/` — scorer, baseline reimplementations, simulator, tournament,
  parameter grid, diagnostics (validation only, not part of the entry).
- `../research/` — real Exophase snapshots A/B, Ghost of Yotei anchors.
- `../README.md` (this file) — method + validation summary.

## Run

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=<out> python3 main.py
```

Writes `/output/<id>.out` for every `/input/<id>.in`.
