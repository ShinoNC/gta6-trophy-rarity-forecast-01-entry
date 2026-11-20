# gta6-trophy-rarity-forecast-01 — entry v2

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for each of the 55 GTA VI trophies, the 5 quantiles (5%, 25%, 50%, 75%,
95%) of the share of tracked PS5 players who will have earned it at
2026-11-26 00:00 UTC (one week after launch). Scored by pinball loss on the
logit scale against the first valid Exophase read in the hour from 00:00 UTC
on 2026-11-26.

v2 re-tunes v1's model on the first post-launch-day Exophase snapshot
(snapshot C, crawled 04:50Z Nov 20, ~29h after launch). The model structure is
unchanged; four parameters changed (see "What v2 changed").

## Method (E/P decomposition)

Exophase's page is a step function: rates only change when Exophase refreshes,
and each refresh dilutes every rate because the tracked-player base grows
faster than earners. So instead of extrapolating rates directly, the model
decomposes each trophy into earners and players:

- **Players tracked P(t):** the launch-night page is a fixed reviewer base P0;
  the general population P(t) − P0 grows as a power law, so `ln(P − P0)` is
  fitted linear in `ln(t)` over the case's distinct snapshots and extrapolated.
  With fewer than 2 usable points the slope falls back to a prior (0.9).
  Snapshot C confirms this: the implied P exponent is 0.81–0.97 across the
  plausible (refresh time, P) range, consistent with the 0.9 prior.
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
  `s = sqrt(resid² + P-resid² + (0.15·√days)² + 0.6² + gap²)`, capped at 2.5,
  where the gap term is the local log-slope of the rate times the expected
  refresh gap (1/lambda days). The 0.6 floor covers refresh risk and
  misspecification: a single Exophase refresh can move a rate by ~1 logit
  unit. No per-role spread multipliers — with the Weibull median the
  per-role biases are gone, so the multipliers only added noise.

Deterministic, stdlib only, runs in ~0.1 s. Falls back to the persistence
baseline if a case has no usable history, so a run never crashes.

## What v2 changed (and why)

Snapshot C (Nov 20) contradicted one v1 assumption: the adoption curve is
FLATTER than a Weibull with shape 1.5. 21 of 31 trophies with a non-zero rate
at B (04:30Z Nov 19) had a LOWER rate at C — new players keep joining faster
than earners, so the population earn rate is diluted even as earners grow.
A single cohort's adoption curve may be S-shaped, but the population curve
the entry fits is the convolution of that curve with a stream of fresh
joiners, and the convolution is flatter. A shape diagnostic (the C refresh
time implied by each shape, robust to 2x tau-prior scaling) puts the
effective shape at ~0.75–1.0.

- **Default shape 1.5 → 1.0.** The backtest (fit on A+B, predict C) ranks
  k=0.75 ≥ k=1.0 > k=1.25 > k=1.5. Simulator tournaments across world shapes
  0.75–2.0 show k=1.0 wins by ~0.25 when the true shape is ≤1.0 and costs
  ~0.13 when it is 1.5; k=0.75 pays a much larger downside if the shape is
  1.5+, so 1.0 is the balanced choice, not the backtest argmax.
- **Shape grid gains 0.75** ([0.75, 1.0, 1.25, 1.5, 1.75, 2.0]) so the
  per-group vote can reach the flat end.
- **Shape vote fires at ≥3 usable points (was ≥4).** The final case will
  most likely hold exactly 3 usable points (the B, C, D refreshes), so a
  threshold of 4 would leave the vote silent there. Three points give one
  residual degree of freedom per trophy, pooled across the role group, and
  the 2x-margin guard still stops near-tie flips.
- **Spread floor 0.5 → 0.6.** The real backtest under-covered at 0.5
  (coverage 0.31–0.55 vs 0.94–0.97 in the simulator): the real world has
  misspecification the simulator cannot see (the dilution dip). The
  simulator cost of 0.6 is ≤0.015.

What did NOT change, and why: tau priors (the backtest is insensitive to
2–3x scaling — with one fit point the level absorbs the tau change), the
refresh-rate logic (the backtest's large t_med cost is specific to the
early fast-growing regime; by the final's 168h horizon the curves are flat
and the pullback costs little), the Weibull offset (its backtest gain is
the same signal as the shape change, expressed differently — changing both
would double-flatten), and BETA_PRIOR (the final case fits P from 2–4
points, so the prior rarely binds).

## Validation

- Scorer reproduces the RULES worked examples exactly (0.403404; invalid-case
  0.46023). Baselines reimplemented from the RULES reproduce the stated
  practice numbers exactly (1.508366 on p00/p01).
- Practice cases run clean and valid (55 ids × 5 quantiles, deterministic,
  ~0.1 s). Practice is a validity check, not the selection criterion: the
  practice history is perfectly flat, so the median is persistence and the
  single refresh dilutes rates by 1–2.4 logit; the final case has 2.6 days
  of real growth history, which is what the simulator models.
- **Real-data backtest** (fit on snapshots A+B, predict snapshot C; 4 cells:
  checkpoints 04:30Z/11:19Z Nov 19 × assumed C refresh 24h/28.5h): v2 beats
  v1 in all 4 cells — 0.453 vs 0.469, 0.349 vs 0.391, 0.623 vs 0.653,
  0.462 vs 0.479 (pinball loss on the logit scale). The curve baseline
  beats both on this backtest (0.19–0.37) — a short-horizon artifact: over
  12–24h dilution dominates and the curve extrapolates the A→B dilution
  trend, but over 4.5 days it runs away (simulator: curve 1.32 vs entry
  0.18), so the model was not switched.
- **Simulator tournament, shape-varied worlds** (40 worlds per cell, seeds
  7/11/99): v2 vs v1 mean loss — world shape 0.75: 0.22 vs 0.41; shape 1.0:
  0.17 vs 0.44; shape 1.5: 0.36 vs 0.19; shape 2.0: 0.58 vs 0.37. v2 wins
  decisively if the true effective shape is ≤1.0 (as the real data says)
  and loses if it is ≥1.5. Under the real-data-informed prior
  (shape 0.75/1.0/1.5/2.0 with weights 0.35/0.35/0.2/0.1), v2's expected
  loss is ~0.26 vs v1's ~0.38. Both entries beat the house baselines
  (persistence ~0.7–1.6, curve ~1.0–1.6) in nearly every world.
- **Honest limits:** the backtest has one usable point per trophy, so it
  measures priors, not the fitting machinery; the simulator cannot validate
  the dilution dip (its rates are monotonic by construction) or the
  Weibull offset (its truth is +6h). The v2 bet is that the real effective
  shape is ≤~1.1 — supported by the shape diagnostic, the backtest, and
  the dilution mechanism, but a single refresh of evidence.

## Layout

- `main.py` — the entry (Python 3.13, stdlib only).
- `../tools/` — scorer, baseline reimplementations, simulator, tournaments,
  backtests, shape diagnostic (validation only, not part of the entry).
- `../research/` — real Exophase snapshots A/B/C, Ghost of Yotei anchors.
- `../README.md` (this file) — method + validation summary.

## Run

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=<out> python3 main.py
```

Writes `/output/<id>.out` for every `/input/<id>.in`.

<sub>Entered on [make.fun](https://make.fun/?ref=1qjvtr9y&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
