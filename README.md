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

- **Players tracked P(t):** `ln P` is linear in `ln(1+t)` (power-law growth,
  fitted on the distinct snapshots in the case history).
- **Earners E(t):** for each trophy, fit `ln(E - E0 + 0.1)` linear in
  `ln(1+t)` on the distinct snapshots (E0 = earners in the first snapshot),
  with recency-weighted OLS and the slope shrunk toward the mean slope of the
  trophy's coarse role group (story / mission / grind / easy / misc / rarest).
  Roles are classified from the case's own descriptions + tiers — no trophy
  id is hard-coded.
- **Median** = 100 · E(T) / P(T), soft-capped toward the role's mature level
  (`M · (1 - exp(-x/M))`, mature levels anchored on Ghost of Yotei's
  7-week Exophase shape) and hard-capped at a generous week-1 ceiling.
  The 100% trophy (Gone Fishing) is forecast at ~99.99%; platinum and 100%
  completion stay near 0% (nobody can finish a ~200 h game in a week).
- **Spread:** normal quantiles on the logit scale, `s = sqrt(resid² +
  P-resid² + (0.15·√days)² + 0.65²)`, capped at 2.5, with role multipliers
  (rarest 1.3×, grind 1.2×, easy 0.9×). The 0.65 floor covers refresh risk:
  a single Exophase refresh can move a rate by 1–2 logit units.

Deterministic, stdlib only, runs in ~0.3 s. Falls back to the persistence
baseline if a case has no usable history, so a run never crashes.

## Validation (offline simulator, `tools/`)

- Scorer reproduces the RULES worked examples exactly (0.403404; invalid-case
  0.46023). Baselines reimplemented from the RULES reproduce the stated
  practice numbers exactly (1.508366 on p00/p01).
- Practice cases (answers ≈ the 04:30Z refresh, snapshot B): **loss 1.142**
  vs persistence 1.508 and curve 1.508. Coverage is low (0.05) — the practice
  history is perfectly flat, so the median is persistence and the single
  refresh dilutes rates by 1–2.4 logit; the final case has 2.6 days of real
  growth history, which is what the simulator models.
- Simulator tournament (two-population generative model: power-law P growth,
  Weibull earn curves, Poisson refreshes; 60 worlds, seeds 7+42): **mean loss
  0.27, median 0.23, worst 0.65, coverage 0.94**; beats both house baselines
  in 56–57 of 60 worlds (persistence 1.08, curve 1.32). Fresh-seed validation
  (seeds 11+99): mean 0.30, wins 59/60. Robust to prior shifts (truths 1.5×
  and 0.6× the assumed mature levels) and to refresh cadence (0.5–2/day).

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
