# GTA VI Trophy Earn-Rate Forecaster (v3)

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ShinoNC/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/ShinoNC/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for each of the 55 GTA VI PS5 trophies, five quantiles of its
Exophase earn rate at the case target (2026-11-26T00:00:00Z for the final
case), scored by mean pinball loss on the logit scale.

## Method (v3): a blend of two medians

No single model wins in every regime the real data has shown, so v3
averages two medians in logit space (weight 0.75 on the E/P median):

1. **E/P median** — earners and players tracked separately. P(t) minus the
   launch-night reviewer base grows as a power law (ln(P - P0) linear in
   ln(t)); the general population's earn rate follows a saturating Weibull
   adoption curve A * (1 - exp(-((t+6)/tau)^k)) with shape k = 1.0,
   per-trophy tau fitted by grid search and shrunk toward the role-group
   median tau. Roles come from the case's own trophy descriptions and
   tiers; levels never pool across roles. This is the model that handles
   the growth-takeover regime.
2. **Trend median** — the house curve's approach, improved: per-trophy OLS
   of logit(rate) on ln(1+t) over the case's distinct snapshots only
   (flat stretches of the step page carry no information), with the slope
   shrunk toward the role-group median slope. This is the model that
   handles the dilution regime.

Both medians are evaluated at the median last-refresh time (target minus
ln(2)/lambda, lambda estimated from the case's own snapshot times),
because the answer is the page state at the last refresh before the
target, not at the target. Spread is logit-normal, from the E/P fit
residuals, the P-fit residual, a horizon term, a refresh-gap term, and a
1.0-logit floor. The 100% trophy gets fixed quantiles
[99.9, 99.97, 99.99, 99.99, 99.99]. Any failure falls back to
persistence, so a run never crashes and never writes an invalid file.

## Why a blend (real-data evidence)

- **Backtest 1** (fit A+B, predict snapshot C, dilution regime): the trend
  beats E/P (0.19-0.37 vs 0.35-0.62). Rates only fell; the trend's
  declining extrapolation was right.
- **Backtest 2** (fit A+B+C, predict snapshot D, live page 11:54Z Nov 20):
  E/P beats the trend in all 12 cells (0.72 vs 0.78-1.25). Snapshot D is
  a regime flip: early trophies exploded (Ch1 0.73% -> 7.99%, Baited
  1.89% -> 10.57%) while late trophies kept diluting. The trend
  extrapolates the dilution decline and gets the direction wrong.
- **v3 vs v2 on both backtests**: v3 wins all 16 cells (backtest 1 mean
  0.454 vs 0.472; backtest 2 mean 0.673 vs 0.726), mostly from the wider
  spread floor (both backtests under-covered at 0.5-0.6) plus the blend.
- **Simulator** (synthetic final worlds, 40 worlds x shapes 0.75/1.0/1.5
  + fresh seeds): v2 beats v3 (mean 0.23 vs 0.33 on seed 7; 0.28 vs 0.40
  on fresh seeds), because the simulator's generative form is Weibull
  adoption + power-law P -- v2's own model class. The simulator is our
  own assumptions; the backtests are real Exophase data. The final is
  one case; v3 trades simulator-optimality for robustness across both
  real regimes.

## The practice leaderboard, and why v3 does not copy it

All 7,026 other entries have their source withheld (ours is the only
public one), so no competitor method is observable. The top practice
losses (0.00005-0.08 vs 1.18 for the best honest model) are only
reachable by embedding the known practice answers (the 04:30Z Nov 19
refresh, public Exophase data) in the program: the practice cases
contain a single launch-night page state, so no honest model can predict
the refresh jump (demonstrated locally: exact-answer oracle 0.002,
best honest model 1.176). That scores practice points and says nothing
about the final case. v3 follows the observable standard method (the
direct trend) as one blend component instead.

## Validation

- Deterministic (two runs byte-identical), runtime ~0.5s per case
  (limit 8s), output 2,656 bytes (limit 65,536).
- Practice loss vs snapshot B: 0.924 (v2: 1.176), coverage 0.51.
- Python 3.13, standard library only, no network. Reads every `*.in`
  in `$MAKEFUN_INPUT_DIR` (else `/input`), writes `<id>.out` to
  `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## License

MIT. See LICENSE.
