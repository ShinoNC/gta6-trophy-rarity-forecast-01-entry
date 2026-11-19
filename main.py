"""Forecast each GTA VI PS5 trophy's Exophase earn rate at the case target.

Model (earn-rate decomposition). A trophy's rate is earners over players tracked:
    rate_i(t) = E_i(t) / P(t).
Exophase's page is a step function: flat between refreshes, jumping at each refresh,
when new (mostly fresh) players join and dilute most rates. So instead of trending
the rate itself, this program models the two counts separately:

  * P(t): players tracked. The launch-night page is a fixed reviewer base P0; the
    general population P(t) - P0 grows as a power law, so ln(P - P0) is fitted
    linear in ln(t) over the case's distinct snapshots and extrapolated.
  * E_i(t): earners of trophy i. The first snapshot's earners are the launch-night
    cohort; what grows afterwards is the general population, whose earn rate
    r_i(t) = (E_i(t) - E_i(0)) / (P(t) - P0) follows a saturating adoption curve
    (Weibull form: A * (1 - exp(-((t+6)/tau)^k))). A log-linear fit cannot
    saturate, so it runs away for fast trophies and under-shoots slow ones;
    the Weibull form captures both the early acceleration (long-tau grinds)
    and the late saturation (short-tau story/easy trophies). The shape
    exponent k is selected per role group from the case itself (a fixed k
    would mislead if the real adoption curve is more or less S-shaped);
    tau is fitted per trophy by grid search and shrunk toward its role-group
    median (roles read from the case's own trophy descriptions and tiers);
    the level A is fitted per trophy by weighted least squares.
  * Refresh timing: the answer is the page state at the LAST refresh before
    the target, not at the target itself. The refresh rate is estimated from
    the case's own snapshot times, and the median is evaluated at the median
    last-refresh time (target minus ln2/lambda); the spread carries the gap
    uncertainty as well as the fits' residuals, a horizon term, and a FLOOR
    that never vanishes (a refresh can move any rate by ~1 logit at any time).

Roles are classified from the case data itself (description keywords + tier), never
from trophy ids: story chapters by number, the final mission, the first mission
(rate 100%), 100%-completion / platinum as rarest, grinds by "Complete N ..." /
"100 locations" / "100 vehicles" / "Bank Vaults" phrasing, easy early activities
(fishing, arcade, cocktail), everything else misc.

Safety: any failure in the model falls back to persistence (last rate, baseline
spread) for that case, so a run never crashes and never writes an invalid file.
Deterministic: no randomness, no dict-order dependence, fixed float order.
Standard library only. Reads every *.in in $MAKEFUN_INPUT_DIR (else /input),
writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Tunable constants (chosen on the offline simulator; see the skill file).
K_TAU = 0.25          # tau shrinkage: effective extra observations from the role median
S_FLOOR = 0.5         # logit spread floor (refresh risk; never vanishes).
# Was 0.65 when the median came from a log-linear E fit; the Weibull median
# is much better calibrated now, so the floor came down. The simulator alone
# would push it to ~0.3, but the real adoption curve is not exactly Weibull
# and the real refresh cadence may differ, so 0.5 keeps a margin for
# misspecification the simulator cannot see (centre value, not argmax).
S_HOR = 0.15          # horizon uncertainty, logit per sqrt-day
S_MAX = 2.5           # spread cap
S_GAP_MAX = 0.5       # cap on the refresh-gap spread term
BETA_PRIOR = 0.9      # P-growth exponent fallback when <2 usable P points
SHAPE_K = 1.5         # Weibull shape exponent of the adoption curve (default;
# the per-group fit below selects from SHAPE_GRID instead when the case
# has enough points to identify the shape)
SHAPE_GRID = [1.0, 1.25, 1.5, 1.75, 2.0]
# Role spread multipliers: all 1.0. Earlier versions widened grind/rarest and
# narrowed easy to compensate for the old log-linear median's role biases;
# with the Weibull median those biases are gone (grind ~0.00, story1 ~+0.06),
# and flattening the multipliers improved every simulator set slightly.
# Only "first" (the 100% trophy) keeps a special value, and it is handled by
# the dedicated branch below, not by this table.
SPREAD_MULT = {"rarest": 1.0, "grind": 1.0, "story1": 1.0, "story2": 1.0,
               "story3": 1.0, "story4": 1.0, "final": 1.0, "mission": 1.0,
               "easy": 1.0, "first": 0.05, "misc": 1.0}
# No median caps: the Weibull fit already saturates (A <= 1, r <= 1), so a
# soft cap toward a "mature level" would double-count the saturation and bias
# fast trophies down. The only bound is clamp_pct (0.01-99.99).
RECENCY = 1.0  # fit weights ramp 1 -> 1+RECENCY over snapshots
# Tau grid for the Weibull fit (hours): log-spaced, 15h (same-day) to 4000h
# (multi-month grinds). 33 points keep the search cheap and deterministic.
TAU_GRID = [15.0 * (4000.0 / 15.0) ** (k / 32.0) for k in range(33)]
# Prior tau per coarse role group (hours), used only when a group has no trophy
# with enough points to fit tau from the case itself. From guide difficulty:
# story chapters take hours-days, grinds take weeks, platinum ~200h of play.
ROLE_TAU_PRIOR = {"story": 160.0, "grind": 400.0, "easy": 80.0,
                  "misc": 220.0, "rarest": 1500.0, "first": 1.0}
# Coarse groups for TAU pooling: chapters share an adoption timescale even though
# their levels differ, so taus pool across the story group; levels do not pool.
SHRINK_GROUP = {"story1": "story", "story2": "story", "story3": "story",
                "story4": "story", "final": "story", "mission": "story",
                "grind": "grind", "easy": "easy", "misc": "misc",
                "rarest": "rarest", "first": "first"}
# Refresh-rate prior: 1 pseudo-refresh over 1 pseudo-day (the observed cadence
# is ~1/day); the estimate shrinks toward it when few refreshes are observed.
# LAM_MIN floors the estimate: with only one refresh observed, the gap-based
# estimate would crash toward 0 (no gaps seen yet), but a front-page game is
# refreshed at least every couple of days, so the rate cannot be near 0.
LAM_PRIOR_N = 1.0
LAM_PRIOR_D = 1.0
LAM_MIN = 0.5
LAM_MAX = 4.0


def clamp_pct(x):
    return min(99.99, max(0.01, float(x)))


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 10000
    return int(v) if float(v).is_integer() else v


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def ols(xs, ys, ws=None):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sd)."""
    if ws is None:
        ws = [1.0] * len(xs)
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        resid = [y - my for y in ys]
        sd = math.sqrt(sum(r * r for r in resid) / max(1, len(ys) - 1)) if len(ys) > 1 else 0.0
        return my, 0.0, sd
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1, len(ys) - 2)
    sd = math.sqrt(sum(r * r for r in resid) / dof)
    return a, b, sd


def classify(trophy):
    """Role of a trophy, from its own description + tier. No id hard-coding."""
    desc = (trophy.get("description") or "").lower()
    tier = (trophy.get("tier") or "").lower()
    name = (trophy.get("name") or "").lower()
    if tier == "platinum" or "100% completion" in desc:
        return "rarest"
    if "gone fishing" in desc or ("complete" in desc and "first" in name):
        return "first"
    m = re.search(r"chapter (\d+)", desc)
    if m:
        return "story" + m.group(1)
    if "final mission" in desc:
        return "final"
    if re.search(r"complete (250|50|8|3) ", desc) or "100 locations" in desc \
            or "100 vehicles" in desc or "bank vaults" in desc \
            or "collectible card sets" in desc:
        return "grind"
    if "fishing" in desc or "arcade game" in desc or "cocktail" in desc \
            or "bowling" in desc or "drive-by" in desc:
        return "easy"
    if "complete" in desc and ("mission" in desc or "takedown" in desc
                               or "heist" in desc or "operation" in desc):
        return "mission"
    return "misc"


def distinct_snapshots(case):
    """Collapse history to distinct page states: (t_hours, players, rates)."""
    release = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
    try:
        rd = (case.get("game") or {}).get("release_date")
        if rd:
            release = datetime.fromisoformat(rd).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        pass
    snaps = []
    seen = set()
    for read in case.get("history") or []:
        rates = read.get("rates")
        if rates is None:
            continue
        p = read.get("players_tracked")
        if not p:
            continue
        key = (p, tuple(rates))
        if key in seen:
            continue
        seen.add(key)
        t = (parse_time(read["at"]) - release).total_seconds() / 3600.0
        snaps.append((t, float(p), [None if r is None else float(r) for r in rates]))
    snaps.sort(key=lambda s: s[0])
    return snaps, release


def weib_z(t, tau, k=SHAPE_K):
    """Saturating adoption shape at time t (hours) for timescale tau."""
    return 1.0 - math.exp(-((max(0.0, t) + 6.0) / tau) ** k)


def fit_level(ts, ys, ws, tau, k=SHAPE_K):
    """Weighted least squares for the level A given tau (through the origin)."""
    zs = [weib_z(t, tau, k) for t in ts]
    szz = sum(w * z * z for w, z in zip(ws, zs))
    if szz <= 1e-12:
        return 0.0, float("inf")
    a = sum(w * z * y for w, z, y in zip(ws, zs, ys)) / szz
    a = min(1.0, max(0.0, a))
    sse = sum(w * (y - a * z) ** 2 for w, z, y in zip(ws, zs, ys))
    return a, sse


def fit_tau(ts, ys, ws, k=SHAPE_K):
    """Grid-search tau; returns (tau, A, sse) or None when the curve is flat at 0."""
    if not ys or max(ys) <= 0.0:
        return None
    best = None
    for tau in TAU_GRID:
        a, sse = fit_level(ts, ys, ws, tau, k)
        if best is None or sse < best[2]:
            best = (tau, a, sse)
    return best if best is not None else None


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    snaps, release = distinct_snapshots(case)
    target = parse_time(case["target"])
    t_target = (target - release).total_seconds() / 3600.0
    horizon_days = max(0.0, case["horizon_minutes"] / 1440.0)
    try:
        t_cp = (parse_time(case["at"]) - release).total_seconds() / 3600.0
    except (KeyError, ValueError, TypeError):
        t_cp = snaps[-1][0] if snaps else 0.0

    last = snaps[-1] if snaps else None
    last_rates = {}
    if last is not None:
        for i, tr in enumerate(trophies):
            r = last[2][i]
            last_rates[tr["id"]] = 0.0 if r is None else r

    # ---- refresh rate from the case's own snapshot times ----
    # Refreshes after the first one arrive as a Poisson process; the estimate
    # counts them over the window after the first refresh, shrunk toward a
    # 1/day prior. The answer is the state at the last refresh before the
    # target, whose median time is target - ln(2)/lambda.
    lam_hat = 1.0
    if len(snaps) >= 2:
        t_first_refresh = snaps[1][0]
        window_days = max(0.0, (t_cp - t_first_refresh) / 24.0)
        n_after = len(snaps) - 2
        lam_hat = (n_after + LAM_PRIOR_N) / (window_days + LAM_PRIOR_D)
        lam_hat = min(LAM_MAX, max(LAM_MIN, lam_hat))
    t_med = t_target - 24.0 * math.log(2.0) / lam_hat
    t_last_snap = snaps[-1][0] if snaps else 0.0
    t_med = min(t_target, max(t_last_snap, t_med))
    # ---- players tracked: ln(P - P0) ~ ln(t) (power-law growth of the
    # general population on top of the fixed launch-night base P0) ----
    p0 = snaps[0][1] if snaps else 0.0
    p_pred, p_resid = None, 0.0
    p_pts = [(math.log(t), math.log(p - p0)) for t, p, _ in snaps
             if t > 0.0 and p > p0]
    if len(p_pts) >= 2:
        ws = [1.0 + RECENCY * j / (len(p_pts) - 1) for j in range(len(p_pts))]
        a, b, sd = ols([p[0] for p in p_pts], [p[1] for p in p_pts], ws)
        b = max(0.0, b)
        p_pred = p0 + math.exp(a + b * math.log(max(1e-6, t_med)))
        p_resid = sd
    elif len(p_pts) == 1:
        x1, y1 = p_pts[0]
        p_pred = p0 + math.exp(y1 + BETA_PRIOR * (math.log(max(1e-6, t_med)) - x1))
    if p_pred is None and last is not None:
        p_pred = last[1] * ((1.0 + max(0.0, t_med)) / (1.0 + max(0.0, last[0]))) ** BETA_PRIOR
    if p_pred is None:
        p_pred = 1.0
    p_pred = max(p_pred, p0 + 1.0)
    p_gen_med = p_pred - p0

    roles = [classify(tr) for tr in trophies]

    # ---- per-trophy Weibull adoption fits ----
    # y_j = excess earners per general player at snapshot j: the general
    # population's earn rate, fitted as A * (1 - exp(-((t+6)/tau)^1.5)).
    fits = []
    for i, tr in enumerate(trophies):
        ts, ys, pgens, e0 = [], [], [], None
        for t, p, rates in snaps:
            r = rates[i]
            if r is None:
                continue
            e = r / 100.0 * p
            if e0 is None:
                e0 = e
            if t > 0.0 and p > p0:
                ts.append(t)
                ys.append(max(0.0, e - e0) / (p - p0))
                pgens.append(p - p0)
        if e0 is None:
            e0 = 0.0
        ws = [1.0 + RECENCY * j / (len(ts) - 1) for j in range(len(ts))] if len(ts) > 1 \
            else [1.0] * len(ts)
        fits.append({"e0": e0, "ts": ts, "ys": ys, "pgens": pgens, "ws": ws,
                     "role": roles[i]})

    # shape vote per coarse group: the adoption exponent is a property of the
    # process, pooled across the group; each trophy votes with its normalized
    # fit quality (SSE over sum of squared y, so big and small trophies count
    # alike) at each candidate shape. Only trophies with >=4 usable points
    # vote: with 3 or fewer points the shape is barely identified (2 params
    # already fit 2 points exactly) and the vote would be tau-grid
    # quantization noise. The default 1.5 stands unless another shape wins by
    # a 2x margin, so a near-tie cannot flip the shape.
    shape_by_group = {}

    def group_shape(role):
        grp = SHRINK_GROUP.get(role, role)
        if grp in shape_by_group:
            return shape_by_group[grp]
        members = [f for f in fits
                   if SHRINK_GROUP.get(f["role"], f["role"]) == grp
                   and len(f["ts"]) >= 4 and max(f["ys"]) > 0.0]
        if not members:
            shape_by_group[grp] = SHAPE_K
            return SHAPE_K
        best_k, best_tot = SHAPE_K, None
        tot_default = None
        for k in SHAPE_GRID:
            tot = 0.0
            for f in members:
                got = fit_tau(f["ts"], f["ys"], f["ws"], k)
                if got is None:
                    continue
                denom = sum(w * y * y for w, y in zip(f["ws"], f["ys"]))
                tot += got[2] / denom if denom > 0 else 0.0
            if k == SHAPE_K:
                tot_default = tot
            if best_tot is None or tot < best_tot:
                best_tot, best_k = tot, k
        if best_k != SHAPE_K and tot_default is not None \
                and best_tot >= 0.5 * tot_default:
            best_k = SHAPE_K
        shape_by_group[grp] = best_k
        return best_k

    for f in fits:
        f["k"] = group_shape(f["role"])
        f["tau_fit"] = fit_tau(f["ts"], f["ys"], f["ws"], f["k"]) \
            if len(f["ts"]) >= 2 else None

    # role-group tau medians (from trophies with a real fit), else the prior
    def role_tau(role):
        grp = SHRINK_GROUP.get(role, role)
        vals = [f["tau_fit"][0] for f in fits
                if SHRINK_GROUP.get(f["role"], f["role"]) == grp and f["tau_fit"] is not None]
        if vals:
            return median(vals)
        return ROLE_TAU_PRIOR.get(grp, 200.0)

    # final per-trophy (tau, A, residual) with tau shrunk toward the group median
    preds = []
    for f in fits:
        tau_grp = role_tau(f["role"])
        if not f["ts"]:
            preds.append(None)
            continue
        if f["tau_fit"] is not None:
            npts = len(f["ts"])
            ln_tau = (npts * math.log(f["tau_fit"][0]) + K_TAU * math.log(tau_grp)) \
                / (npts + K_TAU)
            tau = math.exp(ln_tau)
        else:
            tau = tau_grp
        a, _ = fit_level(f["ts"], f["ys"], f["ws"], tau, f["k"])
        # residual in log space, floored at half an earner so tiny counts do
        # not dominate the spread
        resids = []
        for t, y, pgen in zip(f["ts"], f["ys"], f["pgens"]):
            yhat = a * weib_z(t, tau, f["k"])
            fl = 0.5 / pgen
            resids.append(math.log(y + fl) - math.log(yhat + fl))
        if len(resids) >= 2:
            m = sum(resids) / len(resids)
            sd = math.sqrt(sum((r - m) ** 2 for r in resids) / (len(resids) - 1))
        else:
            sd = 0.0
        r_med = a * weib_z(t_med, tau, f["k"])
        e_pred = f["e0"] + p_gen_med * r_med
        preds.append({"tau": tau, "a": a, "sd": sd, "e_pred": e_pred,
                      "role": f["role"], "k": f["k"]})

    # NOTE on levels: there is deliberately NO blend toward a role-median level.
    # Role groups mix trophies whose levels differ by orders of magnitude (gold =
    # final mission ~30% vs 100% completion ~0%), so pooling levels pulls rare
    # trophies up badly. Each trophy's level is anchored by its own observed data;
    # only the adoption timescale tau is pooled across roles.

    rows = []
    for i, tr in enumerate(trophies):
        tid = tr["id"]
        role = roles[i]
        last_rate = last_rates.get(tid, 0.0)
        # the 100% trophy: it has been 100% in every snapshot; stay at ~100.
        # The scorer clamps the truth to 99.99, so the median must sit there too:
        # a median of 99.95 would already cost 1.6 logit units.
        if last_rate >= 99.99 and all(
                (s[2][i] is not None and s[2][i] >= 99.99) for s in snaps):
            rows.append({"id": tid, "q": [99.9, 99.97, 99.99, 99.99, 99.99]})
            continue
        pr = preds[i]
        if pr is None:
            # no usable history for this trophy: persistence + wide spread
            med = last_rate
            spread = max(S_FLOOR, 0.5 * math.sqrt(horizon_days))
        else:
            med = 100.0 * pr["e_pred"] / p_pred
            # refresh-gap spread: the last refresh before the target is random
            # (exponential gap, SD 1/lambda days); the rate moves with its local
            # log-slope over that gap
            s_gap = 0.0
            if pr["a"] > 0.0:
                t_hi = t_med + 12.0
                t_lo = max(0.0, t_med - 12.0)
                r_hi = pr["a"] * weib_z(t_hi, pr["tau"], pr["k"])
                r_lo = pr["a"] * weib_z(t_lo, pr["tau"], pr["k"])
                if r_hi > 0.0 and r_lo > 0.0 and t_hi > t_lo:
                    slope = abs(math.log(r_hi) - math.log(r_lo)) / (t_hi - t_lo)
                    s_gap = min(S_GAP_MAX, slope * (24.0 / lam_hat))
            spread = math.sqrt(pr["sd"] * pr["sd"] + p_resid * p_resid
                               + (S_HOR * math.sqrt(horizon_days)) ** 2
                               + S_FLOOR ** 2 + s_gap * s_gap)
            spread = min(S_MAX, spread * SPREAD_MULT.get(role, 1.0))
        med = clamp_pct(med)
        lg = logit_pct(med)
        qs = []
        for z in Z:
            q = inv_logit_pct(lg + spread * z)
            qs.append(round4(min(100.0, max(0.0, q))))
        # enforce non-decreasing (rounding can tie, never invert)
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


def fallback_case(case):
    """Persistence baseline (house baseline) — never crashes, always valid."""
    d = max(0.0, case["horizon_minutes"] / 1440.0)
    spread = 0.5 * math.sqrt(d)
    last = case["history"][-1] if case.get("history") else None
    rows = []
    for i, tr in enumerate(case["trophies"]):
        r = 0.0
        if last is not None and last["rates"][i] is not None:
            r = float(last["rates"][i])
        lg = logit_pct(r)
        qs = [round4(inv_logit_pct(lg + z * spread)) for z in Z]
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
            # validate shape before writing: 1 row per trophy, 5 non-decreasing q in [0,100]
            ids = [row["id"] for row in forecast["trophies"]]
            assert len(ids) == len(case["trophies"]) and len(set(ids)) == len(ids)
            for row in forecast["trophies"]:
                q = row["q"]
                assert len(q) == 5 and all(0.0 <= v <= 100.0 for v in q)
                assert all(q[k] >= q[k - 1] for k in range(1, 5))
        except Exception:
            forecast = fallback_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
