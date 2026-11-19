"""Forecast each GTA VI PS5 trophy's Exophase earn rate at the case target.

Model (earn-rate decomposition). A trophy's rate is earners over players tracked:
    rate_i(t) = E_i(t) / P(t).
Exophase's page is a step function: flat between refreshes, jumping at each refresh,
when new (mostly fresh) players join and dilute most rates. So instead of trending
the rate itself, this program fits the two counts separately in log time and divides:

  * P(t): players tracked.  ln P is fitted linear in ln(1 + t) over the case's distinct
    snapshots (t = hours since the game's release); the fit extrapolates to the target.
  * E_i(t): earners of trophy i.  The first snapshot's earners are the launch-night
    cohort (reviewers / earliest players); what grows afterwards is the general
    population, so the fit is on the EXCESS earners E_i(t) - E_i(0), again linear in
    ln(1 + t).  Each trophy's slope is shrunk toward its role-group mean (roles read
    from the case's own trophy descriptions and tiers), because a handful of snapshots
    cannot pin a per-trophy slope on its own.  The predicted level is likewise blended
    toward the role's median prediction at the target.
  * The median rate is 100 * E_i(T) / P(T).  Quantiles are normal on the logit scale,
    with a spread from the fits' residuals plus a horizon term plus a FLOOR that never
    vanishes: a refresh can move any rate by ~1 logit at any time, so a narrow band is
    a guaranteed loss (the house baseline's practice coverage was 0 for exactly this
    reason).

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
K_SHRINK = 1.0        # slope shrinkage: effective extra observations from the role mean
S_FLOOR = 0.65        # logit spread floor (refresh risk; never vanishes)
S_HOR = 0.15          # horizon uncertainty, logit per sqrt-day
S_MAX = 2.5           # spread cap
BETA_PRIOR = 0.9      # P-growth exponent fallback when <2 snapshots
SPREAD_MULT = {"rarest": 1.3, "grind": 1.2, "story1": 1.0, "story2": 1.0,
               "story3": 1.0, "story4": 1.0, "final": 1.1, "mission": 1.0,
               "easy": 0.9, "first": 0.05, "misc": 1.0}
# Generous week-1 ceilings for the median (mature rates from a comparable
# open-world launch are higher; week-1 rates stay below these). Only binds when
# a log-linear extrapolation would run away toward 100%.
ROLE_CEIL = {"rarest": 5.0, "grind": 45.0, "story1": 85.0, "story2": 80.0,
             "story3": 75.0, "story4": 70.0, "final": 55.0, "mission": 60.0,
             "easy": 75.0, "first": 100.0, "misc": 60.0}
RECENCY = 1.0  # fit weights ramp 1 -> 1+RECENCY over snapshots (recent slope is
# steeper: earn-rate curves are convex in log time before they saturate)
EXCESS_OFF = 0.1     # offset added inside ln(excess) so zero growth is finite;
# smaller values let the fit see early growth sooner (less flattening)
# Soft saturation caps for the median (mature rates from a comparable open-world
# launch): med = M * (1 - exp(-med_raw / M)). Only bites when the raw forecast
# approaches the mature level; leaves slow trophies untouched.
ROLE_MAT = {"rarest": 5.0, "grind": 18.0, "story1": 70.0, "story2": 60.0,
            "story3": 52.0, "story4": 45.0, "final": 30.0, "mission": 35.0,
            "easy": 45.0, "first": 100.0, "misc": 22.0}
# Coarse groups for SLOPE shrinkage: chapters share a growth exponent even though
# their levels differ, so slopes pool across the story group; levels do not pool.
SHRINK_GROUP = {"story1": "story", "story2": "story", "story3": "story",
                "story4": "story", "final": "story", "mission": "story",
                "grind": "grind", "easy": "easy", "misc": "misc",
                "rarest": "rarest", "first": "first"}


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


def forecast_case(case):
    trophies = case["trophies"]
    n = len(trophies)
    snaps, release = distinct_snapshots(case)
    target = parse_time(case["target"])
    t_target = (target - release).total_seconds() / 3600.0
    x_target = math.log(1.0 + max(0.0, t_target))
    horizon_days = max(0.0, case["horizon_minutes"] / 1440.0)

    last = snaps[-1] if snaps else None
    last_rates = {}
    if last is not None:
        for i, tr in enumerate(trophies):
            r = last[2][i]
            last_rates[tr["id"]] = 0.0 if r is None else r

    # ---- players tracked: ln P ~ ln(1+t) ----
    p_pred, p_resid = None, 0.0
    if len(snaps) >= 2:
        xs = [math.log(1.0 + max(0.0, t)) for t, _, _ in snaps]
        ys = [math.log(p) for _, p, _ in snaps]
        ws = [1.0 + RECENCY * j / (len(xs) - 1) for j in range(len(xs))]
        a, b, sd = ols(xs, ys, ws)
        # never forecast a shrinking tracked population
        b = max(0.0, b)
        p_pred = math.exp(a + b * x_target)
        p_resid = sd
    if p_pred is None and last is not None:
        t_last = max(0.0, last[0])
        p_pred = last[1] * ((1.0 + max(0.0, t_target)) / (1.0 + t_last)) ** BETA_PRIOR
    if p_pred is None:
        p_pred = 1.0

    roles = [classify(tr) for tr in trophies]

    # ---- per-trophy excess-earner fits ----
    # E0 = earners at the first snapshot; excess E(t) - E0 is the general population.
    fits = []
    for i, tr in enumerate(trophies):
        pts = []
        e0 = None
        for t, p, rates in snaps:
            r = rates[i]
            if r is None:
                continue
            e = r / 100.0 * p
            if e0 is None:
                e0 = e
            excess = max(0.0, e - e0)
            pts.append((math.log(1.0 + max(0.0, t)), math.log(excess + EXCESS_OFF), e))
        if e0 is None:
            e0 = 0.0
        if len(pts) >= 2:
            ws = [1.0 + RECENCY * j / (len(pts) - 1) for j in range(len(pts))]
            a, b, sd = ols([p[0] for p in pts], [p[1] for p in pts], ws)
            b = max(0.0, b)
            fits.append({"e0": e0, "a": a, "b": b, "sd": sd, "n": len(pts),
                         "role": roles[i]})
        else:
            fits.append({"e0": e0, "a": None, "b": 0.0, "sd": 0.0, "n": len(pts),
                         "role": roles[i]})

    # role slope means (from trophies that have a fit), else global mean
    def role_slope(role):
        grp = SHRINK_GROUP.get(role, role)
        vals = [f["b"] for f in fits
                if SHRINK_GROUP.get(f["role"], f["role"]) == grp and f["a"] is not None]
        if vals:
            return median(vals)
        vals = [f["b"] for f in fits if f["a"] is not None]
        return median(vals) if vals else 0.0

    # pass 1: shrunk predictions of ln E(T) per trophy
    pred_ln_e = []
    for f in fits:
        if f["a"] is None:
            pred_ln_e.append(None)
            continue
        b_star = (f["n"] * f["b"] + K_SHRINK * role_slope(f["role"])) / (f["n"] + K_SHRINK)
        # E(T) = E0 + excess(T); excess(T) = exp(a + b* x_T) - EXCESS_OFF
        excess = math.exp(f["a"] + b_star * x_target) - EXCESS_OFF
        pred_ln_e.append(math.log(max(EXCESS_OFF, f["e0"] + max(0.0, excess))))

    # NOTE on levels: there is deliberately NO blend toward a role-median level.
    # Role groups mix trophies whose levels differ by orders of magnitude (gold =
    # final mission ~30% vs 100% completion ~0%), so pooling levels pulls rare
    # trophies up badly. Each trophy's level is anchored by its own observed E0;
    # only slopes are shrunk across roles (role_slope, coarse groups above).

    rows = []
    for i, tr in enumerate(trophies):
        tid = tr["id"]
        role = roles[i]
        f = fits[i]
        last_rate = last_rates.get(tid, 0.0)
        # the 100% trophy: it has been 100% in every snapshot; stay at ~100.
        # The scorer clamps the truth to 99.99, so the median must sit there too:
        # a median of 99.95 would already cost 1.6 logit units.
        if last_rate >= 99.99 and all(
                (s[2][i] is not None and s[2][i] >= 99.99) for s in snaps):
            rows.append({"id": tid, "q": [99.9, 99.97, 99.99, 99.99, 99.99]})
            continue
        if pred_ln_e[i] is None:
            # no usable history for this trophy: persistence + wide spread
            med = last_rate
            spread = max(S_FLOOR, 0.5 * math.sqrt(horizon_days))
        else:
            lv = pred_ln_e[i]
            e_pred = math.exp(lv)
            med = 100.0 * e_pred / p_pred
            # saturation guard: a log-linear extrapolation can run away toward
            # 100% for fast trophies; cap the median at the role ceiling, then
            # soften toward the role's mature level (rates saturate, they do
            # not keep growing linearly in log time).
            med = min(med, ROLE_CEIL.get(role, 60.0))
            m_mat = ROLE_MAT.get(role, 60.0)
            med = m_mat * (1.0 - math.exp(-med / m_mat))
            resid = f["sd"]
            spread = math.sqrt(resid * resid + p_resid * p_resid
                               + (S_HOR * math.sqrt(horizon_days)) ** 2
                               + S_FLOOR ** 2)
            spread = min(S_MAX, spread * SPREAD_MULT.get(role, 1.0))
        med = clamp_pct(med)
        lg = logit_pct(med)
        qs = []
        for z in Z:
            q = inv_logit_pct(lg + z * spread)
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
