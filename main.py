#!/usr/bin/env python3
"""GTA VI Twitch CCU forecaster (gta6-twitch-ccu-forecast-01).

Method: comparable-launch log-ratio ensemble with a locally weighted ridge
regression per forecast hour.

For every comparable launch we build the curve of log ratios
r_h = ln(viewers(h) / viewers(1)) for hours 2..24 of its release day.
Bigger launches have systematically flatter curves (their hour-1 print is
already huge, so the rest of the day grows less in proportion), so the
median ratio is not a constant: it is modelled per hour as a ridge
regression of r_h on three features of the launch

    [1, ln(hour-1 viewers), ln(hour-1 viewers / pre-7d mean viewers),
     ln(hour-1 channels)]

with a Gaussian kernel on ln(hour-1 viewers) (bandwidth 1.5 in log units)
so launches of a similar scale dominate the fit, and a ridge penalty
(5.0) on the slopes for stability when the target scale lies beyond the
largest comparable (as GTA VI's will).  Comparables carry a quality tier
in their unlock note ("Tier: comparable / early-access-contaminated /
weak / target-history"); contaminated and weak launches are downweighted
to 0.5 and the target's own pre-release history is excluded outright,
because early-access and free-to-play curves do not behave like a clean
paid launch.  The forecast quantiles are lognormal around the fitted
value, with the spread taken from the kernel-weighted residual standard
deviation (times 0.85, calibrated by leave-one-out cross-validation over
the comparables and by the four practice cases).  All quantiles are then
multiplied by the target's own hour-1 print.

Inputs:  $MAKEFUN_INPUT_DIR (default /input) holding <id>.in files,
schema makefun/twitch-ccu-case/v1.
Outputs: $MAKEFUN_OUTPUT_DIR (default /output) holding <id>.out files,
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}]} for h = 2..24.

Deterministic, standard library only, no network, no randomness.
"""
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


def norm_ppf(p):
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError("probability outside (0, 1)")
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


Z = tuple(norm_ppf(p) for p in LEVELS)

# Tuned on the four practice cases + leave-one-out over the comparables.
# Round 2 (Nov 17): ridge 5.0 + tier downweight + spread 0.85 beat round 1
# (ridge 1.0, no tiers, spread 0.9) on practice (0.159 vs 0.166), LOO
# (0.362 vs 0.363) and LOO over big launches (0.208 vs 0.215).
KERNEL_BW = 1.5      # Gaussian kernel bandwidth on ln(hour-1 viewers)
RIDGE_LAMBDA = 5.0   # ridge penalty on the slopes (intercept unpenalised)
SPREAD_MULT = 0.85   # multiplier on the residual standard deviation
MIN_POOL = 8         # minimum comparables carrying an hour to fit it

# Quality tiers parsed from each comparable's unlock note ("Tier: ...").
# Unknown tiers get 0.3; the target's own history is never a comparable.
TIER_WEIGHTS = {
    "comparable": 1.0,
    "weak": 0.5,
    "early-access-contaminated": 0.5,
    "target-history": 0.0,
}


def parse_tier(unlock_text):
    """Tier annotation embedded in a comparable's unlock note."""
    if not unlock_text:
        return "unknown"
    match = re.search(r"Tier:\s*([a-z\-]+)", unlock_text)
    return match.group(1) if match else "unknown"


def parse_ts(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def solve_linear(matrix, rhs):
    """Solve a small dense linear system by Gauss-Jordan elimination."""
    n = len(matrix)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            continue
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col] / pv
                for c in range(col, n + 1):
                    aug[r][c] -= factor * aug[col][c]
    out = []
    for i in range(n):
        out.append(aug[i][n] / aug[i][i] if abs(aug[i][i]) > 1e-12 else 0.0)
    return out


def comparable_curve(comp):
    """Hourly viewers {1..24} of a comparable's release day."""
    day_start = parse_ts(comp["day_start"])
    curve = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        delta = (parse_ts(point["from"]) - day_start).total_seconds() / 3600.0
        if delta == int(delta) and 0 <= int(delta) <= 23:
            curve[int(delta) + 1] = viewers
    return curve, day_start


def comparable_features(comp, curve, day_start):
    """Launch-level features used by the regression."""
    if 1 not in curve:
        return None
    v1 = curve[1]
    pre7 = []
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        t = parse_ts(point["from"])
        if day_start - timedelta(days=7) <= t < day_start:
            pre7.append(viewers)
    mean_pre7 = sum(pre7) / len(pre7) if pre7 else None
    channels1 = None
    for point in comp.get("hourly") or []:
        if parse_ts(point["from"]) == day_start:
            channels1 = point.get("channels")
            break
    return {
        "ln_v1": math.log(v1),
        # Pre-release hype: how far the hour-1 print sits above the
        # trailing 7-day mean.  Mean-imputed when the history is missing.
        "ln_hype": math.log(v1 / mean_pre7) if mean_pre7 else 2.2,
        "ln_ch1": math.log(channels1) if channels1 else 7.5,
        # Quality tier from the comparable's unlock note.
        "tier": parse_tier(comp.get("unlock")),
    }


def feature_vector(feat):
    return [1.0, feat["ln_v1"], feat["ln_hype"], feat["ln_ch1"]]


def fit_hour(pool, target_feat, hour):
    """Locally weighted ridge fit of log-ratio at one hour.

    pool: list of (curve, features).  Returns (mu, sd) in log-ratio units,
    or None when too few comparables carry that hour.
    """
    rows = []
    for curve, feat in pool:
        if hour in curve and 1 in curve:
            tier_w = TIER_WEIGHTS.get(feat.get("tier", "unknown"), 0.3)
            if tier_w == 0.0:
                continue  # never learn from the target's own history
            rows.append((feature_vector(feat),
                         math.log(curve[hour] / curve[1]),
                         feat["ln_v1"], tier_w))
    if len(rows) < MIN_POOL:
        return None
    x0 = target_feat["ln_v1"]
    weights = [math.exp(-0.5 * ((ln_v1 - x0) / KERNEL_BW) ** 2) * tier_w
               for _x, _y, ln_v1, tier_w in rows]
    k = len(rows[0][0])
    ata = [[0.0] * k for _ in range(k)]
    atb = [0.0] * k
    for (x, y, _ln, _tw), w in zip(rows, weights):
        for a in range(k):
            atb[a] += w * x[a] * y
            for b in range(k):
                ata[a][b] += w * x[a] * x[b]
    for a in range(1, k):
        ata[a][a] += RIDGE_LAMBDA
    beta = solve_linear(ata, atb)
    target_x = feature_vector(target_feat)
    mu = sum(b * t for b, t in zip(beta, target_x))
    total_w = sum(weights)
    var = 0.0
    for (x, y, _ln, _tw), w in zip(rows, weights):
        fitted = sum(b * v for b, v in zip(beta, x))
        var += w * (y - fitted) ** 2
    sd = math.sqrt(max(var / total_w, 1e-6)) * SPREAD_MULT
    return mu, sd


def pooled_median_ratio(pool, hour):
    """Fallback centre: plain median log-ratio across comparables."""
    ratios = sorted(math.log(curve[hour] / curve[1])
                    for curve, _f in pool if hour in curve and 1 in curve)
    if not ratios:
        return 0.0
    mid = len(ratios) // 2
    if len(ratios) % 2:
        return ratios[mid]
    return 0.5 * (ratios[mid - 1] + ratios[mid])


def round2(value):
    return float(Decimal(repr(value)).quantize(Decimal("0.01"),
                                               rounding=ROUND_HALF_UP))


def forecast_case(case):
    known = case.get("known") or []
    anchor = None
    for entry in known:
        if entry.get("hour") == 1 and entry.get("viewers"):
            anchor = entry
            break
    if anchor is None and known:
        anchor = known[0]
    if anchor is None:
        raise ValueError("case has no known hour-1 print")
    p1 = float(anchor["viewers"])
    if p1 <= 0:
        raise ValueError("hour-1 print is not positive")

    # Target features from the case's own pre-checkpoint history.
    day_start = parse_ts(case["day_start"])
    pre7 = [row["viewers"] for row in case.get("history_hourly") or []
            if row.get("viewers") and row["viewers"] > 0
            and day_start - timedelta(days=7) <= parse_ts(row["from"]) < day_start]
    mean_pre7 = sum(pre7) / len(pre7) if pre7 else None
    target_feat = {
        "ln_v1": math.log(p1),
        "ln_hype": math.log(p1 / mean_pre7) if mean_pre7 else 2.2,
        "ln_ch1": math.log(anchor["channels"]) if anchor.get("channels") else 7.5,
    }

    pool = []
    for comp in case.get("comparables") or []:
        curve, start = comparable_curve(comp)
        if 1 not in curve:
            continue
        if sum(1 for h in range(2, 25) if h in curve) < 20:
            continue
        feat = comparable_features(comp, curve, start)
        if feat is not None:
            pool.append((curve, feat))
    pool.sort(key=lambda item: item[1]["ln_v1"])

    levels = case.get("levels") or list(LEVELS)
    z_scores = [norm_ppf(p) for p in levels]
    hours = []
    for hour in range(2, 25):
        fit = fit_hour(pool, target_feat, hour)
        if fit is None:
            mu = pooled_median_ratio(pool, hour)
            sd = 0.35
        else:
            mu, sd = fit
        quantiles = []
        for z in z_scores:
            value = p1 * math.exp(mu + z * sd)
            value = min(max(value, 1.0), 1e9)
            quantiles.append(round2(value))
        for i in range(1, len(quantiles)):
            quantiles[i] = max(quantiles[i], quantiles[i - 1])
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    for in_path in sorted(input_dir.glob("*.in")):
        case = json.loads(in_path.read_text())
        result = forecast_case(case)
        out_path = output_dir / (in_path.stem + ".out")
        out_path.write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
