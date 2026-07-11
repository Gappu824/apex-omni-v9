"""
APEX OMNI v9.4 — REALIZED-VOL FORECASTER (Pillar 3 core: HAR-RV)
================================================================
The variance risk premium is only measurable against a FORECAST of realized
variance — iv_rank is its shadow. This module is the model layer:

  • RV construction: 1-MINUTE sampled log returns (375/session). One-second
    returns drown in microstructure noise; one-minute is the defensible
    intraday compromise for index futures at this history depth
    (Liu–Patton–Sheppard 2015 on sampling; Andersen–Bollerslev lineage).
    RV_d = Σ r_min²; annualized vol = √(RV_d × 252).
  • Model: HAR-RV on log-RV (Corsi 2009) — the cascade of daily / weekly(5) /
    monthly(≤22) components that remains the standard RV benchmark fifteen
    years on. An |overnight-gap| regressor joins only when ≥8 fit days exist.
    OLS via numpy lstsq; ≥6 rows required to fit. No sklearn, no torch: at
    this vault depth a neural layer would be theater (it arrives in T2,
    gated on BEATING this model's certificate — PROGRAM.md Pillar 3).
  • Intraday projection: a diurnal cumulative-share profile (per-minute mean
    share of the day's RV) turns partial-day RV into an implied full-day
    figure and an annualized REMAINING-day vol — what a seller of premium at
    11:40 actually needs.

Artifacts: state/rv_model_{INDEX}.json {coef, cols, diurnal profile, fit
days, config_hash, ts}. CONSTITUTION: forecasts touch NO gate until
tools/rv_skill_report.py writes a passing skill certificate (walk-forward
QLIKE vs random-walk; prespecified in PROGRAM.md). The brain consumes this
module for TELEMETRY only in v9.4: live rv̂ and the measured VRP spread.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

import config

SESSION_MIN = 375                      # 09:15–15:30 IST
_MIN_FIT_DAYS = 6                      # rows needed for the 4-param core
_GAP_COL_MIN_DAYS = 8                  # add |overnight gap| only past this
_PROFILE_FLOOR = 0.02                  # early-minute share floor (projection)


# ------------------------------------------------------------ RV building
def rv_from_minute_closes(closes: np.ndarray) -> float:
    """Σ of squared 1-minute log returns over valid consecutive minutes.
    NaN minutes (no tick) are bridged by the last valid close — a silent
    minute contributes zero variance, exactly as the tape did."""
    c = np.asarray(closes, float)
    if c.size < 2:
        return 0.0
    # forward-fill
    last = np.nan
    f = c.copy()
    for i in range(f.size):
        if np.isnan(f[i]):
            f[i] = last
        else:
            last = f[i]
    f = f[~np.isnan(f)]
    if f.size < 2:
        return 0.0
    r = np.diff(np.log(f))
    r = r[np.isfinite(r)]
    return float(np.sum(r * r))


def cumulative_profile(closes: np.ndarray) -> np.ndarray | None:
    """This day's cumulative RV share by minute (length SESSION_MIN,
    monotone 0→1). None if the day carries no variance."""
    c = np.asarray(closes, float)
    last = np.nan
    f = c.copy()
    for i in range(f.size):
        if np.isnan(f[i]):
            f[i] = last
        else:
            last = f[i]
    r2 = np.zeros(SESSION_MIN)
    prev = np.nan
    for m in range(min(f.size, SESSION_MIN)):
        if np.isfinite(f[m]) and np.isfinite(prev) and prev > 0:
            r2[m] = math.log(f[m] / prev) ** 2
        if np.isfinite(f[m]):
            prev = f[m]
    tot = r2.sum()
    if tot <= 0:
        return None
    return np.cumsum(r2) / tot


def ann_vol(rv_daily: float) -> float:
    """Annualized vol from one day's realized variance."""
    return math.sqrt(max(rv_daily, 0.0) * 252.0)


# ------------------------------------------------------------ HAR fitting
def _design(logrv: np.ndarray, gaps: np.ndarray | None):
    """HAR design matrix rows for targets logRV_{d} given history < d."""
    rows, ys, idxs = [], [], []
    n = len(logrv)
    use_gap = gaps is not None and n >= _GAP_COL_MIN_DAYS
    for d in range(1, n):
        day = logrv[d - 1]
        week = float(np.mean(logrv[max(0, d - 5):d]))
        month = float(np.mean(logrv[max(0, d - 22):d]))
        row = [1.0, day, week, month]
        if use_gap:
            row.append(float(gaps[d]))
        rows.append(row)
        ys.append(logrv[d])
        idxs.append(d)
    return np.asarray(rows), np.asarray(ys), use_gap, idxs


def fit_har(rv_series: list[float], gaps: list[float] | None,
            profiles: list[np.ndarray], index: str,
            days: list[str]) -> dict | None:
    """Fit the HAR model + diurnal profile from per-day RV. Returns the
    model dict (also written to state/rv_model_{index}.json) or None when
    the history is too thin — thin is a verdict, not an error."""
    rv = np.asarray([max(x, 1e-12) for x in rv_series], float)
    if rv.size < _MIN_FIT_DAYS:
        return None
    logrv = np.log(rv)
    g = np.asarray(gaps, float) if gaps is not None else None
    X, y, use_gap, _ = _design(logrv, g)
    if X.shape[0] < X.shape[1] + 1:
        return None
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    prof = np.mean([p for p in profiles if p is not None], axis=0) \
        if any(p is not None for p in profiles) else None
    model = {"index": index, "coef": [float(c) for c in coef],
             "cols": (["const", "d", "w", "m", "gap"] if use_gap
                      else ["const", "d", "w", "m"]),
             "logrv_tail": [float(x) for x in logrv[-22:]],
             "profile": ([float(x) for x in prof]
                         if prof is not None else None),
             "fit_days": list(days), "n_days": int(rv.size),
             "config_hash": config.CONFIG_HASH, "ts": time.time()}
    try:
        path = config.STATE_DIR / f"rv_model_{index}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(model))
    except Exception:                                     # noqa: BLE001
        pass
    return model


def load_model(index: str) -> dict | None:
    try:
        m = json.loads(
            (config.STATE_DIR / f"rv_model_{index}.json").read_text())
        return m if m.get("coef") else None
    except Exception:                                     # noqa: BLE001
        return None


# ------------------------------------------------------------ prediction
def predict_next_day(model: dict, logrv_hist: list[float] | None = None,
                     gap: float = 0.0) -> float | None:
    """Next-day RV forecast (variance units) from the HAR coefficients and
    the last ≤22 log-RVs (defaults to the tail stored at fit time)."""
    h = np.asarray(logrv_hist if logrv_hist is not None
                   else model.get("logrv_tail") or [], float)
    if h.size < 1:
        return None
    row = [1.0, float(h[-1]), float(np.mean(h[-5:])),
           float(np.mean(h[-22:]))]
    if "gap" in model["cols"]:
        row.append(float(gap))
    coef = np.asarray(model["coef"], float)
    if len(row) != coef.size:
        return None
    return float(math.exp(float(np.dot(coef, row))))


def predict_remaining(model: dict, minute: int,
                      rv_sofar: float) -> dict | None:
    """Intraday projection at session-minute m with partial RV:
      implied_day  — full-day RV implied by the diurnal profile share
      rem_ann_vol  — annualized vol of the REMAINING window (the seller's
                     quantity), √(rem_RV × 252 × 375/(375−m))
      day_ann_vol  — annualized vol of the implied full day."""
    prof = model.get("profile")
    if not prof:
        return None
    m = int(min(max(minute, 0), SESSION_MIN - 1))
    share = max(float(prof[m]), _PROFILE_FLOOR)
    implied = max(rv_sofar, 0.0) / share
    rem = max(implied - rv_sofar, 0.0)
    mins_left = max(SESSION_MIN - m, 1)
    rem_ann = math.sqrt(rem * 252.0 * SESSION_MIN / mins_left)
    return {"implied_day_rv": implied, "rem_ann_vol": rem_ann,
            "day_ann_vol": ann_vol(implied), "minute": m,
            "profile_share": share}