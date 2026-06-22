"""
APEX OMNI v9 — VOLATILITY-SURFACE FORECASTER
============================================
Options P&L is dominated by vega: you can be right on direction and still lose
because the surface crushed under you (classic post-event IV collapse), or win
extra because it expanded. The system already FITS the SVI surface every tick
and persists ATM-IV history — but it never PREDICTED where the surface is going.
This module does, using only data the system already records. No synthetic
series, no fabricated forward curve.

WHAT IT FORECASTS
-----------------
The ATM-IV change over the next VOL_FCAST_HORIZON_MIN minutes, decomposed into
three REAL, separately-justified drivers:

 1. MEAN-REVERSION — intraday IV is strongly mean-reverting to its own recent
    level (well documented; vol spikes decay). We fit an OU-style reversion
    coefficient from the persisted intraday samples and project the current
    deviation forward. This is the dominant term on a normal day.

 2. TERM-STRUCTURE CARRY — the front-expiry vs next-expiry IV slope tells you
    which way the surface is "rolling". A steep front (front IV >> next) on a
    short-dated contract reverts DOWN hard into expiry (the theta/vol bleed);
    an inverted front (event priced in next week) drifts up. Computed from the
    two nearest expiries the macro layer already snapshots.

 3. VOL-OF-VOL BAND — the empirical std of recent IV changes sets how WIDE the
    forecast cone is, and whether the current level is at a |z|>CRUSH_Z extreme
    (about to revert sharply = crush/expansion flag). This is what the regime
    classifier and the exit-shaper consume.

WHAT'S LEARNED vs FIXED
-----------------------
The reversion coefficient and vol-of-vol are REFIT each nightly forge from real
recorded IV (write_vol_model). Until VOL_FCAST_MIN_SAMPLES intraday points
exist, the forecaster returns `None` (no forecast) and every consumer falls back
to its non-forecast path — same dormancy contract as the meta-labeler and trap
learner. Nothing here ever fabricates an IV path.

OUTPUT
------
forecast() returns a VolForecast or None:
  iv_now, iv_fcast (level in HORIZON_MIN), d_iv (signed change), z (current
  level's z-score vs its band), regime in {CRUSH, EXPANSION, STABLE}, conf.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
import datetime as dt
import numpy as np

import config


@dataclass
class VolForecast:
    iv_now: float
    iv_fcast: float          # ATM-IV expected in HORIZON_MIN minutes
    d_iv: float              # signed change (iv_fcast - iv_now)
    z: float                 # z-score of iv_now vs recent band (+ = rich)
    vol_of_vol: float        # std of recent IV changes (the cone half-width)
    regime: str              # "CRUSH" | "EXPANSION" | "STABLE"
    horizon_min: int
    confidence: float        # 0..1, scales with sample count


# ----------------------------------------------------------------- model I/O
_model_cache = {"mtime": None, "params": None}


def _load_model() -> dict | None:
    """Learned reversion/vol params refit nightly; None until they exist."""
    path = config.VOL_FCAST_MODEL_PATH
    try:
        mt = os.path.getmtime(path)
    except OSError:
        _model_cache.update(mtime=None, params=None)
        return None
    if _model_cache["mtime"] == mt:
        return _model_cache["params"]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            p = json.load(fh)
        _model_cache.update(mtime=mt, params=p)
        return p
    except (OSError, ValueError):
        _model_cache.update(mtime=None, params=None)
        return None


def append_intraday_sample(index: str, atm_iv: float,
                           ts: float | None = None) -> None:
    """Record one intraday ATM-IV sample. Called by the macro loop each cycle so
    the forecaster has an intraday series (the daily history is too coarse to
    forecast crush). Bounded ring of the most recent samples per index."""
    path = config.IV_INTRADAY_PATH.format(idx=index)
    ts = ts if ts is not None else dt.datetime.now().timestamp()
    arr = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                arr = json.load(fh)
        except (OSError, ValueError):
            arr = []
    arr.append([round(ts, 1), round(float(atm_iv), 6)])
    # keep ~3 sessions of minute samples
    arr = arr[-max(config.VOL_FCAST_MIN_SAMPLES * 6, 1200):]
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(arr, fh)
        os.replace(tmp, path)
    except OSError:
        pass


def _load_intraday(index: str) -> np.ndarray:
    path = config.IV_INTRADAY_PATH.format(idx=index)
    if not os.path.exists(path):
        return np.empty((0, 2))
    try:
        with open(path, "r", encoding="utf-8") as fh:
            arr = json.load(fh)
        return np.asarray(arr, float)
    except (OSError, ValueError):
        return np.empty((0, 2))


# ----------------------------------------------------------------- forecast
def forecast(index: str, iv_now: float,
             front_iv: float | None = None,
             next_iv: float | None = None,
             dte: float | None = None) -> VolForecast | None:
    """Forecast ATM-IV `VOL_FCAST_HORIZON_MIN` minutes ahead from the REAL
    recorded intraday IV series. Returns None until enough samples exist."""
    samples = _load_intraday(index)
    if len(samples) < config.VOL_FCAST_MIN_SAMPLES:
        return None
    ts = samples[:, 0]
    iv = samples[:, 1]
    # restrict to the trailing window that matters (most recent session-ish)
    iv_recent = iv[-config.VOL_FCAST_MIN_SAMPLES * 3:]
    mu = float(np.mean(iv_recent))
    # Floor the dispersion at a fraction of the IV LEVEL. On a calm day the raw
    # std collapses toward zero, which made the z-score saturate on noise and flip
    # the regime every tick; requiring a deviation of at least a few % of the IV
    # level keeps CRUSH/EXPANSION meaning a genuine move, not jitter.
    raw_sd = float(np.std(iv_recent))
    sd = max(raw_sd,
             float(getattr(config, "VOL_FCAST_SD_FLOOR_FRAC", 0.05)) * abs(mu),
             1e-6)

    # --- driver 1: mean-reversion (learned coefficient, else fixed fallback) ---
    params = _load_model()
    k = float(params["revert_k"]) if params and "revert_k" in params \
        else config.VOL_FCAST_REVERT_K
    # per-sample step ≈ 1 macro cycle; scale to the horizon by cycle spacing
    dts = np.diff(ts[-50:]) if len(ts) >= 3 else np.array([180.0])
    step_s = float(np.median(dts[dts > 0])) if np.any(dts > 0) else 180.0
    n_steps = max(config.VOL_FCAST_HORIZON_MIN * 60.0 / max(step_s, 1.0), 1.0)
    # discrete OU projection: iv_t+ = mu + (iv_now - mu)*(1-k)^n
    reverted = mu + (iv_now - mu) * (1.0 - k) ** n_steps
    d_revert = reverted - iv_now

    # --- driver 2: term-structure carry into expiry ---
    d_carry = 0.0
    if front_iv is not None and next_iv is not None and dte is not None:
        slope = front_iv - next_iv                # >0 = front rich (reverts down)
        # the closer to expiry, the faster a rich front bleeds toward the back
        carry_speed = float(np.clip(1.0 / max(dte, 0.25), 0.0, 2.0))
        cf = (params.get("carry_w", 0.30) if params else 0.30)
        d_carry = -slope * carry_speed * cf * (config.VOL_FCAST_HORIZON_MIN / 60.0)

    # --- driver 3: vol-of-vol band + crush/expansion flag ---
    dIV = np.diff(iv_recent)
    vov = float(np.std(dIV)) * np.sqrt(n_steps) if len(dIV) else sd
    z = (iv_now - mu) / sd

    d_iv = d_revert + d_carry
    iv_fcast = max(iv_now + d_iv, 1e-4)

    regime = "STABLE"
    if z >= config.VOL_FCAST_CRUSH_Z and d_iv < 0:
        regime = "CRUSH"          # rich IV reverting down → vega longs bleed
    elif z <= -config.VOL_FCAST_CRUSH_Z and d_iv > 0:
        regime = "EXPANSION"      # cheap IV expanding → vega longs benefit

    conf = float(np.clip(len(iv_recent) / (config.VOL_FCAST_MIN_SAMPLES * 3),
                         0.0, 1.0))
    return VolForecast(iv_now=iv_now, iv_fcast=iv_fcast, d_iv=d_iv, z=z,
                       vol_of_vol=vov, regime=regime,
                       horizon_min=config.VOL_FCAST_HORIZON_MIN, confidence=conf)


# ----------------------------------------------------------------- nightly fit
def write_vol_model(indices=None) -> dict | None:
    """Refit the mean-reversion coefficient and carry weight from the REAL
    intraday IV series across indices, by maximizing one-step prediction fit
    (OU regression: ΔIV ~ -k·(IV - mean)). Writes VOL_FCAST_MODEL_PATH only when
    there are enough samples; else leaves the fixed fallback in place."""
    indices = indices or config.TRADABLE
    dx, dy = [], []
    for idx in indices:
        s = _load_intraday(idx)
        if len(s) < config.VOL_FCAST_MIN_SAMPLES:
            continue
        iv = s[:, 1]
        mu = np.mean(iv[-config.VOL_FCAST_MIN_SAMPLES * 3:])
        x = (iv[:-1] - mu)          # deviation
        y = (iv[1:] - iv[:-1])      # next-step change
        dx.append(x); dy.append(y)
    if not dx:
        return None
    X = np.concatenate(dx); Y = np.concatenate(dy)
    if len(X) < config.VOL_FCAST_MIN_SAMPLES or np.var(X) < 1e-12:
        return None
    # OLS slope of ΔIV on deviation → -k ; clamp to a sane reversion range
    k = float(-np.cov(X, Y, bias=True)[0, 1] / (np.var(X) + 1e-12))
    k = float(np.clip(k, 0.01, 0.60))
    vov = float(np.std(Y))
    model = {"revert_k": k, "carry_w": 0.30, "vol_of_vol": vov,
             "n_samples": int(len(X)),
             "fit_utc": dt.datetime.utcnow().isoformat()}
    os.makedirs(os.path.dirname(config.VOL_FCAST_MODEL_PATH), exist_ok=True)
    tmp = config.VOL_FCAST_MODEL_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2)
    os.replace(tmp, config.VOL_FCAST_MODEL_PATH)
    return model