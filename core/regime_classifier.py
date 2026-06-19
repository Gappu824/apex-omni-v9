"""
APEX OMNI v9 — MARKET REGIME CLASSIFIER
=======================================
The system computes a dozen state variables — trend strength, GEX sign, IV-rank,
realized vol, PCR, the vol forecast — but until now nothing LABELLED the market
and conditioned behaviour on the label. The same momentum logic ran in a dead
flat tape as in a clean trend, which is exactly why the live logs showed
persistent bullish signals going nowhere on a sideways day.

This classifier names the regime from data the system ALREADY harvests — no new
feed, no fabricated input — and exposes it so the brain can (a) annotate every
heartbeat with the regime, and (b) gate/scale conviction by it. It does NOT
silently veto trades on its own; it surfaces the regime and a recommended
conviction multiplier, leaving the risk envelope untouched.

THE REGIMES (mutually exclusive, from real signals)
---------------------------------------------------
  TREND_UP / TREND_DOWN — directional: |trend efficiency| high AND price holding
        one side of the gamma flip. Momentum entries belong here.
  CHOP — range-bound: low trend efficiency, price pinned between GEX walls near
        max-pain, dealers long gamma (net GEX > 0 = mean-reverting tape). This is
        where momentum signals bleed; conviction is DAMPENED here.
  SQUEEZE_PRONE — dealers SHORT gamma (net GEX << 0) near a wall: hedging
        amplifies moves, breakouts run. Conviction is BOOSTED on a wall break.
  VOL_CRUSH — the vol forecaster flags rich IV reverting down: long-premium
        vega bleed regardless of direction; conviction dampened (theta+vega tax).
  HIGH_VOL — realized/implied vol extreme: wider stops needed, smaller size.

EACH regime carries a `conv_mult` (a multiplier on the entry conviction the brain
already computes) and a one-line `note`. The multiplier only ever SCALES the
existing conviction; it never moves a risk floor and never forces an entry.

LEARNED vs FIXED
----------------
The cut points (what counts as "high" trend efficiency, "deep" negative GEX, etc.)
start as reasoned fixed thresholds and are refit nightly from the real
distribution of those features across harvested sessions (write_regime_model) —
so "high trend efficiency" becomes the empirical 70th percentile of THIS market,
not a hand-set guess. Dormant on the fixed thresholds until enough history.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
import datetime as dt
import numpy as np

import config


@dataclass
class Regime:
    label: str               # TREND_UP|TREND_DOWN|CHOP|SQUEEZE_PRONE|VOL_CRUSH|HIGH_VOL
    conv_mult: float         # multiplier on the brain's entry conviction
    trend_eff: float         # |efficiency ratio| used
    gex_sign: int            # +1 dealers long gamma, -1 short, 0 unknown
    note: str


_cache = {"mtime": None, "params": None}


def _load_model() -> dict | None:
    path = config.REGIME_MODEL_PATH
    try:
        mt = os.path.getmtime(path)
    except OSError:
        _cache.update(mtime=None, params=None)
        return None
    if _cache["mtime"] == mt:
        return _cache["params"]
    try:
        with open(path, "r", encoding="utf-8") as fh:
            p = json.load(fh)
        _cache.update(mtime=mt, params=p)
        return p
    except (OSError, ValueError):
        _cache.update(mtime=None, params=None)
        return None


def _thr(name: str, fallback: float) -> float:
    p = _load_model()
    if p and name in p:
        try:
            return float(p[name])
        except (ValueError, TypeError):
            return fallback
    return fallback


def classify(*, spot: float, trend_efficiency: float,
             net_gex: float | None, flip: float | None,
             call_wall: float | None, put_wall: float | None,
             iv_rank: float | None, realized_vol: float | None,
             vol_regime: str | None = None,
             vol_z: float | None = None) -> Regime:
    """Label the regime from REAL state. All inputs are values the macro/brain
    layers already compute. Returns a Regime with a conviction multiplier."""
    te = abs(float(trend_efficiency))
    gex_sign = 0 if net_gex is None else (1 if net_gex > 0 else -1)

    # learned-or-fixed cut points
    TE_TREND = _thr("te_trend", config.REGIME_TE_TREND)        # high efficiency
    TE_CHOP = _thr("te_chop", config.REGIME_TE_CHOP)           # low efficiency
    GEX_SQUEEZE = _thr("gex_squeeze", config.REGIME_GEX_SQUEEZE)  # deep negative
    IVR_HIGH = _thr("ivr_high", config.IVRANK_HIGH)
    RV_HIGH = _thr("rv_high", config.REGIME_RV_HIGH)

    near_wall = False
    if call_wall and put_wall and spot > 0:
        band = (call_wall - put_wall)
        if band > 0:
            d = min(abs(spot - call_wall), abs(spot - put_wall)) / spot
            near_wall = d <= config.REGIME_WALL_PROX_PCT

    # --- priority order: crush / high-vol guards first, then structure ---
    # 1) VOL_CRUSH — vega bleed dominates; dampen long-premium conviction
    if vol_regime == "CRUSH":
        return Regime("VOL_CRUSH", config.REGIME_MULT_CRUSH, te, gex_sign,
                      f"rich IV reverting down (z {vol_z:+.1f}) — premium bleed; "
                      "conviction dampened")
    # 2) HIGH_VOL — realized/implied extreme: smaller, wider
    if (realized_vol is not None and realized_vol >= RV_HIGH) or \
            (iv_rank is not None and iv_rank >= IVR_HIGH):
        return Regime("HIGH_VOL", config.REGIME_MULT_HIGHVOL, te, gex_sign,
                      "vol extreme — size down, stops wider")
    # 3) SQUEEZE_PRONE — dealers short gamma near a wall → moves amplify
    if gex_sign < 0 and net_gex is not None and net_gex <= GEX_SQUEEZE \
            and near_wall:
        return Regime("SQUEEZE_PRONE", config.REGIME_MULT_SQUEEZE, te, gex_sign,
                      "dealers short gamma at a wall — breakout risk; conviction "
                      "boosted on a wall break")
    # 4) TREND — directional efficiency high and holding a side of the flip
    if te >= TE_TREND:
        side_up = flip is None or spot >= flip
        if side_up:
            return Regime("TREND_UP", config.REGIME_MULT_TREND, te, gex_sign,
                          "clean up-trend (high efficiency, above gamma flip)")
        return Regime("TREND_DOWN", config.REGIME_MULT_TREND, te, gex_sign,
                      "clean down-trend (high efficiency, below gamma flip)")
    # 5) CHOP — low efficiency, dealers long gamma (mean-reverting tape)
    if te <= TE_CHOP and gex_sign >= 0:
        return Regime("CHOP", config.REGIME_MULT_CHOP, te, gex_sign,
                      "range-bound, dealers long gamma — momentum bleeds; "
                      "conviction dampened")
    # neutral / transitional: no scaling
    return Regime("NEUTRAL", 1.0, te, gex_sign, "transitional — no regime edge")


def write_regime_model(feature_rows: list[dict] | None = None) -> dict | None:
    """Refit the regime cut points from the REAL distribution of trend-efficiency
    and net-GEX across harvested sessions: 'high efficiency' becomes the empirical
    70th percentile of THIS market, 'deep negative GEX' the 20th percentile, etc.
    `feature_rows` is the per-cycle macro feature log; None/short → keep fixed
    thresholds. No fabricated data — percentiles of recorded values only."""
    rows = feature_rows or _load_feature_log()
    te = np.array([abs(r["trend_efficiency"]) for r in rows
                   if r.get("trend_efficiency") is not None], float)
    gex = np.array([r["net_gex"] for r in rows
                    if r.get("net_gex") is not None], float)
    if len(te) < config.REGIME_MIN_SAMPLES or len(gex) < config.REGIME_MIN_SAMPLES:
        return None
    params = {
        "te_trend": float(np.clip(np.percentile(te, 70), 0.25, 0.85)),
        "te_chop": float(np.clip(np.percentile(te, 30), 0.10, 0.45)),
        "gex_squeeze": float(np.percentile(gex, 20)),   # deep-negative cut
        "n_samples": int(len(te)),
        "fit_utc": dt.datetime.utcnow().isoformat()}
    os.makedirs(os.path.dirname(config.REGIME_MODEL_PATH), exist_ok=True)
    tmp = config.REGIME_MODEL_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2)
    os.replace(tmp, config.REGIME_MODEL_PATH)
    return params


def _load_feature_log() -> list[dict]:
    path = config.REGIME_FEATURE_LOG
    if not os.path.exists(path):
        return []
    out = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except (OSError, ValueError):
        return []
    return out


def log_features(index: str, trend_efficiency: float,
                 net_gex: float | None) -> None:
    """Append one regime-feature row (JSONL) for the nightly percentile fit.
    Real recorded values only."""
    path = config.REGIME_FEATURE_LOG
    rec = {"ts": round(dt.datetime.now().timestamp(), 1), "index": index,
           "trend_efficiency": round(float(trend_efficiency), 4),
           "net_gex": (None if net_gex is None else float(net_gex))}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except OSError:
        pass
