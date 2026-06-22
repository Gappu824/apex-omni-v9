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
  TREND_UP / TREND_DOWN — directional: |trend efficiency| high. Up/down is taken
        from the SIGNED net move over the window (not spot-vs-flip), so the JUDGE
        "performance by regime" stats are trustworthy. Momentum entries belong
        here; conviction is BOOSTED.
  SQUEEZE_PRONE — dealers SHORT gamma (net GEX deep negative) near a wall:
        hedging amplifies moves, breakouts run. Conviction is BOOSTED.
  CHOP — range-bound: low trend efficiency, dealers long gamma (net GEX > 0 =
        mean-reverting tape). Momentum signals bleed; conviction is DAMPENED.
  VOL_CRUSH — the vol forecaster flags rich IV reverting down: long-premium
        vega bleed regardless of direction; conviction dampened (theta+vega tax).
  HIGH_VOL — realized/implied vol extreme with NO directional or squeeze edge:
        size down. Checked AFTER the directional regimes so a high-vol trend or
        squeeze still boosts (strong trends generate vol — they should not be
        dampened just for being volatile).

PRIORITY (first match wins)
---------------------------
  VOL_CRUSH → SQUEEZE_PRONE → TREND → HIGH_VOL → CHOP → NEUTRAL
HIGH_VOL deliberately sits BELOW the directional regimes: on the cleanest
trending days realized vol routinely clears the high-vol cut, and those days
should be boosted, not dampened. HIGH_VOL therefore only catches the
non-directional volatile tape.

EACH regime carries a `conv_mult` (a multiplier on the entry conviction the brain
already computes) and a one-line `note`. The multiplier only ever SCALES the
existing conviction; it never moves a risk floor and never forces an entry.

LEARNED vs FIXED
----------------
The cut points (what counts as "high" trend efficiency, "deep" negative GEX, etc.)
start as reasoned fixed thresholds and are refit nightly from the real
distribution of those features across harvested sessions (write_regime_model) —
so "high trend efficiency" becomes the empirical 70th percentile of THIS market,
not a hand-set guess. The deep-negative-GEX cut is the 10th percentile FLOORED at
the fixed deep default, so "deep negative" always means deep. Dormant on the
fixed thresholds until enough history.
"""
from __future__ import annotations
import json
import os
from collections import deque
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

# --- hot-path log state (issue #3) ---
_last_log_ts: dict[str, float] = {}   # per-index last-write epoch (rate limit)
_dir_ready = False                    # makedirs once, not every tick

# --- label hysteresis state (issue #5; dormant unless REGIME_HYSTERESIS_N > 1) ---
_regime_state: dict[str, dict] = {}   # per-index {"regime": Regime, "pending": str, "count": int}


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


def _apply_hysteresis(index: str | None, raw: Regime) -> Regime:
    """Require REGIME_HYSTERESIS_N consecutive ticks on a NEW label before the
    regime switches, so the multiplier can't flicker tick-to-tick near a cut
    boundary. Dormant by default (N=1 → returns `raw` unchanged, identical to the
    stateless classifier). Needs the per-index key; without it, no hysteresis."""
    n = int(getattr(config, "REGIME_HYSTERESIS_N", 1))
    if not index or n <= 1:
        return raw
    st = _regime_state.get(index)
    if st is None or st.get("regime") is None:
        _regime_state[index] = {"regime": raw, "pending": raw.label, "count": 0}
        return raw
    held: Regime = st["regime"]
    if raw.label == held.label:                 # same regime → refresh its values
        st["regime"] = raw
        st["pending"] = raw.label
        st["count"] = 0
        return raw
    # a different label proposed → count consecutive agreement before switching
    if st["pending"] == raw.label:
        st["count"] += 1
    else:
        st["pending"] = raw.label
        st["count"] = 1
    if st["count"] >= n:
        st["regime"] = raw
        st["count"] = 0
        return raw
    return held                                 # not enough agreement yet → hold


def classify(*, spot: float, trend_efficiency: float,
             net_gex: float | None, flip: float | None,
             call_wall: float | None, put_wall: float | None,
             iv_rank: float | None, realized_vol: float | None,
             vol_regime: str | None = None,
             vol_z: float | None = None,
             trend_sign: int | None = None,
             index: str | None = None) -> Regime:
    """Label the regime from REAL state. All inputs are values the macro/brain
    layers already compute. Returns a Regime with a conviction multiplier.

    `trend_efficiency` is the (unsigned) efficiency-ratio MAGNITUDE; `trend_sign`
    carries the direction of the net move (+1 up, -1 down, 0/None unknown) and is
    used only to label TREND_UP vs TREND_DOWN. `index` enables per-index
    hysteresis when REGIME_HYSTERESIS_N > 1 (otherwise ignored)."""
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

    hi_vol = ((realized_vol is not None and realized_vol >= RV_HIGH) or
              (iv_rank is not None and iv_rank >= IVR_HIGH))

    # --- priority: CRUSH → SQUEEZE → TREND → HIGH_VOL → CHOP → NEUTRAL ---
    # 1) VOL_CRUSH — vega bleed dominates; dampen long-premium conviction
    if vol_regime == "CRUSH":
        raw = Regime("VOL_CRUSH", config.REGIME_MULT_CRUSH, te, gex_sign,
                     f"rich IV reverting down (z {vol_z:+.1f}) — premium bleed; "
                     "conviction dampened")
    # 2) SQUEEZE_PRONE — dealers short gamma near a wall → moves amplify
    elif gex_sign < 0 and net_gex is not None and net_gex <= GEX_SQUEEZE \
            and near_wall:
        raw = Regime("SQUEEZE_PRONE", config.REGIME_MULT_SQUEEZE, te, gex_sign,
                     "dealers short gamma at a wall — breakout risk; conviction "
                     "boosted on a wall break")
    # 3) TREND — directional efficiency high; up/down from the SIGNED net move
    elif te >= TE_TREND:
        if trend_sign is not None and trend_sign != 0:
            side_up = trend_sign > 0
        else:                                   # fallback: spot vs gamma flip
            side_up = flip is None or spot >= flip
        if side_up:
            raw = Regime("TREND_UP", config.REGIME_MULT_TREND, te, gex_sign,
                         "clean up-trend (high efficiency)")
        else:
            raw = Regime("TREND_DOWN", config.REGIME_MULT_TREND, te, gex_sign,
                         "clean down-trend (high efficiency)")
    # 4) HIGH_VOL — vol extreme with NO directional/squeeze edge: size down
    elif hi_vol:
        raw = Regime("HIGH_VOL", config.REGIME_MULT_HIGHVOL, te, gex_sign,
                     "vol extreme (non-directional) — size down")
    # 5) CHOP — low efficiency, dealers long gamma (mean-reverting tape)
    elif te <= TE_CHOP and gex_sign >= 0:
        raw = Regime("CHOP", config.REGIME_MULT_CHOP, te, gex_sign,
                     "range-bound, dealers long gamma — momentum bleeds; "
                     "conviction dampened")
    # neutral / transitional: no scaling
    else:
        raw = Regime("NEUTRAL", 1.0, te, gex_sign, "transitional — no regime edge")

    return _apply_hysteresis(index, raw)


def write_regime_model(feature_rows: list[dict] | None = None) -> dict | None:
    """Refit the regime cut points from the REAL distribution of trend-efficiency
    and net-GEX across harvested sessions: 'high efficiency' becomes the empirical
    70th percentile of THIS market, 'deep negative GEX' the 10th percentile FLOORED
    at the fixed deep default (so a refit can only make the squeeze cut deeper,
    never shallower). `feature_rows` is the per-cycle macro feature log; None/short
    → keep fixed thresholds. No fabricated data — percentiles of recorded values
    only."""
    rows = feature_rows or _load_feature_log()
    te = np.array([abs(r["trend_efficiency"]) for r in rows
                   if r.get("trend_efficiency") is not None], float)
    gex = np.array([r["net_gex"] for r in rows
                    if r.get("net_gex") is not None], float)
    if len(te) < config.REGIME_MIN_SAMPLES or len(gex) < config.REGIME_MIN_SAMPLES:
        return None
    # deep-negative GEX cut: 10th percentile, but never shallower than the fixed
    # deep default — "deep negative" must actually mean deep (issue #1).
    gex_cut = float(min(np.percentile(gex, 10), config.REGIME_GEX_SQUEEZE))
    params = {
        "te_trend": float(np.clip(np.percentile(te, 70), 0.25, 0.85)),
        "te_chop": float(np.clip(np.percentile(te, 30), 0.10, 0.45)),
        "gex_squeeze": gex_cut,
        "n_samples": int(len(te)),
        "fit_utc": dt.datetime.utcnow().isoformat()}
    os.makedirs(os.path.dirname(config.REGIME_MODEL_PATH), exist_ok=True)
    tmp = config.REGIME_MODEL_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2)
    os.replace(tmp, config.REGIME_MODEL_PATH)
    _trim_feature_log()                          # rotate the JSONL to its tail
    return params


def _load_feature_log() -> list[dict]:
    """Read the regime-feature JSONL, capped to the most recent
    REGIME_FEATURE_LOG_MAX rows so neither memory nor the nightly fit grows with
    an unbounded file (issue #3)."""
    path = config.REGIME_FEATURE_LOG
    if not os.path.exists(path):
        return []
    cap = int(getattr(config, "REGIME_FEATURE_LOG_MAX", 60000))
    keep: deque = deque(maxlen=cap)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    keep.append(line)
    except OSError:
        return []
    out = []
    for line in keep:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _trim_feature_log() -> None:
    """Rewrite the feature log to its most recent REGIME_FEATURE_LOG_MAX rows.
    Called from the nightly refit so the on-disk JSONL is rotated, not appended
    forever (issue #3)."""
    path = config.REGIME_FEATURE_LOG
    if not os.path.exists(path):
        return
    cap = int(getattr(config, "REGIME_FEATURE_LOG_MAX", 60000))
    keep: deque = deque(maxlen=cap)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    keep.append(line.rstrip("\n"))
    except OSError:
        return
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(keep) + ("\n" if keep else ""))
        os.replace(tmp, path)
    except OSError:
        pass


def log_features(index: str, trend_efficiency: float,
                 net_gex: float | None) -> None:
    """Append one regime-feature row (JSONL) for the nightly percentile fit.
    Rate-limited to one row per index per REGIME_LOG_EVERY_S seconds and with a
    one-time directory create, to keep it off the live hot path (issue #3). Real
    recorded values only."""
    global _dir_ready
    every = float(getattr(config, "REGIME_LOG_EVERY_S", 30.0))
    now = dt.datetime.now().timestamp()
    if now - _last_log_ts.get(index, 0.0) < every:
        return
    path = config.REGIME_FEATURE_LOG
    rec = {"ts": round(now, 1), "index": index,
           "trend_efficiency": round(float(trend_efficiency), 4),
           "net_gex": (None if net_gex is None else float(net_gex))}
    try:
        if not _dir_ready:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            _dir_ready = True
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        _last_log_ts[index] = now
    except OSError:
        pass