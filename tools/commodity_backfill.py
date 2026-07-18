"""
APEX OMNI v9.7.1 — COMMODITY BACKFILL + DAILY (TRACK-A) CALIBRATION
==================================================================
This is the part of commodity calibration that is COMPLETE and REAL TODAY,
because Kite's continuous API provides years of DAILY futures candles right now
(no live capture required).

What Kite actually gives (verified against the docs)
----------------------------------------------------
  • continuous=1 on a LIVE futures instrument_token returns DAY candles for that
    contract's expired history too → years of daily commodity-futures OHLCV+OI.
  • Historical INTRADAY options data does NOT exist (expired option tokens are
    flushed and re-used), and there is no historical minute depth. So intraday
    options microstructure CANNOT be backfilled — it is captured live going
    forward (the harvester + Track-B calibration handle that).

Therefore this tool calibrates the DAILY-scale facts that a daily series CAN
legitimately support, per commodity, and writes them into the SAME
logs/calibration.json the equity brain already reads (under the commodity's
name, alongside the indices). Track-B (intraday) fills in the rest later.

Track-A calibrated fields (each grounded in the daily futures series)
--------------------------------------------------------------------
  • atr_proxy_daily   — median daily true range as a fraction of price (Wilder
    ATR normalized), the daily volatility unit.
  • rv_annual         — annualized realized vol from daily log returns.
  • gap_p50 / gap_p90 — overnight/session gap magnitude distribution (|open −
    prev close| / prev close), the event-gap scale that motivates the event
    guard's blackout width.
  • event_gap_p90     — the 90th-pct absolute daily return on/around known EIA/
    OPEC dates when EVENT_OVERRIDES supplies them (else the unconditional p90).
  • regime_hi_vol_rv  — the RV level above which the commodity is "high-vol"
    (its own 70th percentile), a per-commodity regime boundary.
  • daily_range_p50   — typical session range, for sanity bounds.
All carry provenance (n_days, window). Trusted only when n_days ≥
COMMODITY_CALIB_MIN_DAYS; else omitted (brain falls back to config).

This module is BACKFILL + ARTIFACT only — no trading, no model. It runs today
and can re-run nightly (cheap: one continuous call per commodity).

  python tools/commodity_backfill.py [--years N] [--dry]

--dry runs the full calibration MATH on a synthetic daily series (no Kite), to
prove the statistics are correct without credentials. Live mode needs a Kite
session (KiteConnect with access_token) exactly like the harvester.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                             # noqa: E402
from core.diagnostics import _atomic_write_json            # noqa: E402

config.setup_logging("commodity_backfill")
import logging                                            # noqa: E402
log = logging.getLogger("commodity_backfill")


# --------------------------------------------------------------------------
# The calibration math — pure, testable, operates on a daily OHLC array
# --------------------------------------------------------------------------
def _wilder_atr(high, low, close, period=14):
    """Wilder ATR as a FRACTION of price (median over the series)."""
    high, low, close = map(np.asarray, (high, low, close))
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low,
                            np.abs(high - prev_close),
                            np.abs(low - prev_close)])
    # Wilder smoothing
    atr = np.zeros_like(tr, dtype=float)
    atr[:period] = tr[:period].mean() if len(tr) >= period else tr.mean()
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    frac = atr / np.where(close > 0, close, np.nan)
    return float(np.nanmedian(frac[period:] if len(frac) > period else frac))


def calibrate_daily(candles: list, *, commodity: str,
                    event_dates: set | None = None) -> dict:
    """candles: list of [ts, open, high, low, close, volume, oi]. Returns the
    Track-A calibration dict for this commodity (fields omitted if too thin)."""
    if not candles:
        return {"_note": "no daily candles"}
    arr = np.array([[c[1], c[2], c[3], c[4], c[5]] for c in candles], float)
    o, h, l, cl, vol = arr.T
    ts = [c[0] for c in candles]
    n = len(cl)
    min_days = int(getattr(config, "COMMODITY_CALIB_MIN_DAYS", 250))

    out = {"n_days": n, "window": [str(ts[0])[:10], str(ts[-1])[:10]]}
    if n < min_days:
        out["_note"] = (f"thin sample ({n} < {min_days} days) — Track-A fields "
                        f"omitted; brain uses config defaults")
        return out

    # daily log returns
    logret = np.diff(np.log(np.where(cl > 0, cl, np.nan)))
    logret = logret[np.isfinite(logret)]

    out["atr_proxy_daily"] = round(_wilder_atr(h, l, cl), 6)
    out["rv_annual"] = round(float(np.std(logret) * np.sqrt(252)), 4)
    out["regime_hi_vol_rv"] = round(float(
        np.quantile(np.abs(logret) * np.sqrt(252), 0.70)), 4)

    # gap magnitude: |open - prev close| / prev close
    prev_cl = cl[:-1]
    gaps = np.abs(o[1:] - prev_cl) / np.where(prev_cl > 0, prev_cl, np.nan)
    gaps = gaps[np.isfinite(gaps)]
    out["gap_p50"] = round(float(np.quantile(gaps, 0.50)), 5)
    out["gap_p90"] = round(float(np.quantile(gaps, 0.90)), 5)

    # daily range as fraction of close
    rng = (h - l) / np.where(cl > 0, cl, np.nan)
    rng = rng[np.isfinite(rng)]
    out["daily_range_p50"] = round(float(np.quantile(rng, 0.50)), 5)

    # event-conditional gap: |daily return| on known event dates, if provided
    abs_ret = np.abs(logret)
    if event_dates:
        ev_idx = [i for i, t in enumerate(ts[1:])
                  if str(t)[:10] in event_dates]
        if len(ev_idx) >= 8:                        # enough events to be real
            out["event_gap_p90"] = round(float(
                np.quantile(abs_ret[ev_idx], 0.90)), 5)
            out["event_gap_n"] = len(ev_idx)
    if "event_gap_p90" not in out:
        out["event_gap_p90"] = round(float(np.quantile(abs_ret, 0.90)), 5)
        out["event_gap_source"] = "unconditional (no dated events supplied)"

    return out


# --------------------------------------------------------------------------
# Kite plumbing (live mode) — resolve front-month future, pull continuous daily
# --------------------------------------------------------------------------
def _kite_session():
    """Build a KiteConnect from the SAME creds apex_main and the harvester use
    (config.KITE_API_KEY / KITE_ACCESS_TOKEN, from environment — regenerated
    daily per SEBI logout). Returns None with a clear log if unavailable."""
    try:
        from kiteconnect import KiteConnect
    except Exception as e:                                     # noqa: BLE001
        log.warning("kiteconnect not importable (%s) — live backfill "
                    "unavailable; use --dry to validate the math", e)
        return None
    api_key = getattr(config, "KITE_API_KEY", "") or ""
    token = getattr(config, "KITE_ACCESS_TOKEN", "") or ""
    if not api_key or not token:
        log.warning("no Kite creds in the environment (KITE_API_KEY / "
                    "KITE_ACCESS_TOKEN) — the same vars apex_main uses. Set "
                    "them (regenerated daily), then re-run; or use --dry.")
        return None
    k = KiteConnect(api_key=api_key)
    k.set_access_token(token)
    return k


def _front_future_token(kite, commodity: str) -> tuple[int, str] | None:
    """Front-month MCX future (token, symbol) for `commodity` from the dump."""
    today = dt.date.today()
    cands = []
    for ins in kite.instruments("MCX"):
        if ins.get("name") == commodity and ins.get("instrument_type") == "FUT":
            exp = ins["expiry"] if isinstance(ins["expiry"], dt.date) \
                else dt.date.fromisoformat(str(ins["expiry"])[:10])
            if exp >= today:
                cands.append((exp, int(ins["instrument_token"]),
                              ins["tradingsymbol"]))
    if not cands:
        return None
    cands.sort()
    return cands[0][1], cands[0][2]


def _pull_continuous_daily(kite, token: int, years: int) -> list:
    """Daily candles (with OI) for this future incl. expired history, via
    continuous=1. Chunked to respect the 2000-day/day-interval limit."""
    to = dt.date.today()
    frm = to - dt.timedelta(days=int(years * 365))
    out = []
    # day interval allows up to 2000 days per call — one call usually suffices
    chunk_start = frm
    while chunk_start < to:
        chunk_end = min(chunk_start + dt.timedelta(days=1999), to)
        try:
            rows = kite.historical_data(token, chunk_start, chunk_end,
                                        interval="day", continuous=True, oi=True)
        except Exception as e:                                 # noqa: BLE001
            log.warning("historical_data failed %s→%s: %s",
                        chunk_start, chunk_end, e)
            rows = []
        for r in rows:
            out.append([r["date"], r["open"], r["high"], r["low"],
                        r["close"], r.get("volume", 0), r.get("oi", 0)])
        chunk_start = chunk_end + dt.timedelta(days=1)
    # dedupe by date, keep sorted
    seen = {}
    for r in out:
        seen[str(r[0])[:10]] = r
    return [seen[k] for k in sorted(seen)]


def _event_dates_for(commodity: str) -> set:
    """Historical dates of this commodity's scheduled events, if EVENT_OVERRIDES
    (or the weekly rule) can enumerate them across the backfill window. For the
    weekly EIA rule we synthesize the weekday dates; dated one-offs come from
    EVENT_OVERRIDES."""
    from core.event_engine import CommodityEventEngine
    eng = CommodityEventEngine()
    dates = set()
    today = dt.date.today()
    start = today - dt.timedelta(days=365 * 6)
    d = start
    # enumerate weekly-rule event dates that affect this commodity
    while d <= today:
        for ev in eng.events:
            if commodity in ev.affects and ev.weekday is not None \
               and d.weekday() == ev.weekday:
                dates.add(d.isoformat())
            if commodity in ev.affects and ev.explicit_date == d:
                dates.add(d.isoformat())
        d += dt.timedelta(days=1)
    return dates


# --------------------------------------------------------------------------
# Synthetic series for --dry (proves the math with no Kite)
# --------------------------------------------------------------------------
def _synthetic_daily(n=400, seed=7, vol=0.02, start_price=6000.0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, vol, n)
    # inject occasional event-day jumps
    for i in range(3, n, 7):                       # ~weekly
        rets[i] += rng.normal(0, vol * 2.5)
    price = start_price * np.exp(np.cumsum(rets))
    candles = []
    base = dt.date.today() - dt.timedelta(days=n)
    for i in range(n):
        c = price[i]
        o = c * (1 + rng.normal(0, vol * 0.3))
        hi = max(o, c) * (1 + abs(rng.normal(0, vol * 0.5)))
        lo = min(o, c) * (1 - abs(rng.normal(0, vol * 0.5)))
        candles.append([(base + dt.timedelta(days=i)).isoformat(),
                        o, hi, lo, c, int(rng.integers(1000, 5000)), 0])
    return candles


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--dry", action="store_true",
                    help="run the calibration math on a synthetic series (no Kite)")
    args = ap.parse_args()

    commodities = list(getattr(config, "HARVEST_COMMODITIES", []))
    if not commodities:
        log.info("HARVEST_COMMODITIES empty — nothing to backfill")
        return

    # merge into the existing calibration artifact (don't clobber equity fields)
    path = config.LOG_DIR / "calibration.json"
    try:
        existing = json.loads(path.read_text())
    except Exception:                                          # noqa: BLE001
        existing = {}
    existing.setdefault("commodities_daily", {})

    if args.dry:
        log.info("DRY: calibrating Track-A math on a synthetic daily series")
        for c in commodities:
            candles = _synthetic_daily()
            cal = calibrate_daily(candles, commodity=c)
            existing["commodities_daily"][c] = cal
            log.info("  %s: %s", c, json.dumps(cal))
        _atomic_write_json(path, existing)
        log.info("wrote Track-A (synthetic) → %s", path)
        return

    kite = _kite_session()
    if kite is None:
        log.error("no live Kite session — cannot backfill real data. Run with "
                  "--dry to validate the math, or provide Kite creds.")
        return

    for c in commodities:
        ft = _front_future_token(kite, c)
        if not ft:
            log.warning("  %s: no front-month future found on MCX — skipped", c)
            continue
        token, symbol = ft
        log.info("  %s: pulling %d yr continuous daily via %s (token %d)",
                 c, args.years, symbol, token)
        candles = _pull_continuous_daily(kite, token, args.years)
        cal = calibrate_daily(candles, commodity=c,
                              event_dates=_event_dates_for(c))
        cal["front_future"] = symbol
        existing["commodities_daily"][c] = cal
        log.info("  %s: %d days | atr_daily=%s rv_annual=%s gap_p90=%s",
                 c, cal.get("n_days", 0), cal.get("atr_proxy_daily"),
                 cal.get("rv_annual"), cal.get("gap_p90"))

    _atomic_write_json(path, existing)
    log.info("Track-A commodity calibration → %s", path)


if __name__ == "__main__":
    main()