"""
APEX OMNI v9.7.1 — CALIBRATION LOADER & DYNAMIC LEVELS
======================================================
The read side of the nightly calibration. The brain calls calib() to get the
vault-measured numbers (hot-reloaded on mtime change, like the fly-intel
polarity), and dynamic_stop_target() to size the INITIAL stop and target from
the instrument's OWN realized volatility instead of a fixed percent.

Everything degrades to the conservative config default when the artifact is
missing or the field wasn't calibrated (thin sample) — calibration NARROWS
from data, never invents.

Research: volatility-scaled stops/targets (Kaufman, *Trading Systems and
Methods*) — a stop set in the instrument's own vol units is regime-robust; a
fixed-percent stop is too tight in calm regimes and too loose in violent ones.

Self-test:   python core/calibration.py
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass


def _cfg(name: str, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:                                          # pragma: no cover
        return default


_CAL_CACHE: dict = {"mtime": None, "data": {}}


def _path() -> str:
    p = _cfg("CALIBRATION_PATH", None)
    if p:
        return p
    try:
        import config
        return os.path.join(str(getattr(config, "LOG_DIR", ".")),
                            "calibration.json")
    except Exception:                                          # pragma: no cover
        return "calibration.json"


def calib() -> dict:
    """The full calibration artifact (hot-reloaded). {} if none yet."""
    # v9.7.1 (2026-07-19 RED-gate root cause): the regression suite is a
    # PROMOTION GATE and must be deterministic. The night real calibration
    # first landed, 3 scenarios flipped — same code, different thresholds,
    # reproduced to the rupee. Under APEX_HERMETIC_CAL=1 (set ONLY by
    # simulation/run_simulation around its scenarios) the loader returns
    # reference defaults; live and forge behaviour untouched.
    if os.environ.get("APEX_HERMETIC_CAL") == "1":
        return {}
    if not bool(_cfg("USE_CALIBRATION", True)):
        return {}
    if _CAL_CACHE.get("mtime") == "test":
        return _CAL_CACHE["data"]
    path = _path()
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return {}
    if _CAL_CACHE["mtime"] == mt:
        return _CAL_CACHE["data"]
    try:
        with open(path) as fh:
            d = json.load(fh)
    except Exception:                                          # pragma: no cover
        d = {}
    _CAL_CACHE.update({"mtime": mt, "data": d})
    return d


def index_calib(index: str) -> dict:
    """Calibrated fields for one index (empty ⇒ use defaults)."""
    return (calib().get("indices", {}) or {}).get(index, {}) or {}


# --------------------------------------------------------------------------
# Commodity trade-eligibility gate (v9.7.1) — the single source of truth
# --------------------------------------------------------------------------
def commodity_daily_calib(commodity: str) -> dict:
    """Track-A (daily-futures backfill) calibration for a commodity."""
    return (calib().get("commodities_daily", {}) or {}).get(commodity, {}) or {}


def commodity_trade_eligible(commodity: str) -> tuple[bool, str]:
    """(eligible, reason). A commodity may be traded ONLY when ALL hold:
      1. it is EXPLICITLY listed in config.COMMODITY_TRADABLE (operator intent),
      2. Track-A daily calibration exists (historical futures backfilled),
      3. Track-B intraday calibration exists AND intraday_calibrated is True
         (the live vault is deep enough to know its microstructure).
    This is the gate that keeps a fully-built engine from trading a commodity
    blind. Equity is unaffected — this function is commodity-only."""
    tradable = list(_cfg("COMMODITY_TRADABLE", []) or [])
    if commodity not in tradable:
        return False, (f"{commodity} not in COMMODITY_TRADABLE "
                       f"(operator has not enabled it)")
    da = commodity_daily_calib(commodity)
    if not da or "atr_proxy_daily" not in da:
        return False, (f"{commodity} has no Track-A daily calibration — run "
                       f"tools/commodity_backfill.py")
    ic = index_calib(commodity)
    if not ic.get("intraday_calibrated"):
        n = ic.get("n_ticks", 0)
        return False, (f"{commodity} has no Track-B intraday calibration "
                       f"(n_ticks={n}) — harvest more sessions, then run "
                       f"tools/commodity_calibration.py")
    return True, (f"{commodity} eligible: Track-A + Track-B calibrated and "
                  f"operator-enabled")


# --------------------------------------------------------------------------
# Calibrated toxicity thresholds (fall back to config)
# --------------------------------------------------------------------------
def tox_thresholds(index: str) -> tuple[float, float]:
    """(TOX_HIGH, TOX_BLOCK) — vault percentiles if calibrated, else config."""
    ic = index_calib(index)
    high = ic.get("tox_high", _cfg("TOX_HIGH", 0.40))
    block = ic.get("tox_block", _cfg("TOX_BLOCK", 0.55))
    # guard: block must be ≥ high
    return float(high), float(max(block, high))


def bucket_volume(index: str) -> float:
    ic = index_calib(index)
    return float(ic.get("bucket_volume", _cfg("TOX_BUCKET_VOLUME", 5000.0)))


# --------------------------------------------------------------------------
# DYNAMIC stop / target — volatility-scaled, from the vault
# --------------------------------------------------------------------------
@dataclass
class Levels:
    sl_pct: float          # stop as a fraction of entry premium
    tp_pct: float          # target as a fraction of entry premium
    source: str


def dynamic_stop_target(index: str, *, entry_premium: float, delta: float,
                        minutes_to_close: float,
                        atm_iv: float | None) -> Levels:
    """Compute the INITIAL stop/target premium fractions from the instrument's
    realized volatility (ATR proxy) mapped through the option delta, bounded by
    the config floor/ceiling. Falls back to BASE_SL_PCT / BASE_TP_PCT when the
    vault hasn't calibrated this index.

    The spot's expected move over the trade's natural horizon (from ATR proxy)
    × delta = the premium move we should risk to / target. Expressed as a % of
    entry so the existing sizing/manage code consumes it unchanged.
    """
    base_sl = float(_cfg("BASE_SL_PCT", 0.20))
    base_tp = float(_cfg("BASE_TP_PCT", 0.30))
    ic = index_calib(index)
    atr = ic.get("atr_proxy")
    if not atr or entry_premium <= 0 or delta <= 0:
        return Levels(base_sl, base_tp, "config default (no calib/inputs)")
    # ATR proxy is a ~1-minute absolute spot move; scale to the trade's horizon
    # (bounded so a long horizon doesn't explode the stop).
    horizon_min = max(min(minutes_to_close,
                          float(_cfg("DYN_LEVEL_HORIZON_CAP_MIN", 15.0))), 1.0)
    spot_move = float(atr) * (horizon_min ** 0.5)      # √t vol scaling
    prem_move = delta * spot_move
    sl_pct = prem_move / entry_premium
    # target as a risk-multiple of the stop (asymmetric payoff), vol-scaled
    rr = float(_cfg("DYN_LEVEL_RR", 1.6))
    tp_pct = sl_pct * rr
    # bound within sane rails so calibration can't produce absurd levels
    sl_pct = max(min(sl_pct, float(_cfg("DYN_SL_MAX", 0.45))),
                 float(_cfg("DYN_SL_MIN", 0.12)))
    tp_pct = max(min(tp_pct, float(_cfg("DYN_TP_MAX", 1.20))),
                 float(_cfg("DYN_TP_MIN", 0.18)))
    return Levels(round(sl_pct, 3), round(tp_pct, 3),
                  f"vol-scaled (ATR {atr}, {horizon_min:.0f}m, δ{delta:.2f})")


# ----------------------------------------------------------------- self-test
if __name__ == "__main__":
    # inject a calibration artifact
    _CAL_CACHE.update({"mtime": "test", "data": {
        "indices": {
            "SENSEX": {"atr_proxy": 35.0, "tox_high": 0.42, "tox_block": 0.61,
                       "bucket_volume": 8000.0, "n_ticks": 50000},
            "NIFTY": {"n_ticks": 500},      # thin → defaults
        }}})
    print("=== calibrated toxicity thresholds ===")
    print("  SENSEX:", tox_thresholds("SENSEX"), "bucket", bucket_volume("SENSEX"))
    print("  NIFTY (thin→default):", tox_thresholds("NIFTY"),
          "bucket", bucket_volume("NIFTY"))
    print("\n=== dynamic stop/target ===")
    for idx, prem, dlt, mins, iv in [
            ("SENSEX", 60.0, 0.45, 10.0, 0.13),
            ("SENSEX", 60.0, 0.45, 3.0, 0.13),   # less time → tighter
            ("NIFTY", 50.0, 0.40, 10.0, 0.11)]:  # no calib → default
        lv = dynamic_stop_target(idx, entry_premium=prem, delta=dlt,
                                 minutes_to_close=mins, atm_iv=iv)
        print(f"  {idx} prem₹{prem} δ{dlt} {mins}m → SL {lv.sl_pct:.0%} "
              f"TP {lv.tp_pct:.0%} [{lv.source}]")
    print("\n  (SENSEX stop breathes with ATR & horizon; NIFTY falls back "
          "to config until its vault sample is deep enough)")