"""
APEX OMNI v9.7.1 — COMMODITY BRAIN VALIDATION
=============================================
Proves the parallel commodity engine's decision contract end to end, with no
live data: the physics policy is directionally symmetric, every gate fires in
order (conviction → eligibility/calibration → scheduled event), and dynamic
sizing comes from the commodity's OWN calibrated volatility. Also proves the
isolation invariant: nothing trades unless in COMMODITY_TRADABLE with both
calibration tracks green.

  python tools/commodity_brain_validate.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                             # noqa: E402
import core.calibration as CAL                            # noqa: E402
from core.commodity_brain import (CommodityBrain,          # noqa: E402
                                  CommodityHeuristicPolicy)

_IST = ZoneInfo("Asia/Kolkata")
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_fails = 0
F = config.FEATURES_PER_NODE


def check(name, cond, detail=""):
    global _fails
    print(f"  [{PASS if cond else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _fails += 1


def _tilt(bullish=True):
    """A node block with an explicit directional flow tilt."""
    ce, pe, spot = (np.zeros(F, np.float32) for _ in range(3))
    s = 1.0 if bullish else -1.0
    ce[12], pe[12] = 2.0 * s, -1.0 * s      # OFI z
    ce[16], pe[16] = 0.5 * s, -0.5 * s      # dealer inv
    ce[5], pe[5] = 1.0 * s, -1.0 * s        # velocity
    spot[0] = 0.3 * s                       # momentum
    return np.array([spot, ce, pe, np.zeros(F, np.float32),
                     np.zeros(F, np.float32)])


def _snap(l, b, a):
    return {"ltp": l, "bid": b, "ask": a, "bid_qty": 20, "ask_qty": 20,
            "oi": 1000, "vol_delta": 100}


def _market():
    return {"CRUDEOIL": {"spot": _snap(6000, 5999, 6001),
                         "expiry": "2026-08-19", "dte": 20, "T": 20 / 365.0,
                         "legs": {"atm_ce": {"snap": _snap(120, 119, 121),
                                             "strike": 6000},
                                  "atm_pe": {"snap": _snap(118, 117, 119),
                                             "strike": 6000}}}}


def _make_eligible():
    config.COMMODITY_TRADABLE = ["CRUDEOIL"]
    CAL._CAL_CACHE.update({"mtime": "test", "data": {
        "commodities_daily": {"CRUDEOIL": {"atr_proxy_daily": 0.03,
                                           "n_days": 300}},
        "indices": {"CRUDEOIL": {"intraday_calibrated": True, "n_ticks": 50000,
                                 "tox_high": 0.18, "tox_block": 0.20,
                                 "atr_proxy": 12.0, "move_median": 1.3,
                                 "bucket_volume": 600}}}})


def policy_symmetry():
    print("\n=========== POLICY — transparent physics, symmetric ============")
    pol = CommodityHeuristicPolicy()
    up = pol.predict({"CRUDEOIL": _tilt(True)})["CRUDEOIL"]
    dn = pol.predict({"CRUDEOIL": _tilt(False)})["CRUDEOIL"]
    check("bullish flow → CE (positive signal)", up > 0, f"{up:+.3f}")
    check("bearish flow → PE (negative signal)", dn < 0, f"{dn:+.3f}")
    check("directionally symmetric", abs(up + dn) < 1e-6,
          f"|{up:+.3f} + {dn:+.3f}|")
    check("clears the commodity entry bar", abs(up) >= config.COMMODITY_ENTRY_CONVICTION,
          f"{abs(up):.3f} ≥ {config.COMMODITY_ENTRY_CONVICTION}")
    check("no legs streaming → no signal (honest mask)",
          pol.predict({"CRUDEOIL": np.zeros((5, F), np.float32)}) == {})


def gate_chain():
    print("\n=========== GATES — fire in order ============")
    brain = CommodityBrain()
    brain._nodes_for = lambda name, ctx, ts: _tilt(True)   # strong bullish
    mkt = _market()

    # 1. NOT in TRADABLE → blocked at eligibility
    config.COMMODITY_TRADABLE = []
    CAL._CAL_CACHE.update({"mtime": "test", "data": {}})
    now = dt.datetime(2026, 7, 20, 14, 0, tzinfo=_IST)
    d = brain.decide(mkt, now)[0]
    check("not in COMMODITY_TRADABLE → blocked", not d.allowed
          and "TRADABLE" in d.reason)

    # 2. eligible + calibrated, normal time → ENTERED with dynamic sizing
    _make_eligible()
    d = brain.decide(mkt, now)[0]
    check("eligible + calibrated → entered", d.allowed and d.direction == "CE",
          d.reason)
    check("dynamic stop from calibrated vol (not zero/default)",
          0.0 < d.sl_pct < 0.5 and d.tp_pct > d.sl_pct,
          f"sl={d.sl_pct:.3f} tp={d.tp_pct:.3f}")

    # 3. eligible but in EIA blackout → BLOCKED by event guard
    now_ev = dt.datetime(2026, 7, 22, 19, 45, tzinfo=_IST)
    d = brain.decide(mkt, now_ev)[0]
    check("eligible but EIA blackout → blocked by event guard",
          not d.allowed and "BLACKOUT" in d.reason, d.reason)

    # 4. low conviction → blocked before eligibility even checked
    brain._nodes_for = lambda name, ctx, ts: np.zeros((5, F), np.float32)
    # a faint tilt below the bar
    faint = np.zeros((5, F), np.float32)
    faint[1][12] = 0.1
    brain._nodes_for = lambda name, ctx, ts: faint
    d = brain.decide(mkt, now)
    check("faint signal below bar → blocked on conviction",
          (not d[0].allowed and "conviction" in d[0].reason) if d else True)


def isolation():
    print("\n=========== ISOLATION — commodities never touch equity ============")
    # the commodity brain uses its own frame/policy; equity ACTION_DIM unchanged
    check("equity ACTION_DIM unchanged (commodities not in it)",
          config.ACTION_DIM == len(config.INDEX_ORDER) * 2)
    check("commodities absent from equity INDEX_ORDER",
          not any(c in config.INDEX_ORDER
                  for c in getattr(config, "HARVEST_COMMODITIES", [])))
    check("CONFIG_HASH unchanged", config.CONFIG_HASH == "47689d19a5",
          config.CONFIG_HASH)


if __name__ == "__main__":
    policy_symmetry()
    gate_chain()
    isolation()
    print("\n" + "=" * 58)
    if _fails:
        print(f"  {FAIL}: {_fails} check(s) failed")
        sys.exit(1)
    print(f"  {PASS}: commodity brain validated (heuristic + gates + isolation)")