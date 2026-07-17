"""
APEX OMNI v9.7.1 — TOXICITY & DYNAMIC-LEVELS VALIDATION
=======================================================
Proves the behavioural contract of the trap filter (core/order_flow) and the
dynamic stop/target (core/calibration) on controlled scenarios. This is the
offline twin of tools/toxicity_report.py (which measures real lift on the
vault). Run any time:

  python tools/toxicity_validate.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                             # noqa: E402
from core import order_flow as OF                          # noqa: E402
from core import calibration as CAL                        # noqa: E402

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_fails = 0


def check(name, cond, detail=""):
    global _fails
    print(f"  [{PASS if cond else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _fails += 1


def _run(kind, seed=1):
    """Drive the toxicity engine through a scenario, return the final verdict."""
    eng = OF.OrderFlowToxicity("SENSEX")
    rng = random.Random(seed)
    spot = 77000.0
    v = OF.TrapVerdict()
    for t in range(600):
        if kind == "genuine_down":
            spot -= 0.5 + rng.gauss(0, 0.3)
            bid, ask, bq, aq, vol = spot - 1, spot + 1, 300, 900, 200
        elif kind == "genuine_up":
            spot += 0.5 + rng.gauss(0, 0.3)
            bid, ask, bq, aq, vol = spot - 1, spot + 1, 900, 300, 200
        elif kind == "chop":
            spot += rng.gauss(0, 0.4)
            bid, ask, bq, aq, vol = spot - 1, spot + 1, 500, 500, 150
        v = eng.update(spot=spot, bid=bid, bid_qty=bq, ask=ask, ask_qty=aq,
                       vol_delta=vol)
    return v


def toxicity_contract():
    print("\n=========== TOXICITY — trap gate contract ============")
    vd = _run("genuine_down")
    vu = _run("genuine_up")
    vc = _run("chop")
    check("one-sided sell flow reads high toxicity", vd.toxicity > 0.4,
          f"tox {vd.toxicity}")
    check("toxic sell flow points DOWN (−)", vd.tox_dir < 0)
    check("toxic buy flow points UP (+)", vu.tox_dir > 0)
    check("balanced chop reads LOW toxicity", vc.toxicity < vd.toxicity,
          f"chop {vc.toxicity} < down {vd.toxicity}")

    # entry gate: buying WITH the flow is allowed, AGAINST is blocked
    a_with, _ = OF.entry_trap_check(vd, "PE", tox_block=0.55, sweep_fade_ok=True)
    a_against, w_ag = OF.entry_trap_check(vd, "CE", tox_block=0.55,
                                          sweep_fade_ok=True)
    check("PE WITH toxic down-flow is ALLOWED", a_with)
    check("CE AGAINST toxic down-flow is BLOCKED", not a_against, w_ag)
    # a genuine up-flow: buying CE allowed, PE blocked
    a_ce, _ = OF.entry_trap_check(vu, "CE", tox_block=0.55, sweep_fade_ok=True)
    a_pe, _ = OF.entry_trap_check(vu, "PE", tox_block=0.55, sweep_fade_ok=True)
    check("CE WITH toxic up-flow allowed, PE blocked", a_ce and not a_pe)


def sweep_contract():
    print("\n=========== SWEEP — SFP trap detection ============")
    # engineer a clean pierce-above-then-reclaim (swept highs → bearish)
    eng = OF.OrderFlowToxicity("SENSEX")
    spot = 77000.0
    swept = False
    sweep_dir = ""
    for t in range(400):
        if t < 60:
            spot = 77000.0 + (t % 5)          # establish a swing high ~77004
        elif 60 <= t < 70:
            spot = 77000.0 + 40               # pierce well above (>buffer)
        elif 70 <= t < 90:
            spot = 77000.0 - 5                # reclaim back below the high
        else:
            spot = 77000.0
        v = eng.update(spot=spot, bid=spot - 1, bid_qty=800, ask=spot + 1,
                       ask_qty=800, vol_delta=200)
        if v.sweep:
            swept, sweep_dir = True, v.sweep_dir
    check("a genuine pierce+reclaim is detected as a sweep", swept,
          f"dir {sweep_dir}")
    check("swept-highs is labelled 'up' (bearish trap)", sweep_dir == "up")

    # a sweep against the trade blocks the chase; in favour + absorption allows
    tv = OF.TrapVerdict(sweep=True, sweep_dir="up", absorption_z=2.5,
                        toxicity=0.3)
    a_ce, w_ce = OF.entry_trap_check(tv, "CE", tox_block=0.55,
                                     sweep_fade_ok=True)
    a_pe, w_pe = OF.entry_trap_check(tv, "PE", tox_block=0.55,
                                     sweep_fade_ok=True)
    check("CE into a fresh up-sweep is BLOCKED (engineered top)", not a_ce, w_ce)
    check("PE with up-sweep + absorption is HIGH-QUALITY", a_pe,
          w_pe)


def dynamic_levels_contract():
    print("\n=========== DYNAMIC LEVELS — vol-scaled stop/target ============")
    CAL._CAL_CACHE.update({"mtime": "test", "data": {"indices": {
        "SENSEX": {"atr_proxy": 20.0, "n_ticks": 50000},
        "NIFTY": {"n_ticks": 100}}}})   # NIFTY thin → default
    lv_far = CAL.dynamic_stop_target("SENSEX", entry_premium=60.0, delta=0.45,
                                     minutes_to_close=12.0, atm_iv=0.13)
    lv_near = CAL.dynamic_stop_target("SENSEX", entry_premium=60.0, delta=0.45,
                                      minutes_to_close=3.0, atm_iv=0.13)
    lv_def = CAL.dynamic_stop_target("NIFTY", entry_premium=50.0, delta=0.40,
                                     minutes_to_close=12.0, atm_iv=0.11)
    check("more time-to-close ⇒ wider stop (√t scaling)",
          lv_far.sl_pct >= lv_near.sl_pct,
          f"{lv_far.sl_pct:.0%} @12m ≥ {lv_near.sl_pct:.0%} @3m")
    check("target is a risk-multiple of the stop",
          lv_far.tp_pct > lv_far.sl_pct,
          f"TP {lv_far.tp_pct:.0%} > SL {lv_far.sl_pct:.0%}")
    check("stop respects the rails (never absurd)",
          config.DYN_SL_MIN <= lv_far.sl_pct <= config.DYN_SL_MAX)
    check("uncalibrated index falls back to config default",
          lv_def.sl_pct == config.BASE_SL_PCT and "default" in lv_def.source,
          lv_def.source)


if __name__ == "__main__":
    random.seed(20260716)
    toxicity_contract()
    sweep_contract()
    dynamic_levels_contract()
    print("\n" + "=" * 58)
    if _fails:
        print(f"  {FAIL}: {_fails} check(s) failed")
        sys.exit(1)
    print(f"  {PASS}: toxicity + dynamic-levels validated")