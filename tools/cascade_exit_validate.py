"""
APEX OMNI v9.7.1 — CASCADE-EXIT VALIDATION
===========================================
Asserts the two 2026-07-16 fixes on the real scenario:
  A. cascade-aware stop width survives the short-gamma retest wick that the
     fixed base stop is picked off by;
  B. the smart lockout blocks REVENGE (same/weaker re-trigger) but ALLOWS a
     STRONGER, still-aligned cascade re-trigger (trend continuation) — the
     jackpot the blunt lockout muted.

  python tools/cascade_exit_validate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                             # noqa: E402
from core.cascade_exit import cascade_stop_mult, SmartLockout   # noqa: E402

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_fails = 0


def check(name, cond, detail=""):
    global _fails
    print(f"  [{PASS if cond else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _fails += 1


def part_a():
    print("\n=== A: cascade stop width vs the retest wick ===")
    entry = 60.0
    base_sl = config.BASE_SL_PCT
    # deep short-gamma (the 11:28 trigger conditions)
    m, why = cascade_stop_mult(net_gex=-25.1e12, cascade_z=-2.04,
                               is_cascade=True)
    base_stop = entry * (1 - base_sl)
    wide_stop = entry * (1 - base_sl * m)
    wick = entry * (1 - 0.30)      # a 30% premium spike on the up-retest
    check("cascade stop is wider than base", wide_stop < base_stop,
          f"base ₹{base_stop:.1f} vs cascade ₹{wide_stop:.1f} (×{m:.2f})")
    check("base stop would be whipsawed by the wick", wick <= base_stop,
          f"wick ₹{wick:.1f} ≤ base ₹{base_stop:.1f}")
    check("cascade stop SURVIVES the wick", wick > wide_stop,
          f"wick ₹{wick:.1f} > cascade ₹{wide_stop:.1f}")
    # non-cascade and long-gamma must NOT widen
    m0, _ = cascade_stop_mult(net_gex=-25e12, cascade_z=-2.0, is_cascade=False)
    m1, _ = cascade_stop_mult(net_gex=2e12, cascade_z=-2.0, is_cascade=True)
    check("non-cascade entry keeps base stop", m0 == 1.0)
    check("long-gamma cascade keeps base stop", m1 == 1.0)
    # ceiling holds
    m2, _ = cascade_stop_mult(net_gex=-99e12, cascade_z=-5.0, is_cascade=True)
    check("stop multiplier is bounded", m2 <= config.CASCADE_STOP_MULT_MAX,
          f"×{m2} ≤ ×{config.CASCADE_STOP_MULT_MAX}")


def part_b():
    print("\n=== B: smart lockout on the 2026-07-16 sequence ===")
    lk = SmartLockout()
    lk.note_loss("PE", -2.04)                     # trade 1 lost
    v2 = lk.evaluate(ts=2000, direction="PE", is_cascade=True, cascade_z=-2.04,
                     spot=77400, flip=77507, net_gex=-20e12)
    check("same-strength re-trigger is NOT bypassed (revenge)", not v2.bypass,
          v2.reason)
    lk.note_loss("PE", -2.04)
    v3 = lk.evaluate(ts=4900, direction="PE", is_cascade=True, cascade_z=-2.59,
                     spot=77376, flip=77478, net_gex=-25.5e12)
    check("STRONGER aligned re-trigger IS bypassed (the jackpot)", v3.bypass,
          v3.reason)
    lk.register_bypass(4900)
    # misaligned (spot back above flip) must stay blocked
    v4 = lk.evaluate(ts=6000, direction="PE", is_cascade=True, cascade_z=-2.9,
                     spot=77600, flip=77478, net_gex=-25e12)
    check("misaligned strong re-trigger stays blocked", not v4.bypass,
          v4.reason)
    # non-cascade discretionary retry stays blocked
    v5 = lk.evaluate(ts=7000, direction="PE", is_cascade=False, cascade_z=None,
                     spot=77350, flip=77478, net_gex=-25e12)
    check("non-cascade retry stays blocked", not v5.bypass, v5.reason)
    # budget cap
    lk2 = SmartLockout()
    lk2.bypasses_today = config.LOCKOUT_BYPASS_MAX_PER_DAY
    lk2.note_loss("PE", -2.0)
    v6 = lk2.evaluate(ts=100, direction="PE", is_cascade=True, cascade_z=-3.0,
                      spot=77300, flip=77478, net_gex=-25e12)
    check("daily bypass budget is enforced", not v6.bypass, v6.reason)


if __name__ == "__main__":
    part_a()
    part_b()
    print("\n" + "=" * 60)
    if _fails:
        print(f"  {FAIL}: {_fails} check(s) failed")
        sys.exit(1)
    print(f"  {PASS}: cascade-exit fixes validated on the 2026-07-16 scenario")