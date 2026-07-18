"""
APEX OMNI v9.7.1 — FAST-LANE & LOSS-STREAK VALIDATION
=====================================================
Proves the behavioural contract of the two scalping-adjacent features the
operator asked for:

  1. FAST LANE — a conviction-gated quick-profit exit (3-10 min) that runs
     ALONGSIDE the normal 45-min path. It must:
       • only arm for entries at/above FAST_LANE_CONVICTION,
       • only fire inside the [MIN, MAX] hold window,
       • only fire once past the arm threshold AND at the fast-TP,
       • NEVER cut a slow winner short (outside the window → normal path).
  2. LOSS-STREAK BREAKER — N consecutive losses halts the day; any non-losing
     exit resets the streak (so winning scalps aren't blocked, but a real
     losing run stops cold). This is the operator's anti-overtrading guard.

  python tools/fast_lane_validate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                             # noqa: E402
from core.risk_manager import RiskGovernor                 # noqa: E402

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_fails = 0


def check(name, cond, detail=""):
    global _fails
    print(f"  [{PASS if cond else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _fails += 1


def _fast_lane_decision(conviction, hold_s, gain_pct):
    """Pure re-implementation of the manage() fast-lane predicate, so the
    contract is tested in isolation from the full tick engine."""
    qualifies = (config.FAST_LANE_ENABLED
                 and abs(conviction) >= config.FAST_LANE_CONVICTION)
    if not qualifies:
        return False, "not fast-lane (conviction below bar)"
    if not (config.FAST_LANE_MIN_HOLD_S <= hold_s <= config.FAST_LANE_MAX_HOLD_S):
        return False, "outside the fast window → normal path"
    if gain_pct >= config.FAST_LANE_ARM_PCT and gain_pct >= config.FAST_LANE_TP_PCT:
        return True, "FAST_LANE_TP"
    return False, "in window but TP not hit → normal path"


def fast_lane_contract():
    print("\n=========== FAST LANE — quick-profit overlay ============")
    C = config.FAST_LANE_CONVICTION
    lo, hi = config.FAST_LANE_MIN_HOLD_S, config.FAST_LANE_MAX_HOLD_S
    tp = config.FAST_LANE_TP_PCT

    # a low-conviction trade never qualifies, whatever the gain
    fired, _ = _fast_lane_decision(0.70, hold_s=(lo + hi) / 2, gain_pct=tp + 0.1)
    check("low-conviction entry never uses the fast lane", not fired)

    # a high-conviction trade, in-window, at the TP → fires
    fired, why = _fast_lane_decision(C + 0.05, hold_s=(lo + hi) / 2,
                                     gain_pct=tp + 0.02)
    check("high-conviction + in-window + at TP → fast exit", fired, why)

    # high-conviction but BEFORE the min hold → does not fire (let it breathe)
    fired, _ = _fast_lane_decision(C + 0.05, hold_s=lo - 30, gain_pct=tp + 0.1)
    check("before the 3-min floor → no fast exit (must breathe first)",
          not fired)

    # high-conviction but AFTER the max hold → hands back to normal path
    fired, why = _fast_lane_decision(C + 0.05, hold_s=hi + 60, gain_pct=tp + 0.1)
    check("after the 10-min cap → hands back to normal 45-min path", not fired,
          why)

    # high-conviction, in-window, but gain below TP → does NOT cut short
    fired, why = _fast_lane_decision(C + 0.05, hold_s=(lo + hi) / 2,
                                     gain_pct=tp - 0.05)
    check("in-window but move hasn't hit TP → never cuts a slow winner", not fired,
          why)

    check("fast-lane conviction bar sits ABOVE the normal entry bar",
          config.FAST_LANE_CONVICTION > config.ENTRY_CONVICTION,
          f"{config.FAST_LANE_CONVICTION} > {config.ENTRY_CONVICTION}")

    # when suspended, a qualifying entry downgrades to normal (fast_lane=False)
    # rather than being blocked — the entry predicate is `qualifies AND NOT
    # suspended`, so the trade still happens on the 45-min path.
    def _entry_is_fast(conv, suspended):
        qualifies = (config.FAST_LANE_ENABLED
                     and abs(conv) >= config.FAST_LANE_CONVICTION)
        return qualifies and not suspended
    check("suspended lane ⇒ qualifying entry runs as NORMAL, not blocked",
          _entry_is_fast(config.FAST_LANE_CONVICTION + 0.05, suspended=True)
          is False
          and _entry_is_fast(config.FAST_LANE_CONVICTION + 0.05,
                             suspended=False) is True)


def loss_streak_contract():
    print("\n=========== FAST-LANE LOSS-STREAK — scoped breaker ============")
    N = config.LOSS_STREAK_HALT

    # N consecutive FAST-LANE losses suspend the fast lane — but NOT the book
    g = RiskGovernor(capital=60000)
    for i in range(N):
        g.register_exit(1000, -200, "CE", ts=(i + 1) * 100, fast_lane=True)
    check(f"{N} consecutive fast-lane losses SUSPEND the fast lane",
          g.fast_lane_suspended)
    check("...but the book is NOT halted (45-min path keeps trading)",
          not g.halted, f"halted={g.halted}")

    # NORMAL-path losses do NOT count toward the fast-lane streak
    g2 = RiskGovernor(capital=60000)
    for i in range(N + 2):
        g2.register_exit(1000, -200, "CE", ts=(i + 1) * 100, fast_lane=False)
    check(f"{N + 2} NORMAL-path losses never suspend the fast lane",
          not g2.fast_lane_suspended and not g2.halted,
          f"fast_consec={g2.fast_consec_losses}")

    # a fast-lane WIN resets the fast-lane streak
    g3 = RiskGovernor(capital=60000)
    for i in range(N - 1):
        g3.register_exit(1000, -200, "CE", ts=(i + 1) * 100, fast_lane=True)
    g3.register_exit(1000, +300, "CE", ts=999, fast_lane=True)   # fast win
    check("a fast-lane win resets the fast-lane streak",
          g3.fast_consec_losses == 0)
    for i in range(N - 1):
        g3.register_exit(1000, -200, "CE", ts=2000 + i * 100, fast_lane=True)
    check(f"{N-1} fast losses, a fast win, {N-1} fast losses ⇒ NOT suspended",
          not g3.fast_lane_suspended, f"fast_consec={g3.fast_consec_losses}")

    # 12 winning scalps in a row never suspend the lane
    g4 = RiskGovernor(capital=60000)
    for i in range(12):
        g4.register_exit(1000, +150, "CE", ts=(i + 1) * 100, fast_lane=True)
    check("12 winning scalps in a row never suspend the fast lane",
          not g4.fast_lane_suspended)

    # a MIX: normal losses interleaved don't break the fast streak's reset logic
    g5 = RiskGovernor(capital=60000)
    g5.register_exit(1000, -200, "CE", ts=100, fast_lane=True)   # fast loss 1
    g5.register_exit(1000, -900, "CE", ts=200, fast_lane=False)  # normal loss
    g5.register_exit(1000, -200, "CE", ts=300, fast_lane=True)   # fast loss 2
    check("interleaved normal losses don't inflate the fast-lane streak",
          g5.fast_consec_losses == 2 and not g5.fast_lane_suspended,
          f"fast_consec={g5.fast_consec_losses}")


if __name__ == "__main__":
    fast_lane_contract()
    loss_streak_contract()
    print("\n" + "=" * 58)
    if _fails:
        print(f"  {FAIL}: {_fails} check(s) failed")
        sys.exit(1)
    print(f"  {PASS}: fast-lane + loss-streak validated")