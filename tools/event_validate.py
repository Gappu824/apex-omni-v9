"""
APEX OMNI v9.7.1 — EVENT ENGINE & GEMMA ANALYST VALIDATION
==========================================================
Proves the commodity scheduled-event guard's calendar math (the honest "news"
layer) and the Gemma analyst's FAIL-SAFE contract. Deterministic; no Ollama and
no live Kite required.

  python tools/event_validate.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                             # noqa: E402
from core.event_engine import (CommodityEventEngine, MarketEvent,  # noqa: E402
                               event_entry_gate)

_IST = ZoneInfo("Asia/Kolkata")
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_fails = 0


def check(name, cond, detail=""):
    global _fails
    print(f"  [{PASS if cond else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _fails += 1


def ist(y, m, d, hh, mm):
    return dt.datetime(y, m, d, hh, mm, tzinfo=_IST)


def event_contract():
    print("\n=========== EVENT ENGINE — EIA calendar (DST-correct) ============")
    eng = CommodityEventEngine(overrides=[])

    # SUMMER (EDT): 10:30 ET = 20:00 IST. 2026-07-22 = Wednesday (EIA petroleum)
    clear_before = eng.evaluate(ist(2026, 7, 22, 18, 0), "CRUDEOIL")
    blackout = eng.evaluate(ist(2026, 7, 22, 19, 45), "CRUDEOIL")
    settle = eng.evaluate(ist(2026, 7, 22, 20, 10), "CRUDEOIL")
    clear_after = eng.evaluate(ist(2026, 7, 22, 21, 30), "CRUDEOIL")
    check("well before release → clear", not clear_before.in_blackout
          and not clear_before.in_settle)
    check("15 min before → BLACKOUT", blackout.in_blackout,
          f"mins_to_event={blackout.minutes_to_event}")
    check("10 min after → SETTLE", settle.in_settle)
    check("well after → clear", not clear_after.in_blackout
          and not clear_after.in_settle)

    # entry gate follows the verdict
    a1, _ = event_entry_gate(blackout)
    a2, _ = event_entry_gate(settle)
    a3, _ = event_entry_gate(clear_after)
    check("gate BLOCKS in blackout", not a1)
    check("gate BLOCKS in extreme-severity settle", not a2)
    check("gate ALLOWS when clear", a3)

    # WINTER (EST): 10:30 ET = 21:00 IST. 2026-01-21 = Wednesday
    w_before = eng.evaluate(ist(2026, 1, 21, 20, 45), "CRUDEOIL")   # 15 min before 21:00
    check("winter DST: blackout starts at 20:45 (event 21:00), not 19:45",
          w_before.in_blackout and abs(w_before.minutes_to_event - 15) < 0.5,
          f"mins_to_event={w_before.minutes_to_event}")
    w_summer_wrong = eng.evaluate(ist(2026, 1, 21, 19, 45), "CRUDEOIL")
    check("winter: 19:45 is NOT yet blackout (would be wrong if hardcoded EDT)",
          not w_summer_wrong.in_blackout)


def commodity_isolation():
    print("\n=========== EVENT ENGINE — commodity isolation ============")
    eng = CommodityEventEngine(overrides=[])
    # Thursday NatGas storage blackout must not touch Crude
    now = ist(2026, 7, 23, 19, 45)     # Thu, 15 min before NG release
    ng = eng.evaluate(now, "NATURALGAS")
    crude = eng.evaluate(now, "CRUDEOIL")
    check("NatGas in blackout Thursday", ng.in_blackout
          and ng.event == "EIA_NATGAS_STORAGE")
    check("Crude UNAFFECTED by NatGas event", not crude.in_blackout)
    # Gold has no weekly EIA event
    gold = eng.evaluate(now, "GOLD")
    check("Gold unaffected by EIA events", not gold.in_blackout)


def explicit_date_events():
    print("\n=========== EVENT ENGINE — dated one-offs (OPEC/FOMC) ============")
    opec = MarketEvent("OPEC_MEETING", dt.time(6, 0), ("CRUDEOIL",),
                       "extreme", explicit_date=dt.date(2026, 8, 3))
    eng = CommodityEventEngine(overrides=[opec])
    nxt = eng.next_event(ist(2026, 7, 30, 12, 0), "CRUDEOIL")
    check("dated OPEC event is found as next event", nxt is not None
          and nxt[0] == "OPEC_MEETING")
    # 6:00 ET on Aug 3 = 15:30 IST; blackout 15:10-15:30
    v = eng.evaluate(ist(2026, 8, 3, 15, 15), "CRUDEOIL")
    check("blackout active before dated OPEC release", v.in_blackout)


def gemma_failsafe():
    print("\n=========== GEMMA ANALYST — fail-safe contract ============")
    import tools.gemma_analyst as G
    # context gathering must never raise, even with no reports present
    try:
        ctx = G.gather_context()
        ok = isinstance(ctx, dict) and "upcoming_events" in ctx
    except Exception as e:                                     # noqa: BLE001
        ok = False
        print("    gather_context raised:", e)
    check("gather_context never raises, returns structured dict", ok)

    # prompt build must never raise
    try:
        p = G.build_prompt(ctx)
        ok2 = isinstance(p, str) and "SCHEDULED-EVENT RISK" in p
    except Exception:                                          # noqa: BLE001
        ok2 = False
    check("build_prompt produces a scoped prompt", ok2)

    # Ollama call to a dead host must return None, not raise
    res = G._ollama_generate("test", model="x", host="http://127.0.0.1:1",
                             num_ctx=256, timeout=2)
    check("Ollama unavailable → returns None (system proceeds)", res is None)


if __name__ == "__main__":
    event_contract()
    commodity_isolation()
    explicit_date_events()
    gemma_failsafe()
    print("\n" + "=" * 58)
    if _fails:
        print(f"  {FAIL}: {_fails} check(s) failed")
        sys.exit(1)
    print(f"  {PASS}: event engine + Gemma fail-safe validated")