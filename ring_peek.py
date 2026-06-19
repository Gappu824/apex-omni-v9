"""
ring_peek.py — READ-ONLY diagnostic. Attaches to the live BinaryRingBuffer that
data_harvester_v9.py writes and apex_main_v9.py reads, and dumps — per index —
whether the ATM option LEGS are actually populated.

Run it WHILE the harvester is running, from the project folder:
    python ring_peek.py

How to read the output:
  * "<index> spot=... legs=N" with N>0  -> the brain HAS option data for it.
  * "<index> ... NO LEGS"               -> spot is ticking but the ATM option
                                           tokens are subscribed yet NOT quoting,
                                           so _assemble_market drops the legs and
                                           the brain skips that index silently.

So if NIFTY shows spot but NO LEGS while SENSEX shows legs, the problem is the
NIFTY option feed (those specific contracts aren't ticking) — NOT the brain.
If NIFTY shows legs here but the brain still never evaluates it, the problem is
downstream in the brain's read, and we look there next.

This writes NOTHING. It is safe to run alongside the live harvester and brain.
"""
import time
import config
from apex_ipc_core import BinaryRingBuffer

ring = BinaryRingBuffer(writer=False)   # read-only handle on the shared ring


def dump_once():
    state, age = ring.read_state()
    if not state:
        print(f"[ring] no readable state yet (age {age:.1f}s) — "
              f"is data_harvester_v9.py running?")
        return
    market = state.get("market", {})
    present = sorted(k for k in market if not k.startswith("_"))
    print(f"=== ring age {age:.1f}s | indices present: {present} ===")
    vix = market.get("_VIX")
    if vix:
        print(f"  _VIX ltp={vix.get('ltp')}")
    for idx in config.INDICES:
        entry = market.get(idx)
        if entry is None:
            print(f"  {idx:<11} ABSENT (no spot snapshot in ring)")
            continue
        sp = entry.get("spot") or {}
        legs = entry.get("legs") or {}
        flag = "" if legs else "   <-- NO LEGS (ATM option tokens not quoting)"
        print(f"  {idx:<11} spot={sp.get('ltp')} atm={entry.get('atm')} "
              f"expiry={entry.get('expiry')} dte={entry.get('dte')} "
              f"legs={len(legs)}{flag}")
        for leg, info in legs.items():
            snap = info.get("snap") or {}
            print(f"       {leg:<6} K={info.get('strike')} "
                  f"tok={info.get('token')} sym={info.get('symbol')} "
                  f"bid={snap.get('bid')} ask={snap.get('ask')} "
                  f"ltp={snap.get('ltp')}")


if __name__ == "__main__":
    for i in range(5):
        dump_once()
        print()
        if i < 4:
            time.sleep(2)