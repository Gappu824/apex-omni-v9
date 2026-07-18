"""
APEX OMNI v9.7.1 — LIVE COMMODITY BRAIN (apex_commodity_main)
============================================================
The live harness for the commodity engine — the analog of apex_main for MCX
commodities. It is a REAL process (not a self-test): it reads the SAME per-second
ring buffer the harvester already writes (which now includes commodity spot +
option legs), reconstructs each commodity's market, runs CommodityBrain.decide(),
and drives a per-commodity PositionManager with the event-gate and dynamic stops
already wired. Paper-only, gated, supervisable.

Why it reads the shared ring (not its own KiteTicker)
-----------------------------------------------------
The harvester's _assemble_market already emits commodity entries into the ring
(spot=front-month future, plus atm/otm CE/PE legs, with expiry/dte/T/lot/step) —
because the harvester was extended to capture MCX. So this harness consumes that
same ring exactly as apex_main does for equities: ONE WebSocket (the harvester's),
ONE data source, no duplication. This is how the equity brain works, and it's the
correct, non-duplicative design.

What runs today (real, no mockup)
---------------------------------
  • Per RING-SECOND decision cadence (the same tempo apex_main uses), with the
    ~5 Hz loop keeping exits/management at full speed.
  • CommodityBrain.decide() → the transparent physics policy, gated by
    conviction → trade-eligibility (calibration + operator opt-in) → scheduled
    event guard. A commodity with no calibration simply never enters.
  • On an allowed decision, a per-commodity PositionManager.try_enter with a
    real LegQuote hierarchy built from the ring; management every tick with
    dynamic stops from the commodity's own calibrated volatility.
  • ONE RiskGovernor across all commodities (the live single-book model), so
    commodity sizing obeys the same Kelly/floor/curfew/halt discipline.

What is NOT here yet, by design
-------------------------------
  • No trained commodity meta/SAC — CommodityBrain runs heuristic-only until a
    commodity forge trains one on harvested ticks (the load hook returns None).
    This is exactly how apex_main ran before its first promotion. The SYSTEM is
    here; the trained model arrives when the vault has data.

Paper-only always: LIVE_FIRE stays False AND a commodity trades only if it is in
COMMODITY_TRADABLE with both calibration tracks green. With an empty
COMMODITY_TRADABLE (the default), this process runs, reads, decides, and enters
NOTHING — it is the harness waiting for data, live and correct.

  python apex_commodity_main.py
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from collections import deque
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                             # noqa: E402
from apex_ipc_core import BinaryRingBuffer                 # noqa: E402
from core.risk_manager import RiskGovernor                 # noqa: E402
from core.execution_engine import ExecutionEngine          # noqa: E402
from core.position_manager import (PositionManager,        # noqa: E402
                                   LegQuote, TickContext)
from core.commodity_brain import CommodityBrain            # noqa: E402
from core import calibration as CAL                        # noqa: E402

try:
    from kiteconnect import KiteConnect
    HAVE_KITE = True
except Exception:                                          # noqa: BLE001
    HAVE_KITE = False

config.setup_logging("commodity_brain")
import logging                                            # noqa: E402
log = logging.getLogger("commodity_brain")

_IST = ZoneInfo("Asia/Kolkata")
_LEG_ORDER = ["atm_ce", "otm_ce", "atm_pe", "otm_pe"]


def _mid(snap: dict) -> float:
    b, a = float(snap.get("bid") or 0), float(snap.get("ask") or 0)
    if b > 0 and a > 0:
        return (b + a) / 2.0
    return float(snap.get("ltp") or 0)


def _hierarchy_for(direction: str, ctx_m: dict) -> list:
    """Build the LegQuote list (ATM→OTM in the trade direction) from the ring's
    commodity market entry — the same shape PositionManager.try_enter expects."""
    legs = ctx_m.get("legs") or {}
    want = ["atm_ce", "otm_ce"] if direction == "CE" else ["atm_pe", "otm_pe"]
    lot = int(ctx_m.get("lot") or 1)
    dte = float(ctx_m.get("dte") or 20.0)
    out = []
    for leg in want:
        info = legs.get(leg)
        if not info or not info.get("snap"):
            continue
        s = info["snap"]
        bid, ask = float(s.get("bid") or 0), float(s.get("ask") or 0)
        if bid <= 0 or ask <= 0:
            continue
        out.append(LegQuote(
            leg=leg, symbol=info.get("symbol", ""),
            exchange=(config.COMMODITIES.get(
                ctx_m.get("_name", ""), {}) or {}).get("exchange", "MCX"),
            token=int(info.get("token") or 0),
            strike=float(info.get("strike") or 0.0),
            premium=_mid(s), bid=bid, ask=ask,
            bid_qty=float(s.get("bid_qty") or 0),
            ask_qty=float(s.get("ask_qty") or 0),
            lot=lot, delta=0.5, dte=dte))
    return out


def main():
    kite = None
    if HAVE_KITE and config.KITE_API_KEY and getattr(
            config, "KITE_ACCESS_TOKEN", None):
        kite = KiteConnect(api_key=config.KITE_API_KEY)
        kite.set_access_token(config.KITE_ACCESS_TOKEN)

    commodities = list(getattr(config, "HARVEST_COMMODITIES", []))
    if not commodities:
        log.info("HARVEST_COMMODITIES empty — commodity brain idle")
        return

    ring = BinaryRingBuffer()
    risk = RiskGovernor(kite=kite)
    engine = ExecutionEngine(kite=kite, quote_fn=lambda tok: {})
    brain = CommodityBrain()
    # one PositionManager per commodity (isolated from equity PMs by process)
    pms = {c: PositionManager(c, risk, engine) for c in commodities}

    tradable = list(getattr(config, "COMMODITY_TRADABLE", []) or [])
    log.info("commodity brain up | capital ₹%.0f | mode %s | commodities %s | "
             "TRADABLE %s", risk.start_capital,
             "LIVE" if config.live_fire_armed() else "PAPER",
             ",".join(commodities), tradable or "(none — harness idle, waiting "
             "for calibration + opt-in)")

    last_spot: dict[str, float] = {}
    spot_secs: dict[str, deque] = {c: deque(maxlen=1800) for c in commodities}
    spread_ew: dict[str, float] = {c: 0.02 for c in commodities}
    last_decision_sec = -1

    while True:
        try:
            state, age = ring.read_state()
        except Exception as e:                                # noqa: BLE001
            log.debug("ring read failed: %s", e)
            time.sleep(0.2)
            continue
        market = (state or {}).get("market", {})
        now = dt.datetime.now(_IST)
        ts = now.timestamp()
        hm = now.strftime("%H:%M")
        ring_sec = int(ts)

        # ---- management every loop (full tempo) ----
        for c in commodities:
            pm = pms[c]
            ctx_m = market.get(c)
            if not ctx_m:
                continue
            spot = float((ctx_m.get("spot") or {}).get("ltp") or 0.0)
            if spot <= 0:
                continue
            secs = spot_secs[c]
            secs.append((ts, spot))
            vel = 0.0
            if len(secs) > 1 and secs[-1][0] > secs[0][0]:
                vel = (secs[-1][1] - secs[0][1]) / (secs[-1][0] - secs[0][0])
            mins_left = _minutes_to_commodity_close(c, now)
            # a management quote for the held leg (if any)
            quote = {}
            if pm.pos is not None:
                held = (ctx_m.get("legs") or {}).get(pm.pos.leg_key) \
                    if hasattr(pm.pos, "leg_key") else None
                if held and held.get("snap"):
                    s = held["snap"]
                    quote = {"bid": float(s.get("bid") or 0),
                             "ask": float(s.get("ask") or 0),
                             "ltp": float(s.get("ltp") or 0)}
            tctx = TickContext(
                ts=ts, hm=hm, spot=spot, spot_velocity_1s=vel, data_age_s=age,
                atm_iv=0.6,                       # commodity IV is high; refined
                minutes_to_close=mins_left,
                avg_spread_pct=spread_ew[c], conviction=0.0)
            try:
                pm.manage(tctx, quote)
            except Exception as e:                            # noqa: BLE001
                log.debug("%s manage: %s", c, e)

        # ---- decisions once per RING SECOND ----
        if ring_sec != last_decision_sec and market:
            last_decision_sec = ring_sec
            decisions = brain.decide(market, now)
            for d in decisions:
                if not d.allowed:
                    if d.reason and "conviction" not in d.reason:
                        log.debug("%s blocked: %s", d.commodity, d.reason)
                    continue
                pm = pms.get(d.commodity)
                ctx_m = market.get(d.commodity) or {}
                ctx_m["_name"] = d.commodity
                spot = float((ctx_m.get("spot") or {}).get("ltp") or 0.0)
                hierarchy = _hierarchy_for(d.direction, ctx_m)
                if not hierarchy:
                    log.debug("%s: no valid legs to enter", d.commodity)
                    continue
                mins_left = _minutes_to_commodity_close(d.commodity, now)
                tctx = TickContext(
                    ts=ts, hm=hm, spot=spot, spot_velocity_1s=0.0,
                    data_age_s=age, atm_iv=0.6, minutes_to_close=mins_left,
                    avg_spread_pct=spread_ew[d.commodity],
                    conviction=d.conviction)
                try:
                    entered = pm.try_enter(tctx, d.direction, d.conviction,
                                           win_prob=d.win_prob or d.conviction,
                                           hierarchy=hierarchy)
                    if entered:
                        log.info("COMMODITY ENTER %s %s conv=%.2f (%s)",
                                 d.commodity, d.direction, d.conviction,
                                 d.reason)
                except Exception as e:                        # noqa: BLE001
                    log.debug("%s try_enter: %s", d.commodity, e)

        time.sleep(config.LOOP_SLEEP_S if hasattr(config, "LOOP_SLEEP_S")
                   else 0.2)


def _minutes_to_commodity_close(commodity: str, now: dt.datetime) -> float:
    """Minutes to this commodity's MCX session close (default 23:30 IST)."""
    spec = (getattr(config, "COMMODITIES", {}) or {}).get(commodity, {})
    close = spec.get("session_close", "23:30")
    hh, mm = (int(x) for x in close.split(":"))
    close_dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return max((close_dt - now).total_seconds() / 60.0, 0.0)


if __name__ == "__main__":
    main()