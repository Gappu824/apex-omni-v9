"""
APEX OMNI v9 — REAL-DAY REPLAY (zero synthetic data)
====================================================
    python simulation/replay_real_day.py [YYYY-MM-DD] [INDEX]

Replays a HARVESTED day — actual Kite ticks from your vault — through the
complete live decision stack: the same StateBuilder physics, the same
HeuristicPolicy (or whatever the brain would run), the same RiskGovernor,
ExecutionEngine (paper fills against the day's REAL recorded books, with
lookahead taken from the day's REAL future quotes), PositionManager and
TrapShield. Nothing is generated; every premium, spread, OI and feed gap is
what the market actually printed.

This is the answer to "no synthetic data": the 16 authored scenarios remain
the stress exam for disasters reality won't schedule on demand (flash
crashes, reject storms), while THIS replayer + the scenario miner keep the
regression diet 100% real once your vault has days in it.

Honest limits, printed at the end of each run: hierarchy is restricted to
the strikes the harvester actually recorded (ATM ± PRUNE_STEPS); GEX walls
need macro-history (not stored per-second), so the shield's wall factor is
inert here; and paper fills on recorded books are deterministic, slightly
kind in dead tape.
"""
from __future__ import annotations
import datetime as dt
import logging
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                          # noqa: E402
from core.market_state import StateBuilder             # noqa: E402
from core.heuristic_policy import HeuristicPolicy      # noqa: E402
from core.risk_manager import RiskGovernor             # noqa: E402
from core.execution_engine import ExecutionEngine      # noqa: E402
from core.position_manager import (PositionManager, LegQuote,   # noqa: E402
                                   TickContext)
from core.instruments import AsOfMapper                # noqa: E402
from core.quant_core import implied_vol_newton, black76_greeks  # noqa: E402
from simulation.scenario_engine import T0_SEC, N, hm_of         # noqa: E402

log = logging.getLogger("replay")


def load_day(con, day: str, index: str):
    r = con.execute("SELECT token FROM spot_tokens WHERE snap_date<=? AND "
                    "name=? ORDER BY snap_date DESC LIMIT 1",
                    (day, index)).fetchone()
    if not r:
        return None
    spot_tok = int(r[0])
    rows = con.execute(
        "SELECT ts_local_ms/1000, token, ltp, bid, ask, bid_qty, ask_qty, "
        "vol_delta, oi, iceberg FROM ticks_v9 WHERE "
        "date(ts_local_ms/1000,'unixepoch','localtime')=? ORDER BY ts_ms",
        (day,)).fetchall()
    if len(rows) < 600:
        return None
    by_sec: dict[int, dict] = defaultdict(dict)
    toks = set()
    for s, tok, ltp, bid, ask, bq, aq, vd, oi, ice in rows:
        t = int(s) % 86400 - T0_SEC
        if 0 <= t < N:
            by_sec[t][int(tok)] = {"ltp": ltp, "bid": bid, "ask": ask,
                                   "bid_qty": bq, "ask_qty": aq,
                                   "vol_delta": vd, "oi": oi, "iceberg": ice}
            toks.add(int(tok))
    # forward-filled bid/ask arrays → engine's quote feed + REAL lookahead
    ti = {tok: i for i, tok in enumerate(sorted(toks))}
    bidA = np.full((len(ti), N), np.nan, np.float32)
    askA = np.full_like(bidA, np.nan)
    for t, snaps in by_sec.items():
        for tok, sn in snaps.items():
            if sn["bid"] and sn["ask"]:
                bidA[ti[tok], t] = sn["bid"]
                askA[ti[tok], t] = sn["ask"]
    for arr in (bidA, askA):
        for k in range(arr.shape[0]):
            last = np.nan
            row = arr[k]
            for t in range(N):
                if np.isnan(row[t]):
                    row[t] = last
                else:
                    last = row[t]
    return spot_tok, by_sec, ti, bidA, askA


def main():
    day = sys.argv[1] if len(sys.argv) > 1 else None
    index = (sys.argv[2] if len(sys.argv) > 2 else "NIFTY").upper()
    config.setup_logging("replay")
    if not Path(config.DB_PATH).exists():
        sys.exit("No tick vault yet — run the harvester for a session first.")
    con = sqlite3.connect(config.DB_PATH)
    if day is None:
        day = con.execute("SELECT MAX(date(ts_local_ms/1000,'unixepoch',"
                          "'localtime')) FROM ticks_v9").fetchone()[0]
    loaded = load_day(con, day, index)
    con.close()
    if not loaded:
        sys.exit(f"{day}/{index}: not enough recorded ticks to replay.")
    spot_tok, by_sec, ti, bidA, askA = loaded
    mapper = AsOfMapper(dt.date.fromisoformat(day))
    log.info("replaying %s %s — %d tokens, snapshot %s, capital ₹%.0f, "
             "PAPER bar %.2f", day, index, len(ti), mapper.snapshot_used,
             config.TRADING_CAPITAL, config.PAPER_ENTRY_CONVICTION)

    builder = StateBuilder()
    policy = HeuristicPolicy()
    risk = RiskGovernor()
    cur = {"t": 0.0}
    snaps: dict[int, dict] = {}
    last_tick_t: dict[int, int] = {}

    def quote_fn(tok):
        k = ti.get(tok)
        t = int(cur["t"])
        if k is None or np.isnan(bidA[k, t]):
            return {}
        sn = snaps.get(tok, {})
        return {"bid": float(bidA[k, t]), "ask": float(askA[k, t]),
                "ltp": float(sn.get("ltp") or bidA[k, t]),
                "bid_qty": float(sn.get("bid_qty") or 0),
                "ask_qty": float(sn.get("ask_qty") or 0)}

    def lookahead(tok, horizon_s):
        k = ti.get(tok)
        if k is None:
            return None, None
        t = int(cur["t"])
        a = askA[k, t:t + int(horizon_s) + 1]
        b = bidA[k, t:t + int(horizon_s) + 1]
        if np.all(np.isnan(a)):
            return None, None
        return float(np.nanmin(a)), float(np.nanmax(b))

    eng = ExecutionEngine(kite=None, quote_fn=quote_fn,
                          clock=lambda: cur["t"])
    eng.lookahead_fn = lookahead
    ledger = config.LOG_DIR / f"replay_{day}_{index}.csv"
    if ledger.exists():
        ledger.unlink()
    pm = PositionManager(index, risk, eng, ledger_path=ledger)

    chain = None
    last_spot = None
    last_spot_t = -1
    spread_ew = 0.01
    last_try = -1e9
    iidx = config.INDEX_ORDER.index(index)

    for t in range(N):
        cur["t"] = float(t)
        for tok, sn in by_sec.get(t, {}).items():
            snaps[tok] = sn
            last_tick_t[tok] = t
        sp = snaps.get(spot_tok)
        if not sp or not sp.get("ltp"):
            continue
        risk.on_tick()
        spot = float(sp["ltp"])
        if spot_tok in by_sec.get(t, {}):
            last_spot_t = t
        data_age = float(t - last_spot_t)             # REAL feed staleness
        vel = spot - (last_spot if last_spot is not None else spot)
        last_spot = spot

        step = (chain or {}).get("step") or config.INDICES[index]["strike_step"]
        atm = round(spot / step) * step
        if chain is None or chain.get("atm") != atm:
            chain = mapper.chain(index, spot) or chain
        market = {index: {"spot": sp}}
        T = 0.01
        if chain:
            legs = {}
            for leg, info in chain["legs"].items():
                s = snaps.get(info["token"])
                if s:
                    legs[leg] = {"snap": s, "strike": info["strike"]}
            market[index].update({"expiry": chain["expiry"],
                                  "dte": chain["dte"], "T": chain["T"],
                                  "is_weekly": chain["is_weekly"],
                                  "legs": legs})
            T = chain["T"]
        obs = builder.push(market, float(t))
        frame = builder.frames[-1]
        conv = float(policy.predict(frame)[2 * iidx])

        # REAL atm IV via Newton on the recorded ATM mid
        atm_iv = 0.12
        ce = (market[index].get("legs") or {}).get("atm_ce", {}).get("snap")
        if chain and ce and ce.get("bid") and ce.get("ask"):
            mid = (ce["bid"] + ce["ask"]) / 2
            F = spot * math.exp(config.RISK_FREE_RATE * T)
            atm_iv = float(implied_vol_newton(mid, F, chain["atm"], T, True,
                                              config.RISK_FREE_RATE))
            spn = (ce["ask"] - ce["bid"]) / max(mid, 0.05)
            spread_ew = (1 - config.SPREAD_EW_ALPHA) * spread_ew + \
                config.SPREAD_EW_ALPHA * spn
        node = frame[iidx * config.NODES_PER_INDEX]
        absorb = any((v.get("snap") or {}).get("iceberg")
                     for v in (market[index].get("legs") or {}).values())
        t_ce = builder.trk.get(f"{index}:atm_ce")
        t_pe = builder.trk.get(f"{index}:atm_pe")
        opt_flow = ((t_ce.dealer_inv if t_ce else 0.0)
                    + (t_pe.dealer_inv if t_pe else 0.0))
        sell_ratio = float(np.clip(
            0.5 - 0.5 * math.tanh(opt_flow / config.DEALER_INV_SCALE), 0, 1))
        oi_node = node
        if pm.pos is not None:
            oi_node = frame[iidx * config.NODES_PER_INDEX +
                            (1 if pm.pos.direction == "CE" else 2)]
        ctx = TickContext(
            ts=float(t), hm=hm_of(t), spot=spot, spot_velocity_1s=vel,
            data_age_s=data_age, atm_iv=atm_iv,
            minutes_to_close=(N - t) / 60.0,
            gex_put_wall=None, gex_call_wall=None,      # macro history n/a
            absorption=absorb, aggressive_sell_ratio=sell_ratio,
            oi_delta_since=float(oi_node[2]),
            avg_spread_pct=spread_ew, conviction=conv)

        if pm.pos is not None:
            pm.manage(ctx, quote_fn(pm.pos.token))
            continue
        if (risk.halted or abs(conv) < config.PAPER_ENTRY_CONVICTION
                or t - last_try < config.ENTRY_ATTEMPT_THROTTLE_S or not chain):
            continue
        last_try = t
        d = "CE" if conv > 0 else "PE"
        F = spot * math.exp(config.RISK_FREE_RATE * T)
        hier = []
        for r in mapper.hierarchy(index, spot, d):
            tok = r["token"]
            if ti.get(tok) is None or t - last_tick_t.get(tok, -99) > 5:
                continue                       # only strikes the day recorded
            q = quote_fn(tok)
            if not q.get("bid") or not q.get("ask"):
                continue
            mid = (q["bid"] + q["ask"]) / 2
            g = black76_greeks(F, r["strike"], T, max(atm_iv, 0.05),
                               d == "CE", config.RISK_FREE_RATE)
            hier.append(LegQuote(leg=r["symbol"], symbol=r["symbol"],
                                 exchange=r["exchange"], token=tok,
                                 strike=r["strike"], premium=mid,
                                 bid=q["bid"], ask=q["ask"],
                                 bid_qty=q["bid_qty"], ask_qty=q["ask_qty"],
                                 lot=r["lot"], delta=float(g["delta"]),
                                 dte=float(chain["dte"])))
        if hier:
            pm.try_enter(ctx, d, conv, config.PAPER_EXPLORE_WINPROB, hier)

    # ------------------------------------------------------------ verdict
    trades, entries = [], {}
    counts = defaultdict(int)
    for e in pm.events:
        counts[e["event"]] += 1
        if e["event"] == "BUY_FILL":
            entries[e["symbol"]] = e
        elif e["event"] == "SELL_FILL" and e["symbol"] in entries:
            b = entries.pop(e["symbol"])
            trades.append((e["symbol"], float(b["price"]), float(e["price"]),
                           float(e["pnl"]), e["reason"],
                           float(e["ts"]) - float(b["ts"])))
    print("=" * 70)
    print(f" REAL-DAY REPLAY — {index} {day} (100% recorded Kite data)")
    print("=" * 70)
    print(f" trades {len(trades)} | PnL after costs ₹{risk.realized_pnl:+.2f}"
          f" | trap holds {counts['TRAP_HOLD']}"
          f" | blocked/skip {counts['BLOCKED'] + counts['SKIP']}"
          f" | nofills {counts['NOFILL']}"
          f" | halted={risk.halted or '-'}")
    for sym, ep, xp, pnl, why, held in trades:
        print(f"   {sym}: {ep:.2f} → {xp:.2f}  ₹{pnl:+.2f}  "
              f"({held/60:.0f} m, {why})")
    if pm.pos is not None:
        print(" ⚠ INVARIANT: position still open at session end")
    print(f" ledger → {ledger}")
    print(" notes: hierarchy limited to harvested strikes (ATM ± "
          f"{config.PRUNE_STEPS}); GEX wall factor inert (no macro history); "
          "paper fills on recorded books.")


if __name__ == "__main__":
    main()
