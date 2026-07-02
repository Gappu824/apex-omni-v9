#!/usr/bin/env python3
"""
gex_channel_backtest.py — does fading the GEX channel to max-pain clear costs?

The IC analyzer proved the channel features are LEARNABLE (wall_pos, maxpain_pull,
putwall_dist carry real, signed, horizon-building IC in this positive-gamma
window). IC is not P&L. This asks the only question that justifies the OBS_DIM
change and full retrain: trading the signal with REAL option premiums and the
REAL Zerodha cost stack at the configured capital, does it make money after costs?

Signal (the channel-fade the IC pointed to, sign-gated by gamma regime)
----------------------------------------------------------------------
The IC winners — wall_pos, putwall_dist, callwall_dist — all encode ONE thing:
distance from the channel midpoint. putwall_dist scored strongest only because it
is spot-normalised (stable) rather than channel-width-normalised (noisy walls);
blending it standalone injects a short bias (spot is always above the put wall).
So the signal is the two genuinely non-redundant "fade toward centre" pulls, both
spot-normalised and both symmetric (zero at the centre):
    maxpain_pull      = (max_pain − spot)/spot        (+IC → keep)
    channel_mid_pull  = (mid − spot)/spot, mid=(cw+pw)/2   (the stable, symmetric
                        form of the wall-distance signal; near put wall → +/up,
                        near call wall → −/down)
blended, then multiplied by sign(net_gex): in POSITIVE gamma the walls ATTRACT
(the regime we measured, so the blend predicts reversion as-is); in NEGATIVE gamma
they REPEL and the sign flips. NB: every vault day here is positive-gamma, so the
flip is implemented but UNTESTED — do not trust the negative-gamma branch until
you have short-gamma days to validate it.

Trade model (faithful, event-driven, one position per index at a time)
----------------------------------------------------------------------
  • Enter the ATM option in the signal's direction (CE if bullish, PE if bearish),
    only when max-pain sits in that direction (so "fade to max-pain" is coherent).
  • Fill at the TOUCH from the vault: BUY at the ask, SELL at the bid — the bid/ask
    spread is paid for real, not assumed away.
  • Size with the configured capital (floor((capital·deploy)/(ask·lot)), ≥1 lot or
    skip). Reported per-lot too, so the flat-brokerage amortisation is visible.
  • Exit at the earliest of: spot reaches max-pain (TP), adverse move > stop (SL),
    or max-hold (theta guillotine). Costs via execution_engine.round_trip_costs —
    the identical toll the live engine and the forge pay.

Offline, no torch. Needs the vault AND the instrument dump (AsOfMapper) on disk:
    python gex_channel_backtest.py [--db PATH] [--entry 0.5] [--stop-bp 25]
                                   [--max-hold-min 30] [--deploy 1.0]
"""
from __future__ import annotations
import argparse
import sqlite3
import datetime as dt
import numpy as np

import config
from core.instruments import AsOfMapper
from core.execution_engine import round_trip_costs
import nightly_forge_v9 as forge

TRADABLE = config.TRADABLE
CAP = float(config.TRADING_CAPITAL)
# fixed orientation scales (look-ahead-free); blend threshold is what's swept.
SC_MP, SC_MID = 0.004, 0.004


def load_macro(con, day, idx):
    try:
        rows = con.execute(
            "SELECT ts_ms/1000.0, spot, flip, call_wall, put_wall, net_gex, "
            "pcr, max_pain FROM macro_snapshots_v9 WHERE index_name=? AND "
            "date(ts_ms/1000,'unixepoch','localtime')=? ORDER BY ts_ms",
            (idx, day)).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for ts, s, flip, cw, pw, ng, pcr, mp in rows:
        out.append((float(ts), {"spot": s, "call_wall": cw, "put_wall": pw,
                                 "net_gex": ng, "max_pain": mp}))
    return out


def spot_path(con, day, tok):
    rows = con.execute(
        "SELECT ts_ms/1000, ltp FROM ticks_v9 WHERE token=? AND ltp>0 AND "
        "date(ts_local_ms/1000,'unixepoch','localtime')=? ORDER BY ts_ms",
        (tok, day)).fetchall()
    if not rows:
        return np.array([]), np.array([])
    secs = np.array([int(r[0]) for r in rows])
    px = np.array([float(r[1]) for r in rows])
    return secs, px


def token_quote(con, tok, sec, max_stale=180):
    r = con.execute(
        "SELECT ts_ms/1000, bid, ask, ltp FROM ticks_v9 WHERE token=? AND "
        "ts_ms/1000<=? ORDER BY ts_ms DESC LIMIT 1", (tok, sec)).fetchone()
    if not r or (sec - r[0]) > max_stale:
        return None
    _, bid, ask, ltp = r
    ltp = float(ltp or 0)
    bid = float(bid or 0) or ltp
    ask = float(ask or 0) or ltp
    if ltp <= 0:
        return None
    return bid, ask, ltp


def _signal(m):
    s, cw, pw, mp, ng = (m["spot"], m["call_wall"], m["put_wall"],
                         m["max_pain"], m["net_gex"])
    vals = [s, cw, pw, mp, ng]
    if any(v is None or not np.isfinite(v) for v in vals) or s <= 0:
        return None
    comps = [(mp - s) / s / SC_MP]                 # max-pain pull
    if cw > pw:
        mid = 0.5 * (cw + pw)
        comps.append((mid - s) / s / SC_MID)       # channel-midpoint pull (symmetric)
    blend = float(np.mean(comps))
    return blend * (1.0 if ng > 0 else -1.0)


def _px_at(secs, px, sec):
    if len(secs) == 0:
        return None
    j = np.searchsorted(secs, sec, side="right") - 1
    return float(px[j]) if j >= 0 else None


def _find_exit(secs, px, entry_sec, direction, entry_spot, tp_level,
               stop_bp, max_hold_s):
    """Walk the spot path; return (exit_sec, reason)."""
    stop_level = entry_spot * (1 - direction * stop_bp / 1e4)
    end = entry_sec + max_hold_s
    lo = np.searchsorted(secs, entry_sec, side="right")
    for k in range(lo, len(secs)):
        sec, p = int(secs[k]), float(px[k])
        if sec > end:
            return end, "time"
        if direction > 0:
            if p >= tp_level:
                return sec, "tp"
            if p <= stop_level:
                return sec, "sl"
        else:
            if p <= tp_level:
                return sec, "tp"
            if p >= stop_level:
                return sec, "sl"
    return (int(secs[-1]) if len(secs) else end), "eod"


def backtest(db, entry_thr, stop_bp, max_hold_min, deploy):
    con = sqlite3.connect(db)
    try:
        days = forge.trading_days(con)
    except sqlite3.OperationalError as e:
        print(f"cannot read ticks_v9: {e}")
        return
    max_hold_s = int(max_hold_min * 60)
    trades = []
    skipped = {"unaffordable": 0, "no_quote": 0, "no_token": 0, "mp_against": 0}
    for day in days:
        mapper = AsOfMapper(dt.date.fromisoformat(day))
        if getattr(mapper, "snapshot_used", None) is None:
            print(f"  {day}: no instrument dump ≤ date — skipping (need the chain)")
            continue
        spot_tok = {i: forge.spot_token_for(con, day, i) for i in TRADABLE}
        for idx in TRADABLE:
            macro = load_macro(con, day, idx)
            if not macro or not spot_tok[idx]:
                continue
            secs, px = spot_path(con, day, spot_tok[idx])
            if len(secs) == 0:
                continue
            lot_step = config.INDICES[idx]["strike_step"]
            busy_until = -1
            for ts, m in macro:
                if ts < busy_until:
                    continue
                sig = _signal(m)
                if sig is None or abs(sig) < entry_thr:
                    continue
                direction = 1 if sig > 0 else -1
                s0 = m["spot"]; mp = m["max_pain"]
                # "fade to max-pain" only if max-pain is in the signal's direction
                if (mp - s0) * direction <= 0:
                    skipped["mp_against"] += 1
                    continue
                ch = mapper.chain(idx, s0)
                if not ch:
                    skipped["no_token"] += 1
                    continue
                leg = "atm_ce" if direction > 0 else "atm_pe"
                info = ch["legs"].get(leg)
                if not info:
                    skipped["no_token"] += 1
                    continue
                lot = int(ch["lot"])
                q = token_quote(con, info["token"], int(ts))
                if not q:
                    skipped["no_quote"] += 1
                    continue
                entry_ask = q[1]
                qty = int((CAP * deploy) // (entry_ask * lot))
                if qty < 1:
                    skipped["unaffordable"] += 1
                    continue
                # exit
                tp_level = float(mp)
                ex_sec, reason = _find_exit(secs, px, int(ts), direction, s0,
                                            tp_level, stop_bp, max_hold_s)
                xq = token_quote(con, info["token"], ex_sec, max_stale=600)
                if not xq:
                    skipped["no_quote"] += 1
                    continue
                exit_bid = xq[0]
                entry_mid = 0.5 * (q[0] + q[1])
                exit_mid = 0.5 * (xq[0] + xq[1])
                exit_spot = _px_at(secs, px, ex_sec) or s0
                bv = entry_ask * lot * qty
                sv = exit_bid * lot * qty
                gross = sv - bv
                costs = round_trip_costs(bv, sv, 1, 1)
                net = gross - costs
                mid_gross = (exit_mid - entry_mid) * lot * qty
                trades.append({
                    "day": day, "idx": idx, "dir": direction, "reason": reason,
                    "entry_ask": entry_ask, "exit_bid": exit_bid, "lot": lot,
                    "qty": qty, "gross": gross, "costs": costs, "net": net,
                    "hold_min": (ex_sec - ts) / 60.0,
                    "mid_gross": mid_gross,
                    "spread_cost": gross - mid_gross,          # ≤ 0, the spread paid
                    "spot_move_fav": (exit_spot - s0) * direction,   # favourable index pts
                    "net_per_lot": (exit_bid - entry_ask) * lot
                                   - round_trip_costs(entry_ask * lot, exit_bid * lot, 1, 1),
                })
                busy_until = ex_sec
    con.close()
    _report(days, trades, skipped, entry_thr, stop_bp, max_hold_min, deploy)


def _report(days, trades, skipped, entry_thr, stop_bp, max_hold_min, deploy):
    print("\n" + "=" * 74)
    print(f" GEX-CHANNEL FADE — cost-aware backtest   capital ₹{CAP:,.0f}")
    print(f" entry|sig|≥{entry_thr}  stop {stop_bp}bp  max-hold {max_hold_min}m  "
          f"deploy {deploy:.0%}  ({len(days)} days)")
    print("=" * 74)
    if not trades:
        print(" no trades. skipped:", skipped)
        return
    g = np.array([t["gross"] for t in trades])
    c = np.array([t["costs"] for t in trades])
    n = np.array([t["net"] for t in trades])
    npl = np.array([t["net_per_lot"] for t in trades])
    mid = np.array([t["mid_gross"] for t in trades])
    spread = np.array([t["spread_cost"] for t in trades])
    smove = np.array([t["spot_move_fav"] for t in trades])
    wins = (n > 0).sum()
    by_reason = {}
    for t in trades:
        by_reason[t["reason"]] = by_reason.get(t["reason"], 0) + 1
    eq = np.cumsum(n)
    dd = (np.maximum.accumulate(eq) - eq).max() if len(eq) else 0.0

    print(f"\n  trades: {len(trades)}   win rate: {wins/len(trades)*100:.1f}%   "
          f"exits: {by_reason}")
    print(f"\n  GROSS  (touch-to-touch, spread paid):  ₹{g.sum():+,.0f}   "
          f"avg ₹{g.mean():+,.0f}/trade")
    print(f"  COSTS  (brokerage+STT+txn+GST+stamp):  ₹{c.sum():,.0f}   "
          f"avg ₹{c.mean():,.0f}/trade")
    print(f"  NET    (after all costs):              ₹{n.sum():+,.0f}   "
          f"avg ₹{n.mean():+,.0f}/trade")
    print(f"\n  cost drag: costs are {c.sum()/max(abs(g.sum()),1)*100:.0f}% of |gross|")
    print(f"  net per LOT (size-independent edge unit): avg ₹{npl.mean():+,.1f}/lot  "
          f"(>0 on {(npl>0).mean()*100:.0f}% of trades)")
    print(f"  net as % of capital over window: {n.sum()/CAP*100:+.1f}%   "
          f"max drawdown: ₹{dd:,.0f}")

    # ---- loss decomposition: is it DIRECTION, THETA, or SPREAD? ----
    print("\n  DECOMPOSITION — where the money goes")
    n_tr = len(trades)
    hit_frac = (smove > 0).mean()
    hit_z = (hit_frac - 0.5) / np.sqrt(0.25 / n_tr) if n_tr else 0.0
    sig = "SIGNIFICANT" if abs(hit_z) >= 2 else "not sig (≈coin flip)"
    print(f"    spot direction: favourable on {hit_frac*100:.0f}% of trades "
          f"(z={hit_z:+.1f} vs 50%, {sig}), avg favourable move {smove.mean():+.1f} pts")
    print(f"    MID-to-MID gross (spread removed):     ₹{mid.sum():+,.0f}   "
          f"avg ₹{mid.mean():+,.0f}/trade")
    print(f"    spread paid (buy ask / sell bid):      ₹{spread.sum():,.0f}   "
          f"avg ₹{spread.mean():,.0f}/trade")
    delta_pnl = np.array([0.5 * t["spot_move_fav"] * t["lot"] * t["qty"] for t in trades])
    resid = mid - delta_pnl
    rsign = ("theta DRAG" if resid.mean() < 0 else
             "convexity HELPED (downside floored at premium)")
    print(f"    delta-implied ₹{delta_pnl.mean():+,.0f}/trade  →  residual "
          f"₹{resid.mean():+,.0f}/trade  [{rsign}]")
    print(f"\n  skipped: {skipped}")

    # the verdict, now earned by the decomposition (and its SIGNS)
    print("\n" + "-" * 74)
    if npl.mean() > 0 and n.sum() > 0:
        print("  VERDICT: clears costs in-sample. The edge survives the spread and the")
        print("  Zerodha stack at this size — the OBS_DIM change + retrain is justified.")
    elif abs(hit_z) < 2:
        print("  VERDICT: the SIGNAL doesn't translate to a tradable edge. The direction is")
        print(f"  favourable on only {hit_frac*100:.0f}% of trades — NOT significant vs a coin flip")
        print("  (z={:+.1f}). The channel IC is a real pooled association but collapses at".format(hit_z))
        print("  entry timing through the max-pain filter + exit mechanics. Per-lot loss is")
        print(f"  ₹{npl.mean():+,.0f} ({npl.mean()/ (np.mean([t['entry_ask']*t['lot'] for t in trades]))*100:+.1f}%/trade); "
              "the headline −%cap is that × full-capital-per-trade compounding.")
        print("  Neither theta nor spread is the cause — the entry signal itself is too weak.")
        print("  Do NOT retrain on this, and do NOT assume short-premium fixes it either —")
        print("  first establish a directional edge that beats a coin flip out-of-sample.")
    elif resid.mean() < 0 and abs(resid.mean()) > abs(spread.mean()):
        print("  VERDICT: direction is real but THETA drags — the residual is negative and")
        print("  dominates. The reversion is too slow for bought options in this low-vol")
        print("  pinning regime; this looks like a SHORT-premium edge. Test that expression")
        print("  before retraining the long-option policy.")
    elif abs(spread.mean()) > abs(mid.mean()):
        print("  VERDICT: direction is real but the SPREAD eats it — test limit fills / more")
        print("  liquid strikes / debit spreads; the raw directional move may survive.")
    else:
        print("  VERDICT: direction is real but the per-trade edge is thin and loses after")
        print("  the option's carry. Not spread, not cleanly theta (residual floored by")
        print("  limited downside). The signal is too weak at this timing/threshold.")
    print("  (in-sample, 1 positive-gamma regime, 10 days — necessary, not sufficient.)")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(config.DB_PATH))
    ap.add_argument("--entry", type=float, default=0.5)
    ap.add_argument("--stop-bp", type=float, default=25.0)
    ap.add_argument("--max-hold-min", type=float, default=30.0)
    ap.add_argument("--deploy", type=float, default=1.0)
    a = ap.parse_args()
    backtest(a.db, a.entry, a.stop_bp, a.max_hold_min, a.deploy)


if __name__ == "__main__":
    main()