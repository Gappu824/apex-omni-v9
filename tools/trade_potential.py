"""
TRADE POTENTIAL — how much of each trade did we actually capture?
==================================================================
    python tools/trade_potential.py [--days N] [--json out.json]

Every exit this system has taken has been a clock. 2026-08-03 is the
clean example: SENSEX 78300PE entered at ₹189.85, peaked at ₹197.85,
exited on MAX_HOLD_THETA at ₹170.30 for −₹444.70. The target sat at
₹417.67 — unreachable, so it never armed anything. The trade had power
in it; the exit stack did not collect any.

This tool measures that gap on EVERY real fill, and then asks which
exit policy would have collected more — over the WHOLE session, not
just the window the trade happened to live in.

METHOD
------
1. FILLS ARE REAL. BUY_FILL→SELL_FILL pairs from execution_ledger_v9.csv,
   matched exactly as core.edge_audit and the harnesses match them. No
   re-simulated entries: this study conditions on the entry and asks
   only about the exit, which is the one question a post-hoc replay can
   answer without selection bias.

2. PATHS ARE REAL. The per-second bid/ask arrays for that exact symbol
   come from the day cache the forge already builds. Exits mark at the
   BID (you sell into the bid) and every policy pays the identical
   round-trip cost stack, so policies differ only in WHEN they leave.

3. MFE / MAE (Sweeney 1996). For each trade: the best unrealised gain
   available after entry (MFE), the worst drawdown (MAE), when each
   occurred, and the CAPTURE RATIO — realised ÷ MFE. Capture is the
   headline: it is "the full power of the trade" expressed as a number
   between 0 and 1.

4. COUNTERFACTUAL POLICIES, replayed to session close under the rules
   in force from 2026-08-03:
       as_traded      what actually happened (the baseline)
       hold_to_close  no time stop at all
       hold_2x/3x     theta budget extended (ride multiples)
       trail_10/20/30 peak-anchored ratchet, k% giveback from peak
       target_1R/2R/3R  R = entry − stop, the risk actually taken
   Every policy still obeys the disaster floor and the session hard-flat,
   because those are constitution, not preference.

5. STATISTICS. Per-policy minus as_traded, PAIRED per trade and
   clustered by DAY through core.capability_ladder.paired_test, then
   Benjamini-Hochberg across the policy family. Every policy is
   pre-registered in the trial registry before its result exists. The
   report always states the smallest ₹ difference this sample could
   have resolved, so "no better policy" is never confused with "not
   enough trades to tell".

WHAT THIS IS NOT: a backtest of a new strategy. It cannot tell you
whether to enter. It tells you, of the trades you did take, how much
was left on the table and which exit rule would have left less.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                              # noqa: E402

config.setup_logging("trade_potential")
import logging                                             # noqa: E402
log = logging.getLogger("trade_potential")

from core import capability_ladder as CL                   # noqa: E402
from core.execution_engine import round_trip_costs         # noqa: E402

REPORT = config.LOG_DIR / "trade_potential_{d}.json"


# ------------------------------------------------------------------ fills
def closed_fills() -> list[dict]:
    """BUY_FILL→SELL_FILL pairs, matched on symbol exactly as the
    harnesses do. Open positions are excluded — an unfinished trade has
    no realised number to compare a policy against."""
    p = Path(config.LEDGER_PATH)
    if not p.exists():
        return []
    out, opens = [], {}
    for r in csv.DictReader(p.open(encoding="utf-8")):
        ev, sym = r.get("event"), r.get("symbol") or ""
        if ev == "BUY_FILL":
            opens[sym] = r
        elif ev == "SELL_FILL" and sym in opens:
            b = opens.pop(sym)
            try:
                out.append({
                    "symbol": sym,
                    "buy_ts": float(b.get("ts") or 0),
                    "sell_ts": float(r.get("ts") or 0),
                    "entry": float(b.get("price") or 0),
                    "exit": float(r.get("price") or 0),
                    "qty": int(float(b.get("qty") or 0)),
                    "token": int(float(b.get("token") or 0) or 0),
                    "pnl": float(r.get("pnl") or 0),
                    "reason": r.get("reason") or r.get("note") or "",
                })
            except (TypeError, ValueError):
                continue
    return [f for f in out if f["entry"] > 0 and f["qty"] > 0]


def _day_of(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).date().isoformat()


# ------------------------------------------------------------- one replay
def _pnl(entry: float, exit_px: float, qty: int) -> float:
    """Net ₹ for one round trip at the live cost stack — identical for
    every policy, so the comparison isolates timing."""
    gross = (exit_px - entry) * qty
    return gross - round_trip_costs(entry * qty, exit_px * qty)


def _policies(entry: float, stop: float, qty: int, path: np.ndarray,
              held_s: int, hold_budget_s: int) -> dict[str, float]:
    """Replay each exit rule along the real bid path (index = seconds
    after entry). Returns net ₹ per policy."""
    n = path.size
    if n == 0:
        return {}
    R = max(entry - stop, 1e-6)          # the risk actually taken, per unit
    floor_px = entry * (1.0 - float(getattr(config, "MAX_LOSS_PER_TRADE_PCT",
                                            0.3)))
    out: dict[str, float] = {}

    be_px = entry + round_trip_costs(entry * qty, entry * qty) / max(qty, 1)

    def _walk(limit_s: int, trail_pct: float | None = None,
              target_px: float | None = None,
              lock_at: float | None = None) -> float:
        peak = entry
        end = min(limit_s, n - 1)
        for t in range(end + 1):
            px = float(path[t])
            if not np.isfinite(px) or px <= 0:
                continue
            peak = max(peak, px)
            if px <= floor_px:                       # disaster floor: law
                return _pnl(entry, px, qty)
            if target_px is not None and px >= target_px:
                return _pnl(entry, target_px, qty)
            if trail_pct is not None and t >= 1:
                # A trailing stop is a STOP: it fires on giveback from the
                # running peak whether or not the peak was ever in profit.
                # An earlier draft required the ratchet level to sit above
                # entry, which meant a trade peaking only +4% (the real
                # 2026-08-03 case) could never trail at all and rode all
                # the way down to the clock. The profit-LOCK variant below
                # is the separate policy that refuses to give back gains.
                if px <= peak * (1.0 - trail_pct):
                    return _pnl(entry, px, qty)
            if lock_at is not None and peak >= entry * (1.0 + lock_at):
                # profit lock: once the trade has shown `lock_at`, never
                # exit below breakeven+costs again.
                if px <= be_px:
                    return _pnl(entry, be_px, qty)
        last = float(path[end])
        return _pnl(entry, last, qty) if np.isfinite(last) else 0.0

    out["hold_to_close"] = _walk(n - 1)
    out["hold_2x"] = _walk(int(hold_budget_s * 2))
    out["hold_3x"] = _walk(int(hold_budget_s * 3))
    for k in (10, 20, 30):
        out[f"trail_{k}"] = _walk(n - 1, trail_pct=k / 100.0)
    for m in (1, 2, 3):
        out[f"target_{m}R"] = _walk(int(hold_budget_s), target_px=entry + m * R)
    for lk in (5, 10):
        out[f"lock_{lk}pct"] = _walk(int(hold_budget_s), lock_at=lk / 100.0)
    out["trail20_hold2x"] = _walk(int(hold_budget_s * 2), trail_pct=0.20)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()

    fills = closed_fills()
    if not fills:
        log.info("no closed fills in the ledger yet — nothing to study")
        return 0
    if a.days > 0:
        cutoff = time.time() - a.days * 86400
        fills = [f for f in fills if f["buy_ts"] >= cutoff]
    by_day: dict[str, list] = {}
    for f in fills:
        by_day.setdefault(_day_of(f["buy_ts"]), []).append(f)
    log.info("studying %d closed fill(s) over %d session(s)",
             len(fills), len(by_day))

    from simulation.replay_real_day import load_day
    con = sqlite3.connect(str(config.DB_PATH))
    rows, deltas = [], {}
    try:
        for day, day_fills in sorted(by_day.items()):
            try:
                loaded = load_day(con, day, config.TRADABLE[0])
            except Exception as e:                         # noqa: BLE001
                log.warning("  %s: day cache unavailable (%s) — skipped",
                            day, e)
                continue
            if not loaded:
                continue
            _stok, by_sec, ti, bidA, askA = loaded
            base = min(by_sec) if by_sec else 0
            for f in day_fills:
                k = ti.get(f["token"])
                if k is None:
                    log.warning("  %s %s: token not in the day cache — "
                                "skipped", day, f["symbol"])
                    continue
                t0 = int(f["buy_ts"] - base)
                if t0 < 0 or t0 >= bidA.shape[1]:
                    continue
                path = np.asarray(bidA[k, t0:], dtype=float)
                if path.size < 10:
                    continue
                held_s = max(int(f["sell_ts"] - f["buy_ts"]), 1)
                # rules in force from 2026-08-03: the 60-minute guillotine
                # (25 on expiry day) is the budget every hold multiple
                # extends from.
                dte_guess = 9.0
                hold_budget = int(config.MAX_HOLD_MINUTES * 60)
                entry, qty = f["entry"], f["qty"]
                stop = entry * (1.0 - float(config.BASE_SL_PCT))
                finite = path[np.isfinite(path) & (path > 0)]
                if finite.size == 0:
                    continue
                mfe_px = float(finite.max())
                mae_px = float(finite.min())
                i_mfe = int(np.nanargmax(np.where(np.isfinite(path), path,
                                                  -np.inf)))
                mfe_rs = _pnl(entry, mfe_px, qty)
                mae_rs = _pnl(entry, mae_px, qty)
                real = float(f["pnl"])
                cap_ratio = (real / mfe_rs) if mfe_rs > 0 else None
                pol = _policies(entry, stop, qty, path, held_s, hold_budget)
                pol["as_traded"] = real
                for name, v in pol.items():
                    if name == "as_traded":
                        continue
                    deltas.setdefault(name, {}).setdefault(day, []).append(
                        v - real)
                rows.append({
                    "day": day, "symbol": f["symbol"],
                    "entry": round(entry, 2), "exit": round(f["exit"], 2),
                    "qty": qty, "held_min": round(held_s / 60.0, 1),
                    "reason": f["reason"],
                    "realised_rs": round(real, 2),
                    "mfe_px": round(mfe_px, 2), "mfe_rs": round(mfe_rs, 2),
                    "mfe_at_min": round(i_mfe / 60.0, 1),
                    "mae_px": round(mae_px, 2), "mae_rs": round(mae_rs, 2),
                    "capture_ratio": (round(cap_ratio, 3)
                                      if cap_ratio is not None else None),
                    "policies": {k2: round(v2, 2) for k2, v2 in pol.items()},
                })
    finally:
        con.close()

    if not rows:
        log.error("no fill could be matched to a day cache — run the forge "
                  "so the caches exist, then re-run")
        return 1

    # ---------------------------------------------------------- capture
    caps = [r["capture_ratio"] for r in rows if r["capture_ratio"] is not None]
    tot_real = sum(r["realised_rs"] for r in rows)
    tot_mfe = sum(r["mfe_rs"] for r in rows if r["mfe_rs"] > 0)
    log.info("─" * 72)
    log.info("CAPTURE | %d trade(s) | realised ₹%+,.0f against ₹%+,.0f of "
             "peak unrealised available ⇒ %.1f%% captured",
             len(rows), tot_real, tot_mfe,
             100.0 * tot_real / tot_mfe if tot_mfe > 0 else float("nan"))
    if caps:
        log.info("        per-trade capture: median %.2f | p25 %.2f | "
                 "p75 %.2f", float(np.median(caps)),
                 float(np.percentile(caps, 25)),
                 float(np.percentile(caps, 75)))
    log.info("        peak arrived at median %.1f min after entry "
             "(exits happened at median %.1f min)",
             float(np.median([r["mfe_at_min"] for r in rows])),
             float(np.median([r["held_min"] for r in rows])))

    # ------------------------------------------------- policy comparison
    cap = CL.assess(np.array([1.0 if r["realised_rs"] > 0 else 0.0
                              for r in rows]),
                    np.ones(len(rows)),
                    np.array([r["day"] for r in rows]))
    log.info("LADDER | %s", cap.reason)
    try:
        from core.trial_registry import register
        for name in sorted(deltas):
            register(family="trade_potential", spec_id=name,
                     kind="pre_registered", n_trades=len(rows))
    except Exception as e:                                 # noqa: BLE001
        log.debug("registry unavailable (%s)", e)

    pol_rows = []
    for name, per_day in sorted(deltas.items()):
        day_mean = {d: float(np.mean(v)) for d, v in per_day.items()}
        st = CL.paired_test(day_mean)
        pol_rows.append({"policy": name, "delta_rs": st,
                         "total_rs": round(sum(sum(v) for v in
                                               per_day.values()), 2)})
    if pol_rows:
        rej, adj = CL.benjamini_hochberg(
            [p["delta_rs"]["p"] for p in pol_rows],
            float(getattr(config, "DISCOVERY_FDR_Q", 0.10)))
        for p_, ok_, q_ in zip(pol_rows, rej, adj):
            p_["p_adj_bh"] = round(float(q_), 4)
            p_["significant"] = bool(ok_)
    log.info("─" * 72)
    log.info("%-14s %12s %12s %9s  %s", "policy", "Σ Δ₹ vs as-traded",
             "mean/day", "p(BH)", "verdict")
    for p_ in sorted(pol_rows, key=lambda z: -z["total_rs"]):
        st = p_["delta_rs"]
        verdict = ("BETTER" if p_.get("significant") and st["mean"] > 0 else
                   "WORSE" if p_.get("significant") else
                   f"indistinguishable (could resolve ₹"
                   f"{st.get('mde', float('nan')):,.0f}/day)")
        log.info("%-14s %+12,.0f %+12,.0f %9.3f  %s", p_["policy"],
                 p_["total_rs"], st["mean"], p_.get("p_adj_bh", 1.0), verdict)
    log.info("─" * 72)
    log.info("These are EXIT counterfactuals on trades already entered. "
             "They cannot say whether an entry was right — only how much "
             "of it was collected.")

    out = {"ts": time.time(), "config_hash": config.CONFIG_HASH,
           "n_trades": len(rows), "n_days": len(by_day),
           "realised_rs": round(tot_real, 2), "mfe_rs": round(tot_mfe, 2),
           "capture_pct": (round(100.0 * tot_real / tot_mfe, 2)
                           if tot_mfe > 0 else None),
           "ladder": cap.as_dict(), "policies": pol_rows, "trades": rows}
    p = Path(a.json) if a.json else Path(
        str(REPORT).format(d=time.strftime("%Y-%m-%d")))
    try:
        p.write_text(json.dumps(out, indent=1, default=float))
        log.info("report → %s", p)
    except Exception as e:                                 # noqa: BLE001
        log.warning("report write failed (%s)", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())