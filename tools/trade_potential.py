"""
TRADE POTENTIAL v2 — how much of each trade did we actually capture?
====================================================================
    python tools/trade_potential.py [--days N] [--json out.json]
                                    [--promote | --dry-run]

Every exit this system has taken has been a clock. 2026-08-03 is the
clean example: SENSEX 78300PE entered at ₹189.85, peaked at ₹197.85,
exited on MAX_HOLD_THETA at ₹170.30 for −₹444.70. The target sat at
₹417.67 — unreachable, so it never armed anything. The trade had power
in it; the exit stack did not collect any.

WHAT CHANGED IN v2, AND WHY
---------------------------
v1 shipped in v9.9.10 and ran nightly. It never studied a single trade,
and six independent defects were responsible:

  F. `token` was not a column in LEDGER_FIELDS, so every reconstructed
     fill carried token=0, every `ti.get(0)` missed, and every trade hit
     the "not in the day cache" continue. n_trades was 0 every night.
     → the ledger schema now carries token; core.trade_reconstruct
       backfills history from instrument_snapshots.
  B. Pairing required BUY_FILL first, so the entire SHORT book — every
     butterfly body, every shortvol spread — was invisible.
     → signed FIFO; long-first and short-first share one code path.
  C. `opens[sym] = r` destroyed the first of two entries in one symbol
     and mispriced the survivor.
     → FIFO lots; re-entries stack, partials split.
  A. The path array ran 09:15→15:30 from constants written before the
     2026-08-03 reform, so every post-auction fill was out of bounds and
     every trade open at 15:30 was amputated.
     → simulation.session_paths takes the window from the DATE-AWARE
       core.session_calendar, so a pre-reform day still ends 15:30 and
       no ten minutes are ever fabricated.
  E. Unbounded forward-fill turned a pruned, unquoted leg into a
     perfectly flat, perfectly finite price series. Trails could not
     fire on it; targets could not be hit; MFE and capture were computed
     over a path that had stopped existing.
     → bounded carry + a freshness mask. Dead is NaN, and every trade
       reports its live-mark coverage.
  D. Nothing read the output. The loop was open.
     → the verdict goes through core.exit_policy_store, which promotes
       only on paired-by-day, FDR-corrected, MDE-clearing, holdout-
       agreeing evidence.

METHOD
------
1. FILLS ARE REAL, and now completely read (core.trade_reconstruct).
2. PATHS ARE REAL and honestly masked (simulation.session_paths). Exits
   mark at the side you would transact against and every policy pays the
   identical cost stack, so policies differ only in WHEN they leave.
3. MFE / MAE (Sweeney 1996) and the CAPTURE RATIO — realised ÷ MFE.
4. POLICIES come from core.exit_policies — the SAME functions the live
   shadow book steps every second. There is no second implementation to
   drift.
5. STATISTICS: per-day pairing, day-cluster bootstrap + sign-flip
   permutation, Benjamini-Hochberg across the pre-registered family, and
   an explicit MDE so "no better policy" is never confused with "not
   enough trades to tell".

WHAT THIS IS NOT: a backtest of a new strategy. It conditions on the
entry and asks only about the exit — the one question a post-hoc replay
can answer without selection bias.
"""
from __future__ import annotations

import argparse
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

from core import exit_policy_store as EPS                  # noqa: E402
from core.exit_policies import (PolicySpec, TradeCtx,      # noqa: E402
                                replay, pnl_of)
from core.execution_engine import round_trip_costs         # noqa: E402
from core.trade_reconstruct import reconstruct_all         # noqa: E402
from simulation.session_paths import (load_session_paths,  # noqa: E402
                                      traded_tokens)

REPORT = config.LOG_DIR / "trade_potential_{d}.json"


def _register(names) -> None:
    """Pre-register every policy BEFORE its result exists. The names come
    from config.SHADOW_POLICIES, not from this tool's locals, so a policy
    cannot be invented after the fact and back-dated into the family."""
    try:
        from core.trial_registry import register
        for n in sorted(names):
            register(family="trade_potential", spec_id=n,
                     kind="pre_registered", n_trades=0)
    except Exception as e:                                 # noqa: BLE001
        log.debug("registry unavailable (%s)", e)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--promote", action="store_true",
                    help="write the promotion if the gate is cleared")
    ap.add_argument("--dry-run", action="store_true",
                    help="evaluate the gate but never write")
    a = ap.parse_args()

    rec = reconstruct_all(days=a.days)
    rec.log_summary(log)
    trades = [t for t in rec.trades if t.resolved and t.qty > 0]
    if not trades:
        log.info("no reconstructable closed trade with a resolved token — "
                 "nothing to study. If the ledger is non-empty, the gap is "
                 "instrument_snapshots (symbol→token), not the ledger.")
        return 0

    specs = PolicySpec.family()
    _register([s.name for s in specs])
    log.info("policy family (pre-registered in config.SHADOW_POLICIES): %s",
             ", ".join(s.name for s in specs))

    by_day_index: dict[tuple[str, str], list] = {}
    for t in trades:
        by_day_index.setdefault((t.day, t.index), []).append(t)

    con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    rows, per_trade = [], []
    no_path = 0
    try:
        for (day, index), day_trades in sorted(by_day_index.items()):
            ps = load_session_paths(con, day, index,
                                    tokens=traded_tokens(day_trades))
            if ps is None:
                log.warning("  %s %s: no ticks in the vault — %d trade(s) "
                            "skipped", day, index, len(day_trades))
                no_path += len(day_trades)
                continue
            win = ps.window
            log.info("  %s %s: window %s→%s (%ds), %d token(s)",
                     day, index, win.open_hm, win.close_hm, win.n,
                     len(ps.ti))
            for t in day_trades:
                t0 = win.ts_to_t(t.entry_ts)
                if not win.contains(t0):
                    log.warning("    %s entry %s outside the session window "
                                "— skipped", t.symbol, win.t_to_hm(t0))
                    no_path += 1
                    continue
                side_key = "bid" if t.side > 0 else "ask"
                path = ps.path(t.token, t0, side=side_key)
                fresh = ps.fresh_mask(t.token, t0)
                if path is None or path.size < 10:
                    no_path += 1
                    continue
                cov = ps.coverage(t.token, t0)
                budget = int(float(getattr(config, "MAX_HOLD_MINUTES",
                                           60)) * 60)
                ctx = TradeCtx(entry=t.entry_px, qty=t.qty, side=t.side,
                               hold_budget_s=budget,
                               session_end_t=max(win.n - t0 - 1, 1))
                outs = {s.name: replay(s, ctx, path, round_trip_costs, fresh)
                        for s in specs}
                real = float(t.realized_pnl)
                finite = path[np.isfinite(path) & (path > 0)]
                if finite.size == 0:
                    no_path += 1
                    continue
                fav = (finite - t.entry_px) * t.side
                mfe_px = float(finite[int(np.argmax(fav))])
                mae_px = float(finite[int(np.argmin(fav))])
                mfe_rs = pnl_of(ctx, mfe_px, round_trip_costs)
                died = ps.died_at(t.token)

                per_trade.append({
                    "day": t.day, "as_traded": real, "coverage": cov,
                    "policy_pnl": {k: o.pnl for k, o in outs.items()}})
                rows.append({
                    "day": t.day, "index": t.index, "symbol": t.symbol,
                    "kind": t.kind, "side": t.side, "qty": t.qty,
                    "entry": round(t.entry_px, 2),
                    "exit": round(t.exit_px, 2),
                    "held_min": round(t.held_s / 60.0, 1),
                    "reason": t.reason, "realised_rs": round(real, 2),
                    "mfe_px": round(mfe_px, 2), "mfe_rs": round(mfe_rs, 2),
                    "mae_px": round(mae_px, 2),
                    "capture_ratio": (round(real / mfe_rs, 3)
                                      if mfe_rs > 0 else None),
                    "coverage": round(cov, 3),
                    "feed_died_at": (win.t_to_hm(died) if died is not None
                                     else None),
                    "policies": {k: o.as_dict() for k, o in outs.items()}})
    finally:
        con.close()

    if not rows:
        log.info("no trade had a usable path — nothing to conclude")
        return 0

    min_cov = float(getattr(config, "SHADOW_MIN_COVERAGE", 0.60))
    thin = [r for r in rows if r["coverage"] < min_cov]
    tot_real = sum(r["realised_rs"] for r in rows)
    tot_mfe = sum(r["mfe_rs"] for r in rows if r["mfe_rs"] > 0)

    log.info("─" * 72)
    log.info("%d trade(s) over %d session(s) | realised ₹%s | MFE ₹%s | "
             "CAPTURE %s", len(rows), len({r["day"] for r in rows}),
             f"{tot_real:,.0f}", f"{tot_mfe:,.0f}",
             f"{100.0 * tot_real / tot_mfe:.1f}%" if tot_mfe > 0 else "n/a")
    if no_path:
        log.warning("%d trade(s) had no usable path and were EXCLUDED "
                    "(reported, never hidden)", no_path)
    if thin:
        log.warning("%d trade(s) below %.0f%% live-mark coverage — counted "
                    "here, excluded from the verdict. A policy may not be "
                    "promoted on forward-filled corpses.",
                    len(thin), 100 * min_cov)

    verdict = EPS.evaluate(per_trade)
    log.info("─" * 72)
    log.info("%-15s %17s %12s %9s  %s", "policy", "Σ Δ₹ vs as-traded",
             "mean/day", "p(BH)", "verdict")
    for p_ in sorted(verdict.get("policies", []),
                     key=lambda z: -(z.get("total_rs") or -1e18)):
        mean = p_.get("mean", float("nan"))
        mde = p_.get("mde", float("nan"))
        if p_.get("significant") and mean > 0:
            v = "BETTER"
        elif p_.get("significant"):
            v = "WORSE"
        else:
            v = f"indistinguishable (could resolve ₹{mde:,.0f}/day)"
        log.info("%-15s %+17s %+12s %9.3f  %s", p_["policy"],
                 f"{p_.get('total_rs', 0.0):,.0f}", f"{mean:,.0f}",
                 p_.get("p_adj_bh", 1.0), v)
    log.info("─" * 72)
    for k, ok in verdict.get("gates", {}).items():
        log.info("gate %-28s %s", k, "PASS" if ok else "FAIL")

    if a.promote or a.dry_run:
        EPS.promote(verdict, dry_run=a.dry_run)
    else:
        log.info("verdict computed but NOT promoted (pass --promote). "
                 "Winner would be: %s", verdict.get("winner") or "none")

    log.info("These are EXIT counterfactuals on trades already entered. "
             "They cannot say whether an entry was right — only how much "
             "of it was collected.")

    out = {"ts": time.time(), "config_hash": config.CONFIG_HASH,
           "n_trades": len(rows), "n_days": len({r["day"] for r in rows}),
           "n_no_path": no_path, "n_thin_coverage": len(thin),
           "realised_rs": round(tot_real, 2), "mfe_rs": round(tot_mfe, 2),
           "capture_pct": (round(100.0 * tot_real / tot_mfe, 2)
                           if tot_mfe > 0 else None),
           "reconstruction": rec.summary(),
           "verdict": verdict, "trades": rows}
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