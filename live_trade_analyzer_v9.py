"""
APEX OMNI v9 — TRADE ANALYZER → THE JUDGE (audit §10 leap)
==========================================================
v8's analyzer answered "what does the system think right now?" — a question
three other processes already answered. v9 answers the only question that
decides everything: HAS IT BEEN RIGHT, AFTER COSTS?

  verdict  — joins the execution ledger's BUY_FILL/SELL_FILL pairs (fill
             truth, paper or live; in live mode it cross-checks order counts
             against kite.orders()), buckets every closed trade by entry
             conviction, and writes calibration_table.json:
                 {"0.70": [win_rate, n], "0.75": [...], ...}
             That table IS the conviction→probability mapping the brain's
             Kelly sizer reads tomorrow. It also prints the one-page nightly
             verdict: trades, win rate, PnL after fees, fees paid, average
             win/loss, max drawdown, trap-shield saves, blocked entries.
  watch X  — a slim live card for index X off the ring buffer (45 s
             staleness check kept from v8 — it was the right instinct).

Run:  python live_trade_analyzer_v9.py verdict
      python live_trade_analyzer_v9.py watch NIFTY
"""
from __future__ import annotations
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import config


def load_ledger(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verdict(ledger_path: Path = None):
    rows = load_ledger(ledger_path or config.LEDGER_PATH)
    if not rows:
        print("Ledger empty — nothing to judge yet. Run a paper day first.")
        return
    trades, open_by_sym = [], {}
    trap_holds = sum(1 for r in rows if r["event"] == "TRAP_HOLD")
    trap_conf = sum(1 for r in rows if r["event"] == "TRAP_CONFIRMED")
    blocked = sum(1 for r in rows if r["event"] in ("BLOCKED", "SKIP"))
    rejects = sum(1 for r in rows if r["event"] == "REJECT")
    # walk-away (chase-cap NOFILL) evidence: was the slip cap refusing genuine
    # chases (runaway) or possibly too tight (borderline)?
    nofills = [r for r in rows if r["event"] == "NOFILL"]
    wa_run = sum(1 for r in nofills if "runaway" in (r.get("reason") or ""))
    wa_bord = sum(1 for r in nofills if "borderline" in (r.get("reason") or ""))
    for r in rows:
        if r["event"] == "BUY_FILL":
            open_by_sym[r["symbol"]] = r
        elif r["event"] == "SELL_FILL" and r["symbol"] in open_by_sym:
            b = open_by_sym.pop(r["symbol"])
            trades.append({
                "symbol": r["symbol"], "direction": r["direction"],
                "conviction": abs(float(b.get("conviction") or 0)),
                "pnl": float(r.get("pnl") or 0),
                "costs": float(r.get("costs") or 0),
                "reason": r.get("reason", ""),
                "regime": b.get("regime", "") or "—"})
    if not trades:
        print(f"No closed trades. ({blocked} blocked/skipped, "
              f"{rejects} rejects, {trap_holds} trap holds)")
        return

    pnl = [t["pnl"] for t in trades]
    wins = [p for p in pnl if p > 0]
    eq, peak, mdd = 0.0, 0.0, 0.0
    for p in pnl:
        eq += p; peak = max(peak, eq); mdd = min(mdd, eq - peak)
    buckets = defaultdict(list)
    for t in trades:
        w = config.CAL_BUCKET_WIDTH
        b = f"{min(t['conviction'] // w * w, 1 - w):.2f}"
        buckets[b].append(1 if t["pnl"] > 0 else 0)
    table = {b: [round(sum(v) / len(v), 4), len(v)]
             for b, v in sorted(buckets.items())}
    tmp = config.CALIBRATION_TABLE.with_suffix(".tmp")
    tmp.write_text(json.dumps(table, indent=1))
    tmp.replace(config.CALIBRATION_TABLE)

    print("=" * 62)
    print(" APEX v9 NIGHTLY VERDICT — evidence, not vibes "
          f"(config {config.CONFIG_HASH})")
    print("=" * 62)
    print(f" closed trades        : {len(trades)}")
    print(f" win rate             : {len(wins)/len(trades):.1%}")
    print(f" PnL after ALL costs  : ₹{sum(pnl):+,.2f}")
    print(f" fees+taxes paid      : ₹{sum(t['costs'] for t in trades):,.2f}")
    aw = sum(wins) / len(wins) if wins else 0.0
    losses = [p for p in pnl if p <= 0]
    al = sum(losses) / len(losses) if losses else 0.0
    print(f" avg win / avg loss   : ₹{aw:+,.2f} / ₹{al:+,.2f}")
    print(f" max drawdown (intra) : ₹{mdd:,.2f}")
    print(f" trap-shield holds    : {trap_holds}  (confirmed traps: {trap_conf})")
    print(f" entries blocked      : {blocked}   rejects: {rejects}")
    # ---- slip-cap walk-away evidence (tune from data, not theory) ----
    wa_total = wa_run + wa_bord
    if wa_total:
        print("-" * 62)
        print(f" chase-cap walk-aways : {wa_total}  "
              f"({wa_run} runaway / {wa_bord} borderline)")
        bord_frac = wa_bord / wa_total
        if wa_total >= 20 and bord_frac >= 0.40:
            print(f"   → EVIDENCE: {bord_frac:.0%} were BORDERLINE (ask only just "
                  f"past the cap).")
            print(f"     The slip cap ({config.ENTRY_SLIP_CAP_PCT:.2f}) may be "
                  f"slightly tight on genuine fills. Consider widening toward "
                  f"{min(config.ENTRY_SLIP_CAP_PCT + 0.15, 1.0):.2f} and re-checking.")
        elif wa_total >= 20:
            print(f"   → EVIDENCE: {1-bord_frac:.0%} were genuine RUNAWAYS — the "
                  f"cap is correctly refusing chases, not costing real trades.")
            print(f"     Keep the slip cap at {config.ENTRY_SLIP_CAP_PCT:.2f}; "
                  f"widening it would buy tops on moves already over.")
        else:
            print(f"   → too few walk-aways ({wa_total}) to judge the cap yet — "
                  f"need ≥20 for a verdict.")
    print("-" * 62)
    print(" conviction calibration (bucket → win rate, n):")
    for b, (w, n) in table.items():
        marker = " ✓trusted" if n >= config.CAL_MIN_SAMPLES else f" (need ≥{config.CAL_MIN_SAMPLES})"
        print(f"   |conv|≥{b}: {w:.1%}  n={n}{marker}")
    print(f"\n calibration_table.json written → the brain's Kelly sizer "
          f"reads it tomorrow.")
    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t["reason"].split(" ")[0]] += 1
    print(" exit reasons         :", dict(exit_reasons))

    # ---- PER-REGIME breakdown: is each regime helping or hurting? ----
    print("-" * 62)
    print(" performance by REGIME at entry (win rate | avg PnL | n):")
    by_reg = defaultdict(list)
    for t in trades:
        by_reg[t["regime"]].append(t)
    for reg, ts_ in sorted(by_reg.items(), key=lambda kv: -len(kv[1])):
        n = len(ts_)
        wr = sum(1 for t in ts_ if t["pnl"] > 0) / n
        avg = sum(t["pnl"] for t in ts_) / n
        verdict_word = ""
        if n >= 10:
            verdict_word = ("  ← HURTING" if avg < 0 else "  ← helping")
        print(f"   {reg:16s}: {wr:5.0%} win | ₹{avg:+7.1f} avg | n={n}"
              f"{verdict_word if n>=10 else f'  (need ≥10, have {n})'}")

    # ---- PER-EXIT-REASON breakdown: are reversal exits cost-losses? ----
    print(" performance by EXIT reason (avg PnL | avg cost | n):")
    by_exit = defaultdict(list)
    for t in trades:
        by_exit[t["reason"].split(" ")[0]].append(t)
    for rs, ts_ in sorted(by_exit.items(), key=lambda kv: -len(kv[1])):
        n = len(ts_)
        avg = sum(t["pnl"] for t in ts_) / n
        avgc = sum(t["costs"] for t in ts_) / n
        # flag exits that are net-losing AND cost-dominated (churn signature)
        flag = ""
        if n >= 10 and avg < 0 and abs(avg) < avgc:
            flag = "  ← cost-dominated churn"
        print(f"   {rs:20s}: ₹{avg:+7.1f} avg | ₹{avgc:5.1f} cost | n={n}{flag}")


def watch(index: str):
    from apex_ipc_core import BinaryRingBuffer
    ring = BinaryRingBuffer()
    while True:
        state, age = ring.read_state()
        if state is None or age > 45:
            print(f"\r⚠ feed stale ({age:.0f}s) — not trusting it", end="")
            time.sleep(1); continue
        ctx = (state.get("market") or {}).get(index.upper())
        if not ctx:
            print(f"\r{index}: not in ring yet", end=""); time.sleep(1); continue
        sp = ctx.get("spot") or {}
        legs = ctx.get("legs") or {}
        ce = (legs.get("atm_ce") or {}).get("snap") or {}
        pe = (legs.get("atm_pe") or {}).get("snap") or {}
        print(f"\r{index.upper()} {sp.get('ltp', 0):.1f} | ATM {ctx.get('atm')}"
              f" | CE {ce.get('ltp', 0):.2f} ({ce.get('bid',0)}/{ce.get('ask',0)})"
              f" | PE {pe.get('ltp', 0):.2f} ({pe.get('bid',0)}/{pe.get('ask',0)})"
              f" | dte {ctx.get('dte', '?')} | age {age:.1f}s   ",
              end="", flush=True)
        time.sleep(1)


def drift_status():
    """Print the latest live drift assessment (written by the brain)."""
    import json
    try:
        s = json.loads(config.DRIFT_STATE_PATH.read_text())
    except Exception:
        print("No drift assessment yet — run the brain on a live feed first.")
        return
    print("=" * 60)
    print(f" FEATURE DRIFT — grade {s.get('grade')} "
          f"(model {s.get('model_version')})")
    print("=" * 60)
    print(f" live samples {s.get('n_live')} | features considered "
          f"{s.get('features_considered')} | significant {s.get('significant')}"
          f" | moderate {s.get('moderate')}")
    per = s.get("per_feature", {})
    for name, d in sorted(per.items(), key=lambda kv: -kv[1]["psi"]):
        print(f"   {name:14s} PSI {d['psi']:.3f}  KS {d['ks']:.3f}  "
              f"{d['level']:11s} (live μ {d['live_mean']:+.3f} vs ref "
              f"{d['ref_mean']:+.3f})")
    if s.get("grade") == "DRIFTED":
        print(" → LIVE DE-ARMED: regime left the model's training world. "
              "Next nightly forge re-references; paper keeps running.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "verdict"
    if mode == "watch" and len(sys.argv) > 2:
        watch(sys.argv[2])
    elif mode == "drift":
        drift_status()
    else:
        verdict()
