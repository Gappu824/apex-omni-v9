"""
APEX OMNI v9 — EDGE AUDITOR (the SEBI lock)
===========================================
    python -m core.edge_audit          (also runs inside run_nightly)

SEBI's studies are the most important research in Indian retail derivatives:
~93% of individual F&O traders lose money; only ~1% clear ₹1 lakh after
costs; the profits accrue to algorithmic entities; and >75% of losers keep
trading anyway. That last finding is a psychology bug this module makes
structurally impossible: the system may not arm LIVE_FIRE until ITS OWN
ledger — real fills, real costs — clears a statistical bar:

  * ≥ EDGE_MIN_TRADES closed trades across ≥ EDGE_MIN_DAYS distinct sessions
  * bootstrap (EDGE_BOOTSTRAP_N resamples) lower CI bound of mean per-trade
    after-cost PnL > 0  — not "felt profitable": provably so at 95%
  * positive total expectancy and a sane daily Sharpe estimate (reported)

Pass → writes state/edge_certificate.json (valid EDGE_CERT_VALID_DAYS days,
so a decaying edge silently de-arms the system). Fail → it tells you exactly
how far you are, in numbers, and live stays impossible. The certificate is
evidence, not permission to be reckless — it expires, and the drawdown
governor still rules every live day.
"""
from __future__ import annotations
import csv
import datetime as dt
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import config


def _closed_trades(ledger: Path):
    if not ledger.exists():
        return []
    rows = list(csv.DictReader(ledger.open(encoding="utf-8")))
    out, opens = [], {}
    for r in rows:
        if r["event"] == "BUY_FILL":
            opens[r["symbol"]] = r
        elif r["event"] == "SELL_FILL" and r["symbol"] in opens:
            b = opens.pop(r["symbol"])
            try:
                day = dt.datetime.fromtimestamp(float(r["ts"])).date()
            except Exception:                          # noqa: BLE001
                day = None
            out.append({"pnl": float(r.get("pnl") or 0), "day": str(day)})
    return out


def audit_and_certify(ledger: Path | None = None, write: bool = True) -> dict:
    trades = _closed_trades(Path(ledger or config.LEDGER_PATH))
    pnl = np.array([t["pnl"] for t in trades], float)
    days = {t["day"] for t in trades if t["day"]}
    res = {"ts": time.time(), "n": int(len(pnl)), "days": len(days),
           "ok": False, "config_hash": config.CONFIG_HASH}
    if len(pnl) == 0:
        res["verdict"] = "no closed trades yet — paper trade first"
        return _emit(res, write)
    res["total"] = float(pnl.sum())
    res["expectancy"] = float(pnl.mean())
    res["win_rate"] = float((pnl > 0).mean())
    by_day = defaultdict(float)
    for t in trades:
        by_day[t["day"]] += t["pnl"]
    dd = np.array(list(by_day.values()), float)
    res["sharpe_daily"] = float(dd.mean() / dd.std() * np.sqrt(252)) \
        if len(dd) > 2 and dd.std() > 0 else None

    rng = np.random.default_rng(7)
    boots = rng.choice(pnl, size=(config.EDGE_BOOTSTRAP_N, len(pnl)),
                       replace=True).mean(axis=1)
    lo = float(np.quantile(boots, (1 - config.EDGE_CI) / 2))
    hi = float(np.quantile(boots, 1 - (1 - config.EDGE_CI) / 2))
    res["ci"] = [round(lo, 2), round(hi, 2)]

    fails = []
    if len(pnl) < config.EDGE_MIN_TRADES:
        fails.append(f"trades {len(pnl)}/{config.EDGE_MIN_TRADES}")
    if len(days) < config.EDGE_MIN_DAYS:
        fails.append(f"sessions {len(days)}/{config.EDGE_MIN_DAYS}")
    if lo <= 0:
        fails.append(f"{config.EDGE_CI:.0%} CI lower bound "
                     f"₹{lo:.2f} ≤ 0 (mean ₹{pnl.mean():.2f})")
    if pnl.mean() <= 0:
        fails.append("negative expectancy")
    res["ok"] = not fails
    res["verdict"] = ("EDGE PROVEN — certificate issued (valid "
                      f"{config.EDGE_CERT_VALID_DAYS} days)") if res["ok"] \
        else "NOT PROVEN: " + "; ".join(fails)
    return _emit(res, write)


def progress_line(res: dict) -> str:
    """One-line tally of distance to the Edge Certificate — each of the three
    gates marked ✓ (cleared) or ✗ (blocking)."""
    if not res.get("n"):
        return ("📊 progress to certificate: trades 0/"
                f"{config.EDGE_MIN_TRADES} ✗ | sessions 0/"
                f"{config.EDGE_MIN_DAYS} ✗ | edge unmeasured — paper-trade first")
    n, d = res["n"], res["days"]
    lo = res["ci"][0]
    t_ok = "✓" if n >= config.EDGE_MIN_TRADES else "✗"
    s_ok = "✓" if d >= config.EDGE_MIN_DAYS else "✗"
    e_ok = "✓" if lo > 0 else "✗"
    edge = (f"CI lower ₹{lo:+.2f} >0 {e_ok}" if lo > 0
            else f"CI lower still ₹{lo:+.2f} {e_ok}")
    status = "ALL CLEAR → certified" if res.get("ok") else "not yet"
    return (f"📊 progress to certificate: trades {n}/{config.EDGE_MIN_TRADES} "
            f"{t_ok} | sessions {d}/{config.EDGE_MIN_DAYS} {s_ok} | "
            f"{edge} | {status}")


def _emit(res: dict, write: bool) -> dict:
    print("=" * 64)
    print(" EDGE AUDIT — can this account statistically claim an edge?")
    print("=" * 64)
    print(f" trades {res['n']} over {res['days']} session(s)"
          f" | config {res['config_hash']}")
    if res["n"]:
        print(f" total ₹{res['total']:+,.2f} | expectancy ₹"
              f"{res['expectancy']:+.2f}/trade | win {res['win_rate']:.1%}")
        print(f" bootstrap {config.EDGE_CI:.0%} CI of mean: "
              f"₹{res['ci'][0]:+.2f} … ₹{res['ci'][1]:+.2f}"
              f" | daily Sharpe ≈ {res['sharpe_daily']}")
    print(f" → {res['verdict']}")
    print(" " + progress_line(res))
    if write:
        tmp = config.EDGE_CERT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1))
        tmp.replace(config.EDGE_CERT_PATH)
    return res


if __name__ == "__main__":
    audit_and_certify()
