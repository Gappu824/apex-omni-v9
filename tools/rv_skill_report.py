"""
APEX OMNI v9.4 — RV FORECAST-SKILL REPORT (Pillar 3 certifier)
==============================================================
Walk-forward, one-day-ahead exam of the HAR-RV forecaster on YOUR vault:
for each day d (d ≥ 6), fit on days < d, forecast RV_d, score against what
the tape then did. Baselines it must beat: RANDOM WALK (yesterday's RV — the
hardest simple benchmark in the RV literature) and MA5. Losses on the
variance scale via QLIKE (Patton 2011: robust to noisy RV proxies;
QLIKE(F,RV) = RV/F − ln(RV/F) − 1, minimized at F = RV) plus RMSE on log-RV.

PRESPECIFIED acceptance (PROGRAM.md Pillar 3): per index — ≥8 evaluation
days AND mean QLIKE(HAR) < mean QLIKE(RW) AND HAR wins ≥60% of days.
Global ok = every TRADABLE index passes. Certificate:
state/rv_skill_certificate.json (config-hash-stamped, EDGE_CERT_VALID_DAYS
freshness). Until it passes, the forecaster touches NO gate — the brain
shows rv̂/VRP as telemetry only.

Side effects, deliberate: refits the FINAL per-index model on all days
(state/rv_model_{IDX}.json — what tomorrow's brain telemetry loads) and
registers the run in the global trial registry (family "rv").

Run after any close, alongside the harnesses:
    python tools/rv_skill_report.py [--days N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                            # noqa: E402
from core import rv_forecaster as RV                     # noqa: E402
from core import trial_registry as TR                    # noqa: E402
from core.diagnostics import _atomic_write_json          # noqa: E402
from nightly_forge_v9 import trading_days, spot_token_for  # noqa: E402

config.setup_logging("rv_skill")
import logging                                           # noqa: E402
log = logging.getLogger("rv_skill")

_OH, _OM = (int(x) for x in config.SESSION_OPEN.split(":"))
_OPEN_SOD = _OH * 3600 + _OM * 60


def _minute_closes(con, day: str, tok: int) -> np.ndarray:
    arr = np.full(RV.SESSION_MIN, np.nan)
    for ts, ltp in con.execute(
            "SELECT ts_ms/1000, ltp FROM ticks_v9 WHERE token=? AND ltp>0 AND "
            "date(ts_local_ms/1000,'unixepoch','localtime')=? ORDER BY ts_ms",
            (tok, day)):
        m = (int((ts + 19800) % 86400) - _OPEN_SOD) // 60
        if 0 <= m < RV.SESSION_MIN:
            arr[m] = ltp                                  # last tick wins
    return arr


def _qlike(F: float, rv: float) -> float | None:
    if F <= 0 or rv <= 0:
        return None
    x = rv / F
    return float(x - math.log(x) - 1.0)


def _series(con, index: str, days: list[str]):
    """(kept_days, rv[], gaps[], profiles[]) — days without a spot token or
    without variance are dropped with a log line, never silently."""
    kept, rvs, gaps, profs = [], [], [], []
    prev_close = None
    for day in days:
        tok = spot_token_for(con, day, index)
        if not tok:
            log.info("%s %s: no spot token — skipped", index, day)
            continue
        closes = _minute_closes(con, day, tok)
        rv = RV.rv_from_minute_closes(closes)
        if rv <= 0:
            log.info("%s %s: zero variance — skipped", index, day)
            continue
        fin = closes[np.isfinite(closes)]
        gap = (abs(math.log(fin[0] / prev_close))
               if (prev_close and fin.size) else 0.0)
        prev_close = float(fin[-1]) if fin.size else prev_close
        kept.append(day)
        rvs.append(rv)
        gaps.append(gap)
        profs.append(RV.cumulative_profile(closes))
    return kept, rvs, gaps, profs


def _walk_forward(index: str, days, rvs, gaps, profs):
    rows = []
    for d in range(6, len(days)):                          # d ≥ 6
        model = RV.fit_har(rvs[:d], gaps[:d], profs[:d], f"__wf_{index}",
                           days[:d])
        # fit_har writes an artifact even for WF folds — point it at a
        # scratch name; the FINAL refit below owns the real filename.
        if model is None:
            continue
        f_har = RV.predict_next_day(
            model, [math.log(max(x, 1e-12)) for x in rvs[:d]], gaps[d])
        f_rw = rvs[d - 1]
        f_ma5 = float(np.mean(rvs[max(0, d - 5):d]))
        rv_t = rvs[d]
        q_h, q_r, q_m = (_qlike(f_har, rv_t) if f_har else None,
                         _qlike(f_rw, rv_t), _qlike(f_ma5, rv_t))
        if None in (q_h, q_r, q_m):
            continue
        rows.append({"day": days[d], "rv": rv_t, "har": f_har, "rw": f_rw,
                     "ma5": f_ma5, "q_har": q_h, "q_rw": q_r, "q_ma5": q_m,
                     "e_har": (math.log(f_har) - math.log(rv_t)) ** 2,
                     "e_rw": (math.log(f_rw) - math.log(rv_t)) ** 2})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    args = ap.parse_args()
    con = sqlite3.connect(config.DB_PATH)
    days_all = trading_days(con)
    if args.days > 0:
        days_all = days_all[-args.days:]
    per_index, ok_all = {}, True
    for index in config.TRADABLE:
        days, rvs, gaps, profs = _series(con, index, days_all)
        rows = _walk_forward(index, days, rvs, gaps, profs)
        n = len(rows)
        if n:
            mq_h = float(np.mean([r["q_har"] for r in rows]))
            mq_r = float(np.mean([r["q_rw"] for r in rows]))
            mq_m = float(np.mean([r["q_ma5"] for r in rows]))
            wins = sum(1 for r in rows if r["q_har"] < r["q_rw"])
            rmse_h = math.sqrt(float(np.mean([r["e_har"] for r in rows])))
            rmse_r = math.sqrt(float(np.mean([r["e_rw"] for r in rows])))
        else:
            mq_h = mq_r = mq_m = rmse_h = rmse_r = None
            wins = 0
        reasons = []
        if n < 8:
            reasons.append(f"eval days {n} < 8")
        if n and not (mq_h < mq_r):
            reasons.append(f"QLIKE HAR {mq_h:.4f} ≥ RW {mq_r:.4f}")
        if n and wins / max(n, 1) < 0.60:
            reasons.append(f"daily wins {wins}/{n} < 60%")
        ok = not reasons
        ok_all = ok_all and ok
        # FINAL model on all days — tomorrow's live telemetry
        final = RV.fit_har(rvs, gaps, profs, index, days) if rvs else None
        per_index[index] = {
            "ok": ok, "blocked_by": reasons or None, "eval_days": n,
            "qlike_har": mq_h, "qlike_rw": mq_r, "qlike_ma5": mq_m,
            "rmse_log_har": rmse_h, "rmse_log_rw": rmse_r,
            "wins_vs_rw": wins, "series_days": len(days),
            "model_written": bool(final),
            "last_rv_ann": (round(RV.ann_vol(rvs[-1]), 4) if rvs else None)}
        log.info("%s: %s | eval %d | QLIKE HAR %s vs RW %s (MA5 %s) | "
                 "wins %d/%d | logRMSE %s vs %s",
                 index, "SKILL ✓" if ok else "no skill yet", n,
                 f"{mq_h:.4f}" if mq_h is not None else "—",
                 f"{mq_r:.4f}" if mq_r is not None else "—",
                 f"{mq_m:.4f}" if mq_m is not None else "—",
                 wins, n,
                 f"{rmse_h:.3f}" if rmse_h else "—",
                 f"{rmse_r:.3f}" if rmse_r else "—")
        if reasons:
            for r in reasons:
                log.info("  blocked_by: %s", r)
    spec_id = f"har_v1_{config.CONFIG_HASH}"
    TR.register("rv", spec_id, "primary",
                eval_days=sum(v["eval_days"] for v in per_index.values()),
                ok=bool(ok_all))
    cert = {"ok": bool(ok_all), "per_index": per_index, "spec_id": spec_id,
            "criteria": "≥8 eval days AND QLIKE(HAR)<QLIKE(RW) AND wins≥60%,"
                        " every tradable index",
            "family_trials": TR.trials_for_deflation("rv"),
            "config_hash": config.CONFIG_HASH, "ts": time.time()}
    _atomic_write_json(config.STATE_DIR / "rv_skill_certificate.json", cert)
    _atomic_write_json(config.LOG_DIR /
                       f"rv_skill_report_{dt.date.today()}.json",
                       {"certificate": cert})
    # scrub the walk-forward scratch artifacts
    for p in config.STATE_DIR.glob("rv_model___wf_*.json"):
        p.unlink(missing_ok=True)
    log.info("VERDICT: %s | certificate → %s",
             "FORECAST SKILL CERTIFIED ✓" if ok_all
             else "no certificate (forecaster stays telemetry-only)",
             config.STATE_DIR / "rv_skill_certificate.json")


if __name__ == "__main__":
    main()