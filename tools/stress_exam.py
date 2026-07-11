"""
APEX OMNI v9.5 — STRESS WORLDS + CPCV (Pillar 5 T3)
===================================================
Three exams of robustness, all on recombined REALITY — never invented prices:

1) STATIONARY-BOOTSTRAP STRESS (Politis–Romano 1994): each engine cert's
   day-PnL series re-bootstrapped in dependence-preserving blocks; the
   resulting CI is stamped INTO the certificate under `stress` — a cert now
   answers "and under serial dependence?" in its own file.

2) REGIME-STITCHED WORLDS: synthetic sessions spliced from two REAL days —
   the morning of a calm day + the afternoon of a violent one (and the
   reverse), every tick and macro row verbatim from the vault; the splice
   discontinuity IS the stress. Both engine harnesses run on each world;
   deltas vs the component days expose path-dependence fragility. Days are
   ranked by their own minute-RV; worlds live in a throwaway sqlite.

3) CPCV (López de Prado ch.12) on the FITTED model family that can afford
   it nightly — the HAR-RV forecaster: every C(N,k) purged/embargoed split,
   fit on train groups, QLIKE on test groups, reassembled paths. (The forge's
   SAC exam uses the same splitter weekly; per-split model training makes it
   a weekend job, not a nightly one — stated, not hidden.)

Run weekly (or after unusual days):  python tools/stress_exam.py [--worlds N]
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
from core.diagnostics import _atomic_write_json          # noqa: E402
from core.robust_stats import stationary_ci_lo, cpcv_splits  # noqa: E402
from core import rv_forecaster as RVF                    # noqa: E402
import tools.cascade_harness as CH                       # noqa: E402
import tools.shortvol_harness as SH                      # noqa: E402
from tools.rv_skill_report import _series, _qlike        # noqa: E402
from nightly_forge_v9 import trading_days, spot_token_for  # noqa: E402

config.setup_logging("stress_exam")
import logging                                           # noqa: E402
log = logging.getLogger("stress")

_OH, _OM = (int(x) for x in config.SESSION_OPEN.split(":"))
_OPEN_SOD = _OH * 3600 + _OM * 60


# ------------------------------------------------ 1. cert stress stamping
def _stamp_stress(fam: str, cert_path, report_glob: str) -> dict | None:
    try:
        cert = json.loads(cert_path.read_text())
    except Exception:                                     # noqa: BLE001
        return None
    reps = sorted(config.LOG_DIR.glob(report_glob))
    if not reps:
        return None
    rep = json.loads(reps[-1].read_text())
    fills = [r for r in rep.get("backtest_events", []) if "pnl" in r] \
        + rep.get("forward_events", [])
    day_pnl: dict[str, float] = {}
    for r in fills:
        day_pnl[r["day"]] = day_pnl.get(r["day"], 0.0) + float(r["pnl"])
    if len(day_pnl) < 3:
        return None
    lo = stationary_ci_lo(list(day_pnl.values()))
    cert["stress"] = {"stationary_ci_lo": round(lo, 2),
                      "block_mean_len": 3, "days": len(day_pnl),
                      "ts": time.time()}
    _atomic_write_json(cert_path, cert)
    log.info("%s cert stress-stamped: stationary CI90 lo ₹%.2f over %d "
             "day-PnLs (iid cert lo: %s)", fam, lo, len(day_pnl),
             cert.get("ci_lo"))
    return cert["stress"]


# ------------------------------------------------ 2. regime-stitched worlds
def _rank_days_by_rv(con, days):
    out = []
    for day in days:
        tok = spot_token_for(con, day, config.TRADABLE[0])
        if not tok:
            continue
        from tools.rv_skill_report import _minute_closes
        rv = RVF.rv_from_minute_closes(_minute_closes(con, day, tok))
        if rv > 0:
            out.append((day, rv))
    return sorted(out, key=lambda x: x[1])


def _build_world(con, dayA: str, dayB: str, split_hm: str) -> sqlite3.Connection:
    """Throwaway db: dayA's ticks/macro BEFORE split_hm + dayB's AFTER,
    every row verbatim, re-dated onto dayA's calendar clock so the harness
    sees one session. Prices are untouched — the splice jump is the stress."""
    hh, mm = (int(x) for x in split_hm.split(":"))
    split_sod = hh * 3600 + mm * 60
    w = sqlite3.connect(":memory:")
    src_ddl = open("data_harvester_v9.py").read()
    w.executescript(src_ddl.split('TICKS_SCHEMA = """')[1].split('"""')[0])
    import macro_gex_v9 as MG
    w.executescript(MG.MACRO_ARCHIVE_SCHEMA)
    baseA = int(dt.datetime.fromisoformat(dayA + "T00:00:00").timestamp())
    baseB = int(dt.datetime.fromisoformat(dayB + "T00:00:00").timestamp())
    shift_ms = (baseA - baseB) * 1000
    for day, cond, sh in ((dayA, "<", 0), (dayB, ">=", shift_ms)):
        for row in con.execute(
                "SELECT * FROM ticks_v9 WHERE "
                "date(ts_local_ms/1000,'unixepoch','localtime')=? AND "
                f"((ts_ms/1000 + 19800) % 86400) {cond} ?",
                (day, split_sod)):
            r = list(row)
            r[0] += sh
            r[1] += sh
            w.execute(f"INSERT OR IGNORE INTO ticks_v9 VALUES "
                      f"({','.join('?' * len(r))})", r)
        for row in con.execute(
                "SELECT * FROM macro_snapshots_v9 WHERE "
                "date(ts_ms/1000,'unixepoch','localtime')=? AND "
                f"((ts_ms/1000 + 19800) % 86400) {cond} ?",
                (day, split_sod)):
            r = list(row)
            r[0] += sh
            w.execute(f"INSERT OR REPLACE INTO macro_snapshots_v9 VALUES "
                      f"({','.join('?' * len(r))})", r)
    # spot_tokens is (snap_date, name, token) with as-of semantics — copy it
    # whole so the world's day resolves exactly as the vault would.
    for row in con.execute("SELECT * FROM spot_tokens"):
        w.execute("INSERT OR REPLACE INTO spot_tokens VALUES (?,?,?)", row)
    w.commit()
    return w


def _run_world(w, day: str, N: int) -> dict:
    rows, upside = CH._run_day(w, day, N, primary_rows=None)
    c_fills = [r["pnl"] for r in rows if "pnl" in r]
    closes, _sk, _bl = SH._run_day(w, day, N, verbose=False)
    s_fills = [r["pnl"] for r in closes]
    return {"cascade_events": len(rows), "cascade_fills": len(c_fills),
            "cascade_pnl": round(float(np.sum(c_fills)), 2) if c_fills else 0.0,
            "shortvol_fills": len(s_fills),
            "shortvol_pnl": round(float(np.sum(s_fills)), 2) if s_fills else 0.0}


# ------------------------------------------------ 3. CPCV on the RV model
def _cpcv_rv(con, days_all) -> dict:
    per_index = {}
    for index in config.TRADABLE:
        days, rvs, gaps, profs = _series(con, index, days_all)
        if len(days) < 10:
            per_index[index] = {"note": f"only {len(days)} usable days"}
            continue
        splits, paths = cpcv_splits(days, n_groups=min(6, len(days) // 2),
                                    k_test=2, embargo=1)
        by_day = {d: i for i, d in enumerate(days)}
        ql = []
        for sp in splits:
            tr = [by_day[d] for d in sp["train_days"]]
            if len(tr) < 6:
                continue
            model = RVF.fit_har([rvs[i] for i in tr], [gaps[i] for i in tr],
                                [None] * len(tr), f"__wf_{index}",
                                sp["train_days"])
            if model is None:
                continue
            for d in sp["test_days"]:
                i = by_day[d]
                if i < 1:
                    continue
                f = RVF.predict_next_day(
                    model, [math.log(max(x, 1e-12)) for x in rvs[:i]],
                    gaps[i])
                if f:
                    ql.append(_qlike(f, rvs[i]))
        per_index[index] = {
            "splits": len(splits), "paths": len(paths),
            "scored": len(ql),
            "cpcv_qlike": (round(float(np.mean(ql)), 4) if ql else None)}
    for p in config.STATE_DIR.glob("rv_model___wf_*.json"):
        p.unlink(missing_ok=True)
    return per_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=2)
    ap.add_argument("--days", type=int, default=0)
    args = ap.parse_args()
    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)
    if args.days > 0:
        days = days[-args.days:]
    from simulation.scenario_engine import N
    out = {"ts": time.time()}
    out["cert_stress"] = {
        "cascade": _stamp_stress("cascade", config.CASCADE_CERT_PATH,
                                 "cascade_harness_report_*.json"),
        "shortvol": _stamp_stress("shortvol", config.SHORTVOL_CERT_PATH,
                                  "shortvol_harness_report_*.json")}
    ranked = _rank_days_by_rv(con, days)
    worlds = []
    if len(ranked) >= 4:
        calm = [d for d, _ in ranked[:args.worlds]]
        wild = [d for d, _ in ranked[-args.worlds:]]
        for a, b in list(zip(calm, wild)) + list(zip(wild, calm)):
            w = _build_world(con, a, b, config.STRESS_SPLIT_HM)
            res = _run_world(w, a, N)
            res.update({"morning": a, "afternoon": b})
            worlds.append(res)
            log.info("WORLD %s→%s | cascade %d ev/%d fills ₹%+.0f | "
                     "shortvol %d fills ₹%+.0f", a, b,
                     res["cascade_events"], res["cascade_fills"],
                     res["cascade_pnl"], res["shortvol_fills"],
                     res["shortvol_pnl"])
    out["stitched_worlds"] = worlds
    out["cpcv_rv"] = _cpcv_rv(con, days)
    for i, r in out["cpcv_rv"].items():
        log.info("CPCV(HAR-RV) %s: %s", i, r)
    _atomic_write_json(config.LOG_DIR /
                       f"stress_report_{dt.date.today()}.json", out)
    log.info("stress report → logs/stress_report_%s.json", dt.date.today())


if __name__ == "__main__":
    main()