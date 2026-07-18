"""
APEX OMNI v9.7.1 — TOXICITY VAULT REPORT (measure the trap filter's lift)
=========================================================================
The offline validator proves the LOGIC. This measures the real EDGE on the
operator's vault: across the tape, when the toxicity gate would have BLOCKED a
directional entry (adverse flow / engineered sweep), what was the forward
return of that blocked direction vs an allowed one? If the filter is real,
BLOCKED entries have worse forward returns than ALLOWED ones — it is screening
out losers. If not, this report says so and the operator sets
TOXICITY_GATE_ENABLED=False.

Report only. Writes logs/toxicity_report_<date>.json. Reuses the forge DB
plumbing and the SAME streaming OrderFlowToxicity class the brain runs.

  python tools/toxicity_report.py [--days N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                             # noqa: E402
from core import order_flow as OF                          # noqa: E402
from core.calibration import tox_thresholds                # noqa: E402
from core.diagnostics import _atomic_write_json            # noqa: E402
from nightly_forge_v9 import trading_days, spot_token_for  # noqa: E402

config.setup_logging("toxicity_report")
import logging                                            # noqa: E402
log = logging.getLogger("toxicity_report")

_OH, _OM = (int(x) for x in config.SESSION_OPEN.split(":"))
_OPEN_SOD = _OH * 3600 + _OM * 60


def _fwd_ret(spots, t0, hold):
    if t0 + hold >= len(spots):
        return None
    a, b = spots[t0], spots[t0 + hold]
    if a <= 0 or np.isnan(a) or np.isnan(b):
        return None
    return (b - a) / a


def _run_day(con, day, idx, hold_s):
    tok = spot_token_for(con, day, idx)
    if not tok:
        return []
    rows = list(con.execute(
        "SELECT ltp, bid, ask, bid_qty, ask_qty, vol_delta FROM ticks_v9 "
        "WHERE token=? AND ltp>0 "
        "AND date(ts_local_ms/1000,'unixepoch','localtime')=? ORDER BY ts_ms",
        (tok, day)))
    if len(rows) < 120:
        return []
    spots = np.array([r[0] for r in rows], float)
    eng = OF.OrderFlowToxicity(idx)
    thi, tblk = tox_thresholds(idx)
    out = []
    for t, (ltp, bid, ask, bq, aq, vd) in enumerate(rows):
        v = eng.update(spot=ltp, bid=bid, bid_qty=bq, ask=ask, ask_qty=aq,
                       vol_delta=vd)
        fret = _fwd_ret(spots, t, hold_s)
        if fret is None:
            continue
        # for each candidate direction, would the gate allow it, and what was
        # the forward premium-proxy return of taking it?
        for direction in ("CE", "PE"):
            allow, _ = OF.entry_trap_check(v, direction, tox_block=tblk,
                                           sweep_fade_ok=config.TOX_SWEEP_FADE_OK)
            signed = fret if direction == "CE" else -fret   # long CE gains on +
            out.append((allow, signed))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--hold", type=int,
                    default=int(getattr(config, "RIDE_ER_WINDOW_S", 120) * 2))
    args = ap.parse_args()
    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)
    if args.days > 0:
        days = days[-args.days:]
    log.info("toxicity vault report | %d day(s) %s→%s | horizon %ds",
             len(days), days[0] if days else "-", days[-1] if days else "-",
             args.hold)

    allowed, blocked = [], []
    day_sums = []          # per-day (sum_al, n_al, sum_bl, n_bl) for the CI
    for day in days:
        d_al, d_bl = [], []
        for idx in config.TRADABLE:
            for allow, signed in _run_day(con, day, idx, args.hold):
                (d_al if allow else d_bl).append(signed)
        allowed += d_al
        blocked += d_bl
        day_sums.append((float(np.sum(d_al)), len(d_al),
                         float(np.sum(d_bl)), len(d_bl)))

    if not allowed and not blocked:
        log.warning("no data in window — report skipped")
        return

    al = np.array(allowed, float)
    bl = np.array(blocked, float)
    al_mean = float(al.mean()) if len(al) else 0.0
    bl_mean = float(bl.mean()) if len(bl) else 0.0
    al_wr = float((al > 0).mean()) if len(al) else 0.0
    bl_wr = float((bl > 0).mean()) if len(bl) else 0.0
    # DAY-CLUSTER bootstrap CI on (allowed − blocked) mean forward return.
    # v9.7.1 fix for two real bugs: (1) the old tick-level resample tried to
    # allocate (2000 × n_ticks) int64 — 60 GiB at 4M ticks; (2) np.quantile
    # was called with 5/95 instead of 0.05/0.95, so the CI had never actually
    # computed. Resampling DAYS (the cluster unit) is also statistically more
    # correct: intraday ticks are heavily autocorrelated, so an iid tick
    # bootstrap understates the interval — the same autocorrelation lesson the
    # IC study taught. Memory: 2000×D ints, trivial at any vault size.
    edge = None
    if len(al) > 100 and len(bl) > 100 and len(day_sums) >= 5:
        rng = np.random.default_rng(20260716)
        S = np.array(day_sums, float)                      # (D, 4)
        D = len(S)
        idx = rng.integers(0, D, (2000, D))
        sa, na = S[idx, 0].sum(1), S[idx, 1].sum(1)
        sb, nb = S[idx, 2].sum(1), S[idx, 3].sum(1)
        ok = (na > 0) & (nb > 0)
        diff = sa[ok] / na[ok] - sb[ok] / nb[ok]
        lo, hi = float(np.quantile(diff, 0.05)), float(np.quantile(diff, 0.95))
        edge = {"mean": round(float(diff.mean()), 6), "ci90": [round(lo, 6),
                round(hi, 6)], "filter_helps": lo > 0,
                "method": "day-cluster bootstrap (2000 draws)"}

    rep = {"horizon_s": args.hold, "days": len(days),
           "allowed": {"n": len(al), "mean_ret": round(al_mean, 6),
                       "win_rate": round(al_wr, 4)},
           "blocked": {"n": len(bl), "mean_ret": round(bl_mean, 6),
                       "win_rate": round(bl_wr, 4)},
           "allowed_minus_blocked": edge,
           "verdict": (
               "FILTER HELPS — allowed entries beat blocked ones (keep "
               "TOXICITY_GATE_ENABLED=True)" if edge and edge["filter_helps"]
               else "no significant separation in this window — consider "
               "TOXICITY_GATE_ENABLED=False until it appears"),
           "config_hash": config.CONFIG_HASH, "ts": time.time()}

    log.info("  ALLOWED : n=%d mean ret %+.4f%% win %.1f%%", len(al),
             al_mean * 100, al_wr * 100)
    log.info("  BLOCKED : n=%d mean ret %+.4f%% win %.1f%%", len(bl),
             bl_mean * 100, bl_wr * 100)
    if edge:
        log.info("  allowed−blocked: %+.4f%% CI90 [%+.4f%%,%+.4f%%] → %s",
                 edge["mean"] * 100, edge["ci90"][0] * 100,
                 edge["ci90"][1] * 100,
                 "FILTER HELPS" if edge["filter_helps"] else "not significant")
    log.info("  VERDICT: %s", rep["verdict"])

    out = config.LOG_DIR / f"toxicity_report_{dt.date.today()}.json"
    _atomic_write_json(out, rep)
    log.info("report → %s", out)


if __name__ == "__main__":
    main()