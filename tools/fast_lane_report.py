"""
APEX OMNI v9.7.1 — FAST-LANE EDGE REPORT (does the fast lane actually help?)
===========================================================================
The fast-lane validator proves the LOGIC (10/10). This measures the real EDGE
on the operator's vault: across recorded history, for every entry that WOULD
qualify for the fast lane (conviction ≥ FAST_LANE_CONVICTION), what did the
fast-lane exit realize vs. what the SAME entry realized on the normal 45-min
path?

Why this is an honest counterfactual (not a fabricated one)
-----------------------------------------------------------
On a LIVE fill, once the fast lane exits at +22% in 5 min you never observe
what holding to 45 min would have done — that path is gone, and any "fast lane
made/lost ₹X vs holding" claim on live fills would be a lie. But on a REPLAY
over recorded ticks, BOTH exits are observable on the identical entry: the
forward bid path is in the vault, so we can compute the fast-lane exit AND the
normal barrier exit on the very same premium series.

Crucially, this reuses the FORGE'S OWN grader (`_grade_day`) via a read-only
`on_entry` hook — the entry premium (ASK), the forward bid segment, the lot,
and the normal first-touch outcome are exactly what the forge itself graded.
The report never re-implements entry selection or barrier logic, so its numbers
cannot silently diverge from what the engine actually does. It only re-runs the
EXIT rule on the captured segment.

What it reports
---------------
On the set of qualifying (high-conviction) entries:
  • fast-lane mean return & win-rate  vs  normal-path mean return & win-rate
  • the mean per-trade ₹ difference (fast − normal) with a bootstrap 90% CI
  • a clear verdict: FAST LANE HELPS / HURTS / no separation

Report only. Writes logs/fast_lane_report_<date>.json. The number comes from
the operator's vault — this tool just measures it honestly.

  python tools/fast_lane_report.py [--days N]
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                             # noqa: E402
from core.diagnostics import _atomic_write_json            # noqa: E402
import nightly_forge_v9 as F                               # noqa: E402
from nightly_forge_v9 import (trading_days, evaluate_heuristic,  # noqa: E402
                              round_trip_costs)

config.setup_logging("fast_lane_report")
import logging                                            # noqa: E402
log = logging.getLogger("fast_lane_report")

# the same barrier constants the engine uses
_N_LATE = None


def _fast_lane_exit(info: dict) -> tuple[float, int] | None:
    """Re-run ONLY the fast-lane exit rule on the captured forward bid segment.
    Returns (exit_premium, offset_seconds) or None if the segment can't support
    a decision. Mirrors the manage() predicate exactly:
      • only inside [FAST_LANE_MIN_HOLD_S, FAST_LANE_MAX_HOLD_S],
      • fire at the FIRST second bid ≥ e·(1 + FAST_LANE_TP_PCT),
      • else HAND BACK to the normal outcome (the grader's first-touch exit).
    The arm threshold (FAST_LANE_ARM_PCT) is dominated by the TP for the exit
    decision (TP ≥ arm), so the binding condition is the TP crossing.
    """
    e = info["e"]
    seg = info["seg"]
    if e <= 0 or seg is None or seg.size == 0:
        return None
    lo = int(getattr(config, "FAST_LANE_MIN_HOLD_S", 180))
    hi = int(getattr(config, "FAST_LANE_MAX_HOLD_S", 600))
    tp_pct = float(getattr(config, "FAST_LANE_TP_PCT", 0.22))
    tp_px = e * (1.0 + tp_pct)
    # the segment is indexed from t+1; a fill at offset k means k+1 seconds held.
    # restrict to the fast-lane window [lo, hi] seconds.
    win = seg[lo - 1:hi] if lo - 1 < seg.size else np.array([])
    if win.size == 0:
        return None
    hit = np.nonzero(win >= tp_px)[0]
    if hit.size == 0:
        return None                       # fast-TP never reached → hand back
    off = int(hit[0]) + (lo - 1)
    return float(tp_px), off              # filled at the fast-TP


def _forward_evidence(days: list) -> dict:
    """REALIZED fast-lane firings from the execution ledger.

    AUDIT (2026-07-24): this report replays the HEURISTIC+meta path only, and
    the meta currently blocks that path to ~zero entries — so it printed "no
    firings, cannot assess" for 27 straight days while the fast lane was firing
    profitably in production. Every live entry comes from the CASCADE, which
    bypasses the meta, so the replay is structurally blind to the only trades
    that happen. The ledger is not: it records FAST_LANE_ARMED at entry and a
    SELL_FILL with reason FAST_LANE_TP when it banks. Blending that forward
    evidence follows the same pattern cascade_harness already uses.
    """
    out = {"armed": 0, "fired": 0, "pnl_rs": 0.0, "costs_rs": 0.0,
           "wins": 0, "trades": []}
    path = Path(config.LEDGER_PATH)
    if not path.exists() or not days:
        return out
    try:
        import datetime as _dt
        t0 = _dt.datetime.fromisoformat(days[0]).timestamp()
    except Exception:                                          # noqa: BLE001
        t0 = 0.0
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as f:
            for r in csv.DictReader(f):
                try:
                    ts_ = float(r.get("ts") or 0)
                except ValueError:
                    continue
                if ts_ < t0:
                    continue
                if r.get("event") == "FAST_LANE_ARMED":
                    out["armed"] += 1
                elif (r.get("reason") or "").strip() == "FAST_LANE_TP":
                    pnl_ = float(r.get("pnl") or 0.0)
                    out["fired"] += 1
                    out["pnl_rs"] += pnl_
                    out["costs_rs"] += float(r.get("costs") or 0.0)
                    out["wins"] += int(pnl_ > 0)
                    out["trades"].append({"ts": ts_, "index": r.get("index"),
                                          "symbol": r.get("symbol"),
                                          "pnl_rs": round(pnl_, 2)})
    except Exception as e:                                     # noqa: BLE001
        log.warning("forward-evidence read failed (%s) — replay-only report", e)
    return out


_FL_ART = {}          # per-process: {"meta":…, "cal":…} loaded once


def _fl_day_worker(day: str):
    """Grade one day in a pool worker: own sqlite, artifacts loaded once
    per process, qualifying-entry dicts returned to the parent (small,
    picklable). None ⇒ this day failed; the parent logs and skips it."""
    import sqlite3 as _sq
    try:
        if not _FL_ART:
            try:
                _FL_ART["meta"] = F._eval_meta()
            except Exception:                              # noqa: BLE001
                _FL_ART["meta"] = None
            try:
                _FL_ART["cal"] = F._eval_cal()
            except Exception:                              # noqa: BLE001
                _FL_ART["cal"] = {}
        # v9.9.41 CHEAP PRE-FILTER. A fast-lane entry REQUIRES conviction
        # at or above FAST_LANE_CONVICTION. core.signal_stream already holds
        # every signal's conviction for this session at no cost, so a day
        # whose maximum |conv| never reaches the bar CANNOT produce a
        # qualifying entry and does not need a full replay.
        # The 2026-08-18 run graded all 40 sessions in 22 802 s (6.3 h) to
        # find NINE qualifying entries. This skips only days that are
        # provably empty — it can never hide an entry, because the bar is a
        # necessary condition checked against the same conviction the
        # grading path would compute.
        try:
            from core import signal_stream as _SS
            _st = _SS.load(day)
            if _st is not None and len(_st):
                import numpy as _np
                _bar = float(getattr(config, "FAST_LANE_CONVICTION", 0.8))
                if float(_np.abs(_st.conv).max()) < _bar:
                    return []          # provably no qualifying entry
        except Exception:                                  # noqa: BLE001
            pass                       # no stream -> fall through and grade
        rows: list[dict] = []
        con = _sq.connect(str(config.DB_PATH))
        try:
            evaluate_heuristic(con, day, _FL_ART["meta"], _FL_ART["cal"],
                               on_entry=rows.append)
        finally:
            con.close()
        return rows
    except Exception:                                      # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    args = ap.parse_args()
    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)
    if args.days > 0:
        days = days[-args.days:]
    log.info("fast-lane edge report | %d day(s) %s→%s | conviction bar %.2f",
             len(days), days[0] if days else "-", days[-1] if days else "-",
             config.FAST_LANE_CONVICTION)

    # load meta + calibrator EXACTLY as the forge's promotion-day grading does
    # (line 1945: meta, cal = _eval_meta(), _eval_cal()) — using the real
    # trained artifacts so the replayed decisions match what the engine makes.
    # Fall back to the grader's tolerant defaults if an artifact is absent.
    try:
        meta = F._eval_meta()
    except Exception as e_:                                # noqa: BLE001
        log.warning("meta load failed (%s) — using None (heuristic still "
                    "decides; meta only sharpens win-prob)", e_)
        meta = None
    try:
        cal = F._eval_cal()
    except Exception as e_:                                # noqa: BLE001
        log.warning("calibrator load failed (%s) — using {}", e_)
        cal = {}

    fast_r, norm_r = [], []          # per-trade realized RETURN (fraction)
    fast_pnl, norm_pnl = [], []      # per-trade ₹ (after costs), matched pairs
    qualifying = 0

    def on_entry(info: dict):
        nonlocal qualifying
        # only entries that WOULD qualify for the fast lane
        if abs(info["conv"]) < float(config.FAST_LANE_CONVICTION):
            return
        qualifying += 1
        e, lot = info["e"], info["lot"]
        # normal path: exactly what the grader computed (first-touch)
        n_exitp = info["norm_exitp"]
        n_pnl = info["norm_pnl"]
        # fast path: re-run the fast-lane exit on the same segment
        fx = _fast_lane_exit(info)
        if fx is None:
            # fast lane would NOT have fired → it degrades to the normal path,
            # so the two are identical for this entry (no effect). Skip from the
            # DIFFERENCE set (it contributes zero) but count it for context.
            return
        f_exitp, _off = fx
        outlay = e * lot
        f_pnl = (f_exitp - e) * lot - round_trip_costs(outlay, f_exitp * lot)
        fast_pnl.append(f_pnl)
        norm_pnl.append(n_pnl)
        fast_r.append((f_exitp - e) / e)
        norm_r.append((n_exitp - e) / e)

    # v9.9.3: this loop was the LAST serial monster in the evening — 33
    # full-day gradings x ~9 min = 5.9 h, 58% of the whole 2026-08-01
    # night, while every other replay tool had collapsed to seconds on
    # cache hits. Day gradings are independent; they now run through the
    # repo's map_days pool (own sqlite + artifacts per worker, spawn-
    # safe), and the qualifying entries come back as rows aggregated
    # HERE, in day order, through the same on_entry — byte-identical
    # arithmetic, deterministic order, a fraction of the wall time.
    try:
        from core.parallel_days import map_days
        _res = map_days(_fl_day_worker, list(days), desc="fast-lane day")
        for _day, _rows in zip(days, _res):
            if _rows is None:
                log.warning("  %s: grading failed in worker — skipped", _day)
                continue
            for _info in _rows:
                on_entry(_info)
    except Exception as _pe:                               # noqa: BLE001
        log.warning("parallel grading unavailable (%s) — serial path", _pe)
        for day in days:
            try:
                evaluate_heuristic(con, day, meta, cal, on_entry=on_entry)
            except Exception as e_:                        # noqa: BLE001
                log.warning("  %s: grading failed (%s) — skipped", day, e_)

    n_fired = len(fast_pnl)
    if n_fired == 0:
        log.warning("no qualifying entries where the fast lane would have fired "
                    "in this window (conviction ≥ %.2f + a +%.0f%% move inside "
                    "%d–%ds). Nothing to compare.",
                    config.FAST_LANE_CONVICTION,
                    config.FAST_LANE_TP_PCT * 100,
                    config.FAST_LANE_MIN_HOLD_S, config.FAST_LANE_MAX_HOLD_S)
        fwd = _forward_evidence(days)
        if fwd["fired"]:
            log.info("FORWARD EVIDENCE from the live ledger: %d fast-lane "
                     "firing(s), %d armed | Σ ₹%+.2f (costs ₹%.2f) | %d win(s). "
                     "The replay finds none because it only walks the "
                     "heuristic+meta path, which the meta blocks — live "
                     "entries come from the CASCADE.", fwd["fired"],
                     fwd["armed"], fwd["pnl_rs"], fwd["costs_rs"], fwd["wins"])
            for t_ in fwd["trades"][-5:]:
                log.info("    %s %s  ₹%+.2f", t_["index"], t_["symbol"],
                         t_["pnl_rs"])
            _verdict = (f"replay found no qualifying heuristic entries; "
                        f"{fwd['fired']} REAL firing(s) in the ledger, "
                        f"Σ ₹{fwd['pnl_rs']:+.2f} — forward evidence only, "
                        f"far below any certification floor")
        else:
            _verdict = ("no fast-lane firings in window (replay or ledger) "
                        "— cannot assess")
        rep = {"days": len(days), "qualifying_entries": qualifying,
               "fast_lane_fired": 0, "forward_evidence": fwd,
               "verdict": _verdict,
               "config_hash": config.CONFIG_HASH, "ts": time.time()}
        _atomic_write_json(config.LOG_DIR /
                           f"fast_lane_report_{dt.date.today()}.json", rep)
        return

    fp = np.array(fast_pnl, float)
    npn = np.array(norm_pnl, float)
    diff = fp - npn
    fast_mean_r = float(np.mean(fast_r))
    norm_mean_r = float(np.mean(norm_r))
    fast_wr = float((fp > 0).mean())
    norm_wr = float((npn > 0).mean())

    edge = None
    if n_fired >= 30:
        rng = np.random.default_rng(20260717)
        boot = rng.choice(diff, (2000, n_fired), replace=True).mean(1)
        lo, hi = float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95))  # v9.7.1 fix: quantile takes fractions
        edge = {"mean_diff_rs": round(float(diff.mean()), 2),
                "ci90_rs": [round(lo, 2), round(hi, 2)],
                "helps": lo > 0, "hurts": hi < 0}

    if edge and edge["helps"]:
        verdict = ("FAST LANE HELPS — on qualifying entries it beats holding "
                   "to the 45-min path (keep FAST_LANE_ENABLED=True)")
    elif edge and edge["hurts"]:
        verdict = ("FAST LANE HURTS — it exits winners that the normal path "
                   "would have grown; lower FAST_LANE_TP_PCT or set "
                   "FAST_LANE_ENABLED=False")
    else:
        verdict = ("no significant separation in this window — the fast lane is "
                   "neither helping nor hurting on the evidence so far")

    rep = {"days": len(days), "qualifying_entries": qualifying,
           "fast_lane_fired": n_fired,
           "fast_lane": {"mean_ret": round(fast_mean_r, 4),
                         "win_rate": round(fast_wr, 4),
                         "mean_pnl_rs": round(float(fp.mean()), 2)},
           "normal_path": {"mean_ret": round(norm_mean_r, 4),
                           "win_rate": round(norm_wr, 4),
                           "mean_pnl_rs": round(float(npn.mean()), 2)},
           "fast_minus_normal": edge,
           "verdict": verdict,
           "config_hash": config.CONFIG_HASH, "ts": time.time()}

    log.info("  qualifying entries: %d | fast lane fired: %d", qualifying,
             n_fired)
    log.info("  FAST : mean ret %+.2f%% win %.1f%% (₹%.0f/trade)",
             fast_mean_r * 100, fast_wr * 100, fp.mean())
    log.info("  NORM : mean ret %+.2f%% win %.1f%% (₹%.0f/trade)",
             norm_mean_r * 100, norm_wr * 100, npn.mean())
    if edge:
        log.info("  fast−normal: ₹%+.2f/trade CI90 [%+.2f, %+.2f] → %s",
                 edge["mean_diff_rs"], edge["ci90_rs"][0], edge["ci90_rs"][1],
                 "HELPS" if edge["helps"] else
                 "HURTS" if edge["hurts"] else "no separation")
    log.info("  VERDICT: %s", verdict)

    out = config.LOG_DIR / f"fast_lane_report_{dt.date.today()}.json"
    _atomic_write_json(out, rep)
    log.info("report → %s", out)


if __name__ == "__main__":
    main()