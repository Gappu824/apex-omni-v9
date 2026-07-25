"""
APEX OMNI — META LIFT (the last open question for the heuristic path)
=====================================================================
Two measurements now bracket the problem, and this tool closes the gap between
them.

  tools/barrier_sweep.py (2026-07-25, 1,003 signals, 27 days)
      0 of 175 (TP, SL, hold) geometries is positive-expectancy. The BEST
      loses Rs 130/trade. Longer holds lose monotonically more. At TP=SL=0.10
      break-even is 50% and the population wins 29.5% — twenty points short.
      => no geometry rescues the AVERAGE signal.

  forge_report 2026-07-25 (n=947)
      auc_cal 0.5907 (z ~ 2.8 vs 0.5), oof_spread 0.5058.
      => the meta DOES rank signals. Weakly, but not by luck.

The unanswered question is the intersection: a weak ranker applied to a losing
population can still be profitable IF the top slice clears the friction floor.
Averages hide that. This tool cross-tabulates META-SCORE QUANTILE x GEOMETRY
and reports realized win rate and MEAN NET Rs after real costs in every cell.

  * If some (quantile, geometry) cell is positive => a RELATIVE gate ("take
    today's top decile") is justified, and this is the evidence for it. A fixed
    probability bar cannot express that; a quantile gate can.
  * If NO cell is positive => the heuristic path is closed by measurement, not
    opinion, and the forge should be pointed at what does work (the cascade:
    60% win, mean Rs +274, rolling CI lower bound now POSITIVE at +6.35).

FIDELITY
--------
Signal generation, the x-vector, the affordability walk, ASK entry, first-touch
grading and round_trip_costs are the forge's own code, imported not copied. OOF
scores come from `meta_gbm.fit_gbm` itself via the `oof_out` hook, so the
quantiles are the SAME purged day-fold CV the nightly promotion uses — never
in-sample. Alignment between OOF scores and harvested paths is asserted, not
assumed.

This tool writes no model and changes no config. It is a measuring instrument.

  python tools/meta_lift.py --days 27
  python tools/meta_lift.py --days 27 --json logs/meta_lift.json
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                              # noqa: E402
from nightly_forge_v9 import (_Replayer, _kelly_budget,     # noqa: E402
                              round_trip_costs, trading_days)
from core.heuristic_policy import HeuristicPolicy          # noqa: E402
from core import meta_gbm as MG                            # noqa: E402
from simulation.scenario_engine import N as SESSION_N      # noqa: E402

config.setup_logging("meta_lift")
import logging                                             # noqa: E402
log = logging.getLogger("meta_lift")

MAX_PATH_S = 1800


def harvest(con, day: str, budget: float) -> list:
    """One replay -> per signal: forge-identical x-vector + entry + path.

    Mirrors _gen_meta_samples exactly for the x-vector (same frame nodes, same
    t/N, er, capped momentum, direction) so the model we fit here is the model
    the forge fits — only the grading varies.
    """
    rep = _Replayer(con, day, meta=None, cal={}, funnel=None)
    if not rep.ok:
        return []
    pol = HeuristicPolicy()

    def decide(obs, frame, iidx):
        return float(pol.predict(frame)[2 * iidx])

    out = []
    for ev in rep.run(decide):
        if ev[0] != "signal":
            continue
        s = ev[1]
        idx, t, d = s["idx"], s["t"], s["direction"]
        pick = None
        for r in rep.mapper.hierarchy(idx, s["spot"], d):
            k = rep.ti.get(r["token"])
            if k is None or t - rep.last_tick.get(r["token"], -99) > 5:
                continue
            b_, a_ = rep.bidA[k, t], rep.askA[k, t]
            if np.isnan(b_) or np.isnan(a_) or a_ <= 0:
                continue
            if a_ * r["lot"] <= budget:
                pick = (k, float(a_), int(r["lot"]))
                break
        if pick is None:
            continue
        k, e, lot = pick
        seg = rep.bidA[k, t + 1:t + 1 + MAX_PATH_S]
        if seg.size < 60 or np.all(np.isnan(seg)):
            continue
        b0 = s["iidx"] * config.NODES_PER_INDEX
        frame = s["frame"]
        f30 = s["f30"]
        x = np.concatenate([frame[b0], frame[b0 + 1], frame[b0 + 2],
                            [t / SESSION_N, s["er"],
                             math.copysign(min(abs(f30) * 100, 3), f30)
                             if f30 else 0.0,
                             1.0 if d == "CE" else -1.0]]).astype(np.float32)
        out.append({"day": day, "x": x, "e": e, "lot": lot,
                    "path": np.asarray(seg, np.float32)})
    return out


def grade(sig: dict, tp: float, sl: float, hold: int):
    """First-touch net P&L after real costs (identical to barrier_sweep)."""
    e, lot = sig["e"], sig["lot"]
    seg = sig["path"][:hold]
    if seg.size == 0 or np.all(np.isnan(seg)):
        return None
    tpx, slx = e * (1.0 + tp), e * (1.0 - sl)
    ht, hs = seg >= tpx, seg <= slx
    it = int(np.argmax(ht)) if ht.any() else None
    isl = int(np.argmax(hs)) if hs.any() else None
    if it is not None and (isl is None or it < isl):
        px = tpx
    elif isl is not None:
        px = slx
    else:
        v = np.nonzero(~np.isnan(seg))[0]
        if v.size == 0:
            return None
        px = float(seg[v[-1]])
    return float((px - e) * lot - round_trip_costs(e * lot, px * lot))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=27)
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()

    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)[-a.days:]
    budget = _kelly_budget(config.FORGE_EVAL_CAPITAL)
    log.info("meta lift | %d day(s) %s->%s", len(days),
             days[0] if days else "-", days[-1] if days else "-")

    sigs, perday = [], []
    for d in days:
        try:
            got = harvest(con, d, budget)
        except Exception as e:                             # noqa: BLE001
            log.warning("  %s: harvest failed (%s) — skipped", d, e)
            continue
        if not got:
            continue
        # label with the CURRENT geometry so the model we fit is the model the
        # forge fits; the sweep below then re-grades the SAME signals.
        Y = []
        for g in got:
            p = grade(g, config.BASE_TP_PCT, config.BASE_SL_PCT, 900)
            Y.append(1.0 if (p is not None and p > 0) else 0.0)
        sigs += got
        perday.append((d, [g["x"] for g in got], Y, [1.0] * len(got)))
        log.info("  %s: %d signal(s)", d, len(got))

    if len(sigs) < 200:
        log.error("only %d signals — too few for quantile analysis", len(sigs))
        return

    oof: dict = {}
    art = MG.fit_gbm(perday, config.META_MIN_TRAIN, oof_out=oof)
    if art is None or "oof_cal" not in oof:
        log.error("fit_gbm produced no OOF scores (refused or CV too thin) — "
                  "cannot rank without a model. See the reason logged above.")
        return
    mask = oof["mask"]
    if len(mask) != len(sigs):
        log.error("ALIGNMENT BROKEN: %d oof rows vs %d signals — refusing to "
                  "report misaligned quantiles", len(mask), len(sigs))
        return
    scored = [s for s, m in zip(sigs, mask) if m]
    scores = np.asarray(oof["oof_cal"], float)
    log.info("fitted: AUC %.4f | %d of %d signals have OOF scores",
             art.get("auc_cal"), len(scored), len(sigs))

    # geometries: the sweep's best corner, the current setting, and neighbours
    geoms = [(0.10, 0.10, 180), (0.15, 0.10, 180), (0.20, 0.10, 300),
             (0.30, 0.20, 900), (0.30, 0.20, 1800), (0.50, 0.10, 180)]
    qs = [0.50, 0.70, 0.80, 0.90, 0.95]        # keep the TOP (1-q) fraction

    rows, best = [], None
    log.info("")
    log.info("%-22s %-10s %6s %7s %11s %12s", "geometry", "keep-top", "n",
             "win%", "mean Rs", "total Rs")
    log.info("%s", "-" * 74)
    for (tp, sl, hold) in geoms:
        graded = [(grade(s, tp, sl, hold), sc)
                  for s, sc in zip(scored, scores)]
        graded = [(p, sc) for p, sc in graded if p is not None]
        if len(graded) < 100:
            continue
        pnl_all = np.array([p for p, _ in graded], float)
        sc_all = np.array([sc for _, sc in graded], float)
        for q in qs:
            thr = float(np.quantile(sc_all, q))
            sel = sc_all >= thr
            if sel.sum() < 30:
                continue
            pn = pnl_all[sel]
            row = {"tp_pct": tp, "sl_pct": sl, "hold_s": hold,
                   "keep_top_frac": round(1.0 - q, 3),
                   "score_threshold": round(thr, 4),
                   "n": int(pn.size),
                   "win_rate": round(float((pn > 0).mean()), 4),
                   "mean_net_rs": round(float(pn.mean()), 2),
                   "total_net_rs": round(float(pn.sum()), 2),
                   "breakeven_p": round(sl / (tp + sl), 4)}
            rows.append(row)
            if best is None or row["mean_net_rs"] > best["mean_net_rs"]:
                best = row
            log.info("TP%.2f/SL%.2f/%-5d %-10s %6d %6.1f%% %11.2f %12.0f",
                     tp, sl, hold, f"top {100*(1-q):.0f}%", row["n"],
                     100 * row["win_rate"], row["mean_net_rs"],
                     row["total_net_rs"])

    log.info("")
    if best and best["mean_net_rs"] > 0:
        log.info("POSITIVE CELL FOUND: TP %.2f / SL %.2f / hold %ds, keeping "
                 "the top %.0f%% by meta score -> mean Rs %+.2f over %d trades "
                 "(win %.1f%%). A RELATIVE gate can express this; the fixed "
                 "%.2f bar cannot. CANDIDATE ONLY: %d cells were compared, so "
                 "re-run on held-out days before changing anything.",
                 best["tp_pct"], best["sl_pct"], best["hold_s"],
                 100 * best["keep_top_frac"], best["mean_net_rs"], best["n"],
                 100 * best["win_rate"], config.META_ENTRY_P_BAR, len(rows))
    else:
        log.info("NO POSITIVE CELL. The meta ranks (AUC %.4f) but its best "
                 "slice still loses at every geometry: ranking within a "
                 "population whose friction floor is ~2-3%% of premium does "
                 "not reach break-even. The heuristic long-premium path is "
                 "closed by measurement. What is NOT closed: the cascade "
                 "(60%% win, mean Rs +274, rolling CI lo +6.35).",
                 art.get("auc_cal") or float("nan"))
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"days": days, "n_signals": len(sigs), "n_scored": len(scored),
             "auc_cal": art.get("auc_cal"), "cells": rows, "best": best,
             "config_hash": config.CONFIG_HASH}, indent=1), encoding="utf-8")
        log.info("cells -> %s", a.json)


if __name__ == "__main__":
    main()