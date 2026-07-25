"""
APEX OMNI — BARRIER GEOMETRY SWEEP (the question the forge structurally cannot ask)
==================================================================================
The forge trains a meta-labeler GIVEN fixed first-touch barriers. That is an
architectural invariant, and it has a blind spot: the forge can never discover
that the BARRIERS THEMSELVES define an unwinnable game. The evidence says they
currently do —

    payoff b = BASE_TP_PCT / BASE_SL_PCT = 0.30/0.20 = 1.5
    break-even win rate               = 1/(1+b) = 40.0%
    observed base rate (27-day vault) = 26.3%

Under a driftless walk those barriers should be touched TP-first 40% of the time
(SL/(TP+SL)). Observing 26% means the population starts ~14 points underwater
before any signal exists — the cost of paying the ASK and holding long premium
through theta. A meta trained on that population correctly learns "everything
loses" and blocks forever. No amount of extra harvesting fixes a geometry
problem; it just measures the same losing game more precisely.

WHAT THIS TOOL DOES
-------------------
One pass over the vault extracts, for every signal the live HeuristicPolicy
would raise: the ASK actually payable, the lot, and the forward BID path. Those
paths are then graded against MANY (tp_pct, sl_pct, hold_s) geometries — the
expensive replay happens once, the sweep is nearly free.

For each geometry it reports the only metric that settles the question:
MEAN NET P&L PER TRADE after real round-trip costs. Win rate is shown too, but
after-cost expectancy is the verdict — a 70%-win geometry that nets negative is
still a losing game.

FIDELITY (why the numbers can be trusted)
-----------------------------------------
Signals come from the SAME `_Replayer` + `HeuristicPolicy` the forge uses; the
affordability walk, ASK entry, first-touch logic and `round_trip_costs` are the
forge's own functions, imported not reimplemented. Only the barrier geometry
varies. Nothing here writes a model, a config, or a certificate — it is a
measurement instrument, and its output is evidence for a decision you make.

  python tools/barrier_sweep.py --days 12
  python tools/barrier_sweep.py --days 27 --json logs/barrier_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                              # noqa: E402
from nightly_forge_v9 import (_Replayer, _kelly_budget,     # noqa: E402
                              round_trip_costs, trading_days)
from core.heuristic_policy import HeuristicPolicy          # noqa: E402

config.setup_logging("barrier_sweep")
import logging                                             # noqa: E402
log = logging.getLogger("barrier_sweep")

MAX_PATH_S = 1800          # longest hold the grid may ask for


def harvest_paths(con, day: str, budget: float) -> list:
    """One replay of `day` → every affordable signal's (entry ASK, lot, forward
    BID path). Mirrors the forge's own sample generation exactly, minus the
    barriers (which are what we are sweeping)."""
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
            if a_ * r["lot"] <= budget:            # the ASK we would pay
                pick = (k, float(a_), int(r["lot"]))
                break
        if pick is None:
            continue
        k, e, lot = pick
        seg = rep.bidA[k, t + 1:t + 1 + MAX_PATH_S]
        if seg.size < 60 or np.all(np.isnan(seg)):
            continue
        out.append({"day": day, "idx": idx, "dir": d, "e": e, "lot": lot,
                    "path": np.asarray(seg, np.float32)})
    return out


def grade(sig: dict, tp_pct: float, sl_pct: float, hold_s: int) -> tuple | None:
    """First-touch grade of one signal under one geometry. Returns
    (net_pnl, won, exit_offset_s) or None when history cannot price it."""
    e, lot = sig["e"], sig["lot"]
    seg = sig["path"][:hold_s]
    if seg.size == 0 or np.all(np.isnan(seg)):
        return None
    tp, sl = e * (1.0 + tp_pct), e * (1.0 - sl_pct)
    hit_tp = seg >= tp
    hit_sl = seg <= sl
    itp = int(np.argmax(hit_tp)) if hit_tp.any() else None
    isl = int(np.argmax(hit_sl)) if hit_sl.any() else None
    if itp is not None and (isl is None or itp < isl):
        exit_px, off = tp, itp
    elif isl is not None:
        exit_px, off = sl, isl
    else:                                   # neither touched → last valid bid
        valid = np.nonzero(~np.isnan(seg))[0]
        if valid.size == 0:
            return None
        exit_px, off = float(seg[valid[-1]]), int(valid[-1])
    gross = (exit_px - e) * lot
    net = gross - round_trip_costs(e * lot, exit_px * lot)
    return float(net), bool(net > 0), int(off + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=12)
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()

    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)[-a.days:]
    budget = _kelly_budget(config.FORGE_EVAL_CAPITAL)
    log.info("barrier sweep | %d day(s) %s->%s | reference budget Rs %.0f",
             len(days), days[0] if days else "-", days[-1] if days else "-",
             budget)

    sigs: list = []
    for d in days:
        try:
            got = harvest_paths(con, d, budget)
        except Exception as e:                             # noqa: BLE001
            log.warning("  %s: harvest failed (%s) — skipped", d, e)
            continue
        sigs += got
        log.info("  %s: %d signal path(s)", d, len(got))
    if len(sigs) < 50:
        log.error("only %d signals — too few to compare geometries. Harvest "
                  "more days or widen --days.", len(sigs))
        return
    log.info("harvested %d signal paths; sweeping geometries...", len(sigs))

    # the grid: current setting is (0.30, 0.20); we probe around and beyond it
    tps = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    sls = [0.10, 0.15, 0.20, 0.25, 0.30]
    holds = [180, 300, 600, 900, 1800]

    rows = []
    for tp in tps:
        for sl in sls:
            for h in holds:
                res = [g for g in (grade(s, tp, sl, h) for s in sigs)
                       if g is not None]
                if len(res) < 50:
                    continue
                pnl = np.array([r[0] for r in res], float)
                won = np.array([r[1] for r in res], bool)
                off = np.array([r[2] for r in res], float)
                rows.append({
                    "tp_pct": tp, "sl_pct": sl, "hold_s": h,
                    "n": int(pnl.size),
                    "win_rate": round(float(won.mean()), 4),
                    "mean_net_rs": round(float(pnl.mean()), 2),
                    "total_net_rs": round(float(pnl.sum()), 2),
                    "median_hold_s": int(np.median(off)),
                    "breakeven_p": round(sl / (tp + sl), 4)})
    rows.sort(key=lambda r: r["mean_net_rs"], reverse=True)

    cur = next((r for r in rows
                if abs(r["tp_pct"] - config.BASE_TP_PCT) < 1e-9
                and abs(r["sl_pct"] - config.BASE_SL_PCT) < 1e-9), None)
    log.info("")
    log.info("%-6s %-6s %-7s %6s %8s %11s %12s", "TP", "SL", "hold",
             "n", "win%", "mean Rs", "total Rs")
    log.info("%s", "-" * 62)
    for r in rows[:12]:
        log.info("%-6.2f %-6.2f %-7d %6d %7.1f%% %11.2f %12.0f",
                 r["tp_pct"], r["sl_pct"], r["hold_s"], r["n"],
                 100 * r["win_rate"], r["mean_net_rs"], r["total_net_rs"])
    log.info("")
    if cur:
        log.info("CURRENT geometry (TP %.2f / SL %.2f): win %.1f%%, "
                 "mean Rs %+.2f per trade over %d signals",
                 cur["tp_pct"], cur["sl_pct"], 100 * cur["win_rate"],
                 cur["mean_net_rs"], cur["n"])
    best = rows[0] if rows else None
    if best and best["mean_net_rs"] > 0:
        log.info("BEST geometry is POSITIVE-EXPECTANCY: TP %.2f / SL %.2f / "
                 "hold %ds -> mean Rs %+.2f per trade (win %.1f%%, n=%d). "
                 "This is a candidate, NOT a decision: re-run on held-out days "
                 "before changing config — %d geometries were compared, so the "
                 "best is selection-biased.", best["tp_pct"], best["sl_pct"],
                 best["hold_s"], best["mean_net_rs"], 100 * best["win_rate"],
                 best["n"], len(rows))
    else:
        log.info("NO geometry in the grid is positive-expectancy. That is a "
                 "real finding: with ASK entry, these costs and this signal "
                 "population, long-premium first-touch trading does not pay at "
                 "any (TP, SL, hold) tested. More harvesting will not change "
                 "it — the entry rule or the instrument must change.")
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"days": days, "n_signals": len(sigs), "grid": rows,
             "current": cur, "config_hash": config.CONFIG_HASH}, indent=1),
            encoding="utf-8")
        log.info("grid -> %s", a.json)


if __name__ == "__main__":
    main()