"""
APEX OMNI — CROSS-INDEX OVERLAP (is n=1003 really n=1003?)
==========================================================
THE QUESTION THIS ANSWERS

Every gate in this system rests on sample counts: "events 11 < 20",
"event-days 4 < 6", the meta's day-clustered CI, the 3-day lift slice. All of
that arithmetic assumes the samples are INDEPENDENT observations.

They may not be. NIFTY and SENSEX are both large-cap Indian equity indices and
move together ~0.98; NIFTY and BANKNIFTY ~0.85-0.90. If the heuristic fires on
NIFTY and SENSEX in the same second, in the same direction, on a move that is
one market event, that is ONE observation recorded as TWO.

And the forge's AFML uniqueness weighting cannot see it:

    for idx in config.TRADABLE:                       # <- per index
        rows = [j for j, (i2, _, _) in enumerate(spans) if i2 == idx]

Concurrency is counted WITHIN an index. Uniqueness weighting exists precisely
to stop overlapping labels being double-counted, and it is blind to the largest
overlap in the data. Two simultaneous NIFTY/SENSEX labels both get weight ~1.0.

WHAT IT MEASURES (all from the vault, nothing assumed)
  1. CO-FIRING RATE  — what fraction of signals have a same-direction signal on
     another index within +/- WINDOW seconds.
  2. OUTCOME AGREEMENT — when they co-fire, do they win/lose TOGETHER? Compared
     against the agreement you would see by chance at the observed base rate.
  3. INTRA-CLUSTER CORRELATION rho, from that comparison.
  4. EFFECTIVE SAMPLE SIZE via the Kish design effect:
         deff  = 1 + (mean_cluster_size - 1) * rho
         n_eff = n / deff
     If n_eff is materially below n, every CI in the system is too narrow and
     every sample floor is further away than it looks.

This is a measuring instrument. It writes no model, changes no config, and
touches nothing the live system reads.

  python tools/cross_index_overlap.py --days 29
  python tools/cross_index_overlap.py --days 29 --window 5 --json logs/xindex.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                              # noqa: E402
from nightly_forge_v9 import (_Replayer, _kelly_budget,     # noqa: E402
                              _shaped_barriers, _hold_seconds,
                              round_trip_costs, trading_days)
from core.heuristic_policy import HeuristicPolicy          # noqa: E402
from simulation.scenario_engine import N as SESSION_N      # noqa: E402

config.setup_logging("xindex")
import logging                                             # noqa: E402
log = logging.getLogger("xindex")


def harvest(con, day: str, budget: float) -> list:
    """One replay -> every signal as (index, t, direction, won). Mirrors the
    forge's own generation: same _Replayer, same HeuristicPolicy, same
    affordability walk, same shaped barriers, same costs."""
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
                pick = (k, float(a_), int(r["lot"]), float(r["strike"]))
                break
        if pick is None:
            continue
        k, e, lot, Kst = pick
        dte = float(s["ctx"].get("dte", 9.0))
        T_ = float(s["ctx"].get("T") or 0.0)
        # SESSION_N, not rep.N — _Replayer does not expose it; the forge
        # imports the same constant inside its own functions.
        mins_left = max((SESSION_N - t) / 60.0, 1.0)
        try:
            tp, sl = _shaped_barriers(e, s["spot"], Kst, T_, mins_left,
                                      d == "CE")
        except Exception:                                  # noqa: BLE001
            tp = e * (1 + config.BASE_TP_PCT)
            sl = e * (1 - config.BASE_SL_PCT)
        seg = rep.bidA[k, t + 1:t + 1 + _hold_seconds(dte)]
        if seg.size == 0 or np.all(np.isnan(seg)):
            continue
        ht, hs = seg >= tp, seg <= sl
        it = int(np.argmax(ht)) if ht.any() else None
        isl = int(np.argmax(hs)) if hs.any() else None
        if it is not None and (isl is None or it < isl):
            px = float(tp)
        elif isl is not None:
            px = float(sl)
        else:
            v = np.nonzero(~np.isnan(seg))[0]
            px = float(seg[v[-1]]) if v.size else e
        pnl = (px - e) * lot - round_trip_costs(e * lot, px * lot)
        out.append({"day": day, "idx": idx, "t": int(t), "dir": d,
                    "won": bool(pnl > 0), "pnl": float(pnl)})
    return out


def analyse(sigs: list, window: int) -> dict:
    """Cluster same-direction signals that fire within `window` seconds of each
    other on DIFFERENT indices, then size the duplication."""
    by_day = defaultdict(list)
    for s in sigs:
        by_day[s["day"]].append(s)

    clusters: list = []
    co_fired = 0
    pair_agree = pair_total = 0
    for day, rows in by_day.items():
        rows.sort(key=lambda r: r["t"])
        used = [False] * len(rows)
        for i, a in enumerate(rows):
            if used[i]:
                continue
            grp = [i]
            used[i] = True
            for j in range(i + 1, len(rows)):
                b = rows[j]
                if b["t"] - a["t"] > window:
                    break
                if used[j] or b["idx"] == a["idx"] or b["dir"] != a["dir"]:
                    continue
                grp.append(j)
                used[j] = True
            if len(grp) > 1:
                co_fired += len(grp)
                for x in range(len(grp)):
                    for y in range(x + 1, len(grp)):
                        pair_total += 1
                        pair_agree += int(rows[grp[x]]["won"]
                                          == rows[grp[y]]["won"])
            clusters.append([rows[g] for g in grp])

    n = len(sigs)
    n_clusters = len(clusters)
    sizes = np.array([len(c) for c in clusters], float) if clusters else \
        np.array([1.0])
    m_bar = float(sizes.mean())
    base = float(np.mean([s["won"] for s in sigs])) if sigs else 0.0
    # agreement expected by chance at this base rate
    a_exp = base ** 2 + (1.0 - base) ** 2
    a_obs = (pair_agree / pair_total) if pair_total else a_exp
    # intra-cluster correlation (Cohen-style excess agreement)
    rho = ((a_obs - a_exp) / (1.0 - a_exp)) if a_exp < 1.0 else 0.0
    rho = max(0.0, min(1.0, rho))
    deff = 1.0 + (m_bar - 1.0) * rho          # Kish design effect
    n_eff = n / deff if deff > 0 else n
    return {"n_signals": n, "n_clusters": n_clusters,
            "co_fired_signals": co_fired,
            "co_fire_rate": round(co_fired / n, 4) if n else 0.0,
            "mean_cluster_size": round(m_bar, 4),
            "base_rate": round(base, 4),
            "agreement_observed": round(a_obs, 4),
            "agreement_by_chance": round(a_exp, 4),
            "intra_cluster_rho": round(rho, 4),
            "design_effect": round(deff, 4),
            "n_effective": int(round(n_eff)),
            "window_s": window}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=12,
                    help="each day is a FULL replay (~4 min); 12 days "
                         "gives ~400 signals, enough to measure "
                         "co-firing and rho")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()

    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)[-a.days:]
    budget = _kelly_budget(config.FORGE_EVAL_CAPITAL)
    log.info("cross-index overlap | %d day(s) %s->%s | tradable %s | "
             "co-fire window +/-%ds", len(days), days[0] if days else "-",
             days[-1] if days else "-", ",".join(config.TRADABLE), a.window)
    log.info("  each day is a full replay (~4 min) — expect roughly %d min",
             4 * len(days))

    sigs = []
    for d in days:
        try:
            got = harvest(con, d, budget)
        except Exception as e:                             # noqa: BLE001
            log.warning("  %s: harvest failed (%s) — skipped", d, e)
            continue
        sigs += got
        if got:
            per = defaultdict(int)
            for s in got:
                per[s["idx"]] += 1
            log.info("  %s: %4d signal(s)  %s", d, len(got),
                     " ".join(f"{k}:{v}" for k, v in sorted(per.items())))
    if len(sigs) < 100:
        log.error("only %d signals — too few to measure overlap", len(sigs))
        return

    res = analyse(sigs, a.window)
    log.info("")
    log.info("signals %d across %d cluster(s)", res["n_signals"],
             res["n_clusters"])
    log.info("  co-fired (same direction, another index, +/-%ds): %d (%.1f%%)",
             a.window, res["co_fired_signals"], 100 * res["co_fire_rate"])
    log.info("  mean cluster size            : %.3f", res["mean_cluster_size"])
    log.info("  outcome agreement observed   : %.4f", res["agreement_observed"])
    log.info("  agreement expected by chance : %.4f",
             res["agreement_by_chance"])
    log.info("  intra-cluster correlation rho: %.4f", res["intra_cluster_rho"])
    log.info("  design effect 1+(m-1)rho     : %.4f", res["design_effect"])
    log.info("")
    log.info("  EFFECTIVE SAMPLE SIZE: %d  (nominal %d)", res["n_effective"],
             res["n_signals"])
    log.info("")
    shrink = 1.0 - res["n_effective"] / max(res["n_signals"], 1)
    if shrink >= 0.15:
        log.warning("SAMPLES ARE %.0f%% REDUNDANT. Every count this system "
                    "gates on is overstated by roughly that much, and every "
                    "confidence interval is too NARROW by ~sqrt(deff) = %.2fx. "
                    "The fix is to extend the forge's AFML uniqueness "
                    "concurrency ACROSS indices (it currently loops per index), "
                    "so simultaneous NIFTY/SENSEX/BANKNIFTY labels share weight "
                    "instead of each counting as one. Measure first, then "
                    "change one thing.", 100 * shrink,
                    res["design_effect"] ** 0.5)
    else:
        log.info("Overlap is small (%.0f%% redundancy). The indices fire "
                 "largely independently at this window, so the existing counts "
                 "and CIs are close to honest. No change warranted on this "
                 "evidence.", 100 * shrink)
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"days": days, "tradable": list(config.TRADABLE), **res,
             "config_hash": config.CONFIG_HASH}, indent=1), encoding="utf-8")
        log.info("-> %s", a.json)


if __name__ == "__main__":
    main()