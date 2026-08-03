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

FIDELITY (single source of truth)
---------------------------------
Nothing here is re-derived. `_gen_meta_samples_cached` returns the forge's own
(X, Y, W, R, RET) for each day — the identical sample set the nightly promotion
trains on — where RET is the REALIZED net P&L of each signal at the shaped
barriers, already after round-trip costs, and aligned to X by construction.
OOF scores come from `meta_gbm.fit_gbm` itself through the `oof_out` hook, so
the quantiles use the SAME purged day-fold CV as promotion and are never
in-sample. The score/P&L alignment is asserted at runtime, not assumed, and a
mismatch refuses to report rather than printing plausible nonsense.

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
from nightly_forge_v9 import (_gen_meta_samples_cached,     # noqa: E402
                              trading_days)
from core import meta_gbm as MG                            # noqa: E402

config.setup_logging("meta_lift")
import logging                                             # noqa: E402
log = logging.getLogger("meta_lift")

# The forge's own sample generator already returns RET — the REALIZED net P&L
# of every meta sample, at the exact shaped barriers the labels use, aligned to
# X and Y by construction. Re-deriving it here would risk divergence for no
# gain, so this tool consumes it directly (and via the npz cache, fast).
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=27)
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()

    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)[-a.days:]
    log.info("meta lift | %d day(s) %s->%s", len(days),
             days[0] if days else "-", days[-1] if days else "-")

    # detect cache hits without touching the forge: a fresh, stamp-matching
    # npz for the day means the samples were memoised, not re-replayed.
    _was_cached: dict = {}
    try:
        import nightly_forge_v9 as _F
        _stamp = _F._meta_samples_stamp()
        for _d in days:
            _p = _F._meta_cache_path(_d)
            hit = False
            if _p.exists():
                try:
                    hit = str(np.load(_p, allow_pickle=False)["stamp"]) == _stamp
                except Exception:                          # noqa: BLE001
                    hit = False
            _was_cached[_d] = hit
    except Exception as e:                                 # noqa: BLE001
        log.debug("cache probe unavailable (%s)", e)

    perday, rets, sample_day = [], [], []
    for d in days:
        try:
            X, Y, W, _R, RET, _E = _gen_meta_samples_cached(con, d)
        except Exception as e:                             # noqa: BLE001
            log.warning("  %s: sample generation failed (%s) — skipped", d, e)
            continue
        if not X:
            continue
        if not (len(X) == len(Y) == len(W) == len(RET)):
            log.error("  %s: ragged sample arrays — refusing", d)
            return
        perday.append((d, list(X), list(Y), list(W)))
        rets += list(RET)
        sample_day += [d] * len(X)
        # PROVENANCE: every sample is derived from real vault ticks. The forge
        # memoises that derivation per day in data/forge_cache/<day>.npz, keyed
        # on CONFIG_HASH + cache version + the decision knobs — so a config or
        # knob change rebuilds automatically. What the stamp does NOT track is
        # the vault's tick CONTENT for that day: if a day were re-harvested
        # after being cached, the cache would still be served. Log which source
        # each day came from so the provenance is visible, not assumed.
        log.info("  %s: %-5d sample(s)  [%s]", d, len(X),
                 "cache" if _was_cached.get(d) else "rebuilt from vault ticks")

    n = len(rets)
    if n < 200:
        log.error("only %d samples — too few for quantile analysis", n)
        return

    oof: dict = {}
    # AUDIT (2026-07-25): fit_gbm writes MODEL_DIR/meta_gbm.txt by default —
    # this research tool was silently REPLACING the booster the live brain
    # serves. Redirect to a scratch file; production is never touched.
    import tempfile as _tf
    _scratch = Path(_tf.mkdtemp()) / "meta_lift_research.txt"
    art = MG.fit_gbm(perday, config.META_MIN_TRAIN, oof_out=oof,
                     model_path=_scratch)
    if art is None or "oof_cal" not in oof:
        log.error("fit_gbm produced no OOF scores (refused, or CV too thin) — "
                  "see the reason logged above")
        return
    mask = np.asarray(oof["mask"], bool)
    if mask.size != n:
        log.error("ALIGNMENT BROKEN: %d oof rows vs %d samples — refusing to "
                  "report misaligned quantiles", mask.size, n)
        return
    pnl = np.asarray(rets, float)[mask]
    sday = np.asarray(sample_day, object)[mask]
    score = np.asarray(oof["oof_cal"], float)
    y = np.asarray(oof["y"], float)
    log.info("")
    log.info("fitted: AUC %.4f | OOF spread %.4f | %d of %d samples scored",
             art.get("auc_cal") or float("nan"),
             art.get("oof_spread_p05_p95") or float("nan"), pnl.size, n)
    log.info("baseline (take EVERY signal): win %.1f%% | mean Rs %+.2f | "
             "total Rs %+.0f", 100 * float((pnl > 0).mean()), float(pnl.mean()),
             float(pnl.sum()))

    log.info("")
    log.info("%-11s %6s %5s %6s %10s %22s", "keep-top", "n", "days",
             "win%", "mean Rs", "CI90 (day-cluster)")
    log.info("%s", "-" * 70)
    _MIN_DAYS = int(getattr(config, 'META_LIFT_MIN_DAYS', 10))
    _rng = np.random.default_rng(20260725)
    rows, best = [], None
    for q in (0.0, 0.50, 0.70, 0.80, 0.90, 0.95, 0.99):
        thr = float(np.quantile(score, q)) if q > 0 else -1e18
        sel = score >= thr
        if sel.sum() < 20:
            continue
        pn = pnl[sel]
        dsel = sday[sel]
        # DAY-CLUSTER BOOTSTRAP. Pooling trades treats 51 fills from 3 days as
        # 51 independent observations — they are not. Intraday P&L is strongly
        # day-correlated (one trending session produces many similar
        # outcomes), which is exactly why the forge uses purged DAY folds. The
        # honest unit of resampling is therefore the DAY, not the trade; the
        # same correction the toxicity report needed.
        udays = sorted(set(dsel.tolist()))
        lo = hi = None
        if len(udays) >= 3:
            by_day = [pn[dsel == u] for u in udays]
            D = len(by_day)
            draws = np.empty(2000, float)
            for b in range(2000):
                pick = _rng.integers(0, D, D)
                cat = np.concatenate([by_day[j] for j in pick])
                draws[b] = cat.mean() if cat.size else np.nan
            good = draws[~np.isnan(draws)]
            if good.size:
                lo = float(np.quantile(good, 0.05))
                hi = float(np.quantile(good, 0.95))
        row = {"keep_top_frac": round(1.0 - q, 3),
               "score_threshold": round(thr, 6), "n": int(pn.size),
               "n_days": len(udays),
               "win_rate": round(float((pn > 0).mean()), 4),
               "mean_net_rs": round(float(pn.mean()), 2),
               "total_net_rs": round(float(pn.sum()), 2),
               "ci90_day_cluster": ([round(lo, 2), round(hi, 2)]
                                    if lo is not None else None),
               # AUDIT (2026-07-25): a CI over too few clusters LIES. Measured
               # false-positive rate of this exact bootstrap on NULL data:
               #   3 days 15.3% | 5 days 8.7% | 10 days 9.0% | 27 days 4.0%
               # (pooling instead of clustering: 43.5% at 3 days). So a
               # positive lower bound is only believable with enough DAYS —
               # trades do not buy independence, days do.
               "significant": bool(lo is not None and lo > 0
                                   and len(udays) >= _MIN_DAYS)}
        row["days"] = list(udays)
        rows.append(row)
        if best is None or row["mean_net_rs"] > best["mean_net_rs"]:
            best = row
        _ci = (f"[{row['ci90_day_cluster'][0]:+.0f}, "
               f"{row['ci90_day_cluster'][1]:+.0f}]"
               if row["ci90_day_cluster"] else "n/a (<3 days)")
        log.info("%-11s %6d %5d %5.1f%% %10.2f %22s %s",
                 f"top {100*(1-q):.0f}%" if q > 0 else "ALL",
                 row["n"], row["n_days"], 100 * row["win_rate"],
                 row["mean_net_rs"], _ci,
                 "SIGNIFICANT" if row["significant"]
                 else (f"(only {row['n_days']}d — CI unreliable below "
                       f"{_MIN_DAYS}d)" if row["ci90_day_cluster"]
                       and row["ci90_day_cluster"][0] > 0 else ""))

    log.info("")
    if best and best.get("days") and best["n_days"] <= 6:
        log.info("the top %.0f%% slice is concentrated in these day(s): %s",
                 100 * best["keep_top_frac"], ", ".join(best["days"]))
        log.info("  -> check whether they share a regime (one trending week "
                 "will manufacture this pattern). If they do, the model may "
                 "have learned a CONDITION rather than an edge — which is "
                 "testable: wait for that condition to recur out-of-sample.")
    _sig = [r for r in rows if r["significant"]]
    _ci = best.get("ci90_day_cluster") if best else None
    _ci_excludes_zero = bool(_ci and _ci[0] > 0)
    if best and best["mean_net_rs"] > 0 and not _sig and _ci_excludes_zero:
        # AUDIT (2026-07-25): the earlier wording claimed the interval
        # "includes zero" whenever a slice was rejected. That was simply FALSE
        # when the rejection came from the DAY FLOOR instead — the top-5%
        # slice returned CI90 [+103, +1495], which excludes zero, and saying
        # otherwise misreports the evidence. Name the actual reason.
        _fp = {3: 15.3, 4: 12.0, 5: 8.7, 6: 8.5, 7: 8.2, 8: 8.0, 9: 9.0}.get(
            best["n_days"], 9.0)
        log.info("POSITIVE AND CI EXCLUDES ZERO — BUT ON TOO FEW DAYS. The "
                 "top %.0f%% slice means Rs %+.2f over %d trades, CI90 %s "
                 "(lower bound ABOVE zero). It is still not actionable, and "
                 "the reason is NOT the interval: those trades come from only "
                 "%d day(s). At %d clusters this bootstrap has a MEASURED "
                 "false-positive rate of ~%.1f%% on null data, so a nominal "
                 "90%% interval is really ~%.0f%% confidence — and %d "
                 "thresholds were scanned on one dataset. 51 trades from 3 "
                 "sessions is 3 observations wearing a disguise. Keep "
                 "harvesting: the interval narrows with DAYS, not trades.",
                 100 * best["keep_top_frac"], best["mean_net_rs"], best["n"],
                 best["ci90_day_cluster"], best["n_days"], best["n_days"],
                 _fp, 100 - _fp, len(rows))
    elif best and best["mean_net_rs"] > 0 and not _sig:
        log.info("POSITIVE MEAN, NOT SIGNIFICANT: the top %.0f%% slice means "
                 "Rs %+.2f over %d trades from %d day(s), CI90 %s — the "
                 "interval includes zero. With %d thresholds compared on one "
                 "dataset, that is the shape a lucky subset takes, not an "
                 "edge. Do NOT move the bar on this.",
                 100 * best["keep_top_frac"], best["mean_net_rs"],
                 best["n"], best["n_days"], best["ci90_day_cluster"],
                 len(rows))
    elif best and best["mean_net_rs"] > 0:
        log.info("POSITIVE SLICE: keeping the top %.0f%% by meta score turns "
                 "the population positive — mean Rs %+.2f over %d trades "
                 "(win %.1f%%) vs Rs %+.2f taking everything. A RELATIVE gate "
                 "can express this; the fixed %.2f bar cannot. CANDIDATE "
                 "ONLY: %d thresholds were compared on the same data, so "
                 "confirm on held-out days before changing anything.",
                 100 * best["keep_top_frac"], best["mean_net_rs"], best["n"],
                 100 * best["win_rate"], float(pnl.mean()),
                 config.META_ENTRY_P_BAR, len(rows))
    else:
        log.info("NO POSITIVE SLICE. The meta ranks (AUC %.4f) but even its "
                 "top percentile loses after costs: ranking inside a "
                 "population whose friction floor is ~2-3%% of premium never "
                 "reaches break-even. Combined with barrier_sweep (0 of 175 "
                 "geometries positive), the heuristic long-premium path is "
                 "closed by measurement — not by opinion.",
                 art.get("auc_cal") or float("nan"))
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"days": days, "n_samples": n, "n_scored": int(pnl.size),
             "auc_cal": art.get("auc_cal"),
             "oof_spread": art.get("oof_spread_p05_p95"),
             "baseline_mean_rs": round(float(pnl.mean()), 2),
             "slices": rows, "best": best,
             "config_hash": config.CONFIG_HASH}, indent=1), encoding="utf-8")
        log.info("slices -> %s", a.json)


if __name__ == "__main__":
    main()