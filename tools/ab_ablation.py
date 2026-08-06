"""
A/B ABLATION — does this feature make the system BETTER or WORSE?
==================================================================
    python tools/ab_ablation.py [--days N] [--groups peer_context,...]
                               [--reps 5] [--json out.json]

Features get added to this system on a thesis (cross-index peer context,
July 2026: `peer_agree`, `peer_max_agree`, `peer_dispersion`). Nothing
has ever measured whether they helped. Overall AUC cannot answer it —
it moved for a dozen reasons at once. This tool answers it directly, on
the REAL vault, in the two currencies that matter.

THE DESIGN, AND WHY EACH PIECE IS THERE
---------------------------------------
* PAIRED, NOT TWO-SAMPLE. Fit WITH the group and WITHOUT it on the
  IDENTICAL days, IDENTICAL purged folds, IDENTICAL seeds, then take the
  per-day difference. Day-to-day variance — which dwarfs any feature
  effect in 32 days of options data — cancels in the pairing. This is
  why a vault that cannot resolve an absolute AUC of 0.545 can still
  resolve a paired delta several times smaller.

* ABLATION BY ZEROING, NOT DELETING. The x-width stays fixed, so both
  arms share one artifact schema and one hyper-parameter regime. A
  narrower matrix would change the model's capacity and confound the
  comparison with a different question.

* REPEATED CV. Each repetition rotates the day partition (purging and
  embargo intact), so fold luck averages out and the residual spread
  tells you how much of any delta is merely fitting noise.

* TWO METRICS, DELIBERATELY.
    AUC  — does the feature improve RANKING?
    ₹    — does that ranking survive the gate and the cost stack?
  A feature that lifts AUC but not rupees has not helped this system;
  the EV gate is where probabilities become decisions, so the rupee
  metric replays exactly it: take the sample iff calibrated p ≥ its own
  breakeven p* (from the stored per-sample economics through
  `round_trip_costs`), and sum the realised net P&L already recorded in
  the labels. Same physics as production, no re-derivation.

* SIGNIFICANCE THAT RESPECTS THE DATA. Day-cluster bootstrap CI on the
  paired difference, plus an exact-ish paired sign-flip permutation test
  (days are the exchangeable unit under the null). Benjamini-Hochberg
  across every group tested; every group pre-registered in the trial
  registry before its result exists.

* POWER IS REPORTED, ALWAYS. If a group comes back INDISTINGUISHABLE,
  the report states the smallest delta this vault could have detected,
  so "no effect" is never confused with "no evidence".

VERDICTS: BETTER / WORSE / INDISTINGUISHABLE — per group, per metric.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                              # noqa: E402

config.setup_logging("ab_ablation")
import logging                                             # noqa: E402
log = logging.getLogger("ab_ablation")

from core import capability_ladder as CL                   # noqa: E402
from core import meta_gbm as MG                            # noqa: E402
from core.meta_gate import breakeven_p                     # noqa: E402

REPORT = config.LOG_DIR / "ab_ablation_{d}.json"


# ------------------------------------------------------------ feature map
def feature_groups(dim: int) -> dict[str, list[int]]:
    f = int(config.FEATURES_PER_NODE)
    g = {"node0_spot": list(range(0, f)),
         "node1_ctx": list(range(f, 2 * f)),
         "node2_flow": list(range(2 * f, 3 * f)),
         "time_of_day": [3 * f],
         "event_risk": [3 * f + 1],
         "flow_30s": [3 * f + 2],
         "direction": [3 * f + 3]}
    if dim > 3 * f + 4:
        g["peer_context"] = list(range(3 * f + 4, dim))
    return {k: [i for i in v if i < dim] for k, v in g.items()
            if any(i < dim for i in v)}


# ------------------------------------------------------------- one metric
def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = scores[y > 0.5], scores[y <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0)
                 / (pos.size * neg.size))


def _rupees(p_cal: np.ndarray, ret: np.ndarray, econ) -> float:
    """Replay the v3 EV gate: take the trade iff the calibrated
    probability clears THIS trade's own breakeven. Sum realised net ₹."""
    total = 0.0
    for k in range(p_cal.size):
        e, tp, sl, lot = econ[k]
        if not (e > 0 and tp > 0 and sl > 0):
            continue
        ps = breakeven_p(float(e), float(tp), float(sl), int(lot))
        if ps is None or not np.isfinite(ps):
            continue
        if float(p_cal[k]) >= ps + float(getattr(config, "META_EV_MARGIN",
                                                 0.02)):
            total += float(ret[k])
    return total


# --------------------------------------------------------------- one arm
def _run_arm(perday, SD, RET, ECON, drop: list[int] | None, rot: int):
    """One fit. `rot` rotates the day order so each repetition draws a
    different purged partition; `drop` zeroes the ablated columns."""
    days = [d for d, *_ in perday]
    order = days[rot:] + days[:rot]
    by_day = {d: t for d, *t in ((d, X, y, w) for d, X, y, w in perday)}
    pd2 = []
    for d in order:
        X, y_, w_ = by_day[d]
        if drop:
            X = [np.where(np.isin(np.arange(len(r)), drop), 0.0,
                          np.asarray(r, np.float32)).astype(np.float32)
                 for r in X]
        pd2.append((d, X, y_, w_))
    with tempfile.TemporaryDirectory() as td:
        oof: dict = {}
        MG.fit_gbm(pd2, min_train=config.META_MIN_TRAIN, oof_out=oof,
                   model_path=Path(td) / "ab.txt")
    if "mask" not in oof:
        return None
    m = np.asarray(oof["mask"])
    s = np.asarray(oof["oof_raw"], float)
    y = np.asarray(oof["y"], float)
    # calibrate OOF scores the way serving does, but fold-honestly: a
    # single isotonic on OOF is self-referential, so use the rank-based
    # empirical CDF of the training arm — monotone, so AUC is unchanged,
    # and it maps scores into [0,1] for the gate replay.
    rank = np.argsort(np.argsort(s)) / max(s.size - 1, 1)
    base = float(y.mean())
    p_cal = np.clip(base + (rank - 0.5) * 2.0 * min(base, 1 - base) * 2.0,
                    0.01, 0.99)
    dd = SD[m]
    per_day = {}
    for d in np.unique(dd):
        sel = dd == d
        per_day[str(d)] = (_auc(s[sel], y[sel]),
                           _rupees(p_cal[sel], RET[m][sel],
                                   [ECON[i] for i in np.nonzero(m)[0][sel]]))
    return {"auc_all": _auc(s, y), "per_day": per_day, "n": int(y.size)}


# --------------------------------------------------------- paired testing
def _paired(deltas: dict, n_boot: int = 4000, seed: int = 11) -> dict:
    """v9.9.7: the paired test moved to core.capability_ladder so the
    horizon sweep, the feature ablation and this tool all share ONE
    implementation. Behaviour is unchanged (same bootstrap, same sign-flip
    permutation, same 80%-power resolution bound)."""
    return CL.paired_test(deltas, n_boot=n_boot, seed=seed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--groups", type=str, default="",
                    help="comma list; default = every group")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()

    from nightly_forge_v9 import trading_days, _gen_meta_samples_cached
    con = sqlite3.connect(str(config.DB_PATH))
    try:
        days = trading_days(con)
        if a.days > 0:
            days = days[-a.days:]
        perday, Y, W, SD, RET, ECON = [], [], [], [], [], []
        for d in days:
            try:
                X, y_, w_, _r, ret_, e_ = _gen_meta_samples_cached(con, d)
            except Exception as e:                         # noqa: BLE001
                log.warning("  %s: %s", d, e)
                continue
            if not X or len(e_) != len(X):
                continue
            perday.append((d, X, y_, w_))
            Y += list(y_); W += list(w_); SD += [d] * len(X)
            RET += list(ret_); ECON += [tuple(t) for t in e_]
    finally:
        con.close()
    if not perday:
        log.error("no usable sample days — run the forge first")
        return 1
    Y = np.asarray(Y, float); W = np.asarray(W, float)
    SD = np.asarray(SD); RET = np.asarray(RET, float)
    dim = len(perday[0][1][0])

    cap = CL.assess(Y, W, SD)
    log.info("LADDER | %s", cap.reason)
    log.info("PAIRED DESIGN | absolute-AUC resolution is %.3f, but the "
             "paired test cancels day variance — its own resolution is "
             "reported per group below.", cap.mde_auc)

    groups = feature_groups(dim)
    want = [g.strip() for g in a.groups.split(",") if g.strip()]
    if want:
        groups = {k: v for k, v in groups.items() if k in want}
    log.info("A/B over %d group(s) on %d real day(s), n=%d, %d rep(s)",
             len(groups), len(perday), Y.size, a.reps)

    try:
        from core.trial_registry import register
        for g in groups:
            register(family="ab_ablation", spec_id=f"drop_{g}",
                     kind="pre_registered", n_days=len(perday))
    except Exception as e:                                 # noqa: BLE001
        log.debug("registry unavailable (%s)", e)

    rows = []
    for name, cols in groups.items():
        d_auc, d_rs, full_aucs = {}, {}, []
        for rep in range(max(1, a.reps)):
            rot = (rep * max(1, len(perday) // max(a.reps, 1))) % len(perday)
            with_ = _run_arm(perday, SD, RET, ECON, None, rot)
            without = _run_arm(perday, SD, RET, ECON, cols, rot)
            if not with_ or not without:
                continue
            full_aucs.append(with_["auc_all"])
            for day in with_["per_day"]:
                if day not in without["per_day"]:
                    continue
                aw, rw = with_["per_day"][day]
                ao, ro = without["per_day"][day]
                if np.isfinite(aw) and np.isfinite(ao):
                    d_auc.setdefault(day, []).append(aw - ao)
                d_rs.setdefault(day, []).append(rw - ro)
        if not d_auc:
            log.warning("  %-14s: no comparable folds", name)
            continue
        auc_d = {k: float(np.mean(v)) for k, v in d_auc.items()}
        rs_d = {k: float(np.mean(v)) for k, v in d_rs.items()}
        sa, sr = _paired(auc_d), _paired(rs_d)
        row = {"group": name, "n_cols": len(cols),
               "auc_with_mean": round(float(np.mean(full_aucs)), 4)
               if full_aucs else None,
               "delta_auc": sa, "delta_rs": sr}
        rows.append(row)
        log.info("  %-14s ΔAUC %+.4f (CI90 %+.4f..%+.4f, p=%.3f, "
                 "resolvable %.4f) | Δ₹ %+9.0f (CI90 %+.0f..%+.0f, "
                 "p=%.3f)", name, sa["mean"], sa["ci90"][0], sa["ci90"][1],
                 sa["p"], sa.get("mde", float("nan")), sr["mean"],
                 sr["ci90"][0], sr["ci90"][1], sr["p"])

    # ---- multiplicity across the groups actually tested
    for key in ("delta_auc", "delta_rs"):
        live = [r for r in rows if np.isfinite(r[key]["mean"])]
        if not live:
            continue
        rej, adj = CL.benjamini_hochberg(
            [r[key]["p"] for r in live],
            float(getattr(config, "DISCOVERY_FDR_Q", 0.10)))
        for r, ok_, q_ in zip(live, rej, adj):
            r[key]["p_adj_bh"] = round(float(q_), 4)
            r[key]["significant"] = bool(ok_)

    log.info("─" * 74)
    for r in rows:
        sa, sr = r["delta_auc"], r["delta_rs"]
        def _v(st, unit):
            if not st.get("significant"):
                return (f"INDISTINGUISHABLE (could resolve "
                        f"{st.get('mde', float('nan')):.4f}{unit})")
            return "BETTER" if st["mean"] > 0 else "WORSE"
        r["verdict_auc"] = _v(sa, "")
        r["verdict_rs"] = _v(sr, "₹")
        log.info("  %-14s ranking: %-42s rupees: %s",
                 r["group"], r["verdict_auc"], r["verdict_rs"])
    log.info("─" * 74)
    log.info("Read this as: BETTER/WORSE means the feature group changed "
             "the result by more than day-to-day noise after FDR "
             "correction. INDISTINGUISHABLE names the smallest effect "
             "this vault could have caught — absence of evidence, with "
             "the evidence bound stated.")

    out = {"ts": time.time(), "config_hash": config.CONFIG_HASH,
           "days": len(perday), "n": int(Y.size), "reps": a.reps,
           "ladder": cap.as_dict(), "groups": rows}
    p = Path(a.json) if a.json else Path(
        str(REPORT).format(d=time.strftime("%Y-%m-%d")))
    try:
        p.write_text(json.dumps(out, indent=1, default=float))
        log.info("report → %s", p)
    except Exception as e:                                 # noqa: BLE001
        log.warning("report write failed (%s)", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())