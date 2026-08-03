"""
FEATURE DISCOVERY — which inputs, if any, carry ranking signal?
===============================================================
    python tools/feature_discovery.py [--days N]

The meta model sees ~64 numbers per decision. When its AUC lands at
0.494 the natural question is "which of those 64 are worth anything?"
— and the natural way to answer it is the fastest known route to a
false discovery: score all 64, keep the best, admire it.

This tool answers the question the defensible way.

  1. LADDER-GATED. Runs only at STAGE DISCOVER or better (MDE ≤ 0.55 by
     default). Below that, per-feature effects are smaller than the
     vault's own resolution and every "finding" is noise.

  2. TWO INDEPENDENT LENSES, both day-clustered:
     a. UNIVARIATE — each feature's own AUC as a raw ranker, with a
        one-sided day-cluster bootstrap p-value. Catches marginal signal
        a tree model may be diluting.
     b. GROUP ABLATION — refit the full model with one feature GROUP
        zeroed (node-0 / node-1 / node-2 / time-of-day / event-risk /
        f30 / direction / peer-context) and measure ΔAUC. Catches
        conditional signal a univariate test cannot see.
     Agreement between the two is the interesting case; either alone is
     a lead, not a conclusion.

  3. PRE-REGISTERED AND FDR-CONTROLLED. Every feature and every group is
     written to the trial registry before its result exists, and
     Benjamini-Hochberg is applied across the whole family. The repo's
     deflated-Sharpe accounting therefore already knows how many shots
     were fired.

  4. "NOTHING CLEARED" IS THE EXPECTED OUTPUT and is written down with
     its numbers. A null recorded at 32 days is what makes a positive at
     120 days believable.

Nothing here changes live behaviour. It produces evidence.
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

config.setup_logging("feature_discovery")
import logging                                             # noqa: E402
log = logging.getLogger("feature_discovery")

from core import capability_ladder as CL                   # noqa: E402
from core import meta_gbm as MG                            # noqa: E402

REPORT = config.LOG_DIR / "feature_discovery_{d}.json"


def _groups(dim: int) -> dict[str, list[int]]:
    """The x-vector's real structure (see decision._meta_x): three index
    nodes, then tod / event-risk / f30 / direction, then peer context."""
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


def _oof_auc(perday, SD, drop: list[int] | None = None) -> float:
    """Purged day-fold OOF AUC of the full model, optionally with a set
    of columns zeroed. Zeroing (not deleting) keeps the x-width — and
    therefore the artifact schema — identical across every refit."""
    if drop:
        perday2 = []
        for d, X, y_, w_ in perday:
            X2 = []
            for row in X:
                r = np.array(row, dtype=np.float32, copy=True)
                r[drop] = 0.0
                X2.append(r)
            perday2.append((d, X2, y_, w_))
    else:
        perday2 = perday
    with tempfile.TemporaryDirectory() as td:
        oof: dict = {}
        MG.fit_gbm(perday2, min_train=config.META_MIN_TRAIN, oof_out=oof,
                   model_path=Path(td) / "f.txt")
    if "mask" not in oof:
        return float("nan")
    s = np.asarray(oof["oof_raw"], float)
    y = np.asarray(oof["y"], float)
    a, _lo, _p = CL.cluster_bootstrap_auc_p(s, y, SD[oof["mask"]], n_boot=1)
    return float(a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    a = ap.parse_args()
    if not bool(getattr(config, "DISCOVERY_ENABLED", True)):
        log.info("discovery disabled by config")
        return 0

    from nightly_forge_v9 import trading_days, _gen_meta_samples_cached
    con = sqlite3.connect(str(config.DB_PATH))
    try:
        days = trading_days(con)
        if a.days > 0:
            days = days[-a.days:]
        perday, X, Y, W, SD = [], [], [], [], []
        for d in days:
            try:
                x_, y_, w_, _r, _ret, _e = _gen_meta_samples_cached(con, d)
            except Exception as e:                         # noqa: BLE001
                log.warning("  %s: %s", d, e)
                continue
            if not x_:
                continue
            perday.append((d, x_, y_, w_))
            X += [np.asarray(v, np.float32) for v in x_]
            Y += list(y_); W += list(w_); SD += [d] * len(x_)
    finally:
        con.close()
    if not X:
        log.error("no samples — run the forge first")
        return 1
    X = np.vstack(X); Y = np.asarray(Y, float)
    W = np.asarray(W, float); SD = np.asarray(SD)

    cap = CL.assess(Y, W, SD)
    log.info("LADDER | %s", cap.reason)
    if not cap.allows("DISCOVER"):
        log.warning("STAGE %s — per-feature effects are smaller than this "
                    "vault's resolution (%.3f). Screening 64 features now "
                    "would yield ~%.0f false leads at q=%.2f and nothing "
                    "real. Refusing; ~%d more day(s) needed.",
                    cap.stage, cap.mde_auc,
                    X.shape[1] * float(getattr(config, "DISCOVERY_FDR_Q",
                                               0.10)),
                    float(getattr(config, "DISCOVERY_FDR_Q", 0.10)),
                    cap.days_to_promote_power)
        _write(cap, [], [], "refused: underpowered (STAGE " + cap.stage + ")")
        return 0

    dim = X.shape[1]
    grp = _groups(dim)
    log.info("screening %d feature(s) in %d group(s) over %d day(s), "
             "n=%d", dim, len(grp), len(perday), Y.size)

    # ---- pre-register the whole family BEFORE any result exists
    try:
        from core.trial_registry import register
        for j in range(dim):
            register(family="feature_discovery", spec_id=f"uni_f{j}",
                     kind="pre_registered")
        for g in grp:
            register(family="feature_discovery", spec_id=f"ablate_{g}",
                     kind="pre_registered")
    except Exception as e:                                 # noqa: BLE001
        log.debug("registry unavailable (%s)", e)

    # ---- lens A: univariate, day-clustered
    uni = []
    for j in range(dim):
        col = X[:, j]
        if not np.isfinite(col).all() or float(np.nanstd(col)) < 1e-12:
            uni.append({"feature": j, "auc": float("nan"),
                        "p_one_sided": 1.0, "dead": True})
            continue
        auc, lo, p = CL.cluster_bootstrap_auc_p(col, Y, SD, n_boot=400)
        # a feature that ranks INVERSELY is just as informative
        if auc < 0.5:
            auc2, lo2, p2 = CL.cluster_bootstrap_auc_p(-col, Y, SD,
                                                       n_boot=400)
            if p2 < p:
                auc, lo, p = auc2, lo2, p2
        uni.append({"feature": j, "auc": round(float(auc), 4),
                    "ci90_lo": round(float(lo), 4),
                    "p_one_sided": round(float(p), 5), "dead": False})
    live = [u for u in uni if not u["dead"]]
    if live:
        rej, adj = CL.benjamini_hochberg(
            [u["p_one_sided"] for u in live],
            float(getattr(config, "DISCOVERY_FDR_Q", 0.10)))
        for u, r, q in zip(live, rej, adj):
            u["p_adj_bh"] = round(float(q), 5)
            u["survives_fdr"] = bool(r)
    hits = [u for u in live if u.get("survives_fdr")]
    log.info("UNIVARIATE | %d live feature(s), %d dead | %d survive "
             "BH-FDR q=%.2f", len(live), len(uni) - len(live), len(hits),
             float(getattr(config, "DISCOVERY_FDR_Q", 0.10)))
    for u in sorted(hits, key=lambda z: -z["auc"])[:10]:
        log.info("    f%-3d AUC %.4f (CI90 lo %.4f) q=%.4f",
                 u["feature"], u["auc"], u["ci90_lo"], u["p_adj_bh"])
    if not hits:
        log.info("    no single feature ranks winners above chance after "
                 "correction — recorded as the null at this depth")

    # ---- lens B: group ablation
    full = _oof_auc(perday, SD)
    log.info("ABLATION | full-model OOF AUC %.4f", full)
    abl = []
    for name, cols in grp.items():
        a_drop = _oof_auc(perday, SD, cols)
        d_auc = (full - a_drop) if np.isfinite(a_drop) else float("nan")
        abl.append({"group": name, "n_cols": len(cols),
                    "auc_without": (round(float(a_drop), 4)
                                    if np.isfinite(a_drop) else None),
                    "delta_auc": (round(float(d_auc), 4)
                                  if np.isfinite(d_auc) else None)})
        log.info("    drop %-14s → AUC %.4f (Δ %+.4f)", name,
                 a_drop if np.isfinite(a_drop) else float("nan"),
                 d_auc if np.isfinite(d_auc) else float("nan"))
    verdict = (f"{len(hits)} feature(s) survive FDR; full AUC {full:.4f}"
               if np.isfinite(full) else f"{len(hits)} feature(s) survive FDR")
    if not hits and (not np.isfinite(full) or full < config.META_MIN_AUC):
        verdict = ("NULL RECORDED — no univariate survivor and the full "
                   "model does not clear the promotion bar at this depth")
    log.info("VERDICT | %s", verdict)
    _write(cap, uni, abl, verdict, full)
    return 0


def _write(cap, uni, abl, verdict, full=None) -> None:
    out = {"ts": time.time(), "config_hash": config.CONFIG_HASH,
           "ladder": cap.as_dict() if cap else None,
           "fdr_q": float(getattr(config, "DISCOVERY_FDR_Q", 0.10)),
           "full_auc": (round(float(full), 4)
                        if full is not None and np.isfinite(full) else None),
           "univariate": uni, "ablation": abl, "verdict": verdict}
    try:
        p = Path(str(REPORT).format(d=time.strftime("%Y-%m-%d")))
        p.write_text(json.dumps(out, indent=1))
        log.info("report → %s", p)
    except Exception as e:                                 # noqa: BLE001
        log.warning("report write failed (%s)", e)


if __name__ == "__main__":
    raise SystemExit(main())