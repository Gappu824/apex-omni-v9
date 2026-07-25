"""
APEX OMNI v9.8 — META-FORGE v2 (the labeler that finally tells the truth)
=========================================================================
Replaces the hand-rolled logistic meta-labeler's TRAINING with the tabular
state of the art, sized to this machine (i7-13650HX; the 4060 stays free for
RV-Net — gradient-boosted trees on a few thousand rows train in SECONDS on
CPU and, per Grinsztajn et al. 2022, still beat deep nets on tabular data):

  • LightGBM binary classifier, uniqueness-weighted (López de Prado ch.4 —
    overlapping triple-barrier labels are not independent evidence).
  • PURGED day-fold cross-validation with embargo (LdP ch.7): folds are whole
    DAYS, and ±META_EMBARGO_DAYS neighbours of the test day are dropped from
    training — intraday serial correlation cannot leak across the split.
  • ISOTONIC CALIBRATION (Zadrozny–Elkan 2002) fit on the out-of-fold
    predictions: the artifact ships a monotone map from raw score to HONEST
    probability. This is the direct cure for the live finding "said 0.50,
    did 0.05" — the number the Kelly blend consumes becomes a frequency.
  • Per-decile WILSON LOWER BOUNDS on the calibrated OOF outcomes, stored in
    the artifact: a conservative P(win) floor per probability bin, ready for
    a conformal-style gate (registered trial before it gates anything).
  • Final model refit on ALL data at the median early-stopped round; artifact
    stays a single JSON at META_MODEL_PATH (engine:"gbm" + booster file
    beside it in MODEL_DIR), atomically written, config-hash-stamped.

FAIL-OPEN BY DESIGN: if lightgbm is not importable this module returns None
and nightly_forge falls straight through to the proven logistic path — a
missing pip package can never cost a nightly. Install once:
    pip install lightgbm
"""
from __future__ import annotations

import json
import math
import time
from statistics import NormalDist

import numpy as np
import logging

log = logging.getLogger("forge")   # AUDIT: the skill report below needs a
#                                     logger; this module had none.

import config


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U). 0.5 = no ordering ability at all.

    Brier/BSS conflate DISCRIMINATION (does the model rank winners above
    losers?) with CALIBRATION (are the numbers the right probabilities?). A
    model can rank well and still be badly calibrated — in which case a FIXED
    bar like META_ENTRY_P_BAR never fires while a RELATIVE bar (today's top
    decile) would. AUC separates the two, so the operator can tell "there is no
    signal" apart from "there is signal, expressed on the wrong scale".
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    n_pos = float((y > 0.5).sum())
    n_neg = float((y <= 0.5).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1, dtype=float)
    # average ranks over ties — a constant predictor must score exactly 0.5
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt), float)
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[y > 0.5].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def _wilson_lo(wins: float, n: float, ci: float = 0.90) -> float:
    if n <= 0:
        return 0.0
    z = NormalDist().inv_cdf(0.5 + ci / 2.0)
    p = wins / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    rad = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0))
    return max((ctr - rad) / den, 0.0)


def _purged_day_folds(days: list[str], embargo: int):
    """Leave-one-day-out when few days, else 5 grouped folds; each yields
    (test_days, train_days) with ±embargo neighbours of test purged."""
    uniq = sorted(set(days))
    k = len(uniq) if len(uniq) <= 8 else 5
    chunks = [uniq[i::k] for i in range(k)] if k == 5 else [[d] for d in uniq]
    idx = {d: i for i, d in enumerate(uniq)}
    for test in chunks:
        if not test:
            continue
        banned = set()
        for d in test:
            i = idx[d]
            for j in range(i - embargo, i + embargo + 1):
                if 0 <= j < len(uniq):
                    banned.add(uniq[j])
        train = [d for d in uniq if d not in banned]
        if train:
            yield test, train


def _isotonic(p: np.ndarray, y: np.ndarray, w: np.ndarray):
    """Weighted isotonic regression via PAVA. Returns breakpoints (x, y_hat)
    suitable for np.interp at serve time. Pure numpy — no sklearn needed."""
    order = np.argsort(p, kind="stable")
    x, t, ww = p[order], y[order].astype(float), w[order].astype(float)
    # pool adjacent violators
    vals = t.copy()
    wts = ww.copy()
    xs = x.copy()
    i = 0
    n = len(vals)
    v, wv, xv = list(vals), list(wts), list(xs)
    out_v, out_w, out_x = [], [], []
    for j in range(n):
        cv, cw, cx = v[j], wv[j], xv[j]
        while out_v and out_v[-1] > cv:
            pv, pw = out_v.pop(), out_w.pop()
            out_x.pop()
            cv = (cv * cw + pv * pw) / (cw + pw)
            cw += pw
        out_v.append(cv)
        out_w.append(cw)
        out_x.append(cx)
    # expand pooled blocks back to breakpoints (x = block right edge)
    bx = np.asarray(out_x, float)
    by = np.clip(np.asarray(out_v, float), 0.0, 1.0)
    if len(bx) == 1:                       # degenerate: constant map
        bx = np.array([0.0, 1.0])
        by = np.array([by[0], by[0]])
    return bx, by


def fit_gbm(perday: list[tuple], min_train: int) -> dict | None:
    """perday: [(day, X_list, y_list, w_list)] exactly as train_meta builds.
    Returns the artifact dict (engine:'gbm') or None → caller falls back."""
    try:
        import lightgbm as lgb
    except Exception:                                     # noqa: BLE001
        return None
    days, X, Y, W, D = [], [], [], [], []
    for day, xs, ys, ws in perday:
        days.append(day)
        for x, y, w in zip(xs, ys, ws):
            X.append(np.asarray(x, np.float32))
            Y.append(float(y))
            W.append(float(w))
            D.append(day)
    n = len(X)
    if n < min_train:
        return None
    X = np.stack(X)
    Y = np.asarray(Y, np.float32)
    W = np.asarray(W, np.float32)
    W = W / max(float(W.mean()), 1e-9)
    D = np.asarray(D)
    params = {"objective": "binary", "metric": "binary_logloss",
              "num_leaves": config.META_GBM_LEAVES,
              "learning_rate": config.META_GBM_LR,
              "min_child_samples": config.META_GBM_MINCHILD,
              "feature_fraction": 0.85, "bagging_fraction": 0.85,
              "bagging_freq": 1, "lambda_l2": 1.0, "verbosity": -1,
              "num_threads": 0, "seed": 20260714}
    t0 = time.time()
    oof_p = np.full(n, np.nan, np.float32)
    best_iters = []
    for test_days, train_days in _purged_day_folds(list(D), 
                                                   config.META_EMBARGO_DAYS):
        tr = np.isin(D, train_days)
        te = np.isin(D, test_days)
        if tr.sum() < 50 or te.sum() < 5:
            continue
        dtr = lgb.Dataset(X[tr], label=Y[tr], weight=W[tr])
        dva = lgb.Dataset(X[te], label=Y[te], weight=W[te], reference=dtr)
        bst = lgb.train(params, dtr, num_boost_round=config.META_GBM_ROUNDS,
                        valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_p[te] = bst.predict(X[te],
                                num_iteration=bst.best_iteration)
        best_iters.append(bst.best_iteration or config.META_GBM_ROUNDS)
    got = ~np.isnan(oof_p)
    if got.sum() < min_train // 2 or len(best_iters) < 2:
        return None                       # CV too thin to trust — fall back
    # honest OOF metrics BEFORE calibration
    brier_raw = float(np.mean((oof_p[got] - Y[got]) ** 2))
    # isotonic calibration on OOF
    iso_x, iso_y = _isotonic(oof_p[got], Y[got], W[got])
    p_cal = np.interp(oof_p[got], iso_x, iso_y)
    brier_cal = float(np.mean((p_cal - Y[got]) ** 2))
    # reliability table + Wilson lower bounds per decile of CALIBRATED p
    bins = []
    edges = np.linspace(0.0, 1.0, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p_cal >= lo) & (p_cal < hi if hi < 1 else p_cal <= hi)
        nb = int(m.sum())
        if nb == 0:
            bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": 0,
                         "said": None, "did": None, "p_lo": None})
            continue
        said = float(p_cal[m].mean())
        did = float(Y[got][m].mean())
        bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": nb,
                     "said": round(said, 4), "did": round(did, 4),
                     "p_lo": round(_wilson_lo(float(Y[got][m].sum()), nb),
                                   4)})
    # final refit on ALL data at the median early-stopped round
    rounds = int(np.median(best_iters))
    bst = lgb.train(params, lgb.Dataset(X, label=Y, weight=W),
                    num_boost_round=max(rounds, 25))
    mpath = config.MODEL_DIR / "meta_gbm.txt"
    tmp = mpath.with_suffix(".tmp")
    bst.save_model(str(tmp), num_iteration=rounds)
    tmp.replace(mpath)
    fit_s = time.time() - t0
    acc = float((((np.interp(oof_p[got], iso_x, iso_y)) > 0.5)
                 == (Y[got] > 0.5)).mean())
    # AUDIT (2026-07-22): a probabilistic gate must be scored against the
    # trivial CLIMATOLOGY forecast (always predict the base rate), not against
    # nothing. Brier Skill Score = 1 − Brier_model / Brier_climatology; ≤0 means
    # the model carries no information the base rate did not already have. The
    # 720-signal GBM scored BSS −0.038 while silently gating 100% of entries —
    # exactly the failure rvnet already guards against ("does NOT beat HAR — no
    # artifact written"). META_MIN_BSS applies that same discipline here.
    # 2026-07-23 DEGENERACY GUARD: with ~5 winners in 384 samples the commodity
    # forge promoted a model whose "98.8% holdout accuracy" was purely the class
    # imbalance (always predict LOSS). Isotonic calibration and a P(win) gate
    # are meaningless on that few positives, so refuse before reporting skill.
    _pos = int(Y[got].sum())
    _neg = int(got.sum() - _pos)
    _minpos = int(getattr(config, "META_MIN_POSITIVES", 0) or 0)
    if _minpos and min(_pos, _neg) < _minpos:
        log.warning("META NOT PROMOTED: only %d positive / %d negative "
                    "outcomes (base rate %.4f) — below META_MIN_POSITIVES=%d. "
                    "A calibrated probability gate cannot be fit on that few "
                    "events; the reported accuracy would be the class "
                    "imbalance, not information. No artifact written; the "
                    "brain stays heuristic-only. Keep harvesting.",
                    _pos, _neg, float(Y[got].mean()), _minpos)
        return None
    # AUDIT (2026-07-23): live telemetry showed the promoted meta emitting a
    # CONSTANT — p50 = p90 = p99 = max = 0.23 across 13,791 evaluations on BOTH
    # indices, while conviction varied normally. A gate fed a constant cannot
    # discriminate at all: it blocks 100% or passes 100% forever, whatever the
    # market. Measure the calibrated OOF spread so this is visible every night.
    _cal_oof = np.interp(oof_p[got], iso_x, iso_y)
    _spread = float(np.quantile(_cal_oof, 0.95) - np.quantile(_cal_oof, 0.05))
    _distinct = int(len(np.unique(np.round(_cal_oof, 4))))
    # DISCRIMINATION vs CALIBRATION. auc_raw is the booster's own ordering;
    # auc_cal is what serving actually sees after isotonic. Isotonic is
    # monotone, so the two differ only through TIES — and a calibration that
    # collapses to a constant destroys ranking that the booster did have. If
    # auc_raw >> 0.5 while auc_cal == 0.5, the signal exists and the mapping is
    # eating it; that is a fixable problem and argues for a RELATIVE gate.
    _auc_raw = _auc(Y[got], oof_p[got])
    _auc_cal = _auc(Y[got], _cal_oof)
    log.info("META DISCRIMINATION | AUC raw %.4f | AUC calibrated %.4f "
             "(0.500 = no ordering ability) -> %s", _auc_raw, _auc_cal,
             "NO RANKING SIGNAL" if not (_auc_cal > 0.53)
             else "ranks better than chance")
    if _auc_raw > 0.55 >= _auc_cal:
        log.warning("CALIBRATION IS DESTROYING RANKING: the booster orders "
                    "signals (AUC %.4f) but the calibrated output does not "
                    "(AUC %.4f). A fixed probability bar cannot exploit this; "
                    "a relative gate (top-quantile of the day) could.",
                    _auc_raw, _auc_cal)
    _min_spread = getattr(config, "META_MIN_OOF_SPREAD", None)
    if _spread < 0.02:
        log.warning("META OUTPUT NEARLY CONSTANT: calibrated OOF p05-p95 "
                    "spread = %.4f over %d distinct values. A gate model that "
                    "does not vary cannot separate winners from losers — it is "
                    "the base rate wearing a model's clothes.", _spread,
                    _distinct)
    if _min_spread is not None and _spread < float(_min_spread):
        # NOTE deliberately NOT the default. Unlike the commodity positives
        # guard (where refusing left that brain heuristic-only — strictly
        # safer), refusing the EQUITY meta removes the gate that is currently
        # blocking a population the counterfactual grades at 0 wins in 400 and
        # -Rs 631k. Withholding it would UNBLOCK those trades. Report loudly,
        # act only on an explicit operator decision.
        log.warning("META NOT PROMOTED: OOF spread %.4f < META_MIN_OOF_SPREAD "
                    "%.4f — refusing a constant-output gate.", _spread,
                    float(_min_spread))
        return None
    p_clim = float(Y[got].mean())
    brier_clim = float(np.mean((p_clim - Y[got]) ** 2))
    bss_cal = (1.0 - brier_cal / brier_clim) if brier_clim > 0 else 0.0
    bss_raw = (1.0 - brier_raw / brier_clim) if brier_clim > 0 else 0.0
    _b = config.BASE_TP_PCT / max(config.BASE_SL_PCT, 1e-9)
    _breakeven = 1.0 / (1.0 + _b)
    log.info("META SKILL | base_rate %.4f | Brier climatology %.5f vs model "
             "%.5f (cal) → BSS %+.4f %s | payoff b=%.2f ⇒ break-even p=%.3f, "
             "serving bar %.2f", p_clim, brier_clim, brier_cal, bss_cal,
             "NO SKILL" if bss_cal <= 0 else "has skill", _b, _breakeven,
             getattr(config, "META_ENTRY_P_BAR", float("nan")))
    _min_bss = getattr(config, "META_MIN_BSS", None)
    if _min_bss is not None and bss_cal < float(_min_bss):
        log.warning("META NOT PROMOTED: BSS %+.4f < META_MIN_BSS %.4f — the "
                    "model does not beat always-predicting the base rate. No "
                    "artifact written; the gate degrades to the honest "
                    "conviction bar. The refusal is the system working.",
                    bss_cal, float(_min_bss))
        return None
    return {"engine": "gbm", "model_file": mpath.name,
            "oof_spread_p05_p95": round(_spread, 5),
            "auc_raw": (None if _auc_raw != _auc_raw
                        else round(_auc_raw, 4)),
            "auc_cal": (None if _auc_cal != _auc_cal
                        else round(_auc_cal, 4)),
            "oof_distinct_values": _distinct,
            "brier_climatology": round(brier_clim, 5),
            "bss_cal": round(bss_cal, 4), "bss_raw": round(bss_raw, 4),
            "breakeven_p": round(_breakeven, 4),
            "iso_x": np.round(iso_x, 6).tolist(),
            "iso_y": np.round(iso_y, 6).tolist(),
            "bins": bins, "n": n,
            "base_rate": round(float(Y.mean()), 4),
            "oof_brier_raw": round(brier_raw, 5),
            "oof_brier_cal": round(brier_cal, 5),
            "holdout_acc": round(acc, 4),
            "holdout": f"purged-dayfold(k={len(best_iters)},"
                       f"embargo={config.META_EMBARGO_DAYS})",
            "cv_rounds_median": rounds, "fit_seconds": round(fit_s, 2),
            "uniqueness_mean": round(float(W.mean()), 4),
            "days": sorted(set(days)), "ts": time.time(),
            "config_hash": config.CONFIG_HASH}


_BOOSTERS: dict = {}


def score_vec(meta: dict, x: np.ndarray,
              clamp: bool = True) -> float | None:
    """Serve-time scorer for engine:'gbm' artifacts — lazy-cached booster,
    isotonic map, floor/cap clamp. Returns None if the booster is missing so
    the caller can degrade exactly as with no meta at all."""
    try:
        import lightgbm as lgb
    except Exception:                                     # noqa: BLE001
        return None
    mpath = config.MODEL_DIR / meta.get("model_file", "meta_gbm.txt")
    # AUDIT (2026-07-24): the cache was keyed on the PATH alone with no
    # invalidation, so any process that had already loaded meta_gbm.txt kept
    # serving that booster even after the forge overwrote the file. The nightly
    # forge does exactly that within a single process — fit, write, then score
    # the exam — so a stale booster could grade the very model that replaced
    # it. Key on (path, mtime): a rewritten model is picked up automatically.
    try:
        _mt = mpath.stat().st_mtime
    except FileNotFoundError:
        return None
    key = (str(mpath), _mt)
    bst = _BOOSTERS.get(key)
    if bst is None:
        bst = lgb.Booster(model_file=str(mpath))
        _BOOSTERS.clear()          # only ever one live model; don't leak
        _BOOSTERS[key] = bst
    p_raw = float(bst.predict(np.asarray(x, np.float32)[None, :])[0])
    p_cal = float(np.interp(p_raw, np.asarray(meta["iso_x"], float),
                            np.asarray(meta["iso_y"], float)))
    if getattr(config, "META_USE_PLO", False):
        for bn in meta.get("bins") or []:
            if (bn.get("p_lo") is not None
                    and bn.get("n", 0) >= config.META_PLO_MIN_N
                    and bn["lo"] <= p_cal < bn["hi"] + (1e-9 if bn["hi"] == 1
                                                        else 0)):
                p_cal = float(bn["p_lo"])   # serve the Wilson LOWER bound
                break
    if not clamp:
        # AUDIT: META_P_FLOOR (0.5) clamps every sub-floor probability UP to
        # exactly 0.50 — which is why 4,656 block reasons read an identical
        # "meta P(win) 0.50<0.55" and the model's real distribution was
        # invisible. The floor belongs to the SIZING path; telemetry and
        # diagnosis must see the true calibrated value.
        return float(p_cal)
    return float(min(max(p_cal, config.META_P_FLOOR), config.META_P_CAP))