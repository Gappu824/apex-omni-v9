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

import config


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
    return {"engine": "gbm", "model_file": mpath.name,
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


def score_vec(meta: dict, x: np.ndarray) -> float | None:
    """Serve-time scorer for engine:'gbm' artifacts — lazy-cached booster,
    isotonic map, floor/cap clamp. Returns None if the booster is missing so
    the caller can degrade exactly as with no meta at all."""
    try:
        import lightgbm as lgb
    except Exception:                                     # noqa: BLE001
        return None
    mpath = config.MODEL_DIR / meta.get("model_file", "meta_gbm.txt")
    key = str(mpath)
    bst = _BOOSTERS.get(key)
    if bst is None:
        if not mpath.exists():
            return None
        bst = lgb.Booster(model_file=key)
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
    return float(min(max(p_cal, config.META_P_FLOOR), config.META_P_CAP))