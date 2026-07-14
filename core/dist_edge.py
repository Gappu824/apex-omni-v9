"""
APEX OMNI v10 — DISTRIBUTIONAL EDGE ENGINE (what SAC should have been)
======================================================================
Instead of a policy chasing a falsified directional thesis for 3 hours a
night, learn — in seconds — the CONDITIONAL P&L DISTRIBUTION of taking the
gate's trade: three LightGBM quantile regressors (q10 / q50 / q90, pinball
loss) on the same feature vector the meta-labeler sees, trained under the
same purged day-fold CV. The meta says "how OFTEN does this win"; this says
"what does the P&L DISTRIBUTION look like" — together they are strictly more
information than any scalar policy output.

Sizing consumption (Rockafellar–Uryasev CVaR spirit, fractional-Kelly form):
    mult = clip( q50 / (q50 − q10), 0, 1 )   if q50 > 0 else 0
i.e. median edge over downside width — a distribution-aware Kelly proxy that
shrinks size exactly when the left tail fattens. TELEMETRY + artifact only
until a registered spec passes; DEE_SIZE_ENABLE stays False by constitution.

Labels are the triple-barrier NORMALIZED RETURNS the forge's own sample
generator produces (stamp-bumped cache adds them; old caches rebuild once).
Fail-open like Meta-Forge: no lightgbm / thin data ⇒ returns None, nothing
in the nightly breaks.
"""
from __future__ import annotations

import time

import numpy as np

import config
from core.meta_gbm import _purged_day_folds


def _pinball(y, q, alpha):
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def fit_dee(rows: list[tuple], min_train: int) -> dict | None:
    """rows: [(day, x_vec, ret, w)] with ret = normalized barrier return.
    Returns artifact dict or None (caller ignores — fail-open)."""
    try:
        import lightgbm as lgb
    except Exception:                                     # noqa: BLE001
        return None
    rows = [r for r in rows if r[2] is not None and np.isfinite(r[2])]
    if len(rows) < min_train:
        return None
    D = np.asarray([r[0] for r in rows])
    X = np.stack([np.asarray(r[1], np.float32) for r in rows])
    Y = np.asarray([float(r[2]) for r in rows], np.float32)
    W = np.asarray([float(r[3]) for r in rows], np.float32)
    W = W / max(float(W.mean()), 1e-9)
    alphas = (0.10, 0.50, 0.90)
    base = {"num_leaves": config.META_GBM_LEAVES,
            "learning_rate": config.META_GBM_LR,
            "min_child_samples": config.META_GBM_MINCHILD,
            "feature_fraction": 0.85, "bagging_fraction": 0.85,
            "bagging_freq": 1, "lambda_l2": 1.0, "verbosity": -1,
            "num_threads": 0, "seed": 20260714}
    t0 = time.time()
    oof = {a: np.full(len(Y), np.nan, np.float32) for a in alphas}
    iters = {a: [] for a in alphas}
    for test_d, train_d in _purged_day_folds(list(D),
                                             config.META_EMBARGO_DAYS):
        tr, te = np.isin(D, train_d), np.isin(D, test_d)
        if tr.sum() < 50 or te.sum() < 5:
            continue
        for a in alphas:
            p = dict(base, objective="quantile", alpha=a,
                     metric="quantile")
            dtr = lgb.Dataset(X[tr], label=Y[tr], weight=W[tr])
            dva = lgb.Dataset(X[te], label=Y[te], weight=W[te],
                              reference=dtr)
            bst = lgb.train(p, dtr,
                            num_boost_round=config.META_GBM_ROUNDS,
                            valid_sets=[dva],
                            callbacks=[lgb.early_stopping(50,
                                                          verbose=False)])
            oof[a][te] = bst.predict(X[te],
                                     num_iteration=bst.best_iteration)
            iters[a].append(bst.best_iteration or config.META_GBM_ROUNDS)
    got = ~np.isnan(oof[0.50])
    if got.sum() < min_train // 2 or len(iters[0.50]) < 2:
        return None
    # honest OOF: pinball per quantile + empirical coverage of [q10,q90]
    lo = np.minimum(oof[0.10][got], oof[0.50][got])       # enforce order
    hi = np.maximum(oof[0.90][got], oof[0.50][got])
    cover = float(np.mean((Y[got] >= lo) & (Y[got] <= hi)))
    metrics = {f"pinball_q{int(a*100)}":
               round(_pinball(Y[got], oof[a][got], a), 5) for a in alphas}
    files = {}
    for a in alphas:
        r = int(np.median(iters[a]))
        bst = lgb.train(dict(base, objective="quantile", alpha=a),
                        lgb.Dataset(X, label=Y, weight=W),
                        num_boost_round=max(r, 25))
        f = config.MODEL_DIR / f"dee_q{int(a*100)}.txt"
        tmp = f.with_suffix(".tmp")
        bst.save_model(str(tmp), num_iteration=r)
        tmp.replace(f)
        files[f"q{int(a*100)}"] = f.name
    return {"engine": "dee", "files": files, "n": int(len(Y)),
            "oof_cover_q10_q90": round(cover, 4), **metrics,
            "ret_mean": round(float(Y.mean()), 5),
            "ret_q": [round(float(np.quantile(Y, q)), 4)
                      for q in (0.1, 0.5, 0.9)],
            "fit_seconds": round(time.time() - t0, 2),
            "days": sorted(set(D.tolist())), "ts": time.time(),
            "config_hash": config.CONFIG_HASH}


_B: dict = {}


def predict_quantiles(art: dict, x: np.ndarray):
    """(q10,q50,q90) with monotonicity enforced; None if boosters missing."""
    try:
        import lightgbm as lgb
    except Exception:                                     # noqa: BLE001
        return None
    qs = []
    for k in ("q10", "q50", "q90"):
        f = config.MODEL_DIR / art["files"][k]
        b = _B.get(str(f))
        if b is None:
            if not f.exists():
                return None
            b = lgb.Booster(model_file=str(f))
            _B[str(f)] = b
        qs.append(float(b.predict(np.asarray(x, np.float32)[None, :])[0]))
    q10, q50, q90 = qs
    q50 = max(q50, q10)
    q90 = max(q90, q50)
    return q10, q50, q90


def size_multiplier(q10: float, q50: float, q90: float) -> float:
    """CVaR-aware fractional-Kelly proxy in [0,1]: median edge over downside
    width. Zero when the median take is not positive."""
    if q50 <= 0:
        return 0.0
    dn = max(q50 - q10, 1e-9)
    return float(min(max(q50 / dn, 0.0), 1.0))