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

from pathlib import Path

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


def serve_spread(scores, y, w, cap: int = 4000, probe: int = 1500,
                 seed: int = 0):
    """What the GATE WILL ACTUALLY EMIT — measured, not proxied.

    meta_gbm has measured a calibrated OOF spread since the 2026-07-23
    audit (the promoted meta was serving a constant 0.23). That check fits
    ONE isotonic map over the OOF set and measures its spread. Serving does
    something else entirely: core.meta_gate builds an IVAP — TWO isotonic
    fits PER QUERY — and reports (p0, p1) merged. Those are different
    estimators, and passing one says nothing about the other. On matched
    synthetic data the isotonic map carried 925 distinct values while the
    Venn-Abers merge carried 36: a ~26x resolution collapse the old check
    could not see.

    2026-08-10 is what that costs. The artifact loaded clean
    (holdout_acc=0.7449, n=1901, va=yes) and the feature-integrity tripwire
    stayed silent all session — live features genuinely varied. Yet across
    14 655 evaluations per index the served win probability took two
    values, 0.204 and 0.4557, and "EV: optimistic p1 0.20 < p*" became the
    single largest block reason on every index. The gate was deciding on
    p* alone.

    So this builds the REAL VennAbers serving uses, over a stride of the
    same OOF payload, and reports the spread and distinct count of the
    merged probability. It imports meta_gate lazily: meta_gate imports this
    module, and a top-level import would close the cycle.
    """
    import numpy as _np
    try:
        from core.meta_gate import VennAbers
    except Exception as e:                                 # noqa: BLE001
        log.warning("serve-spread diagnostic unavailable (%s) — promotion "
                    "will fall back to the isotonic proxy, which cannot "
                    "see a Venn-Abers collapse", e)
        return None

    s = _np.asarray(scores, float)
    if s.size < 50:
        return None
    if s.size > cap:                       # same cap serving applies
        idx = _np.linspace(0, s.size - 1, cap).astype(int)
        s, y, w = s[idx], _np.asarray(y)[idx], _np.asarray(w)[idx]
    try:
        va = VennAbers(s, _np.asarray(y, float), _np.asarray(w, float))
    except Exception as e:                                 # noqa: BLE001
        log.warning("VennAbers construction failed (%s)", e)
        return None

    rng = _np.random.default_rng(seed)
    q = rng.choice(s, size=min(probe, s.size), replace=True)
    merged, p0s, p1s = [], [], []
    for v in q:
        a, b = va.interval(float(v))
        p0s.append(a)
        p1s.append(b)
        merged.append(VennAbers.merge(a, b))
    merged = _np.asarray(merged, float)
    return {
        "spread": float(_np.quantile(merged, 0.95)
                        - _np.quantile(merged, 0.05)),
        "distinct": int(len(_np.unique(_np.round(merged, 4)))),
        "p1_distinct": int(len(_np.unique(_np.round(p1s, 4)))),
        "p0_distinct": int(len(_np.unique(_np.round(p0s, 4)))),
        "median": float(_np.median(merged)), "n_probe": int(merged.size),
    }



def _embed_booster(mpath) -> dict:
    """Read the booster text and package it for the artifact.

    zlib + base64 because the LightGBM text format compresses ~5x and a
    10 KB model becomes a couple of KB of JSON — small enough that carrying
    it costs nothing and large enough that losing it costs everything.
    sha256 over the RAW text, so a same-size corruption is caught where a
    byte count would miss it.
    """
    import base64
    import hashlib
    import zlib
    try:
        raw = Path(mpath).read_bytes()
    except Exception as e:                                 # noqa: BLE001
        log.warning("could not embed the booster (%s) — the artifact will "
                    "fall back to the external file and remains vulnerable "
                    "to a sibling overwrite", e)
        return {}
    return {"b64": base64.b64encode(zlib.compress(raw, 6)).decode("ascii"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "raw_bytes": len(raw)}


def _extract_booster(meta: dict):
    """The embedded model text, verified. None if absent or corrupt.

    Verification is not ceremony: the whole point of embedding is that the
    served model is provably the one the artifact describes, and an
    unverified extract would just move the drift inside the file.
    """
    import base64
    import hashlib
    import zlib
    b64 = meta.get("booster_b64")
    if not b64:
        return None
    try:
        raw = zlib.decompress(base64.b64decode(b64))
    except Exception as e:                                 # noqa: BLE001
        log.error("embedded booster failed to decompress (%s) — refusing "
                  "to fall back to the external file, which is exactly the "
                  "path that produced the mismatch this design removes", e)
        return False
    want = str(meta.get("booster_sha256") or "")
    if want and hashlib.sha256(raw).hexdigest() != want:
        log.error("embedded booster FAILED its sha256 — the artifact is "
                  "corrupt. Refusing to serve.")
        return False
    return raw.decode("utf-8", errors="strict")


def _x_schema(width: int) -> str:
    """Fingerprint of the FEATURE LAYOUT the artifact was trained on.

    Deliberately separate from CONFIG_HASH. CONFIG_HASH answers "is this the
    same feature WORLD" (and so governs cache invalidation); this answers
    "is this the same COLUMN LAYOUT" (and so governs whether a stored model
    may be served). They diverge exactly where a serving-side switch changes
    the vector without changing the data — META_CROSS_INDEX being the case
    that broke on 2026-08-14.
    """
    import hashlib as _h
    parts = [f"w={int(width)}",
             f"xidx={int(bool(getattr(config, 'META_CROSS_INDEX', False)))}",
             f"news={int(bool(getattr(config, 'NEWS_FEED_META', False)))}"]
    return _h.sha1("|".join(parts).encode()).hexdigest()[:10]


def fit_gbm(perday: list[tuple], min_train: int,
            oof_out: dict | None = None,
            model_path: Path | None = None) -> dict | None:
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
    # AUDIT (2026-07-25): this write is UNCONDITIONAL, so any caller that fits
    # a model replaces the booster the live brain serves. tools/meta_lift.py
    # did exactly that — a read-only research tool silently clobbered
    # production, and two probe runs minutes apart scored different models
    # while reading identical metadata. Research callers now pass model_path
    # to redirect; the forge passes nothing and behaves exactly as before.
    mpath = Path(model_path) if model_path else (config.MODEL_DIR /
                                                 "meta_gbm.txt")
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
    if oof_out is not None:
        # Research hook (tools/meta_lift.py): expose the OOF predictions the CV
        # already computed, WITHOUT bloating the artifact that serving loads.
        # `mask` maps these back to the caller's sample order.
        oof_out["mask"] = got.copy()
        oof_out["oof_raw"] = oof_p[got].copy()
        oof_out["oof_cal"] = _cal_oof.copy()
        oof_out["y"] = Y[got].copy()
    # ---- v9.9 META-GATE v3: Venn-Abers calibration payload. The OOF
    # (raw score, label, uniqueness weight) triples ARE the calibration
    # set (cross-VA: every score here was produced out-of-fold under the
    # purged day split). Serving builds two isotonic fits per query for a
    # finite-sample-valid interval [p0,p1] — see core/meta_gate.py.
    _vs, _vy, _vw = oof_p[got], Y[got], W[got]
    _cap = int(getattr(config, "META_VA_MAX_CAL", 4000))
    if len(_vs) > _cap:                     # stride-thin on the score order
        _o = np.argsort(_vs, kind="stable")
        _o = _o[np.linspace(0, len(_o) - 1, _cap).astype(int)]
        _vs, _vy, _vw = _vs[_o], _vy[_o], _vw[_o]
    va_payload = {"s": np.round(_vs, 6).tolist(),
                  "y": np.round(_vy, 1).tolist(),
                  "w": np.round(_vw, 4).tolist()}
    # per-feature TRAINING liveness — the serve-time skew tripwire's
    # reference (a feature that never varied in training cannot "freeze")
    _span = X.max(axis=0) - X.min(axis=0)
    feat_alive = (_span > 1e-9).tolist()
    # honest VA diagnostics on a stride of OOF, calibrated on the rest
    _va_width = _va_brier = float("nan")
    try:
        from core.meta_gate import VennAbers as _VA
        _probe = np.linspace(0, len(_vs) - 1,
                             min(200, len(_vs))).astype(int)
        _mask = np.ones(len(_vs), bool); _mask[_probe] = False
        if _mask.sum() >= 50:
            _va = _VA(_vs[_mask], _vy[_mask], _vw[_mask])
            _ws, _bs = [], []
            for _j in _probe:
                _p0, _p1 = _va.interval(float(_vs[_j]))
                _pm = _VA.merge(_p0, _p1)
                _ws.append(_p1 - _p0)
                _bs.append((_pm - _vy[_j]) ** 2)
            _va_width = float(np.mean(_ws))
            _va_brier = float(np.mean(_bs))
            log.info("META-GATE v3 | VA mean interval width %.4f | "
                     "merged-Brier %.5f (isotonic %.5f) over %d probes",
                     _va_width, _va_brier, brier_cal, len(_probe))
    except Exception as _e:                                # noqa: BLE001
        log.warning("VA diagnostics skipped (%s)", _e)
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
    # v9.9.23: ONE threshold, used by the verdict line AND the gate below.
    # They disagreed: the report called anything <= 0.53 "NO RANKING SIGNAL"
    # while the gate refused only below 0.52. On 2026-08-11 the equity meta
    # scored _auc_cal = 0.5210 — the log printed NO RANKING SIGNAL and the
    # gate promoted it anyway, because the stricter number was cosmetic.
    # A system that states a model cannot rank and then ships it is worse
    # than one that never measured: it produces an audit trail that says
    # the right thing while the wrong thing happens.
    _min_auc = getattr(config, "META_MIN_AUC", 0.53)
    _ranks = (_auc_cal == _auc_cal and _min_auc is not None
              and _auc_cal > float(_min_auc))
    log.info("META DISCRIMINATION | AUC raw %.4f | AUC calibrated %.4f "
             "(0.500 = no ordering ability; gate at %.3f) -> %s",
             _auc_raw, _auc_cal, float(_min_auc if _min_auc else 0.0),
             "ranks better than chance" if _ranks else "NO RANKING SIGNAL")
    if _auc_raw > 0.55 >= _auc_cal:
        log.warning("CALIBRATION IS DESTROYING RANKING: the booster orders "
                    "signals (AUC %.4f) but the calibrated output does not "
                    "(AUC %.4f). A fixed probability bar cannot exploit this; "
                    "a relative gate (top-quantile of the day) could.",
                    _auc_raw, _auc_cal)
    # AUDIT (2026-07-28): the commodity forge promoted a model that this very
    # block had just described as "NO RANKING SIGNAL" — AUC 0.4915, i.e. worse
    # than a coin flip at ordering winners above losers, with holdout accuracy
    # 92.6% that is exactly the 92.8% class imbalance. It cleared
    # META_MIN_POSITIVES (35 > 30) and META_MIN_BSS is report-only, so nothing
    # stopped it. I had added the diagnostic that DETECTS this and no guard
    # that ACTS on it. A gate model that cannot rank cannot gate: served
    # through Kelly it sizes on noise. Refusing leaves the brain heuristic-only,
    # which is strictly safer, so this one defaults to ARMED.
    if not _ranks and _min_auc is not None and _auc_cal == _auc_cal:
        log.warning("META NOT PROMOTED: calibrated AUC %.4f < META_MIN_AUC "
                    "%.4f — the model does not order winners above losers "
                    "(0.500 = chance). Any headline accuracy here is just the "
                    "%.1f%% majority class. No artifact written; the brain "
                    "stays heuristic-only.", _auc_cal, float(_min_auc),
                    100.0 * (1.0 - float(Y[got].mean())))
        return None
    # ---- v9.9.18: gate on what SERVING emits, not the isotonic proxy.
    # Reached only if the model can rank at all — spread without ordering
    # is a model that emits varied numbers in no useful sequence, and
    # 2026-08-11 is the proof: serve spread 0.0684 over 17 distinct values
    # cleared this block while AUC sat at 0.4988 raw / 0.5210 calibrated.
    # Spread is necessary, never sufficient.
    _srv = serve_spread(_vs, _vy, _vw,
                        cap=int(getattr(config, "META_VA_MAX_CAL", 4000)))
    if _srv is not None:
        log.info("META SERVE-PATH (Venn-Abers, what the gate actually "
                 "emits): spread %.4f over %d distinct value(s) "
                 "[p0:%d p1:%d] | isotonic proxy said spread %.4f over %d",
                 _srv["spread"], _srv["distinct"], _srv["p0_distinct"],
                 _srv["p1_distinct"], _spread, _distinct)
        _need_sp = float(getattr(config, "META_MIN_SERVE_SPREAD", 0.05))
        _need_di = int(getattr(config, "META_MIN_SERVE_DISTINCT", 12))
        if _srv["spread"] < _need_sp or _srv["distinct"] < _need_di:
            log.error("META NOT PROMOTED: the SERVE path is degenerate — "
                      "spread %.4f (need %.4f) over %d distinct value(s) "
                      "(need %d). On 2026-08-10 a model that passed the "
                      "isotonic check served two values, 0.204 and 0.4557, "
                      "across 14 655 evaluations and 'optimistic p1 0.20 < "
                      "p*' became the largest block reason on every index. "
                      "A gate fed a near-constant is the base rate wearing "
                      "a model's clothes: it cannot discriminate, so it "
                      "blocks or passes on p* alone. Keeping the incumbent.",
                      _srv["spread"], _need_sp, _srv["distinct"], _need_di)
            return None
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
    try:
        _bst_bytes = mpath.stat().st_size
    except Exception:                                      # noqa: BLE001
        _bst_bytes = -1
    # ---- v9.9.43: EMBED THE MODEL IN THE ARTIFACT.
    # The artifact and the booster were TWO FILES with no atomic binding, so
    # they could desynchronise for many reasons — a sibling process writing
    # the shared default path, a partial write, a crash between the two
    # writes, one restored from backup. `model_bytes` was the only tie, and
    # a byte count is a weak one: it collides trivially and cannot see
    # same-size corruption at all.
    # A model that lives INSIDE its own artifact cannot drift from it. One
    # file, one atomic replace, one content hash. The .txt is still written
    # for debuggability but it is no longer the source of truth, so nothing
    # that overwrites it can change what gets served.
    _emb = _embed_booster(mpath)
    return {"engine": "gbm", "model_file": mpath.name,
            "booster_b64": _emb.get("b64"),
            "booster_sha256": _emb.get("sha256"),
            "booster_raw_bytes": _emb.get("raw_bytes"),
            # AUDIT (2026-07-28): the artifact never recorded how WIDE its
            # x-vector was. Enabling cross-index peer features takes x from 61
            # to 64, and a 61-dim booster fed a 64-dim vector does not fail
            # loudly — LightGBM will happily score garbage. Record the width and
            # refuse a mismatch at serving.
            "x_dim": int(X.shape[1]),
            # v9.9.34: the SCHEMA, not just the width. CONFIG_HASH cannot
            # catch this — META_CROSS_INDEX is hash-EXCLUDED (correctly: it
            # is a serving knob) yet it appends 3 peer columns, so X went
            # 61 -> 64 on 2026-08-14 with the hash unchanged, the day caches
            # untouched, and a 61-feature artifact still on disk. The result
            # was a LightGBMError per evaluation. A model must describe the
            # feature world it was fitted in, independently of the hash.
            "x_schema": _x_schema(int(X.shape[1])),
            "model_bytes": _bst_bytes,
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
            "va": va_payload, "feat_alive": feat_alive,
            "va_mean_width": (None if _va_width != _va_width
                              else round(_va_width, 5)),
            "va_brier_merged": (None if _va_brier != _va_brier
                                else round(_va_brier, 5)),
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
_XDIM_WARNED: dict = {}   # (expected, got) -> last-logged ts (throttle)


def _raw_score(meta: dict, x: np.ndarray) -> float | None:
    """Booster load + integrity guards + raw prediction. Shared by the
    legacy point scorer (score_vec) and the v3 interval scorer
    (score_interval) so the guard rails can never drift apart."""
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
        # PREFER THE EMBEDDED MODEL. It cannot have drifted from the
        # artifact because it IS the artifact.
        _txt = _extract_booster(meta)
        if _txt is False:
            return None                       # corrupt: refuse, never guess
        if _txt:
            bst = lgb.Booster(model_str=_txt)
        else:
            # LEGACY artifact with no embedded model. Keep the byte check —
            # but REFUSE on mismatch instead of warning and serving anyway.
            # The old code logged "The served model is NOT the one this
            # artifact describes" and then loaded it on the very next line,
            # so every session since 2026-08-14 scored against a booster the
            # metadata did not describe while the log said so plainly. A
            # guard that detects a fault and proceeds is worse than no
            # guard: it produces the audit trail of a system that is
            # protected, and the behaviour of one that is not.
            _want = meta.get("model_bytes")
            if _want and _want > 0:
                try:
                    _have = mpath.stat().st_size
                except Exception:                          # noqa: BLE001
                    _have = -1
                if _have != _want:
                    log.error("BOOSTER MISMATCH: %s is %s bytes, artifact "
                              "says %s. REFUSING to serve — the gate falls "
                              "back to the conviction bar. Re-run the forge "
                              "to write a self-contained artifact.",
                              mpath.name, _have, _want)
                    return None
            bst = lgb.Booster(model_file=str(mpath))
        _BOOSTERS.clear()          # only ever one live model; don't leak
        _BOOSTERS[key] = bst
    # AUDIT (2026-07-28): x_dim guard. My first attempt anchored on a line that
    # does not exist here, so str.replace() silently did nothing and I shipped a
    # guard that was never inserted — hence the assert above. Note the real
    # failure mode is NOT silent scoring: LightGBM raises LightGBMError on a
    # width mismatch. But an uncaught exception in the serving path would
    # propagate into the live brain, so catch it here and return None.
    _xa = np.asarray(x, np.float32)
    # AUDIT (2026-07-29 LIVE CRASH): this checked meta["x_dim"] only, so it was
    # a NO-OP for any artifact trained BEFORE that field existed — which is
    # exactly the model in production during a migration. The brain died on
    # LightGBMError instead of degrading. Ask the BOOSTER how wide it is: it is
    # authoritative, present for every artifact old or new, and cannot drift
    # from the model it describes.
    # Ask every way the object might answer. `num_feature()` is a raw
    # lgb.Booster method; a sklearn LGBMClassifier exposes `n_features_in_`
    # and hides the booster behind `booster_`. When the artifact is the
    # sklearn form, num_feature() raises, _bwidth falls to 0, the guard below
    # is skipped by `if _want and ...`, and the LightGBMError surfaces from
    # predict() instead — per evaluation, thousands of lines.
    _bwidth = 0
    for _probe in ("num_feature", "n_features_in_", "num_features"):
        try:
            _v = getattr(bst, _probe, None)
            _v = _v() if callable(_v) else _v
            if _v:
                _bwidth = int(_v)
                break
        except Exception:                                  # noqa: BLE001
            continue
    if not _bwidth:
        try:
            _bwidth = int(bst.booster_.num_feature())
        except Exception:                                  # noqa: BLE001
            _bwidth = 0
    # SCHEMA FIRST. The width check below catches 61-vs-64; this catches the
    # harder case — same width, different layout — which no shape check can
    # see and which would score silently against the wrong columns.
    _have_sch = str(meta.get("x_schema") or "")
    if _have_sch:
        _now_sch = _x_schema(int(_xa.size))
        if _have_sch != _now_sch:
            _k2 = ("schema", _have_sch, _now_sch)
            if time.time() - _XDIM_WARNED.get(_k2, 0.0) > float(
                    getattr(config, "XDIM_REMIND_S", 900)):
                _XDIM_WARNED[_k2] = time.time()
                log.error("META SCHEMA MISMATCH: artifact was trained under "
                          "layout %s, the builder now emits %s (width %d, "
                          "META_CROSS_INDEX=%s). Refusing to score — a model "
                          "fed the wrong columns produces a confident number "
                          "about nothing. Re-run the forge.", _have_sch,
                          _now_sch, int(_xa.size),
                          bool(getattr(config, "META_CROSS_INDEX", False)))
            return None
    _want = int(meta.get("x_dim") or 0) or _bwidth
    if _want and int(_want) != int(_xa.size):
        # THROTTLED. This fires on EVERY evaluation — several per second, per
        # index — so an un-throttled ERROR buried the 2026-07-29 session log in
        # thousands of identical lines and drowned the brain's real output. The
        # condition is static until the forge re-runs, so say it once per
        # (expected, got) pair, then repeat only every X-DIM REMIND_S so it can
        # never be silently forgotten either.
        _k = (int(_want), int(_xa.size))
        _now = time.time()
        _last = _XDIM_WARNED.get(_k, 0.0)
        if _now - _last > float(getattr(config, "XDIM_REMIND_S", 900)):
            _XDIM_WARNED[_k] = _now
            log.error("X-DIM MISMATCH: artifact expects %d features, got %d — "
                      "the model was trained on a different feature set "
                      "(META_CROSS_INDEX on/off?). Refusing to score; the gate "
                      "falls back to the conviction bar until the forge "
                      "re-runs. (Repeating at most every %.0f min.)",
                      int(_want), int(_xa.size),
                      float(getattr(config, "XDIM_REMIND_S", 900)) / 60.0)
        return None
    try:
        p_raw = float(bst.predict(_xa[None, :])[0])
    except Exception as e:                                 # noqa: BLE001
        # Belt to the guard's braces. Nothing in the serving path may take the
        # brain down; a scoring failure degrades to "no meta opinion".
        _ek = ("predict", str(e)[:60])
        if time.time() - _XDIM_WARNED.get(_ek, 0.0) > float(
                getattr(config, "XDIM_REMIND_S", 900)):
            _XDIM_WARNED[_ek] = time.time()
            log.error("META ARTIFACT UNUSABLE (%s). Throttled: this runs per "
                      "evaluation and un-throttled it buries the session log.",
                      e)
        return None
    if False:
        log.error("meta scoring failed (%s) — returning None so the gate "
                  "falls back to the conviction bar", e)
        return None
    return p_raw


def score_vec(meta: dict, x: np.ndarray,
              clamp: bool = True) -> float | None:
    """Serve-time scorer for engine:'gbm' artifacts — lazy-cached booster,
    isotonic map, floor/cap clamp. Returns None if the booster is missing so
    the caller can degrade exactly as with no meta at all."""
    p_raw = _raw_score(meta, x)
    if p_raw is None:
        return None
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


# ---------------------------------------------------------------------------
# v9.9 META-GATE v3 serving: Venn-Abers interval + feature-skew status.
# ---------------------------------------------------------------------------
_INTEGRITY: dict = {}    # artifact ts -> FeatureIntegrity (one live model)


def score_interval(meta: dict, x: np.ndarray
                   ) -> tuple[float, float, float, str] | None:
    """Returns (p0, p1, p_merged, integrity_status) or None when the
    artifact has no VA payload / booster is unavailable — the caller
    falls back to the legacy point gate, fail-open."""
    from core import meta_gate as MGT
    va = MGT.va_from_artifact(meta)
    if va is None:
        return None
    p_raw = _raw_score(meta, x)
    if p_raw is None:
        return None
    key = float(meta.get("ts", 0.0))
    fi = _INTEGRITY.get(key)
    if fi is None:
        _INTEGRITY.clear()
        fi = _INTEGRITY[key] = MGT.FeatureIntegrity()
    status = fi.observe(np.asarray(x, np.float32), meta.get("feat_alive"))
    p0, p1 = va.interval(float(p_raw))
    return p0, p1, MGT.VennAbers.merge(p0, p1), status