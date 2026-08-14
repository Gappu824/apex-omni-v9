"""
PAYOFF TARGET — predict the magnitude, because the sign is not predictable
==========================================================================
THE CASE FOR CHANGING THE QUESTION
-----------------------------------
The equity meta predicts P(win). On 2026-08-11 it scored AUC 0.4988 raw /
0.5210 calibrated over n=1901. Corrected for the fact that samples inside a
session share one tape (m̄ = 63.4 rows per day, 30 non-empty days), the
effective sample is 141–846 depending on ICC, and at EVERY level 0.5210 is
indistinguishable from 0.500. The detectability floor is AUC 0.587 (ICC
0.05) to 0.658 (ICC 0.20). No replacement classifier is going to clear that
on daily index-option direction — and the July diagnostic chain already
concluded there is no tradable directional edge in this vault at any
horizon.

That is not an argument for a better classifier. It is an argument that the
TARGET is wrong. Sign prediction on index returns is close to a coin; the
DISPERSION of outcomes is not — volatility clusters and is persistent
(Engle 1982, Bollerslev 1986), which is the entire reason rv_forecaster
exists in this system and shows skill where direction shows none.

So this module predicts R — the payoff in units of the trade's own initial
risk — instead of whether the trade wins:

    R      = P&L / risk_rs           where risk_rs = |entry − stop| × qty
    R_mfe  = (MFE − entry)·side·qty / risk_rs      the POTENTIAL, ≥ 0
    R_real = realised P&L / risk_rs                what was collected

R_mfe is a non-negative magnitude. That is the quantity with a chance of
being forecastable, and the shadow book has been recording exactly its
ingredients since 2026-08-10.

WHY THIS FITS THE EV GATE BETTER THAN A PROBABILITY
----------------------------------------------------
The gate today needs p versus p* = stop/(target+stop). That decomposition
ASSUMES the target and stop are the outcomes — but the theta guillotine,
the trail and the disaster floor all intervene, so the realised payoff is
not the specified one. Estimating E[R] directly skips the decomposition:
it uses the part of the problem that has signal and drops the part that
does not. A conservative lower QUANTILE of R plays the role p0 plays now —
the pessimistic bound the gate compares against.

THE DISCIPLINE THAT MAKES THIS RESEARCH AND NOT HOPE
-----------------------------------------------------
Nothing here fits a model until predictability has been MEASURED and has
cleared its own MDE. `measure()` runs first and can return "not
predictable"; `fit_quantiles()` refuses to run if it did. The test is the
one that killed the directional hypothesis in July and must be applied to
this idea with equal force:

  * PER-DAY Spearman IC, never pooled. Pooled IC t-stats on 1 Hz data are
    autocorrelation-inflated artifacts — that is exactly how the phantom
    directional edge survived for weeks.
  * DAY-CLUSTERED significance and an explicit MDE, so "no signal" is never
    confused with "not enough sessions to tell".
  * BENJAMINI-HOCHBERG across the feature set, because scanning features
    and reporting the best one is how noise gets promoted.
  * PRE-REGISTERED quantiles and horizon. Chosen here, before any result.

If the measurement says no, that is the finding, and this module is a
two-page answer instead of a model nobody should trust.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

import config

log = logging.getLogger("payoff_target")

# Pre-registered. Fixed before any result exists.
QUANTILES = (0.25, 0.50, 0.75)
MIN_SESSIONS = 20          # the user's own bar: measure before fitting
MIN_TRADES = 60


@dataclass
class PayoffRow:
    day: str
    index: str
    risk_rs: float
    r_real: float
    r_mfe: float
    r_best: float
    coverage: float

    def as_dict(self) -> dict:
        return asdict(self)


def build_rows(trades) -> tuple[list[PayoffRow], dict]:
    """Turn shadow-labelled trades into R-multiples.

    `trades` are core.shadow_labels.LabelledTrade. The risk denominator is
    the trade's OWN initial risk, so a ₹200 option and a ₹15 option are on
    one scale — without that, the target is dominated by whichever
    instrument happened to be expensive that week.
    """
    stats = {"in": len(trades), "no_risk": 0, "thin": 0, "kept": 0}
    min_cov = float(getattr(config, "SHADOW_MIN_COVERAGE", 0.60))
    out: list[PayoffRow] = []
    for t in trades:
        if float(t.coverage) < min_cov:
            stats["thin"] += 1
            continue
        # risk in rupees. The shadow ledger carries entry and the policy
        # engine's stop_pct; when neither is recoverable the row is dropped
        # rather than given a guessed denominator.
        risk = getattr(t, "risk_rs", None)
        if risk is None:
            entry = getattr(t, "entry_px", None)
            qty = getattr(t, "qty", None)
            if entry and qty:
                risk = abs(float(entry)) * float(
                    getattr(config, "BASE_SL_PCT", 0.20)) * float(qty)
        if not risk or risk <= 0:
            stats["no_risk"] += 1
            continue
        mfe = getattr(t, "mfe_px", None)
        entry = getattr(t, "entry_px", None)
        side = int(getattr(t, "side", 1) or 1)
        qty = float(getattr(t, "qty", 0) or 0)
        r_mfe = (float(((mfe - entry) * side * qty) / risk)
                 if (mfe is not None and entry is not None and qty)
                 else float("nan"))
        out.append(PayoffRow(
            day=t.day, index=t.index, risk_rs=float(risk),
            r_real=float(t.realized_pnl) / float(risk),
            r_mfe=max(r_mfe, 0.0) if r_mfe == r_mfe else float("nan"),
            r_best=float(t.best_pnl) / float(risk),
            coverage=float(t.coverage)))
        stats["kept"] += 1
    return out, stats



def load_forge_matrix(path=None):
    """Read the matrix nightly_forge publishes. Returns (rows, X, names).

    This is the whole sample, not the shadow ledger: the forge already
    grades every replayed signal with a barrier P&L and the payoff geometry
    it was graded on, so R = ret / risk exists for all ~1900 rows across 38
    sessions without waiting for live shadows to accumulate. The shadow
    book remains the LIVE cross-check on the same quantity — a target
    measured only in replay and never observed forward is how a backtest
    becomes a belief.
    """
    import numpy as _np
    p = Path(path or (config.STATE_DIR / "meta_train_matrix.npz"))
    if not p.exists():
        return None, None, None
    try:
        z = _np.load(p, allow_pickle=False)
    except Exception as e:                                 # noqa: BLE001
        log.warning("payoff matrix unreadable (%s)", e)
        return None, None, None
    try:
        ch = str(z["config_hash"][0])
    except Exception:                                      # noqa: BLE001
        ch = ""
    if ch and ch != config.CONFIG_HASH:
        log.warning("payoff matrix was built under CONFIG_HASH %s != %s — "
                    "ignored. A matrix from a different feature world is "
                    "not evidence about this one.", ch, config.CONFIG_HASH)
        return None, None, None
    X = _np.asarray(z["X"], float)
    ret = _np.asarray(z["ret"], float)
    risk = _np.asarray(z["risk"], float)
    days = [str(d) for d in z["day"]]
    rows = [PayoffRow(day=days[i], index="", risk_rs=float(risk[i]),
                      r_real=float(ret[i] / risk[i]),
                      r_mfe=float("nan"),          # replay grades barriers,
                      r_best=float("nan"),         # not excursions
                      coverage=1.0)
            for i in range(len(ret)) if risk[i] > 0]
    keep = [i for i in range(len(ret)) if risk[i] > 0]
    names = [f"f{j}" for j in range(X.shape[1])]
    return rows, X[keep], names


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 4:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def measure(rows: list[PayoffRow], X: np.ndarray, feat_names,
            target: str = "r_real") -> dict:
    """IS R PREDICTABLE AT ALL? Run this before fitting anything.

    Per-day Spearman IC per feature, then a day-clustered t-test on the
    per-day ICs, then BH across features. Returns `predictable: False`
    unless something clears — and `fit_quantiles` refuses without it.
    """
    from core import capability_ladder as CL

    y = np.array([getattr(r, target) for r in rows], float)
    days = np.array([r.day for r in rows])
    ok = np.isfinite(y)
    y, X, days = y[ok], np.asarray(X, float)[ok], days[ok]
    uniq = sorted(set(days.tolist()))

    gates = {"enough_sessions": len(uniq) >= MIN_SESSIONS,
             "enough_trades": int(ok.sum()) >= MIN_TRADES}
    base = {"target": target, "n": int(ok.sum()), "n_days": len(uniq),
            "gates": gates,
            "quantiles": list(QUANTILES),
            "config_hash": config.CONFIG_HASH}
    if len(uniq) < 4 or ok.sum() < 20:
        return {**base, "predictable": False,
                "reason": f"only {len(uniq)} session(s) / {int(ok.sum())} "
                          f"trade(s) — not yet measurable"}

    # DEDUPLICATE IDENTICAL COLUMNS BEFORE TESTING.
    # 2026-08-13: 64 features produced only 41 distinct ICs — f17/f36/f55
    # and f18/f37/f56 matched byte-for-byte, because the matrix stacks the
    # SAME feature per index (NIFTY/BANKNIFTY/SENSEX). Benjamini-Hochberg
    # then corrected over 64 hypotheses when ~21 were distinct, making the
    # test far more conservative than the evidence warranted. Duplicate
    # columns are not independent tests; counting them as such is a
    # correction error in the SAFE direction, which makes it easy to miss.
    _seen: dict = {}
    _keep, _alias = [], {}
    for j in range(X.shape[1]):
        key = np.asarray(X[:, j], float).tobytes()
        if key in _seen:
            _alias.setdefault(_seen[key], []).append(j)
        else:
            _seen[key] = j
            _keep.append(j)
    if len(_keep) < X.shape[1]:
        log.info("payoff features: %d column(s) -> %d DISTINCT; %d exact "
                 "duplicate(s) dropped before BH so the correction counts "
                 "hypotheses, not copies", X.shape[1], len(_keep),
                 X.shape[1] - len(_keep))

    feats = []
    _names = list(feat_names) if feat_names else [f"f{i}"
                                                 for i in range(X.shape[1])]
    for j in _keep:
        name = _names[j]
        if j in _alias:
            name = f"{name}(={','.join(_names[k] for k in _alias[j])})"
        per_day = []
        for d in uniq:
            m = days == d
            if m.sum() >= 4:
                ic = _spearman(X[m, j], y[m])
                if ic == ic:
                    per_day.append(ic)
        if len(per_day) < 4:
            continue
        st = CL.paired_test({uniq[i]: v for i, v in enumerate(per_day)})
        feats.append({"feature": str(name), "n_days": len(per_day),
                      "mean_ic": st["mean"], "p": st.get("p", 1.0),
                      "mde": st.get("mde", float("nan")),
                      "ci90": st.get("ci90")})
    if not feats:
        return {**base, "predictable": False,
                "reason": "no feature had enough same-day observations"}

    rej, adj = CL.benjamini_hochberg(
        [f["p"] for f in feats],
        float(getattr(config, "PAYOFF_FDR_Q", 0.10)))
    for f, r_, q_ in zip(feats, rej, adj):
        f["p_adj_bh"] = round(float(q_), 4)
        f["significant"] = bool(r_)
        f["above_mde"] = bool(abs(f["mean_ic"]) >
                              float(f.get("mde", float("inf"))))

    winners = [f for f in feats if f["significant"] and f["above_mde"]]
    predictable = bool(winners) and all(gates.values())
    return {**base, "predictable": predictable,
            "n_features": len(feats), "n_significant": len(winners),
            "features": sorted(feats, key=lambda z: -abs(z["mean_ic"])),
            "winners": [f["feature"] for f in winners],
            "reason": ("" if predictable else
                       "no feature cleared BH + MDE on per-day IC"
                       if all(gates.values()) else
                       "sample gates not met")}


def report(m: dict, logger: logging.Logger | None = None) -> None:
    lg = logger or log
    lg.info("PAYOFF TARGET %r | n=%d over %d session(s) (need %d/%d)",
            m.get("target"), m.get("n", 0), m.get("n_days", 0),
            MIN_TRADES, MIN_SESSIONS)
    for k, v in m.get("gates", {}).items():
        lg.info("  gate %-20s %s", k, "PASS" if v else "FAIL")
    for f in (m.get("features") or [])[:8]:
        lg.info("  %-22s IC %+.4f over %2d day(s) | p(BH) %.3f | MDE %.4f "
                "%s", f["feature"], f["mean_ic"], f["n_days"],
                f.get("p_adj_bh", 1.0), f.get("mde", float("nan")),
                "<-- clears" if f.get("significant") and f.get("above_mde")
                else "")
    if m.get("predictable"):
        lg.info("PREDICTABLE: %d feature(s) clear BH + MDE on per-day IC — "
                "quantile fitting is now permitted.", m.get("n_significant"))
    else:
        lg.info("NOT PREDICTABLE (%s). No model is fitted. This is a "
                "finding, not a failure: the alternative is a quantile "
                "regression on noise, sized through Kelly.",
                m.get("reason") or "n/a")


def pinball(y: np.ndarray, q_hat: np.ndarray, q: float) -> float:
    d = y - q_hat
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def fit_quantiles(rows: list[PayoffRow], X: np.ndarray, feat_names,
                  measurement: dict, target: str = "r_real") -> dict | None:
    """Purged day-fold quantile regression on R. REFUSES unless measured.

    Linear pinball regression on purged day folds, scored against the
    UNCONDITIONAL quantile — the only honest baseline, because beating
    "predict the same number every time" is exactly the bar a conditional
    model has to clear to be worth serving.
    """
    if not measurement.get("predictable"):
        log.warning("fit_quantiles refused: measure() returned NOT "
                    "PREDICTABLE (%s). Fitting anyway is how a quantile "
                    "regression on noise reaches production.",
                    measurement.get("reason"))
        return None

    y = np.array([getattr(r, target) for r in rows], float)
    days = np.array([r.day for r in rows])
    ok = np.isfinite(y)
    y, Xa, days = y[ok], np.asarray(X, float)[ok], days[ok]
    Xz = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-9)
    Xz = np.hstack([Xz, np.ones((len(Xz), 1))])
    uniq = sorted(set(days.tolist()))
    n_fold = min(5, len(uniq))
    folds = [set(uniq[i::n_fold]) for i in range(n_fold)]

    out = {"target": target, "n": int(len(y)), "n_days": len(uniq),
           "quantiles": {}, "config_hash": config.CONFIG_HASH}
    for q in QUANTILES:
        oof = np.full(len(y), np.nan)
        for f in folds:
            te = np.array([d in f for d in days])
            tr = ~te
            if tr.sum() < 20:
                continue
            w = np.zeros(Xz.shape[1])
            w[-1] = float(np.quantile(y[tr], q))
            for _ in range(300):               # subgradient on pinball
                r = y[tr] - Xz[tr] @ w
                g = Xz[tr].T @ np.where(r >= 0, q, q - 1.0) / max(tr.sum(), 1)
                w += 0.05 * g - 1e-4 * w
            oof[te] = Xz[te] @ w
        if np.isnan(oof).any():
            continue
        base = np.full(len(y), float(np.quantile(y, q)))
        lm, lb = pinball(y, oof, q), pinball(y, base, q)
        out["quantiles"][str(q)] = {
            "pinball_model": round(lm, 5),
            "pinball_unconditional": round(lb, 5),
            "skill": round(1.0 - lm / lb, 5) if lb > 0 else None,
            "beats_unconditional": bool(lm < lb)}
    return out