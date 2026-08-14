"""
EPISODE RANKER — a different question, a different unit, a different model
==========================================================================
WHY THE FIFTH CLASSIFIER WOULD HAVE FAILED TOO
-----------------------------------------------
Four architectures have now been fitted to "will this trade win?" and all
four scored at chance. The reason is not capacity, it is resolution:

    n = 1953 rows over 31 sessions  ->  m̄ = 63 rows per session
    ICC 0.05  ->  n_eff  476  ->  detectable AUC >= 0.585
    ICC 0.20  ->  n_eff  146  ->  detectable AUC >= 0.655
    observed AUC = 0.5070

Rows one second apart are near-duplicates: they share the tape, the chain,
the regime and very nearly the outcome. Stacking 63 of them per session
inflates n by 63x and n_eff by almost nothing. The same code promoted the
COMMODITY meta at AUC 0.5979 the same night, so the machinery works — the
equity target does not have resolvable signal at this sample size.

THREE CHANGES, EACH FORCED BY A MEASURED FINDING
-------------------------------------------------
1. THE UNIT: episodes, not seconds.
   One observation per NON-OVERLAPPING episode. Two signals whose outcome
   windows overlap are one observation, not two — that is what "independent
   sample" means here. n falls from ~1953 to a few hundred and n_eff rises
   TOWARD n. Fewer honest rows beat many redundant ones.

2. THE TARGET: R = P&L / initial risk, not win/lose.
   Sign prediction on index options is a coin at this n_eff, and the July
   diagnostic chain already established there is no tradable intraday
   directional edge in this vault. Dispersion is a different question and
   payoff_target found one feature clearing at p(BH)=0.099 — marginal, but
   the only thing that has ever cleared on equity here.

3. THE PROBLEM: RANK within a session, not calibrate across sessions.
   This is the change that matters most, and it comes from the book, not
   from statistics. MAX_CONCURRENT_POSITIONS=1 with a 60-minute hold and a
   180s cooldown caps the day at ~5 trades: on 2026-08-11 six cascade
   signals fired and all six were refused because the single slot was
   occupied. The book does not need to know E[R]. It needs to know WHICH
   of today's candidates to spend the slot on.
   Ranking is a strictly weaker requirement than calibrated probability,
   and weaker requirements are what low n_eff can support. A model that
   cannot say "this trade wins 58% of the time" may still say "this one is
   better than that one", and the second statement is the one the book
   consumes.

MODEL CLASS: DELIBERATELY SMALL
--------------------------------
At n≈300 episodes a 64-feature booster memorises. So: features are
rank-transformed within session (scale-free, outlier-robust — option R has
a fat right tail that would otherwise dominate any least-squares fit), a
handful are selected, and a ridge scorer is fitted. Total capacity is a
few coefficients.

FEATURE SELECTION HAPPENS INSIDE THE FOLD. Selecting on all sessions and
then cross-validating is the most common leak in this literature and it
inflates every number downstream. Here the selection sees only the training
sessions of its own fold.

WHAT IT IS SCORED ON
--------------------
* WITHIN-SESSION SPEARMAN IC — did it order today's candidates correctly?
  Per session, then day-clustered. Never pooled: pooled IC on overlapping
  windows is the autocorrelation artifact that produced the phantom
  directional edge in July.
* TOP-1 REALISED R vs RANDOM-PICK R — the economic question, in the book's
  own units. RANDOM_SLOT is the same baseline core/entry_counterfactual.py
  uses, deliberately: a ranker that cannot beat picking at random is not
  selecting, whatever its IC says.

It can conclude NO SKILL, and on this vault that remains the likeliest
outcome. Nothing here promotes itself.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict

import numpy as np

import config

log = logging.getLogger("episode_ranker")

MIN_SESSIONS = 25
MIN_EPISODES = 120
MAX_FEATURES = 5          # pre-registered. At n≈300 this is already generous.


@dataclass
class Episode:
    day: str
    index: str
    t: int                 # session-second of the episode's signal
    r: float               # realised R = P&L / initial risk
    x: np.ndarray = field(repr=False, default=None)
    row: int = -1          # v9.9.38: the SOURCE ROW this episode came from.
    #                        Callers that join a second array (the approach
    #                        windows) must index by THIS, never by position.
    #                        Positional alignment held only because
    #                        build_episodes happens to sort the way the forge
    #                        emits — an implicit contract that, if it ever
    #                        broke, would pair windows with the wrong labels
    #                        and produce a confident number with no error.

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("x", None)
        return d


def build_episodes(days, ts, r, X, index=None, hold_s: int | None = None,
                   rows=None) -> tuple[list[Episode], dict]:
    """Collapse per-second rows into NON-OVERLAPPING episodes.

    Two signals whose outcome windows overlap are not two observations: they
    resolve against substantially the same tape. The rule is a hard gap of
    one hold budget — the horizon the label is measured over — which is the
    same embargo logic the purged day-folds already use, applied within the
    day instead of across it.

    Within an episode the FIRST signal is kept, never the best. Keeping the
    best would select on the outcome and quietly bake a look-ahead into
    every row.
    """
    if hold_s is None:
        hold_s = int(float(getattr(config, "MAX_HOLD_MINUTES", 60)) * 60)
    days = np.asarray(days)
    ts = np.asarray(ts, float)
    r = np.asarray(r, float)
    X = np.asarray(X, float)

    # `rows` maps local index -> index in the CALLER's original arrays, so a
    # caller that pre-filtered (dropping non-finite R, say) can still join
    # other row-aligned arrays correctly.
    rows = np.arange(len(ts)) if rows is None else np.asarray(rows)
    order = np.lexsort((ts, days))
    eps: list[Episode] = []
    stats = {"rows": int(len(order)), "episodes": 0, "collapsed": 0}
    last_day, last_t = None, -10 ** 9
    for i in order:
        d = str(days[i])
        if d != last_day:
            last_day, last_t = d, -10 ** 9
        if ts[i] - last_t < hold_s:
            stats["collapsed"] += 1
            continue
        if not np.isfinite(r[i]):
            continue
        eps.append(Episode(day=d, index=str(index) if index else "",
                           t=int(ts[i]), r=float(r[i]), x=X[i],
                           row=int(rows[i])))
        last_t = ts[i]
    stats["episodes"] = len(eps)
    if stats["rows"]:
        log.info("episodes: %d row(s) -> %d independent episode(s) "
                 "(%d collapsed as overlapping). n_eff rises toward n; that "
                 "is the point, not a loss of data.",
                 stats["rows"], stats["episodes"], stats["collapsed"])
    return eps, stats


def _rank_within(v: np.ndarray) -> np.ndarray:
    """Rank-transform to [-1, 1]. Scale-free and outlier-robust: option R has
    a fat right tail that would otherwise dominate a least-squares fit."""
    if v.size < 2 or np.std(v) == 0:
        return np.zeros_like(v)
    ranks = np.argsort(np.argsort(v)).astype(float)
    return 2.0 * ranks / (v.size - 1) - 1.0


def _prep(eps: list[Episode]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank features and target WITHIN each session.

    Within-session because that is the comparison the book makes: it never
    chooses between a Tuesday signal and a Thursday one, only among today's.
    Ranking within the day also removes any level effect a regime shift
    would otherwise inject as spurious signal.
    """
    X = np.vstack([e.x for e in eps])
    y = np.array([e.r for e in eps], float)
    d = np.array([e.day for e in eps])
    Xr = np.zeros_like(X, dtype=float)
    yr = np.zeros_like(y)
    for day in np.unique(d):
        m = d == day
        if m.sum() < 2:
            continue
        for j in range(X.shape[1]):
            Xr[m, j] = _rank_within(X[m, j])
        yr[m] = _rank_within(y[m])
    return Xr, yr, d


def _select(Xr: np.ndarray, yr: np.ndarray, d: np.ndarray,
            k: int) -> list[int]:
    """Top-k features by mean within-session IC — TRAIN sessions only.

    This function is called inside the fold, never outside it. Selecting on
    all sessions and then cross-validating is the standard leak in this
    literature; it inflates every number that follows.
    """
    scores = []
    for j in range(Xr.shape[1]):
        ics = []
        for day in np.unique(d):
            m = d == day
            if m.sum() >= 4 and np.std(Xr[m, j]) > 0 and np.std(yr[m]) > 0:
                ics.append(float(np.corrcoef(Xr[m, j], yr[m])[0, 1]))
        scores.append(abs(float(np.mean(ics))) if len(ics) >= 3 else 0.0)
    return list(np.argsort(scores)[::-1][:k])


def _ridge(Xs: np.ndarray, ys: np.ndarray, lam: float = 10.0) -> np.ndarray:
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    try:
        return np.linalg.solve(A, Xs.T @ ys)
    except np.linalg.LinAlgError:
        return np.zeros(Xs.shape[1])


def cross_validate(eps: list[Episode], n_fold: int = 5,
                   embargo_days: int = 1, seed: int = 0) -> dict:
    """Purged day-folds with an embargo. Returns out-of-fold scores.

    Whole sessions are held out and the sessions adjacent to the test block
    are PURGED from training, because an episode near a day boundary can
    share tape with the next session's open. Same discipline as the forge's
    purged-dayfold(k=5, embargo=1).
    """
    Xr, yr, d = _prep(eps)
    uniq = sorted(set(d.tolist()))
    if len(uniq) < n_fold + 2:
        return {"ok": False, "reason": f"{len(uniq)} session(s)"}
    folds = [set(uniq[i::n_fold]) for i in range(n_fold)]
    oof = np.full(len(yr), np.nan)
    chosen: dict[int, int] = {}

    for f in folds:
        te = np.array([x in f for x in d])
        emb = set()
        for i, day in enumerate(uniq):
            if day in f:
                for o in range(-embargo_days, embargo_days + 1):
                    if 0 <= i + o < len(uniq):
                        emb.add(uniq[i + o])
        tr = np.array([(x not in emb) for x in d])
        if tr.sum() < 40 or te.sum() < 5:
            continue
        k = int(min(MAX_FEATURES, max(1, tr.sum() // 40)))
        cols = _select(Xr[tr], yr[tr], d[tr], k)
        for c in cols:
            chosen[c] = chosen.get(c, 0) + 1
        w = _ridge(Xr[tr][:, cols], yr[tr])
        oof[te] = Xr[te][:, cols] @ w

    ok = ~np.isnan(oof)
    return {"ok": bool(ok.sum() > 0), "oof": oof, "y": yr, "day": d,
            "raw_r": np.array([e.r for e in eps], float),
            "n_scored": int(ok.sum()),
            "feature_votes": {int(c): int(v) for c, v in
                              sorted(chosen.items(), key=lambda z: -z[1])},
            "stability": (max(chosen.values()) / n_fold) if chosen else 0.0}


def evaluate(cv: dict, seed: int = 0) -> dict:
    """Score the ranker on the two questions that matter."""
    from core import capability_ladder as CL
    if not cv.get("ok"):
        return {"ok": False, "reason": cv.get("reason", "no oof")}

    oof, y, d, raw = cv["oof"], cv["y"], cv["day"], cv["raw_r"]
    rng = np.random.default_rng(seed)
    per_day_ic, top1, rnd1 = {}, {}, {}
    for day in np.unique(d):
        m = (d == day) & ~np.isnan(oof)
        if m.sum() < 3:
            continue
        if np.std(oof[m]) > 0 and np.std(y[m]) > 0:
            per_day_ic[day] = float(np.corrcoef(
                np.argsort(np.argsort(oof[m])).astype(float),
                np.argsort(np.argsort(y[m])).astype(float))[0, 1])
        rr = raw[m]
        top1[day] = float(rr[int(np.argmax(oof[m]))])
        rnd1[day] = float(np.mean(rr))      # expectation of a random pick

    if len(per_day_ic) < 5:
        return {"ok": False, "reason": f"{len(per_day_ic)} scored session(s)"}

    ic = CL.paired_test(per_day_ic)
    econ = CL.paired_test({k: top1[k] - rnd1[k] for k in top1})
    alpha = float(getattr(config, "EPISODE_ALPHA", 0.05))
    checks = {
        "ic_positive": ic["mean"] > 0,
        "ic_significant": ic.get("p", 1.0) <= alpha,
        "ic_above_mde": ic["mean"] > float(ic.get("mde", float("inf"))),
        "beats_random_pick": econ["mean"] > 0,
        "econ_significant": econ.get("p", 1.0) <= alpha,
        "econ_above_mde": econ["mean"] > float(econ.get("mde", float("inf"))),
        "features_stable": cv.get("stability", 0.0) >= float(
            getattr(config, "EPISODE_MIN_STABILITY", 0.6)),
    }
    gates = {"enough_sessions": len(per_day_ic) >= MIN_SESSIONS,
             "enough_episodes": int(cv["n_scored"]) >= MIN_EPISODES}
    return {"ok": all(checks.values()) and all(gates.values()),
            "n_sessions": len(per_day_ic), "n_episodes": int(cv["n_scored"]),
            "ic_mean": ic["mean"], "ic_p": ic.get("p", 1.0),
            "ic_mde": ic.get("mde", float("nan")), "ic_ci90": ic.get("ci90"),
            "econ_mean_R": econ["mean"], "econ_p": econ.get("p", 1.0),
            "econ_mde": econ.get("mde", float("nan")),
            "stability": cv.get("stability", 0.0),
            "feature_votes": cv.get("feature_votes", {}),
            "gates": gates, "checks": checks,
            "config_hash": config.CONFIG_HASH}


def report(v: dict, logger=None) -> None:
    lg = logger or log
    if not v.get("ok") and "n_sessions" not in v:
        lg.info("episode ranker: %s", v.get("reason"))
        return
    lg.info("EPISODE RANKER | %d episode(s) over %d session(s)",
            v["n_episodes"], v["n_sessions"])
    lg.info("  within-session IC   %+.4f | p %.4f | MDE %.4f",
            v["ic_mean"], v["ic_p"], v["ic_mde"])
    lg.info("  top-1 R vs random   %+.4f R/session | p %.4f | MDE %.4f",
            v["econ_mean_R"], v["econ_p"], v["econ_mde"])
    lg.info("  feature stability   %.2f (fraction of folds picking the same "
            "top feature)", v["stability"])
    for k, ok in {**v.get("gates", {}), **v.get("checks", {})}.items():
        lg.info("  %-22s %s", k, "PASS" if ok else "FAIL")
    if not v.get("ok"):
        lg.info("NOT PROMOTED. Ranking is a weaker claim than calibration "
                "and this vault still cannot support it. The MDE columns "
                "say what a larger sample could have resolved — that is a "
                "sample verdict, not proof of no effect.")