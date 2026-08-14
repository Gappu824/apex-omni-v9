"""
RANGE REGIME — is this market going anywhere, on ANY timescale?
================================================================
THE NUMBER THAT MOTIVATES THIS MODULE
--------------------------------------
    2026-08-11  NIFTY      TREND      2s of 18047   (0.01%)
    2026-08-11  BANKNIFTY  TREND      7s of 18047   (0.04%)
    2026-08-10  BANKNIFTY  TREND    240s of 17793   (1.35%)

On 2026-08-11 the system took FIVE directional NIFTY put trades — the
24400 strike three times — in a session that trended for two seconds.
Every one stopped out. Buying premium for direction in a market that does
not travel is a theta donation with extra steps.

The regime label ALREADY EXISTS. `regime_share_s` reports it every night.
It is simply not a gate: the funnel keys are stale_feed, no_market,
cas_auction, in_position, risk_halted, below_bar, not_persistent,
retest_guard, toxicity_trap, throttled, no_chain, no_quotes, risk_blocked,
no_fill, entered. There is no `range_bound` among them. The system knows
and does not act.

WHY A NEW DETECTOR RATHER THAN GATING ON THE EXISTING LABEL
------------------------------------------------------------
The existing labels are a state machine over thresholds with hysteresis,
and they are single-timescale by construction. A market can be flat over
five minutes and travelling over an hour, and a 60-minute option position
cares about the hour. Gating on a one-window label would refuse good
trades in the morning and allow bad ones after lunch.

So this measures the SAME question at several horizons at once and
requires agreement — which is precisely the "variable windows" reading:
one window is an opinion, several windows agreeing is evidence.

THE STATISTIC: LO-MACKINLAY VARIANCE RATIO (1988)
--------------------------------------------------
For a price series, VR(q) = Var(q-period return) / (q · Var(1-period
return)).

    VR ≈ 1   random walk — no exploitable persistence
    VR > 1   positively autocorrelated — trending, travel accumulates
    VR < 1   mean-reverting — range-bound, moves get retraced

This is the canonical test for exactly the question being asked, and it
is a TEST, not an indicator: it comes with a null and a sampling
distribution, so "range-bound" becomes a statement with a z-score rather
than a threshold someone picked.

Two implementation points that matter and are usually got wrong:

  1. THE HETEROSKEDASTICITY-ROBUST STATISTIC (Lo & MacKinlay's M2/z*), not
     the homoskedastic one. Index tick data is violently heteroskedastic —
     the open and the CAS window alone guarantee it — and the homoskedastic
     z over-rejects the random walk badly under changing variance. Using it
     here would manufacture "trending" verdicts out of volatility clusters.
  2. OVERLAPPING q-period returns with the small-sample bias correction,
     because non-overlapping windows throw away most of the sample and a
     session only has ~22 000 seconds to begin with.

Reference: Lo, A.W. & MacKinlay, A.C. (1988), "Stock Market Prices Do Not
Follow Random Walks: Evidence from a Simple Specification Test", Review of
Financial Studies 1(1). The Kaufman Efficiency Ratio is carried alongside
as a model-free cross-check — core/signal_persistence.py already uses it,
so the two agree on vocabulary.

WHAT THIS MODULE WILL AND WILL NOT CLAIM
-----------------------------------------
It reports a verdict per horizon and an aggregate. It does NOT assert that
refusing range-bound entries is profitable — that is a claim for
core/entry_counterfactual.py to settle on the real tape, since it can
replay the identical session with and without the gate. RANGE_GATE_ENABLED
is False by default for exactly that reason.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict

import numpy as np

import config

log = logging.getLogger("range_regime")

# Pre-registered horizons, in seconds. Chosen to bracket the hold budget:
# the shortest is scalp-scale, the longest exceeds MAX_HOLD_MINUTES so a
# 60-minute position is judged on a window at least as long as itself.
HORIZONS_S = (300, 900, 1800, 3600, 5400)

TRENDING, RANDOM, RANGE = "TRENDING", "RANDOM_WALK", "RANGE_BOUND"


@dataclass
class HorizonVerdict:
    horizon_s: int
    vr: float
    z: float
    verdict: str
    er: float               # Kaufman efficiency ratio, model-free check
    n: int

    def as_dict(self) -> dict:
        return asdict(self)


def variance_ratio(px: np.ndarray, q: int) -> tuple[float, float, int]:
    """Lo-MacKinlay VR(q) with the heteroskedasticity-robust z.

    Returns (VR, z*, n_used). z* is asymptotically N(0,1) under the null
    of a random walk WITH heteroskedastic increments — the only version
    defensible on intraday index data.
    """
    p = np.asarray(px, float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size < max(3 * q, 60):
        return float("nan"), float("nan"), int(p.size)
    x = np.diff(np.log(p))                       # 1-period log returns
    n = x.size
    mu = x.mean()
    var1 = np.sum((x - mu) ** 2) / (n - 1)
    if var1 <= 0:
        return float("nan"), float("nan"), n
    # overlapping q-period returns, unbiased denominator (Lo-MacKinlay eq 9)
    m = q * (n - q + 1) * (1.0 - q / n)
    if m <= 0:
        return float("nan"), float("nan"), n
    xq = np.convolve(x, np.ones(q), mode="valid")
    varq = np.sum((xq - q * mu) ** 2) / m
    vr = varq / var1

    # heteroskedasticity-robust variance of VR: sum of weighted delta_j
    theta = 0.0
    d = (x - mu) ** 2
    denom = float(np.sum(d)) ** 2
    if denom <= 0:
        return float(vr), float("nan"), n
    for j in range(1, q):
        # delta_hat(j) is a RATIO and carries its own 1/n — multiplying by
        # n here (as an earlier draft did) inflated theta by n and drove
        # every z to ~0.00, so a strongly mean-reverting OU series scored
        # z=-0.02 and read RANDOM_WALK. The test would have passed
        # everything. Lo-MacKinlay (1988) eq. 10: delta_hat(j) =
        # sum[(x_t-mu)^2 (x_{t-j}-mu)^2] / [sum (x_t-mu)^2]^2, and for iid
        # this is ~1/n, giving theta ~ (4q/3)/n as it should.
        delta = float(np.sum(d[j:] * d[:-j])) / denom
        w = 2.0 * (q - j) / q
        theta += (w ** 2) * delta
    if theta <= 0:
        return float(vr), float("nan"), n
    z = (vr - 1.0) / math.sqrt(theta)
    return float(vr), float(z), n


def efficiency_ratio(px: np.ndarray) -> float:
    """Kaufman ER: net displacement / total path. 0 = pure noise, 1 = a
    straight line. Model-free, no null — carried as a sanity check on the
    VR verdict rather than as a decision input."""
    p = np.asarray(px, float)
    p = p[np.isfinite(p) & (p > 0)]
    if p.size < 3:
        return float("nan")
    path = float(np.sum(np.abs(np.diff(p))))
    if path <= 0:
        return 0.0
    return float(abs(p[-1] - p[0]) / path)


def assess(px: np.ndarray, horizons=HORIZONS_S,
           alpha: float | None = None) -> dict:
    """Judge one spot series across every horizon.

    `px` is a per-second spot series for the session so far.
    """
    if alpha is None:
        alpha = float(getattr(config, "RANGE_Z_ALPHA", 1.96))
    p = np.asarray(px, float)
    out: list[HorizonVerdict] = []
    for q in horizons:
        if p.size < max(3 * q, 60):
            continue
        vr, z, n = variance_ratio(p, q)
        if not (vr == vr and z == z):
            continue
        if z > alpha:
            v = TRENDING
        elif z < -alpha:
            v = RANGE
        else:
            v = RANDOM
        out.append(HorizonVerdict(horizon_s=q, vr=round(vr, 4),
                                  z=round(z, 3), verdict=v,
                                  er=round(efficiency_ratio(p[-q:]), 4),
                                  n=n))
    if not out:
        return {"ok": False, "reason": "not enough session yet",
                "n_samples": int(p.size), "horizons": []}

    n_trend = sum(1 for h in out if h.verdict == TRENDING)
    n_range = sum(1 for h in out if h.verdict == RANGE)
    need = int(getattr(config, "RANGE_MIN_AGREE", 2))

    # AGREEMENT, not a single window. One horizon calling RANGE is an
    # opinion; several independent horizons agreeing is evidence, and the
    # asymmetry is deliberate — refusing to trade needs less proof than
    # deciding to.
    if n_trend >= need and n_trend > n_range:
        agg = TRENDING
    elif n_range >= need and n_range > n_trend:
        agg = RANGE
    else:
        agg = RANDOM
    return {"ok": True, "aggregate": agg, "n_trending": n_trend,
            "n_range": n_range, "n_horizons": len(out),
            "min_agree": need,
            "mean_er": round(float(np.nanmean([h.er for h in out])), 4),
            "horizons": [h.as_dict() for h in out],
            "n_samples": int(p.size)}


def may_trade_directional(a: dict, enabled: bool | None = None
                          ) -> tuple[bool, str]:
    """The gate. Directional PREMIUM BUYING only; it says nothing about
    range-selling structures, which want exactly this environment.

    Off by default. Whether refusing these entries actually pays is a
    question for core/entry_counterfactual.py on the real tape, not an
    assumption to ship.
    """
    # `enabled` lets a STUDY arm force the gate on while the live default
    # stays off. Without it tools/gate_ab_study.py read the global flag,
    # found it False, and the range arm returned (True,"") for every
    # signal — so on 2026-08-13 that arm was byte-identical to the
    # incumbent (-20,548 both, 91 trades both) and measured nothing at all,
    # while looking like a clean null result.
    if enabled is None:
        enabled = bool(getattr(config, "RANGE_GATE_ENABLED", False))
    if not enabled:
        return True, ""
    if not a.get("ok"):
        return True, ""                        # too early to refuse
    if a.get("aggregate") != RANGE:
        return True, ""
    hz = ", ".join(f"{h['horizon_s']}s VR {h['vr']:.2f} z {h['z']:+.1f}"
                   for h in a.get("horizons", [])
                   if h["verdict"] == RANGE)
    return False, (f"range_bound: {a['n_range']}/{a['n_horizons']} horizon(s) "
                   f"mean-reverting (need {a['min_agree']}) [{hz}] — buying "
                   f"premium for direction here pays theta for travel the "
                   f"tape is not delivering")


def report(a: dict, index: str = "", logger=None) -> None:
    lg = logger or log
    if not a.get("ok"):
        lg.info("range %s: %s (%d sample(s))", index, a.get("reason"),
                a.get("n_samples", 0))
        return
    lg.info("range %s: %s — %d trending / %d range of %d horizon(s), "
            "mean ER %.3f", index, a["aggregate"], a["n_trending"],
            a["n_range"], a["n_horizons"], a.get("mean_er", float("nan")))
    for h in a["horizons"]:
        lg.info("   %5ds  VR %5.2f  z %+6.2f  ER %.3f  -> %s",
                h["horizon_s"], h["vr"], h["z"], h["er"], h["verdict"])