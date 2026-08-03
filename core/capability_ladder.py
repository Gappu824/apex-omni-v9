"""
CAPABILITY LADDER — what can this vault actually detect, today?
===============================================================
The forge has refused three nights running: AUC 0.494 / 0.492 / 0.442,
all below chance. Two explanations fit that evidence equally well:

    (a) there is no signal in these features at this horizon, or
    (b) there IS signal, and 1,584 day-clustered samples cannot see it.

Every night the system was answering (a) by default and searching no
further. This module makes the question quantitative and therefore
answerable — and makes the ANSWER drive what the system is allowed to
do next.

WHAT IT COMPUTES
----------------
1. Effective sample size. Raw n lies twice over:
   - AFML uniqueness weights mean samples overlap in time. Kish's
     effective size n_kish = (Σw)² / Σw² is the standard correction.
   - Samples inside one trading day share a regime. The design effect
     DEFF = 1 + (m̄ − 1)·ICC (Kish 1965) discounts for that clustering;
     ICC is estimated by one-way ANOVA on the day factor, not assumed.
   n_eff = n_kish / DEFF, floored at the number of days — you never
   have more independent evidence than you have days.

2. The standard error of AUC at that n_eff, via Hanley & McNeil (1982):
       Q1 = A/(2−A), Q2 = 2A²/(1+A)
       SE = √[(A(1−A) + (n₁−1)(Q1−A²) + (n₀−1)(Q2−A²)) / (n₁n₀)]

3. The MINIMUM DETECTABLE AUC — the smallest true ranking ability this
   vault could distinguish from chance at the configured power, solved
   by bisection on A: (A − 0.5)/SE(A) ≥ z_α + z_β.

4. DAYS-TO-POWER: at the observed samples-per-day rate, how many more
   trading days until the minimum detectable AUC drops below the
   promotion bar. This is the honest answer to "when will we know?"

WHY IT GATES SEARCH
-------------------
Searching an underpowered vault does not find weak signal — it finds
noise, and the more horizons and features you try, the more certainly
it does. So the ladder unlocks stages only as detection becomes
possible:

    STAGE 0  BLIND     MDE > 0.60   collect data; no search at all
    STAGE 1  SCREEN    MDE ≤ 0.60   few pre-registered hypotheses
                                    (horizon sweep) under FDR control
    STAGE 2  DISCOVER  MDE ≤ 0.55   per-feature screening, ablations
    STAGE 3  PROMOTE   MDE ≤ bar    the guard's own AUC floor is
                                    detectable; promotion means something

Refusing to search is not timidity — at STAGE 0 a search that "finds"
something is reporting its own noise floor.

Everything here is pure arithmetic over (weights, labels, day tags).
No I/O, no config mutation, no side effects.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

import config


# ----------------------------------------------------------------- normal
def _ppf(p: float) -> float:
    """Inverse standard normal (Acklam's rational approximation, |ε|<1e-9
    over the range we use). Avoids a scipy dependency in a hot path."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > ph:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ------------------------------------------------------------ effective n
def kish_effective_n(w: np.ndarray) -> float:
    """(Σw)²/Σw² — the effective size of a weighted sample (Kish 1965).
    Equals n when all weights are equal; collapses toward the count of
    genuinely distinct observations as weights become uneven."""
    w = np.asarray(w, dtype=np.float64)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.0
    return float(w.sum() ** 2 / np.square(w).sum())


def day_icc(y: np.ndarray, day: np.ndarray) -> float:
    """Intra-class correlation of the LABEL across trading days, by the
    one-way random-effects ANOVA estimator:
        ICC = (MSB − MSW) / (MSB + (m̄ − 1)·MSW)
    Measures how much of the win/lose variance is 'which day was it'.
    Clipped to [0, 1); 0 when days are indistinguishable."""
    y = np.asarray(y, dtype=np.float64)
    day = np.asarray(day)
    groups = [y[day == d] for d in np.unique(day)]
    groups = [g for g in groups if g.size > 0]
    k = len(groups)
    n = sum(g.size for g in groups)
    if k < 2 or n <= k:
        return 0.0
    gm = y.mean()
    ssb = sum(g.size * (g.mean() - gm) ** 2 for g in groups)
    ssw = sum(float(np.square(g - g.mean()).sum()) for g in groups)
    msb = ssb / (k - 1)
    msw = ssw / (n - k)
    # Kish's m̄ for unequal clusters
    sizes = np.array([g.size for g in groups], dtype=np.float64)
    m_bar = (n - (np.square(sizes).sum() / n)) / (k - 1)
    if m_bar <= 1:
        return 0.0
    if msw <= 0:
        # zero within-day variance: the day fully determines the label.
        # That is ICC = 1 (maximal clustering), not 0 — returning 0 here
        # would have told the ladder that perfectly clustered samples are
        # perfectly independent, the exact opposite of the truth.
        return 0.999 if msb > 0 else 0.0
    icc = (msb - msw) / (msb + (m_bar - 1) * msw)
    return float(min(max(icc, 0.0), 0.999))


def design_effect(sizes: np.ndarray, icc: float) -> float:
    """DEFF = 1 + (m̄ − 1)·ICC. Clustered samples carry less information
    than independent ones; this is how much less."""
    sizes = np.asarray(sizes, dtype=np.float64)
    if sizes.size == 0:
        return 1.0
    n = sizes.sum()
    m_bar = float(np.square(sizes).sum() / n) if n else 1.0
    return float(max(1.0, 1.0 + (m_bar - 1.0) * icc))


# ------------------------------------------------------------------- AUC
def auc_se(auc: float, n_pos: float, n_neg: float) -> float:
    """Hanley & McNeil (1982) standard error of a single AUC."""
    a = float(min(max(auc, 1e-6), 1 - 1e-6))
    n1, n0 = max(float(n_pos), 1.0), max(float(n_neg), 1.0)
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    var = (a * (1 - a) + (n1 - 1) * (q1 - a * a)
           + (n0 - 1) * (q2 - a * a)) / (n1 * n0)
    return float(math.sqrt(max(var, 1e-12)))


def min_detectable_auc(n_pos: float, n_neg: float,
                       power: float = 0.80, alpha: float = 0.05) -> float:
    """Smallest AUC distinguishable from 0.5 at this sample size, one-
    sided. Bisection on A of f(A) = (A−0.5)/SE(A) − (z_α + z_β)."""
    if n_pos < 2 or n_neg < 2:
        return 1.0
    need = _ppf(1 - alpha) + _ppf(power)

    def f(a: float) -> float:
        return (a - 0.5) / auc_se(a, n_pos, n_neg) - need

    lo, hi = 0.5 + 1e-6, 0.999
    if f(hi) < 0:
        return 1.0                       # even a perfect ranker is unclear
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return float(hi)


# ----------------------------------------------------------------- ladder
STAGES = ("BLIND", "SCREEN", "DISCOVER", "PROMOTE")


@dataclass
class Capability:
    n: int
    n_days: int
    base_rate: float
    n_kish: float
    icc: float
    deff: float
    n_eff: float
    n_eff_pos: float
    n_eff_neg: float
    mde_auc: float                 # minimum detectable AUC, this vault
    promote_bar: float             # META_MIN_AUC
    stage: str
    stage_idx: int
    samples_per_day: float
    days_to_promote_power: int     # more trading days needed, 0 = ready
    reason: str

    def allows(self, stage: str) -> bool:
        return self.stage_idx >= STAGES.index(stage)

    def as_dict(self) -> dict:
        return asdict(self)


def assess(y, w, day, power: float | None = None,
           alpha: float | None = None) -> Capability:
    """The nightly verdict on the vault's own resolving power."""
    y = np.asarray(y, dtype=np.float64)
    w = (np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64))
    day = np.asarray(day)
    power = float(power if power is not None
                  else getattr(config, "LADDER_POWER", 0.80))
    alpha = float(alpha if alpha is not None
                  else getattr(config, "LADDER_ALPHA", 0.05))
    bar = float(getattr(config, "META_MIN_AUC", 0.52))
    n = int(y.size)
    days = np.unique(day)
    n_days = int(days.size)
    if n == 0 or n_days == 0:
        return Capability(0, 0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                          bar, "BLIND", 0, 0.0, 9999,
                          "empty vault — nothing to assess")
    base = float(y.mean())
    n_kish = kish_effective_n(w)
    icc = day_icc(y, day)
    sizes = np.array([int((day == d).sum()) for d in days], dtype=np.float64)
    # v9.9.5 CORRECTION: the Kish weight collapse and the cluster design
    # effect were charging for the SAME dependency twice. AFML uniqueness
    # weights already discount overlapping samples — an overlap that is
    # overwhelmingly intra-day — so applying DEFF built on RAW day sizes on
    # top of n_kish double-counts it. On the 2026-08-03 vault that drove
    # n_eff to the n_days floor (26) and reported an impossible-looking
    # MDE 0.784 / "999 days". The standard combined form (Kish 1965;
    # Gabler, Häder & Lahiri 1999) multiplies a weighting design effect by
    # a clustering one whose mean cluster size is the EFFECTIVE size, so
    # scale the cluster sizes by the same Kish ratio the weights implied.
    _ratio = (n_kish / float(n)) if n else 1.0
    deff = design_effect(sizes * max(_ratio, 1e-6), icc)
    n_eff = max(min(n_kish / deff, float(n)), float(n_days))
    n_eff_pos = n_eff * base
    n_eff_neg = n_eff * (1.0 - base)
    mde = min_detectable_auc(n_eff_pos, n_eff_neg, power, alpha)
    if mde <= bar:
        stage = "PROMOTE"
    elif mde <= float(getattr(config, "LADDER_DISCOVER_MDE", 0.55)):
        stage = "DISCOVER"
    elif mde <= float(getattr(config, "LADDER_SCREEN_MDE", 0.60)):
        stage = "SCREEN"
    else:
        stage = "BLIND"
    spd = n / max(n_days, 1)
    # days-to-power: n_eff scales ∝ n for fixed weight/ICC structure, so
    # project forward at the observed rate and re-solve.
    need_days = 0
    if stage != "PROMOTE":
        scale = n_eff / max(float(n), 1.0)       # eff samples per raw sample
        for extra in range(1, 501):
            n2 = (n_days + extra) * spd
            e2 = max(n2 * scale, float(n_days + extra))
            if min_detectable_auc(e2 * base, e2 * (1 - base),
                                  power, alpha) <= bar:
                need_days = extra
                break
        else:
            need_days = 999
    reason = (f"n={n} over {n_days} day(s); Kish {n_kish:.0f}, ICC "
              f"{icc:.3f} ⇒ DEFF {deff:.2f} ⇒ n_eff {n_eff:.0f}; at "
              f"power {power:.0%} the smallest AUC this vault can "
              f"separate from chance is {mde:.3f} (promotion bar "
              f"{bar:.2f})")
    return Capability(n, n_days, base, n_kish, icc, deff, n_eff,
                      n_eff_pos, n_eff_neg, mde, bar, stage,
                      STAGES.index(stage), spd, need_days, reason)


# ------------------------------------------------- multiple-testing tools
def benjamini_hochberg(pvals, q: float = 0.10):
    """BH step-up. Returns (reject_mask, adjusted_p). Controls the false
    DISCOVERY rate — the right criterion when screening many candidate
    horizons or features, where a few false positives are tolerable but
    a flood is not (Benjamini & Hochberg 1995)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.zeros(0, bool), np.zeros(0)
    order = np.argsort(p)
    ranked = p[order]
    # BH adjusted p at rank i (1-based) = min_{j≥i} (m/j)·p_(j), then
    # made monotone by a right-to-left running minimum. The rank vector
    # is 1..m — an earlier m..1 here inverted the correction and let
    # noise through.
    adj = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out_adj = np.empty(m)
    out_adj[order] = adj
    return out_adj <= q, out_adj


def cluster_bootstrap_auc_p(scores, y, day, n_boot: int = 2000,
                            seed: int = 7) -> tuple[float, float, float]:
    """One-sided p-value for AUC > 0.5 by DAY-CLUSTER bootstrap, plus the
    90% CI. Resampling whole days (not rows) is what keeps the intra-day
    correlation from manufacturing significance — the same idiom the
    toxicity report already uses."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    day = np.asarray(day)
    days = np.unique(day)
    idx_by_day = {d: np.nonzero(day == d)[0] for d in days}
    rng = np.random.default_rng(seed)

    def _auc(s, yy):
        pos, neg = s[yy > 0.5], s[yy <= 0.5]
        if pos.size == 0 or neg.size == 0:
            return np.nan
        r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
        return float((r[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0)
                     / (pos.size * neg.size))

    point = _auc(scores, y)
    if not np.isfinite(point):
        return float("nan"), float("nan"), 1.0
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(days, size=days.size, replace=True)
        rows = np.concatenate([idx_by_day[d] for d in pick])
        boots[b] = _auc(scores[rows], y[rows])
    boots = boots[np.isfinite(boots)]
    if boots.size < 50:
        return point, float("nan"), 1.0
    lo = float(np.percentile(boots, 5))
    # one-sided p: bootstrap mass at or below chance
    p = float((boots <= 0.5).mean())
    p = min(max(p, 1.0 / (boots.size + 1)), 1.0)
    return point, lo, p


# ---------------------------------------------------------------- selftest
if __name__ == "__main__":                                  # pragma: no cover
    import sys
    ok = 0

    def chk(name, cond):
        global ok
        print(("  PASS  " if cond else "  FAIL  ") + name)
        ok += bool(cond)

    # --- normal quantiles
    chk("ppf(0.975)≈1.95996", abs(_ppf(0.975) - 1.959964) < 1e-4)
    chk("ppf(0.80)≈0.84162", abs(_ppf(0.80) - 0.841621) < 1e-4)

    # --- Kish
    chk("kish equal weights = n", abs(kish_effective_n(np.ones(100)) - 100) < 1e-9)
    w = np.array([1.0] * 50 + [0.0001] * 50)
    chk("kish uneven < n", kish_effective_n(w) < 55)

    # --- ICC: identical days ⇒ ~0; day-driven labels ⇒ high
    rng = np.random.default_rng(0)
    d = np.repeat(np.arange(30), 50)
    y_rand = (rng.random(1500) < 0.3).astype(float)
    chk("ICC ≈ 0 when days alike", day_icc(y_rand, d) < 0.05)
    y_day = np.repeat((rng.random(30) < 0.5).astype(float), 50)
    chk("ICC high when label is the day", day_icc(y_day, d) > 0.9)

    # --- DEFF monotone in ICC
    sz = np.full(30, 50.0)
    chk("DEFF grows with ICC",
        design_effect(sz, 0.0) < design_effect(sz, 0.1)
        < design_effect(sz, 0.5))

    # --- Hanley SE shrinks with n; MDE shrinks with n
    chk("SE falls with n", auc_se(0.7, 50, 150) > auc_se(0.7, 500, 1500))
    m_small = min_detectable_auc(50, 150)
    m_big = min_detectable_auc(5000, 15000)
    chk("MDE falls with n", m_small > m_big)
    chk("MDE small-n is large (>0.6)", m_small > 0.60)
    chk("MDE big-n is tight (<0.53)", m_big < 0.53)

    # --- the ladder on THEIR vault shape: 1584 samples, 32 days, 26.8% wins
    rng = np.random.default_rng(3)
    days32 = np.repeat([f"d{i:02d}" for i in range(32)], 1584 // 32)
    yv = (rng.random(days32.size) < 0.268).astype(float)
    wv = rng.uniform(0.05, 0.25, days32.size)      # uniqueness ≈ 0.0996 mean
    capv = assess(yv, wv, days32)
    print(f"\n  → live-shaped vault: {capv.reason}")
    print(f"  → stage {capv.stage} | days to promotion power: "
          f"{capv.days_to_promote_power}")
    chk("live-shaped vault is NOT promote-powered", capv.stage != "PROMOTE")
    chk("assess reports a finite MDE", 0.5 < capv.mde_auc <= 1.0)
    chk("days-to-power is a positive projection",
        capv.days_to_promote_power > 0)

    # a deep vault must unlock PROMOTE
    days300 = np.repeat([f"d{i:03d}" for i in range(300)], 60)
    yb = (rng.random(days300.size) < 0.30).astype(float)
    capb = assess(yb, np.ones(days300.size), days300)
    chk("deep vault unlocks PROMOTE", capb.stage == "PROMOTE")
    chk("deep vault needs 0 more days", capb.days_to_promote_power == 0)
    chk("allows() respects the ladder",
        capb.allows("SCREEN") and not capv.allows("PROMOTE"))

    # --- BH: all-null keeps discoveries near zero; planted signal is found
    pnull = np.random.default_rng(5).random(200)
    rej, _ = benjamini_hochberg(pnull, 0.10)
    chk("BH on pure noise rejects ≤ 5% of 200", rej.sum() <= 10)
    pmix = np.concatenate([np.full(5, 1e-6), np.random.default_rng(6).random(95)])
    rej2, _ = benjamini_hochberg(pmix, 0.10)
    chk("BH finds the 5 planted", rej2[:5].all())

    # --- cluster bootstrap: honest on noise, powered on real signal
    dd = np.repeat(np.arange(40), 40)
    yy = (np.random.default_rng(8).random(1600) < 0.3).astype(float)
    ss = np.random.default_rng(9).random(1600)
    _a, _lo, pnoise = cluster_bootstrap_auc_p(ss, yy, dd, n_boot=400)
    chk("cluster bootstrap: noise not significant", pnoise > 0.05)
    ss2 = yy * 0.6 + np.random.default_rng(10).random(1600) * 0.4
    a2, lo2, psig = cluster_bootstrap_auc_p(ss2, yy, dd, n_boot=400)
    chk("cluster bootstrap: planted signal significant",
        psig < 0.05 and a2 > 0.6 and lo2 > 0.5)

    total = 21
    print(f"\n{ok}/{total} checks passed")
    sys.exit(0 if ok == total else 1)