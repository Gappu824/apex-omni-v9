"""
APEX OMNI v9.9.1 — BOCPD (Bayesian Online Changepoint Detection), corrected
============================================================================
Adams & MacKay (2007) on the RV-forecast innovations x_t = log(rv̂_t/rv̂_{t−1}).
The regime machine answers "which regime am I in"; BOCPD answers the sharper
question "did the world just BREAK".

Two corrections over the first cut, found by its own planted-break proof:
  1. DETECTION STATISTIC. P(r_t = 0) renormalizes to ≈hazard every step BY
     CONSTRUCTION (the changepoint and growth terms share the same Σ r·pred),
     so it can never spike. The correct statistic is the posterior mass on
     SHORT run-lengths — after a true break the recursion collapses onto them
     within a few observations. cp_prob := P(run ≤ 5), with a 30-obs warmup
     (early on, all runs are short by definition).
  2. TRIM. A boolean-mask trim destroys the index↔run-length identity the
     recursion depends on. Trim is now PREFIX-preserving: keep [0..last
     above threshold], cap at max_run, fold truncated tail mass into the
     last kept run.

Model: unknown-mean/variance Gaussian segments, Normal-Inverse-Gamma prior →
exact Student-t predictive. O(run-length) per update. TELEMETRY ONLY by
constitution: cp_prob/map_run ride the rv block; any gate consuming them is
a registered spec first.
"""
from __future__ import annotations

import math

import numpy as np

_SHORT_RUNS = 6          # cp_prob = mass on runs 0..5
_WARMUP = 30             # below this n, "all runs are short" is vacuous


class BOCPD:
    def __init__(self, hazard_lambda: float = 250.0,
                 mu0: float = 0.0, kappa0: float = 1.0,
                 alpha0: float = 1.0, beta0: float = 1e-4,
                 trim: float = 1e-6, max_run: int = 5000):
        self.h = 1.0 / float(hazard_lambda)
        self.mu0, self.k0 = float(mu0), float(kappa0)
        self.a0, self.b0 = float(alpha0), float(beta0)
        self.trim, self.max_run = float(trim), int(max_run)
        self.r = np.array([1.0])
        self.mu = np.array([self.mu0])
        self.k = np.array([self.k0])
        self.a = np.array([self.a0])
        self.b = np.array([self.b0])
        self.n = 0

    @staticmethod
    def _t_logpdf(x, df, loc, scale):
        z = (x - loc) / scale
        return (math.lgamma((df + 1) / 2) - math.lgamma(df / 2)
                - 0.5 * math.log(df * math.pi) - math.log(scale)
                - (df + 1) / 2 * math.log1p(z * z / df))

    def update(self, x: float) -> dict:
        x = float(x)
        df = 2.0 * self.a
        scale = np.sqrt(self.b * (self.k + 1.0) / (self.a * self.k))
        logpred = np.array([self._t_logpdf(x, df[i], self.mu[i], scale[i])
                            for i in range(len(self.r))])
        m = logpred.max()
        pred = np.exp(logpred - m)                 # relative densities: the
        growth = self.r * pred * (1.0 - self.h)    # common e^m cancels in the
        cp = float(np.sum(self.r * pred) * self.h)  # normalization below
        new_r = np.concatenate([[cp], growth])
        tot = float(new_r.sum())
        if tot <= 0 or not math.isfinite(tot):
            hz = 1.0 / self.h
            self.__init__(hz, self.mu0, self.k0, self.a0, self.b0,
                          self.trim, self.max_run)
            return {"cp_prob": 1.0, "map_run": 0, "n": self.n}
        new_r /= tot
        mu_n = np.concatenate([[self.mu0],
                               (self.k * self.mu + x) / (self.k + 1.0)])
        k_n = np.concatenate([[self.k0], self.k + 1.0])
        a_n = np.concatenate([[self.a0], self.a + 0.5])
        b_n = np.concatenate([[self.b0],
                              self.b + self.k * (x - self.mu) ** 2
                              / (2.0 * (self.k + 1.0))])
        above = np.nonzero(new_r > self.trim)[0]
        last = int(above.max()) + 1 if above.size else 2
        last = min(max(last, 2), self.max_run, len(new_r))
        tail = float(new_r[last:].sum())
        self.r = new_r[:last].copy()
        self.r[-1] += tail
        self.r /= self.r.sum()
        self.mu, self.k = mu_n[:last], k_n[:last]
        self.a, self.b = a_n[:last], b_n[:last]
        self.n += 1
        short = float(self.r[:min(_SHORT_RUNS, len(self.r))].sum())
        cp_prob = 0.0 if self.n < _WARMUP else short
        return {"cp_prob": round(cp_prob, 4),
                "map_run": int(np.argmax(self.r)), "n": self.n}