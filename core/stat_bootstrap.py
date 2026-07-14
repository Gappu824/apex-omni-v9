"""
APEX OMNI v10 — STATIONARY BOOTSTRAP (Politis–Romano 1994)
==========================================================
Event P&Ls inside a session are serially dependent (shared regime, shared
vol day); the iid bootstrap's CI is therefore too NARROW — flattering. The
stationary bootstrap resamples geometric-length blocks (wrapping), preserving
short-range dependence, and its lower bound is the honest one. Mean block
length defaults to the Politis–White-style n^(1/3) rule. Reported in every
certificate beside the iid bound as `stationary_ci_lo` (diagnostic tier; the
pass rule stays on the iid bound until a doctrine trial says otherwise).
"""
from __future__ import annotations

import numpy as np


def stat_boot_lo(pnls, ci: float, n_boot: int,
                 mean_block: float | None = None,
                 seed: int = 20260709) -> float | None:
    r = np.asarray(pnls, float)
    n = len(r)
    if n < 5:
        return None
    L = float(mean_block) if mean_block else max(n ** (1.0 / 3.0), 2.0)
    p = 1.0 / L
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        out = np.empty(n)
        i = rng.integers(0, n)
        for t in range(n):
            out[t] = r[i]
            i = rng.integers(0, n) if rng.random() < p else (i + 1) % n
        means[b] = out.mean()
    return float(np.quantile(means, (1.0 - ci) / 2.0))