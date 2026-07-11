"""
APEX OMNI v9.5 — ROBUST STATISTICS CORE (Pillar 5 shared instruments)
=====================================================================
Two instruments the certificates and exams lean on:

• STATIONARY BOOTSTRAP (Politis–Romano 1994): resamples a dependent series
  in geometric-length blocks (mean length 1/p), preserving serial dependence
  the iid bootstrap destroys. Day-PnL series are autocorrelated (regimes
  cluster); a certificate CI that ignores that is optimistic by construction.

• CPCV SPLITTER (López de Prado ch.12): combinatorial purged cross-validation
  — N groups of days, every C(N,k) choice of k test groups, PURGING training
  days adjacent to test boundaries (embargo) so labels never leak across the
  fit. Returns splits AND the reassembled backtest paths.

Pure numpy/stdlib; deterministic under a supplied seed.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def stationary_bootstrap_means(x, n_boot: int = 4000, p: float = 1.0 / 3.0,
                               seed: int = 20260710) -> np.ndarray:
    """Bootstrap distribution of the MEAN of a dependent series.
    Mean block length = 1/p (default 3 observations — days)."""
    x = np.asarray(x, float)
    n = x.size
    if n == 0:
        return np.zeros(0)
    if n == 1:
        return np.full(n_boot, float(x[0]))
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(0, n)
        for t in range(n):
            idx[t] = i
            if rng.random() < p:
                i = rng.integers(0, n)          # new block start
            else:
                i = (i + 1) % n                 # continue block (circular)
        out[b] = float(x[idx].mean())
    return out


def stationary_ci_lo(x, ci: float = 0.90, n_boot: int = 4000,
                     p: float = 1.0 / 3.0, seed: int = 20260710) -> float | None:
    m = stationary_bootstrap_means(x, n_boot, p, seed)
    if m.size == 0:
        return None
    return float(np.quantile(m, (1.0 - ci) / 2.0))


def cpcv_splits(days: list[str], n_groups: int = 6, k_test: int = 2,
                embargo: int = 1):
    """Combinatorial purged CV over an ORDERED day list.
    Returns (splits, paths):
      splits: list of dicts {test_groups, test_days, train_days} where
              train excludes test days AND `embargo` days on each side of
              every test group (purge — LdP 7.4/12).
      paths:  φ = C(N,k)·k/N reassembled full-sample test paths — lists of
              (split_index, day) covering every day exactly once per path.
    """
    days = list(days)
    n = len(days)
    if n < n_groups:
        n_groups = max(2, n)
    # contiguous groups (chronology preserved)
    bounds = np.linspace(0, n, n_groups + 1).astype(int)
    groups = [days[bounds[g]:bounds[g + 1]] for g in range(n_groups)]
    gidx = {d: g for g, grp in enumerate(groups) for d in grp}
    splits = []
    for combo in combinations(range(n_groups), k_test):
        test_days = [d for g in combo for d in groups[g]]
        test_pos = {days.index(d) for d in test_days}
        banned = set()
        for pos in test_pos:
            for e in range(-embargo, embargo + 1):
                banned.add(pos + e)
        train_days = [d for i, d in enumerate(days)
                      if i not in banned and gidx[d] not in combo]
        splits.append({"test_groups": combo, "test_days": test_days,
                       "train_days": train_days})
    # path assembly: each group appears in C(N-1,k-1) splits ⇒ that many paths
    per_group = [ [si for si, s in enumerate(splits)
                   if g in s["test_groups"]] for g in range(n_groups)]
    n_paths = len(per_group[0]) if per_group else 0
    paths = []
    for pth in range(n_paths):
        path = []
        for g in range(n_groups):
            si = per_group[g][pth]
            path += [(si, d) for d in groups[g]]
        paths.append(path)
    return splits, paths