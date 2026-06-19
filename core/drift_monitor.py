"""
APEX OMNI v9 — FEATURE-DRIFT MONITOR (the live regime-shift guard)
=================================================================
A certified edge is proven on PAST tape. The day the market moves into a
regime the live model never trained on, that model's win-probabilities
become extrapolation — confident fiction. This module is the guard that
catches it, with real statistics, per feature, live.

How it works, end to end:
  1. TRAINING TIME — the forge calls build_reference() on the exact feature
     matrix it trained the meta-model on. For every signal-carrying feature
     it stores quantile bin edges (equal-frequency) + the reference histogram
     + mean/std. That JSON is the model's "this is the world I learned"
     fingerprint, tagged with the config hash and the model timestamp.

  2. LIVE — the brain feeds every per-second frame's features into a rolling
     window (DriftMonitor.observe). Each heartbeat it calls assess(), which
     for every key feature computes:
        • PSI  Σ (live% − ref%)·ln(live%/ref%)  over the reference bins
               — the industry-standard population-stability index.
        • KS   max |CDF_live − CDF_ref|  — distribution-free two-sample gap.
     A feature is "moderate" if PSI ≥ DRIFT_PSI_MODERATE, "significant" if
     PSI ≥ DRIFT_PSI_SIGNIFICANT or KS ≥ DRIFT_KS_SIGNIFICANT.

  3. GRADE — from the fraction of key features breaching:
        GREEN    nothing meaningful moved → trade normally.
        WATCH    ≥ DRIFT_WATCH_FRAC moderate → log it, keep trading; you've
                 been warned the tape is shifting.
        DRIFTED  ≥ DRIFT_DEARM_FRAC significant → write drift_state and
                 de-certify: live arming is blocked until a fresh nightly
                 forge re-references the model to the new regime (or the
                 regime reverts). Paper keeps running so you keep learning.

Pure numpy. No look-ahead, no synthetic anything — the reference is YOUR
training data, the live sample is YOUR live tape, both through the one
StateBuilder. Honest limits printed in the state file: drift detection is
necessary, not sufficient — a stationary-looking tape can still be one the
edge simply stops working on; this catches the DISTRIBUTIONAL shifts, which
are the ones that silently invalidate a trained model.
"""
from __future__ import annotations
import json
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np

import config
from core.market_state import FEATURE_NAMES

log = logging.getLogger("drift")

_FEAT_IDX = {name: i for i, name in enumerate(FEATURE_NAMES)}


# ============================================================ reference
def build_reference(feature_matrix: np.ndarray, *, model_version: str,
                    path: Path | None = None) -> dict:
    """Called by the forge. feature_matrix: (rows, 19) of the SAME per-node
    features the meta-model trained on (spot/atm nodes pooled is fine — the
    point is the feature marginals). Stores equal-frequency bin edges + the
    reference histogram per key feature."""
    path = Path(path or config.DRIFT_PROFILE_PATH)
    X = np.asarray(feature_matrix, float)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"reference matrix must be (rows,{len(FEATURE_NAMES)})")
    if X.shape[0] > config.DRIFT_REF_MAX_SAMPLES:
        idx = np.random.default_rng(7).choice(
            X.shape[0], config.DRIFT_REF_MAX_SAMPLES, replace=False)
        X = X[idx]
    prof = {}
    for name in config.DRIFT_KEY_FEATURES:
        col = X[:, _FEAT_IDX[name]]
        col = col[np.isfinite(col)]
        if col.size < 50 or np.allclose(col, col[0]):
            continue                       # near-constant → not informative
        qs = np.linspace(0, 1, config.DRIFT_BINS + 1)
        edges = np.quantile(col, qs)
        edges[0], edges[-1] = -np.inf, np.inf     # open tails
        edges = _dedupe(edges)
        hist, _ = np.histogram(col, bins=edges)
        ref_pct = hist / max(hist.sum(), 1)
        prof[name] = {"edges": edges.tolist(),
                      "ref_pct": ref_pct.tolist(),
                      "mean": float(col.mean()), "std": float(col.std())}
    out = {"ts": time.time(), "model_version": model_version,
           "config_hash": config.CONFIG_HASH, "n_ref": int(X.shape[0]),
           "features": prof}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(path)
    log.info("drift reference written: %d features over %d rows (model %s)",
             len(prof), X.shape[0], model_version)
    return out


def _dedupe(edges: np.ndarray) -> np.ndarray:
    """Quantile edges can tie on heavy point-mass features (e.g. depth_grad=0).
    Merge duplicates so bins are strictly increasing."""
    out = [edges[0]]
    for e in edges[1:]:
        if e > out[-1]:
            out.append(e)
    if len(out) < 3:                       # degenerate → simple 3-bin spread
        lo = out[0] if np.isfinite(out[0]) else -1.0
        hi = out[-1] if np.isfinite(out[-1]) else 1.0
        return np.array([-np.inf, (lo + hi) / 2, np.inf])
    return np.array(out)


# ============================================================ live monitor
def _psi(ref_pct: np.ndarray, live_pct: np.ndarray) -> float:
    eps = 1e-6
    r = np.clip(ref_pct, eps, None)
    l = np.clip(live_pct, eps, None)
    return float(np.sum((l - r) * np.log(l / r)))


def _ks_from_hist(ref_pct: np.ndarray, live_pct: np.ndarray) -> float:
    return float(np.max(np.abs(np.cumsum(ref_pct) - np.cumsum(live_pct))))


class DriftMonitor:
    """Rolling live feature window vs the forge's reference profile."""

    def __init__(self, profile_path: Path | None = None):
        self.path = Path(profile_path or config.DRIFT_PROFILE_PATH)
        self.profile = None
        self.loaded_ts = 0.0
        self.buf: dict[str, deque] = {
            n: deque(maxlen=config.DRIFT_REF_MAX_SAMPLES)
            for n in config.DRIFT_KEY_FEATURES}
        self._reload()

    def _reload(self):
        try:
            mt = self.path.stat().st_mtime
        except FileNotFoundError:
            self.profile = None
            return
        if mt <= self.loaded_ts:
            return
        try:
            self.profile = json.loads(self.path.read_text())
            self.loaded_ts = mt
            log.info("drift reference loaded (model %s, %d features)",
                     self.profile.get("model_version"),
                     len(self.profile.get("features", {})))
        except Exception as e:                            # noqa: BLE001
            log.warning("drift reference unreadable: %s", e)

    def observe(self, frame_30x19: np.ndarray):
        """Feed one live frame. Pools the SPOT + ATM CE/PE nodes of every
        TRADABLE index (the nodes that actually drive decisions)."""
        for idx in config.TRADABLE:
            b = config.INDEX_ORDER.index(idx) * config.NODES_PER_INDEX
            for node in (frame_30x19[b], frame_30x19[b + 1], frame_30x19[b + 2]):
                if not node.any():
                    continue
                for name in config.DRIFT_KEY_FEATURES:
                    v = float(node[_FEAT_IDX[name]])
                    if np.isfinite(v):
                        self.buf[name].append(v)

    def _live_count(self) -> int:
        return min((len(d) for d in self.buf.values()), default=0)

    def assess(self) -> dict:
        """Compute PSI + KS per key feature; grade GREEN/WATCH/DRIFTED."""
        self._reload()
        n = self._live_count()
        if self.profile is None:
            return {"grade": "NO_REF", "n_live": n,
                    "msg": "no reference yet — forge has not trained a model"}
        if self.profile.get("config_hash") != config.CONFIG_HASH:
            return {"grade": "NO_REF", "n_live": n,
                    "msg": "reference is for a different config — retrain"}
        if n < config.DRIFT_MIN_LIVE_SAMPLES:
            return {"grade": "WARMUP", "n_live": n,
                    "need": config.DRIFT_MIN_LIVE_SAMPLES}
        feats = self.profile["features"]
        per = {}
        moderate = significant = considered = 0
        for name, ref in feats.items():
            live = np.array(self.buf[name], float)
            if live.size < config.DRIFT_MIN_LIVE_SAMPLES:
                continue
            edges = np.array(ref["edges"], float)
            ref_pct = np.array(ref["ref_pct"], float)
            hist, _ = np.histogram(live, bins=edges)
            live_pct = hist / max(hist.sum(), 1)
            psi = _psi(ref_pct, live_pct)
            ks = _ks_from_hist(ref_pct, live_pct)
            sig = psi >= config.DRIFT_PSI_SIGNIFICANT or \
                ks >= config.DRIFT_KS_SIGNIFICANT
            mod = (not sig) and psi >= config.DRIFT_PSI_MODERATE
            considered += 1
            significant += int(sig)
            moderate += int(mod)
            per[name] = {"psi": round(psi, 3), "ks": round(ks, 3),
                         "level": "significant" if sig else
                         ("moderate" if mod else "stable"),
                         "live_mean": round(float(live.mean()), 4),
                         "ref_mean": round(ref["mean"], 4)}
        if considered == 0:
            return {"grade": "WARMUP", "n_live": n}
        sig_frac = significant / considered
        mod_frac = (moderate + significant) / considered
        if sig_frac >= config.DRIFT_DEARM_FRAC:
            grade = "DRIFTED"
        elif mod_frac >= config.DRIFT_WATCH_FRAC:
            grade = "WATCH"
        else:
            grade = "GREEN"
        worst = sorted(per.items(), key=lambda kv: -kv[1]["psi"])[:5]
        res = {"ts": time.time(), "grade": grade, "n_live": n,
               "features_considered": considered,
               "significant": significant, "moderate": moderate,
               "sig_frac": round(sig_frac, 3), "mod_frac": round(mod_frac, 3),
               "model_version": self.profile.get("model_version"),
               "worst": {k: v for k, v in worst}, "per_feature": per}
        self._persist(res)
        return res

    def _persist(self, res: dict):
        try:
            tmp = config.DRIFT_STATE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(res))
            tmp.replace(config.DRIFT_STATE_PATH)
        except Exception as e:                            # noqa: BLE001
            log.debug("drift state write: %s", e)


# ============================================================ gate helper
def drift_blocks_live() -> bool:
    """Read by config.live_fire_armed(): True iff the latest live assessment
    graded DRIFTED and is recent (within one trading day). A regime the model
    never saw blocks REAL orders; paper keeps running."""
    try:
        s = json.loads(config.DRIFT_STATE_PATH.read_text())
        if s.get("grade") != "DRIFTED":
            return False
        return (time.time() - float(s.get("ts", 0))) < 86400
    except Exception:                                     # noqa: BLE001
        return False
