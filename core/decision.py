"""
APEX OMNI v9.1 — SHARED DECISION PATH (audit fix: one exam, one system)
=======================================================================
The v9 audit found the forge grading a DIFFERENT trading system than the one
deployed: the brain gated entries at the fused, regime-multiplied conviction
with a persistence check, while `_gen_meta_samples` / `_grade_like_live` gated
raw heuristic output with none of that. This module is the fix: every stage of
the entry decision — advisory shock, logit fusion, regime scaling, effective
bar, meta/calibration win-probability, persistence, the entry gate itself —
lives HERE and only here. `apex_main_v9` and `nightly_forge_v9` both import it,
so train/serve skew in the decision distribution is structurally impossible,
the same way core.market_state made it impossible for features.

Second audit fix, applied here: the regime multiplier operates in LOGIT space.
The old `conv *= mult` composed a bounded tanh with a sub-unity multiplier and
a fixed 0.70 bar, which made CHOP (×0.70) and VOL_CRUSH (×0.65) arithmetically
un-enterable — a hard veto the classifier's contract explicitly forbids.
`tanh(atanh(conv) · mult)` dampens by RAISING the raw-signal requirement
(CHOP now needs pre-mult |conv| ≥ 0.845 instead of ≥ 1.0) and boosts without
saturating, so "scale, never veto" is true by construction:

        regime      old requirement      new requirement
        CHOP  ×0.70   |conv| ≥ 1.000 ✗    |conv| ≥ 0.845 ✓
        CRUSH ×0.65   |conv| ≥ 1.077 ✗    |conv| ≥ 0.869 ✓

Third audit fix: the persistence window is WALL-CLOCK. The old deque of the
last 4 brain-loop iterations spanned ~0.8 s at the 5 Hz loop — not the
"sustained read" the docstrings promised — and its config knobs
(SIGNAL_PERSIST_FRAC / AVG_MULT) were dead. PersistenceTracker keys samples on
timestamps and evicts by SIGNAL_PERSIST_WINDOW_S, so "held for N seconds"
means N seconds at any loop cadence (brain ~5 Hz, forge replay 1 Hz — same
window, same test).

No torch, no kiteconnect: importable by every process.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config
from core.quant_core import bayesian_signal_fusion
from core.signal_persistence import assess_persistence


# --------------------------------------------------------------------------
# 1. ADVISORY SHOCK — the exact stack the brain applied inline (VPIN pressure,
#    gamma-flip side, PCR extremes, expiry max-pain gravity, prev-day level
#    breaks, Gao–Han–Li–Zhou late-day intraday momentum). One implementation;
#    the forge passes whatever real context the vault/archive holds and each
#    missing piece degrades to zero exactly as a missing macro JSON does live.
# --------------------------------------------------------------------------
def compute_shock(*, ai: float, vpin: float, dealer_inv: float,
                  mac: dict | None, spot: float, dte: float,
                  levels: dict | None, f30: float, hm: str) -> float:
    shock = 0.0
    if vpin > config.ADVISORY_VPIN_THRESHOLD:
        shock += config.ADVISORY_SHOCK * math.copysign(1.0, dealer_inv or ai or 1.0)
    flip = (mac or {}).get("flip")
    if flip:
        shock += config.ADVISORY_SHOCK * (1.0 if spot > flip else -1.0)
    pcr = (mac or {}).get("pcr")
    if pcr is not None:
        if pcr >= config.PCR_HIGH:            # crowded puts → contrarian up
            shock += config.ADVISORY_SHOCK_PCR
        elif pcr <= config.PCR_LOW:
            shock -= config.ADVISORY_SHOCK_PCR
    mp = (mac or {}).get("max_pain")
    if mp and float(dte or 9.0) < 1.0:        # expiry-day pin gravity
        shock += config.ADVISORY_SHOCK_MAXPAIN * (1.0 if mp > spot else -1.0)
    if levels:
        if levels.get("pdh") and spot > levels["pdh"]:
            shock += config.ADVISORY_SHOCK_LEVELS
        elif levels.get("pdl") and spot < levels["pdl"]:
            shock -= config.ADVISORY_SHOCK_LEVELS
    if f30 and hm >= config.IMOM_AFTER:       # first-30-min → last-hour i-mom
        shock += config.ADVISORY_SHOCK_IMOM * (1.0 if f30 > 0 else -1.0)
    return shock


def fuse(ai: float, shock: float) -> float:
    """Logit-space fusion, single shared copy (delegates to quant_core)."""
    return bayesian_signal_fusion(ai, shock,
                                  quant_weight=config.FUSION_QUANT_WEIGHT)


# --------------------------------------------------------------------------
# 2. REGIME SCALING — logit space (the veto fix; see module docstring).
# --------------------------------------------------------------------------
def apply_regime(conv: float, conv_mult: float) -> float:
    if conv == 0.0 or conv_mult == 1.0:
        return conv
    c = float(np.clip(conv, -0.999999, 0.999999))
    return math.tanh(math.atanh(c) * float(conv_mult))


def effective_bar(base_bar: float, vix_bump: float,
                  iv_rank: float | None) -> float:
    """Base conviction bar plus the VIX-spike and IV-rank premiums — the
    identical composition the brain used inline."""
    bar = base_bar + vix_bump
    if iv_rank is not None and iv_rank >= config.IVRANK_HIGH:
        bar += config.IVRANK_BAR_BUMP
    return bar


# --------------------------------------------------------------------------
# 3. WIN PROBABILITY — meta-model logistic + calibration-table blend. Moved
#    verbatim from apex_main (meta_win_prob / win_prob_for / the 50-50 blend)
#    so the forge's grader computes P(win) with the same bytes.
# --------------------------------------------------------------------------
def meta_win_prob(meta: dict | None, frame: np.ndarray, iidx: int,
                  tod: float, er: float, f30: float, dirn: int,
                  clamp: bool = True,
                  conv_by_index=None) -> float | None:
    """clamp=False returns the TRUE calibrated probability (no META_P_FLOOR
    lift) — for telemetry/diagnosis. Decisions keep clamp=True."""
    if not meta or int(meta.get("n", 0)) < config.META_MIN_TRAIN:
        return None
    b0 = iidx * config.NODES_PER_INDEX
    # CROSS-INDEX PEER CONTEXT (config.META_CROSS_INDEX). Appended AFTER the
    # existing 61 so an old artifact's feature order is untouched; the x_dim
    # guard in meta_gbm refuses a model trained on the other width.
    _peer = []
    if bool(getattr(config, "META_CROSS_INDEX", False)):
        from core.cross_index import peer_features, N_PEER_FEATURES
        if conv_by_index is None:
            _peer = [0.0] * N_PEER_FEATURES
            if not globals().get("_PEER_WARNED"):
                globals()["_PEER_WARNED"] = True
                import logging as _lg
                _lg.getLogger("decision").warning(
                    "META_CROSS_INDEX is ON but no conviction vector was "
                    "passed — serving zeros where the forge trained on real "
                    "peer context. That is a train/serve SKEW; pass "
                    "conv_by_index at every call site.")
        else:
            _peer = peer_features(conv_by_index, iidx,
                                  "CE" if dirn > 0 else "PE")
    x = np.concatenate([frame[b0], frame[b0 + 1], frame[b0 + 2],
                        [tod, er,
                         math.copysign(min(abs(f30) * 100, 3), f30)
                         if f30 else 0.0,
                         1.0 if dirn > 0 else -1.0], _peer]).astype(np.float32)
    if meta.get("engine") == "gbm":
        from core import meta_gbm as MG
        return MG.score_vec(meta, x, clamp=clamp)
    mu = np.asarray(meta["mu"], np.float32)
    sd = np.asarray(meta["sd"], np.float32)
    w = np.asarray(meta["w"], np.float32)
    z = (x - mu) / np.where(sd > 0.0, sd, 1.0)
    pr = 1.0 / (1.0 + math.exp(-float(z @ w) - float(meta["b"])))
    if not clamp:
        return float(pr)
    return float(min(max(pr, config.META_P_FLOOR), config.META_P_CAP))


def cal_bucket(conv: float) -> str:
    w = config.CAL_BUCKET_WIDTH
    return f"{min(abs(conv) // w * w, 1 - w):.2f}"


def blend_winprob(wp_meta: float | None, conv: float, cal: dict) -> float:
    bkey = cal_bucket(conv)
    cal_hit = bkey in cal and cal[bkey][1] >= config.CAL_MIN_SAMPLES
    if wp_meta is None:
        return float(cal[bkey][0]) if cal_hit else config.uncalibrated_winprob()
    if cal_hit:
        return 0.5 * (wp_meta + float(cal[bkey][0]))   # blend both judges
    return wp_meta


# --------------------------------------------------------------------------
# 4. PERSISTENCE — wall-clock window over the SAME assess_persistence physics.
# --------------------------------------------------------------------------
class PersistenceTracker:
    """Rolling (ts, conv) window evicted by wall-clock seconds. `check` runs
    core.signal_persistence.assess_persistence over the surviving samples, so
    the coherence/ER/tape-agreement science is unchanged — only the window's
    meaning is fixed (seconds, not loop iterations)."""

    __slots__ = ("samples",)

    def __init__(self):
        self.samples: deque = deque()

    def push(self, ts: float, conv: float) -> None:
        self.samples.append((float(ts), float(conv)))
        horizon = float(getattr(config, "SIGNAL_PERSIST_WINDOW_S", 12.0))
        while self.samples and self.samples[0][0] < ts - horizon:
            self.samples.popleft()

    @property
    def latest(self) -> float | None:
        return self.samples[-1][1] if self.samples else None

    def check(self, conv_now: float, spot_window,
              conv_floor: float) -> tuple[bool, str, dict]:
        if not config.SIGNAL_PERSIST_ENABLED:
            return True, "persistence disabled", {}
        min_n = int(getattr(config, "SIGNAL_PERSIST_MIN_SAMPLES", 4))
        if len(self.samples) < min_n:
            return True, "warming up", {"n": len(self.samples)}
        window = [c for _, c in self.samples]
        return assess_persistence(conv_now, window, list(spot_window or ()),
                                  conv_floor=conv_floor)


# --------------------------------------------------------------------------
# 5. THE GATE — model-driven when a trained meta exists, else the bootstrap
#    conviction bar. Byte-for-byte the brain's decision block, importable.
# --------------------------------------------------------------------------
@dataclass
class GateResult:
    ok: bool
    reason: str            # why blocked (or the gate description if ok)
    model_driven: bool
    floor: float           # the persistence conv_floor the caller must use


def entry_gate(conv: float, wp: float, wp_meta: float | None,
               eff_bar: float) -> GateResult:
    model_driven = bool(config.META_DECISION_ENABLED and wp_meta is not None)
    if model_driven:
        if abs(conv) < config.META_ENTRY_CONV_FLOOR:
            return GateResult(False, f"conv {abs(conv):.2f}<"
                              f"{config.META_ENTRY_CONV_FLOOR:.2f} floor",
                              True, config.META_ENTRY_CONV_FLOOR)
        if wp < config.META_ENTRY_P_BAR:
            return GateResult(False, f"meta P(win) {wp:.2f}<"
                              f"{config.META_ENTRY_P_BAR:.2f}",
                              True, config.META_ENTRY_CONV_FLOOR)
        return GateResult(True, f"meta P(win) {wp:.2f}≥"
                          f"{config.META_ENTRY_P_BAR:.2f}",
                          True, config.META_ENTRY_CONV_FLOOR)
    if abs(conv) < eff_bar:
        return GateResult(False, f"conv {abs(conv):.2f}<{eff_bar:.2f} bar",
                          False, eff_bar)
    return GateResult(True, f"conv {abs(conv):.2f}≥{eff_bar:.2f}",
                      False, eff_bar)


# --------------------------------------------------------------------------
# 6. GATE FUNNEL — the "why is it flat" diagnostic both the brain and the
#    forge grader report. Every second, every index, records exactly one
#    outcome: which gate stopped the entry, or that one was placed. The daily
#    JSON report turns "it doesn't trade" from a mystery into a table.
# --------------------------------------------------------------------------
_GATES = ("stale_feed", "no_market", "in_position", "risk_halted",
          "below_bar", "not_persistent", "retest_guard", "toxicity_trap",
          "throttled", "no_chain", "no_quotes", "risk_blocked", "no_fill",
          "entered")


class GateFunnel:
    def __init__(self, indices):
        self.counts = {i: {g: 0 for g in _GATES} for i in indices}
        self.block_detail: dict[str, dict] = {i: {} for i in indices}

    def record(self, idx: str, gate: str, detail: str | None = None) -> None:
        c = self.counts.setdefault(idx, {g: 0 for g in _GATES})
        c[gate] = c.get(gate, 0) + 1
        if detail:
            d = self.block_detail.setdefault(idx, {})
            d[detail] = d.get(detail, 0) + 1

    def as_dict(self) -> dict:
        out = {}
        for idx, c in self.counts.items():
            top = sorted(self.block_detail.get(idx, {}).items(),
                         key=lambda kv: -kv[1])[:8]
            out[idx] = {"gates": dict(c), "top_block_reasons": dict(top)}
        return out

    def line(self, idx: str) -> str:
        c = self.counts.get(idx, {})
        seen = {g: n for g, n in c.items() if n}
        order = sorted(seen.items(), key=lambda kv: -kv[1])[:5]
        body = " ".join(f"{g}={n}" for g, n in order) or "no ticks yet"
        return f"{idx}[{body}]"