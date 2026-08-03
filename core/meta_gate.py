"""
APEX OMNI v9.9 — META-GATE v3 (the gate that knows what it doesn't know)
=========================================================================
Retires the fixed-bar point-probability veto. Three research pillars:

1. VENN-ABERS PREDICTION INTERVALS (Vovk & Petej 2014; van der Laan &
   Alaa 2025 generalize the framework). For a query score s, fit TWO
   isotonic regressions on the calibration set augmented with (s, y=0)
   and (s, y=1); the fitted values at s are (p0, p1). Under
   exchangeability the TRUE probability lies in [p0, p1] — a FINITE-
   SAMPLE guarantee, unlike lone isotonic (asymptotic only), which is
   exactly the difference that matters at this vault's sample counts.
   The log-loss-optimal single number is p = p1 / (1 - p0 + p1).
   Interval WIDTH is honest epistemic uncertainty where scores are
   sparse; on a COLLAPSED calibrator (the "constant 0.23" / "OOF spread
   0.004" incidents) it degrades to the honest base rate — and the EV
   gate then judges THAT against p*, instead of a fake edge.

2. PER-TRADE EV GATE. The labels are first-touch on SHAPED barriers
   (per-signal tp/sl), so P(win) means P(hit tp before sl). The break-
   even probability p* therefore differs per trade:
        win  = (tp - e)·lot - costs(e, tp)
        loss = (e - sl)·lot + costs(e, sl)
        p*   = loss / (win + loss)
   computed with the SAME _shaped_barriers physics and the SAME
   round_trip_costs the label generator uses (single copy, both import
   from here). A fixed 0.55 bar demanded p ≥ 0.55 from a 2:1-odds
   trade that breaks even at 0.34 — that is the mechanism that blocked
   good trades. The EV gate asks the only question that pays:
   is the trade +EV at the probability the model can DEFEND?

3. THREE-ZONE DECISION + ACI-STYLE ADAPTATION.
        FULL  : p0 ≥ p* + m   (even the pessimistic bound is +EV)
        VETO  : p1 <  p* + m   (even the optimistic bound loses money)
        PROBE : otherwise      (the model cannot rule the edge in OR
                                out → enter at MINIMUM size inside the
                                full risk constitution; buy information
                                instead of asserting ignorance as "no")
   m is an online margin in the spirit of Adaptive Conformal Inference
   (Gibbs & Candès 2021): after each resolved meta-mode trade,
   m += γ·(p_served − outcome). A model that systematically overstates
   p tightens its own gate; one that understates unblocks itself. m is
   clipped to ±META_ACI_MAX and soft-resets on artifact promotion.

STRUCTURAL SAFETY (the 0.23-constant class of bug, killed at the root):
   the artifact carries per-feature TRAINING stats (q05/q95/alive). At
   serve time a rolling window watches the live x-vectors; if features
   that varied in training freeze live (train/serve skew — dead frame
   nodes, wrong builder, schema drift), the gate drops to MONITOR: it
   stops vetoing (passes at probe size), and says so loudly. A gate
   must never fail-closed because of its own plumbing.

FAIL-OPEN BY DESIGN: no VA payload / no meta / mode!="ev" → callers
fall back to the legacy fixed-bar gate byte-for-byte. Nothing here can
cost a session; it can only refuse to pretend.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
from core.quant_core import implied_vol_newton, black76_greeks
from core.execution_engine import round_trip_costs

log = logging.getLogger("meta_gate")


# ═══════════════════════════════════════════════════════════════════════
# 0. SHAPED BARRIERS — the single copy. Was nightly_forge_v9._shaped_
#    barriers; the forge now imports it from here so the LABEL physics,
#    the LIVE p* physics and the GRADER p* physics can never drift.
#    Byte-identical math to the forge original (v9.1 ask-entry basis).
# ═══════════════════════════════════════════════════════════════════════
def shaped_barriers(e, spot, K, T, mins, is_call,
                    call_wall=None, put_wall=None):
    """The live PositionManager.try_enter exit target, reproduced for the
    reward AND for the EV gate's per-trade breakeven.

        em        = spot · atm_iv · √(minutes_to_close / (252·375))
        spot_room = min(em, runway)                       # GEX wall cap
        prem_room = delta_at_entry · spot_room
        target    = entry + max(prem_room, entry · BASE_TP_PCT)
        stop      = entry · (1 − BASE_SL_PCT)

    atm_iv is Newton-inverted from the leg's OWN entry price `e` (the
    ASK live pays on a momentum cross); delta is Black-76 on that same
    iv, `abs(delta) or 0.4` exactly as live. Vectorized or scalar."""
    r = config.RISK_FREE_RATE
    e = np.asarray(e, float); spot = np.asarray(spot, float)
    K = np.asarray(K, float); T = np.maximum(np.asarray(T, float), 1e-6)
    F = spot * np.exp(r * T)
    iv = implied_vol_newton(e, F, K, T, is_call, r)
    delta = np.abs(np.asarray(
        black76_greeks(F, K, T, iv, is_call, r)["delta"], float))
    delta = np.where(delta > 1e-9, delta, 0.4)
    em = spot * iv * np.sqrt(np.maximum(mins, 1.0) / (252.0 * 375.0))
    spot_room = em
    if is_call and call_wall is not None and call_wall > 0:
        runway = call_wall - spot
        spot_room = np.where(runway > 0, np.minimum(em, runway), em)
    elif (not is_call) and put_wall is not None and put_wall > 0:
        runway = spot - put_wall
        spot_room = np.where(runway > 0, np.minimum(em, runway), em)
    prem_room = delta * spot_room
    tp = e + np.maximum(prem_room, e * config.BASE_TP_PCT)
    sl = e * (1.0 - config.BASE_SL_PCT)
    return tp, sl


# ═══════════════════════════════════════════════════════════════════════
# 1. WEIGHTED PAVA returning the FITTED VECTOR (not pooled breakpoints —
#    Venn-Abers needs the fitted value at one specific inserted index).
# ═══════════════════════════════════════════════════════════════════════
def _pava_fit(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-decreasing) fit of y ordered as given.
    Stack-based pool-adjacent-violators; O(n). Returns fitted values."""
    n = len(y)
    # blocks: (value, weight, count)
    vals: list[float] = []
    wts: list[float] = []
    cnt: list[int] = []
    for i in range(n):
        cv, cw, cc = float(y[i]), float(w[i]), 1
        while vals and vals[-1] > cv:
            pv, pw, pc = vals.pop(), wts.pop(), cnt.pop()
            cv = (cv * cw + pv * pw) / (cw + pw)
            cw += pw
            cc += pc
        vals.append(cv); wts.append(cw); cnt.append(cc)
    out = np.empty(n, float)
    pos = 0
    for v, c in zip(vals, cnt):
        out[pos:pos + c] = v
        pos += c
    return np.clip(out, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# 2. VENN-ABERS PREDICTOR — exact IVAP on a (score, label, weight)
#    calibration set. Weights extend the classical (unweighted) validity
#    guarantee heuristically, consistent with the uniqueness-weighted
#    training upstream (AFML ch.4); the query point carries weight 1.0
#    (training weights are normalized to mean 1, so this is the mean).
# ═══════════════════════════════════════════════════════════════════════
class VennAbers:
    __slots__ = ("s", "y", "w", "n")

    def __init__(self, scores, labels, weights=None):
        s = np.asarray(scores, float)
        y = np.asarray(labels, float)
        w = (np.ones_like(s) if weights is None
             else np.asarray(weights, float))
        order = np.argsort(s, kind="stable")
        self.s, self.y, self.w = s[order], y[order], np.maximum(w[order], 1e-6)
        self.n = len(self.s)

    def _fit_with(self, q: float, label: float) -> float:
        """Isotonic fit of cal ∪ {(q, label)}; return fitted value at q.
        TIES ARE POOLED FIRST: isotonic on x must be constant on equal x,
        so exact score ties (LightGBM emits many) collapse to one weighted
        point BEFORE PAVA — the query joins its tie block. Without this,
        a constant-score calibrator (the 0.23 incident) yields invalid
        step functions inside the tie and a wrong interval."""
        pos = int(np.searchsorted(self.s, q, side="right"))
        s2 = np.insert(self.s, pos, q)
        y2 = np.insert(self.y, pos, label)
        w2 = np.insert(self.w, pos, 1.0)
        ux, inv = np.unique(s2, return_inverse=True)
        gw = np.bincount(inv, weights=w2)
        gy = np.bincount(inv, weights=y2 * w2) / np.maximum(gw, 1e-12)
        fitted = _pava_fit(gy, gw)
        return float(fitted[inv[pos]])

    def interval(self, q: float) -> tuple[float, float]:
        """(p0, p1) — the Venn-Abers span for query score q."""
        if self.n == 0:
            return 0.0, 1.0
        p0 = self._fit_with(q, 0.0)
        p1 = self._fit_with(q, 1.0)
        if p1 < p0:                       # numerically possible on ties
            p0, p1 = p1, p0
        return p0, p1

    @staticmethod
    def merge(p0: float, p1: float) -> float:
        """Log-loss-optimal single probability from the VA pair."""
        den = 1.0 - p0 + p1
        return float(p1 / den) if den > 1e-9 else 0.5


# serve-time cache: one VennAbers per artifact (keyed on artifact ts) —
# same discipline as meta_gbm._BOOSTERS (one live model; don't leak).
_VA_CACHE: dict = {}


def va_from_artifact(meta: dict) -> VennAbers | None:
    va = meta.get("va")
    if not va or not va.get("s"):
        return None
    key = (float(meta.get("ts", 0.0)), len(va["s"]))
    hit = _VA_CACHE.get(key)
    if hit is None:
        hit = VennAbers(va["s"], va["y"], va.get("w"))
        _VA_CACHE.clear()
        _VA_CACHE[key] = hit
    return hit


# ═══════════════════════════════════════════════════════════════════════
# 3. PER-TRADE BREAKEVEN — the same cost stack the labels charged.
# ═══════════════════════════════════════════════════════════════════════
def breakeven_p(entry: float, tp: float, sl: float, lot: int) -> float:
    """p* such that p·win − (1−p)·loss = 0, Zerodha costs included on
    BOTH exit branches. Returns 1.0 (untradeable) if the win branch
    cannot even pay its costs."""
    lot = max(int(lot), 1)
    e_v = float(entry) * lot
    win = (float(tp) - float(entry)) * lot - round_trip_costs(
        e_v, float(tp) * lot)
    loss = (float(entry) - float(sl)) * lot + round_trip_costs(
        e_v, float(sl) * lot)
    if win <= 0.0:
        return 1.0
    if loss <= 0.0:                       # cannot lose ⇒ any p is +EV
        return 0.0
    return float(loss / (win + loss))


def candidate_economics(entry_ask: float, spot: float, strike: float,
                        T: float, mins_left: float, is_call: bool,
                        lot: int, call_wall=None, put_wall=None
                        ) -> tuple[float, float, float] | None:
    """(p_star, tp_pct, sl_pct) for THIS candidate leg — the shaped-
    barrier payoff the labels were trained on, expressed both as the
    breakeven the GATE needs and the percentages the KELLY sizer needs,
    so gating and sizing can never price two different trades. None ⇒
    inputs cannot support the physics (caller falls back, fail-open)."""
    try:
        if not (entry_ask and entry_ask > 0 and spot > 0 and strike > 0
                and T and T > 0):
            return None
        tp, sl = shaped_barriers(entry_ask, spot, strike, T,
                                 max(mins_left, 1.0), is_call,
                                 call_wall, put_wall)
        tp, sl = float(tp), float(sl)
        ps = breakeven_p(entry_ask, tp, sl, lot)
        return (ps, max(tp / entry_ask - 1.0, 1e-4),
                max(1.0 - sl / entry_ask, 1e-4))
    except Exception as e:                                 # noqa: BLE001
        log.debug("candidate_economics failed (%s)", e)
        return None


def pstar_for_candidate(entry_ask: float, spot: float, strike: float,
                        T: float, mins_left: float, is_call: bool,
                        lot: int, call_wall=None, put_wall=None
                        ) -> float | None:
    """Live/grader helper: shaped barriers for THIS candidate leg → p*.
    Returns None when inputs cannot support the physics (caller falls
    back to the legacy gate for this evaluation — fail-open)."""
    try:
        if not (entry_ask and entry_ask > 0 and spot > 0 and strike > 0
                and T and T > 0):
            return None
        tp, sl = shaped_barriers(entry_ask, spot, strike, T,
                                 max(mins_left, 1.0), is_call,
                                 call_wall, put_wall)
        return breakeven_p(entry_ask, float(tp), float(sl), lot)
    except Exception as e:                                 # noqa: BLE001
        log.debug("pstar_for_candidate failed (%s)", e)
        return None


# ═══════════════════════════════════════════════════════════════════════
# 4. ACI-STYLE ONLINE MARGIN — persisted, per book, soft-reset on
#    artifact promotion. Direction: m rises when the model OVERSTATES
#    p (served − outcome > 0), tightening the gate; falls when it
#    understates, unblocking. Clipped to ±META_ACI_MAX.
# ═══════════════════════════════════════════════════════════════════════
class AdaptiveMargin:
    def __init__(self, book: str):
        self.book = str(book)
        self.path = Path(getattr(config, "META_GATE_STATE",
                                 config.STATE_DIR / "meta_gate_aci.json"))
        self._j = self._load()

    def _load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except Exception:                                  # noqa: BLE001
            return {}

    def _save(self) -> None:
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._j))
            os.replace(tmp, self.path)
        except Exception as e:                             # noqa: BLE001
            log.warning("ACI state save failed (%s)", e)

    @property
    def m(self) -> float:
        d = self._j.get(self.book) or {}
        cap = float(getattr(config, "META_ACI_MAX", 0.10))
        return float(min(max(float(d.get("m", 0.0)), -cap), cap))

    def update(self, p_served: float, won: bool) -> float:
        g = float(getattr(config, "META_ACI_GAMMA", 0.02))
        cap = float(getattr(config, "META_ACI_MAX", 0.10))
        d = self._j.setdefault(self.book, {"m": 0.0, "n": 0})
        d["m"] = float(min(max(d.get("m", 0.0)
                               + g * (float(p_served) - (1.0 if won else 0.0)),
                               -cap), cap))
        d["n"] = int(d.get("n", 0)) + 1
        d["ts"] = time.time()
        self._save()
        log.info("ACI[%s] update: served %.3f outcome %d → margin %+.4f "
                 "(n=%d)", self.book, p_served, 1 if won else 0,
                 d["m"], d["n"])
        return d["m"]

    def on_promotion(self, artifact_ts: float) -> None:
        """New artifact ⇒ the margin was learned against the OLD model.
        Soft-reset (halve) rather than zero: miscalibration is partly a
        property of the pipeline, not only of one fit."""
        d = self._j.setdefault(self.book, {"m": 0.0, "n": 0})
        if abs(float(d.get("art_ts", 0.0)) - float(artifact_ts)) > 1.0:
            d["art_ts"] = float(artifact_ts)
            old = float(d.get("m", 0.0))
            d["m"] = old * 0.5
            self._save()
            if abs(old) > 1e-6:
                log.info("ACI[%s] soft-reset on promotion: %+.4f → %+.4f",
                         self.book, old, d["m"])


# ═══════════════════════════════════════════════════════════════════════
# 5. FEATURE INTEGRITY — train/serve skew tripwire. The artifact carries
#    per-feature training stats; a rolling window of live x-vectors is
#    compared against them. Trip ⇒ MONITOR (never veto), loud, throttled.
# ═══════════════════════════════════════════════════════════════════════
class FeatureIntegrity:
    def __init__(self):
        self._buf: list[np.ndarray] = []
        self._status = "OK"
        self._last_err = 0.0
        self._frozen: list[int] = []

    def observe(self, x: np.ndarray, feat_alive) -> str:
        """feat_alive: artifact's boolean list — features whose TRAINING
        variance was alive. Returns "OK" | "SKEW"."""
        win = int(getattr(config, "META_FEAT_WINDOW", 240))
        need = max(win // 2, 30)
        self._buf.append(np.asarray(x, np.float32))
        if len(self._buf) > win:
            self._buf.pop(0)
        if feat_alive is None or len(self._buf) < need:
            return self._status if self._buf else "OK"
        A = np.stack(self._buf)
        alive = np.asarray(feat_alive, bool)
        if alive.size != A.shape[1]:
            return self._status                 # x-dim guard owns this case
        span = A.max(axis=0) - A.min(axis=0)
        frozen = np.nonzero(alive & (span < 1e-9))[0]
        trip = int(getattr(config, "META_FEAT_FROZEN_MIN", 8))
        new_status = "SKEW" if len(frozen) >= trip else "OK"
        if new_status != self._status or (
                new_status == "SKEW"
                and time.time() - self._last_err > 900):
            self._last_err = time.time()
            self._frozen = [int(i) for i in frozen[:16]]
            if new_status == "SKEW":
                log.error("META FEATURE SKEW: %d features that varied in "
                          "TRAINING are frozen LIVE over the last %d "
                          "evaluations (first: %s). The model is being fed "
                          "a different world than it learned — its vetoes "
                          "are not trustworthy. Gate → MONITOR (no vetoes, "
                          "probe-size passes) until the feed heals or the "
                          "forge re-runs. (Repeating ≤ every 15 min.)",
                          len(frozen), len(self._buf), self._frozen)
            else:
                log.info("META FEATURE SKEW cleared — gate resumes EV mode.")
        self._status = new_status
        return self._status


# ═══════════════════════════════════════════════════════════════════════
# 6. THE DECISION — zones from (interval, p*, margin, integrity).
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class GateDecision:
    zone: str            # FULL | PROBE | VETO | MONITOR | LEGACY
    ok: bool             # may the entry proceed at all?
    probe: bool          # if ok: minimum-size entry?
    reason: str
    p0: float = float("nan")
    p1: float = float("nan")
    p: float = float("nan")      # merged (log-loss-optimal) probability
    p_star: float = float("nan")
    margin: float = 0.0


def decide(p0: float, p1: float, p_star: float, margin: float,
           integrity: str = "OK") -> GateDecision:
    p = VennAbers.merge(p0, p1)
    ev_m = float(getattr(config, "META_EV_MARGIN", 0.02))
    bar = float(p_star) + float(margin) + ev_m
    if integrity != "OK":
        return GateDecision(
            "MONITOR", True, True,
            f"feature skew — monitor mode (p∈[{p0:.2f},{p1:.2f}] "
            f"unreliable); probe size only", p0, p1, p, p_star, margin)
    if p0 >= bar:
        return GateDecision(
            "FULL", True, False,
            f"EV: pessimistic p0 {p0:.2f} ≥ p* {p_star:.2f}"
            f"{margin + ev_m:+.2f}", p0, p1, p, p_star, margin)
    if p1 < bar:
        return GateDecision(
            "VETO", False, False,
            f"EV: optimistic p1 {p1:.2f} < p* {p_star:.2f}"
            f"{margin + ev_m:+.2f}", p0, p1, p, p_star, margin)
    return GateDecision(
        "PROBE", True, True,
        f"EV ambiguous: p*[{p_star:.2f}]∈(p0 {p0:.2f}, p1 {p1:.2f}) "
        f"— probe size", p0, p1, p, p_star, margin)


# ═══════════════════════════════════════════════════════════════════════
# 7. SELF-TEST — python -m core.meta_gate  (synthetic; no vault needed)
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    rng = np.random.default_rng(7)
    fails = 0

    def check(name, cond):
        global fails
        print(("  ✓ " if cond else "  ✗ ") + name)
        if not cond:
            fails += 1

    # --- 1. PAVA correctness vs brute expectation on a known case
    y = np.array([1, 0, 1, 1, 0, 1], float)
    w = np.ones(6)
    f = _pava_fit(y, w)
    check("PAVA monotone", bool(np.all(np.diff(f) >= -1e-12)))
    check("PAVA preserves weighted mean",
          abs(f.mean() - y.mean()) < 1e-9)

    # --- 2. VA validity on synthetic: p_true = sigmoid(score)
    n = 1200
    s = rng.normal(0, 1.4, n)
    p_true = 1 / (1 + np.exp(-s))
    yy = (rng.random(n) < p_true).astype(float)
    va = VennAbers(s, yy)
    qs = np.clip(rng.normal(0, 1.4, 300), -2.0, 2.0)   # bulk region
    hits = wid = 0.0
    for q in qs:
        p0, p1 = va.interval(float(q))
        pt = 1 / (1 + math.exp(-q))
        hits += (p0 - 0.05 <= pt <= p1 + 0.05)
        wid += (p1 - p0)
    check(f"VA brackets truth ({hits/300:.0%} within ±0.05, bulk)",
          hits / 300 >= 0.92)
    check(f"VA intervals informative (mean width {wid/300:.3f} < 0.15)",
          wid / 300 < 0.15)

    # --- 3. THE COLLAPSE SCENARIO — constant scores (the 0.23 incident).
    # Lone isotonic served a confident constant; VA must serve ≈base-rate
    # with an interval that admits ignorance beyond it.
    sc = np.zeros(400)
    yc = (rng.random(400) < 0.42).astype(float)
    vac = VennAbers(sc, yc)
    p0, p1 = vac.interval(0.0)
    pm = VennAbers.merge(p0, p1)
    check(f"collapse → honest base-rate {pm:.3f}≈0.42 (not a fake edge)",
          abs(pm - yc.mean()) < 0.05)
    check("collapse → interval contains base rate",
          p0 - 1e-9 <= yc.mean() <= p1 + 1e-9)

    # --- 4. Tiny calibration set → wide interval (epistemic honesty)
    v3 = VennAbers([0.1, 0.9], [0, 1])
    p0t, p1t = v3.interval(0.5)
    check(f"n=2 → wide interval ({p1t - p0t:.2f} ≥ 0.30)", p1t - p0t >= 0.30)

    # --- 5. Breakeven math vs brute-force EV zero-crossing
    e_, tp_, sl_, lot_ = 100.0, 140.0, 80.0, 75
    ps = breakeven_p(e_, tp_, sl_, lot_)
    win = (tp_ - e_) * lot_ - round_trip_costs(e_ * lot_, tp_ * lot_)
    loss = (e_ - sl_) * lot_ + round_trip_costs(e_ * lot_, sl_ * lot_)
    ev_at = ps * win - (1 - ps) * loss
    check(f"p* {ps:.4f}: EV(p*) = ₹{ev_at:+.4f} ≈ 0", abs(ev_at) < 1e-6)
    check("2:1 odds ⇒ p* well below 0.55 (the trades the old bar killed)",
          ps < 0.40)
    check("cost-swamped micro trade ⇒ p*→1 (untradeable, veto)",
          breakeven_p(1.0, 1.05, 0.9, 1) == 1.0)

    # --- 6. Zone truth table
    d = decide(0.50, 0.62, 0.34, 0.0)          # asymmetric winner
    check("FULL when p0 clears p*", d.zone == "FULL" and d.ok
          and not d.probe)
    d = decide(0.30, 0.44, 0.55, 0.0)
    check("VETO when even p1 < p*", d.zone == "VETO" and not d.ok)
    d = decide(0.40, 0.60, 0.50, 0.0)
    check("PROBE on straddle", d.zone == "PROBE" and d.ok and d.probe)
    d = decide(0.40, 0.60, 0.50, 0.0, integrity="SKEW")
    check("MONITOR never vetoes", d.zone == "MONITOR" and d.ok and d.probe)
    d = decide(0.56, 0.62, 0.50, 0.08)         # margin tightens
    check("ACI margin shifts the bar", d.zone == "PROBE")

    # --- 7. ACI convergence: model overstates by +0.10 ⇒ m → ≈ +0.10
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        config.META_GATE_STATE = Path(td) / "aci.json"
        am = AdaptiveMargin("equity")
        for _ in range(400):
            p_served = 0.60
            won = rng.random() < 0.50          # truth 0.50: overstated
            am.update(p_served, won)
        check(f"ACI margin ≈ overconfidence (+0.10): m={am.m:+.3f}",
              0.06 <= am.m <= 0.10)
        am.on_promotion(123.0)
        check("promotion soft-reset halves m", abs(am.m) < 0.06)

    # --- 8. Integrity tripwire on frozen features
    fi = FeatureIntegrity()
    alive = [True] * 20
    st = "OK"
    for k in range(200):
        x = np.zeros(20, np.float32)
        x[:10] = rng.normal(size=10)           # 10 live, 10 frozen-alive
        st = fi.observe(x, alive)
    check("skew trips on ≥8 frozen-but-trained features", st == "SKEW")
    fi2 = FeatureIntegrity()
    for k in range(200):
        st2 = fi2.observe(rng.normal(size=20).astype(np.float32), alive)
    check("healthy stream stays OK", st2 == "OK")

    # --- 9. shaped_barriers: sane geometry + wall cap engages
    tp1, sl1 = shaped_barriers(120.0, 24000.0, 24000.0, 5 / 365, 180.0,
                               True)
    tp2, _ = shaped_barriers(120.0, 24000.0, 24000.0, 5 / 365, 180.0,
                             True, call_wall=24010.0)
    check("tp>entry>sl and wall caps the target",
          float(tp1) > 120.0 > float(sl1) and float(tp2) <= float(tp1))

    print("\n%d/17 checks passed" % (17 - fails))
    raise SystemExit(1 if fails else 0)