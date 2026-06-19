"""
APEX OMNI v9 — TRAP SHIELD (the anti-"institutional flush" layer)
=================================================================
The pattern you described — you enter, price gets slammed just far enough to
panic retail out of their longs, then the contract rips — is a real and
well-documented microstructure event. It goes by many names: stop hunt,
liquidity sweep, Wyckoff "spring", stop-run. Mechanically it has a signature
that differs from a GENUINE breakdown, and that signature is what this
module scores. Six fingerprints, each cheap to compute from data the system
already has:

 1. VELOCITY ANOMALY   — the flush is a vertical spike (z-score of 1-second
                         spot velocity vs the day's own distribution), not a
                         grind. Real trends walk; hunts teleport.
 2. ABSORPTION         — at the lows, heavy volume prints AT THE BID while
                         price stops falling: someone large is quietly
                         buying everything panicking retail sells (the
                         iceberg detector + tick-rule flow measure this).
                         A real breakdown shows the opposite: price keeps
                         giving way under selling.
 3. OI NON-CONFIRMATION— a genuine bearish break attracts NEW short interest
                         (ΔOI rising on the break). A hunt is position
                         CLOSING — ΔOI flat or falling while price spikes
                         down means no real money is committing to the move.
 4. PREMIUM DISLOCATION— your option's premium drops MORE than its delta ×
                         the spot move justifies → market makers yanked and
                         repriced quotes to maximize panic, the move is in
                         the quotes, not the underlying.
 5. SPREAD BLOWOUT     — bid/ask gaps to several × its rolling average at
                         the exact moment of the spike. Liquidity being
                         pulled is the hunter turning the lights off.
 6. WALL PROXIMITY     — the spike terminates within a fraction of a percent
                         of the GEX put-wall / round-number support, where
                         resting retail stops cluster and where dealer
                         hedging flow flips supportive.

Score ≥ TRAP_SCORE_THRESHOLD ⇒ the normal stop is SUSPENDED for a bounded
grace window (TRAP_MAX_HOLD_S). Non-negotiables that keep the shield from
becoming a bag-holding machine:

  * The DISASTER FLOOR always fires. Always. If price keeps falling to
    1.6× the normal stop distance (capped at −45% of premium), we are wrong,
    trap or no trap, and we exit. The shield bends the stop; nothing breaks it.
  * The grace window is short and the shield may trigger at most
    TRAP_MAX_USES_PER_TRADE times per position.
  * If, during the hold, OI starts CONFIRMING the break or absorption
    vanishes, the shield releases early and the stop executes.
  * If price reclaims TRAP_RECLAIM_PCT of the spike, the trap is confirmed:
    the stop is re-anchored to the hunt's low (now a defended level).

Honesty clause: no detector is perfect. Some real breakdowns begin life
looking exactly like hunts. The floor is the price of that humility.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from collections import deque
import numpy as np

import config


# ----------------------------------------------------------------- learned model
# The nightly forge refits the trap score WEIGHTS and THRESHOLD from real
# stopped-out trades (labeled by whether price reclaimed = hunt, or kept falling
# = genuine breakdown) and writes config.TRAP_MODEL_PATH. Until that file exists
# — i.e. until there are enough real stop-outs to learn from — these fall back to
# the fixed config.TRAP_WEIGHTS / TRAP_SCORE_THRESHOLD (the reasoned guess).
#
# CRITICAL: only the pattern-recognition knobs are learned. The grace window
# (TRAP_MAX_HOLD_S), use cap (TRAP_MAX_USES_PER_TRADE) and disaster floor are the
# risk constitution for the shield — they are NEVER loaded from the model, never
# learned, never moved. A learner optimizing on outcomes would loosen them; their
# job is to bound the tail the data can't show.
_trap_cache: dict = {"mtime": None, "weights": None, "threshold": None}


def _load_trap_model() -> None:
    import json
    import os
    path = config.TRAP_MODEL_PATH
    try:
        mt = os.path.getmtime(path)
    except OSError:
        _trap_cache.update(mtime=None, weights=None, threshold=None)
        return
    if _trap_cache["mtime"] == mt:
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            m = json.load(fh)
        w = {k: float(m["weights"][k]) for k in config.TRAP_WEIGHTS}
        th = float(m["threshold"])
        # sanity clamp: a learned threshold can shift the hold/release balance
        # but is bounded so a degenerate fit can't disable the shield entirely
        # or make it hold on noise. These bounds are fixed, not learned.
        th = float(np.clip(th, config.TRAP_THRESHOLD_MIN,
                           config.TRAP_THRESHOLD_MAX))
        _trap_cache.update(mtime=mt, weights=w, threshold=th)
    except (OSError, KeyError, ValueError, TypeError):
        _trap_cache.update(mtime=None, weights=None, threshold=None)


def _trap_weights() -> dict:
    _load_trap_model()
    return _trap_cache["weights"] or config.TRAP_WEIGHTS


def _trap_threshold() -> float:
    _load_trap_model()
    t = _trap_cache["threshold"]
    return t if t is not None else config.TRAP_SCORE_THRESHOLD


@dataclass
class TrapSignals:
    spot_velocity_1s: float          # signed Δspot over last second
    spot: float
    absorption: bool                 # iceberg-style buying at the lows
    aggressive_sell_ratio: float     # 0..1 share of volume hitting the bid
    oi_delta_break: float            # ΔOI since spike began (norm, signed)
    premium_move_pct: float          # actual premium change since spike start
    delta_implied_move_pct: float    # |delta|·Δspot/entry_premium (what's "fair")
    spread_pct: float
    avg_spread_pct: float
    gex_put_wall: float | None
    gex_call_wall: float | None


@dataclass
class _ShieldState:
    active: bool = False
    started_ts: float = 0.0
    spike_high: float = 0.0          # pre-spike reference price
    spike_low: float = 1e18
    uses: int = 0
    vel_hist: deque = field(default_factory=lambda: deque(maxlen=config.TRAP_VEL_WINDOW_S))


class TrapShield:
    def __init__(self, direction: str):
        self.dirn = 1 if direction == "CE" else -1   # CE long hurt by down-spike
        self.s = _ShieldState()

    # ----------------------------------------------------------- scoring
    def observe(self, spot_velocity_1s: float):
        """Feed EVERY tick's velocity (adverse-signed) so the z-score at a
        breach is measured against the day's own distribution, not an
        empty history."""
        self.s.vel_hist.append(-spot_velocity_1s * self.dirn)

    def _vel_z(self, v: float) -> float:
        h = self.s.vel_hist
        if len(h) < 30:
            return 0.0
        sd = float(np.std(h)) or 1e-9
        return (v - float(np.mean(h))) / sd

    def score(self, sig: TrapSignals) -> tuple[float, dict]:
        adverse_v = -sig.spot_velocity_1s * self.dirn      # >0 = against us
        z = self._vel_z(adverse_v)
        f = {}
        f["velocity"] = float(np.clip(z / config.TRAP_VELOCITY_Z, 0, 1.5)) \
            if z > 0 else 0.0
        f["absorption"] = 1.0 if (sig.absorption and
                                  sig.aggressive_sell_ratio > 0.6) else \
            (0.5 if sig.absorption else 0.0)
        # OI: confirmation of the break is NEGATIVE evidence for a trap
        f["oi"] = float(np.clip(-sig.oi_delta_break * config.TRAP_OI_CONFIRM_SCALE, -1, 1)) * 0.5 + 0.5
        disl = sig.premium_move_pct - (-abs(sig.delta_implied_move_pct))
        f["dislocation"] = float(np.clip(-disl / config.TRAP_DISLOCATION_FULL, 0, 1))
        ratio = sig.spread_pct / max(sig.avg_spread_pct, 1e-4)
        f["spread"] = float(np.clip((ratio - 1) / (config.TRAP_SPREAD_BLOWOUT - 1),
                                    0, 1))
        f["wall"] = 0.0
        wall = sig.gex_put_wall if self.dirn == 1 else sig.gex_call_wall
        if wall and sig.spot > 0:
            prox = abs(sig.spot - wall) / sig.spot
            f["wall"] = 1.0 if prox <= config.TRAP_WALL_PROX_PCT else \
                max(0.0, 1 - prox / (config.TRAP_WALL_PROX_PCT * 4))
        W = _trap_weights()
        return sum(W[k] * min(f[k], 1.0) for k in W), f

    # ----------------------------------------------------------- decision
    def on_stop_breach(self, ts: float, sig: TrapSignals) -> tuple[bool, str, float]:
        """Called ONLY when the normal stop has been breached.
        Returns (hold, reason, score). Disaster floor is checked by the
        PositionManager separately and OVERRIDES any hold."""
        if self.s.active:
            held = ts - self.s.started_ts
            if held > config.TRAP_MAX_HOLD_S:
                self.s.active = False
                return False, f"shield grace expired ({held:.0f}s) — honoring stop", 0.0
            # early release: the market started confirming the break
            if sig.oi_delta_break > 0.02 and not sig.absorption:
                self.s.active = False
                return False, "OI confirming breakdown — releasing shield", 0.0
            self.s.spike_low = min(self.s.spike_low, sig.spot)
            return True, f"shield holding ({held:.0f}s/" \
                         f"{config.TRAP_MAX_HOLD_S}s)", 1.0
        if self.s.uses >= config.TRAP_MAX_USES_PER_TRADE:
            return False, "shield uses exhausted — honoring stop", 0.0
        sc, parts = self.score(sig)
        self.last_fingerprints = parts          # for the nightly learner's dataset
        if sc >= _trap_threshold():
            self.s.active = True
            self.s.started_ts = ts
            self.s.uses += 1
            self.s.spike_high = sig.spot
            self.s.spike_low = sig.spot
            top = sorted(parts.items(), key=lambda kv: -kv[1])[:3]
            why = ", ".join(f"{k}={v:.2f}" for k, v in top)
            return True, f"TRAP suspected (score {sc:.2f}: {why}) — holding", sc
        return False, f"no trap signature (score {sc:.2f}) — honoring stop", sc

    def reclaim_check(self, spot: float) -> bool:
        """True the moment price reclaims enough of the spike → trap CONFIRMED.
        Caller re-anchors the stop just under the hunt's low."""
        if not self.s.active:
            return False
        span = self.s.spike_high - self.s.spike_low
        if span <= 0:
            return False
        reclaimed = (spot - self.s.spike_low) / span if self.dirn == 1 else \
                    (self.s.spike_high - spot) / span
        if reclaimed >= config.TRAP_RECLAIM_PCT:
            self.s.active = False
            return True
        return False

    @property
    def hunt_low(self) -> float:
        return self.s.spike_low

    @property
    def holding(self) -> bool:
        return self.s.active
