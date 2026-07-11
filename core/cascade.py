"""
APEX OMNI v9.2 — GAMMA-CASCADE DETECTOR (shared: harness ≡ brain)
=================================================================
One state machine, imported verbatim by tools/cascade_harness.py (which grades
every historical trigger on the vault) and by apex_main_v9 (which runs it live,
certificate-gated) — the core/decision.py constitutional pattern: the exam and
the trader cannot diverge because they are the same bytes.

THE MECHANISM (published; see config's cascade section for citations): with
spot BELOW the gamma flip and dealers net SHORT gamma, hedging flow trades
WITH the move — impulses are amplified, in BOTH directions. The trigger is
therefore state × impulse, direction from the impulse sign:

    zone(t)   : spot < flip − hyst   AND   net_gex ≤ CASCADE_NET_GEX_MAX
                (hyst = CASCADE_HYST_MULT × flip_width, floored at one strike
                 step — a full-bracket cross, never a tick flicker)
    impulse(t): |Δspot over CASCADE_VEL_WINDOW_S| ≥ CASCADE_VEL_Z σ, where σ
                is that same window-return's own rolling std over
                CASCADE_VOL_LOOKBACK_S (self-normalizing: "large for THIS
                tape", not an absolute point count)
    EVENT     : zone ∧ impulse ∧ cooldown clear ∧ under the daily cap
    direction : PE if the impulse is down, CE if up (a squeeze rally under
                negative gamma is the same amplification with opposite sign)
    kind      : "flip_break" when the zone was entered within the last
                impulse window (the cross IS the news), else "zone_impulse"

NOT prediction — regime-state detection. The trade the detector emits is
graded by the harness EXACTLY as the brain would place it: ask entry, the
standard shaped triple barrier, the 0-DTE-aware guillotine, real costs — so a
certificate, if earned, certifies the deployable trade and nothing else.

Feed cadence contract: update() once per DECISION SECOND (the v9.1.2 ring
cadence live; 1 Hz in replay). Pure stdlib+numpy; no torch, no kite.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from dataclasses import dataclass, asdict

import config


@dataclass
class CascadeEvent:
    ts: float
    index: str
    direction: str            # "PE" (down-impulse) | "CE" (up-impulse)
    kind: str                 # "flip_break" | "zone_impulse"
    z: float                  # impulse z-score (signed)
    spot: float
    flip: float
    flip_width: float
    net_gex: float
    flip_source: str          # "nowcast" | "radar" (stale-sweep fallback)
    flip_age_s: float

    def as_dict(self) -> dict:
        return asdict(self)


class CascadeDetector:
    """Per-index. Deterministic given the fed sequence — replay-identical."""

    def __init__(self, index: str, zone_side: str = "below"):
        # zone_side: "below" (the certified base — spot under the flip) or
        # "above" (the registered upside-variant TRIAL: deep-negative total
        # gamma with spot ABOVE the flip; admitted to the factory by the
        # upside_zone_candidate_s diagnostic). Constructor argument, NOT a
        # config knob — the base spec's knob-hash is untouched.
        self.index = index
        self.zone_side = zone_side
        self.last_z: float | None = None      # v9.2.1: telemetry/diagnostics
        self._spots: deque = deque()          # (ts, spot) for the window return
        self._rets: deque = deque()           # (ts, r_window) history for σ
        self._m = 0.0                         # Welford over current ret deque
        self._s2 = 0.0
        self._in_zone_since: float | None = None
        self._cooldown_until = -1e18
        self._fired_today = 0
        self._day: str | None = None

    # ------------------------------------------------------------ internals
    def _window_return(self, ts: float, spot: float) -> float | None:
        self._spots.append((ts, spot))
        horizon = config.CASCADE_VEL_WINDOW_S
        while self._spots and self._spots[0][0] < ts - config.CASCADE_VOL_LOOKBACK_S:
            self._spots.popleft()
        # oldest sample at-or-before ts − window (right-continuous anchor)
        anchor = None
        for t0, s0 in self._spots:
            if t0 <= ts - horizon:
                anchor = s0
            else:
                break
        return None if anchor is None else spot - anchor

    def _sigma(self, ts: float, r: float) -> float | None:
        self._rets.append((ts, r))
        while self._rets and self._rets[0][0] < ts - config.CASCADE_VOL_LOOKBACK_S:
            self._rets.popleft()
        n = len(self._rets)
        if n < config.CASCADE_VOL_MIN_N:
            return None
        vals = [x for _, x in self._rets]
        m = sum(vals) / n
        var = sum((x - m) ** 2 for x in vals) / max(n - 1, 1)
        return math.sqrt(var) if var > 0 else None

    # ------------------------------------------------------------ public
    def update(self, *, ts: float, day: str, spot: float,
               flip: float | None, flip_width: float | None,
               net_gex: float | None, strike_step: float,
               flip_source: str, flip_age_s: float) -> CascadeEvent | None:
        if day != self._day:                          # fresh session
            self._day = day
            self._fired_today = 0
            self._in_zone_since = None
            self._spots.clear()
            self._rets.clear()
        r = self._window_return(ts, spot)
        if r is None:
            return None
        sig = self._sigma(ts, r)
        if flip is None or net_gex is None or sig is None:
            self.last_z = None
            self._in_zone_since = None
            return None
        z = r / sig
        self.last_z = float(z)
        hyst = max(config.CASCADE_HYST_MULT * float(flip_width or 0.0),
                   float(strike_step))
        if self.zone_side == "above":
            in_zone = (spot > flip + hyst) and \
                (net_gex <= config.CASCADE_NET_GEX_MAX)
        else:
            in_zone = (spot < flip - hyst) and \
                (net_gex <= config.CASCADE_NET_GEX_MAX)
        if in_zone and self._in_zone_since is None:
            self._in_zone_since = ts
        if not in_zone:
            self._in_zone_since = None
            return None
        if abs(z) < config.CASCADE_VEL_Z:
            return None
        if ts < self._cooldown_until:
            return None
        if self._fired_today >= config.CASCADE_MAX_EVENTS_DAY:
            return None
        self._cooldown_until = ts + config.CASCADE_COOLDOWN_S
        self._fired_today += 1
        kind = ("flip_break"
                if ts - (self._in_zone_since or ts) <= config.CASCADE_VEL_WINDOW_S
                else "zone_impulse")
        return CascadeEvent(ts=ts, index=self.index,
                            direction="PE" if z < 0 else "CE", kind=kind,
                            z=float(z), spot=float(spot), flip=float(flip),
                            flip_width=float(flip_width or 0.0),
                            net_gex=float(net_gex), flip_source=flip_source,
                            flip_age_s=float(flip_age_s))


# ==========================================================================
# CERTIFICATE — the falsification lock. Written ONLY by the harness; read by
# the brain. Fingerprints CONFIG_HASH + every knob that shapes the trigger or
# the graded trade, so tuning anything re-opens the lock until re-proven.
#
# v9.2.1 STAGING (backtest → paper → live, the AFML pipeline made executable):
# the 2026-07-09 harness run proved the backtest tier structurally CANNOT
# certify alone — 36 triggers detected, 35 unfillable because pre-2026-07-04
# vaults carry ATM±3 while the ladder walks 8 rungs. Paper forward-testing is
# therefore the evidence engine: cascade_mode() unlocks PAPER-ONLY entries so
# every future trigger becomes an out-of-sample, real-execution-path
# observation (the strongest evidence class — no backtest artifact can touch
# it); log_forward_entry() records the join keys; the harness blends realized
# forward fills into the certificate. LIVE cascade still requires the full
# certificate AND the four live locks — paper explore can never leak real
# money.
# ==========================================================================
FORWARD_LOG = config.STATE_DIR / "cascade_forward.jsonl"


def cascade_mode(cert: dict | None) -> str:
    """'certified' | 'paper-explore' | 'telemetry'. Paper-explore is active
    only when explicitly enabled AND the account is not live-armed — the
    exploration tier is physically incapable of touching real money."""
    if not config.CASCADE_LIVE_ENABLED:
        return "telemetry"
    if cert is not None:
        return "certified"
    if getattr(config, "CASCADE_PAPER_EXPLORE", False) \
            and not config.live_fire_armed():
        return "paper-explore"
    return "telemetry"


def log_forward_entry(row: dict) -> None:
    """Append-only forward-evidence record at ENTRY. symbol + entry_ts are
    the deterministic join keys the harness matches against the execution
    ledger's BUY_FILL/SELL_FILL pairs. Never raises into the trading loop."""
    try:
        FORWARD_LOG.parent.mkdir(exist_ok=True)
        with FORWARD_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:                                 # noqa: BLE001
        pass


def cascade_knob_hash() -> str:
    knobs = (config.CONFIG_HASH,
             config.CASCADE_VEL_WINDOW_S, config.CASCADE_VEL_Z,
             config.CASCADE_VOL_LOOKBACK_S, config.CASCADE_VOL_MIN_N,
             config.CASCADE_NET_GEX_MAX, config.CASCADE_HYST_MULT,
             config.CASCADE_COOLDOWN_S, config.CASCADE_MAX_EVENTS_DAY,
             config.BASE_TP_PCT, config.BASE_SL_PCT, config.MAX_HOLD_MINUTES,
             config.MAX_HOLD_MINUTES_0DTE, config.EXPIRY_DTE_LT,
             config.MAX_ENTRY_SPREAD_PCT)
    return hashlib.sha1(repr(knobs).encode()).hexdigest()[:10]


def load_certificate() -> dict | None:
    """The valid, matching certificate — or None (detector stays
    telemetry-only). Fail-closed on any doubt, exactly like live_fire_armed."""
    try:
        c = json.loads(config.CASCADE_CERT_PATH.read_text())
    except Exception:                                 # noqa: BLE001
        return None
    if not (bool(c.get("ok"))
            and c.get("knob_hash") == cascade_knob_hash()
            and (time.time() - float(c.get("ts", 0)))
            < config.EDGE_CERT_VALID_DAYS * 86400):
        return None
    return c


def certificate_wp(cert: dict) -> float:
    """Sizing win-probability for a certified cascade entry: the harness's
    LOWER-bound win rate (not the point estimate), clamped to a sane band.
    Feeds the normal Kelly governor — sizing conservatism is structural."""
    wr = float(cert.get("win_rate_lo", cert.get("win_rate", 0.5)))
    return float(min(max(wr, 0.50), 0.75))