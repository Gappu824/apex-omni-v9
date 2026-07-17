"""
APEX OMNI v9.7.1 — CASCADE EXIT & SMART LOCKOUT
================================================
The 2026-07-16 SENSEX jackpot that got away (from brain_report_2026-07-16.json)
-----------------------------------------------------------------------------
Three cascade PE triggers fired on a genuine short-gamma breakdown — spot
falling THROUGH the gamma flip with deeply negative net_gex, the regime where
dealer hedging AMPLIFIES the move (Baltussen et al., JFE 2021: negative dealer
gamma ⇒ intraday MOMENTUM, the mirror of the pinning case):

    11:28  spot 77430  flip 77552  z −2.04  → BUY 77400 PE   (trade 1)
    12:04  spot 77400  flip 77507  z −2.04  → BUY 77400 PE   (trade 2)
    12:53  spot 77376  flip 77478  z −2.59  → BLOCKED "post-loss PE lockout"

Spot fell 77430 → 77376. A PE should have paid. Instead the day realized
−₹998.89 and the STRONGEST trigger (z −2.59, the start of the real leg) was
BLOCKED. Two failures compounded:

  1. WHIPSAW STOP-OUT. Short-gamma breakdowns are violent and mean-revert
     second-to-second even as they trend down over minutes. Trades 1 & 2 were
     stopped on an upward retest wick, THEN the down-move continued — the exact
     "second candle didn't sustain, retest gave SL" trap. The fixed base stop
     (BASE_SL_PCT) is calibrated for normal regimes; in a short-gamma cascade
     it sits INSIDE the regime's own noise band and gets picked off.

  2. DUMB LOCKOUT. The post-loss directional lockout counts LOSSES, not WHY.
     Two whipsaw losses armed a blanket "no PE for DIRECTION_LOCKOUT_S", which
     then muted the one trigger that mattered. In CHOP that lockout is correct
     (it stops revenge trading); in a CONFIRMED, STRENGTHENING short-gamma
     downtrend the same-direction re-trigger is not revenge — it is the TREND.

This module fixes both, each mechanism carrying its rationale, and neither
loosening any hard floor (disaster / profit-lock / EOD / stale are untouched
and still fire first).

Part A — CASCADE-AWARE STOP WIDTH
---------------------------------
When a position is opened FROM a cascade trigger in a short-gamma regime, the
stop must respect the regime's realized noise, not a fixed percent. The stop
distance is widened by a regime factor derived from |net_gex| depth and the
cascade z (both proxies for how violently dealers are amplifying), bounded so
it can never exceed the disaster floor. This is the standard "volatility-scaled
stop" (Kaufman, *Trading Systems and Methods*): a stop set inside the noise is
not a stop, it is a donation. Wider stop + smaller size (the risk governor
already sizes off the stop distance) = same rupee risk, far fewer whipsaw
exits. The peak-capture trail then protects the gain once the real move runs.

Part B — SMART POST-LOSS LOCKOUT
--------------------------------
A same-direction re-entry after a loss is only BLOCKED when it looks like
revenge. It is ALLOWED (lockout bypassed) when ALL hold:
  • the new trigger is a CASCADE event (structural, not a discretionary retry),
  • it is STRICTLY STRONGER than the trigger that just lost (|z| greater by a
    margin) — the trend is intensifying, not the trader tilting,
  • it is still structurally ALIGNED (spot still beyond the flip in the trade
    direction, net_gex still short-gamma, flip fresh),
  • within a bounded number of bypasses per day (a strengthening trend re-arms
    a few times; a countertrend churn does not get infinite lives).
Anything short of that keeps the full lockout. This is the opportunity-cost /
trend-continuation refinement of a revenge-trade guard: punish tilt, not trend.

Both parts are ADVISORY inputs to the existing machinery — Part A adjusts the
stop the RiskGovernor sizes against; Part B returns a bypass decision the
RiskGovernor consults before enforcing its lockout. Neither can widen a stop
past the disaster floor nor re-arm beyond the daily bypass cap.

All knobs hash-EXCLUDED (they change stop WIDTH and lockout TEMPO for cascade
trades — not any feature or triple-barrier LABEL the forge trains on), so
tuning never forces a re-forge.

Self-test:   python core/cascade_exit.py
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _cfg(name: str, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:                                          # pragma: no cover
        return default


# --------------------------------------------------------------------------
# Part A — cascade-aware stop width
# --------------------------------------------------------------------------
def cascade_stop_mult(*, net_gex: float | None, cascade_z: float | None,
                      is_cascade: bool) -> tuple[float, str]:
    """Multiplier on the base stop DISTANCE for a cascade-opened position in a
    short-gamma regime. 1.0 (no change) for non-cascade or long-gamma entries.

    Rationale: |net_gex| depth and cascade |z| both scale how violently dealers
    amplify — a deeper short-gamma / stronger break needs a wider stop to sit
    OUTSIDE the regime's own retest noise. Bounded by CASCADE_STOP_MULT_MAX; the
    disaster floor (applied by the caller) is the absolute backstop regardless.
    """
    if not is_cascade or net_gex is None or net_gex >= 0:
        return 1.0, "non-cascade/long-gamma: base stop"
    gex_floor = float(_cfg("SV_NET_GEX_MIN", 1.0e12))          # scale anchor
    # depth in "threshold units": 1.0 at −1e12, grows with log magnitude
    depth = math.log10(max(abs(net_gex), gex_floor)) - math.log10(gex_floor)
    z_term = 0.0
    if cascade_z is not None:
        z_floor = float(_cfg("CASCADE_Z_FLOOR", 2.0))
        z_term = max(abs(cascade_z) - z_floor, 0.0)            # extra beyond the trigger
    base = float(_cfg("CASCADE_STOP_MULT_BASE", 1.5))
    k_depth = float(_cfg("CASCADE_STOP_K_DEPTH", 0.35))
    k_z = float(_cfg("CASCADE_STOP_K_Z", 0.25))
    mult = base + k_depth * depth + k_z * z_term
    mult = max(1.0, min(mult, float(_cfg("CASCADE_STOP_MULT_MAX", 2.5))))
    return round(mult, 3), (f"short-gamma cascade: stop ×{mult:.2f} "
                            f"(gex depth {depth:.2f}, z+{z_term:.2f})")


# --------------------------------------------------------------------------
# Part B — smart post-loss lockout
# --------------------------------------------------------------------------
@dataclass
class LockoutVerdict:
    bypass: bool
    reason: str
    diag: dict


class SmartLockout:
    """Decides whether a post-loss directional lockout should be BYPASSED for a
    strengthening, still-aligned cascade re-trigger. One instance per brain
    session; consulted by the RiskGovernor gate."""

    def __init__(self):
        self.bypasses_today = 0
        self.last_loss_z: dict[str, float] = {}     # direction → |z| that lost
        self.last_bypass_ts = -1e18

    def note_loss(self, direction: str, cascade_z: float | None) -> None:
        """Record the strength of the trigger that just lost, so the next
        re-trigger can be judged 'stronger than the loss'."""
        if cascade_z is not None:
            prev = self.last_loss_z.get(direction, 0.0)
            # keep the STRONGEST recent losing trigger as the bar to beat
            self.last_loss_z[direction] = max(prev, abs(cascade_z))

    def evaluate(self, *, ts: float, direction: str, is_cascade: bool,
                 cascade_z: float | None, spot: float, flip: float | None,
                 net_gex: float | None) -> LockoutVerdict:
        """Return whether to bypass the lockout for THIS candidate. All the
        structural checks must pass; otherwise the lockout stands."""
        diag = {"dir": direction, "is_cascade": is_cascade,
                "z": (round(cascade_z, 2) if cascade_z is not None else None),
                "loss_z": self.last_loss_z.get(direction)}
        if not bool(_cfg("SMART_LOCKOUT_ENABLED", True)):
            return LockoutVerdict(False, "smart-lockout disabled", diag)
        if not is_cascade:
            return LockoutVerdict(False, "not a cascade trigger — lockout stands",
                                  diag)
        if self.bypasses_today >= int(_cfg("LOCKOUT_BYPASS_MAX_PER_DAY", 3)):
            return LockoutVerdict(False, "daily bypass budget spent", diag)
        if ts - self.last_bypass_ts < float(_cfg("LOCKOUT_BYPASS_COOLDOWN_S",
                                                  60.0)):
            return LockoutVerdict(False, "bypass cooldown", diag)
        # strictly STRONGER than the trigger that lost (trend intensifying)
        if cascade_z is None:
            return LockoutVerdict(False, "no z on re-trigger", diag)
        loss_z = self.last_loss_z.get(direction, 0.0)
        margin = float(_cfg("LOCKOUT_BYPASS_Z_MARGIN", 0.30))
        if abs(cascade_z) < loss_z + margin:
            return LockoutVerdict(
                False, f"z {abs(cascade_z):.2f} not > loss z {loss_z:.2f}"
                       f"+{margin:.2f} — looks like revenge, lockout stands",
                diag)
        # still structurally ALIGNED with the trade direction
        if flip is None or net_gex is None:
            return LockoutVerdict(False, "no flip/gex to confirm alignment",
                                  diag)
        aligned = ((direction == "PE" and spot < flip and net_gex < 0) or
                   (direction == "CE" and spot > flip and net_gex < 0))
        if not aligned:
            return LockoutVerdict(
                False, f"not aligned (spot {spot:.0f} vs flip {flip:.0f}, "
                       f"gex {net_gex:.1e}) — lockout stands", diag)
        return LockoutVerdict(
            True, f"STRENGTHENING aligned cascade: z {abs(cascade_z):.2f} > "
                  f"loss {loss_z:.2f}, spot beyond flip, short-gamma — trend "
                  f"continuation, not revenge; lockout BYPASSED", diag)

    def register_bypass(self, ts: float) -> None:
        self.bypasses_today += 1
        self.last_bypass_ts = float(ts)


# --------------------------------------------------------------------------
# SELF-TEST
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Part A: cascade-aware stop width ===")
    cases = [
        ("non-cascade", dict(net_gex=2e12, cascade_z=None, is_cascade=False)),
        ("long-gamma cascade (shouldn't widen)",
         dict(net_gex=2e12, cascade_z=-2.0, is_cascade=True)),
        ("mild short-gamma z-2.0",
         dict(net_gex=-2e12, cascade_z=-2.0, is_cascade=True)),
        ("deep short-gamma z-2.6 (the 12:53 trigger)",
         dict(net_gex=-25e12, cascade_z=-2.59, is_cascade=True)),
        ("extreme (clamped)",
         dict(net_gex=-80e12, cascade_z=-4.0, is_cascade=True)),
    ]
    for name, kw in cases:
        m, why = cascade_stop_mult(**kw)
        print(f"  {name:42s} → ×{m:<5.2f} {why}")

    print("\n=== Part B: smart lockout (replaying 2026-07-16) ===")
    lk = SmartLockout()
    # trade 1 lost (z -2.04)
    lk.note_loss("PE", -2.04)
    # trade 2 lost (z -2.04) — same strength, this WOULD be revenge if retried
    v_rev = lk.evaluate(ts=100, direction="PE", is_cascade=True,
                        cascade_z=-2.04, spot=77400, flip=77507, net_gex=-20e12)
    print(f"  same-strength re-trigger (z-2.04 after z-2.04 loss):\n"
          f"    bypass={v_rev.bypass} — {v_rev.reason}")
    lk.note_loss("PE", -2.04)
    # trade 3: STRONGER trigger (z -2.59), still aligned → should BYPASS
    v_jack = lk.evaluate(ts=200, direction="PE", is_cascade=True,
                         cascade_z=-2.59, spot=77376, flip=77478,
                         net_gex=-25.5e12)
    print(f"  STRONGER aligned re-trigger (z-2.59, the jackpot):\n"
          f"    bypass={v_jack.bypass} — {v_jack.reason}")
    # a countertrend CE after PE losses (revenge/flip) → must stay locked
    v_ct = lk.evaluate(ts=300, direction="PE", is_cascade=True,
                       cascade_z=-2.7, spot=77600, flip=77478, net_gex=-25e12)
    print(f"  strong but MIS-ALIGNED (spot back above flip):\n"
          f"    bypass={v_ct.bypass} — {v_ct.reason}")

    ok = (v_rev.bypass is False and v_jack.bypass is True
          and v_ct.bypass is False)
    print(f"\n  {'✓ PASS' if ok else '✗ FAIL'}: revenge blocked, "
          f"strengthening-trend jackpot allowed, misaligned blocked")