"""
APEX OMNI v9.7.1 — FLY INTELLIGENCE (the pin engine's read, mined for the
directional book instead of expressed as a trade)
=====================================================================
The operator's directive, stated repeatedly: the butterfly must NOT be the
trade — but its read is valuable, so use that read to buy BETTER directional
options. This module is that translation layer. It never opens anything; it
turns the fly's gate verdict (core.shortvol.evaluate_gate — the SAME bytes,
no second signal invented) into a directional MODULATION and a BOUNDARY MAP
the long book consumes.

Why the fly's read is a real directional edge (not decoration)
--------------------------------------------------------------
The gate grants precisely when: net dealer gamma is POSITIVE and large, IV
rank is RICH, and spot sits INSIDE a call/put wall corridor with buffer. That
is the textbook signature of a PINNING / MEAN-REVERTING regime:

  • Positive dealer gamma ⇒ market makers hedge AGAINST price (sell rallies,
    buy dips) — realized vol is damped and price reverts toward gamma walls.
    (Baltussen, Da, Lammers, Van der Wel, JFE 2021, "Hedging demand and market
    intraday momentum"; Barbon–Buraschi 2021 on dealer gamma & price
    stabilisation.) The gamma "flip" / walls are where that hedging flow
    changes sign — the standard dealer-positioning map used by the desk.
  • So in THIS regime a directional BREAKOUT toward a near wall is a trade the
    dealers will actively fade — the exact "entered after the breakdown candle,
    second candle didn't sustain, retest gave SL" trap the operator described.
    The fly's grant is therefore a SKEPTIC on momentum INTO a wall, and a
    CONFIRMER on a fade FROM a corridor edge BACK toward the pin.

Three actionable products (all advisory; none is a hard veto)
-------------------------------------------------------------
Given a granted fly read {side (nearer wall), corridor_steps, iv_rank,
net_gex, call_wall, put_wall} and the live spot + a directional conviction:

  1. CONVICTION MODULATION (logit space, like core.decision.apply_regime):
     • conviction pointing INTO the nearer wall, with little room, is
       DAMPENED (raise the bar it must clear) — proportional to how little
       corridor room remains and how deep the +gamma is. You can still take
       it if the raw signal is strong enough; you just have to mean it.
     • conviction pointing AWAY from the near wall (toward open corridor) or
       FADING a corridor edge back toward the pin is mildly BOOSTED — it is
       trading WITH the dealer flow, not against it.
     This is "scale, never veto" — identical philosophy to the regime
     multiplier, composed multiplicatively with it (both live in logit space,
     so they stack without saturating).

  2. BOUNDARY-AWARE TARGET CAP: the near wall is a hard magnet, so a
     directional target should not be planted PAST it. The long book already
     caps the target at the GEX wall runway; the fly read makes that cap
     REGIME-CONDITIONAL — tighter (a fraction of the runway) when +gamma is
     deep, because the pin will arrest the move BEFORE the wall in a strong
     pinning regime. Returned as a runway multiplier ∈ (0,1].

  3. MEAN-REVERSION ENTRY HINT: when spot is pressed against a corridor edge
     (within a small buffer of a wall) in a granted +gamma regime, a fade
     BACK toward the pin is the regime's highest-probability directional
     move. Surfaced as a discrete hint (side + strength) the brain MAY act on
     through the SAME entry constitution — it is not an auto-entry.

Honesty / safety
----------------
• Pure function of already-computed state. No new market data, no new signal.
• Advisory only: every product degrades to "no change" when the fly gate is
  not granting or the read is stale. The directional book's floors, gates and
  persistence are untouched.
• All knobs hash-EXCLUDED (they change WHEN/HOW a directional entry fires and
  where its target sits — not any feature or triple-barrier LABEL the forge
  trains on), so tuning never forces a re-forge. Validated on the vault by
  tools/fly_intel_report.py before being trusted, exactly like every organ.

Self-test:   python core/fly_intel.py
Vault study: python tools/fly_intel_report.py [--days N]
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


@dataclass
class FlyIntel:
    """The mined directional read. `active` is False whenever the fly gate is
    not granting (all products then neutral)."""
    active: bool
    regime: str = "none"          # "pinning" when a +gamma grant is live
    near_wall: str = ""           # "CE" (call wall above) | "PE" (put wall below)
    near_wall_px: float = 0.0
    far_wall_px: float = 0.0
    room_to_near_steps: float = 0.0   # corridor steps from spot to the near wall
    corridor_steps: float = 0.0
    iv_rank: float | None = None
    net_gex: float | None = None
    pin_pressure: float = 0.0     # 0..1 composite: how strong the pin is here
    # products
    conv_mult: float = 1.0        # multiplicative modulation for directional conv
    target_runway_mult: float = 1.0   # shrink the directional target's wall runway
    revert_hint_side: str = ""    # "CE"/"PE" fade-to-pin hint (edge-pressed only)
    revert_hint_strength: float = 0.0
    note: str = "inactive"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def assess(*, granted: bool, side: str | None, spot: float,
           call_wall: float | None, put_wall: float | None,
           corridor_steps: float, iv_rank: float | None,
           net_gex: float | None, strike_step: float,
           direction: str | None, conviction: float) -> FlyIntel:
    """Translate a fly gate verdict into the directional read.

    granted / side / corridor_steps / iv_rank / net_gex come straight from
    core.shortvol.evaluate_gate. `direction`/`conviction` are the directional
    book's CURRENT candidate (post-regime-mult); pass direction=None to get the
    regime map without a per-candidate modulation.
    """
    if not granted or not (call_wall and put_wall and call_wall > put_wall):
        return FlyIntel(active=False)

    cw, pw = float(call_wall), float(put_wall)
    step = max(float(strike_step), 1e-6)
    room_ce = max((cw - spot) / step, 0.0)     # steps up to the call wall
    room_pe = max((spot - pw) / step, 0.0)     # steps down to the put wall
    near = "CE" if room_ce <= room_pe else "PE"
    near_px = cw if near == "CE" else pw
    far_px = pw if near == "CE" else cw
    room_near = room_ce if near == "CE" else room_pe

    # ---- pin pressure: deeper +gamma, richer IV, and LESS room ⇒ stronger pin
    gex_min = float(_cfg("SV_NET_GEX_MIN", 1.0e12))
    gex_norm = (math.log10(max(float(net_gex or gex_min), gex_min))
                - math.log10(gex_min))               # 0 at threshold, grows
    gex_term = _sigmoid(gex_norm / max(_cfg("FLY_INTEL_GEX_LOG_SCALE", 0.5),
                                       1e-6) - 1.0)
    ivr_term = min(max((float(iv_rank or 0.0)
                        - _cfg("SV_IVRANK_MIN", 0.60))
                       / max(1.0 - _cfg("SV_IVRANK_MIN", 0.60), 1e-6), 0.0), 1.0)
    # closeness: 1 when spot is AT the near wall, 0 at corridor centre
    half = max(float(corridor_steps) / 2.0, 1e-6)
    closeness = min(max(1.0 - (room_near / half), 0.0), 1.0)
    pin = float(min(max(0.5 * gex_term + 0.3 * ivr_term + 0.2 * closeness,
                        0.0), 1.0))

    intel = FlyIntel(
        active=True, regime="pinning", near_wall=near, near_wall_px=near_px,
        far_wall_px=far_px, room_to_near_steps=round(room_near, 2),
        corridor_steps=round(float(corridor_steps), 2), iv_rank=iv_rank,
        net_gex=net_gex, pin_pressure=round(pin, 3))

    # ---- product 2: regime-conditional target runway cap (always available) --
    # deep pin ⇒ the move arrests BEFORE the wall; shrink the runway the long
    # book grants the directional target toward the near wall.
    floor_mult = float(_cfg("FLY_INTEL_RUNWAY_MULT_FLOOR", 0.45))
    intel.target_runway_mult = round(1.0 - (1.0 - floor_mult) * pin, 3)

    # ---- product 3: mean-reversion hint when pressed against a corridor edge -
    edge_steps = float(_cfg("FLY_INTEL_EDGE_STEPS", 0.75))
    if room_near <= edge_steps:
        # fade BACK toward the pin: pressed at the call wall ⇒ bearish (PE),
        # pressed at the put wall ⇒ bullish (CE).
        intel.revert_hint_side = "PE" if near == "CE" else "CE"
        intel.revert_hint_strength = round(pin * (1.0 - room_near / max(edge_steps, 1e-6)), 3)

    # ---- product 1: directional conviction modulation (needs a candidate) ----
    if direction in ("CE", "PE") and conviction != 0.0:
        into_near = (direction == near)          # buying toward the near wall?
        max_damp = float(_cfg("FLY_INTEL_MAX_DAMP", 0.45))   # ≥ this fraction
        max_boost = float(_cfg("FLY_INTEL_MAX_BOOST", 0.15))
        # scarcity of room amplifies the effect (1 at the wall, 0 with full room)
        scarcity = 1.0 - min(room_near / max(half, 1e-6), 1.0)
        if into_near:
            # dampen: multiplier < 1, deeper with more pin AND less room
            intel.conv_mult = round(1.0 - max_damp * pin * scarcity, 3)
            intel.note = (f"pinning: {direction} INTO {near} wall "
                          f"({room_near:.1f} steps) — dampened ×"
                          f"{intel.conv_mult} (pin {pin:.2f})")
        else:
            # boost: trading away from the near wall / with the fade
            intel.conv_mult = round(1.0 + max_boost * pin, 3)
            intel.note = (f"pinning: {direction} AWAY from {near} wall — "
                          f"boosted ×{intel.conv_mult} (pin {pin:.2f})")
    else:
        intel.note = (f"pinning regime map only (near {near} wall @ "
                      f"{near_px:.0f}, pin {pin:.2f})")

    return intel


def apply_conv(conv: float, conv_mult: float) -> float:
    """Compose the fly modulation in LOGIT space (same as
    core.decision.apply_regime), so it stacks with the regime multiplier
    without saturating the tanh. Identity when mult == 1."""
    if conv == 0.0 or conv_mult == 1.0:
        return conv
    c = max(min(conv, 0.999999), -0.999999)
    return math.tanh(math.atanh(c) * float(conv_mult))


# ----------------------------------------------------------------- self-test
if __name__ == "__main__":
    step = 100.0
    # SENSEX-like: corridor 77000(PE)..77500(CE), spot pressed near the call wall
    scenarios = [
        ("spot mid-corridor, no candidate",
         dict(spot=77250, direction=None, conviction=0.0)),
        ("bullish INTO the near call wall (should DAMPEN)",
         dict(spot=77460, direction="CE", conviction=0.80)),
        ("bearish AWAY from the near call wall (should BOOST)",
         dict(spot=77460, direction="PE", conviction=0.80)),
        ("bullish with full room (mild)",
         dict(spot=77120, direction="CE", conviction=0.80)),
        ("edge-pressed at call wall (revert hint = PE)",
         dict(spot=77490, direction=None, conviction=0.0)),
    ]
    base = dict(granted=True, side="CE", call_wall=77500.0, put_wall=77000.0,
                corridor_steps=5.0, iv_rank=0.72, net_gex=2.4e12,
                strike_step=step)
    print(f"{'scenario':46s} | mult  | runway | revert | note")
    print("-" * 118)
    for name, kw in scenarios:
        fi = assess(**base, **kw)
        rv = (f"{fi.revert_hint_side}:{fi.revert_hint_strength:.2f}"
              if fi.revert_hint_side else "—")
        print(f"{name:46s} | {fi.conv_mult:<5.3f} | {fi.target_runway_mult:<6.3f} "
              f"| {rv:6s} | {fi.note}")
    # ungranted ⇒ fully neutral
    fi0 = assess(granted=False, side=None, spot=77250, call_wall=None,
                 put_wall=None, corridor_steps=0, iv_rank=None, net_gex=None,
                 strike_step=step, direction="CE", conviction=0.8)
    print(f"\nungranted gate → active={fi0.active} conv_mult={fi0.conv_mult} "
          f"runway_mult={fi0.target_runway_mult} (fully neutral ✓)")
    # logit composition sanity
    print("\nlogit compose: conv 0.80 × dampen 0.70 →",
          round(apply_conv(0.80, 0.70), 3),
          "| × boost 1.15 →", round(apply_conv(0.80, 1.15), 3))