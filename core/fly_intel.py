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

Why the fly's read is a real directional edge — AND WHY THE SIGN IS MEASURED,
NOT ASSUMED
------------------------------------------------------------------------------
The gate grants when net dealer gamma is positive and large, IV rank is rich,
and spot sits inside a call/put wall corridor. Two OPPOSITE things can be true
of that regime, and WHICH ONE holds is an empirical question about a specific
tape — not something to hard-code:

  • REVERSION reading (textbook +gamma): dealers hedge against price (sell
    rallies, buy dips), realized vol is damped, price reverts toward the walls
    (Baltussen–Da–Lammers–Van der Wel, JFE 2021; Barbon–Buraschi 2021). Here a
    breakout INTO a near wall gets faded — fade the edge, ride the pin.
  • MOMENTUM reading: on some indices/regimes the wall BREAK is the signal —
    once price reaches the wall the dealer hedging that was pinning it flips
    and the move accelerates THROUGH. Here a breakout INTO the wall FOLLOWS
    THROUGH, and fading it loses. This is the operator's BankNifty case: the
    breakout DID work, but entrants were shaken out on the RETEST wick before
    the real move ("second candle didn't sustain, retest gave SL").

  *** On this operator's own vault (fly_intel_report, 2026-07-16, 5 granted
      days), INTO-wall won 61.6% and fade won 38.3% — a decisive MOMENTUM
      sign, the OPPOSITE of the textbook reversion default. ***

So the module does NOT bake in a direction. It reads a PERSISTED POLARITY that
fly_intel_report writes from the vault: +1 = momentum (ride the break), −1 =
reversion (fade to pin), 0 = undecided. The polarity is set ONLY when the
report's confidence interval is decisive AND the sample clears a day/second
floor; otherwise it stays 0 and the modulation is IDENTITY (no tilt at all).
Ships at 0 — nothing is applied until the operator's own data earns a sign.
This is the same telemetry → evidence → certificate → action ladder the rest
of the system uses; the modulation is a CERTIFIED organ, not a guess.

Products (all advisory; none a hard veto; all NEUTRAL at polarity 0)
--------------------------------------------------------------------
  1. CONVICTION MODULATION (logit space): with a decided polarity, a read
     trading WITH the vault-measured edge is mildly BOOSTED and a read AGAINST
     it is DAMPENED. Momentum polarity ⇒ boost the wall-break direction,
     dampen the fade. Reversion polarity ⇒ the reverse. "Scale, never veto",
     stacked multiplicatively on the regime multiplier.

  2. BOUNDARY-AWARE TARGET CAP: the near wall is a hard level. Under REVERSION
     polarity the target is pulled IN (the pin arrests the move before the
     wall). Under MOMENTUM polarity the wall is a level to break THROUGH, so
     the runway is NOT shrunk (cap → 1.0). Neutral (1.0) at polarity 0.

  3. RETEST-SURVIVAL ENTRY FILTER (the BankNifty trap-killer, always on when
     the fly grants regardless of polarity sign): when a wall-break entry is
     forming, the naive fill is on the FIRST breakout candle — which is
     exactly what gets stopped on the retest. This product tells the brain to
     REQUIRE a sustained reclaim/hold through the retest before arming the
     directional entry (the entry-side twin of the exit dwell), so the system
     stops taking the bait the video describes. Surfaced as a hint + a
     recommended entry-arm delay; the brain applies it through the normal
     persistence gate.

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


# --------------------------------------------------------------------------
# VAULT-MEASURED POLARITY — the sign of the fly's directional edge, learned
# from the operator's own tape by tools/fly_intel_report.py and persisted to
# a small JSON. +1 = MOMENTUM (ride the wall break), −1 = REVERSION (fade to
# pin), 0 = UNDECIDED (no modulation at all). Cached with mtime invalidation
# so a fresh report is picked up without a restart. Missing/short-sample ⇒ 0.
# --------------------------------------------------------------------------
_POLARITY_CACHE: dict = {"mtime": None, "val": 0, "meta": {}}


def polarity() -> int:
    """Current vault-measured polarity ∈ {-1, 0, +1}. 0 (identity) whenever no
    decisive, sufficiently-sampled sign has been written. Never raises."""
    import json
    import os
    if not bool(_cfg("FLY_INTEL_USE_POLARITY", True)):
        return 0
    # test/warm injection: a sentinel mtime short-circuits the file read
    if _POLARITY_CACHE.get("mtime") == "test":
        return int(_POLARITY_CACHE["val"])
    path = _cfg("FLY_INTEL_POLARITY_PATH", None)
    if path is None:
        try:
            import config
            path = os.path.join(str(getattr(config, "LOG_DIR", ".")),
                                "fly_intel_polarity.json")
        except Exception:                                      # pragma: no cover
            return 0
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return 0                                   # no artifact yet ⇒ neutral
    if _POLARITY_CACHE["mtime"] == mt:
        return int(_POLARITY_CACHE["val"])
    try:
        with open(path, "r") as fh:
            d = json.load(fh)
        val = int(d.get("polarity", 0))
        if val not in (-1, 0, 1):
            val = 0
    except Exception:                                          # pragma: no cover
        val, d = 0, {}
    _POLARITY_CACHE.update({"mtime": mt, "val": val, "meta": d})
    return val


def polarity_meta() -> dict:
    """The metadata block from the last-loaded polarity artifact (win rates,
    CI, sample), for telemetry. Empty until one is loaded."""
    polarity()
    return dict(_POLARITY_CACHE.get("meta", {}))


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
    polarity: int = 0             # vault-measured edge sign in effect (+1/-1/0)
    # products
    conv_mult: float = 1.0        # multiplicative modulation for directional conv
    target_runway_mult: float = 1.0   # shrink the directional target's wall runway
    revert_hint_side: str = ""    # edge-direction hint (fade OR ride per polarity)
    revert_hint_strength: float = 0.0
    # product 3 — retest survival (the BankNifty trap-killer; polarity-agnostic)
    retest_arm_delay_s: float = 0.0   # require a sustained hold this long before
    #                                   arming a wall-break directional entry
    at_wall: bool = False         # spot is within the edge buffer of the near wall
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

    pol = polarity()
    intel = FlyIntel(
        active=True, regime=("momentum" if pol > 0 else
                             "reversion" if pol < 0 else "undecided"),
        near_wall=near, near_wall_px=near_px, far_wall_px=far_px,
        room_to_near_steps=round(room_near, 2),
        corridor_steps=round(float(corridor_steps), 2), iv_rank=iv_rank,
        net_gex=net_gex, pin_pressure=round(pin, 3), polarity=pol)

    edge_steps = float(_cfg("FLY_INTEL_EDGE_STEPS", 0.75))
    intel.at_wall = room_near <= edge_steps

    # ---- product 3: RETEST-SURVIVAL (polarity-AGNOSTIC — always on when the
    # fly grants). The BankNifty trap: a wall-break entry filled on the first
    # breakout candle gets stopped on the retest. When spot is AT the wall,
    # require a sustained hold before the directional entry arms — the entry-
    # side twin of the exit dwell. Scales with pin (a tighter pin ⇒ a nastier
    # retest wick ⇒ demand a longer hold).
    if intel.at_wall and bool(_cfg("FLY_INTEL_RETEST_FILTER", True)):
        base_delay = float(_cfg("FLY_INTEL_RETEST_ARM_S", 20.0))
        intel.retest_arm_delay_s = round(base_delay * (0.5 + pin), 1)

    # ---- product 2: target runway cap — polarity-conditional.
    # REVERSION ⇒ pull the target IN (pin arrests before the wall).
    # MOMENTUM  ⇒ the wall is to be broken THROUGH; do NOT shrink (→1.0).
    # UNDECIDED ⇒ neutral (1.0).
    if pol < 0:
        floor_mult = float(_cfg("FLY_INTEL_RUNWAY_MULT_FLOOR", 0.45))
        intel.target_runway_mult = round(1.0 - (1.0 - floor_mult) * pin, 3)
    else:
        intel.target_runway_mult = 1.0

    # ---- product 1 (hint): the edge direction at a corridor edge, per polarity.
    # REVERSION ⇒ fade back toward the pin. MOMENTUM ⇒ ride the break through
    # the wall. UNDECIDED ⇒ no hint.
    if intel.at_wall and pol != 0:
        break_dir = near                     # breaking THROUGH the near wall
        fade_dir = "PE" if near == "CE" else "CE"
        intel.revert_hint_side = break_dir if pol > 0 else fade_dir
        intel.revert_hint_strength = round(
            pin * (1.0 - room_near / max(edge_steps, 1e-6)), 3)

    # ---- product 1 (modulation): needs a candidate AND a decided polarity.
    # A read trading WITH the vault edge is boosted; AGAINST it, dampened.
    if direction in ("CE", "PE") and conviction != 0.0 and pol != 0:
        into_near = (direction == near)      # candidate points toward near wall
        # "with the edge" = into-wall under momentum, fade under reversion
        with_edge = (into_near if pol > 0 else (not into_near))
        max_damp = float(_cfg("FLY_INTEL_MAX_DAMP", 0.45))
        max_boost = float(_cfg("FLY_INTEL_MAX_BOOST", 0.15))
        scarcity = 1.0 - min(room_near / max(half, 1e-6), 1.0)
        regime_word = "momentum" if pol > 0 else "reversion"
        if with_edge:
            intel.conv_mult = round(1.0 + max_boost * pin, 3)
            intel.note = (f"{regime_word}: {direction} WITH edge "
                          f"(near {near} wall, {room_near:.1f} steps) — "
                          f"boosted ×{intel.conv_mult} (pin {pin:.2f})")
        else:
            intel.conv_mult = round(1.0 - max_damp * pin * scarcity, 3)
            intel.note = (f"{regime_word}: {direction} AGAINST edge — "
                          f"dampened ×{intel.conv_mult} (pin {pin:.2f})")
    else:
        _pw = ("undecided — modulation OFF until the vault earns a sign"
               if pol == 0 else
               f"{'momentum' if pol > 0 else 'reversion'} map "
               f"(near {near} wall @ {near_px:.0f}, pin {pin:.2f})")
        intel.note = _pw

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
    base = dict(granted=True, side="CE", call_wall=77500.0, put_wall=77000.0,
                corridor_steps=5.0, iv_rank=0.72, net_gex=2.4e12,
                strike_step=step)
    # Force each polarity via the module cache (bypasses the file for the test).
    def _force(p):
        _POLARITY_CACHE.update({"mtime": "test", "val": p, "meta": {}})

    scenarios = [
        ("CE toward near call wall", dict(spot=77350, direction="CE",
                                          conviction=0.80)),
        ("PE away from near call wall", dict(spot=77350, direction="PE",
                                             conviction=0.80)),
        ("edge-pressed at call wall", dict(spot=77440, direction=None,
                                           conviction=0.0)),
    ]
    for pol, label in ((0, "UNDECIDED (default — must be NEUTRAL)"),
                       (+1, "MOMENTUM (ride the break — from the vault)"),
                       (-1, "REVERSION (fade to pin — textbook)")):
        _force(pol)
        print(f"\n=== polarity {pol:+d}: {label} ===")
        print(f"{'scenario':30s} | mult  | runway | hint     | retest | note")
        print("-" * 116)
        for name, kw in scenarios:
            fi = assess(**base, **kw)
            hint = (f"{fi.revert_hint_side}:{fi.revert_hint_strength:.2f}"
                    if fi.revert_hint_side else "—")
            rt = (f"{fi.retest_arm_delay_s:.0f}s" if fi.retest_arm_delay_s
                  else "—")
            print(f"{name:30s} | {fi.conv_mult:<5.3f} | "
                  f"{fi.target_runway_mult:<6.3f} | {hint:8s} | {rt:6s} | "
                  f"{fi.note[:44]}")

    _force(0)
    ok = True
    for kw in (dict(spot=77350, direction="CE", conviction=0.80),
               dict(spot=77350, direction="PE", conviction=0.80)):
        fi = assess(**base, **kw)
        if fi.conv_mult != 1.0 or fi.target_runway_mult != 1.0:
            ok = False
    print(f"\nUNDECIDED neutrality (conv & runway untouched): "
          f"{'✓' if ok else '✗ FAIL'}")
    # ungranted ⇒ inactive regardless of polarity
    _force(+1)
    fi0 = assess(granted=False, side=None, spot=77250, call_wall=None,
                 put_wall=None, corridor_steps=0, iv_rank=None, net_gex=None,
                 strike_step=step, direction="CE", conviction=0.8)
    print(f"ungranted gate → active={fi0.active} (inactive ✓)"
          if not fi0.active else "ungranted FAIL")
    print("\nlogit compose: 0.80 × boost 1.13 →",
          round(apply_conv(0.80, 1.13), 3),
          "| × dampen 0.87 →", round(apply_conv(0.80, 0.87), 3))