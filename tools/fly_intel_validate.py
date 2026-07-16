"""
APEX OMNI v9.7.1 — FLY-INTEL VALIDATION (offline, no DB)
========================================================
Proves the behavioral contract of core/fly_intel.py on reconstructed pinning-
regime scenarios, and — the part that matters — demonstrates the EDGE
HYPOTHESIS the modulation rests on: in a positive-gamma pinning regime, a
directional breakout INTO the near wall reverts (loses), while a fade FROM a
corridor edge back toward the pin follows through (wins). If that asymmetry is
real in the scenarios, dampening the former and boosting the latter is
correct; if not, the modulation would be noise and this harness FAILS.

This is the offline twin of tools/fly_intel_report.py (which measures the same
asymmetry on the real vault). Run this any time; run that after a close.

  python tools/fly_intel_validate.py
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                             # noqa: E402
from core import fly_intel as FI                          # noqa: E402

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_fails = 0


def check(name, cond, detail=""):
    global _fails
    print(f"  [{PASS if cond else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        _fails += 1


# --------------------------------------------------------------------------
# 1. CONTRACT: the three products behave as specified.
# --------------------------------------------------------------------------
def contract():
    print("\n=============== CONTRACT — fly_intel products ================")
    step = 100.0
    base = dict(granted=True, side="CE", call_wall=77500.0, put_wall=77000.0,
                corridor_steps=5.0, iv_rank=0.72, net_gex=2.4e12,
                strike_step=step)
    into = FI.assess(**base, spot=77460, direction="CE", conviction=0.80)
    away = FI.assess(**base, spot=77460, direction="PE", conviction=0.80)
    check("conv INTO near wall is dampened (<1)", into.conv_mult < 1.0,
          f"×{into.conv_mult}")
    check("conv AWAY from near wall is boosted (>1)", away.conv_mult > 1.0,
          f"×{away.conv_mult}")
    check("dampening ⇒ logit conv shrinks",
          abs(FI.apply_conv(0.80, into.conv_mult)) < 0.80,
          f"0.80→{FI.apply_conv(0.80, into.conv_mult):.3f}")
    deep = FI.assess(**{**base, "net_gex": 8.0e12}, spot=77480,
                     direction=None, conviction=0.0)
    shallow = FI.assess(**{**base, "net_gex": 1.05e12}, spot=77250,
                        direction=None, conviction=0.0)
    check("deeper pin ⇒ tighter target runway",
          deep.target_runway_mult < shallow.target_runway_mult,
          f"deep {deep.target_runway_mult} < shallow {shallow.target_runway_mult}")
    edge = FI.assess(**base, spot=77492, direction=None, conviction=0.0)
    check("edge-pressed at call wall ⇒ revert hint PE",
          edge.revert_hint_side == "PE" and edge.revert_hint_strength > 0,
          f"{edge.revert_hint_side}:{edge.revert_hint_strength}")
    ung = FI.assess(granted=False, side=None, spot=77250, call_wall=None,
                    put_wall=None, corridor_steps=0, iv_rank=None,
                    net_gex=None, strike_step=step, direction="CE",
                    conviction=0.8)
    check("ungranted gate ⇒ fully neutral",
          (not ung.active) and ung.conv_mult == 1.0
          and ung.target_runway_mult == 1.0)


# --------------------------------------------------------------------------
# 2. EDGE HYPOTHESIS: simulate a pinning regime and measure whether the
#    modulation improves the P&L of directional entries vs unmodulated.
# --------------------------------------------------------------------------
def _pin_path(n, pin_strength, cw, pw, seed):
    """A positive-gamma pin: an Ornstein–Uhlenbeck reversion of spot toward the
    corridor CENTRE, strength ∝ pin. Breakouts toward a wall get pulled back;
    the wall behaves as a reflecting magnet. Returns per-second spot."""
    rng = random.Random(seed)
    centre = (cw + pw) / 2.0
    half = (cw - pw) / 2.0
    sp = centre + rng.uniform(-half * 0.6, half * 0.6)
    out = []
    for _ in range(n):
        # OU pull toward centre + noise; stronger pin ⇒ stronger pull, less noise
        pull = -pin_strength * (sp - centre) / half
        sp += pull * half * 0.02 + rng.gauss(0, half * 0.03 * (1.2 - pin_strength))
        sp = max(min(sp, cw + half * 0.05), pw - half * 0.05)   # walls reflect
        out.append(sp)
    return out


def _entry_pnl(spots, t0, direction, delta, hold, entry_prem):
    """Directional long P&L over `hold` seconds from t0: premium moves delta ×
    spot move in the position's favour."""
    if t0 + hold >= len(spots):
        hold = len(spots) - t0 - 1
    if hold <= 0:
        return 0.0
    d = spots[t0 + hold] - spots[t0]
    signed = d if direction == "CE" else -d
    exit_prem = max(entry_prem + delta * signed, 0.5)
    return exit_prem - entry_prem


def edge_hypothesis():
    print("\n============ EDGE HYPOTHESIS — does the read help? ===========")
    step = 100.0
    cw, pw = 77500.0, 77000.0
    delta, hold, entry_prem = 0.45, 300, 120.0
    bar = 0.55

    # The modulation is a SOFT tilt (scale-never-veto), so its effect shows on
    # MARGINAL signals — those near the bar, where a dampen/boost flips the
    # decision. We sample conviction across a realistic band [0.56, 0.85] (all
    # nominally tradeable) so the tilt can move borderline into-wall breakouts
    # below the bar and lift borderline fades. A signal is CE or PE with its
    # true P&L in the pin; RAW takes all above the bar, MOD applies the fly
    # tilt first. If the read is informative, MOD's taken set has better P&L.
    raw_take, mod_take = [], []
    n_days = 300
    conv_band = [0.56, 0.60, 0.64, 0.68, 0.72, 0.78, 0.84]
    for day in range(n_days):
        pin = random.Random(day).uniform(0.45, 0.9)
        spots = _pin_path(1500, pin, cw, pw, seed=1000 + day)
        rng = random.Random(5000 + day)
        for t0 in range(0, 1200, 60):
            spot = spots[t0]
            room_ce = (cw - spot) / step
            room_pe = (spot - pw) / step
            near = "CE" if room_ce <= room_pe else "PE"
            for direction in ("CE", "PE"):
                conv0 = rng.choice(conv_band)
                pnl = _entry_pnl(spots, t0, direction, delta, hold, entry_prem)
                if abs(conv0) >= bar:
                    raw_take.append(pnl)
                fi = FI.assess(granted=True, side=near, spot=spot,
                               call_wall=cw, put_wall=pw, corridor_steps=5.0,
                               iv_rank=0.72, net_gex=1.0e12 * (1 + 6 * pin),
                               strike_step=step, direction=direction,
                               conviction=conv0)
                if abs(FI.apply_conv(conv0, fi.conv_mult)) >= bar:
                    mod_take.append(pnl)

    raw_mean = sum(raw_take) / max(len(raw_take), 1)
    mod_mean = sum(mod_take) / max(len(mod_take), 1)
    raw_wr = sum(1 for p in raw_take if p > 0) / max(len(raw_take), 1)
    mod_wr = sum(1 for p in mod_take if p > 0) / max(len(mod_take), 1)
    shed = len(raw_take) - len(mod_take)
    print(f"    RAW  (fly ignored): {len(raw_take):5d} entries | mean "
          f"₹{raw_mean:+.2f} | win {raw_wr:.1%}")
    print(f"    MOD  (fly-intel)  : {len(mod_take):5d} entries | mean "
          f"₹{mod_mean:+.2f} | win {mod_wr:.1%}  ({shed} marginal into-wall shed)")
    check("fly-intel improves mean P&L per taken entry", mod_mean > raw_mean,
          f"Δ ₹{mod_mean - raw_mean:+.2f}/entry")
    check("fly-intel improves win rate", mod_wr > raw_wr,
          f"Δ {(mod_wr - raw_wr) * 100:+.1f}pp")
    check("fly-intel is selective at the margin", shed > 0,
          f"{shed} borderline into-wall breakouts tilted below the bar")


# --------------------------------------------------------------------------
# 3. NEUTRALITY: outside a granted regime, the modulation must not touch a
#    single directional decision (no silent drift).
# --------------------------------------------------------------------------
def neutrality():
    print("\n================ NEUTRALITY — no silent drift ===============")
    step, cw, pw = 100.0, 77500.0, 77000.0
    unchanged = True
    for _ in range(500):
        conv = random.uniform(-0.99, 0.99)
        fi = FI.assess(granted=False, side=None, spot=77250.0, call_wall=cw,
                       put_wall=pw, corridor_steps=5.0, iv_rank=None,
                       net_gex=None, strike_step=step,
                       direction=("CE" if conv > 0 else "PE"), conviction=conv)
        if FI.apply_conv(conv, fi.conv_mult) != conv:
            unchanged = False
            break
    check("ungranted regime leaves every conviction identical", unchanged)


if __name__ == "__main__":
    random.seed(20260716)
    contract()
    edge_hypothesis()
    neutrality()
    print("\n" + "=" * 62)
    if _fails:
        print(f"  {FAIL}: {_fails} check(s) failed")
        sys.exit(1)
    print(f"  {PASS}: all fly-intel validation checks passed")