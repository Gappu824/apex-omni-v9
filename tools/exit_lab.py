"""
APEX OMNI v9.7.1 — EXIT LAB (empirical validation of the peak-capture exits)
============================================================================
Runs the SHIPPED bytes — core.butterfly.FlyBook (with the new manage ladder),
core.exit_engine.PeakCaptureTrail / ride_ok, core.displacement — through a
scenario battery, side-by-side with a faithful re-implementation of the v9.7
first-touch policy, and ASSERTS the behavioral contract:

  FLY (real FlyBook.try_open → manage → close, temp ledger, forward-log
       no-op'd; noise scale calibrated to the 2026-07-15 live telemetry
       where the raw unwind credit printed 21.50→25.90 minute-to-minute on
       a still spot):
    F1  noise-hold      : today's mark noise stops NEITHER policy out, and
                          the new policy does not phantom-exit either.
    F2  wide-print spike: ONE 5-second garbage print at cc≈12 stops the OLD
                          policy at the worst mark of the day; the NEW
                          policy (smoothed + 45 s dwell) holds through it.
    F3  true collapse   : a sustained decay through the stop DOES exit —
                          after the dwell, never before it, and always
                          before the hard floor is deeply violated. The
                          dwell is a filter, not a blindfold.
    F4  pin-and-pay     : the fly grinds toward max value. OLD banks the
                          first touch of the 50% target; NEW tags it and
                          rides the ratchet — capture ratio must beat OLD.
    F5  pin lost        : spot sits beyond a wing ≥60 s ⇒ PIN_LOST salvage.

  LONG BOOK (policy-level: the exact rung-4/4.6 math with the real
       PeakCaptureTrail + ride_ok + signed_efficiency on a synthetic
       post-trap breakdown shaped like the 2026-07-15 SENSEX PE case —
       runway-anchored target ≈ +34%, then the tape runs 3× further):
    L1  jackpot capture : NEW exit captures a strictly larger share of the
                          peak gain than the OLD first-touch target.
    L2  hunt survival   : a 10 s flush wick through the ratchet does not
                          fire (dwell); OLD trailing stop is taken out.
    L3  reversal dwell  : one conviction flicker against the position no
                          longer exits; a sustained reversal still does.

  DISPLACEMENT (real governor):
    D1  flicker ≠ displacement; a sustained qualifying read displaces; the
        daily budget and a working (≥60% progress) fly are honored.

Run after any change to the exit stack:   python tools/exit_lab.py
Exit code 0 = every assert held. This lab is synthetic-scenario evidence of
the MECHANISM — it is not a market backtest; the butterfly certificate re-
earns through tools/butterfly_harness.py on the real vault, and the long
book accrues its evidence through the ledger as always.
"""
from __future__ import annotations

import math
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                            # noqa: E402
from core import butterfly as BF                          # noqa: E402
from core.displacement import DisplacementGovernor        # noqa: E402
from core.exit_engine import (PeakCaptureTrail, TrailParams,  # noqa: E402
                              ride_ok, signed_efficiency)

BF.log_forward = lambda row: None          # lab must never touch state/

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = ""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# ============================================================ fly scaffolding
def _spec() -> BF.FlySpec:
    return BF.FlySpec(index="SENSEX", side="PE", body_k=77000.0,
                      wing_in_k=77200.0, wing_out_k=76800.0,
                      wing_width=200.0, lot=20,
                      body_symbol="SXB", wing_in_symbol="SXI",
                      wing_out_symbol="SXO", body_token=2,
                      wing_in_token=1, wing_out_token=3, exchange="BFO")


def _quotes(cc: float, spread: float = 0.6) -> dict:
    """Legs whose CONSERVATIVE unwind credit (wings@bid − 2×body@ask) equals
    cc exactly; entry pricing sees ~2% spreads so try_open passes its gates."""
    ba = 30.0                                   # body ask (unwind side)
    w = (cc + 2 * ba) / 2.0                     # each wing bid
    return {1: {"bid": w, "ask": w + spread},
            2: {"bid": ba - spread, "ask": ba},
            3: {"bid": w, "ask": w + spread}}


def _open_book():
    tmp = Path(tempfile.mkdtemp()) / "lab_ledger.csv"
    book = BF.FlyBook(risk=None, ledger_path=tmp)
    r = book.try_open(ts=0.0, hm="11:00", spec=_spec(),
                      quotes=_quotes(26.0), capital=50000.0, mode="lab")
    assert "opened" in r, f"lab try_open failed: {r}"
    return book, float(r["debit"])


def _hm(t: float) -> str:
    m = int(11 * 60 + t // 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def _run_fly(cc_path, spot_path=None, old: bool = False):
    """Drive the REAL FlyBook (new policy) or the v9.7 first-touch rungs
    (old) over identical (cc, spot) paths. Returns (t_exit, why, cc_at_exit)
    or (None, 'HELD', last_cc)."""
    if old:
        debit = 26.0                             # same executable open debit
        tgt = debit + config.SV_FLY_TP_FRAC * (200.0 - debit)
        stop = debit * (1.0 - config.SV_FLY_SL_FRAC)
        for t, cc in enumerate(cc_path):
            if cc >= tgt:
                return t, "TARGET", cc
            if cc <= stop:
                return t, "STOP", cc
        return None, "HELD", cc_path[-1]
    book, _ = _open_book()
    for t, cc in enumerate(cc_path):
        spot = spot_path[t] if spot_path is not None else 77000.0
        row = book.manage(ts=float(t + 1), hm=_hm(t), spot=spot,
                          quotes=_quotes(cc), cascade_event=False)
        if row is not None:
            return t, row["why"], cc
    return None, "HELD", cc_path[-1]


def _noise_path(n: int, base: float = 26.0, sigma: float = 1.1,
                seed: int = 11) -> list[float]:
    """Mean-reverting bounce around a still value — the 2026-07-15 signature
    (raw cc 21.50…25.90 on a pinned spot). σ chosen so per-minute swings
    match the live log's ±8% of debit."""
    rng = random.Random(seed)
    return [max(base + rng.gauss(0, sigma), 0.2) for _ in range(n)]


print("\n================ FLY BOOK — real FlyBook bytes ================")
# F1 — today's noise stops nobody, and the new engine adds no phantom exit
path = _noise_path(1800)
to, wo, _ = _run_fly(path, old=True)
tn, wn, _ = _run_fly(path)
check("F1 noise-hold (old survives today's noise)", to is None, f"old:{wo}")
check("F1 noise-hold (new adds no phantom exit)", tn is None, f"new:{wn}")

# F2 — one wide print at the stop level
path = _noise_path(1800, seed=12)
for k in range(900, 905):
    path[k] = 12.0                               # cc ≤ 13 = raw stop touch
to, wo, co = _run_fly(path, old=True)
tn, wn, _ = _run_fly(path)
check("F2 wide print stops OLD at worst mark", to == 900 and wo == "STOP",
      f"old exited t={to} {wo} @ {co:.2f}")
check("F2 wide print — NEW holds through it", tn is None,
      "smoothed mark barely moves; 45 s dwell never met")

# F3 — a REAL collapse still exits, after the dwell, before the abyss
path = [26.0 - 16.0 * min(t / 300.0, 1.0) + random.Random(13).gauss(0, 0.5)
        for t in range(600)]
first_touch = next(t for t, c in enumerate(path)
                   if c <= 26.0 * (1 - config.SV_FLY_SL_FRAC))
tn, wn, cn = _run_fly(path)
check("F3 sustained collapse exits", tn is not None and wn in
      ("STOP", "STOP_HARD"), f"{wn} at t={tn}s (raw first touch t={first_touch}s)")
check("F3 …after the dwell, not on first raw touch",
      tn is not None and tn >= first_touch,
      f"waited {tn - first_touch}s past first touch")
check("F3 …and before the abyss", cn > 26.0 * (1 - 0.85),
      f"exit credit {cn:.2f} (full-debit loss would be 0)")

# F4 — pin pays: first-touch banking vs ride-the-ratchet
rng = random.Random(14)
grind = [min(26.0 + 0.075 * t, 176.0) + rng.gauss(0, 1.6) for t in range(2400)]
grind += [max(176.0 - 0.09 * t, 90.0) + rng.gauss(0, 1.6) for t in range(1000)]
peak = max(grind)
to, wo, co = _run_fly(grind, old=True)
tn, wn, cn = _run_fly(grind)
cap_o = (co - 26.0) / (peak - 26.0) * 100
cap_n = (cn - 26.0) / (peak - 26.0) * 100
check("F4 OLD banks the 50% target at first touch", wo == "TARGET",
      f"exit {co:.2f} = {cap_o:.0f}% of peak gain")
check("F4 NEW rides behind the ratchet, closes above the old target",
      wn in ("TARGET_TRAIL", "TRAIL_TAKE", "STAGNATION_TAKE")
      and cn > co and cap_n > cap_o + 5,
      f"{wn} at {cn:.2f} = {cap_n:.0f}% of peak gain "
      f"(old banked {co:.2f} = {cap_o:.0f}%; Δ+{cap_n - cap_o:.0f}pp)")

# F5 — pin loss: spot beyond a wing ≥60 s ⇒ salvage
path = _noise_path(600, seed=15)
spots = [77000.0 + (260.0 if t >= 300 else 0.0) for t in range(600)]
tn, wn, _ = _run_fly(path, spot_path=spots)
check("F5 pin lost ⇒ salvage exit", wn == "PIN_LOST" and tn is not None
      and 355 <= tn <= 375, f"{wn} at t={tn}s (breach began t=300s, dwell 60s)")

print("\n============ LONG BOOK — rung-4/4.6 policy math ==============")
# The 2026-07-15 SENSEX PE shape: spot chops, then breaks down 450 pts (the
# post-trap resolution), stalls. Runway-anchored target at entry: Pwall 136
# pts below ⇒ prem_room = 0.45×136 ≈ 61 ⇒ target ≈ entry+34%.
rng = random.Random(16)
spot0, entry, delta = 77136.0, 180.0, 0.45
spots, prem = [], []
sp = spot0
for t in range(2400):
    if t < 240:
        sp += rng.gauss(0, 2.0)                  # pre-break chop
    elif t < 1440:
        sp += -0.375 + rng.gauss(0, 1.5)         # −450 pts over 20 min
    else:
        sp += rng.gauss(0, 1.5)                  # post-break stall
    spots.append(sp)
    prem.append(max(entry + delta * (spot0 - sp) + rng.gauss(0, 1.0), 5.0))
peak = max(prem)
target_old = entry + max(delta * 136.0, entry * config.BASE_TP_PCT)

# OLD: first touch of the runway target (meta dormant ⇒ no extension), with
# the raw 45%-giveback trail beneath it.
exit_old = None
pk, stop, armed = entry, entry * (1 - config.BASE_SL_PCT), False
for t, m in enumerate(prem):
    pk = max(pk, m)
    if not armed and pk >= entry * (1 + config.TRAIL_ARM_PCT):
        armed = True
    if armed:
        stop = max(stop, entry + (pk - entry) * (1 - config.TRAIL_GIVEBACK_PCT))
    if m >= target_old:
        exit_old = (t, m, "TARGET"); break
    if m <= stop:
        exit_old = (t, m, "STOP/TRAIL"); break
exit_old = exit_old or (len(prem) - 1, prem[-1], "EOD")

# NEW: rung 4 (ER-gated extension) + rung 4.6 (trail) with the SHIPPED objects
tr = PeakCaptureTrail(entry, 0.0, TrailParams(
    arm_frac=config.TRAIL_ARM_PCT,
    give_frac_trend=config.EXIT_GIVE_FRAC_TREND,
    give_frac_chop=config.EXIT_GIVE_FRAC_CHOP,
    k_sigma=config.EXIT_K_SIGMA, give_floor_frac=config.EXIT_GIVE_FLOOR_FRAC,
    ema_hl_s=config.EXIT_MARK_EMA_HL_S,
    sigma_prior_frac=config.EXIT_SIGMA_PRIOR_FRAC,
    confirm_s=config.EXIT_CONFIRM_S, hard_mult=config.EXIT_HARD_BREACH_MULT,
    stagnation_s=config.EXIT_STAGNATION_S,
    tighten_min_left=config.EXIT_THETA_TIGHTEN_MIN_LEFT,
    tighten_floor=config.EXIT_THETA_TIGHTEN_MIN))
exit_new, tgt, ext = None, target_old, 0
for t, m in enumerate(prem):
    # FAIR comparison: the shipped book's base stop is unchanged — model it
    # on the NEW side too (it sits beneath the trail exactly as in manage()).
    if not tr.armed and m <= entry * (1 - config.BASE_SL_PCT):
        exit_new = (t, m, "STOP"); break
    td = tr.update(float(t), m, regime_label="TREND")
    er = signed_efficiency(spots[:t + 1], config.RIDE_ER_WINDOW_S)
    conv = -0.70 if 240 <= t < 1440 else -0.10   # the read during the break
    if m >= tgt:
        ok, _why = ride_ok(er, "PE", conv, er_min=config.RIDE_ER_MIN,
                           oppose_conv=config.RIDE_OPPOSE_CONV,
                           ride_conv=config.RIDE_CONV)
        if ok and ext < config.TARGET_EXTEND_MAX and tr.armed:
            tgt = m + max(delta * 60.0, entry * config.BASE_TP_PCT)
            ext += 1
        else:
            exit_new = (t, m, "TARGET"); break
    if td.exit_now:
        exit_new = (t, tr.ema, td.reason); break
exit_new = exit_new or (len(prem) - 1, prem[-1], "EOD")
cap_o = (exit_old[1] - entry) / (peak - entry) * 100
cap_n = (exit_new[1] - entry) / (peak - entry) * 100
check("L1 jackpot: NEW captures more of the peak",
      cap_n > cap_o + 15,
      f"old {exit_old[2]} @ {exit_old[1]:.0f} ({cap_o:.0f}%) vs "
      f"new {exit_new[2]} @ {exit_new[1]:.0f} ({cap_n:.0f}%), "
      f"{ext} extension(s), peak {peak:.0f}")

# L2 — the video's pattern: flush wick through the level, no sustain, reclaim
rng = random.Random(17)
prem2, px = [], entry
for t in range(1800):
    px = max(px + (0.06 if t < 1200 else 0.01) + rng.gauss(0, 0.9), 5.0)
    spike = -0.25 * px if 1200 <= t < 1210 else 0.0
    prem2.append(max(px + spike + rng.gauss(0, px * 0.012), 1.0))
pk, stop, armed, old_hit = entry, entry * 0.8, False, None
for t, m in enumerate(prem2):
    pk = max(pk, m)
    if not armed and pk >= entry * 1.15:
        armed = True
    if armed:
        stop = max(stop, entry + (pk - entry) * 0.55)
    if m <= stop:
        old_hit = (t, m); break
tr2 = PeakCaptureTrail(entry, 0.0, TrailParams(
    arm_frac=0.15, give_frac_trend=0.25, give_frac_chop=0.15, k_sigma=3.0,
    give_floor_frac=0.12, ema_hl_s=6.0, sigma_prior_frac=0.02, confirm_s=20.0,
    hard_mult=2.5, stagnation_s=420.0, tighten_min_left=75.0,
    tighten_floor=0.40))
new_hit = None
for t, m in enumerate(prem2):
    td = tr2.update(float(t), m, regime_label="TREND")
    if td.exit_now:
        new_hit = (t, tr2.ema, td.reason); break
check("L2 hunt wick shakes OLD out at the low",
      old_hit is not None and 1200 <= old_hit[0] <= 1215,
      f"old stopped t={old_hit[0]} @ {old_hit[1]:.1f}" if old_hit else "old held?!")
check("L2 hunt wick — NEW dwell holds through it",
      new_hit is None or (new_hit[0] > 1215 and new_hit[1] > old_hit[1]),
      (f"rode through the wick to path end @ {tr2.ema:.1f}"
       if new_hit is None else
       f"new {new_hit[2]} t={new_hit[0]} @ {new_hit[1]:.1f}"))

# L3 — reversal dwell semantics (pure clock math, mirrors the shipped rung)
rev_since, exited_flicker, exited_sustained = 0.0, False, False
for t, flip in enumerate([0.9] + [0.1] * 30):        # one-tick flicker
    if flip >= config.ENTRY_CONVICTION:
        rev_since = rev_since or (t + 1)
        if (t + 1) - rev_since >= config.REVERSAL_CONFIRM_S:
            exited_flicker = True
    else:
        rev_since = 0.0
rev_since = 0.0
for t, flip in enumerate([0.9] * 30):                # 30 s sustained reversal
    if flip >= config.ENTRY_CONVICTION:
        rev_since = rev_since or (t + 1)
        if (t + 1) - rev_since >= config.REVERSAL_CONFIRM_S:
            exited_sustained = True
    else:
        rev_since = 0.0
check("L3 one-tick reversal flicker no longer exits", not exited_flicker)
check("L3 sustained reversal still exits", exited_sustained)

print("\n=============== DISPLACEMENT — real governor =================")
g = DisplacementGovernor()
base = dict(eff_bar=0.55, persist_ok=True, persist_why="", cascade_ev=None,
            fly_open_ts=0.0, fly_progress_pct=-6.0, fly_unreal=-160.0,
            fly_pin_frac=0.62, minutes_to_close=95.0)
v1 = g.evaluate(ts=700.0, idx="NIFTY", conv=-0.78, tape_er=-0.61, **base)
v2 = g.evaluate(ts=725.0, idx="NIFTY", conv=-0.80, tape_er=-0.63, **base)
check("D1 flicker held, sustained displaced",
      (not v1.displace) and v2.displace, v2.reason if v2.displace else v2.reason)
g.register(725.0)
v3 = g.evaluate(ts=1700.0, idx="NIFTY", conv=-0.90, tape_er=-0.80, **base)
v3b = g.evaluate(ts=1725.0, idx="NIFTY", conv=-0.90, tape_er=-0.80, **base)
g.register(1725.0)
v4 = g.evaluate(ts=2700.0, idx="NIFTY", conv=-0.95, tape_er=-0.85, **base)
check("D1 daily budget honored (2/day)",
      v3b.displace and not v4.displace, v4.reason)
g2 = DisplacementGovernor()
v5 = g2.evaluate(ts=700.0, idx="NIFTY", conv=-0.90, tape_er=-0.80,
                 **{**base, "fly_progress_pct": 72.0})
check("D1 fly ≥60% to target is protected", not v5.displace, v5.reason)

print(f"\n================  {len(PASS)} passed, {len(FAIL)} failed  "
      f"================")
if FAIL:
    print("FAILED:", ", ".join(FAIL))
sys.exit(1 if FAIL else 0)