"""
APEX OMNI v9.7.1 — PEAK-CAPTURE EXIT ENGINE (one exit science, both books)
==========================================================================
Why this module exists (2026-07-15 live evidence)
-------------------------------------------------
Two independent exit failures observed live, one root cause:

1. LONG BOOK: the target is fixed at entry (delta × min(expected-move, GEX
   runway) — with the runway measured to the very wall a genuine breakdown is
   about to smash through) and pays out at FIRST TOUCH. The model-driven
   extension (META_HOLD_PAST_TARGET_P) is structurally dormant until
   META_MIN_TRAIN labeled trades exist, so in heuristic mode the system
   ALWAYS banks the first touch — a jackpot trend leaves with a fraction of
   the move ("exited early, at some lower level instead of the peak").

2. FLY BOOK: TP/SL fire at the first touch of the RAW conservative unwind
   credit (wings@bid − 2×body@ask). Live telemetry (2026-07-15) shows that
   mark oscillating 21.50 → 25.90 minute-to-minute on an essentially still
   spot — ±8% of the debit of pure quote/spread noise. A single wide-quote
   print can stop the structure out at the worst mark of the day, or bank a
   noise-touch "target" that value never reached.

Both are the same disease: exits triggered by ONE TICK of a NOISY executable
mark against a NON-RATCHETING level. This module is the cure, shared by both
books so live == harness == simulator by construction (the same way
core/decision unified the entry path).

The science (each mechanism carries its literature)
---------------------------------------------------
• SMOOTHED MARK + NOISE SCALE — decisions run on an EMA of the executable
  mark; a robust EWMA absolute-innovation estimate σ̂ prices how much of a
  giveback is indistinguishable from microstructure noise. A raw option mark
  bid-ask-bounces every tick even when value doesn't move (Roll 1984); a
  four-leg conservative unwind bounces four times as hard.
• CHANDELIER RATCHET — the trailing exit level is HWM − giveback and NEVER
  loosens (LeBeau's chandelier exit; Kaufman, *Trading Systems and Methods*),
  with giveback = max(K·σ̂, frac × peak-gain): the σ̂ term floors it above
  noise, the fraction term bounds how much of a runaway move can ever be
  surrendered — far tighter near the peak than the legacy fixed 45%.
• REGIME-CONDITIONAL WIDTH — Kaminski & Lo (2014), "When Do Stop-Loss Rules
  Stop Losses?" (Journal of Financial Markets): trailing exits ADD value
  under momentum and DESTROY it under mean-reversion. TREND regimes trail
  wide and ride; CHOP tightens and adds a stagnation clock (bank a stalled
  gain instead of paying theta to watch it decay).
• DWELL CONFIRMATION — a breach must SUSTAIN for a bounded number of seconds
  (or exceed a hard multiple of the giveback → waterfall escape) before it
  fires. This is the exit-side twin of core/trap_shield: the stop-hunt
  pattern — break, FAIL TO SUSTAIN, reclaim, then the real move — that
  shakes retail out at the low ("second candle sustain nahi kiya, retest me
  SL de diya") is filtered on the way OUT by the same sustained-read test the
  entry side already applies. The disaster floor NEVER waits (unchanged).
• EFFICIENCY-GATED RIDE — at the old take-profit line the question is "is
  the tape still trending?", answered by the Kaufman Efficiency Ratio on the
  spot path (the SAME core/signal_persistence.efficiency_ratio the entry
  gate trusts, signed by the window's net displacement). ER high and with the
  position ⇒ extend the target by expected-move increments (the exact
  mechanics the dormant meta extension already uses) and let the ratchet
  protect the gain underneath; ER low or against ⇒ bank the touch as today.

Nothing here loosens the constitution: disaster floor, profit-lock floor,
EOD flatten, stale-feed flatten, cascade veto and SV_CLOSE_HM are untouched
and still fire first. This engine can only ever exit a WINNER closer to its
peak, refuse to be shaken out by an unsustained spike, or bank a stalled
gain sooner — it cannot extend a loss past any existing floor.

Self-test:   python core/exit_engine.py     (old-vs-new scenario battery)
Full lab:    python tools/exit_lab.py       (calibrated to live telemetry)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# --------------------------------------------------------------------------
# Parameters — passed at construction (no config import): each book binds its
# own knob namespace, and the harness/spec-factory knob-patching keeps working
# because a fresh position constructs a fresh trail from the patched values.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrailParams:
    arm_frac: float           # arm the ratchet once smoothed HWM ≥ entry×(1+this)
    give_frac_trend: float    # giveback ≤ this fraction of peak gain (momentum)
    give_frac_chop: float     # …in mean-reverting regimes (tighter)
    k_sigma: float            # noise floor: giveback ≥ k · σ̂
    give_floor_frac: float    # …and ≥ this fraction of entry (pullback floor:
    #                           tick-noise σ̂ under-scales multi-second trend
    #                           pullbacks; premium retraces in % chunks)
    ema_hl_s: float           # mark-smoothing half-life, seconds
    sigma_prior_frac: float   # σ̂ warm-start = entry × this (adapts from data)
    confirm_s: float          # a ratchet breach must sustain this long (dwell)
    hard_mult: float          # smoothed mark ≤ HWM − this×giveback ⇒ fire NOW
    stagnation_s: float       # armed + no new HWM for this long ⇒ bank the gain
    tighten_min_left: float   # start theta-tightening inside this many minutes
    tighten_floor: float      # giveback multiplier floor as the clock runs out
    chop_labels: tuple = ("CHOP", "VOL_CRUSH")   # regimes that tighten/stagnate;
    #                                              ("*",) = stagnation any regime


@dataclass
class TrailDecision:
    exit_now: bool = False
    reason: str = ""
    urgent: bool = False


class PeakCaptureTrail:
    """Stateful per-position trailing engine over a noisy executable mark.

    Deterministic given the (ts, mark) stream — replayable byte-identically by
    the harness and the scenario simulator. All state is visible via vitals()
    for the heartbeat / Ⓕ line.
    """

    __slots__ = ("entry", "p", "ema", "sigma", "hwm", "hwm_ts", "ratchet",
                 "armed", "give", "breach_since", "last_ts", "n")

    def __init__(self, entry: float, ts: float, params: TrailParams):
        self.entry = float(entry)
        self.p = params
        self.ema = float(entry)
        self.sigma = max(float(entry) * params.sigma_prior_frac, 1e-6)
        self.hwm = float(entry)
        self.hwm_ts = float(ts)
        self.ratchet = 0.0            # inactive until armed
        self.armed = False
        self.give = 0.0
        self.breach_since = 0.0
        self.last_ts = float(ts)
        self.n = 0

    # ------------------------------------------------------------------ core
    def update(self, ts: float, raw_mark: float, *,
               regime_label: str = "",
               minutes_to_close: float | None = None) -> TrailDecision:
        """Advance the smoothed mark / noise / HWM / ratchet state and return
        the exit decision for THIS tick. Call once per management tick with
        the same executable mark the book would actually unwind at."""
        raw = float(raw_mark)
        if raw <= 0:                       # dead/one-sided book: no read
            return TrailDecision()
        dt = min(max(float(ts) - self.last_ts, 0.05), 10.0) if self.n else 1.0
        self.last_ts = float(ts)
        self.n += 1
        alpha = 1.0 - 0.5 ** (dt / max(self.p.ema_hl_s, 0.1))
        inn = raw - self.ema
        self.ema += alpha * inn
        # EWMA absolute innovation ≈ the tick-noise scale of the raw mark
        # around smoothed value (robust to fat tails vs a squared estimator).
        self.sigma = (1.0 - alpha) * self.sigma + alpha * abs(inn)

        if self.ema > self.hwm:
            self.hwm = self.ema
            self.hwm_ts = float(ts)
            self.breach_since = 0.0        # new high void any pending breach

        if not self.armed and self.hwm >= self.entry * (1.0 + self.p.arm_frac):
            self.armed = True

        if not self.armed:
            return TrailDecision()

        gain = self.hwm - self.entry
        chop = ("*" in self.p.chop_labels
                or (regime_label or "").upper() in self.p.chop_labels)
        frac = self.p.give_frac_chop if chop else self.p.give_frac_trend
        give = max(self.p.k_sigma * self.sigma,
                   self.p.give_floor_frac * self.entry, frac * gain)
        if (minutes_to_close is not None
                and minutes_to_close < self.p.tighten_min_left):
            give *= max(self.p.tighten_floor,
                        max(minutes_to_close, 0.0) / self.p.tighten_min_left)
        self.give = give
        self.ratchet = max(self.ratchet, self.hwm - give)

        # ---- waterfall escape: a collapse far past the giveback fires NOW —
        # waiting a dwell window through a genuine cliff is how -5% becomes
        # -20% on 0-2 DTE premium.
        if self.ema <= self.hwm - self.p.hard_mult * give:
            return TrailDecision(True, "TRAIL_HARD", urgent=True)

        # ---- dwell-confirmed ratchet breach: the anti-hunt exit. A spike
        # through the level that RECLAIMS inside confirm_s (the "second candle
        # didn't sustain" signature) never fires.
        if self.ema <= self.ratchet:
            if self.breach_since <= 0.0:
                self.breach_since = float(ts)
            elif float(ts) - self.breach_since >= self.p.confirm_s:
                return TrailDecision(True, "TRAIL_RATCHET")
        else:
            self.breach_since = 0.0

        # ---- stagnation take (Kaminski–Lo mean-reversion case): an armed
        # winner that has stopped making highs in a chopping tape is paying
        # theta for nothing — bank it.
        if (self.p.stagnation_s > 0 and chop
                and float(ts) - self.hwm_ts >= self.p.stagnation_s
                and self.ema > self.entry):
            return TrailDecision(True, "STAGNATION_TAKE")

        return TrailDecision()

    # ------------------------------------------------------------- telemetry
    def vitals(self) -> dict:
        return {"ema": round(self.ema, 2), "sigma": round(self.sigma, 3),
                "hwm": round(self.hwm, 2),
                "ratchet": (round(self.ratchet, 2) if self.armed else None),
                "give": round(self.give, 2), "armed": self.armed,
                "breach_s": (round(self.last_ts - self.breach_since, 1)
                             if self.breach_since > 0 else 0.0)}


# --------------------------------------------------------------------------
# EFFICIENCY-GATED RIDE — should a tagged target be banked or extended?
# --------------------------------------------------------------------------
def signed_efficiency(spot_window, window_s: int) -> float | None:
    """Kaufman ER over the last `window_s` per-second spots, SIGNED by the
    window's net displacement (+ = tape trending up, − = down). None until
    enough history. Reuses core/signal_persistence.efficiency_ratio — the
    same physics the entry gate trusts."""
    from core.signal_persistence import efficiency_ratio
    sw = list(spot_window or ())
    if len(sw) < max(int(window_s), 3):
        return None
    seg = sw[-int(window_s):]
    er = efficiency_ratio(seg)
    return er if seg[-1] >= seg[0] else -er


def ride_ok(signed_tape_er: float | None, direction: str, conviction: float,
            *, er_min: float, oppose_conv: float,
            ride_conv: float = 0.62) -> tuple[bool, str]:
    """The tagged target may be EXTENDED (ride) when the edge is demonstrably
    still on. Two independent qualifiers, EITHER sufficient, with hard vetoes
    that bind in both cases:

      QUALIFIER A — efficient tape: |ER| ≥ er_min AND pointing WITH the
                    position (a clean directional move on the underlying).
      QUALIFIER B — strong aligned conviction: |conv| ≥ ride_conv WITH the
                    position (a fused read this strong into the wall IS the
                    directional edge, even when the second-by-second path is
                    choppy — a −450-pt breakdown with realistic per-second
                    noise nets ER≈0.25–0.35, below er_min, yet is exactly the
                    'jackpot' the operator must not exit early). The tape
                    still may not be AGAINST the position for B to apply.

    VETOES (both qualifiers): the tape efficiently AGAINST the position, or
    conviction hard against (≤ −oppose_conv), always banks the target. Anything
    that clears neither qualifier banks it too — the conservative default."""
    if signed_tape_er is None and abs(conviction) < ride_conv:
        return False, "no tape read yet"
    want = 1 if direction == "CE" else -1
    conv_aligned = conviction * want
    # hard opposition veto — binds regardless of which qualifier is tried
    if conv_aligned < -abs(oppose_conv):
        return False, f"conv {conviction:+.2f} hard against"
    if signed_tape_er is not None:
        er = abs(signed_tape_er)
        tape_dir = 1 if signed_tape_er > 0 else -1
        # tape efficiently AGAINST the position vetoes the ride outright
        if er >= er_min and tape_dir != want:
            return False, f"tape ER {er:.2f} AGAINST position"
        # QUALIFIER A: efficient tape with the position
        if er >= er_min and tape_dir == want:
            return True, f"tape ER {er:.2f} with position"
    # QUALIFIER B: strong aligned conviction carries a choppy-but-real trend
    if conv_aligned >= ride_conv:
        _er_s = (f"{abs(signed_tape_er):.2f}"
                 if signed_tape_er is not None else "n/a")
        return True, f"conv {conviction:+.2f} strong with position (ER {_er_s})"
    _er_s = (f"{abs(signed_tape_er):.2f}"
             if signed_tape_er is not None else "n/a")
    return False, f"tape ER {_er_s}<{er_min:.2f} & conv {conviction:+.2f} weak"


# --------------------------------------------------------------------------
# SELF-TEST — old first-touch policy vs the peak-capture engine on the four
# scenario classes that motivated it. `python core/exit_engine.py`
# --------------------------------------------------------------------------
def _mk_path(kind: str, entry: float = 25.0, seed: int = 7,
             n: int = 2400) -> list[float]:
    import random
    rng = random.Random(seed)
    px, out = entry, []
    for t in range(n):
        if kind == "jackpot":              # trend to ~3.6× then fade to ~2.8×
            drift = 0.045 if t < 1500 else -0.02
        elif kind == "chop":
            drift = 0.0
        elif kind == "hunt":               # grind up, violent 20 s flush at
            drift = 0.02 if t < 1200 else 0.004   # t=1200, full reclaim, run on
        else:
            drift = 0.0
        px = max(px + drift + rng.gauss(0, 0.30), 0.5)
        spike = 0.0
        if kind == "hunt" and 1200 <= t < 1210:
            spike = -0.25 * px             # a 10 s flush wick, then reclaim.
            # (Deeper/longer flushes are the TrapShield's case: in the long
            # book every trail fire is shown to the shield before it executes,
            # so absorption/OI/spread fingerprints can hold a confirmed hunt —
            # dwell filters the wicks, the shield filters the sweeps.)
        out.append(max(px + spike + rng.gauss(0, px * 0.015), 0.05))
    return out


def _old_policy(path, entry, tp_pct=0.30, arm=0.15, give=0.45, sl=0.20):
    """The legacy exit: first touch of the fixed target, 45%-of-gain trail,
    raw-mark stop (shield not modeled — favourable to OLD in the hunt case)."""
    target = entry * (1 + tp_pct)
    stop, peak, armed = entry * (1 - sl), entry, False
    for t, m in enumerate(path):
        peak = max(peak, m)
        if not armed and peak >= entry * (1 + arm):
            armed = True
        if armed:
            stop = max(stop, entry + (peak - entry) * (1 - give))
        if m >= target:
            return t, m, "TARGET"
        if m <= stop:
            return t, m, "STOP/TRAIL"
    return len(path) - 1, path[-1], "EOD"


def _new_policy(path, entry, params: TrailParams, regime="TREND"):
    tr = PeakCaptureTrail(entry, 0.0, params)
    for t, m in enumerate(path):
        d = tr.update(float(t), m, regime_label=regime)
        if d.exit_now:
            return t, tr.ema, d.reason
    return len(path) - 1, tr.ema, "EOD"


if __name__ == "__main__":
    P = TrailParams(arm_frac=0.15, give_frac_trend=0.25, give_frac_chop=0.15,
                    k_sigma=3.0, give_floor_frac=0.12, ema_hl_s=6.0,
                    sigma_prior_frac=0.02, confirm_s=20.0, hard_mult=2.5,
                    stagnation_s=420.0, tighten_min_left=75.0,
                    tighten_floor=0.40)
    print(f"{'scenario':34s} | {'OLD exit':22s} | {'NEW exit':24s} | verdict")
    print("-" * 110)
    for kind, regime in (("jackpot", "TREND"), ("chop", "CHOP"),
                         ("hunt", "TREND")):
        path = _mk_path(kind)
        peak = max(path)
        ot, om, ow = _old_policy(path, 25.0)
        nt, nm, nw = _new_policy(path, 25.0, P, regime)
        cap_o = (om - 25.0) / max(peak - 25.0, 1e-9) * 100
        cap_n = (nm - 25.0) / max(peak - 25.0, 1e-9) * 100
        print(f"{kind:34s} | {ow:12s} {om:6.2f} ({cap_o:4.0f}%) | "
              f"{nw:14s} {nm:6.2f} ({cap_n:4.0f}%) | "
              f"{'NEW captures more of the peak' if cap_n > cap_o + 5 else 'parity'}")