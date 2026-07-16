"""
APEX OMNI v9.7.1 — DISPLACEMENT GOVERNOR (opportunity-cost exit for the fly)
============================================================================
The problem (2026-07-15 live evidence)
--------------------------------------
The v10.1 GLOBAL SINGLE-POSITION LOCK is the right capital constitution — one
live position in the whole system — but it was implemented as a BLIND lock:
while a fly is open, the brain `continue`s past the cascade path and the
entire 1 Hz entry stack on BOTH indices. No incoming signal, however strong,
is ever weighed against the position being held. On 2026-07-15 a SENSEX fly
pinned in CHOP bled −60…−405 for 30+ minutes while both indices sat locked —
the exact session the operator watched a directional breakdown pay elsewhere.

A held position is not free: its opportunity cost is the best trade the
locked capital cannot take. That comparison is a portfolio decision, and this
module makes it explicit, conservative, and fully logged — instead of the
lock silently deciding "hold" every second by omission.

What this is NOT
----------------
Not a second entry gate, and not a shortcut around the first one. The
governor can only ever CLOSE the fly (releasing the lock); the freed capital
then re-enters through the untouched constitution — decision gate,
persistence, throttle, RiskGovernor, spread gate, chase cap. Buy-only is
preserved on both sides: unwinding a debit fly sells owned wings and buys
back the covered body (no naked leg at any instant), and the replacement
trade is a plain bought option through the normal path.

Displacement discipline (ALL must hold — every rejection is named)
------------------------------------------------------------------
 TIER A — structural: a CASCADE event (the certified crash-regime trigger)
   on EITHER index displaces immediately. The fly's own gate already treats
   cascade state as a veto on its index; a cascade on the OTHER index was
   being swallowed by the blind lock.
 TIER B — statistical: a directional read that clears the ENTRY bar PLUS a
   margin, passes the SAME persistence physics as an entry (coherence +
   Kaufman ER + tape agreement via core/decision.PersistenceTracker), whose
   signed tape ER agrees, and which has itself sustained for a minimum
   number of seconds. One flicker never displaces.
 AND the fly must be fairly beaten, not merely present:
   • minimum hold honoured (the pin thesis gets its fair chance),
   • the fly is NOT close to paying (progress-to-target below a cap),
   • a fly that is pinning AND green demands the STRONG margin,
   • enough session left for the freed capital to actually work
     (waived for Tier A — a cascade is precisely the storm the fly must
      not be holding through),
   • daily displacement budget + cooldown (rotation is a scalpel, not a
     metronome — churn is how four-leg costs eat an account).

Research anchor: this is the standard opportunity-cost/switching-option
treatment — hold value vs best alternative net of switching costs, with
hysteresis so the comparison doesn't chatter (cf. Dixit–Pindyck, *Investment
under Uncertainty*, on switching with frictions). The hysteresis here is the
margin + sustain + cooldown triplet.

Certification honesty: a DISPLACED close is a PORTFOLIO decision, not a fly
policy outcome — the fly's certificate must not be graded on trades the
governor censored. tools/butterfly_harness.py therefore excludes
why=DISPLACED rows from the certificate blend and reports them separately.

Self-test:   python core/displacement.py
"""
from __future__ import annotations

from dataclasses import dataclass, field

try:
    import config
except Exception:                                          # pragma: no cover
    config = None                  # standalone self-test / external import


def _cfg(name: str, default):
    return getattr(config, name, default) if config is not None else default


@dataclass
class DisplacementVerdict:
    displace: bool
    reason: str                    # human-readable grant or the named refusal
    tier: str = ""                 # "A" (cascade) | "B" (statistical)
    index: str = ""                # candidate index
    side: str = ""                 # candidate direction CE/PE
    diag: dict = field(default_factory=dict)


class DisplacementGovernor:
    """One instance per brain session. evaluate() is called at the 1 Hz
    decision cadence for each index while a fly holds the global lock."""

    def __init__(self):
        self.count_today = 0
        self.last_disp_ts = -1e18
        self._cand_since: dict[tuple[str, str], float] = {}
        self.last_refusal = ""     # surfaced on the heartbeat

    # ------------------------------------------------------------- helpers
    def _sustained(self, ts: float, idx: str, side: str,
                   qualifies: bool) -> float:
        """Wall-clock seconds the candidate read has continuously qualified.
        Any second it fails, the clock resets — same philosophy as the entry
        persistence window."""
        key = (idx, side)
        if not qualifies:
            self._cand_since.pop(key, None)
            return 0.0
        t0 = self._cand_since.setdefault(key, float(ts))
        # a qualifying read on the OTHER side voids this side's clock
        self._cand_since.pop((idx, "CE" if side == "PE" else "PE"), None)
        return float(ts) - t0

    def register(self, ts: float) -> None:
        self.count_today += 1
        self.last_disp_ts = float(ts)
        self._cand_since.clear()

    # ------------------------------------------------------------ decision
    def evaluate(self, *, ts: float, idx: str, conv: float, eff_bar: float,
                 persist_ok: bool, persist_why: str,
                 tape_er: float | None, cascade_ev,
                 fly_open_ts: float, fly_progress_pct: float | None,
                 fly_unreal: float | None, fly_pin_frac: float | None,
                 minutes_to_close: float) -> DisplacementVerdict:
        """All fly_* readings come from FlyBook.mark() SMOOTHED vitals where
        available (never a single raw print).

        fly_progress_pct : % of the way from debit to the profit target
        fly_pin_frac     : |spot − body| / wing_width  (0 = perfect pin)
        tape_er          : SIGNED Kaufman ER (core/exit_engine.signed_efficiency)
        """
        if not _cfg("DISP_ENABLED", True):
            return DisplacementVerdict(False, "displacement disabled")

        side = "CE" if conv > 0 else "PE"
        tier_a = cascade_ev is not None

        # ---------- budget / tempo (checked first: cheap, absolute) --------
        if self.count_today >= int(_cfg("DISP_MAX_PER_DAY", 2)):
            self.last_refusal = "disp budget spent"
            return DisplacementVerdict(False, self.last_refusal)
        if ts - self.last_disp_ts < float(_cfg("DISP_COOLDOWN_S", 900.0)):
            self.last_refusal = "disp cooldown"
            return DisplacementVerdict(False, self.last_refusal)

        # ---------- the fly gets its fair chance ---------------------------
        held = ts - float(fly_open_ts)
        min_hold = float(_cfg("FLY_MIN_HOLD_BEFORE_DISP_S", 600.0))
        if held < min_hold and not tier_a:
            self.last_refusal = f"fly held {held:.0f}s<{min_hold:.0f}s"
            return DisplacementVerdict(False, self.last_refusal)
        prog = fly_progress_pct if fly_progress_pct is not None else 0.0
        prog_cap = float(_cfg("DISP_FLY_PROGRESS_MAX", 0.60)) * 100.0
        if prog >= prog_cap and not tier_a:
            self.last_refusal = f"fly {prog:.0f}%→target (≥{prog_cap:.0f}%)"
            return DisplacementVerdict(False, self.last_refusal)

        # ---------- TIER A: cascade on either index ------------------------
        if tier_a:
            self.last_refusal = ""
            return DisplacementVerdict(
                True, f"CASCADE {getattr(cascade_ev, 'kind', '?')} "
                      f"z={getattr(cascade_ev, 'z', 0.0):+.1f} on {idx} — "
                      f"structural displacement",
                tier="A", index=idx,
                side=getattr(cascade_ev, "direction", side),
                diag={"held_s": round(held, 0)})

        # ---------- session runway (Tier B only) ----------------------------
        min_left = float(_cfg("DISP_MIN_MINUTES_LEFT", 45.0))
        if minutes_to_close < min_left:
            self.last_refusal = f"{minutes_to_close:.0f}m left<{min_left:.0f}m"
            return DisplacementVerdict(False, self.last_refusal)

        # ---------- TIER B: statistical candidate --------------------------
        margin = float(_cfg("DISP_CONV_MARGIN", 0.10))
        pinning_green = ((fly_pin_frac is not None and fly_pin_frac < 0.5)
                         and (fly_unreal is not None and fly_unreal >= 0))
        if pinning_green:                      # working fly ⇒ higher burden
            margin = float(_cfg("DISP_CONV_MARGIN_STRONG", 0.20))
        bar = eff_bar + margin
        er_min = float(_cfg("DISP_ER_MIN", 0.45))
        want = 1 if conv > 0 else -1
        er_ok = (tape_er is not None and abs(tape_er) >= er_min
                 and (1 if tape_er > 0 else -1) == want)
        qualifies = abs(conv) >= bar and persist_ok and er_ok
        sustained = self._sustained(ts, idx, side, qualifies)
        need_s = float(_cfg("DISP_CAND_SUSTAIN_S", 20.0))

        diag = {"conv": round(conv, 3), "bar": round(bar, 3),
                "tape_er": (round(tape_er, 3) if tape_er is not None else None),
                "persist": persist_ok, "sustained_s": round(sustained, 1),
                "fly_prog_pct": round(prog, 1),
                "fly_pin_frac": (round(fly_pin_frac, 2)
                                 if fly_pin_frac is not None else None),
                "fly_unreal": fly_unreal, "held_s": round(held, 0)}

        if not qualifies:
            if abs(conv) < bar:
                self.last_refusal = f"conv {abs(conv):.2f}<{bar:.2f}+margin"
            elif not persist_ok:
                self.last_refusal = f"not persistent ({persist_why})"
            else:
                self.last_refusal = (f"tape ER "
                                     f"{(abs(tape_er) if tape_er is not None else 0):.2f}"
                                     f" weak/against")
            return DisplacementVerdict(False, self.last_refusal, diag=diag)
        if sustained < need_s:
            self.last_refusal = f"candidate {sustained:.0f}s<{need_s:.0f}s"
            return DisplacementVerdict(False, self.last_refusal, diag=diag)

        self.last_refusal = ""
        return DisplacementVerdict(
            True, f"{idx} {side} conv {conv:+.2f} ≥ bar+margin {bar:.2f}, "
                  f"persistent, tape ER {abs(tape_er):.2f} agreeing, "
                  f"sustained {sustained:.0f}s vs fly at {prog:.0f}%→target "
                  f"(held {held:.0f}s)",
            tier="B", index=idx, side=side, diag=diag)


# ----------------------------------------------------------------- self-test
if __name__ == "__main__":
    class _Ev:                                             # cascade stand-in
        kind, z, direction = "FLIP_BREAK", -2.7, "PE"

    g = DisplacementGovernor()
    base = dict(eff_bar=0.55, persist_ok=True, persist_why="",
                fly_open_ts=0.0, fly_progress_pct=-6.0, fly_unreal=-160.0,
                fly_pin_frac=0.62, minutes_to_close=95.0, cascade_ev=None)
    cases = [
        ("one strong flicker (0 s sustain)",
         dict(ts=700, idx="NIFTY", conv=-0.78, tape_er=-0.61, **base)),
        ("same read 25 s later (sustained)",
         dict(ts=725, idx="NIFTY", conv=-0.80, tape_er=-0.63, **base)),
        ("strong read, tape against",
         dict(ts=800, idx="NIFTY", conv=-0.80, tape_er=+0.55, **base)),
        ("fly 70% to target — protected",
         dict(ts=900, idx="NIFTY", conv=-0.85, tape_er=-0.70,
              **{**base, "fly_progress_pct": 70.0})),
        ("fly only 4 min old — fair chance",
         dict(ts=240, idx="NIFTY", conv=-0.85, tape_er=-0.70, **base)),
        ("cascade on the OTHER index (tier A, immediate)",
         dict(ts=250, idx="NIFTY", conv=-0.30, tape_er=-0.10,
              **{**base, "cascade_ev": _Ev()})),
    ]
    for name, kw in cases:
        # rebuild the sustain clock realistically for the 2nd case
        v = g.evaluate(**kw)
        print(f"{name:48s} → {'DISPLACE' if v.displace else 'hold':8s} "
              f"[{v.tier or '-'}] {v.reason}")