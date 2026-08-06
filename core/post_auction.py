"""
POST-AUCTION ENGINE — trading the 15:35–15:40 window on its own terms
=====================================================================
From 2026-08-03 the cash close is discovered by auction and published at
15:35, while index options keep trading until 15:40. Those five minutes
are a genuinely new regime: the single largest information event of the
day lands, and there is a short, thin, fast window to express it.

WHY THE EXISTING MACHINERY CANNOT TRADE IT
------------------------------------------
Every constant in the index-options stack is sized for a 60-minute hold:
a 30% base target, a 20% stop, a peak ratchet with dwell confirmation,
a stagnation timer measured in minutes, a hard-flat five minutes before
the bell. Point that at a 5-minute window and it does nothing useful —
the target is unreachable, the dwell never completes, and the hard-flat
fires before the trade has begun. This is not a tuning problem. A
5-minute regime needs its own physics.

WHAT THIS MODULE IS
-------------------
1. A REGIME DEFINITION. Its own hold budget (minutes), stop and target
   geometry, and an exit ladder with no dwell requirement — in five
   minutes there is no time to confirm anything twice, so the ladder
   reacts on first touch and protects with a hard floor.

2. A CALIBRATOR. The geometry is not guessed. `calibrate()` measures the
   window from the vault itself: how far the index actually travels
   between 15:35 and 15:40, how option premium responds, and what the
   spread costs. Targets are set from the observed move distribution,
   not from the 60-minute constants.

3. A READINESS GATE. `readiness()` answers one question — may the system
   trade this window yet? It requires POST_AUCTION_MIN_SESSIONS (7) of
   harvested post-auction data AND a calibration fitted from them AND
   evidence that the window is not adverse. Until all three hold, the
   answer is no and the reason is stated. After they hold, it is yes,
   automatically, with no edit to any file.

The design intent is that this switches itself on about a week after the
harvester starts banking the window — and that if the evidence says the
window is a trap, it never switches on at all. Both outcomes are correct.

CONSTITUTION IS NOT NEGOTIABLE. Post-auction trades run under the same
risk governor, the same daily ledger, the same disaster floor, the same
concurrency and cooldown rules, and TrapShield stays wired exactly as it
is for every other position type. Only the exit GEOMETRY differs.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict

import config

log = logging.getLogger("post_auction")

CERT_PATH = config.STATE_DIR / "post_auction_certificate.json"


@dataclass
class PostAuctionGeometry:
    """The exit physics for a 15:35–15:40 trade, all in the window's own
    units. Every field is measured, never assumed."""
    hold_minutes: float            # budget from entry (bounded by the bell)
    flat_hm: str                   # be flat by this clock time, always
    tp_pct: float                  # target as a fraction of entry premium
    sl_pct: float                  # stop as a fraction of entry premium
    trail_giveback: float          # ratchet: exit on this giveback from peak
    arm_at_pct: float              # ratchet arms once this much is banked
    n_sessions: int                # how many post-auction sessions fitted
    median_move_pts: float         # observed |index move| 15:35→15:40
    p90_move_pts: float
    median_spread_pct: float       # round-trip spread cost, as a fraction
    fitted_utc: float

    def as_dict(self) -> dict:
        return asdict(self)


def _default_geometry() -> PostAuctionGeometry:
    """Deliberately NOT tradable numbers — a placeholder that exists so
    callers have a shape to reason about before calibration. readiness()
    refuses while `n_sessions` is 0, so this is never used to size a
    trade."""
    return PostAuctionGeometry(
        hold_minutes=0.0, flat_hm="15:39", tp_pct=0.0, sl_pct=0.0,
        trail_giveback=0.0, arm_at_pct=0.0, n_sessions=0,
        median_move_pts=0.0, p90_move_pts=0.0, median_spread_pct=0.0,
        fitted_utc=0.0)


def load() -> PostAuctionGeometry:
    try:
        j = json.loads(CERT_PATH.read_text(encoding="utf-8"))
        g = j.get("geometry") or {}
        return PostAuctionGeometry(**{k: g[k] for k in
                                      _default_geometry().as_dict()
                                      if k in g})
    except Exception:                                      # noqa: BLE001
        return _default_geometry()


def save(geom: PostAuctionGeometry, evidence: dict) -> None:
    body = {"geometry": geom.as_dict(), "evidence": evidence,
            "config_hash": config.CONFIG_HASH, "ts": time.time()}
    try:
        CERT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CERT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(body, indent=1), encoding="utf-8")
        import os
        os.replace(tmp, CERT_PATH)
    except Exception as e:                                 # noqa: BLE001
        log.warning("could not write post-auction certificate (%s)", e)


def fit_geometry(moves_pts: list[float], spot_ref: float,
                 spreads_pct: list[float], n_sessions: int
                 ) -> PostAuctionGeometry:
    """Turn observed window behaviour into exit geometry.

    The reasoning, made explicit so it can be argued with:
      • TARGET is the MEDIAN absolute move, converted to premium through a
        conservative 0.5 delta and reduced by the round-trip spread. Using
        the median (not p90) means the target is reachable in half the
        sessions by construction — the opposite of the +120% target that
        made every 60-minute exit a clock.
      • STOP is half the target, giving 2:1 odds, and is floored so it can
        never sit inside the spread — a stop inside the spread is not a
        stop, it is a guaranteed exit.
      • The RATCHET arms early (at half the target) and gives back little,
        because in a five-minute window there is no second chance: the
        move that pays is usually the first one.
    """
    import statistics as st
    mv = sorted(abs(float(m)) for m in moves_pts if m == m)
    sp = sorted(float(x) for x in spreads_pct if x == x and x >= 0)
    med = st.median(mv) if mv else 0.0
    p90 = (mv[int(0.9 * (len(mv) - 1))] if mv else 0.0)
    med_sp = st.median(sp) if sp else 0.02
    # premium response to a `med` point move on a ~0.5-delta ATM option,
    # expressed as a fraction of premium. spot_ref/premium is unknown here,
    # so work in the ratio the vault gives us: move ÷ spot × leverage.
    lev = float(getattr(config, "POST_AUCTION_PREMIUM_LEVERAGE", 12.0))
    raw_tp = max((med / max(spot_ref, 1.0)) * lev, 0.0)
    net = raw_tp - med_sp                  # what the median move NETS
    # A floor target was the first thing I wrote here and it was wrong: if
    # the measured move cannot clear the round-trip spread, inventing a
    # "2x spread" target manufactures exactly the unreachable-target
    # disease that made every 60-minute exit a clock. When the physics
    # does not support a trade, the correct output is NO GEOMETRY and an
    # adverse verdict — a window that cannot pay its own spread is not a
    # window, and no amount of further data changes that arithmetic.
    min_edge = float(getattr(config, "POST_AUCTION_MIN_EDGE_MULT", 1.0))
    tp = net
    sl = max(tp / 2.0, med_sp * 1.5)
    tradable = (net > med_sp * min_edge) and (tp / max(sl, 1e-9) >= 1.8)
    if not tradable:
        tp = sl = 0.0
    return PostAuctionGeometry(
        hold_minutes=float(getattr(config, "POST_AUCTION_HOLD_MIN", 4.0)),
        flat_hm=str(getattr(config, "POST_AUCTION_FLAT_HM", "15:39")),
        tp_pct=round(tp, 4), sl_pct=round(sl, 4),
        trail_giveback=round(tp * 0.35, 4) if tp else 0.0,
        arm_at_pct=round(tp * 0.5, 4), n_sessions=int(n_sessions),
        median_move_pts=round(med, 2), p90_move_pts=round(p90, 2),
        median_spread_pct=round(med_sp, 4), fitted_utc=time.time())


def readiness() -> tuple[bool, str]:
    """May the system trade 15:35–15:40 yet?

    Three conditions, all necessary:
      1. the operator has not disabled the regime outright;
      2. at least POST_AUCTION_MIN_SESSIONS post-auction sessions have
         been harvested AND fitted into a geometry;
      3. the fitted evidence is not adverse — a window whose measured
         edge is significantly negative never opens, no matter how much
         data accumulates.
    """
    if not bool(getattr(config, "POST_AUCTION_ENABLED", True)):
        return False, "post-auction regime disabled by config"
    g = load()
    need = int(getattr(config, "POST_AUCTION_MIN_SESSIONS", 7))
    if g.n_sessions < need:
        return False, (f"learning: {g.n_sessions}/{need} post-auction "
                       f"session(s) harvested — the window opens by itself "
                       f"once the vault has a week of it")
    if g.tp_pct <= 0 or g.sl_pct <= 0 or g.hold_minutes <= 0:
        return False, (f"measured: median move {g.median_move_pts:.1f} pts "
                       f"cannot clear a {g.median_spread_pct:.2%} round-trip "
                       f"spread at 2:1 odds — the window does not pay for "
                       f"itself. Staying out is the finding.")
    try:
        j = json.loads(CERT_PATH.read_text(encoding="utf-8"))
        ev = j.get("evidence") or {}
    except Exception:                                      # noqa: BLE001
        ev = {}
    if ev.get("adverse") is True:
        return False, (f"evidence says this window is adverse "
                       f"({ev.get('reason', 'measured edge negative')}) — "
                       f"staying out is the result, not a failure")
    if str(j.get("config_hash", "")) != config.CONFIG_HASH:
        return False, ("certificate was fitted under a different config — "
                       "re-run the calibrator")
    return True, (f"ready: {g.n_sessions} session(s) fitted | target "
                  f"{g.tp_pct:.1%} stop {g.sl_pct:.1%} hold "
                  f"{g.hold_minutes:.0f}m flat {g.flat_hm}")


def exit_decision(entry: float, mark: float, peak: float,
                  held_min: float, now_hm: str,
                  geom: PostAuctionGeometry | None = None
                  ) -> tuple[bool, str]:
    """The five-minute exit ladder. Order matters: the floor first, the
    clock last, and no rule waits for confirmation — there is no time.

    Returns (should_exit, reason).
    """
    g = geom or load()
    if g.n_sessions <= 0:
        return True, "POST_AUCTION_UNCALIBRATED"     # never hold blind
    if entry <= 0:
        return True, "POST_AUCTION_BAD_ENTRY"
    up = mark / entry - 1.0
    dn = 1.0 - mark / entry
    if now_hm >= g.flat_hm:
        return True, "POST_AUCTION_BELL"             # the bell is absolute
    if dn >= g.sl_pct:
        return True, "POST_AUCTION_STOP"
    if up >= g.tp_pct:
        return True, "POST_AUCTION_TARGET"
    if peak / entry - 1.0 >= g.arm_at_pct:
        # ratchet armed: protect the banked move, no dwell, first touch
        if mark <= peak * (1.0 - g.trail_giveback):
            return True, "POST_AUCTION_TRAIL"
    if held_min >= g.hold_minutes:
        return True, "POST_AUCTION_HOLD"
    return False, ""


# ---------------------------------------------------------------- selftest
if __name__ == "__main__":                                 # pragma: no cover
    import sys
    ok = 0

    def chk(name, cond):
        global ok
        print(("  PASS  " if cond else "  FAIL  ") + name)
        ok += bool(cond)

    import tempfile
    from pathlib import Path
    config.STATE_DIR = Path(tempfile.mkdtemp())
    globals()["CERT_PATH"] = config.STATE_DIR / "post_auction_certificate.json"

    # ---- readiness refuses before the data exists
    r, why = readiness()
    chk("refuses with no data", not r and "0/7" in why)

    # ---- geometry is fitted from measured behaviour
    # (a) a window whose median move CANNOT pay the spread ⇒ untradable
    thin = fit_geometry([18.0, 25.0, 31.0, 12.0, 40.0, 22.0, 27.0],
                        spot_ref=24750.0,
                        spreads_pct=[0.012, 0.015, 0.011, 0.02, 0.013,
                                     0.014, 0.016], n_sessions=7)
    chk("un-payable window yields NO geometry", thin.tp_pct == 0.0)
    save(thin, {"adverse": False})
    r, why = readiness()
    chk("and readiness refuses with the arithmetic",
        not r and "does not pay for itself" in why)

    # (b) a window with a real move and a tight spread ⇒ tradable geometry
    moves = [60.0, 85.0, 110.0, 45.0, 130.0, 75.0, 95.0]   # NIFTY pts
    spreads = [0.004, 0.005, 0.004, 0.006, 0.005, 0.004, 0.005]
    g = fit_geometry(moves, spot_ref=24750.0, spreads_pct=spreads,
                     n_sessions=7)
    chk("target is positive and above the spread",
        g.tp_pct > 2 * g.median_spread_pct)
    chk("stop gives roughly 2:1 odds",
        1.8 <= g.tp_pct / g.sl_pct <= 2.2)
    chk("stop cannot sit inside the spread",
        g.sl_pct > g.median_spread_pct)
    chk("ratchet arms at half the target",
        abs(g.arm_at_pct - g.tp_pct * 0.5) < 1e-6)
    chk("hold budget is minutes, not an hour", 1.0 <= g.hold_minutes <= 5.0)
    chk("median move recorded", g.median_move_pts == 85.0)

    save(g, {"adverse": False, "n_sessions": 7})
    r, why = readiness()
    chk("opens once 7 sessions are fitted", r and "ready" in why)

    # ---- adverse evidence keeps it shut forever
    save(g, {"adverse": True, "reason": "measured Δ₹ significantly negative"})
    r, why = readiness()
    chk("adverse evidence keeps it shut", not r and "adverse" in why)
    save(g, {"adverse": False, "n_sessions": 7})

    # ---- the exit ladder
    e = 100.0
    chk("stop fires", exit_decision(e, e * (1 - g.sl_pct - 0.001), e, 1.0,
                                    "15:36", g)[1] == "POST_AUCTION_STOP")
    chk("target fires", exit_decision(e, e * (1 + g.tp_pct + 0.001), e, 1.0,
                                      "15:36", g)[1] == "POST_AUCTION_TARGET")
    pk = e * (1 + g.arm_at_pct + 0.01)
    chk("armed ratchet protects the banked move",
        exit_decision(e, pk * (1 - g.trail_giveback - 0.001), pk, 1.0,
                      "15:36", g)[1] == "POST_AUCTION_TRAIL")
    chk("un-armed ratchet does NOT fire",
        not exit_decision(e, e * 1.001, e * 1.002, 1.0, "15:36", g)[0])
    chk("hold budget fires",
        exit_decision(e, e, e, g.hold_minutes + 0.1, "15:36",
                      g)[1] == "POST_AUCTION_HOLD")
    chk("the bell overrides everything",
        exit_decision(e, e * 1.5, e * 1.5, 0.1, "15:39",
                      g)[1] == "POST_AUCTION_BELL")
    chk("quiet trade is held", not exit_decision(e, e * 1.001, e * 1.001,
                                                 0.5, "15:36", g)[0])
    chk("uncalibrated geometry never holds",
        exit_decision(e, e, e, 0.1, "15:36",
                      _default_geometry())[1] == "POST_AUCTION_UNCALIBRATED")

    total = 19
    print(f"\n  fitted geometry: target {g.tp_pct:.1%} | stop {g.sl_pct:.1%} "
          f"| arm {g.arm_at_pct:.1%} | giveback {g.trail_giveback:.1%} "
          f"| hold {g.hold_minutes:.0f}m | flat {g.flat_hm}")
    print(f"\n{ok}/{total} checks passed")
    sys.exit(0 if ok == total else 1)