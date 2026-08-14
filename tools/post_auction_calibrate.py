"""
POST-AUCTION CALIBRATOR — measure the 15:35–15:40 window, then decide
=====================================================================
    python tools/post_auction_calibrate.py [--json out.json]

Runs nightly. Every post-reform session in the vault contributes one
observation of the new window: how far the index actually travelled from
the auction close to the bell, and what the round-trip spread on the ATM
options cost. From those two distributions the exit geometry is fitted
(core.post_auction.fit_geometry) and the regime either opens or does not.

THE DECISION RULE, STATED PLAINLY
---------------------------------
  • Fewer than POST_AUCTION_MIN_SESSIONS sessions ⇒ keep learning. The
    tool reports how many more are needed and changes nothing.
  • Enough sessions, and the median move NETS more than the round-trip
    spread at 2:1 odds ⇒ write the geometry, and the window opens by
    itself on the next session. No file edit, no flag to remember.
  • Enough sessions, and it does not ⇒ write an ADVERSE verdict. The
    window stays shut permanently unless the measured arithmetic
    changes. A five-minute window that cannot pay its own spread is not
    an opportunity, and waiting longer will not make the spread smaller.

Every fit is registered as a trial, so the deflation accounting used by
every other claim in this system also counts this one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                              # noqa: E402

config.setup_logging("post_auction_calibrate")
import logging                                             # noqa: E402
log = logging.getLogger("post_auction_calibrate")

from core import post_auction as PA                        # noqa: E402
from core import session_calendar as SC                    # noqa: E402


def _hm_to_min(hm: str) -> int:
    h, m = str(hm).split(":")[:2]
    return int(h) * 60 + int(m)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()
    if not bool(getattr(config, "POST_AUCTION_ENABLED", True)):
        log.info("post-auction regime disabled by config — nothing to do")
        return 0

    from nightly_forge_v9 import trading_days
    from simulation.replay_real_day import load_day
    con = sqlite3.connect(str(config.DB_PATH))
    moves, spreads, sessions, spot_ref = [], [], [], []
    degenerate = 0   # sessions whose window held no REAL spot tick
    try:
        days = [d for d in trading_days(con) if SC.cas_in_force(d)]
        log.info("%d post-reform session(s) in the vault", len(days))
        for day in days:
            idx = config.TRADABLE[0]
            if SC.is_bse(idx):
                continue
            # v9.9.32: THE 0/7 BUG. This used load_day, whose array is
            # scenario_engine's 09:15->15:30 window (22 500 columns) —
            # constants written before the 2026-08-03 reform. The
            # post-auction window starts at t=22 800 and ends at 23 100, so
            # `t40 >= bidA.shape[1]` was ALWAYS true and every session was
            # rejected. The message blamed the harvester ("process shut at
            # 15:30?"), which was never true: the 2026-08-11 harvester
            # report has updated_utc 13:07Z = 18:37 IST, hours past the
            # close. The tape was captured all along; the LOADER could not
            # see it, so the counter could never leave 0/7 no matter how
            # many sessions passed.
            #
            # simulation.session_paths is DATE-AWARE (session_calendar), so
            # a pre-reform day still ends 15:30 and no window is fabricated.
            try:
                from simulation.session_paths import load_session_paths
                ps = load_session_paths(con, day, idx)
            except Exception as e:                         # noqa: BLE001
                log.warning("  %s: %s", day, e)
                continue
            if ps is None:
                log.info("  %s: no ticks in the vault for this session", day)
                continue
            stok = None
            try:
                _r = con.execute("SELECT token FROM spot_tokens WHERE "
                                 "snap_date=? AND name=?",
                                 (day, idx)).fetchone()
                stok = int(_r[0]) if _r else None
            except Exception:                              # noqa: BLE001
                stok = None
            win = ps.window
            t35 = win.sod_to_t(_hm_to_min("15:35") * 60)
            t40 = win.sod_to_t(_hm_to_min("15:40") * 60)
            k = ps.row(stok) if stok else None
            if t35 < 0 or t40 > win.n or t40 <= t35:
                log.info("  %s: session closes %s — the 15:35-15:40 window "
                         "does not exist on this date (pre-reform)", day,
                         win.close_hm)
                continue
            if k is None:
                # Distinguish "no window" from "no data for the spot". The
                # first is a fact about the date; the second is a plumbing
                # failure, and conflating them cost six sessions of silence.
                log.warning("  %s: session closes %s so the window EXISTS, "
                            "but the spot token %s has no path in the vault "
                            "— this is a data/plumbing problem, not a "
                            "calendar one", day, win.close_hm, stok)
                continue
            # Bind EVERY name the old load_day tuple provided. The first
            # port bound only `bidA`, and the spread loop eleven lines below
            # still referenced `ti`/`askA` — NameError at runtime, after
            # 267s of replay. A partial rename is worse than none: it fails
            # late, in the one branch that only runs on post-reform days.
            ti, bidA, askA = ps.ti, ps.bid, ps.ask
            # COUNT REAL TICKS, NOT CARRIED COPIES.
            # session_paths forward-fills a quote for up to
            # SHADOW_MAX_STALE_S. If the spot feed stops at the auction, the
            # window fills with ~120 IDENTICAL carried samples, sails past a
            # size>=30 floor, and reports a range of EXACTLY 0.00. On
            # 2026-08-14 all seven sessions did precisely that (FRESH=1,
            # usable=121) and the tool then wrote an ADVERSE certificate
            # shutting the window permanently — on an artifact, not a
            # market fact. A wrong certificate is worse than no certificate.
            seg_raw = np.asarray(bidA[k, t35:t40], dtype=float)
            fresh_m = ps.fresh_mask(stok)
            fresh_m = (np.asarray(fresh_m[t35:t40], dtype=bool)
                       if fresh_m is not None
                       else np.isfinite(seg_raw))
            seg = seg_raw[fresh_m & np.isfinite(seg_raw) & (seg_raw > 0)]
            min_fresh = int(getattr(config, "POST_AUCTION_MIN_FRESH", 30))
            if seg.size < min_fresh:
                log.warning("  %s: only %d REAL tick(s) in 15:35-15:40 "
                            "(%d sample(s) survive once carried copies are "
                            "removed). The spot is not disseminated through "
                            "this window on this session — NOT counted, and "
                            "no certificate is issued from it.",
                            day, int(seg.size),
                            int((np.isfinite(seg_raw) & (seg_raw > 0)).sum()))
                degenerate += 1
                continue
            mv = float(seg.max() - seg.min())
            if mv <= 0.0:
                log.warning("  %s: %d real tick(s) but range is EXACTLY 0.00 "
                            "— a disseminated index does not hold one value "
                            "for five minutes. Treating as no data.",
                            day, int(seg.size))
                degenerate += 1
                continue
            moves.append(mv)
            spot_ref.append(float(seg[0]))
            # round-trip spread on the tokens that DID quote in the window
            sp = []
            # TWO-SIDED ONLY. ps.two_sided is the set that had a real bid
            # AND ask; an index spot is a computed level carrying bid=ask=0,
            # so including it would contribute a 0% spread and drag the
            # median toward zero — the exact artifact that made SENSEX look
            # zero-depth in the July audit.
            _quoted = [(tk, kk) for tk, kk in ti.items()
                       if tk in (ps.two_sided or set())]
            for tok, kk in _quoted[:400]:
                b = np.asarray(bidA[kk, t35:t40], float)
                s_ = np.asarray(askA[kk, t35:t40], float)
                m = np.isfinite(b) & np.isfinite(s_) & (b > 0) & (s_ > 0)
                if m.sum() < 10:
                    continue
                sp.append(float(np.median((s_[m] - b[m]) / s_[m])))
            if sp:
                spreads.append(float(np.median(sp)))
            sessions.append(day)
            log.info("  %s: window range %.1f pts | median spread %.2f%%",
                     day, mv, 100 * (spreads[-1] if spreads else float('nan')))
    finally:
        con.close()

    need = int(getattr(config, "POST_AUCTION_MIN_SESSIONS", 7))
    n = len(sessions)
    if n < need:
        log.warning("LEARNING: %d/%d post-auction session(s) captured. The "
                    "window opens by itself once the vault holds %d. "
                    "Nothing changed.", n, need, need)
        g = PA.load()
        g.n_sessions = n
        PA.save(g, {"adverse": False, "learning": True, "sessions": sessions})
        return 0

    geom = PA.fit_geometry(moves, float(np.median(spot_ref)), spreads, n)
    adverse = geom.tp_pct <= 0
    ev = {"adverse": bool(adverse), "sessions": sessions, "n": n,
          "median_move_pts": geom.median_move_pts,
          "p90_move_pts": geom.p90_move_pts,
          "median_spread_pct": geom.median_spread_pct,
          "reason": ("median move cannot net the round-trip spread at 2:1 "
                     "odds" if adverse else "geometry fitted")}
    PA.save(geom, ev)
    try:
        from core.trial_registry import register
        register(family="post_auction", spec_id=f"geom_{n}d",
                 kind="calibration", adverse=bool(adverse),
                 tp_pct=geom.tp_pct, sl_pct=geom.sl_pct)
    except Exception as e:                                 # noqa: BLE001
        log.debug("registry unavailable (%s)", e)

    ok, why = PA.readiness()
    log.info("─" * 70)
    if adverse:
        log.warning("ADVERSE: %d session(s), median move %.1f pts against a "
                    "%.2f%% round-trip spread. The window does not pay for "
                    "itself; entries stay shut. This is the result.",
                    n, geom.median_move_pts, 100 * geom.median_spread_pct)
    else:
        log.warning("POST-AUCTION REGIME OPEN: %s", why)
        log.info("  target %.2f%% | stop %.2f%% | ratchet arms %.2f%% then "
                 "gives back %.2f%% | hold %.0f min | flat %s",
                 100 * geom.tp_pct, 100 * geom.sl_pct,
                 100 * geom.arm_at_pct, 100 * geom.trail_giveback,
                 geom.hold_minutes, geom.flat_hm)
    log.info("certificate → %s", PA.CERT_PATH)
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"geometry": geom.as_dict(), "evidence": ev,
             "ready": ok, "ts": time.time()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())