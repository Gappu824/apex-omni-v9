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
    try:
        days = [d for d in trading_days(con) if SC.cas_in_force(d)]
        log.info("%d post-reform session(s) in the vault", len(days))
        for day in days:
            idx = config.TRADABLE[0]
            if SC.is_bse(idx):
                continue
            try:
                loaded = load_day(con, day, idx)
            except Exception as e:                         # noqa: BLE001
                log.warning("  %s: %s", day, e)
                continue
            if not loaded:
                continue
            stok, by_sec, ti, bidA, askA = loaded
            base = min(by_sec) if by_sec else 0
            day0 = dt.datetime.combine(dt.date.fromisoformat(day),
                                       dt.time(0, 0)).timestamp()
            t35 = int(day0 + _hm_to_min("15:35") * 60 - base)
            t40 = int(day0 + _hm_to_min("15:40") * 60 - base)
            k = ti.get(stok)
            if k is None or t35 < 0 or t40 >= bidA.shape[1] or t40 <= t35:
                log.info("  %s: window not harvested (process shut at "
                         "15:30?) — this session cannot contribute", day)
                continue
            seg = np.asarray(bidA[k, t35:t40], dtype=float)
            seg = seg[np.isfinite(seg) & (seg > 0)]
            if seg.size < 30:
                log.info("  %s: only %d tick(s) in the window — skipped",
                         day, seg.size)
                continue
            mv = float(seg.max() - seg.min())
            moves.append(mv)
            spot_ref.append(float(seg[0]))
            # round-trip spread on the tokens that DID quote in the window
            sp = []
            for tok, kk in list(ti.items())[:400]:
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