"""
OVERNIGHT GAP — what the night did, measured before it is believed
==================================================================
THE DATA PROBLEM, STATED FIRST
-------------------------------
GIFT NIFTY trades on NSE IX — a different exchange from NSE cash and NFO.
Kite Connect's instrument dump for this account carries 16 514 option rows
across NFO/BFO/MCX and NOTHING from NSE IX; a grep for gift/sgx/nseix
across the entire repo returns nothing. So a module that simply assumed a
GIFT feed would have shipped a feature column of zeros that quietly
degraded the model, which is worse than not shipping it.

This module therefore separates two things that are usually conflated:

  A. THE OVERNIGHT INFORMATION, which is observable TODAY from the vault:
     the realised open gap, its size against the instrument's own ATR,
     and how the first minutes treat it. For a book that commits at 09:50
     (core/day_plan.py) the 09:15 gap is ALREADY REALISED and fully
     available — no offshore feed is needed to use it.

  B. THE PRE-OPEN FORECAST, which is what GIFT NIFTY actually adds: a
     price for NIFTY risk while the domestic market is shut. That is only
     worth wiring once a token resolves, and `gift_token()` is the hook.
     Until it resolves, `gift_gap_pct` is absent — not zero. Absent means
     "no column"; zero means "the night was flat", and a model cannot
     tell those apart if the feature lies.

WHY THE GAP IS WORTH MEASURING AT ALL
--------------------------------------
Two competing regularities, both documented and both regime-dependent:
gaps that FILL (mean reversion toward the previous close, the dominant
intraday tendency in index products) and gaps that RUN (continuation on
genuine overnight repricing). Which one dominates is exactly the sort of
thing that cannot be assumed and can be measured — and it interacts
directly with core/range_regime.py, since a filling gap IS a
mean-reverting tape.

THE DISCIPLINE
--------------
Identical to core/payoff_target.py, and for the same reason: the July
diagnostic chain proved that pooled IC t-stats on 1 Hz data are
autocorrelation-inflated artifacts. So per-day Spearman IC, day-clustered
significance, an explicit MDE, and Benjamini-Hochberg across the feature
set. `measure()` can return "no feature clears" and that is the finding,
not a failure — a gap feature that does not predict is a column that adds
variance to the meta and nothing else.
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from dataclasses import dataclass, asdict

import numpy as np

import config

log = logging.getLogger("overnight_gap")

FEATURES = ("gap_pct", "gap_atr", "gap_abs_atr", "first15_follow",
            "gift_gap_pct")


@dataclass
class GapRow:
    day: str
    index: str
    prev_close: float
    open_px: float
    gap_pct: float          # signed, vs previous close
    gap_atr: float          # signed, in units of the instrument's own ATR
    gap_abs_atr: float      # magnitude only
    first15_follow: float   # did the first 15 min extend the gap (+) or
    #                         fade it (-)? in ATR units
    gift_gap_pct: float | None = None   # None = NO FEED, never 0.0

    def as_dict(self) -> dict:
        return asdict(self)

    def features(self) -> dict:
        d = {"gap_pct": self.gap_pct, "gap_atr": self.gap_atr,
             "gap_abs_atr": self.gap_abs_atr,
             "first15_follow": self.first15_follow}
        if self.gift_gap_pct is not None:
            d["gift_gap_pct"] = self.gift_gap_pct
        return d


def gift_token(con: sqlite3.Connection, day: str) -> int | None:
    """Resolve a GIFT NIFTY token from the instrument snapshot, if the
    broker ever exposes one.

    Returns None today, and None is the CORRECT answer: this account's
    dump is NFO/BFO/MCX only. The hook exists so that enabling the feed
    is a data change rather than a code change, and so the absence is
    explicit in the logs instead of silently becoming a zero column.
    """
    try:
        r = con.execute(
            "SELECT token FROM instrument_snapshots WHERE snap_date<=? AND "
            "(symbol LIKE 'GIFTNIFTY%' OR symbol LIKE 'GIFT NIFTY%' OR "
            " name LIKE 'GIFTNIFTY%') ORDER BY snap_date DESC LIMIT 1",
            (day,)).fetchone()
        return int(r[0]) if r else None
    except sqlite3.Error:
        return None


def build_rows(con: sqlite3.Connection, days: list[str],
               index: str = "NIFTY") -> tuple[list[GapRow], dict]:
    """One row per session: the night, as the 09:50 book would see it."""
    from simulation.session_paths import window_for

    stats = {"days": len(days), "kept": 0, "no_prev": 0, "no_open": 0,
             "gift": 0}
    rows: list[GapRow] = []
    prev_close = None
    prev_ranges: list[float] = []

    for day in sorted(days):
        try:
            win = window_for(day, index)
            mid = dt.datetime.combine(dt.date.fromisoformat(day),
                                      dt.time(0, 0)).timestamp()
            tok = con.execute(
                "SELECT token FROM spot_tokens WHERE snap_date=? AND name=?",
                (day, index)).fetchone()
            if not tok:
                continue
            lo = int((mid + win.t0_sod) * 1000)
            hi = int((mid + win.t0_sod + win.n) * 1000)
            q = con.execute(
                "SELECT ts_local_ms/1000.0, ltp FROM ticks_v9 WHERE token=? "
                "AND ts_ms>=? AND ts_ms<? ORDER BY ts_ms", (tok[0], lo, hi)
            ).fetchall()
            if len(q) < 120:
                continue
            ts = np.array([r[0] for r in q], float)
            px = np.array([r[1] for r in q], float)
            ok = np.isfinite(px) & (px > 0)
            ts, px = ts[ok], px[ok]
            if px.size < 120:
                continue

            day_close = float(px[-1])
            day_range = float(px.max() - px.min())
            if prev_close is None or not prev_ranges:
                prev_close, prev_ranges = day_close, [day_range]
                stats["no_prev"] += 1
                continue

            atr = float(np.mean(prev_ranges[-14:])) or 1.0
            open_px = float(px[0])
            gap = open_px - prev_close
            t0 = ts[0]
            m15 = px[ts <= t0 + 900]
            follow = ((float(m15[-1]) - open_px) / atr
                      if m15.size > 2 else 0.0)
            # continuation is same-signed as the gap; fading is opposite
            follow *= (1.0 if gap >= 0 else -1.0)

            g = None
            gt = gift_token(con, day)
            if gt:
                gq = con.execute(
                    "SELECT ltp FROM ticks_v9 WHERE token=? AND ts_ms<? "
                    "ORDER BY ts_ms DESC LIMIT 1", (gt, lo)).fetchone()
                if gq and gq[0]:
                    g = float(gq[0]) / prev_close - 1.0
                    stats["gift"] += 1

            rows.append(GapRow(day=day, index=index, prev_close=prev_close,
                               open_px=open_px,
                               gap_pct=gap / prev_close,
                               gap_atr=gap / atr,
                               gap_abs_atr=abs(gap) / atr,
                               first15_follow=follow, gift_gap_pct=g))
            stats["kept"] += 1
            prev_close = day_close
            prev_ranges.append(day_range)
        except Exception as e:                             # noqa: BLE001
            log.debug("gap row failed for %s (%s)", day, e)
    if stats["gift"] == 0:
        log.info("no GIFT NIFTY token resolved — `gift_gap_pct` is ABSENT, "
                 "not zero. This account's dump is NFO/BFO/MCX only; the "
                 "hook is live the day a token appears. A zero column would "
                 "tell the model the night was flat.")
    return rows, stats


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 4 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def measure(rows: list[GapRow], outcome: dict) -> dict:
    """Does any gap feature predict the session outcome?

    `outcome` maps day -> realised R (or ₹). One observation per SESSION —
    there is exactly one gap per day, so the day IS the sample and there is
    no clustering to correct for. That also means the sample is small by
    construction: 38 sessions is 38 points, and the MDE will say so.
    """
    from core import capability_ladder as CL

    days = [r.day for r in rows if r.day in outcome]
    if len(days) < 12:
        return {"ok": False, "predictable": False,
                "reason": f"{len(days)} paired session(s) — one gap per day "
                          f"means the day is the sample, so this needs "
                          f"sessions, not ticks",
                "n_days": len(days)}
    y = np.array([float(outcome[d]) for d in days])
    feats = []
    for name in FEATURES:
        vals = [getattr(r, name, None) for r in rows if r.day in outcome]
        if any(v is None for v in vals):
            continue                       # absent feature, never imputed
        x = np.array(vals, float)
        if not np.isfinite(x).all():
            continue
        ic = _spearman(x, y)
        # leave-one-session-out for a distribution, since n IS the day count
        loo = [ _spearman(np.delete(x, i), np.delete(y, i))
                for i in range(len(y)) ]
        st = CL.paired_test({days[i]: float(loo[i]) for i in range(len(loo))})
        feats.append({"feature": name, "ic": round(float(ic), 4),
                      "loo_mean": st["mean"], "p": st.get("p", 1.0),
                      "mde": st.get("mde", float("nan")),
                      "n_days": len(days)})
    if not feats:
        return {"ok": False, "predictable": False,
                "reason": "no complete feature column", "n_days": len(days)}
    rej, adj = CL.benjamini_hochberg(
        [f["p"] for f in feats], float(getattr(config, "GAP_FDR_Q", 0.10)))
    for f, r_, q_ in zip(feats, rej, adj):
        f["p_adj_bh"] = round(float(q_), 4)
        f["significant"] = bool(r_)
        f["above_mde"] = bool(abs(f["loo_mean"]) >
                              float(f.get("mde", float("inf"))))
    win = [f for f in feats if f["significant"] and f["above_mde"]]
    return {"ok": True, "predictable": bool(win), "n_days": len(days),
            "features": sorted(feats, key=lambda z: -abs(z["ic"])),
            "winners": [f["feature"] for f in win],
            "config_hash": config.CONFIG_HASH}


def report(m: dict, logger=None) -> None:
    lg = logger or log
    if not m.get("ok"):
        lg.info("overnight gap: %s", m.get("reason"))
        return
    lg.info("overnight gap: %d paired session(s)", m["n_days"])
    for f in m["features"]:
        lg.info("  %-16s IC %+.4f | p(BH) %.3f | MDE %.4f %s",
                f["feature"], f["ic"], f.get("p_adj_bh", 1.0),
                f.get("mde", float("nan")),
                "<-- clears" if f.get("significant") and f.get("above_mde")
                else "")
    if not m["predictable"]:
        lg.info("NO gap feature clears BH + MDE. One gap per session means "
                "38 days is 38 points; this is a small-sample verdict, not "
                "proof of no effect — the MDE column says what could have "
                "been resolved.")