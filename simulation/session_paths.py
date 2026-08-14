"""
SESSION PATHS — the whole session, and an honest mask over it
==============================================================
simulation.replay_real_day.load_day is the engine's replay feed and is
deliberately frozen (its docstring says so). This module is the STUDY
feed. It differs from load_day in exactly two ways, and both are
corrections to defects that made "over the whole market session" false:

  A. THE WINDOW WAS HARD-CODED TO 15:30. load_day builds its array from
     scenario_engine's T0_SEC/N = 09:15→15:30, constants written before
     the 2026-08-03 reform. Equity derivatives now close 15:40 and
     POST_AUCTION_ENABLED is True, so a fill at 15:36 lands at t0=22860
     in a 22500-column array and is dropped by the bounds check — every
     post-auction trade was invisible, and every trade still open at
     15:30 had its path amputated there. Here the window comes from
     core.session_calendar.session_close_hm(day, index), which is
     DATE-AWARE: a pre-reform day still ends 15:30, so replaying history
     under a 15:40 close never fabricates ten minutes that did not exist.

  B. FORWARD-FILL WAS UNBOUNDED. The harvester unsubscribes any leg
     beyond PRUNE_STEPS of the running ATM (data_harvester_v9:370). Ticks
     stop; load_day then carries the last bid to the close. A dead
     instrument becomes a perfectly flat, perfectly finite price series —
     so a trailing stop can never fire on it, a target can never be hit,
     and MFE/MAE are computed over a path that stopped existing. Nothing
     raises, because np.isfinite() is True the whole way.

     Here every sample carries a FRESH bit. A quote is carried forward
     for at most SHADOW_MAX_STALE_S seconds — the horizon over which a
     stale option quote is still a defensible mark — and after that the
     path is NaN. NaN is the correct value: not "the price did not move",
     but "we stopped looking". Policies must then decide explicitly what
     to do about a dead feed instead of silently trading a corpse.

The right long-run fix for (B) is upstream — pin held tokens so the
harvester never prunes something we are marking (see core.token_pins).
This module is what makes the damage VISIBLE and bounded until every day
in the vault was captured under pinning.
"""
from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from dataclasses import dataclass

import numpy as np

import config
from core import session_calendar as SC

log = logging.getLogger("session_paths")

_EQUITY_OPEN_HM = "09:15"


def _hm_to_sod(hm: str) -> int:
    h, m = str(hm).split(":")[:2]
    return int(h) * 3600 + int(m) * 60


@dataclass
class SessionWindow:
    index: str
    day: str
    t0_sod: int
    n: int
    open_hm: str
    close_hm: str

    def sod_to_t(self, sod: int) -> int:
        return int(sod) - self.t0_sod

    def ts_to_t(self, ts: float) -> int:
        lt = dt.datetime.fromtimestamp(ts)
        return (lt.hour * 3600 + lt.minute * 60 + lt.second) - self.t0_sod

    def t_to_hm(self, t: int) -> str:
        sod = self.t0_sod + int(t)
        return f"{sod // 3600:02d}:{(sod % 3600) // 60:02d}"

    def contains(self, t: int) -> bool:
        return 0 <= int(t) < self.n


def window_for(day: str, index: str) -> SessionWindow:
    """The real tradable window for this index on this date.

    Equity: 09:15 → session_calendar.session_close_hm (15:30 before the
    reform, 15:40 after; BSE follows its own switch).
    Commodity: COMMODITY_SESSION_OPEN → that contract's session_close.
    """
    comms = getattr(config, "COMMODITIES", {}) or {}
    if index in comms:
        open_hm = str(getattr(config, "COMMODITY_SESSION_OPEN", "09:00"))
        close_hm = str(comms[index].get("session_close", "23:55"))
    else:
        open_hm = _EQUITY_OPEN_HM
        try:
            close_hm = SC.session_close_hm(day, index)
        except Exception as e:                             # noqa: BLE001
            log.warning("session_close_hm failed for %s %s (%s) — "
                        "falling back to 15:30", day, index, e)
            close_hm = "15:30"
    t0 = _hm_to_sod(open_hm)
    n = max(_hm_to_sod(close_hm) - t0, 60)
    return SessionWindow(index=index, day=day, t0_sod=t0, n=n,
                         open_hm=open_hm, close_hm=close_hm)


@dataclass
class PathSet:
    """Per-second bid/ask over the session, with a freshness mask."""
    window: SessionWindow
    ti: dict[int, int]                 # token → row index
    bid: np.ndarray                    # (tokens, n) float32, NaN where dead
    ask: np.ndarray
    fresh: np.ndarray                  # (tokens, n) bool — True = real tick
    last_tick_t: dict[int, int]        # token → index of its final real tick
    two_sided: set = None              # tokens that had a real bid AND ask;
    #                                    anything else is a computed level
    #                                    (index spot) and is NOT tradeable

    def row(self, token: int) -> int | None:
        return self.ti.get(int(token))

    def path(self, token: int, t0: int = 0, side: str = "bid"
             ) -> np.ndarray | None:
        """Marks from t0 to session close. NaN once the feed is dead."""
        k = self.row(token)
        if k is None:
            return None
        t0 = max(int(t0), 0)
        if t0 >= self.window.n:
            return None
        arr = self.bid if side == "bid" else self.ask
        return np.asarray(arr[k, t0:], dtype=float)

    def fresh_mask(self, token: int, t0: int = 0) -> np.ndarray | None:
        k = self.row(token)
        if k is None:
            return None
        return np.asarray(self.fresh[k, max(int(t0), 0):], dtype=bool)

    def coverage(self, token: int, t0: int = 0) -> float:
        """Fraction of the post-entry session for which we had a live mark.
        A study must report this: a 0.18 coverage trade cannot support a
        claim about what a hold-to-close policy would have earned."""
        m = self.fresh_mask(token, t0)
        if m is None or m.size == 0:
            return 0.0
        return float(m.mean())

    def died_at(self, token: int) -> int | None:
        """Index after which this token has no real ticks, or None if it
        was quoted to the close."""
        k = self.row(token)
        if k is None:
            return None
        last = self.last_tick_t.get(int(token), -1)
        return None if last >= self.window.n - 1 else last


def load_session_paths(con: sqlite3.Connection, day: str, index: str,
                       tokens: set[int] | None = None,
                       max_stale_s: int | None = None) -> PathSet | None:
    """Build the session path set for `day`.

    `tokens` restricts the array to the instruments a study actually
    needs — a full day is thousands of legs and the dense array is
    tokens × n float32. Passing the traded set keeps this a few MB.
    """
    win = window_for(day, index)
    if max_stale_s is None:
        max_stale_s = int(getattr(config, "SHADOW_MAX_STALE_S", 120))
    # TIMEZONE. `int(epoch) % 86400` is seconds-of-day in UTC, NOT local
    # time. On a UTC host that is accidentally correct and every test
    # passes; on the IST host this actually runs on, a tick at local 09:30
    # carries a 04:00 UTC epoch, every t comes out negative, every row is
    # filtered away and this function returns None for EVERY session — so
    # tools/trade_potential.py reports "no ticks in the vault" all night and
    # never raises. nightly_forge_v9 avoids this with a hard-coded
    # (ts + 19800) % 86400; hard-coding the offset works but breaks the day
    # the host or the exchange moves. Anchoring on the day's own LOCAL
    # midnight is correct on any host, in any zone, without a constant.
    midnight = dt.datetime.combine(dt.date.fromisoformat(day),
                                   dt.time(0, 0)).timestamp()
    lo_ms = int((midnight + win.t0_sod) * 1000)
    hi_ms = int((midnight + win.t0_sod + win.n) * 1000)
    try:
        # Bounded on ts_local_ms instead of date(...,'localtime'): the date()
        # form cannot use an index, so it full-scans a 4.6M-row day (the
        # 2026-08-07 harvest was 4 621 320 rows) on every call.
        rows = con.execute(
            # `ltp` is selected too, and that is not cosmetic: an INDEX SPOT is a
            # computed level, not a quoted instrument, so its rows carry
            # bid=ask=0. The bid/ask filter below therefore dropped the spot
            # entirely, `ti` had no row for it, and every caller asking for
            # the spot got None. post_auction_calibrate then reported
            # "window outside this session (close 15:40) — pre-reform day",
            # which is self-contradictory: a 15:40 close IS post-reform.
            # The window arithmetic was right; the spot was simply missing.
            "SELECT ts_local_ms/1000.0, token, bid, ask, ltp FROM ticks_v9 "
            "WHERE ts_local_ms >= ? AND ts_local_ms < ? ORDER BY ts_ms",
            (lo_ms, hi_ms)).fetchall()
    except sqlite3.Error as e:                             # noqa: BLE001
        log.warning("tick query failed for %s (%s)", day, e)
        return None
    if not rows:
        return None

    want = {int(t) for t in tokens} if tokens else None
    seen: dict[int, list[tuple[int, float, float]]] = {}
    for s, tok, bid, ask, ltp in rows:
        tok = int(tok)
        if want is not None and tok not in want:
            continue
        t = int(round(float(s) - midnight)) - win.t0_sod
        if not (0 <= t < win.n):
            continue
        if bid and ask and bid > 0 and ask > 0:
            seen.setdefault(tok, []).append((t, float(bid), float(ask)))
        elif ltp and float(ltp) > 0:
            # Quote-less instrument (index spot). Both sides carry the last
            # level, which is the honest representation: there is no spread
            # to model because there is nothing to transact against. A
            # caller that needs a TRADEABLE quote must check `two_sided`.
            seen.setdefault(tok, []).append((t, float(ltp), float(ltp)))
    if not seen:
        return None

    two_sided = set()
    for _tok, _rows in seen.items():
        if any(abs(a - b) > 1e-9 for _t, b, a in _rows):
            two_sided.add(_tok)
    ti = {tok: i for i, tok in enumerate(sorted(seen))}
    shape = (len(ti), win.n)
    bidA = np.full(shape, np.nan, np.float32)
    askA = np.full(shape, np.nan, np.float32)
    fresh = np.zeros(shape, dtype=bool)
    last_tick: dict[int, int] = {}

    for tok, samples in seen.items():
        k = ti[tok]
        for t, b, a in samples:
            bidA[k, t] = b
            askA[k, t] = a
            fresh[k, t] = True
        last_tick[tok] = max(t for t, _b, _a in samples)

    # BOUNDED carry. `age` counts seconds since the last real tick; once it
    # passes max_stale_s the mark becomes NaN rather than a flat line.
    for k in range(shape[0]):
        b_row, a_row, f_row = bidA[k], askA[k], fresh[k]
        last_b = last_a = np.nan
        age = 10 ** 9
        for t in range(win.n):
            if f_row[t]:
                last_b, last_a, age = b_row[t], a_row[t], 0
                continue
            age += 1
            if age <= max_stale_s and np.isfinite(last_b):
                b_row[t] = last_b
                a_row[t] = last_a
            else:
                b_row[t] = np.nan
                a_row[t] = np.nan

    return PathSet(window=win, ti=ti, bid=bidA, ask=askA, fresh=fresh,
                   last_tick_t=last_tick, two_sided=two_sided)


def traded_tokens(trades) -> set[int]:
    """Every token any of these TradeRecords touches."""
    out: set[int] = set()
    for t in trades:
        for leg in getattr(t, "legs", []) or []:
            if getattr(leg, "token", 0):
                out.add(int(leg.token))
    return out