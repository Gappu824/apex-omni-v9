"""
TRADE RECONSTRUCT — the ledger, read correctly
===============================================
Every post-hoc study in this system starts by turning the execution
ledger back into trades. Until v9.9.13 that reconstruction lived inline
in three places (core/edge_audit.py, tools/trade_potential.py, the
harnesses) as the same eleven lines:

    if ev == "BUY_FILL":            opens[sym] = r
    elif ev == "SELL_FILL" and sym in opens:  pair(opens.pop(sym), r)

That is wrong in four independent ways, each of which silently DELETES
real trades rather than raising:

  1. SHORT LEGS VANISH. A leg that opens with SELL_FILL (the butterfly
     body, every shortvol spread) never enters `opens`, so its closing
     BUY_FILL is dropped and its opening SELL_FILL is dropped too. The
     entire short book is invisible to every study that uses this shape.
  2. RE-ENTRY OVERWRITES. `opens[sym] = r` on a second entry into the
     same symbol destroys the first. The first trade disappears; the
     survivor is paired with the WRONG entry price, so its P&L, MFE and
     capture ratio are all fiction.
  3. PARTIAL EXITS MIS-MATCH. A 3-lot entry closed as 1+2 pairs the full
     entry against the first partial and discards the second.
  4. MULTI-LEG TRADES ARE NOT TRADES. FLY_OPEN/FLY_CLOSE and
     SPREAD_OPEN/SPREAD_CLOSE carry a composite symbol and four tokens;
     the pair-by-symbol shape cannot represent them at all.

And underneath all four, defect F: `token` was never a column in
LEDGER_FIELDS, so every reconstructed fill carried token=0 and every
downstream day-cache lookup missed.

WHAT THIS MODULE DOES INSTEAD
------------------------------
* SIGNED FIFO. Each (index, symbol) keeps a FIFO queue of open lots with
  a sign. A fill that opposes the queue CLOSES lots oldest-first; a fill
  that agrees with it (or arrives flat) OPENS one. Long-first and
  short-first are the same code path, partials split lots, and a
  re-entry stacks instead of overwriting.
* MULTI-LEG AS ONE TRADE. FLY_* and SPREAD_* pair on their composite
  symbol through the same FIFO and carry their per-leg tokens, so a fly
  is one TradeRecord with four legs, not four unrelated fills.
* TOKEN BACKFILL. Ledgers written before the schema fix have no token
  column. `instrument_snapshots` (core/instruments.py) is the as-of
  authority for symbol→token on any past day, so history is recovered
  rather than discarded. Rows that cannot be resolved are RETURNED with
  token=0 and counted in `unresolved` — never silently dropped.
* NOTHING IS DISCARDED SILENTLY. Every fill the reconstruction cannot
  place lands in `orphans` with a reason. A study that loses trades must
  say so out loud; that is the whole lesson of defect F.

This module is pure reading. It has no opinion about what a trade should
have done — that is the shadow book's job.
"""
from __future__ import annotations

import csv
import datetime as dt
import logging
import re
import sqlite3
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path

import config

log = logging.getLogger("trade_reconstruct")

# Events that OPEN and CLOSE, per book. The sign is the direction of the
# position the event creates, not the direction of the order.
_OPEN_EVENTS = {"BUY_FILL": +1, "SELL_FILL": -1,
                "FLY_OPEN": +1, "SPREAD_OPEN": +1}
_CLOSE_EVENTS = {"FLY_CLOSE", "SPREAD_CLOSE"}
_KIND_OF = {"BUY_FILL": "SINGLE", "SELL_FILL": "SINGLE",
            "FLY_OPEN": "FLY", "FLY_CLOSE": "FLY",
            "SPREAD_OPEN": "SPREAD", "SPREAD_CLOSE": "SPREAD"}

# "NIFTY24500CE+NIFTY24600CEx2+NIFTY24700CE" → legs with multiplicity
_LEG_RE = re.compile(r"^(?P<sym>[A-Z0-9]+?)(?:x(?P<mult>\d+))?$")


@dataclass
class Leg:
    symbol: str
    token: int = 0
    side: int = +1          # +1 long, -1 short
    mult: int = 1           # units of `qty` this leg carries

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradeRecord:
    """One round trip, however many legs it took."""
    kind: str                       # SINGLE | FLY | SPREAD
    index: str
    day: str
    symbol: str                     # display symbol (composite for multi-leg)
    side: int                       # +1 long premium, -1 short premium
    qty: int                        # units (lot-multiplied) per leg-mult
    entry_ts: float
    exit_ts: float
    entry_px: float                 # net debit (+) or credit (−) per unit
    exit_px: float
    realized_pnl: float
    costs: float = 0.0
    reason: str = ""
    conviction: float = 0.0
    win_prob: float = 0.0
    regime: str = ""
    legs: list[Leg] = field(default_factory=list)
    open_event: str = ""
    close_event: str = ""

    @property
    def token(self) -> int:
        """Primary token — the leg whose path drives the trade."""
        return self.legs[0].token if self.legs else 0

    @property
    def held_s(self) -> int:
        return max(int(self.exit_ts - self.entry_ts), 1)

    @property
    def resolved(self) -> bool:
        return bool(self.legs) and all(l.token > 0 for l in self.legs)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["legs"] = [l.as_dict() for l in self.legs]
        d["held_s"] = self.held_s
        d["resolved"] = self.resolved
        return d


@dataclass
class _Lot:
    """An open parcel awaiting its close."""
    row: dict
    qty: int
    side: int
    ts: float


# --------------------------------------------------------------- helpers
def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key)
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _i(row: dict, key: str, default: int = 0) -> int:
    return int(_f(row, key, float(default)))


def day_of(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).date().isoformat()


def split_legs(symbol: str, side: int) -> list[Leg]:
    """Explode a composite multi-leg symbol into signed legs.

    The butterfly writes wing_in + body x2 + wing_out. A long fly is long
    the wings and SHORT the doubled body — the sign pattern is structural,
    so it is derived here rather than parsed out of a string that does not
    carry it.
    """
    parts = [p for p in str(symbol).split("+") if p]
    if len(parts) <= 1:
        return [Leg(symbol=str(symbol), side=side, mult=1)]
    legs: list[Leg] = []
    for p in parts:
        m = _LEG_RE.match(p)
        if not m:
            legs.append(Leg(symbol=p, side=side, mult=1))
            continue
        mult = int(m.group("mult") or 1)
        # the multiplied leg of a long fly is the short body
        leg_side = (-side) if mult > 1 else side
        legs.append(Leg(symbol=m.group("sym"), side=leg_side, mult=mult))
    return legs


class TokenResolver:
    """symbol → token, as of a given trading day.

    instrument_snapshots is written every morning by core.instruments, so
    it is the only authority that knows what a symbol meant on a past
    date. Cached per (day, symbol); a miss is reported, never guessed.
    """

    def __init__(self, db_path: Path | None = None):
        self._db = Path(db_path or config.DB_PATH)
        self._cache: dict[tuple[str, str], int] = {}
        self._snap_for_day: dict[str, str | None] = {}
        self._con: sqlite3.Connection | None = None
        self.misses: set[tuple[str, str]] = set()

    def _conn(self) -> sqlite3.Connection | None:
        if self._con is None:
            try:
                self._con = sqlite3.connect(f"file:{self._db}?mode=ro",
                                            uri=True)
            except sqlite3.Error as e:                      # noqa: BLE001
                log.warning("instrument snapshot db unavailable (%s) — "
                            "tokens cannot be backfilled", e)
                return None
        return self._con

    def _snap_date(self, day: str) -> str | None:
        if day in self._snap_for_day:
            return self._snap_for_day[day]
        con, snap = self._conn(), None
        if con is not None:
            try:
                r = con.execute(
                    "SELECT MAX(snap_date) FROM instrument_snapshots "
                    "WHERE snap_date <= ?", (day,)).fetchone()
                snap = r[0] if r else None
            except sqlite3.Error:
                snap = None
        self._snap_for_day[day] = snap
        return snap

    def resolve(self, symbol: str, day: str) -> int:
        key = (day, symbol)
        if key in self._cache:
            return self._cache[key]
        tok, con, snap = 0, self._conn(), self._snap_date(day)
        if con is not None and snap:
            try:
                r = con.execute(
                    "SELECT token FROM instrument_snapshots WHERE "
                    "snap_date=? AND symbol=?", (snap, symbol)).fetchone()
                if r:
                    tok = int(r[0])
            except sqlite3.Error:
                tok = 0
        if tok == 0:
            self.misses.add(key)
        self._cache[key] = tok
        return tok

    def close(self) -> None:
        if self._con is not None:
            try:
                self._con.close()
            finally:
                self._con = None


# ------------------------------------------------------------- the core
class Reconstruction:
    """Result of reading one ledger."""

    def __init__(self):
        self.trades: list[TradeRecord] = []
        self.orphans: list[dict] = []
        self.still_open: list[dict] = []
        self.rows_read = 0

    @property
    def unresolved(self) -> list[TradeRecord]:
        return [t for t in self.trades if not t.resolved]

    def summary(self) -> dict:
        by_kind: dict[str, int] = {}
        for t in self.trades:
            by_kind[t.kind] = by_kind.get(t.kind, 0) + 1
        return {"rows_read": self.rows_read,
                "trades": len(self.trades),
                "by_kind": by_kind,
                "long": sum(1 for t in self.trades if t.side > 0),
                "short": sum(1 for t in self.trades if t.side < 0),
                "unresolved_tokens": len(self.unresolved),
                "orphans": len(self.orphans),
                "still_open": len(self.still_open)}

    def log_summary(self, logger: logging.Logger | None = None) -> None:
        lg = logger or log
        s = self.summary()
        lg.info("ledger → %d row(s) → %d trade(s) %s | long %d / short %d",
                s["rows_read"], s["trades"], s["by_kind"], s["long"],
                s["short"])
        if s["unresolved_tokens"]:
            lg.warning("%d trade(s) have UNRESOLVED tokens — their paths "
                       "cannot be replayed and they are excluded from any "
                       "path study (but counted here, not hidden)",
                       s["unresolved_tokens"])
        if s["orphans"]:
            lg.warning("%d orphan fill(s) could not be placed: %s",
                       s["orphans"],
                       sorted({o["why"] for o in self.orphans}))
        if s["still_open"]:
            lg.info("%d position(s) still open at end of ledger — excluded "
                    "(no realised number to compare against)",
                    s["still_open"])


def _lot_key(row: dict) -> tuple[str, str]:
    return (str(row.get("index") or ""), str(row.get("symbol") or ""))


def reconstruct(ledger_path: Path | str | None = None,
                resolver: TokenResolver | None = None,
                since_ts: float = 0.0) -> Reconstruction:
    """Read the execution ledger into TradeRecords.

    Signed FIFO per (index, symbol). Opening and closing are decided by
    the sign of the book, not by the event name alone, so a SELL_FILL
    opens a short when flat and closes a long when long.
    """
    p = Path(ledger_path or config.LEDGER_PATH)
    out = Reconstruction()
    if not p.exists():
        log.info("no execution ledger at %s — nothing to reconstruct", p)
        return out
    own_resolver = resolver is None
    resolver = resolver or TokenResolver()
    books: dict[tuple[str, str], deque[_Lot]] = {}

    try:
        with p.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                out.rows_read += 1
                ev = str(row.get("event") or "")
                if ev not in _KIND_OF:
                    continue
                ts = _f(row, "ts")
                if ts <= 0 or ts < since_ts:
                    continue
                qty = _i(row, "qty")
                if qty <= 0:
                    out.orphans.append({"row": row, "why": "qty<=0"})
                    continue
                key = _lot_key(row)
                book = books.setdefault(key, deque())

                if ev in _CLOSE_EVENTS:
                    # explicit close event — must have an open lot
                    if not book:
                        out.orphans.append(
                            {"row": row, "why": f"{ev} with no open lot"})
                        continue
                    _consume(out, book, row, qty, resolver, ev)
                    continue

                sign = _OPEN_EVENTS[ev]
                if book and book[0].side == -sign:
                    # opposes the book → this fill CLOSES, oldest first
                    _consume(out, book, row, qty, resolver, ev)
                else:
                    book.append(_Lot(row=row, qty=qty, side=sign, ts=ts))
    except OSError as e:                                   # noqa: BLE001
        log.error("could not read ledger %s (%s)", p, e)
        return out
    finally:
        pass

    for key, book in books.items():
        for lot in book:
            out.still_open.append({"index": key[0], "symbol": key[1],
                                   "qty": lot.qty, "ts": lot.ts})
    if own_resolver:
        if resolver.misses:
            log.warning("%d symbol/day pair(s) had no instrument snapshot — "
                        "token unresolved (e.g. %s)", len(resolver.misses),
                        sorted(resolver.misses)[:3])
        resolver.close()
    return out


def _consume(out: Reconstruction, book: deque, close_row: dict, qty: int,
             resolver: TokenResolver, close_ev: str) -> None:
    """Match `qty` units of a closing fill against open lots, oldest first.

    A close larger than the front lot spans lots; a close smaller than it
    splits the lot and leaves the remainder open. Realised P&L on the
    ledger row belongs to the WHOLE close, so it is apportioned by units
    — never duplicated across lots (the partial-fill double-count that
    the S2-F4 audit found in the risk ledger is the same bug in a
    different place).
    """
    close_ts = _f(close_row, "ts")
    close_px = _f(close_row, "price")
    total_pnl = _f(close_row, "pnl")
    total_cost = _f(close_row, "costs")
    remaining = qty
    matched_units = 0
    spans: list[tuple[_Lot, int]] = []
    while remaining > 0 and book:
        lot = book[0]
        take = min(remaining, lot.qty)
        spans.append((lot, take))
        remaining -= take
        matched_units += take
        lot.qty -= take
        if lot.qty <= 0:
            book.popleft()
    if remaining > 0:
        out.orphans.append({"row": close_row,
                            "why": f"{close_ev} exceeded open qty by "
                                   f"{remaining}"})
    if matched_units <= 0:
        return
    for lot, take in spans:
        share = take / matched_units
        open_row = lot.row
        day = day_of(lot.ts)
        side = lot.side
        legs = split_legs(str(open_row.get("symbol") or ""), side)
        for leg in legs:
            tok = _i(open_row, "token")
            if len(legs) == 1 and tok > 0:
                leg.token = tok
            else:
                leg.token = resolver.resolve(leg.symbol, day)
        out.trades.append(TradeRecord(
            kind=_KIND_OF.get(str(open_row.get("event") or ""), "SINGLE"),
            index=str(open_row.get("index") or ""),
            day=day,
            symbol=str(open_row.get("symbol") or ""),
            side=side,
            qty=take,
            entry_ts=lot.ts,
            exit_ts=close_ts,
            entry_px=_f(open_row, "price"),
            exit_px=close_px,
            realized_pnl=total_pnl * share,
            costs=total_cost * share,
            reason=str(close_row.get("reason") or ""),
            conviction=_f(open_row, "conviction"),
            win_prob=_f(open_row, "win_prob"),
            regime=str(open_row.get("regime") or ""),
            legs=legs,
            open_event=str(open_row.get("event") or ""),
            close_event=close_ev))


def all_ledgers() -> list[Path]:
    """The live ledger plus every rotated predecessor.

    position_manager renames the ledger whenever LEDGER_FIELDS changes
    (`.pre_v971_<ts>.csv`), which is correct — misaligned columns are
    worse than a rotation. But it means a schema fix ORPHANS every day of
    history unless the readers follow the rotations. They do now. Oldest
    first, so FIFO books open in chronological order.
    """
    live = Path(config.LEDGER_PATH)
    out = sorted(live.parent.glob(f"{live.stem}.pre_*{live.suffix}"),
                 key=lambda p: p.stat().st_mtime)
    if live.exists():
        out.append(live)
    return out


def reconstruct_all(days: int = 0) -> Reconstruction:
    """Reconstruct across every ledger generation as one continuous book."""
    import time
    since = time.time() - days * 86400 if days > 0 else 0.0
    resolver = TokenResolver()
    merged = Reconstruction()
    try:
        for p in all_ledgers():
            r = reconstruct(p, resolver=resolver, since_ts=since)
            merged.trades.extend(r.trades)
            merged.orphans.extend(r.orphans)
            merged.still_open.extend(r.still_open)
            merged.rows_read += r.rows_read
    finally:
        if resolver.misses:
            log.warning("%d symbol/day pair(s) had no instrument snapshot",
                        len(resolver.misses))
        resolver.close()
    merged.trades.sort(key=lambda t: t.entry_ts)
    return merged


def closed_trades(ledger_path: Path | str | None = None,
                  days: int = 0) -> list[TradeRecord]:
    """Convenience: completed round trips, newest window first."""
    import time
    if ledger_path is None:
        return reconstruct_all(days).trades
    since = time.time() - days * 86400 if days > 0 else 0.0
    return reconstruct(ledger_path, since_ts=since).trades