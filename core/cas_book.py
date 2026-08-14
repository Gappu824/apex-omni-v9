"""
CAS BOOK — the closing auction is a different market, so it gets its own book
=============================================================================
WHY SEPARATE AND NOT JUST ANOTHER ENTRY
----------------------------------------
CAS pre-print entries already exist (apex_main_v9:1948 calls `_attempt`
with a "CAS-PREPRINT" tag), but they run through the SAME PositionManager,
the same single slot and the same risk ledger as the continuous session.
Three things break because of that:

  1. THE SLOT. MAX_CONCURRENT_POSITIONS=1 is shared, so a day-session
     position still open at 15:15 makes the auction unreachable. The two
     books compete for one slot even though they trade different
     microstructures at non-overlapping times.
  2. THE THESIS. Under core/day_plan.py the session commits ONE thesis. A
     CAS entry is not a continuation of it — the auction prices a
     different thing (the closing print) on a different mechanism — so it
     must not consume, inherit, or be blocked by the day's commitment.
  3. THE ATTRIBUTION. Shared P&L makes the auction's edge unmeasurable.
     If CAS and the day session pay into one ledger, no study can ever say
     whether the auction was worth trading, and CAS_MIN_SESSIONS exists
     precisely to answer that question.

WHAT MAKES CAS A DIFFERENT MARKET, CONCRETELY
----------------------------------------------
core/session_calendar.CAS_PHASES: 15:15 REFERENCE, 15:20 ENTRY, 15:25
LIMIT_ONLY, 15:30 MATCHING, 15:35 POST_AUCTION. Through the first four the
disseminated index is INDICATIVE — it moves as the order book builds and
jumps again at the random closure — so conviction, GEX, the flip and the
regime label are all reading auction mechanics rather than a traded price.
That is why the continuous gate blocks entries there, and why this book
does not simply re-run the same gates a few minutes later.

Only POST_AUCTION (15:35-15:40) is continuously traded, and the vault has
0 of 7 sessions of it captured — so it is not tradable yet by the system's
own rule, and this book will say so rather than guess.

HOW THE CAPITAL IS SPLIT
------------------------
CAS_CAPITAL_FRAC of TRADING_CAPITAL, carved out and NOT shared. A book
that can borrow from the day session's capital is not a separate book; it
is the same book with a different label, and the first bad auction would
be paid for out of tomorrow's day-session sizing.

EVERYTHING HERE IS OFF UNTIL EARNED
------------------------------------
CAS_BOOK_ENABLED defaults False. Even enabled, entries require the
existing cas_capture evidence gate (CAS_MIN_SESSIONS tapes proving the
basis forecasts the print) — this module does not create a new authority
to trade, it creates a separate container for trades the existing gates
already permit.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict

import config

log = logging.getLogger("cas_book")

TRADABLE_PHASE = "POST_AUCTION"


@dataclass
class CasPosition:
    symbol: str
    token: int
    index: str
    qty: int
    entry_px: float
    entry_ts: float
    phase: str
    reason: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CasBook:
    """One per process. Its own capital, its own ledger, its own slot."""
    day: str = ""
    pos: CasPosition | None = None
    closed: list = field(default_factory=list)
    realized: float = 0.0
    entries: int = 0
    blocked: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.day:
            self.day = dt.date.today().isoformat()

    # ------------------------------------------------------------ capital
    @staticmethod
    def capital() -> float:
        """The carved-out slice. Never borrows from the day session."""
        frac = float(getattr(config, "CAS_CAPITAL_FRAC", 0.25))
        frac = max(min(frac, 0.5), 0.0)
        return float(getattr(config, "TRADING_CAPITAL", 60000.0)) * frac

    def deployable(self) -> float:
        """Capital minus what this book has already lost today. A separate
        book that ignores its own drawdown is a separate book only on the
        way up."""
        return max(self.capital() + min(self.realized, 0.0), 0.0)

    # -------------------------------------------------------------- gates
    def may_enter(self, now: float | None = None, index: str = "NIFTY"
                  ) -> tuple[bool, str]:
        """Total. Returns (ok, reason) and never raises into the loop."""
        try:
            if not bool(getattr(config, "CAS_BOOK_ENABLED", False)):
                return False, "cas book disabled"
            if self.pos is not None:
                return False, f"cas book: already in {self.pos.symbol}"
            if self.entries >= int(getattr(config, "CAS_MAX_ENTRIES", 1)):
                return False, ("cas book: session entry cap reached — the "
                               "auction is one event, not a market to trade "
                               "repeatedly")
            from core import session_calendar as SC
            ph = SC.cas_phase(dt.datetime.fromtimestamp(
                now if now is not None else time.time()), index=index)
            if ph != TRADABLE_PHASE:
                return False, (f"cas book: phase {ph} — the disseminated "
                               f"index is INDICATIVE here, not a traded "
                               f"price; only {TRADABLE_PHASE} matches "
                               f"continuously")
            if not bool(getattr(config, "POST_AUCTION_ENTRIES", False)):
                return False, ("cas book: POST_AUCTION_ENTRIES is False — "
                               "the window is captured for evidence before "
                               "it is traded")
            if self.deployable() <= 0:
                return False, ("cas book: its own capital slice is exhausted "
                               "— it does not borrow from the day session")
            return True, ""
        except Exception as e:                             # noqa: BLE001
            log.warning("cas gate failed (%s) — refusing, since an auction "
                        "entry on an unknown phase is worse than none", e)
            return False, f"cas book: gate error ({e})"

    def record_block(self, why: str) -> None:
        k = str(why).split("—")[0].strip()[:56] or "blocked"
        self.blocked[k] = self.blocked.get(k, 0) + 1

    # ------------------------------------------------------------ actions
    def enter(self, symbol: str, token: int, index: str, qty: int,
              px: float, phase: str, reason: str = "",
              ts: float | None = None) -> bool:
        try:
            cost = float(px) * int(qty)
            if cost > self.deployable():
                self.record_block("cas book: unaffordable in its own slice")
                return False
            self.pos = CasPosition(symbol=symbol, token=int(token),
                                   index=index, qty=int(qty),
                                   entry_px=float(px),
                                   entry_ts=float(ts or time.time()),
                                   phase=phase, reason=reason)
            self.entries += 1
            log.info("CAS ENTER %s %s x%d @ %.2f (%s) | own capital Rs%.0f",
                     index, symbol, qty, px, phase, self.deployable())
            self._persist()
            return True
        except Exception as e:                             # noqa: BLE001
            log.warning("cas enter failed (%s)", e)
            return False

    def must_exit(self, now: float | None = None) -> tuple[bool, str]:
        """The auction ends; the book is flat. There is no overnight in
        this container and no hand-off to the day session."""
        if self.pos is None:
            return False, ""
        try:
            hm = str(getattr(config, "POST_AUCTION_FLAT_HM", "15:39"))
            h, m = (int(x) for x in hm.split(":")[:2])
            lt = dt.datetime.fromtimestamp(now if now is not None
                                           else time.time())
            if lt.hour * 60 + lt.minute >= h * 60 + m:
                return True, f"CAS_FLAT {hm}"
        except Exception:                                  # noqa: BLE001
            return True, "CAS_FLAT (clock error — flattening)"
        return False, ""

    def exit(self, px: float, reason: str, costs: float = 0.0,
             ts: float | None = None) -> float:
        if self.pos is None:
            return 0.0
        p = self.pos
        pnl = (float(px) - p.entry_px) * p.qty - float(costs)
        self.realized += pnl
        rec = p.as_dict()
        rec.update(exit_px=float(px), exit_ts=float(ts or time.time()),
                   pnl=round(pnl, 2), exit_reason=reason)
        self.closed.append(rec)
        self.pos = None
        log.info("CAS EXIT %s @ %.2f (%s) Rs%+.2f | CAS book today Rs%+.2f "
                 "— attributed separately from the day session, which is "
                 "the only way the auction's edge can ever be measured",
                 p.symbol, px, reason, pnl, self.realized)
        self._persist()
        return pnl

    # -------------------------------------------------------------- state
    def _path(self):
        return config.STATE_DIR / "cas_book.json"

    def _persist(self) -> None:
        try:
            p = self._path()
            p.parent.mkdir(parents=True, exist_ok=True)
            body = {"day": self.day, "config_hash": config.CONFIG_HASH,
                    "pos": self.pos.as_dict() if self.pos else None,
                    "closed": self.closed[-20:], "realized": self.realized,
                    "entries": self.entries, "blocked": self.blocked,
                    "capital": self.capital()}
            tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.json")
            tmp.write_text(json.dumps(body, indent=1, default=float),
                           encoding="utf-8")
            os.replace(tmp, p)
        except Exception as e:                             # noqa: BLE001
            log.debug("cas persist failed (%s)", e)

    @classmethod
    def load_or_new(cls) -> "CasBook":
        today = dt.date.today().isoformat()
        try:
            p = config.STATE_DIR / "cas_book.json"
            if p.exists():
                b = json.loads(p.read_text(encoding="utf-8"))
                if (b.get("day") == today
                        and b.get("config_hash") == config.CONFIG_HASH):
                    bk = cls(day=today,
                             realized=float(b.get("realized") or 0.0),
                             entries=int(b.get("entries") or 0),
                             closed=list(b.get("closed") or []),
                             blocked=dict(b.get("blocked") or {}))
                    if b.get("pos"):
                        bk.pos = CasPosition(**b["pos"])
                    return bk
        except Exception as e:                             # noqa: BLE001
            log.warning("cas restore failed (%s)", e)
        return cls(day=today)

    def summary(self) -> dict:
        return {"day": self.day, "entries": self.entries,
                "realized": round(self.realized, 2),
                "open": self.pos.symbol if self.pos else None,
                "capital": round(self.capital(), 2),
                "blocked": dict(self.blocked),
                "closed": len(self.closed)}