"""
APEX OMNI v9.7.1 — COMMODITY EVENT ENGINE (the honest "news" guard)
===================================================================
Commodity markets move on scheduled, KNOWN-IN-ADVANCE data releases far more
violently than equity indices. This engine encodes those releases and guards
the book around them. It is the honest version of "commodity moves by news":
the highest-impact commodity news is a CALENDAR, not a mystery — and a calendar
is deterministic, testable, and needs no paid feed.

Scope, stated plainly
----------------------
This is a SCHEDULED-EVENT + GAP guard, NOT a real-time sentiment engine. It does
not read headlines and trade their tone — that needs a low-latency paid news
feed and is a research project of its own; faking it would be a mockup. What it
DOES, and what actually protects a commodity book, is:
  • know the release windows (EIA petroleum, EIA natgas storage, OPEC, FOMC,
    US CPI, US NFP) in IST with correct US-Eastern → IST DST handling,
  • BLOCK new entries in a pre-release window (don't buy into a coin flip),
  • FLATTEN / widen protection through the release + settle window,
  • expose an "event pressure" the (future) commodity engine and the nightly
    Gemma analyst consume.

The releases (authoritative cadence, US Eastern; holiday dates shift — handled
by explicit overrides):
  • EIA Weekly Petroleum Status Report — Wed 10:30 ET  → CRUDEOIL (huge)
  • EIA Weekly Natural Gas Storage      — Thu 10:30 ET  → NATURALGAS (huge)
  • FOMC decision                       — 14:00 ET      → GOLD/SILVER (macro)
  • US CPI                              — 08:30 ET      → GOLD/SILVER
  • US NFP                              — 08:30 ET (1st Fri) → GOLD/SILVER
  • OPEC(+) meetings                    — date-driven    → CRUDEOIL
DST: 10:30 ET = 20:00 IST (EDT, summer) or 21:00 IST (EST, winter). Computed
from a real US-Eastern clock, never hardcoded.

This module is PURE + DETERMINISTIC (calendar math only) so it replays and
self-tests. Live wiring is a separate, later step (the commodity engine).

  python core/event_engine.py
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")
_IST = ZoneInfo("Asia/Kolkata")


def _cfg(name, default):
    try:
        import config
        return getattr(config, name, default)
    except Exception:                                          # pragma: no cover
        return default


# --------------------------------------------------------------------------
# Event model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketEvent:
    name: str                 # "EIA_PETROLEUM"
    et_time: dt.time          # release time in US Eastern
    affects: tuple            # ("CRUDEOIL",) — commodities it moves
    severity: str             # "extreme" | "high" | "medium"
    # scheduling: either a weekly weekday rule OR an explicit date (overrides)
    weekday: int | None = None       # 0=Mon … 6=Sun (weekly cadence)
    explicit_date: dt.date | None = None


# The recurring weekly rules (the backbone). Holiday shifts and one-off events
# (OPEC, FOMC, CPI, NFP dates) come from EVENT_OVERRIDES, which the operator or
# the nightly refresh updates — this module never guesses a shifted date.
_WEEKLY_EVENTS = [
    MarketEvent("EIA_PETROLEUM", dt.time(10, 30), ("CRUDEOIL",),
                "extreme", weekday=2),                         # Wednesday
    MarketEvent("EIA_NATGAS_STORAGE", dt.time(10, 30), ("NATURALGAS",),
                "extreme", weekday=3),                         # Thursday
]


def _default_overrides() -> list[MarketEvent]:
    """Explicit-date events + holiday shifts. Seeded from the authoritative
    calendars; the operator extends this list (or a nightly job refreshes it).
    Empty-safe: with no overrides, only the weekly EIA rules fire."""
    ov = _cfg("EVENT_OVERRIDES", None)
    if ov is not None:
        return ov
    return []               # operator/nightly supplies dated OPEC/FOMC/CPI/NFP


@dataclass
class EventVerdict:
    in_blackout: bool = False        # inside pre-release entry-block window
    in_settle: bool = False          # release just fired; volatile settle window
    severity: str = ""               # of the governing event
    event: str = ""                  # its name
    affects: tuple = ()              # commodities it moves
    minutes_to_event: float | None = None   # signed: <0 = already released
    reason: str = "clear"


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------
class CommodityEventEngine:
    """Given a timezone-aware 'now', tells the book whether a commodity is in a
    scheduled-event blackout or settle window. Deterministic and replayable."""

    def __init__(self, overrides: list[MarketEvent] | None = None):
        self.events = list(_WEEKLY_EVENTS) + (
            overrides if overrides is not None else _default_overrides())

    def _event_datetime_ist(self, ev: MarketEvent, day: dt.date) -> dt.datetime:
        """The release instant for `ev` on `day`, converted ET→IST (DST-correct)."""
        et = dt.datetime.combine(day, ev.et_time, tzinfo=_ET)
        return et.astimezone(_IST)

    def _events_on(self, day_ist: dt.date) -> list[tuple[MarketEvent, dt.datetime]]:
        """All events whose release lands on `day_ist` (in IST), as (ev, ts)."""
        out = []
        for ev in self.events:
            if ev.explicit_date is not None:
                # explicit ET date → its IST instant; include if that IST date
                # matches (a late-ET event can spill to next IST day)
                ts = self._event_datetime_ist(ev, ev.explicit_date)
                if ts.date() == day_ist:
                    out.append((ev, ts))
            elif ev.weekday is not None:
                # weekly: the ET release date is the one whose IST instant lands
                # today. Check yesterday/today ET to cover the date-line shift.
                for delta in (-1, 0):
                    et_day = day_ist + dt.timedelta(days=delta)
                    if et_day.weekday() == ev.weekday:
                        ts = self._event_datetime_ist(ev, et_day)
                        if ts.date() == day_ist:
                            out.append((ev, ts))
        return out

    def evaluate(self, now_ist: dt.datetime, commodity: str) -> EventVerdict:
        """Is `commodity` in a scheduled-event window at `now_ist`?"""
        if now_ist.tzinfo is None:
            now_ist = now_ist.replace(tzinfo=_IST)
        pre = float(_cfg("EVENT_BLACKOUT_PRE_MIN", 20))       # block N min before
        post = float(_cfg("EVENT_SETTLE_POST_MIN", 30))       # settle N min after
        v = EventVerdict()
        best_abs = None
        # consider events landing today and tomorrow (a pre-window can straddle)
        for day in (now_ist.date(), now_ist.date() + dt.timedelta(days=1)):
            for ev, ts in self._events_on(day):
                if commodity not in ev.affects:
                    continue
                delta_min = (ts - now_ist).total_seconds() / 60.0
                # blackout: [ts - pre, ts)
                if -0.0001 <= delta_min <= pre:
                    if not v.in_blackout or (best_abs is None
                                             or abs(delta_min) < best_abs):
                        v.in_blackout, v.severity, v.event = True, ev.severity, ev.name
                        v.affects, v.minutes_to_event = ev.affects, round(delta_min, 1)
                        best_abs = abs(delta_min)
                # settle: (ts, ts + post]
                elif -post <= delta_min < 0:
                    if not v.in_settle and not v.in_blackout:
                        v.in_settle, v.severity, v.event = True, ev.severity, ev.name
                        v.affects, v.minutes_to_event = ev.affects, round(delta_min, 1)
        if v.in_blackout:
            v.reason = (f"BLACKOUT: {v.event} in {v.minutes_to_event:.0f} min "
                        f"(severity {v.severity}) — no new {commodity} entries")
        elif v.in_settle:
            v.reason = (f"SETTLE: {v.event} released {abs(v.minutes_to_event):.0f} "
                        f"min ago — {commodity} volatile, widen/flatten only")
        return v

    def next_event(self, now_ist: dt.datetime,
                   commodity: str | None = None) -> tuple[str, dt.datetime] | None:
        """The next upcoming event (optionally for one commodity), for telemetry
        and the nightly digest."""
        if now_ist.tzinfo is None:
            now_ist = now_ist.replace(tzinfo=_IST)
        cands = []
        for day_off in range(0, 8):
            day = now_ist.date() + dt.timedelta(days=day_off)
            for ev, ts in self._events_on(day):
                if commodity and commodity not in ev.affects:
                    continue
                if ts > now_ist:
                    cands.append((ts, ev))
        if not cands:
            return None
        cands.sort(key=lambda x: x[0])
        ts, ev = cands[0]
        return ev.name, ts


# --------------------------------------------------------------------------
# Entry gate the (future) commodity engine will consult
# --------------------------------------------------------------------------
def event_entry_gate(v: EventVerdict) -> tuple[bool, str]:
    """(allow_entry, why). Blackout blocks new entries outright; settle blocks
    too (the release just fired — let it stabilize). Advisory: it only ever
    BLOCKS, never forces a trade."""
    if v.in_blackout:
        return False, v.reason
    if v.in_settle and v.severity in ("extreme", "high"):
        return False, v.reason
    return True, "clear"


# --------------------------------------------------------------------------
# SELF-TEST (deterministic calendar math)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    eng = CommodityEventEngine(overrides=[])     # weekly EIA rules only

    def ist(y, m, d, hh, mm):
        return dt.datetime(y, m, d, hh, mm, tzinfo=_IST)

    print("=== EIA petroleum (Wed 10:30 ET) → IST window, DST-correct ===")
    # July = EDT (UTC-4); 10:30 ET = 20:00 IST. 2026-07-22 is a Wednesday.
    for label, now in [
            ("18:00 IST (well before)", ist(2026, 7, 22, 18, 0)),
            ("19:45 IST (in blackout)", ist(2026, 7, 22, 19, 45)),
            ("20:10 IST (in settle)",   ist(2026, 7, 22, 20, 10)),
            ("21:30 IST (clear after)", ist(2026, 7, 22, 21, 30))]:
        v = eng.evaluate(now, "CRUDEOIL")
        allow, why = event_entry_gate(v)
        print(f"  {label}: allow={allow} — {why}")

    print("\n=== NatGas storage (Thu 10:30 ET) only affects NATURALGAS ===")
    now = ist(2026, 7, 23, 19, 45)      # Thursday, in NG blackout
    vg = eng.evaluate(now, "NATURALGAS")
    vc = eng.evaluate(now, "CRUDEOIL")
    print(f"  NATURALGAS 19:45 Thu: blackout={vg.in_blackout} ({vg.event})")
    print(f"  CRUDEOIL   19:45 Thu: blackout={vc.in_blackout} (unaffected → "
          f"{vc.reason})")

    print("\n=== winter DST check (Jan = EST, 10:30 ET = 21:00 IST) ===")
    # 2026-01-21 is a Wednesday. 20:45 IST should NOT yet be blackout (event 21:00)
    now = ist(2026, 1, 21, 20, 45)
    v = eng.evaluate(now, "CRUDEOIL")
    print(f"  20:45 IST Jan (event at 21:00): blackout={v.in_blackout} "
          f"minutes_to_event={v.minutes_to_event}")
    now = ist(2026, 1, 21, 20, 55)
    v = eng.evaluate(now, "CRUDEOIL")
    print(f"  20:55 IST Jan: blackout={v.in_blackout} "
          f"minutes_to_event={v.minutes_to_event}")

    print("\n=== explicit-date event (e.g. OPEC on a specific day) ===")
    opec = MarketEvent("OPEC_MEETING", dt.time(6, 0), ("CRUDEOIL",),
                       "extreme", explicit_date=dt.date(2026, 8, 3))
    eng2 = CommodityEventEngine(overrides=[opec])
    nxt = eng2.next_event(ist(2026, 7, 30, 12, 0), "CRUDEOIL")
    print(f"  next CRUDEOIL event from 2026-07-30: {nxt}")