"""
SESSION CALENDAR — India equity market structure, before and after
   the 2026-08-03 Closing Auction Session reform
===================================================================
Effective **3 August 2026** (NSE circular 30 May 2026, SEBI CAS
framework), the shape of the Indian trading day changed:

  • EQUITY DERIVATIVES (index AND stock F&O) now close at **15:40**,
    not 15:30. The open is unchanged at 09:15. The session is therefore
    **385 minutes**, not 375.
  • The CASH market gained a **Closing Auction Session**. Stocks that
    have F&O contracts stop CONTINUOUS trading at **15:15** and are in
    auction until **15:35**; the auction price — not a VWAP — becomes
    the official close. Non-F&O stocks continue to 15:30. A cash
    post-close session runs 15:50–16:00.
  • The derivatives closing-price VWAP window moved to 15:10–15:40.

WHY THIS MATTERS TO A GAMMA/DELTA SYSTEM — the part that is easy to miss
-----------------------------------------------------------------------
NIFTY and BANKNIFTY are computed from constituents that are ALL F&O
stocks. From 15:15 to 15:35 those constituents are in auction, not in
continuous trade. For twenty minutes the option is live and liquid while
the INDEX PRINT BEHIND IT is derived from securities that are no longer
continuously priced.

Everything this system reasons with in that window is therefore suspect:
spot, and so delta, expected move, GEX walls, the gamma flip, cascade
z-scores, regime labels. The options themselves keep trading — so exits
remain both possible and necessary — but a NEW entry taken on a
spot-derived signal between 15:15 and 15:35 is an entry taken on a
number that no longer means what the model was trained to think it
means. This module therefore exposes a CAS blackout the brain uses to
suspend entries while leaving exit management fully armed.

HISTORY IS NOT REWRITTEN
------------------------
The vault holds sessions from June 2026 that really did end at 15:30.
Replaying them under a 15:40 close would fabricate ten minutes of
market that never existed and shift every time-of-day feature. Every
function here is therefore keyed on the SESSION DATE and returns the
rules that were actually in force that day.

BSE (SENSEX / BANKEX): SEBI's framework covers both exchanges and BSE
has published a CAS indicator in its scrip master, but as of this
writing BSE's own timing circular is still awaited. BSE_FOLLOWS_NSE_CAS
defaults False — SENSEX/BANKEX keep the 15:30 close until you flip it,
because assuming an extension that has not been confirmed would leave
positions open in a market that has shut.
"""
from __future__ import annotations

import datetime as dt

import config

# The reform date. Sessions on or after this trade under the new rules.
CAS_EFFECTIVE = dt.date(2026, 8, 3)


def _as_date(day) -> dt.date:
    # datetime subclasses date, so it MUST be tested first — otherwise a
    # datetime falls through unchanged and every comparison against a
    # plain date raises.
    if isinstance(day, dt.datetime):
        return day.date()
    if isinstance(day, dt.date):
        return day
    if isinstance(day, (int, float)):
        return dt.datetime.fromtimestamp(float(day)).date()
    return dt.date.fromisoformat(str(day)[:10])


def _hm_to_min(hm: str) -> int:
    h, m = str(hm).split(":")[:2]
    return int(h) * 60 + int(m)


def cas_in_force(day) -> bool:
    """Did the 2026-08-03 reform apply on this session?"""
    return _as_date(day) >= CAS_EFFECTIVE


def is_bse(index: str) -> bool:
    return str(index).upper() in {"SENSEX", "BANKEX"}


def session_close_hm(day, index: str = "NIFTY") -> str:
    """Derivatives close for this index on this date."""
    if not cas_in_force(day):
        return "15:30"
    if is_bse(index) and not bool(getattr(config, "BSE_FOLLOWS_NSE_CAS",
                                          False)):
        return "15:30"
    return str(getattr(config, "SESSION_CLOSE_CAS", "15:40"))


def session_minutes(day, index: str = "NIFTY") -> int:
    """Length of the tradable derivatives session in minutes — 375 before
    the reform, 385 after (NSE). This is the denominator for every
    time-of-day feature and every √t scaling; getting it wrong shifts
    the entire feature space by 2.7%."""
    return max(_hm_to_min(session_close_hm(day, index))
               - _hm_to_min(str(getattr(config, "SESSION_OPEN", "09:15"))), 1)


def minutes_to_close(now, day=None, index: str = "NIFTY") -> float:
    """Minutes remaining in the derivatives session at `now` (a datetime
    or epoch). Never negative."""
    t = (now if isinstance(now, dt.datetime)
         else dt.datetime.fromtimestamp(float(now)))
    d = _as_date(day if day is not None else t)
    return max(_hm_to_min(session_close_hm(d, index))
               - (t.hour * 60 + t.minute + t.second / 60.0), 0.0)


# NSE's actual CAS phase structure (official CAS page, updated 2026-08-02;
# circulars NSE/CMTR/74466 and NSE/FAOP/74467 of 2026-05-29). Each phase
# behaves differently and a system that lumps them into one 20-minute blob
# will misread all four transitions.
CAS_PHASES = (
    # (start, end, phase, index_price_quality)
    ("15:15", "15:20", "CAS_REFERENCE", "indicative"),
    #   reference price = VWAP of 15:00–15:15 trades; CTS→CAS transition
    ("15:20", "15:25", "CAS_ENTRY", "indicative"),
    #   limit AND market orders may be entered/modified/cancelled
    ("15:25", "15:30", "CAS_LIMIT_ONLY", "indicative"),
    #   limit orders only; market orders frozen; system-driven RANDOM
    #   closure somewhere in the final two minutes (15:28–15:30)
    ("15:30", "15:35", "CAS_MATCHING", "settling"),
    #   uncrossing at the equilibrium price; the official close is born here
    ("15:35", "15:40", "POST_AUCTION", "traded"),
    #   cash close is KNOWN and published; derivatives still trade. This is
    #   new information the market did not previously have at any point.
)


def cas_window(day, index: str = "NIFTY") -> tuple[str, str] | None:
    """(start, end) of the cash Closing Auction Session, or None when it
    does not apply to this index on this date."""
    if not cas_in_force(day):
        return None
    if is_bse(index) and not bool(getattr(config, "BSE_FOLLOWS_NSE_CAS",
                                          False)):
        return None
    return (str(getattr(config, "CAS_START", "15:15")),
            str(getattr(config, "CAS_END", "15:35")))


def cas_phase(now, day=None, index: str = "NIFTY") -> str:
    """Which phase of the day are we in?

    CTS            continuous trading, everything normal
    CAS_REFERENCE  cash frozen, reference VWAP being struck
    CAS_ENTRY      auction book building, market+limit orders
    CAS_LIMIT_ONLY limit only, random close imminent
    CAS_MATCHING   uncrossing; the official close is being determined
    POST_AUCTION   cash closed and PUBLISHED; derivatives still trading
    CLOSED         after the derivatives bell
    """
    t = (now if isinstance(now, dt.datetime)
         else dt.datetime.fromtimestamp(float(now)))
    d = _as_date(day if day is not None else t)
    cur = t.hour * 60 + t.minute + t.second / 60.0
    if cur >= _hm_to_min(session_close_hm(d, index)):
        return "CLOSED"
    if not cas_window(d, index):
        return "CTS"
    for start, end, phase, _q in CAS_PHASES:
        if _hm_to_min(start) <= cur < _hm_to_min(end):
            return phase
    return "CTS"


def index_price_quality(now, day=None, index: str = "NIFTY") -> str:
    """What KIND of number is the spot feed carrying right now?

    'traded'      — a continuously-traded index. Normal.
    'indicative'  — NSE disseminates an INDICATIVE INDEX during CAS,
                    computed from indicative equilibrium prices of the
                    constituents. It is not stale: it moves, and it can
                    jump as the auction book builds and again at the
                    random closure. Any z-score, flip-break, regime label
                    or GEX read taken from it is measuring auction
                    mechanics, not the market the model was trained on.
    'settling'    — uncrossing in progress; the print is converging on the
                    official close.
    """
    ph = cas_phase(now, day, index)
    for _s, _e, phase, quality in CAS_PHASES:
        if phase == ph:
            return quality
    return "traded" if ph != "CLOSED" else "closed"


def in_cas_blackout(now, day=None, index: str = "NIFTY") -> bool:
    """True while the index print is INDICATIVE or SETTLING — i.e. through
    CAS_REFERENCE, CAS_ENTRY, CAS_LIMIT_ONLY and CAS_MATCHING (15:15–15:35).
    Entries are suspended; exits are NOT, because the option is still live
    and an open position still needs managing. Deliberately does NOT cover
    POST_AUCTION (15:35–15:40), where the cash close is known and published
    and the index is a real number again."""
    if not bool(getattr(config, "CAS_BLACKOUT_ENABLED", True)):
        return False
    return index_price_quality(now, day, index) in ("indicative", "settling")


def in_post_auction(now, day=None, index: str = "NIFTY") -> bool:
    """15:35–15:40: the ten minutes that did not exist before 2026-08-03.
    The cash closing price has been discovered and published, and index
    derivatives are still open. Liquidity is thin and the move can be
    sharp, so entries here are governed by POST_AUCTION_ENTRIES (default
    False: the system has no vault evidence about this regime yet — it
    has never traded a single second of it)."""
    return cas_phase(now, day, index) == "POST_AUCTION"


def entry_curfew_hm(day, index: str = "NIFTY") -> str:
    """Last clock time a NEW entry may be taken. Derived from this
    session's close, so it extended itself on 2026-08-03 instead of
    leaving a hand-set 15:05 that shut entries 35 minutes early."""
    close = _hm_to_min(session_close_hm(day, index))
    m = close - int(getattr(config, "NO_ENTRY_BEFORE_CLOSE_MIN", 5))
    # v9.9.14: on a post-reform NIFTY session that lands at 15:35 — exactly
    # when the POST-AUCTION window opens. The curfew would therefore have
    # slammed shut the very window the session extension exists to reach:
    # every post-auction entry refused with "entry curfew after 15:35".
    # Once the regime is open, the curfew moves to just inside the
    # post-auction flat time, which is itself inside the bell.
    if cas_window(day, index):
        try:
            from core import cas_capture as _CC
            from core import post_auction as _PA
            if _PA.readiness()[0] or _CC.preprint_readiness(index)[0]:
                pa_flat = _hm_to_min(str(getattr(config,
                                                 "POST_AUCTION_FLAT_HM",
                                                 "15:39")))
                m = max(m, pa_flat - 1)
        except Exception:                                  # noqa: BLE001
            pass
    return f"{m // 60:02d}:{m % 60:02d}"


def entries_allowed(now, day=None, index: str = "NIFTY") -> tuple[bool, str]:
    """THE session-phase entry rule — one copy, used by the live brain and
    by the forge grader, so the model is never trained on decisions
    serving cannot make.

    Returns (allowed, reason). Exits are never governed here: an open
    position must be managed in every phase the option trades in.
    """
    ph = cas_phase(now, day, index)
    if ph == "CLOSED":
        return False, "session closed"
    if in_cas_blackout(now, day, index):
        # v9.9.15: the blackout is not permanent — it is the DEFAULT while
        # the system has no way to see through the auction. Once
        # cas_capture proves the option-implied basis forecasts the print,
        # entries inside 15:15–15:35 open, because that is the only place
        # from which the 15:35 move can be captured at all. The conviction
        # stack still does not vote here: the basis decides, alone.
        if ph in ("CAS_REFERENCE", "CAS_ENTRY", "CAS_LIMIT_ONLY"):
            from core import cas_capture as _CC
            ok_pp, why_pp = _CC.preprint_readiness(index)
            if ok_pp:
                return True, f"CAS_PREPRINT ({why_pp})"
            return False, f"{ph} — {why_pp}"
        return False, (f"{ph} — index print is "
                       f"{index_price_quality(now, day, index)}, not a "
                       f"traded price; uncrossing in progress")
    if ph == "POST_AUCTION":
        # v9.9.12: no hand-set flag any more. core.post_auction.readiness()
        # opens this window BY ITSELF once the vault holds a week of it and
        # the fitted geometry proves the median move can pay the spread.
        from core import post_auction as _PA
        ok, why = _PA.readiness()
        if not ok:
            return False, f"post-auction 15:35-15:40 — {why}"
        return True, f"POST_AUCTION ({why})"
    return True, ph


def flatten_hm(day, index: str = "NIFTY") -> str:
    """When open positions are force-flattened. Normally the hard-flat, but
    once the post-auction regime is open the book must be allowed to hold a
    trade INTO that window — otherwise EOD_FLATTEN fires at 15:35 and kills
    every post-auction position in the second it is born. The post-auction
    ladder owns its own bell (POST_AUCTION_FLAT_HM) and is stricter."""
    base = hard_flat_hm(day, index)
    if cas_window(day, index):
        try:
            from core import cas_capture as _CC
            from core import post_auction as _PA
            if _PA.readiness()[0] or _CC.preprint_readiness(index)[0]:
                return str(getattr(config, "POST_AUCTION_FLAT_HM", "15:39"))
        except Exception:                                  # noqa: BLE001
            pass
    return base


def hard_flat_hm(day, index: str = "NIFTY") -> str:
    """When every position must be flat. Kept a fixed margin inside the
    close so the flatten has time to fill — extending the session must
    never mean flattening later into thinner liquidity than before."""
    margin = int(getattr(config, "HARD_FLAT_MARGIN_MIN", 5))
    m = _hm_to_min(session_close_hm(day, index)) - margin
    return f"{m // 60:02d}:{m % 60:02d}"


# ---------------------------------------------------------------- selftest
if __name__ == "__main__":                                 # pragma: no cover
    import sys
    ok = 0

    def chk(name, cond):
        global ok
        print(("  PASS  " if cond else "  FAIL  ") + name)
        ok += bool(cond)

    OLD, NEW = "2026-07-31", "2026-08-03"
    chk("pre-reform NIFTY closes 15:30", session_close_hm(OLD) == "15:30")
    chk("post-reform NIFTY closes 15:40", session_close_hm(NEW) == "15:40")
    chk("pre-reform session is 375 min", session_minutes(OLD) == 375)
    chk("post-reform session is 385 min", session_minutes(NEW) == 385)
    chk("reform date itself is INCLUDED", cas_in_force("2026-08-03"))
    chk("the day before is not", not cas_in_force("2026-08-02"))

    chk("BSE stays 15:30 until confirmed",
        session_close_hm(NEW, "SENSEX") == "15:30"
        and session_minutes(NEW, "SENSEX") == 375)
    config.BSE_FOLLOWS_NSE_CAS = True
    chk("BSE follows once the flag is set",
        session_close_hm(NEW, "SENSEX") == "15:40")
    config.BSE_FOLLOWS_NSE_CAS = False

    chk("no CAS window before the reform", cas_window(OLD) is None)
    chk("CAS window is 15:15–15:35 after", cas_window(NEW) == ("15:15", "15:35"))

    d = dt.datetime(2026, 8, 3, 15, 20)
    chk("15:20 is inside the blackout", in_cas_blackout(d))
    chk("15:14 is not", not in_cas_blackout(dt.datetime(2026, 8, 3, 15, 14)))
    chk("15:35 is not (auction over)",
        not in_cas_blackout(dt.datetime(2026, 8, 3, 15, 35)))
    chk("15:20 on 2026-07-31 is not (no reform yet)",
        not in_cas_blackout(dt.datetime(2026, 7, 31, 15, 20)))
    chk("SENSEX ignores the blackout while BSE is unconfirmed",
        not in_cas_blackout(d, index="SENSEX"))

    chk("minutes_to_close at 15:20 post-reform is 20",
        abs(minutes_to_close(d) - 20.0) < 1e-6)
    chk("minutes_to_close at 15:20 pre-reform is 10",
        abs(minutes_to_close(dt.datetime(2026, 7, 31, 15, 20)) - 10.0) < 1e-6)
    chk("never negative after the bell",
        minutes_to_close(dt.datetime(2026, 8, 3, 16, 30)) == 0.0)

    chk("hard-flat sits inside the close",
        hard_flat_hm(NEW) == "15:35" and hard_flat_hm(OLD) == "15:25")

    total = 19
    print(f"\n{ok}/{total} checks passed")
    sys.exit(0 if ok == total else 1)