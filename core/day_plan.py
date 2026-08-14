"""
DAY PLAN — one thesis per day, formed once, reviewed once, exited on the clock
==============================================================================
WHAT THE 2026-08-11 TAPE ACTUALLY SHOWS
---------------------------------------
    09:20  24400PE  conv -0.600  15.5m  STOP        +157.27
    09:40  24350PE  conv -0.483   5.1m  STOP        -270.23
    10:40  24400PE  conv -0.660  13.5m  STOP        -257.03
    12:54  24400PE  conv -0.457   8.9m  STOP        -254.66
    13:06  SENSEX CE conv 0.790  60.7m  THETA       -670.50
    14:31  24350PE  conv -0.704   6.8m  PROFIT_LOCK   -2.29
                                                    -1297.44

FIVE of six trades were NIFTY puts. The 24400 strike was entered THREE
times and stopped out every time; the 24350 twice. That is one directional
thesis — NIFTY down — re-expressed six times in a session that never
delivered it, paying the round-trip cost and the stop each time. The
operator's read is correct and the ledger is unambiguous: the losses are
not six independent bad trades, they are one bad thesis re-entered.

MAX_CONCURRENT_POSITIONS=1 does not prevent this. It bounds CONCURRENCY,
not REPETITION: exit, wait out COOLDOWN_S, re-enter the same strike. The
cooldown is 180 seconds; the gap between the 24400PE attempts was 65
minutes and 121 minutes. Nothing in the stack was ever going to stop it.

WHAT THIS MODULE DOES
---------------------
Forms ONE thesis per session and holds the book to it:

  09:15-09:45  OBSERVE. No entries. The open is the widest-spread,
               highest-variance window of the day and the regime label is
               still settling; on 2026-08-11 NIFTY spent 5050s in
               VOL_CRUSH and 9977s in SQUEEZE_PRONE, and those labels are
               not stable in the first thirty minutes.
  09:50        COMMIT. The best-scoring candidate at the decision instant
               becomes the day's thesis. One entry. If nothing clears the
               bar, the day is FLAT and stays flat.
  ~12:30       REVIEW. Re-score the open position's own entry conviction.
               A thesis that has REVERSED is closed then, rather than
               waiting for a stop that may be far away. This is the piece
               the exit stack genuinely lacked: it ratchets and consults
               TrapShield, but it never re-asks "is the reason I entered
               still true?"
  15:05        FLAT. The session exit replaces the theta guillotine.

  CAS (15:15-15:40) is a SEPARATE session with its own book and its own
  gates — auction microstructure, indicative prices, no continuous
  matching. It is not a continuation of the day plan and never inherits
  its thesis.

ON REMOVING MAX_HOLD_THETA — STATED PLAINLY
--------------------------------------------
The operator asked for it and it is implemented. The evidence points the
other way and that is recorded here rather than silently overridden: on
2026-08-11 the theta guillotine cut the single worst trade of the day
(SENSEX 78300CE, -Rs670.50). That position peaked at 304.80 against an
entry of 300.35 — up 1.5%, once — and was at 270.20 when the clock cut
it. Removing the guillotine does not save that trade; it lets it bleed
from 14:06 to 15:05.

What makes the removal defensible is that the day plan replaces an
UNBOUNDED hold with a BOUNDED one: DAYPLAN_EXIT_HM is a hard flat, and
the mid-session review can close a reversed thesis long before it. Theta
decay does not stop being real because the clock was removed — so the
disaster floor and the stop remain untouched, and the review exists
precisely to catch what the guillotine used to catch by brute force.

This is A/B-able, not a one-way door. DAYPLAN_ENABLED=False restores the
incumbent exactly, and core/entry_counterfactual.py can simulate the day
plan against the incumbent on the same tape before either is trusted.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict

import config

log = logging.getLogger("day_plan")

OBSERVE, ARMED, COMMITTED, REVIEWED, FLAT, DONE = (
    "OBSERVE", "ARMED", "COMMITTED", "REVIEWED", "FLAT", "DONE")


def _hm_to_sod(hm: str) -> int:
    h, m = str(hm).split(":")[:2]
    return int(h) * 3600 + int(m) * 60


def _now_sod(ts: float | None = None) -> int:
    lt = dt.datetime.fromtimestamp(ts if ts is not None else time.time())
    return lt.hour * 3600 + lt.minute * 60 + lt.second


@dataclass
class Candidate:
    """One index's standing at the decision instant."""
    index: str
    conv: float
    win_prob: float
    symbol: str = ""
    token: int = 0
    regime: str = ""
    news_score: float = 0.0
    ts: float = 0.0

    @property
    def score(self) -> float:
        """Ranking score at commit. Conviction is the spine; the news read
        is a TILT with a bounded weight, never a veto and never a driver —
        an unbounded LLM term would make the day's thesis unauditable."""
        w = float(getattr(config, "DAYPLAN_NEWS_WEIGHT", 0.15))
        w = max(min(w, 0.35), 0.0)
        tilt = 1.0 + w * max(min(self.news_score, 1.0), -1.0) * (
            1.0 if self.conv > 0 else -1.0)
        return abs(self.conv) * tilt

    def as_dict(self) -> dict:
        d = asdict(self)
        d["score"] = round(self.score, 4)
        return d


@dataclass
class DayPlan:
    day: str
    state: str = OBSERVE
    thesis: Candidate | None = None
    committed_ts: float = 0.0
    reviewed_ts: float = 0.0
    review_verdict: str = ""
    closed_ts: float = 0.0
    close_reason: str = ""
    observed: list = field(default_factory=list)

    # ------------------------------------------------------------ windows
    @staticmethod
    def _t(name: str, default: str) -> int:
        return _hm_to_sod(str(getattr(config, name, default)))

    @classmethod
    def analysis_end(cls) -> int:
        return cls._t("DAYPLAN_ANALYSIS_END_HM", "09:45")

    @classmethod
    def entry_t(cls) -> int:
        return cls._t("DAYPLAN_ENTRY_HM", "09:50")

    @classmethod
    def commit_deadline(cls) -> int:
        """The commit window is not a single instant: a one-second window
        would miss the day whenever the loop jitters or a quote is stale.
        It opens at DAYPLAN_ENTRY_HM and closes here."""
        return cls._t("DAYPLAN_COMMIT_END_HM", "10:20")

    @classmethod
    def review_t(cls) -> int:
        return cls._t("DAYPLAN_REVIEW_HM", "12:30")

    @classmethod
    def exit_t(cls) -> int:
        return cls._t("DAYPLAN_EXIT_HM", "15:05")

    @classmethod
    def cas_start(cls) -> int:
        return cls._t("DAYPLAN_CAS_START_HM", "15:15")

    # -------------------------------------------------------------- phase
    def phase(self, ts: float | None = None) -> str:
        sod = _now_sod(ts)
        if sod < self.analysis_end():
            return "OBSERVE"
        if sod < self.entry_t():
            return "ANALYSE"
        if sod < self.commit_deadline():
            return "COMMIT_WINDOW"
        if sod < self.review_t():
            return "HOLD"
        if sod < self.exit_t():
            return "REVIEW"
        if sod < self.cas_start():
            return "FLATTEN"
        return "CAS"

    # ------------------------------------------------------------ gating
    def may_enter(self, ts: float | None = None) -> tuple[bool, str]:
        """THE single-trade rule. One commit per session, full stop.

        This is what the incumbent stack could not express: it bounded
        concurrency and cooled down for 180s, so the same strike came back
        65 and 121 minutes later. Here a day that has committed is done
        committing, whatever the tape does afterwards."""
        if not bool(getattr(config, "DAYPLAN_ENABLED", False)):
            return True, ""
        ph = self.phase(ts)
        if self.state in (COMMITTED, REVIEWED):
            return False, ("day plan: already committed to "
                           f"{self.thesis.symbol if self.thesis else '?'} — "
                           f"one thesis per session")
        if self.state in (FLAT, DONE):
            return False, "day plan: session closed to new entries"
        if ph == "OBSERVE":
            return False, (f"day plan: observing until "
                           f"{getattr(config, 'DAYPLAN_ANALYSIS_END_HM', '09:45')}"
                           f" — the open is the widest-spread window and the "
                           f"regime label is still settling")
        if ph == "ANALYSE":
            return False, (f"day plan: analysis window — commit opens "
                           f"{getattr(config, 'DAYPLAN_ENTRY_HM', '09:50')}")
        if ph == "COMMIT_WINDOW":
            return True, ""
        return False, f"day plan: commit window has closed ({ph})"

    def must_exit(self, ts: float | None = None) -> tuple[bool, str]:
        """Session flat. Replaces MAX_HOLD_THETA with a BOUNDED hold — the
        position ends at DAYPLAN_EXIT_HM whatever else happens."""
        if not bool(getattr(config, "DAYPLAN_ENABLED", False)):
            return False, ""
        if self.state not in (COMMITTED, REVIEWED):
            return False, ""
        if _now_sod(ts) >= self.exit_t():
            return True, (f"DAYPLAN_SESSION_EXIT "
                          f"{getattr(config, 'DAYPLAN_EXIT_HM', '15:05')}")
        return False, ""

    def due_for_review(self, ts: float | None = None) -> bool:
        return (bool(getattr(config, "DAYPLAN_ENABLED", False))
                and self.state == COMMITTED
                and _now_sod(ts) >= self.review_t())

    def committed(self) -> bool:
        """Has the session's one thesis already been taken? The live loop
        asks this on every fill, so it must be cheap and total."""
        return self.state in (COMMITTED, REVIEWED)

    # ------------------------------------------------------------ actions
    def observe(self, cands: list[Candidate]) -> None:
        """Record the analysis window. Kept for the report, and so the
        commit can be audited against what was actually visible."""
        if not cands:
            return
        self.observed.append({"ts": time.time(),
                              "top": max(cands, key=lambda c: c.score
                                         ).as_dict()})
        self.observed = self.observed[-40:]

    def commit(self, cands: list[Candidate], bar: float | None = None
               ) -> Candidate | None:
        """Pick the day's thesis. Returns None if nothing clears the bar —
        and a FLAT day is a legitimate, common outcome, not a failure."""
        if bar is None:
            bar = float(config.entry_conviction_bar())
        ok = [c for c in cands if abs(c.conv) >= bar]
        if not ok:
            self.state = FLAT
            log.info("DAY PLAN: no candidate cleared the %.2f bar at the "
                     "commit instant — FLAT for the session. A flat day is "
                     "an outcome, not a miss.", bar)
            self._persist()
            return None
        best = max(ok, key=lambda c: c.score)
        self.thesis = best
        self.state = COMMITTED
        self.committed_ts = time.time()
        log.info("DAY PLAN COMMIT: %s %s conv %+.3f (score %.3f, news "
                 "%+.2f) — the session's ONE thesis. Runner-up: %s",
                 best.index, best.symbol or "(pending)", best.conv,
                 best.score, best.news_score,
                 ", ".join(f"{c.index} {c.conv:+.2f}"
                           for c in sorted(ok, key=lambda z: -z.score)[1:3])
                 or "none")
        self._persist()
        return best

    def review(self, live_conv: float, unrealized_r: float = 0.0
               ) -> tuple[bool, str]:
        """Mid-session: is the reason we entered still true?

        The exit stack ratchets, trails and consults TrapShield, but it
        never re-asks the ENTRY question. On 2026-08-11 four NIFTY puts
        were stopped out at -Rs254 to -Rs270 each while the thesis had
        already stopped being supported; a reversal check closes that
        position on evidence instead of waiting for a stop that may sit
        far away.

        Returns (should_close, reason).
        """
        self.state = REVIEWED
        self.reviewed_ts = time.time()
        if self.thesis is None:
            return False, ""
        entry_sign = 1.0 if self.thesis.conv > 0 else -1.0
        live_sign = 1.0 if live_conv > 0 else -1.0
        rev_bar = float(getattr(config, "DAYPLAN_REVERSAL_CONV", 0.40))
        if live_sign != entry_sign and abs(live_conv) >= rev_bar:
            self.review_verdict = "REVERSED"
            log.info("DAY PLAN REVIEW: thesis REVERSED — entered on conv "
                     "%+.3f, now %+.3f (|.| >= %.2f). Closing on the "
                     "reason, not waiting for the stop.",
                     self.thesis.conv, live_conv, rev_bar)
            self._persist()
            return True, "DAYPLAN_THESIS_REVERSED"
        decay = float(getattr(config, "DAYPLAN_DECAY_CONV", 0.15))
        if abs(live_conv) < decay:
            self.review_verdict = "DECAYED"
            log.info("DAY PLAN REVIEW: thesis DECAYED — conv %+.3f -> "
                     "%+.3f (< %.2f). The edge that justified the position "
                     "is gone; holding is now a theta bet.",
                     self.thesis.conv, live_conv, decay)
            self._persist()
            return True, "DAYPLAN_THESIS_DECAYED"
        self.review_verdict = "INTACT"
        log.info("DAY PLAN REVIEW: thesis INTACT (conv %+.3f -> %+.3f, "
                 "unrealised %+.2fR) — holding to %s",
                 self.thesis.conv, live_conv, unrealized_r,
                 getattr(config, "DAYPLAN_EXIT_HM", "15:05"))
        self._persist()
        return False, ""

    def note_close(self, reason: str) -> None:
        self.state = DONE
        self.closed_ts = time.time()
        self.close_reason = str(reason)
        self._persist()

    # -------------------------------------------------------------- state
    def _path(self):
        return config.STATE_DIR / "day_plan.json"

    def _persist(self) -> None:
        try:
            p = self._path()
            p.parent.mkdir(parents=True, exist_ok=True)
            body = {"day": self.day, "state": self.state,
                    "config_hash": config.CONFIG_HASH,
                    "thesis": self.thesis.as_dict() if self.thesis else None,
                    "committed_ts": self.committed_ts,
                    "reviewed_ts": self.reviewed_ts,
                    "review_verdict": self.review_verdict,
                    "closed_ts": self.closed_ts,
                    "close_reason": self.close_reason,
                    "observed": self.observed[-10:]}
            tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.json")
            tmp.write_text(json.dumps(body, indent=1, default=float),
                           encoding="utf-8")
            os.replace(tmp, p)
        except Exception as e:                             # noqa: BLE001
            log.debug("day plan persist failed (%s)", e)

    @classmethod
    def load_or_new(cls) -> "DayPlan":
        """Restore today's plan across a restart. A plan that reset on
        restart would let the book commit a SECOND thesis — the exact
        failure this module exists to prevent."""
        today = dt.date.today().isoformat()
        try:
            p = config.STATE_DIR / "day_plan.json"
            if p.exists():
                b = json.loads(p.read_text(encoding="utf-8"))
                if (b.get("day") == today
                        and b.get("config_hash") == config.CONFIG_HASH):
                    dp = cls(day=today, state=str(b.get("state") or OBSERVE))
                    if b.get("thesis"):
                        t = dict(b["thesis"])
                        t.pop("score", None)
                        dp.thesis = Candidate(**t)
                    dp.committed_ts = float(b.get("committed_ts") or 0)
                    dp.reviewed_ts = float(b.get("reviewed_ts") or 0)
                    dp.review_verdict = str(b.get("review_verdict") or "")
                    dp.observed = list(b.get("observed") or [])
                    log.info("day plan restored: state=%s thesis=%s", dp.state,
                             dp.thesis.symbol if dp.thesis else "none")
                    return dp
        except Exception as e:                             # noqa: BLE001
            log.warning("day plan restore failed (%s) — starting fresh", e)
        return cls(day=today)