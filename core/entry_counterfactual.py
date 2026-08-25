"""
ENTRY COUNTERFACTUAL — what the BOOK would have done, not what a signal was worth
=================================================================================
nightly_forge_v9._shadow_trade grades a blocked signal and says so plainly in
its own docstring:

    "NO governor, NO affordability: the question is the signal's worth,
     not the account's size"

That is a defensible question. It is not the question anyone can act on, and
treating its answer as evidence about the entry bar is the single most
expensive mistake available on this side of the system. Four reasons:

  1. THE NUMBER IS UNCAPITALISABLE. The live book is
     MAX_CONCURRENT_POSITIONS=1 across ALL indices, with a 60-minute theta
     guillotine and a 180s cooldown. Entry window 09:15→15:05 is 21 000s;
     one slot costs up to 3 780s. THE DAY HOLDS AT MOST FIVE TRADES. A
     counterfactual that grades 400 blocked signals per gate and sums them
     is describing a book that cannot exist at any bar, on any capital.
  2. THE STATISTICS ARE INFLATED BY CONSTRUCTION. Signals arrive at 1 Hz.
     Second t and second t+1 are near-identical convictions on near-identical
     paths, and their "trades" overlap almost completely. Pooling them and
     testing gives t-stats inflated by autocorrelation — the exact artifact
     the July diagnostic chain proved was responsible for the apparent
     directional edge. Independence must be earned by the SIMULATION
     (one book, sequential, non-overlapping), not assumed by the estimator.
  3. IT ANSWERS ABOUT A DIFFERENT SYSTEM. _shadow_trade exits on
     _shaped_barriers first-touch. The live stack is the v9.7.1 ratchet with
     TrapShield, dwell-confirmed stops and a dead-trade cut. Comparing a
     first-touch counterfactual to live P&L confounds the entry question
     with an exit change.
  4. IT CANNOT ANSWER THE QUESTION ASKED. The near-miss band is
     |conv| ≥ bar − CF_NEAR_MISS, a 0.05-wide sliver. "Where should the bar
     be?" needs the whole conviction range, swept.

WHAT THIS MODULE DOES INSTEAD
-----------------------------
It replays the session as a POLICY, K times in one pass — once per candidate
bar — with each candidate running its own book under the real constraints:
one position, the throttle, the cooldown, the entry curfew, affordability
against TRADING_CAPITAL, and ONE shared exit rule so the comparison isolates
the entry decision and nothing else.

Because every bar sees the identical signal stream in the identical order,
the comparison is exactly paired at the day level, and each bar's trades are
non-overlapping by construction. That is what makes the day-clustered
statistics in tools/entry_bar_study.py honest rather than decorative.

THE REAL QUESTION, RESTATED
---------------------------
With five slots and thousands of candidates, "should the bar be lower" is the
wrong frame. The book is a SECRETARY PROBLEM: take this 0.56, or hold the
slot for a possible 0.80 that may never come? A lower bar does not buy more
trades — it buys EARLIER, WEAKER ones and spends the slot. So this module
also simulates two reference policies that bracket any threshold rule:

  * ORACLE_TOPK  — perfect hindsight: the k highest-conviction signals of the
    day, subject to the same book. The ceiling any bar could reach.
  * RANDOM_SLOT  — fills slots from the eligible pool at random, seeded. The
    floor that says whether selection is doing anything at all.

A bar that cannot beat RANDOM_SLOT is not selecting; a bar near ORACLE_TOPK
has no headroom left and the bar is not where the problem is.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

import config
from core.exit_policies import PolicySpec, PolicyState, TradeCtx, step, pnl_of

log = logging.getLogger("entry_counterfactual")

ORACLE = "ORACLE_TOPK"
RANDOM = "RANDOM_SLOT"


@dataclass
class Signal:
    """One second's candidate, exactly as the replayer produced it."""
    t: int
    index: str
    conv: float
    wp: float = 0.0
    spot: float = 0.0
    ts: float = 0.0
    dte: float = 9.0
    blocked_by: str = ""          # why LIVE refused it ("" = live took it)

    @property
    def direction(self) -> str:
        return "CE" if self.conv > 0 else "PE"

    @property
    def strength(self) -> float:
        return abs(float(self.conv))


@dataclass
class CFTrade:
    index: str
    token: int
    symbol: str
    entry_t: int
    exit_t: int
    entry_px: float
    exit_px: float
    qty: int
    conv: float
    pnl: float
    reason: str
    stale: bool = False
    coverage: float = 1.0

    @property
    def held_s(self) -> int:
        return max(self.exit_t - self.entry_t, 1)


@dataclass
class Book:
    """One candidate policy's book. The live constraints, honestly applied."""
    name: str
    bar: float
    # v9.9.29: an ARM may carry extra gates (day plan, range regime). They
    # are consulted at the same point the live gate chain would consult
    # them — after the bar, before affordability — so an A/B measures the
    # gate and not a different ordering.
    day_plan: object = None
    range_fn: object = None
    blocked_by_gate: dict = field(default_factory=dict)
    pos: dict | None = None
    last_exit_t: int = -10 ** 9
    last_try_t: int = -10 ** 9
    trades: list[CFTrade] = field(default_factory=list)
    skipped_busy: int = 0          # eligible but the slot was full
    skipped_cooldown: int = 0
    skipped_afford: int = 0
    skipped_nochain: int = 0
    offered: int = 0

    def free(self, t: int, cooldown_s: int, throttle_s: float) -> bool:
        if self.pos is not None:
            return False
        if t - self.last_exit_t < cooldown_s:
            return False
        if t - self.last_try_t < throttle_s:
            return False
        return True

    def occupancy(self, window_s: int) -> float:
        return min(sum(x.held_s for x in self.trades) / max(window_s, 1), 1.0)

    def pnl(self) -> float:
        return float(sum(x.pnl for x in self.trades))


class BarSweep:
    """Replay one session under K candidate bars simultaneously.

    Callers supply two closures, so this module never has to know whether it
    is driving the nightly forge's dense arrays or a test fixture:

        chain_fn(index, spot, direction, t) -> [{token, symbol, lot, strike,
                                                 bid, ask, fresh}, ...]
            the affordability-ordered rung walk, freshest first.
        quote_fn(token, t) -> (bid, ask, fresh)
            the mark at second t. fresh=False means the feed is dead there
            and NO exit rule may trigger on it (see core.exit_policies).
    """

    def __init__(self, bars, chain_fn, quote_fn, *,
                 capital: float | None = None,
                 exit_policy: str = "as_live",
                 curfew_t: int | None = None,
                 session_n: int | None = None,
                 costs_fn=None, seed: int = 0,
                 include_reference: bool = True):
        self.chain_fn = chain_fn
        self.quote_fn = quote_fn
        self.capital = float(capital if capital is not None
                             else getattr(config, "TRADING_CAPITAL", 60000.0))
        self.cooldown = int(getattr(config, "COOLDOWN_S", 180))
        self.throttle = float(getattr(config, "ENTRY_ATTEMPT_THROTTLE_S", 5.0))
        self.hold_s = int(float(getattr(config, "MAX_HOLD_MINUTES", 60)) * 60)
        self.session_n = int(session_n or 22500)
        self.curfew_t = int(curfew_t if curfew_t is not None
                            else self.session_n)
        self.rng = np.random.default_rng(seed)
        if costs_fn is None:
            try:
                from core.execution_engine import round_trip_costs
                costs_fn = round_trip_costs
            except Exception:                              # noqa: BLE001
                costs_fn = None
        self.costs = costs_fn
        self.spec = _live_equivalent_spec(exit_policy)

        self.books: dict[str, Book] = {
            f"bar_{b:.2f}": Book(name=f"bar_{b:.2f}", bar=float(b))
            for b in bars}
        self._arm_mode = False
        self.reference = include_reference
        if include_reference:
            self.books[RANDOM] = Book(name=RANDOM, bar=0.0)
            # ORACLE needs the whole day before it can choose, so it is not
            # a streaming book — it is resolved in finish().
        self._all_signals: list[Signal] = []

    def add_arm(self, name: str, bar: float, day_plan=None, range_fn=None
                ) -> Book:
        """Register an A/B arm alongside (or instead of) the bar grid.

        Every arm sees the IDENTICAL signal stream in the identical order,
        so the comparison is exactly paired at the day level and no arm can
        benefit from a different sample.
        """
        bk = Book(name=name, bar=float(bar), day_plan=day_plan,
                  range_fn=range_fn)
        self.books[name] = bk
        return bk

    def clear_bar_grid(self) -> None:
        """Drop the sweep books — an A/B study wants only its own arms."""
        for k in [k for k in self.books if k.startswith("bar_")]:
            self.books.pop(k, None)

    # ---------------------------------------------------------------- feed
    def offer(self, sig: Signal) -> None:
        """Present one second's candidate to every book."""
        self._all_signals.append(sig)
        for name, bk in self.books.items():
            if name == ORACLE:
                continue
            eligible = (sig.strength >= bk.bar if name != RANDOM
                        else self.rng.random() < _RANDOM_RATE)
            if not eligible:
                continue
            bk.offered += 1
            self._try_enter(bk, sig)

    def _try_enter(self, bk: Book, sig: Signal) -> None:
        if sig.t >= self.curfew_t:
            return
        if bk.pos is not None:
            bk.skipped_busy += 1
            return
        if sig.t - bk.last_exit_t < self.cooldown:
            bk.skipped_cooldown += 1
            return
        if sig.t - bk.last_try_t < self.throttle:
            return
        # ---- arm-specific gates, at the live position in the chain
        if bk.range_fn is not None:
            ok, why = bk.range_fn(sig.t, sig.index)
            if not ok:
                k = why.split(":")[0] or "range_bound"
                bk.blocked_by_gate[k] = bk.blocked_by_gate.get(k, 0) + 1
                return
        if bk.day_plan is not None:
            ok, why = bk.day_plan.may_enter_t(sig.t)
            if not ok:
                k = why.split("—")[0].strip()[:48] or "day_plan"
                bk.blocked_by_gate[k] = bk.blocked_by_gate.get(k, 0) + 1
                return
        bk.last_try_t = sig.t
        rung = self._affordable(sig)
        if rung is None:
            bk.skipped_nochain += 1
            return
        entry = float(rung["ask"])
        qty = int(rung["lot"])
        if entry * qty > self.capital:
            bk.skipped_afford += 1
            return
        ctx = TradeCtx(entry=entry, qty=qty, side=+1,
                       hold_budget_s=self.hold_s,
                       session_end_t=max(self.session_n - sig.t - 1, 1))
        bk.pos = {"sig": sig, "ctx": ctx, "rung": rung,
                  "st": PolicyState.start(ctx), "t0": sig.t,
                  "marks": 0, "fresh": 0}
        if bk.day_plan is not None:
            bk.day_plan.commit_t(sig.t, sig.conv, sig.index)

    def _affordable(self, sig: Signal):
        """First rung that is quoted two-sided, inside the spread cap, and
        that the capital can actually hold. The live first_affordable walk."""
        try:
            rows = self.chain_fn(sig.index, sig.spot, sig.direction, sig.t)
        except Exception as e:                             # noqa: BLE001
            log.debug("chain_fn failed (%s)", e)
            return None
        cap = float(getattr(config, "MAX_ENTRY_SPREAD_PCT", 0.10))
        for r in rows or []:
            b, a = float(r.get("bid") or 0), float(r.get("ask") or 0)
            if not (b > 0 and a > 0) or not r.get("fresh", True):
                continue
            mid = (a + b) / 2.0
            if mid <= 0 or (a - b) / max(mid, 0.05) > cap:
                continue
            if a * int(r.get("lot") or 0) > self.capital:
                continue
            return r
        return None

    # ---------------------------------------------------------------- mark
    def mark(self, t: int) -> None:
        """Advance every open book position one second."""
        for name, bk in self.books.items():
            if name == ORACLE or bk.pos is None:
                continue
            self._advance(bk, t)

    def _advance(self, bk: Book, t: int) -> None:
        p = bk.pos
        # DAY PLAN: the mid-session review and the hard session flat both
        # close a live position. Without these the arm would measure the
        # entry restriction alone and credit the day plan with an exit
        # discipline it never applied — the two are inseparable in the
        # real design, so they must be inseparable here.
        if bk.day_plan is not None:
            close, why = bk.day_plan.tick_t(t, p["sig"].conv)
            if close:
                st = p["st"]
                st.closed = True
                st.exit_px = st.last_fresh_px or p["ctx"].entry
                st.exit_t = t - p["t0"]
                st.exit_reason = why
                self._close(bk, t)
                return
        tok = int(p["rung"]["token"])
        bid, _ask, fresh = self.quote_fn(tok, t)
        p["marks"] += 1
        if fresh:
            p["fresh"] += 1
        px = float(bid) if (fresh and bid and bid > 0) else float("nan")
        st = step(p["st"], self.spec, p["ctx"], px, t - p["t0"], self.costs)
        if st.closed:
            self._close(bk, t)

    def _close(self, bk: Book, t: int) -> None:
        p = bk.pos
        st, ctx, sig = p["st"], p["ctx"], p["sig"]
        cov = (p["fresh"] / p["marks"]) if p["marks"] else 0.0
        bk.trades.append(CFTrade(
            index=sig.index, token=int(p["rung"]["token"]),
            symbol=str(p["rung"].get("symbol") or p["rung"]["token"]),
            entry_t=p["t0"], exit_t=p["t0"] + st.exit_t,
            entry_px=ctx.entry, exit_px=st.exit_px, qty=ctx.qty,
            conv=sig.conv, pnl=pnl_of(ctx, st.exit_px, self.costs),
            reason=st.exit_reason, stale=st.stale_exit, coverage=cov))
        bk.pos = None
        bk.last_exit_t = t

    # -------------------------------------------------------------- finish
    def finish(self, t_end: int | None = None) -> dict[str, Book]:
        """Close anything still open at the bell and resolve the oracle."""
        t_end = int(t_end if t_end is not None else self.session_n - 1)
        for name, bk in self.books.items():
            if name == ORACLE or bk.pos is None:
                continue
            p = bk.pos
            st = p["st"]
            st.closed = True
            st.exit_px = st.last_fresh_px or p["ctx"].entry
            st.exit_t = t_end - p["t0"]
            st.exit_reason = "SESSION_END"
            st.stale_exit = st.last_fresh_t < (t_end - p["t0"] - 5)
            self._close(bk, t_end)
        if self.reference:
            self.books[ORACLE] = self._run_oracle(t_end)
        return self.books

    def _run_oracle(self, t_end: int) -> Book:
        """Perfect-hindsight ceiling: greedily fill slots with the highest-
        conviction signals of the day, still honouring one position, the
        cooldown and the curfew. This is NOT a strategy — it is the number
        no causal bar can exceed, so a bar close to it has no headroom."""
        bk = Book(name=ORACLE, bar=0.0)
        order = sorted(self._all_signals, key=lambda s: -s.strength)
        taken: list[tuple[int, int]] = []
        for sig in order:
            if sig.t >= self.curfew_t:
                continue
            if any(not (sig.t + self.hold_s + self.cooldown <= a
                        or sig.t >= b + self.cooldown) for a, b in taken):
                continue
            rung = self._affordable(sig)
            if rung is None:
                continue
            entry, qty = float(rung["ask"]), int(rung["lot"])
            if entry * qty > self.capital:
                continue
            ctx = TradeCtx(entry=entry, qty=qty, side=+1,
                           hold_budget_s=self.hold_s,
                           session_end_t=max(self.session_n - sig.t - 1, 1))
            st = PolicyState.start(ctx)
            marks = fresh_n = 0
            tok = int(rung["token"])
            for t in range(sig.t, min(t_end + 1, self.session_n)):
                b, _a, fr = self.quote_fn(tok, t)
                marks += 1
                fresh_n += int(bool(fr))
                px = float(b) if (fr and b and b > 0) else float("nan")
                step(st, self.spec, ctx, px, t - sig.t, self.costs)
                if st.closed:
                    break
            if not st.closed:
                st.closed = True
                st.exit_px = st.last_fresh_px or ctx.entry
                st.exit_t = t_end - sig.t
                st.exit_reason = "SESSION_END"
            bk.trades.append(CFTrade(
                index=sig.index, token=tok,
                symbol=str(rung.get("symbol") or tok), entry_t=sig.t,
                exit_t=sig.t + st.exit_t, entry_px=ctx.entry,
                exit_px=st.exit_px, qty=ctx.qty, conv=sig.conv,
                pnl=pnl_of(ctx, st.exit_px, self.costs),
                reason=st.exit_reason, stale=st.stale_exit,
                coverage=(fresh_n / marks) if marks else 0.0))
            taken.append((sig.t, sig.t + st.exit_t))
            if len(taken) >= _max_slots(self.curfew_t, self.hold_s,
                                        self.cooldown):
                break
        return bk

    # --------------------------------------------------------------- report
    def summary(self, window_s: int | None = None) -> dict:
        w = int(window_s or self.curfew_t)
        out = {}
        for name, bk in self.books.items():
            trades = bk.trades
            out[name] = {
                "bar": bk.bar, "n_trades": len(trades),
                "pnl": round(bk.pnl(), 2),
                "wins": sum(1 for x in trades if x.pnl > 0),
                "mean_conv": (round(float(np.mean([abs(x.conv)
                                                   for x in trades])), 3)
                              if trades else None),
                "occupancy": round(bk.occupancy(w), 3),
                "offered": bk.offered,
                "skipped_busy": bk.skipped_busy,
                "skipped_cooldown": bk.skipped_cooldown,
                "skipped_afford": bk.skipped_afford,
                "skipped_nochain": bk.skipped_nochain,
                "blocked_by_gate": dict(bk.blocked_by_gate),
                "thin_coverage": sum(1 for x in trades if x.coverage <
                                     float(getattr(config,
                                                   "SHADOW_MIN_COVERAGE",
                                                   0.6))),
            }
        return out


_RANDOM_RATE = 0.002    # the random reference offers ~1 candidate per 500s


def _max_slots(curfew_t: int, hold_s: int, cooldown_s: int) -> int:
    """The hard ceiling on trades per session. This is the number that makes
    'lower the bar to get more trades' false: it does not depend on the bar
    at all.

    v9.9.17 — count entry START opportunities, not COMPLETED slots. An entry
    is legal at any t <= curfew; the slot it occupies may run past curfew and
    be closed by the guillotine. With a 21000s window and a 3780s slot the
    starts are 0, 3780, 7560, 11340, 15120, 18900 — SIX, the last one 2100s
    inside the window. The old `win // slot` returned 5 and the 2026-08-21
    and 2026-08-23 sweeps then printed "AT MOST 5 TRADE(S) PER SESSION, at
    ANY bar" as their headline while their own books took 6 on 92 book-days.
    The claim the tool exists to make was contradicted by the tool's own
    output on the same screen.

    Only the printed ceiling was wrong; the books were right to allow the
    sixth entry, so no sweep result changes. The reframing argument is
    unaffected — 6 is still a hard bar-independent cap.
    """
    return max(int(curfew_t // max(hold_s + cooldown_s, 1)), 0) + 1


def _live_equivalent_spec(name: str) -> PolicySpec:
    """ONE exit rule for every candidate bar.

    The point of holding this fixed is causal, not convenient: if bars were
    compared under different exits, a bar difference and an exit difference
    would be perfectly confounded and neither could be attributed. The
    default mirrors the live single-leg stack's shape — initial stop, theta
    guillotine, disaster floor — through the same core.exit_policies engine
    the live shadow book steps.
    """
    if name and name != "as_live":
        for s in PolicySpec.family():
            if s.name == name:
                return s
        raise ValueError(f"unknown exit policy '{name}' — it must come from "
                         f"config.SHADOW_POLICIES so the comparison stays "
                         f"pre-registered")
    return PolicySpec(name="as_live", hold_mult=1.0, target_r=None,
                      trail_pct=None, lock_at=None)


def bar_grid() -> list[float]:
    """The pre-registered sweep. Fixed here, not chosen after seeing a
    result — the whole point of a registry."""
    g = getattr(config, "ENTRY_BAR_GRID", None)
    if g:
        return [float(x) for x in g]
    return [round(x, 2) for x in np.arange(0.20, 0.86, 0.05)]


def capacity_note(curfew_t: int | None = None) -> dict:
    """The arithmetic that reframes the question. Callers print this before
    any sweep result so nobody reads a bar difference as a volume story."""
    hold = int(float(getattr(config, "MAX_HOLD_MINUTES", 60)) * 60)
    cd = int(getattr(config, "COOLDOWN_S", 180))
    win = int(curfew_t or 21000)
    return {"entry_window_s": win, "hold_s": hold, "cooldown_s": cd,
            "slot_s": hold + cd, "max_trades_per_session":
                _max_slots(win, hold, cd),
            "max_concurrent": int(getattr(config,
                                          "MAX_CONCURRENT_POSITIONS", 1))}