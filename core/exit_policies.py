"""
EXIT POLICIES — one implementation, two drivers
================================================
The nightly study and the live shadow book must run the SAME exit code.
If the study has its own copy, its verdict describes a policy that never
ran; that is train/serve skew, and it is exactly what made meta_gbm.py
emit a constant in production while scoring well offline. So this module
owns the policy family, and both callers drive it:

    live    core.shadow_book        → step() once per mark, forever
    offline tools/trade_potential   → replay() over a stored path

replay() is literally a loop over step(). There is no vectorised fast
path, because a vectorised fast path is a second implementation.
tools/shadow_audit.py asserts the two drivers agree bit-for-bit.

WHAT A POLICY MAY ASSUME
------------------------
* SIDE-AWARE. `side` is +1 long premium, −1 short premium. Every
  comparison is written in FAVOURABLE/ADVERSE terms and flipped by side,
  so a short spread trails on the same code path as a long option
  instead of needing an inverted twin.
* THE FLOOR AND THE BELL ARE LAW. The disaster floor and the session
  hard-flat are constitution, not preference (config MAX_LOSS_PER_TRADE_
  PCT, the session close). Every policy obeys both; policies differ only
  in WHEN they choose to leave inside those bounds.
* A DEAD FEED IS NOT A FLAT PRICE. A NaN mark means the harvester
  stopped quoting this leg — pruned, or the feed dropped. A policy may
  NOT trigger on a NaN: you cannot sell into a quote that does not
  exist. If the feed never returns, the policy closes at the last mark
  it actually saw and reports reason DEAD_FEED with the timestamp. That
  outcome is flagged `stale_exit=True` and every study downstream must
  either exclude it or report it — silence here is the flat-line bug
  that made `hold_to_close` look free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict

import config

# ------------------------------------------------------------ the family
# name → kwargs for PolicySpec. Mirrors config.SHADOW_POLICIES; that tuple
# is the pre-registration and this dict is its implementation.
_FAMILY: dict[str, dict] = {
    "as_traded":     dict(),                       # baseline; never replayed
    "hold_to_close": dict(hold_mult=None),
    "hold_2x":       dict(hold_mult=2.0),
    "hold_3x":       dict(hold_mult=3.0),
    "trail_10":      dict(hold_mult=None, trail_pct=0.10),
    "trail_20":      dict(hold_mult=None, trail_pct=0.20),
    "trail_30":      dict(hold_mult=None, trail_pct=0.30),
    "target_1R":     dict(hold_mult=1.0, target_r=1.0),
    "target_2R":     dict(hold_mult=1.0, target_r=2.0),
    "target_3R":     dict(hold_mult=1.0, target_r=3.0),
    "lock_5pct":     dict(hold_mult=1.0, lock_at=0.05),
    "lock_10pct":    dict(hold_mult=1.0, lock_at=0.10),
    "trail20_hold2x": dict(hold_mult=2.0, trail_pct=0.20),
}


@dataclass(frozen=True)
class PolicySpec:
    name: str
    hold_mult: float | None = 1.0   # × hold budget; None = to the bell
    trail_pct: float | None = None  # giveback from peak that closes
    target_r: float | None = None   # take profit at N × risk
    lock_at: float | None = None    # once +N% shown, never exit below BE

    @staticmethod
    def family() -> list["PolicySpec"]:
        names = tuple(getattr(config, "SHADOW_POLICIES", tuple(_FAMILY)))
        out = []
        for n in names:
            if n == "as_traded":
                continue
            if n not in _FAMILY:
                raise ValueError(
                    f"config.SHADOW_POLICIES names '{n}' but "
                    f"core.exit_policies has no implementation — a "
                    f"pre-registered policy that cannot run is worse than "
                    f"no policy")
            out.append(PolicySpec(name=n, **_FAMILY[n]))
        return out


@dataclass
class TradeCtx:
    """Everything a policy needs about the trade it is managing."""
    entry: float
    qty: int
    side: int = +1                  # +1 long premium, −1 short premium
    stop_pct: float | None = None   # initial stop as a fraction of entry
    hold_budget_s: int = 3600       # the theta guillotine, in seconds
    session_end_t: int = 10 ** 9    # ticks after entry at which the bell rings
    floor_pct: float | None = None  # disaster floor as fraction of entry

    def __post_init__(self):
        if self.stop_pct is None:
            self.stop_pct = float(getattr(config, "BASE_SL_PCT", 0.20))
        if self.floor_pct is None:
            self.floor_pct = float(getattr(config, "MAX_LOSS_PER_TRADE_PCT",
                                           0.30))

    @property
    def r_unit(self) -> float:
        """Risk per unit — the distance to the initial stop."""
        return max(self.entry * float(self.stop_pct), 1e-6)

    def favourable(self, px: float) -> float:
        """Signed move in the direction that makes money."""
        return (px - self.entry) * self.side

    def floor_px(self) -> float:
        return self.entry * (1.0 - self.side * float(self.floor_pct))

    def breakeven_px(self, costs_fn=None) -> float:
        if costs_fn is None:
            return self.entry
        c = costs_fn(self.entry * self.qty, self.entry * self.qty)
        return self.entry + self.side * (c / max(self.qty, 1))


@dataclass
class PolicyState:
    """Mutable per-policy state. Serialisable — the live book persists it."""
    peak: float = 0.0               # best FAVOURABLE price seen
    peak_t: int = 0
    locked: bool = False
    closed: bool = False
    exit_px: float = 0.0
    exit_t: int = 0
    exit_reason: str = ""
    stale_exit: bool = False
    last_fresh_px: float = 0.0
    last_fresh_t: int = -1
    mfe_px: float = 0.0
    mae_px: float = 0.0
    marks: int = 0
    fresh_marks: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def start(ctx: TradeCtx) -> "PolicyState":
        return PolicyState(peak=ctx.entry, mfe_px=ctx.entry,
                           mae_px=ctx.entry, last_fresh_px=ctx.entry)


def step(st: PolicyState, spec: PolicySpec, ctx: TradeCtx,
         px: float, t: int, costs_fn=None) -> PolicyState:
    """Advance one policy by one mark. `t` = seconds since entry.

    Returns the same object (mutated) so the live book can hold one state
    per policy without reallocating every second.
    """
    if st.closed:
        return st
    st.marks += 1
    fresh = (px is not None and math.isfinite(px) and px > 0)

    # --- the bell is law, whatever the feed is doing
    bell = t >= ctx.session_end_t
    budget = (None if spec.hold_mult is None
              else int(ctx.hold_budget_s * spec.hold_mult))
    clock = budget is not None and t >= budget

    if not fresh:
        # A dead feed can trigger nothing. If the trade must end now, it
        # ends at the last price we actually saw — and says so.
        if bell or clock:
            _close(st, ctx, st.last_fresh_px, t,
                   "DEAD_FEED_BELL" if bell else "DEAD_FEED_CLOCK",
                   stale=True)
        return st

    st.fresh_marks += 1
    st.last_fresh_px, st.last_fresh_t = px, t
    if ctx.favourable(px) > ctx.favourable(st.mfe_px):
        st.mfe_px = px
    if ctx.favourable(px) < ctx.favourable(st.mae_px):
        st.mae_px = px
    if ctx.favourable(px) > ctx.favourable(st.peak):
        st.peak, st.peak_t = px, t

    # --- 1. disaster floor: constitution, checked before anything else
    floor = ctx.floor_px()
    if ctx.favourable(px) <= ctx.favourable(floor):
        return _close(st, ctx, px, t, "FLOOR")

    # --- 2. profit lock: once `lock_at` has been shown, never give back
    #        past breakeven again.
    if spec.lock_at is not None:
        shown = ctx.favourable(st.peak) / max(ctx.entry, 1e-9)
        if shown >= spec.lock_at:
            st.locked = True
        if st.locked:
            be = ctx.breakeven_px(costs_fn)
            if ctx.favourable(px) <= ctx.favourable(be):
                return _close(st, ctx, be, t, "PROFIT_LOCK")

    # --- 3. target
    if spec.target_r is not None:
        tgt = ctx.entry + ctx.side * spec.target_r * ctx.r_unit
        if ctx.favourable(px) >= ctx.favourable(tgt):
            return _close(st, ctx, tgt, t, "TARGET")

    # --- 4. peak-anchored trail. A trail is a STOP: it fires on giveback
    #        from the running peak whether or not the peak was in profit.
    if spec.trail_pct is not None and t >= 1:
        give = abs(st.peak) * spec.trail_pct
        level = st.peak - ctx.side * give
        if ctx.favourable(px) <= ctx.favourable(level):
            return _close(st, ctx, px, t, "TRAIL")

    # --- 5. the clock, then the bell
    if clock:
        return _close(st, ctx, px, t, "CLOCK")
    if bell:
        return _close(st, ctx, px, t, "BELL")
    return st


def _close(st: PolicyState, ctx: TradeCtx, px: float, t: int,
           reason: str, stale: bool = False) -> PolicyState:
    st.closed = True
    st.exit_px = float(px if (px and math.isfinite(px)) else ctx.entry)
    st.exit_t = int(t)
    st.exit_reason = reason
    st.stale_exit = bool(stale)
    return st


@dataclass
class Outcome:
    policy: str
    exit_px: float
    exit_t: int
    reason: str
    pnl: float
    stale_exit: bool
    coverage: float
    mfe_px: float
    mae_px: float
    peak_t: int
    state: PolicyState = field(repr=False, default=None)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("state", None)
        return d


def pnl_of(ctx: TradeCtx, exit_px: float, costs_fn=None) -> float:
    """Net ₹ for one round trip. Side-aware; identical cost stack for
    every policy, so a comparison isolates timing and nothing else."""
    gross = (exit_px - ctx.entry) * ctx.qty * ctx.side
    if costs_fn is None:
        return gross
    return gross - costs_fn(ctx.entry * ctx.qty, exit_px * ctx.qty)


def replay(spec: PolicySpec, ctx: TradeCtx, path, costs_fn=None,
           fresh_mask=None) -> Outcome:
    """Drive `step` across a stored path. Index 0 is the entry second.

    This is the offline driver. It is a loop over the live function on
    purpose — see the module docstring.
    """
    st = PolicyState.start(ctx)
    n = len(path)
    last_t = 0
    for t in range(n):
        px = float(path[t])
        if fresh_mask is not None and not bool(fresh_mask[t]):
            px = float("nan")
        last_t = t
        step(st, spec, ctx, px, t, costs_fn)
        if st.closed:
            break
    if not st.closed:
        # Path ran out before any rule fired: the session ended here.
        _close(st, ctx, st.last_fresh_px, last_t, "PATH_END",
               stale=(st.last_fresh_t < last_t))
    cov = (st.fresh_marks / st.marks) if st.marks else 0.0
    return Outcome(policy=spec.name, exit_px=st.exit_px, exit_t=st.exit_t,
                   reason=st.exit_reason,
                   pnl=pnl_of(ctx, st.exit_px, costs_fn),
                   stale_exit=st.stale_exit, coverage=cov,
                   mfe_px=st.mfe_px, mae_px=st.mae_px, peak_t=st.peak_t,
                   state=st)


def replay_family(ctx: TradeCtx, path, costs_fn=None, fresh_mask=None
                  ) -> dict[str, Outcome]:
    return {s.name: replay(s, ctx, path, costs_fn, fresh_mask)
            for s in PolicySpec.family()}