"""
APEX OMNI v9 — RISK GOVERNOR (audit §2 leap: the constitution gets a court)
===========================================================================
v8 *defined* MAX_DAILY_DRAWDOWN and never checked it anywhere in the audited
code. v9 routes every entry through exactly one object. Nothing else in the
codebase is allowed to place an order without a permission slip from here.

What it enforces:
  * Capital truth      — works off config.TRADING_CAPITAL; in live mode it
                         queries kite.margins() and uses the SMALLER number.
  * Affordability      — "only take a trade you can HOLD": premium×lot must
                         fit the Kelly budget AND the worst case at the
                         disaster floor must cost ≤ MAX_LOSS_PER_TRADE_PCT of
                         capital AND cash must cover premium + costs buffer.
                         It can also walk DOWN a strike hierarchy to the
                         first leg the account can genuinely carry.
  * Daily drawdown     — realized, after costs, vs MAX_DAILY_DRAWDOWN_PCT.
  * Kill switches      — order-reject storm, stale-data feed, manual halt.
  * Tempo              — cooldowns, post-loss directional lockouts, entry
                         curfew, concurrent-position cap, warm-up ticks.
"""
from __future__ import annotations
import logging
import math
import time
from dataclasses import dataclass

import config

log = logging.getLogger("risk")


@dataclass
class TradePermit:
    ok: bool
    reason: str
    qty: int = 0
    lots: int = 0
    budget: float = 0.0


class RiskGovernor:
    def __init__(self, capital: float | None = None, kite=None,
                 book: str = "equity", persist: bool = False):
        # persist=True is passed ONLY by the live entrypoints (apex_main,
        # apex_commodity_main). Simulation, harnesses and validators construct
        # governors freely and must NEVER couple to the live day ledger — the
        # regression gate caught exactly that contamination when persistence
        # was ambient in the constructor.
        self.start_capital = float(capital if capital is not None
                                   else config.TRADING_CAPITAL)
        self.kite = kite
        self.book = book
        self._persist = bool(persist) and \
            bool(getattr(config, "RISK_STATE_PERSIST", True))
        self.realized_pnl = 0.0
        self.deployed = 0.0
        self.open_positions = 0
        self.halted = False
        self.halt_reason = ""
        self.reject_count = 0
        self.ticks_seen = 0
        self.last_exit_ts = 0.0
        self.lockout_until = 0.0
        self.lockout_direction = None
        # v9.7.1 fast-lane loss-streak breaker: N consecutive FAST-LANE losses
        # suspends the FAST LANE for the rest of the day — it does NOT halt the
        # book. The normal 45-min path keeps running on its own risk controls
        # (drawdown halt, cascade lockout). This is a scoped anti-overtrading
        # guard: overtrading is a fast-lane risk, so the breaker governs only
        # the fast lane. Only fast-lane losses count; a fast-lane win (or any
        # non-losing fast-lane exit) resets the streak. Normal-path trades never
        # touch this counter.
        self.fast_consec_losses = 0
        self.fast_lane_suspended = False
        # v9.7.1 AUDIT F1: the daily ledger must SURVIVE a crash-restart. The
        # supervisor's backoff restart used to hand a book that was 0.3% from
        # the drawdown halt a FRESH allowance (and cleared halts, lockouts and
        # the fast-lane streak). Day-scoped state now persists per book and
        # reloads on construction; positions/warm-up deliberately do NOT (a
        # fresh process holds no positions and must re-settle physics).
        self._load_day_state()

    # ---------------------------------------------- day-state persistence
    def _state_path(self):
        import datetime as _dt
        return (config.STATE_DIR /
                f"risk_day_{self.book}_{_dt.date.today()}.json")

    def _save_day_state(self):
        if not self._persist:
            return
        try:
            import json as _json
            p = self._state_path()
            tmp = p.with_suffix(".tmp")
            tmp.write_text(_json.dumps({
                "realized_pnl": self.realized_pnl,
                "fast_consec_losses": self.fast_consec_losses,
                "fast_lane_suspended": self.fast_lane_suspended,
                "halted": self.halted, "halt_reason": self.halt_reason,
                "lockout_until": self.lockout_until,
                "lockout_direction": self.lockout_direction,
                "reject_count": self.reject_count}))
            tmp.replace(p)
        except Exception as e:                              # noqa: BLE001
            log.warning("risk day-state save failed: %s", e)

    def _load_day_state(self):
        if not self._persist:
            return
        try:
            import json as _json
            p = self._state_path()
            if not p.exists():
                return
            d = _json.loads(p.read_text())
            self.realized_pnl = float(d.get("realized_pnl", 0.0))
            self.fast_consec_losses = int(d.get("fast_consec_losses", 0))
            self.fast_lane_suspended = bool(d.get("fast_lane_suspended", False))
            self.lockout_until = float(d.get("lockout_until", 0.0))
            self.lockout_direction = d.get("lockout_direction")
            self.reject_count = int(d.get("reject_count", 0))
            if d.get("halted"):
                self.halted = True
                self.halt_reason = d.get("halt_reason", "restored halt")
            if self.realized_pnl or self.halted:
                log.warning("day risk-state RESTORED (%s): realized ₹%+.0f, "
                            "halted=%s, fast_streak=%d", self.book,
                            self.realized_pnl, self.halted,
                            self.fast_consec_losses)
        except Exception as e:                              # noqa: BLE001
            log.warning("risk day-state load failed: %s", e)

    # ------------------------------------------------------------ capital
    def available_cash(self) -> float:
        cash = self.start_capital + self.realized_pnl - self.deployed
        if self.kite is not None and config.live_fire_armed():
            try:                          # live: never believe we have more
                live = float(self.kite.margins("equity")["available"]["cash"])
                cash = min(cash, live)    # than the broker actually shows
            except Exception as e:        # noqa: BLE001
                log.warning("margins() failed (%s) — using local cash", e)
        return max(cash, 0.0)

    def equity(self) -> float:
        return self.start_capital + self.realized_pnl

    # ------------------------------------------------------------ switches
    def kill(self, reason: str):
        if not self.halted:
            log.critical("🛑 TRADING HALTED: %s", reason)
        self.halted = True
        self.halt_reason = reason
        self._save_day_state()

    def register_reject(self):
        self.reject_count += 1
        if self.reject_count >= config.MAX_ORDER_REJECTS:
            self.kill(f"order-reject storm ({self.reject_count} rejects)")

    def on_tick(self):
        self.ticks_seen += 1

    # ------------------------------------------------------------ outcomes
    def register_entry(self, premium_outlay: float):
        self.deployed += premium_outlay
        self.open_positions += 1

    def register_exit(self, premium_outlay: float, pnl_after_costs: float,
                      direction: str, ts: float | None = None,
                      fast_lane: bool = False):
        ts = ts or time.time()
        self.deployed = max(self.deployed - premium_outlay, 0.0)
        self.open_positions = max(self.open_positions - 1, 0)
        self.realized_pnl += pnl_after_costs
        self.last_exit_ts = ts
        if pnl_after_costs < 0:
            self.lockout_until = ts + config.DIRECTION_LOCKOUT_S
            self.lockout_direction = direction
            # fast-lane loss-streak breaker — counts ONLY fast-lane losses and
            # suspends ONLY the fast lane (the 45-min path keeps trading).
            if fast_lane:
                self.fast_consec_losses += 1
                _streak_max = int(getattr(config, "LOSS_STREAK_HALT", 3))
                if _streak_max > 0 and self.fast_consec_losses >= _streak_max:
                    if not self.fast_lane_suspended:
                        log.warning("⏸ FAST LANE SUSPENDED for the day "
                                    "(%d consecutive fast-lane losses) — the "
                                    "normal 45-min path continues",
                                    self.fast_consec_losses)
                    self.fast_lane_suspended = True
        elif fast_lane:
            self.fast_consec_losses = 0   # a fast-lane win resets the streak
        self._save_day_state()
        dd = -self.realized_pnl / self.start_capital
        if dd >= config.MAX_DAILY_DRAWDOWN_PCT:
            self.kill(f"daily drawdown {dd:.1%} ≥ "
                      f"{config.MAX_DAILY_DRAWDOWN_PCT:.0%} limit")

    # ------------------------------------------------------------ the gate
    def request_entry(self, *, direction: str, premium: float, lot: int,
                      win_prob: float, sl_pct: float, tp_pct: float,
                      data_age_s: float, now_hm: str,
                      ts: float | None = None, symbol: str | None = None,
                      exchange: str | None = None, price: float | None = None,
                      ann_vol: float | None = None,
                      lockout_bypass: bool = False,
                      curfew_hm: str | None = None) -> TradePermit:
        ts = ts or time.time()
        if self.halted:
            return TradePermit(False, f"halted: {self.halt_reason}")
        if self.ticks_seen < config.MIN_TICKS_BEFORE_TRADING:
            return TradePermit(False, "warm-up: physics not settled yet")
        if data_age_s > config.DATA_STALE_BLOCK_S:
            return TradePermit(False, f"stale feed ({data_age_s:.1f}s)")
        _curfew = curfew_hm or config.NO_ENTRY_AFTER   # AUDIT F2: the equity
        # curfew silently killed every commodity EVENING entry ("20:00" ≥
        # "14:45"); each book now passes the curfew of ITS OWN session.
        if now_hm >= _curfew:
            return TradePermit(False, f"entry curfew after {_curfew}")
        if self.open_positions >= config.MAX_CONCURRENT_POSITIONS:
            return TradePermit(False, "max concurrent positions")
        if ts - self.last_exit_ts < config.COOLDOWN_S:
            return TradePermit(False, "cooldown")
        if (ts < self.lockout_until and direction == self.lockout_direction
                and not lockout_bypass):
            # v9.7.1: lockout_bypass is granted by core/cascade_exit.SmartLockout
            # ONLY for a STRONGER, still-aligned cascade re-trigger (trend
            # continuation, not revenge). Everything else is still locked out.
            left = int(self.lockout_until - ts)
            return TradePermit(False, f"post-loss {direction} lockout ({left}s)")

        # half-Kelly budget on CALIBRATED win prob (no invented (|a|+1)/2)
        b = max(tp_pct, 1e-3) / max(sl_pct, 1e-3)
        kelly = max(win_prob - (1 - win_prob) / b, 0.0)
        budget = min(self.equity() * config.MAX_KELLY_BUDGET_PCT,
                     self.equity() * kelly * config.KELLY_FRACTION)
        if ann_vol:        # volatility-managed sizing: hot vol → smaller bets
            budget *= float(min(max(config.VOL_TARGET_ANN / max(ann_vol, 1e-3),
                                    config.VOL_SCALE_MIN), 1.0))
        if budget <= 0:
            return TradePermit(False,
                f"Kelly says no edge (p={win_prob:.2f}, b={b:.2f})")

        outlay = premium * lot
        if outlay > budget:
            return TradePermit(False,
                f"₹{outlay:,.0f} exceeds Kelly budget ₹{budget:,.0f}", budget=budget)
        cash = self.available_cash()
        buffer = outlay * 0.02                 # heuristic cost buffer …
        if self.kite is not None and symbol and config.live_fire_armed():
            # AUDIT F4: read-only, but an HTTP call per entry ATTEMPT — the
            # affordability walker made N of them per decision-second in
            # PAPER too. Live keeps exact charges; paper uses the 2%% buffer.
            try:                               # … EXACT charges, paper & live
                om = self.kite.order_margins([{
                    "exchange": exchange, "tradingsymbol": symbol,
                    "transaction_type": "BUY", "variety": "regular",
                    "product": "MIS", "order_type": "LIMIT",
                    "quantity": lot, "price": float(price or premium)}])
                ch = (om[0].get("charges") or {}).get("total")
                if ch is not None:
                    buffer = float(ch)
            except Exception as e:             # noqa: BLE001
                log.debug("order_margins fallback (%s)", e)
        if outlay + buffer > cash:
            return TradePermit(False,
                f"₹{outlay:,.0f} + charges ₹{buffer:,.0f} exceeds cash "
                f"₹{cash:,.0f}")
        floor_pct = min(sl_pct * config.DISASTER_FLOOR_MULT,
                        config.ABS_DISASTER_PCT)
        worst = outlay * floor_pct
        if worst > self.equity() * config.MAX_LOSS_PER_TRADE_PCT:
            return TradePermit(False,
                f"disaster-floor loss ₹{worst:,.0f} > "
                f"{config.MAX_LOSS_PER_TRADE_PCT:.0%} of capital — cannot HOLD this")
        return TradePermit(True, "approved", qty=lot, lots=1, budget=budget)

    # --------------------------------------------- affordability walker
    def first_affordable(self, hierarchy: list[dict], **kw) -> tuple[dict | None,
                                                                     TradePermit]:
        """hierarchy: preferred-first list of {'leg','premium','lot',...}.
        Walks down (ATM → OTM → deeper) and returns the first leg the
        account can genuinely hold, with its permit."""
        last = TradePermit(False, "empty hierarchy")
        for leg in hierarchy:
            extras = {k: leg[k] for k in ("symbol", "exchange", "price")
                      if leg.get(k) is not None}
            p = self.request_entry(premium=leg["premium"], lot=leg["lot"],
                                   **extras, **kw)
            if p.ok:
                return leg, p
            last = p
        return None, last