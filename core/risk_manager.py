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
    def __init__(self, capital: float | None = None, kite=None):
        self.start_capital = float(capital if capital is not None
                                   else config.TRADING_CAPITAL)
        self.kite = kite
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
                      direction: str, ts: float | None = None):
        ts = ts or time.time()
        self.deployed = max(self.deployed - premium_outlay, 0.0)
        self.open_positions = max(self.open_positions - 1, 0)
        self.realized_pnl += pnl_after_costs
        self.last_exit_ts = ts
        if pnl_after_costs < 0:
            self.lockout_until = ts + config.DIRECTION_LOCKOUT_S
            self.lockout_direction = direction
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
                      ann_vol: float | None = None) -> TradePermit:
        ts = ts or time.time()
        if self.halted:
            return TradePermit(False, f"halted: {self.halt_reason}")
        if self.ticks_seen < config.MIN_TICKS_BEFORE_TRADING:
            return TradePermit(False, "warm-up: physics not settled yet")
        if data_age_s > config.DATA_STALE_BLOCK_S:
            return TradePermit(False, f"stale feed ({data_age_s:.1f}s)")
        if now_hm >= config.NO_ENTRY_AFTER:
            return TradePermit(False, f"entry curfew after {config.NO_ENTRY_AFTER}")
        if self.open_positions >= config.MAX_CONCURRENT_POSITIONS:
            return TradePermit(False, "max concurrent positions")
        if ts - self.last_exit_ts < config.COOLDOWN_S:
            return TradePermit(False, "cooldown")
        if ts < self.lockout_until and direction == self.lockout_direction:
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
        if self.kite is not None and symbol:   # read-only; no order placed
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
