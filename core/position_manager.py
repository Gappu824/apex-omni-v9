"""
APEX OMNI v9 — POSITION MANAGER
===============================
One lifecycle, shared verbatim by the live brain and the scenario simulator.

Entry:   conviction → calibration table → RiskGovernor.first_affordable()
         walks the strike hierarchy (ATM → OTM → deeper) to the first leg the
         capital can genuinely HOLD → maker limit at the micro-price →
         position registered ONLY from the fill (qty & price from the broker
         /paper fill, never from intent).

Exit ladder, evaluated every tick, strictly in this order:
   1. DISASTER FLOOR     — absolute; overrides everything incl. the shield.
   2. EOD FLATTEN        — 15:15, before the broker's MIS square-off.
   3. STALE-FEED FLATTEN — feed dead > DATA_STALE_FLATTEN_S.
   4. TARGET             — corrected expected-move (single √T) + GEX runway.
   5. THETA GUILLOTINE   — max-hold minutes for short-dated longs.
   6. CONVICTION REVERSAL— model flips hard against the position.
   7. STOP (trail/base)  — but a breach is first shown to the TrapShield;
                           a suspected institutional flush is HELD through,
                           under the floor, for a bounded grace window. A
                           confirmed trap re-anchors the stop under the
                           hunt's low instead of panic-selling it.

Ledger: one CSV row per FILL event with the symbol written BEFORE any state
is cleared (audit: v8 nulled self.symbol first and logged `None` on every
single exit).
"""
from __future__ import annotations
import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import config
from core.execution_engine import ExecutionEngine, Fill, round_trip_costs
from core.risk_manager import RiskGovernor, TradePermit
from core.trap_shield import TrapShield, TrapSignals
from core.quant_core import expected_move, micro_price

log = logging.getLogger("pm")

LEDGER_FIELDS = ["ts", "event", "index", "symbol", "direction", "qty",
                 "price", "value", "conviction", "win_prob", "pnl",
                 "costs", "reason", "order_id"]


@dataclass
class LegQuote:
    leg: str                  # atm_ce / otm_ce / atm_pe / otm_pe
    symbol: str
    exchange: str
    token: int
    strike: float
    premium: float            # mid
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    lot: int
    delta: float
    dte: float


@dataclass
class TickContext:
    ts: float
    hm: str                   # "HH:MM"
    spot: float
    spot_velocity_1s: float
    data_age_s: float
    atm_iv: float             # annualized (the ONE sigma)
    minutes_to_close: float
    gex_put_wall: float | None = None
    gex_call_wall: float | None = None
    absorption: bool = False
    aggressive_sell_ratio: float = 0.5
    oi_delta_since: float = 0.0
    avg_spread_pct: float = 0.01
    conviction: float = 0.0
    live_win_prob: float | None = None   # model's LIVE P(win) for the HELD
    #                                      position's direction (None until a
    #                                      trained meta-model exists). Drives the
    #                                      model-shaped early exit; never overrides
    #                                      the disaster floor / EOD / hard stop.
    regime_label: str = ""               # market regime at this tick (for the
    #                                      per-regime performance breakdown)


@dataclass
class Position:
    index: str
    direction: str            # CE | PE
    symbol: str
    exchange: str
    token: int
    strike: float
    qty: int
    entry: float
    entry_ts: float
    delta_at_entry: float
    conviction: float
    win_prob: float
    order_id: str
    n_buy_orders: int
    gtt_id: str | None = None
    dte: float = 2.0
    stop: float = 0.0
    target: float = 0.0
    floor: float = 0.0
    profit_lock: float = 0.0      # breakeven+costs; active once trail arms. A
    #                               trap-hold can never give back below this.
    breakeven_px: float = 0.0     # entry + round-trip cost per unit
    extends_used: int = 0         # model-driven target extensions taken so far
    peak: float = 0.0
    trail_armed: bool = False
    shield: TrapShield = field(default=None)            # type: ignore
    spike_ref_spot: float = 0.0
    spike_ref_oi: float = 0.0


class PositionManager:
    def __init__(self, index: str, risk: RiskGovernor, engine: ExecutionEngine,
                 ledger_path: Path | None = None):
        self.index = index
        self.risk = risk
        self.engine = engine
        self.pos: Position | None = None
        self.last_block_reason = ""     # why the most recent try_enter didn't fill
                                        # (read by the brain's heartbeat why-flat line)
        # walk-away diagnostics: classify chase-cap NOFILLs so the operator can
        # SEE whether the slip cap is refusing genuine chases (runaway) or might
        # be too tight on real fills (borderline). Evidence to tune the cap.
        self._walkaway_tally = {"runaway": 0, "borderline": 0,
                                "worst_overshoot": 0.0}
        self.ledger = Path(ledger_path or config.LEDGER_PATH)
        if not self.ledger.exists():
            with self.ledger.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, LEDGER_FIELDS).writeheader()
        self.events: list[dict] = []      # in-memory mirror (simulator reads)

    # ------------------------------------------------------------ ledger
    def _log(self, **row):
        row = {k: row.get(k, "") for k in LEDGER_FIELDS}
        self.events.append(row)
        with self.ledger.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, LEDGER_FIELDS).writerow(row)

    # ------------------------------------------------------------ entry
    def try_enter(self, ctx: TickContext, direction: str, conviction: float,
                  win_prob: float, hierarchy: list[LegQuote]) -> bool:
        if self.pos is not None:
            self.last_block_reason = "already in position"
            return False
        cands = []
        for q in hierarchy:
            sp = (q.ask - q.bid) / max(q.premium, 0.05)
            if q.bid <= 0 or q.ask <= 0:
                continue
            if sp > config.MAX_ENTRY_SPREAD_PCT:
                continue          # illiquidity gate (audit §12 spread reality)
            cands.append({"q": q, "premium": q.premium, "lot": q.lot,
                          "symbol": q.symbol, "exchange": q.exchange,
                          "price": q.ask})
        if not cands:
            self._log(ts=ctx.ts, event="SKIP", index=self.index,
                      direction=direction, reason="no leg passes spread gate",
                      conviction=f"{conviction:.3f}")
            log.info("SKIP %s %s (conv %+.2f): no leg passes spread gate",
                     self.index, direction, conviction)
            self.last_block_reason = "no leg passes spread gate"
            return False
        leg, permit = self.risk.first_affordable(
            cands, direction=direction, win_prob=win_prob,
            sl_pct=config.BASE_SL_PCT, tp_pct=config.BASE_TP_PCT,
            data_age_s=ctx.data_age_s, now_hm=ctx.hm, ts=ctx.ts,
            ann_vol=ctx.atm_iv or None)
        if leg is None:
            self._log(ts=ctx.ts, event="BLOCKED", index=self.index,
                      direction=direction, reason=permit.reason,
                      conviction=f"{conviction:.3f}")
            log.info("BLOCKED %s %s (conv %+.2f): %s", self.index,
                     direction, conviction, permit.reason)
            self.last_block_reason = permit.reason
            return False
        q: LegQuote = leg["q"]
        tick = 0.05
        micro = micro_price(q.bid, q.ask, q.bid_qty, q.ask_qty)
        spread_pct = (q.ask - q.bid) / max((q.ask + q.bid) / 2, 0.05)
        step = config.INDICES[self.index]["strike_step"]
        cross = abs(conviction) >= config.ENTRY_CROSS_CONVICTION

        if cross:
            # MOMENTUM ENTRY → cross the spread. The leg already cleared the hard
            # liquidity gate (MAX_ENTRY_SPREAD_PCT) to even be a candidate, so the
            # ONLY remaining question is anti-chase: has the ask run too far from
            # the decision-time micro-price? One strike-step of SPOT move ≈
            # delta×step of PREMIUM; allow ENTRY_SLIP_CAP_PCT of that. Past it,
            # the runner is gone — walk away, don't buy exhaustion (critical at
            # 0–2 DTE). There is deliberately NO separate "comfortable spread"
            # band: the hard gate is the liquidity filter, the slip cap is the
            # chase filter, and a third guessed threshold only starved fills.
            prem_cap = max(abs(q.delta), 0.05) * step * config.ENTRY_SLIP_CAP_PCT
            max_pay = micro + prem_cap
            if q.ask > max_pay:
                # how far past the cap did the ask run? overshoot as a FRACTION
                # of the cap distance tells us WHY we walked: a tiny overshoot
                # (≤ BORDERLINE) hints the cap may be slightly tight on a real
                # fill; a large one is a genuine runaway the cap rightly refused.
                overshoot = (q.ask - max_pay) / max(prem_cap, 1e-6)
                borderline = overshoot <= config.SLIPCAP_BORDERLINE_FRAC
                kind = "borderline" if borderline else "runaway"
                self._walkaway_tally["borderline" if borderline else "runaway"] += 1
                self._walkaway_tally["worst_overshoot"] = max(
                    self._walkaway_tally["worst_overshoot"], overshoot)
                self._log(ts=ctx.ts, event="NOFILL", index=self.index,
                          symbol=q.symbol, direction=direction,
                          price=f"{q.ask:.2f}",
                          reason=f"signal stale ({kind}) — ask {q.ask:.2f} ran "
                                 f"{overshoot*100:.0f}% past chase cap "
                                 f"{max_pay:.2f} (micro {micro:.2f} +{prem_cap:.2f}"
                                 f", spread {spread_pct*100:.1f}%)",
                          conviction=f"{conviction:.3f}")
                log.info("NOFILL %s %s — %s: ask %.2f is %.0f%% past chase cap "
                         "%.2f (spread %.1f%%), walked away",
                         self.index, q.symbol, kind, q.ask, overshoot * 100,
                         max_pay, spread_pct * 100)
                self.last_block_reason = f"signal stale ({kind}) — ask past chase cap"
                return False
            # cross: take the ask, hard-ceilinged a couple ticks beyond it
            limit = round(min(q.ask + config.ENTRY_CROSS_CAP_TICKS * tick,
                              max_pay), 2)
        else:
            # passive maker — only reached if the cross bar is set ABOVE the
            # entry bar (e.g. a mean-reversion variant or a regression override).
            # In the default momentum config cross bar == entry bar, so entries
            # always cross and this path is not used.
            limit = round(min(micro, q.ask), 2)
        fill: Fill = self.engine.buy_limit(symbol=q.symbol, exchange=q.exchange,
                                           token=q.token, qty=permit.qty,
                                           limit=limit)
        if fill.status == "REJECTED":
            self.risk.register_reject()
            self._log(ts=ctx.ts, event="REJECT", index=self.index,
                      symbol=q.symbol, direction=direction,
                      reason=fill.reason, order_id=fill.order_id)
            self.last_block_reason = f"rejected ({fill.reason})"
            return False
        if fill.status == "NOFILL" or fill.qty <= 0:
            self._log(ts=ctx.ts, event="NOFILL", index=self.index,
                      symbol=q.symbol, direction=direction,
                      price=limit, reason="maker did not fill — walked away",
                      order_id=fill.order_id)
            log.info("NOFILL %s %s @ %.2f — walked away", self.index,
                     q.symbol, limit)
            self.last_block_reason = "maker no-fill — walked away"
            return False

        outlay = fill.avg_price * fill.qty
        self.risk.register_entry(outlay)
        sl_dist = fill.avg_price * config.BASE_SL_PCT
        p = Position(index=self.index, direction=direction, symbol=q.symbol,
                     exchange=q.exchange, token=q.token, strike=q.strike,
                     qty=fill.qty, entry=fill.avg_price, entry_ts=ctx.ts,
                     delta_at_entry=abs(q.delta) or 0.4,
                     conviction=conviction, win_prob=win_prob,
                     order_id=fill.order_id, n_buy_orders=max(fill.n_orders, 1),
                     dte=float(q.dte))
        p.stop = fill.avg_price - sl_dist
        p.floor = fill.avg_price - min(sl_dist * config.DISASTER_FLOOR_MULT,
                                       fill.avg_price * config.ABS_DISASTER_PCT)
        # profit-lock floor: per-unit breakeven = entry + round-trip cost/unit.
        # A trap-hold on a winner can never give back below this (set live once
        # the trail arms). Constitution — fixed, not learned.
        rt_cost = round_trip_costs(fill.avg_price * fill.qty,
                                   fill.avg_price * fill.qty,
                                   n_buy_orders=max(fill.n_orders, 1),
                                   n_sell_orders=max(fill.n_orders, 1))
        p.breakeven_px = fill.avg_price + rt_cost / max(fill.qty, 1)
        em = expected_move(ctx.spot, ctx.atm_iv, ctx.minutes_to_close)
        runway = None
        if direction == "CE" and ctx.gex_call_wall and ctx.gex_call_wall > ctx.spot:
            runway = ctx.gex_call_wall - ctx.spot
        if direction == "PE" and ctx.gex_put_wall and ctx.gex_put_wall < ctx.spot:
            runway = ctx.spot - ctx.gex_put_wall
        spot_room = min(em, runway) if runway else em
        prem_room = p.delta_at_entry * spot_room
        p.target = fill.avg_price + max(prem_room,
                                        fill.avg_price * config.BASE_TP_PCT)
        p.peak = fill.avg_price
        p.shield = TrapShield(direction)
        p.spike_ref_spot = ctx.spot
        p.gtt_id = self.engine.arm_gtt_floor(symbol=p.symbol,
            exchange=p.exchange, qty=p.qty, floor_px=p.floor,
            last_price=p.entry)
        self.pos = p
        self._log(ts=ctx.ts, event="BUY_FILL", index=self.index,
                  symbol=p.symbol, direction=direction, qty=p.qty,
                  price=f"{p.entry:.2f}", value=f"{outlay:.2f}",
                  conviction=f"{conviction:.3f}", win_prob=f"{win_prob:.3f}",
                  reason=fill.reason or "maker fill", order_id=p.order_id,
                  regime=ctx.regime_label)
        log.info("ENTER %s %s ×%d @ %.2f | stop %.2f floor %.2f target %.2f",
                 self.index, p.symbol, p.qty, p.entry, p.stop, p.floor, p.target)
        self.last_block_reason = ""
        return True

    # ------------------------------------------------------------ manage
    def live_snapshot(self, ctx: TickContext, quote: dict) -> str | None:
        """Build a one-line live read of the OPEN position for the heartbeat:
        mark-to-market PnL, distance to stop/target, peak, trail/lock state, and
        the institutional read (OI delta, trap-shield score, model P(win)). Pure
        reporting — computes nothing that changes behavior. None if flat."""
        p = self.pos
        if p is None:
            return None
        bid = float(quote.get("bid") or 0)
        ltp = float(quote.get("ltp") or bid or p.entry)
        mark = bid if bid > 0 else ltp
        upnl = (mark - p.entry) * p.qty
        upnl_pct = (mark / p.entry - 1.0) * 100 if p.entry else 0.0
        # distance to the rungs that matter, as % of premium
        to_stop = (mark / p.stop - 1.0) * 100 if p.stop else 0.0
        to_tgt = (p.target / mark - 1.0) * 100 if mark else 0.0
        held_s = ctx.ts - p.entry_ts
        # institutional read (already computed upstream; surfaced here)
        oi = ctx.oi_delta_since
        # live trap score for THIS position's direction, if the shield can read it
        trap = ""
        try:
            from core.trap_shield import TrapSignals
            sig = TrapSignals(
                spot=ctx.spot, spot_velocity=ctx.spot_velocity_1s,
                absorption=ctx.absorption,
                aggressive_sell_ratio=ctx.aggressive_sell_ratio,
                oi_delta_break=ctx.oi_delta_since,
                premium_move=abs(mark - p.entry),
                delta_implied=max(abs(p.delta_at_entry) *
                                  abs(ctx.spot_velocity_1s), 1e-6),
                spread_pct=ctx.avg_spread_pct)
            sc, _ = p.shield.score(sig)
            trap = f" trap {sc:.2f}"
        except Exception:                                  # noqa: BLE001
            trap = ""
        wp = f" P(win) {ctx.live_win_prob:.2f}" if ctx.live_win_prob is not None \
            else ""
        armed = "TRAIL" if p.trail_armed else "warm"
        lock = f" lock {p.profit_lock:.2f}" if p.profit_lock > 0 else ""
        return (f"  ↳ {p.symbol} {p.direction} ×{p.qty} | mark ₹{mark:.2f} "
                f"(entry {p.entry:.2f}) | uPnL ₹{upnl:+.0f} ({upnl_pct:+.1f}%) | "
                f"peak {p.peak:.2f} | stop {p.stop:.2f} (+{to_stop:.0f}%) "
                f"target {p.target:.2f} (+{to_tgt:.0f}%) | {armed}{lock} | "
                f"held {held_s:.0f}s | OIΔ {oi:+.2%}{trap}{wp}")

    def manage(self, ctx: TickContext, quote: dict) -> str | None:
        p = self.pos
        if p is None:
            return None
        p.shield.observe(ctx.spot_velocity_1s)
        bid = float(quote.get("bid") or 0)
        ask = float(quote.get("ask") or 0)
        ltp = float(quote.get("ltp") or bid or p.entry)
        mark = bid if bid > 0 else ltp
        p.peak = max(p.peak, mark)
        if not p.trail_armed and p.peak >= p.entry * (1 + config.TRAIL_ARM_PCT):
            p.trail_armed = True
        if p.trail_armed:
            gain = p.peak - p.entry
            p.stop = max(p.stop, p.entry + gain * (1 - config.TRAIL_GIVEBACK_PCT))
            # set/raise the profit-lock floor: breakeven, optionally lifted to
            # lock a fraction of the best gain. Ratchets up with the peak, never
            # down. A trap-hold can ride a flush down to here but NO further.
            if config.PROFIT_LOCK_ENABLED:
                locked = p.entry + gain * (1.0 - config.PROFIT_LOCK_GIVEBACK)
                p.profit_lock = max(p.profit_lock, p.breakeven_px, locked)

        # 1) DISASTER FLOOR — absolute, shield-proof
        if mark <= p.floor:
            return self._exit(ctx, quote, "DISASTER_FLOOR", urgent=True)
        # 1.5) PROFIT-LOCK FLOOR — once armed, guarantees a winner can never
        # become a loser. BUT while the trap shield is ACTIVELY holding a
        # confirmed hunt, breakeven is suspended (the hunt is expected to
        # reclaim, and the shield's 150s / 2-use caps bound the exposure) — only
        # the disaster floor binds during a hold, so the lock can't flush us out
        # of the very sweep the shield is riding. The instant the shield is NOT
        # holding, breakeven applies in full.
        shield_holding = bool(getattr(p.shield, "holding", False))
        if (config.PROFIT_LOCK_ENABLED and p.trail_armed and not shield_holding
                and p.profit_lock > 0 and mark <= p.profit_lock):
            return self._exit(ctx, quote, "PROFIT_LOCK", urgent=True)
        # 2) EOD
        if ctx.hm >= config.FORCE_FLATTEN_AT:
            return self._exit(ctx, quote, "EOD_FLATTEN", urgent=True)
        # 3) stale feed
        if ctx.data_age_s > config.DATA_STALE_FLATTEN_S:
            return self._exit(ctx, quote, "STALE_FEED_FLATTEN", urgent=True)
        # 4) target — model may EXTEND it when the edge is still strong
        if mark >= p.target:
            if (config.META_DECISION_ENABLED
                    and ctx.live_win_prob is not None
                    and ctx.live_win_prob >= config.META_HOLD_PAST_TARGET_P
                    and p.extends_used < config.TARGET_EXTEND_MAX
                    and p.trail_armed):
                # edge says there's more in the move: push the target out by
                # another expected-move increment and keep riding. The armed
                # trail and the profit-lock floor protect the gain underneath,
                # so this can never give back to a loss. Re-evaluated next tag.
                em = expected_move(ctx.spot, ctx.atm_iv, ctx.minutes_to_close)
                step_prem = max(p.delta_at_entry * em, p.entry * config.BASE_TP_PCT)
                old_t = p.target
                p.target = mark + step_prem
                p.extends_used += 1
                self._log(ts=ctx.ts, event="TARGET_EXTEND", index=self.index,
                          symbol=p.symbol, direction=p.direction,
                          price=f"{mark:.2f}",
                          reason=f"P(win) {ctx.live_win_prob:.2f} ≥ "
                                 f"{config.META_HOLD_PAST_TARGET_P:.2f} — riding "
                                 f"(#{p.extends_used}/{config.TARGET_EXTEND_MAX}), "
                                 f"target {old_t:.2f}→{p.target:.2f}, lock "
                                 f"{p.profit_lock:.2f}",
                          conviction=f"{ctx.live_win_prob:.3f}")
                # fall through: do NOT exit this tick
            else:
                return self._exit(ctx, quote, "TARGET")
        # 5) theta guillotine (0-DTE regime cuts faster)
        hold_lim = config.MAX_HOLD_MINUTES_0DTE \
            if p.dte < config.EXPIRY_DTE_LT else config.MAX_HOLD_MINUTES
        if (ctx.ts - p.entry_ts) / 60.0 > hold_lim:
            return self._exit(ctx, quote, "MAX_HOLD_THETA")
        # 6) conviction reversal
        flip = -ctx.conviction if p.direction == "CE" else ctx.conviction
        if flip >= config.ENTRY_CONVICTION:
            return self._exit(ctx, quote, "CONVICTION_REVERSAL")
        # 6.5) MODEL-SHAPED EXIT — only when a trained meta-model is live. If the
        # model's fresh P(win) for THIS position's direction has decayed below the
        # floor, the edge is gone: exit early. This sits BELOW the disaster floor,
        # EOD and stale rungs (they already returned above), so it can only ever
        # cut a trade SOONER — never extend one past the fixed bounds. After the
        # min-hold guard, so entry noise doesn't whipsaw it.
        if (config.META_DECISION_ENABLED
                and ctx.live_win_prob is not None
                and (ctx.ts - p.entry_ts) >= config.META_EXIT_MIN_HOLD_S
                and ctx.live_win_prob < config.META_EXIT_P_FLOOR):
            self._log(ts=ctx.ts, event="META_EDGE_GONE", index=self.index,
                      symbol=p.symbol, direction=p.direction,
                      price=f"{mark:.2f}",
                      reason=f"model P(win) {ctx.live_win_prob:.2f} < floor "
                             f"{config.META_EXIT_P_FLOOR:.2f} — edge decayed",
                      conviction=f"{ctx.live_win_prob:.3f}")
            return self._exit(ctx, quote, "META_EDGE_GONE")
        # 7) stop — gated by the Trap Shield
        if mark <= p.stop:
            prem_move = (mark - p.entry) / p.entry
            sig = TrapSignals(
                spot_velocity_1s=ctx.spot_velocity_1s, spot=ctx.spot,
                absorption=ctx.absorption,
                aggressive_sell_ratio=ctx.aggressive_sell_ratio,
                oi_delta_break=ctx.oi_delta_since,
                premium_move_pct=prem_move,
                delta_implied_move_pct=p.delta_at_entry
                    * abs(ctx.spot - p.spike_ref_spot) / max(p.entry, 0.5),
                spread_pct=(ask - bid) / max((ask + bid) / 2, 0.05)
                    if ask > bid > 0 else 0.0,
                avg_spread_pct=ctx.avg_spread_pct,
                gex_put_wall=ctx.gex_put_wall, gex_call_wall=ctx.gex_call_wall)
            hold, why, score = p.shield.on_stop_breach(ctx.ts, sig)
            fp = getattr(p.shield, "last_fingerprints", {}) or {}
            fp_str = ";".join(f"{k}={fp[k]:.3f}" for k in sorted(fp)) if fp else ""
            if hold:
                self._log(ts=ctx.ts, event="TRAP_HOLD", index=self.index,
                          symbol=p.symbol, direction=p.direction,
                          price=f"{mark:.2f}", reason=why,
                          conviction=f"{score:.2f}", fingerprints=fp_str)
                return None
            # breach honored — record fingerprints too: these are the NEGATIVE
            # (genuine-breakdown) labels the forge needs alongside the holds.
            self._log(ts=ctx.ts, event="STOP_BREACH_HONORED", index=self.index,
                      symbol=p.symbol, direction=p.direction,
                      price=f"{mark:.2f}", reason=why,
                      conviction=f"{score:.2f}", fingerprints=fp_str)
            return self._exit(ctx, quote, f"STOP ({why})")
        # trap confirmed? re-anchor instead of celebrating prematurely
        if p.shield.holding and p.shield.reclaim_check(ctx.spot):
            old = p.stop
            hunt_prem = max(p.floor, mark * 0.97)
            p.stop = min(p.stop, hunt_prem)
            p.stop = max(p.floor, min(old, hunt_prem))
            self._log(ts=ctx.ts, event="TRAP_CONFIRMED", index=self.index,
                      symbol=p.symbol, direction=p.direction,
                      price=f"{mark:.2f}",
                      reason=f"flush reclaimed — stop re-anchored {old:.2f}→{p.stop:.2f}")
        return None

    # ------------------------------------------------------------ exit
    def _exit(self, ctx: TickContext, quote: dict, reason: str,
              urgent: bool = False) -> str:
        p = self.pos
        bid = float(quote.get("bid") or 0) or p.entry * 0.9
        bq = float(quote.get("bid_qty") or 0)
        aq = float(quote.get("ask_qty") or 0)
        ask = float(quote.get("ask") or bid)
        limit = round(max(micro_price(bid, ask, bq, aq), bid), 2) \
            if not urgent else round(bid, 2)
        fill = self.engine.sell_limit(symbol=p.symbol, exchange=p.exchange,
                                      token=p.token, qty=p.qty, limit=limit,
                                      urgent=urgent)
        if fill.qty <= 0:
            self._log(ts=ctx.ts, event="EXIT_NOFILL", index=self.index,
                      symbol=p.symbol, direction=p.direction, price=limit,
                      reason=f"{reason} — sell unfilled, will retry",
                      order_id=fill.order_id)
            return "RETRY"
        sold_value = fill.avg_price * fill.qty
        buy_value = p.entry * fill.qty
        costs = round_trip_costs(buy_value, sold_value,
                                 n_buy_orders=p.n_buy_orders,
                                 n_sell_orders=max(fill.n_orders, 1))
        pnl = sold_value - buy_value - costs
        # ★ audit fix: log BEFORE clearing any state — symbol never None again
        self._log(ts=ctx.ts, event="SELL_FILL", index=self.index,
                  symbol=p.symbol, direction=p.direction, qty=fill.qty,
                  price=f"{fill.avg_price:.2f}", value=f"{sold_value:.2f}",
                  pnl=f"{pnl:.2f}", costs=f"{costs:.2f}", reason=reason,
                  order_id=fill.order_id,
                  conviction=f"{p.conviction:.3f}",
                  win_prob=f"{p.win_prob:.3f}")
        remaining = p.qty - fill.qty
        self.risk.register_exit(p.entry * p.qty, pnl, p.direction, ctx.ts)
        log.info("EXIT %s %s ×%d @ %.2f | %s | PnL ₹%.2f (costs ₹%.2f)",
                 self.index, p.symbol, fill.qty, fill.avg_price, reason,
                 pnl, costs)
        if remaining > 0:      # partial exit: keep the residual position honest
            p.qty = remaining
            self._log(ts=ctx.ts, event="PARTIAL_REMAINDER", index=self.index,
                      symbol=p.symbol, direction=p.direction, qty=remaining,
                      reason="sell partially filled — residual still managed")
            return "PARTIAL"
        self.engine.disarm_gtt(p.gtt_id)
        self.pos = None
        return reason