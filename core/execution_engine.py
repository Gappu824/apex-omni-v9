"""
APEX OMNI v9 — EXECUTION ENGINE (audit §8: "execution truth")
=============================================================
The single gravest v8 gap: fire-and-forget orders, positions registered at
the limit price whether or not anything filled. v9's contract:

  A POSITION EXISTS ONLY IF A FILL SAYS SO.

Live mode (LIVE_FIRE + confirmation phrase) drives pykiteconnect:
  place_order → order_id → poll order_history/order_trades within a time
  budget → handle COMPLETE / partial / OPEN → one cancel-and-repost a tick
  worse → walk away. Exits prefer LIMIT at the touch; the protected-market
  fallback sends market_protection (mandatory for API market orders since
  the April-2026 SEBI framework; pykiteconnect v4 exposes the parameter).
  reconcile() pulls kite.positions() at startup so a restart can never
  hallucinate a flat book.

Paper mode is an EVENT-DRIVEN fill simulator, not a rubber stamp: a maker
limit buy rests until the market's ask trades down to it (or times out and
goes through the same cancel/repost path as live). Crossing the spread fills
at the touch + slippage. This is what makes the scenario simulator's
results mean something.

The cost model lives here too, so the paper engine, the forge's reward and
the analyzer's verdict all pay the SAME toll: ₹20/executed order brokerage,
0.1% STT on sell premium, exchange txn + SEBI on both legs, GST on
brokerage+txn, stamp on buys.
"""
from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass, field

import config

log = logging.getLogger("exec")

try:
    from kiteconnect import KiteConnect  # noqa: F401
    HAVE_KITE = True
except Exception:                         # noqa: BLE001
    HAVE_KITE = False


# ----------------------------------------------------------------- costs
def round_trip_costs(buy_value: float, sell_value: float,
                     n_buy_orders: int = 1, n_sell_orders: int = 1) -> float:
    c = config.COSTS
    brokerage = c["brokerage_per_order"] * (n_buy_orders + n_sell_orders)
    txn = c["exch_txn_pct"] * (buy_value + sell_value)
    sebi = c["sebi_pct"] * (buy_value + sell_value)
    stt = c["stt_sell_pct"] * sell_value
    gst = c["gst_pct"] * (brokerage + txn)
    stamp = c["stamp_buy_pct"] * buy_value
    return brokerage + txn + sebi + stt + gst + stamp


# ----------------------------------------------------------------- results
@dataclass
class Fill:
    status: str            # FILLED | PARTIAL | NOFILL | REJECTED
    qty: int = 0
    avg_price: float = 0.0
    order_id: str = ""
    reason: str = ""
    n_orders: int = 1      # executed orders (for brokerage truth)


@dataclass
class _Resting:
    order_id: str
    side: str              # BUY | SELL
    token: int
    symbol: str
    qty: int
    limit: float
    placed_ts: float
    reposted: bool = False
    filled_qty: int = 0
    filled_value: float = 0.0


class _RateLimiter:
    def __init__(self, per_s: int):
        self.per_s = per_s; self.stamps: list[float] = []
    def wait(self):
        now = time.time()
        self.stamps = [t for t in self.stamps if now - t < 1.0]
        if len(self.stamps) >= self.per_s:
            time.sleep(max(0.0, 1.0 - (now - self.stamps[0])))
        self.stamps.append(time.time())


class ExecutionEngine:
    """quote_fn(token) -> {'bid','ask','bid_qty','ask_qty','ltp'} — live mode
    wires this to the ring buffer; the simulator wires it to the scenario."""

    def __init__(self, kite=None, quote_fn=None, clock=time.time):
        self.live = config.live_fire_armed() and kite is not None and HAVE_KITE
        self.kite = kite
        self.quote_fn = quote_fn or (lambda token: {})
        self.clock = clock
        self.resting: dict[str, _Resting] = {}
        self.order_rl = _RateLimiter(config.RATE["order_per_s"])
        self.inject_rejects = 0            # simulator fault-injection
        self.lookahead_fn = None           # sim: (token, horizon)->(min_ask,max_bid)
        self.paper_slippage_ticks = config.PAPER_SLIPPAGE_TICKS
        self.paper_fill_realism = config.PAPER_FILL_REALISM
        if config.LIVE_FIRE and not self.live:
            log.warning("LIVE_FIRE=True but confirmation phrase absent — "
                        "running PAPER. (This is the double-switch working.)")
        log.info("ExecutionEngine mode: %s", "LIVE" if self.live else "PAPER")

    # ------------------------------------------------------------ helpers
    def _tick_size(self, price: float) -> float:
        return 0.05

    # ------------------------------------------------------------ entries
    def buy_limit(self, *, symbol: str, exchange: str, token: int, qty: int,
                  limit: float, budget_s: float | None = None) -> Fill:
        budget_s = budget_s or config.ORDER_POLL_BUDGET_S
        return (self._live_limit("BUY", symbol, exchange, token, qty, limit,
                                 budget_s) if self.live else
                self._paper_limit("BUY", symbol, token, qty, limit, budget_s))

    def sell_limit(self, *, symbol: str, exchange: str, token: int, qty: int,
                   limit: float, budget_s: float | None = None,
                   urgent: bool = False) -> Fill:
        budget_s = budget_s or config.ORDER_POLL_BUDGET_S
        f = (self._live_limit("SELL", symbol, exchange, token, qty, limit,
                              budget_s) if self.live else
             self._paper_limit("SELL", symbol, token, qty, limit, budget_s))
        if urgent and f.status in ("NOFILL", "PARTIAL"):
            rem = qty - f.qty
            q = self.quote_fn(token) or {}
            bid = float(q.get("bid") or limit)
            chase = max(bid - config.URGENT_CHASE_TICKS * self._tick_size(bid), 0.05)
            f2 = (self._live_limit("SELL", symbol, exchange, token, rem, chase,
                                   budget_s) if self.live else
                  self._paper_limit("SELL", symbol, token, rem, chase, budget_s))
            tot = f.qty + f2.qty
            if tot:
                avg = (f.avg_price * f.qty + f2.avg_price * f2.qty) / tot
                return Fill("FILLED" if tot == qty else "PARTIAL", tot, avg,
                            f2.order_id or f.order_id, "urgent chase",
                            f.n_orders + f2.n_orders)
        return f

    # ------------------------------------------------------------ LIVE path
    def _live_limit(self, side, symbol, exchange, token, qty, limit,
                    budget_s) -> Fill:
        try:
            self.order_rl.wait()
            oid = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR, exchange=exchange,
                tradingsymbol=symbol, transaction_type=side, quantity=qty,
                product=self.kite.PRODUCT_MIS,
                order_type=self.kite.ORDER_TYPE_LIMIT,
                price=round(limit, 2), validity=self.kite.VALIDITY_DAY,
                tag="APEXv9")
        except Exception as e:                                   # noqa: BLE001
            log.error("place_order rejected: %s", e)
            return Fill("REJECTED", reason=str(e), n_orders=0)
        fill = self._poll_until(oid, qty, budget_s)
        if fill.status == "FILLED":
            return fill
        # one polite repost a tick worse, then walk away (maker discipline)
        try:
            self.order_rl.wait()
            q = self.quote_fn(token) or {}
            tick = self._tick_size(limit)
            new = limit + tick if side == "BUY" else limit - tick
            new = min(new, float(q.get("ask") or new)) if side == "BUY" \
                else max(new, float(q.get("bid") or new))
            self.kite.modify_order(variety=self.kite.VARIETY_REGULAR,
                                   order_id=oid, price=round(new, 2))
            fill2 = self._poll_until(oid, qty, budget_s)
            if fill2.status == "FILLED":
                fill2.reason = "filled on repost"
                return fill2
            self.order_rl.wait()
            self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR,
                                   order_id=oid)
        except Exception as e:                                   # noqa: BLE001
            log.error("repost/cancel error: %s", e)
        return self._truth_from_trades(oid, qty)

    def _push_status(self, oid: str) -> Fill | None:
        """WS order pushes (written by the harvester) beat REST polling —
        zero rate-limit cost, postback-equivalent payload."""
        try:
            import json
            p = config.ORDER_UPDATES_PATH
            if not p.exists():
                return None
            d = json.loads(p.read_text()).get(str(oid))
            if not d:
                return None
            if d.get("status") == "COMPLETE" and d.get("avg"):
                return Fill("FILLED", int(d.get("filled") or 0),
                            float(d["avg"]), oid, "ws push")
            if d.get("status") == "REJECTED":
                return Fill("REJECTED", reason=d.get("msg") or "rejected",
                            order_id=oid, n_orders=0)
        except Exception:                                # noqa: BLE001
            pass
        return None

    def _poll_until(self, oid, qty, budget_s) -> Fill:
        t0 = time.time()
        while time.time() - t0 < budget_s:
            f = self._push_status(oid)
            if f is not None:
                if f.status == "FILLED" and f.qty in (0, qty):
                    f.qty = qty
                    return f
                if f.status == "REJECTED":
                    return f
            try:
                hist = self.kite.order_history(oid)
                st = hist[-1]
                if st["status"] == "COMPLETE":
                    return Fill("FILLED", qty,
                                float(st.get("average_price") or 0), oid)
                if st["status"] in ("REJECTED", "CANCELLED"):
                    return Fill("REJECTED" if st["status"] == "REJECTED"
                                else "NOFILL", reason=st.get(
                                    "status_message", st["status"]),
                                order_id=oid, n_orders=0)
            except Exception as e:                               # noqa: BLE001
                log.warning("order_history: %s", e)
            time.sleep(config.LIVE_POLL_INTERVAL_S)
        return self._truth_from_trades(oid, qty)

    def _truth_from_trades(self, oid, qty) -> Fill:
        """Partial-fill truth straight from the trades endpoint."""
        try:
            trades = self.kite.order_trades(oid)
        except Exception:                                        # noqa: BLE001
            trades = []
        fq = sum(int(t["quantity"]) for t in trades)
        if fq <= 0:
            return Fill("NOFILL", order_id=oid, n_orders=0)
        fv = sum(int(t["quantity"]) * float(t["average_price"]) for t in trades)
        return Fill("PARTIAL" if fq < qty else "FILLED", fq, fv / fq, oid)

    # ------------------------------------------------------------ PAPER path
    def _paper_limit(self, side, symbol, token, qty, limit, budget_s) -> Fill:
        if self.inject_rejects > 0:
            self.inject_rejects -= 1
            return Fill("REJECTED", reason="paper-injected reject", n_orders=0)
        if self.clock is not time.time:
            return self._sim_limit(side, token, qty, limit, budget_s)
        oid = f"P{uuid.uuid4().hex[:10]}"
        r = _Resting(oid, side, token, symbol, qty, limit, self.clock())
        # immediate marketable check
        f = self._try_fill(r, self.quote_fn(token) or {})
        if f:
            return f
        self.resting[oid] = r
        # In the live loop / simulator, on_quote() is pumped every tick; here
        # we also poll briefly so callers may use this synchronously.
        deadline = r.placed_ts + budget_s
        while self.clock() < deadline:
            if oid not in self.resting:
                break
            f = self._try_fill(r, self.quote_fn(token) or {})
            if f:
                self.resting.pop(oid, None)
                return f
            if self.clock == time.time:
                time.sleep(0.1)
            else:
                break        # simulated clock: the sim pumps on_quote itself
        if oid in self.resting and self.clock() >= deadline:
            self.resting.pop(oid, None)
            if not r.reposted:        # one repost, a tick worse — same as live
                tick = self._tick_size(limit)
                r2 = _Resting(oid + "R", side, token, symbol, qty - r.filled_qty,
                              limit + tick if side == "BUY" else limit - tick,
                              self.clock(), reposted=True)
                f = self._try_fill(r2, self.quote_fn(token) or {})
                if f:
                    f.reason = "filled on repost"
                    return f
                self.resting[r2.order_id] = r2
                return Fill("NOFILL", r.filled_qty,
                            (r.filled_value / r.filled_qty) if r.filled_qty
                            else 0.0, oid, "resting after repost", 0)
        if r.filled_qty:
            return Fill("PARTIAL", r.filled_qty,
                        r.filled_value / r.filled_qty, oid)
        return Fill("NOFILL", order_id=oid, n_orders=0)

    def _sim_limit(self, side, token, qty, limit, budget_s) -> Fill:
        """Deterministic paper fills under a simulated clock. A maker limit
        fills only if the opposing touch trades THROUGH it within the poll
        budget (scenario provides the lookahead); marketable orders fill at
        the touch + slippage; otherwise one repost a tick worse, then cancel
        — exactly the live lifecycle's shape, compressed."""
        oid = f"S{uuid.uuid4().hex[:10]}"
        q = self.quote_fn(token) or {}
        bid, ask = float(q.get("bid") or 0), float(q.get("ask") or 0)
        tick = self._tick_size(limit)
        slip = self.paper_slippage_ticks * tick

        def attempt(px) -> Fill | None:
            if bid <= 0 or ask <= 0:
                return None
            if side == "BUY":
                if px >= ask:                       # marketable → taker
                    return Fill("FILLED", qty, round(min(ask + slip, px), 2), oid)
                if self.lookahead_fn:
                    mn_ask, _ = self.lookahead_fn(token, budget_s)
                    hit = (mn_ask < px) if self.paper_fill_realism \
                        else (mn_ask <= px)   # strict = behind-the-queue proxy
                    if mn_ask is not None and hit:
                        return Fill("FILLED", qty, round(px, 2), oid,
                                    "maker fill")
            else:
                if px <= bid:
                    return Fill("FILLED", qty, round(max(bid - slip, px), 2), oid)
                if self.lookahead_fn:
                    _, mx_bid = self.lookahead_fn(token, budget_s)
                    hit = (mx_bid > px) if self.paper_fill_realism \
                        else (mx_bid >= px)
                    if mx_bid is not None and hit:
                        return Fill("FILLED", qty, round(px, 2), oid,
                                    "maker fill")
            return None

        f = attempt(limit)
        if f:
            return f
        worse = limit + tick if side == "BUY" else max(limit - tick, 0.05)
        f = attempt(worse)
        if f:
            f.reason = "filled on repost"
            return f
        return Fill("NOFILL", order_id=oid,
                    reason="maker unfilled within budget — walked away",
                    n_orders=0)

    def _try_fill(self, r: _Resting, q: dict) -> Fill | None:
        bid, ask = float(q.get("bid") or 0), float(q.get("ask") or 0)
        if bid <= 0 or ask <= 0:
            return None
        slip = self.paper_slippage_ticks * self._tick_size(ask)
        if r.side == "BUY" and r.limit >= ask:
            px = min(ask + (slip if r.limit > ask else 0.0), r.limit)
            return Fill("FILLED", r.qty, round(px, 2), r.order_id)
        if r.side == "SELL" and r.limit <= bid:
            px = max(bid - (slip if r.limit < bid else 0.0), r.limit)
            return Fill("FILLED", r.qty, round(px, 2), r.order_id)
        return None

    def on_quote(self, token: int, q: dict) -> list[tuple[str, Fill]]:
        """Simulator/live-loop pump: lets resting maker orders fill the moment
        the market trades through them. Returns [(order_id, Fill), …]."""
        out = []
        for oid in list(self.resting):
            r = self.resting[oid]
            if r.token != token:
                continue
            if self.clock() - r.placed_ts > config.ORDER_POLL_BUDGET_S * 2:
                self.resting.pop(oid, None)
                out.append((oid, Fill("NOFILL", r.filled_qty,
                                      (r.filled_value / r.filled_qty)
                                      if r.filled_qty else 0.0, oid,
                                      "expired", 0)))
                continue
            f = self._try_fill(r, q)
            if f:
                self.resting.pop(oid, None)
                out.append((oid, f))
        return out

    # ------------------------------------------------------------ GTT floor
    def arm_gtt_floor(self, *, symbol, exchange, qty, floor_px,
                      last_price) -> str | None:
        """Server-side disaster floor: a GTT single-trigger SELL LIMIT that
        survives a dead laptop or a cut feed. Config-gated (LIVE_GTT_FLOOR)
        because GTTs are good-till-cancelled and product-NRML — pair it with
        NRML positions, not MIS, and remember the 250-active-GTT account cap.
        Triggered GTT fills surface on the WS order updates, not postbacks."""
        if not (self.live and config.LIVE_GTT_FLOOR):
            if config.LIVE_GTT_FLOOR:
                log.info("paper: would arm GTT floor %s ≤ %.2f", symbol,
                         floor_px)
            return None
        try:
            self.order_rl.wait()
            trig = self.kite.place_gtt(
                trigger_type=self.kite.GTT_TYPE_SINGLE,
                tradingsymbol=symbol, exchange=exchange,
                trigger_values=[round(floor_px, 2)], last_price=last_price,
                orders=[{"transaction_type": self.kite.TRANSACTION_TYPE_SELL,
                         "quantity": qty,
                         "order_type": self.kite.ORDER_TYPE_LIMIT,
                         "product": self.kite.PRODUCT_NRML,
                         "price": round(max(floor_px * 0.97, 0.05), 2)}])
            tid = str(trig.get("trigger_id"))
            log.info("GTT floor armed %s @ %.2f (trigger %s)", symbol,
                     floor_px, tid)
            return tid
        except Exception as e:                           # noqa: BLE001
            log.error("GTT floor failed (%s) — in-process floor still live", e)
            return None

    def disarm_gtt(self, trigger_id: str | None):
        if not (self.live and trigger_id):
            return
        try:
            self.order_rl.wait()
            self.kite.delete_gtt(trigger_id=trigger_id)
            log.info("GTT floor disarmed (%s)", trigger_id)
        except Exception as e:                           # noqa: BLE001
            log.warning("delete_gtt(%s): %s", trigger_id, e)

    # ------------------------------------------------------------ truth sync
    def reconcile(self) -> list[dict]:
        """Startup truth: what does the BROKER say we hold? (audit §8)"""
        if not self.live:
            return []
        try:
            net = self.kite.positions().get("net", [])
            open_pos = [p for p in net if int(p.get("quantity") or 0) != 0]
            for p in open_pos:
                log.warning("RECONCILE: broker shows OPEN %s qty=%s avg=%.2f",
                            p["tradingsymbol"], p["quantity"],
                            float(p.get("average_price") or 0))
            return open_pos
        except Exception as e:                                   # noqa: BLE001
            log.error("positions() failed: %s — refusing to assume flat", e)
            raise
