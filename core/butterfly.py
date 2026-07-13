"""
APEX OMNI v9.7 — LONG DEBIT BUTTERFLY ENGINE (buy-only VRP expression)
======================================================================
The shortvol SIGNAL, expressed as a BOUGHT structure. evaluate_gate() (in
core.shortvol) already identifies the moment vol is rich and price is likely
to pin a wall — that read is instrument-agnostic. The credit spread expressed
it by SELLING; this expresses the identical thesis by BUYING a long debit
butterfly centred on the tested wall:

    BUY  1× inner wing   (wall − W·step)
    SELL 2× body         (wall)                 ← wrapped on BOTH sides
    BUY  1× outer wing   (wall + W·step)

Economics (CE example; PE is the mirror):
  • Entered for a NET DEBIT you PAY — max risk = debit × qty, full stop.
    There is no naked short: the two body shorts are covered by the two long
    wings straddling them. Buy-only in the sense that matters — defined,
    prepaid, capped risk with no assignment tail.
  • Max VALUE at expiry when spot pins the body (wall): the structure is worth
    ≈ W·step per unit; profit = that − debit. Away from the body in either
    direction it decays to zero — you lose only the debit.
  • Same VRP thesis as the credit spread (short vol, pin near wall), so it
    wins under the SAME conditions the gate certifies — but it PAYS theta to
    the wings while COLLECTING it on the body, netting a small long-vega-at-
    the-wings / short-vega-at-the-body profile that is richest when IV is
    high and mean-reverts (exactly SV_IVRANK_MIN territory).

Research anchors: the long butterfly as a defined-risk pin/vol-harvest is
standard (Natenberg, *Option Volatility & Pricing*, ch. on butterflies;
the 0DTE-fly literature, e.g. pinning studies around high-gamma expiries).

Execution honesty (the four-leg cost reality, stated not hidden): a fly is
FOUR legs = four toll-booth round trips. price_fly() therefore demands BOTH
wings and the body pass the per-leg spread cap AND that the debit clears a
minimum fraction of the wing width (junk-structure reject), and size_fly()
sizes off the debit like the long book sizes off premium. At retail costs a
fly only makes sense in liquid ATM-ish strikes — which is precisely where the
wall sits, so the gate's wall-anchoring keeps the legs liquid by construction.

This module is MECHANICS ONLY. The gate lives in shortvol.evaluate_gate;
certification runs through the spec factory (specs/butterfly_*.json) exactly
like every other engine — a fly trades live only behind its OWN certificate.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import config
from core.execution_engine import round_trip_costs
from core.shortvol import log_forward         # reuse the shared forward log


import hashlib as _hashlib


def fly_knob_hash() -> str:
    """Stable hash of the butterfly's certifiable knobs — the cert locks to
    this, fails closed if any structural parameter changes (Bailey–LdP: a new
    knob set is a new hypothesis and must re-earn its certificate)."""
    payload = "|".join(str(x) for x in (
        config.SV_FLY_WING_STEPS, config.SV_FLY_MIN_DEBIT_FRAC,
        config.SV_FLY_MAX_DEBIT_FRAC, config.SV_FLY_TP_FRAC,
        config.SV_FLY_SL_FRAC, config.SV_IVRANK_MIN,
        config.SV_NET_GEX_MIN, config.SV_CORRIDOR_MIN_STEPS,
        config.SV_WALL_BUFFER_STEPS, config.SV_DTE_MIN, config.SV_DTE_MAX,
        config.SV_RISK_PCT))
    return _hashlib.sha1(payload.encode()).hexdigest()[:10]


# ============================================================ pricing (pure)
def price_fly(*, wing_in: dict, body: dict, wing_out: dict,
              wing_width: float) -> tuple[float | None, str]:
    """Executable opening DEBIT of a long butterfly, quoted conservatively:
    BUY both wings at their ASK, SELL 2× body at its BID.
        debit = wing_in.ask + wing_out.ask - 2·body.bid   (per 1-lot unit)
    Rejects one-sided/illiquid legs and junk structures (debit must be a
    sensible fraction of the wing width, and strictly positive — a credit
    'butterfly' means the quotes are stale/crossed, never a free lunch)."""
    legs = (("wing_in", wing_in), ("body", body), ("wing_out", wing_out))
    for name, q in legs:
        b, a = float(q.get("bid") or 0), float(q.get("ask") or 0)
        if b <= 0 or a <= 0:
            return None, f"{name} leg one-sided"
        mid = (a + b) / 2.0
        if (a - b) / max(mid, 0.05) > config.MAX_ENTRY_SPREAD_PCT:
            return None, f"{name} leg spread>cap"
    debit = (float(wing_in["ask"]) + float(wing_out["ask"])
             - 2.0 * float(body["bid"]))
    if debit <= 0:
        return None, f"non-positive debit {debit:.2f} (stale/crossed quotes)"
    # a fly's theoretical max value is the wing width; a debit above some
    # fraction of it means no edge left to capture
    if debit > config.SV_FLY_MAX_DEBIT_FRAC * wing_width:
        return None, (f"debit {debit:.2f} > "
                      f"{config.SV_FLY_MAX_DEBIT_FRAC:.0%} of wing "
                      f"width {wing_width:.0f} (no room)")
    if debit < config.SV_FLY_MIN_DEBIT_FRAC * wing_width:
        return None, (f"debit {debit:.2f} < "
                      f"{config.SV_FLY_MIN_DEBIT_FRAC:.0%} of width "
                      f"(structure too cheap — quotes suspect)")
    return float(debit), "ok"


def fly_close_credit(*, wing_in_bid: float, body_ask: float,
                     wing_out_bid: float) -> float:
    """Executable unwind CREDIT: SELL both wings at BID, BUY 2× body at ASK.
    Conservative on every leg (the mirror of price_fly)."""
    return float(wing_in_bid + wing_out_bid - 2.0 * body_ask)


def fly_pnl(*, debit: float, close_credit: float, lot: int, lots: int,
            open_q: dict, close_q: dict) -> float:
    """Realized after-cost ₹ across all FOUR legs' round trips (each leg
    bought-then-sold or sold-then-bought through the shared toll booth).
      gross = (close_credit − debit) × qty
    open_q / close_q carry the per-leg fill prices used at entry and exit."""
    q = lot * lots
    gross = (close_credit - debit) * q
    # wing_in: bought@open_ask, sold@close_bid
    c_wi = round_trip_costs(open_q["wing_in_ask"] * q,
                            close_q["wing_in_bid"] * q)
    # wing_out: bought@open_ask, sold@close_bid
    c_wo = round_trip_costs(open_q["wing_out_ask"] * q,
                            close_q["wing_out_bid"] * q)
    # body: 2 contracts, sold@open_bid, bought@close_ask
    c_bd = round_trip_costs(close_q["body_ask"] * 2 * q,
                            open_q["body_bid"] * 2 * q)
    return float(gross - c_wi - c_wo - c_bd)


def fly_max_value(wing_width: float) -> float:
    """Per-unit value if spot pins the body at expiry (= the wing width)."""
    return float(wing_width)


def size_fly(debit: float, wing_width: float, lot: int,
             capital: float) -> int:
    """Fixed-fractional, IDENTICAL to shortvol.size_lots: the debit paid IS
    the max loss (debit·lot·lots), capped at SV_RISK_PCT% of capital AND the
    global deployment cap. Same two-layer mechanism the whole system uses —
    fractional risk, no Kelly (the VRP edge is true-p > risk-neutral-p)."""
    risk_per = debit * lot
    if risk_per <= 0:
        return 0
    budget = min(capital * config.SV_RISK_PCT / 100.0,
                 capital * config.MAX_KELLY_BUDGET_PCT)
    return int(budget // risk_per)


def fly_pop_prior(debit: float, wing_width: float) -> float:
    """Crude prior P(profit): the fly profits inside a band around the body
    whose half-width is (wing_width − breakeven_offset). Breakevens sit at
    body ± debit, so the profitable span is 2·(wing_width − debit) out of the
    total 2·wing_width the structure covers. Clamped to [0.05, 0.95]."""
    if wing_width <= 0:
        return 0.5
    span = max(wing_width - debit, 0.0)
    return float(min(max(span / wing_width, 0.05), 0.95))


# ============================================================ spec + book
@dataclass
class FlySpec:
    index: str
    side: str                       # body option type (CE/PE) — the wall's
    body_k: float                   # the wall
    wing_in_k: float                # nearer-the-money wing
    wing_out_k: float               # further-OTM wing
    wing_width: float               # |body − wing| (symmetric)
    lot: int
    body_symbol: str = ""
    wing_in_symbol: str = ""
    wing_out_symbol: str = ""
    body_token: int = 0
    wing_in_token: int = 0
    wing_out_token: int = 0
    exchange: str = "NFO"


@dataclass
class OpenFly:
    spec: FlySpec
    fly_id: str
    debit: float
    lots: int
    open_ts: float
    open_hm: str
    open_q: dict                    # per-leg entry fills (for cost accounting)
    pop: float
    mode: str
    max_loss: float = field(init=False)

    def __post_init__(self):
        # a long fly's max loss is exactly the debit paid × qty
        self.max_loss = self.debit * self.spec.lot * self.lots


def build_fly(index: str, side: str, strike_step: float,
              call_wall: float, put_wall: float,
              rungs: list[dict]) -> tuple[FlySpec | None, str]:
    """Centre the fly on the TESTED wall (same wall the gate grants), wings
    SV_FLY_WING_STEPS on each side, all legs resolved from the SAME OTM ladder
    the long engine walks. Returns (spec, reason)."""
    wall = call_wall if side == "CE" else put_wall
    body_k = float(wall)
    w = strike_step * config.SV_FLY_WING_STEPS
    # wing_in = toward the money, wing_out = further OTM (mirror for PE)
    if side == "CE":
        wing_in_k, wing_out_k = body_k - w, body_k + w
    else:
        wing_in_k, wing_out_k = body_k + w, body_k - w
    by_k = {float(r["strike"]): r for r in rungs}
    b = by_k.get(body_k)
    wi = by_k.get(wing_in_k)
    wo = by_k.get(wing_out_k)
    for leg, k, nm in ((b, body_k, "body"), (wi, wing_in_k, "inner wing"),
                       (wo, wing_out_k, "outer wing")):
        if leg is None:
            return None, f"{nm} {k:.0f} beyond harvested ladder"
    return FlySpec(
        index=index, side=side, body_k=body_k,
        wing_in_k=wing_in_k, wing_out_k=wing_out_k,
        wing_width=abs(w), lot=int(b["lot"]),
        body_symbol=b["symbol"], wing_in_symbol=wi["symbol"],
        wing_out_symbol=wo["symbol"], body_token=int(b["token"]),
        wing_in_token=int(wi["token"]), wing_out_token=int(wo["token"]),
        exchange=b.get("exchange", "NFO")), "ok"


class FlyBook:
    """ONE butterfly at a time, globally (capital truth). Mirrors SpreadBook:
    mechanics only, writes FLY_OPEN/FLY_CLOSE to the SAME ledger (ignored by
    the long book's edge-auditor — the fly certifies through its OWN harness),
    and registers max-loss (= debit) into the single RiskGovernor so one
    drawdown constitution governs everything.

    BUY-ONLY: try_open pays a debit; there is no live-routing short path and
    no naked leg. Live routing (all four legs) is a future tranche gated on
    the fly's certificate — until then this is the paper-explore engine that
    accrues the forward evidence, exactly as shortvol did before it.
    """

    def __init__(self, risk=None, ledger_path=None):
        self.risk = risk
        self.ledger = Path(ledger_path or config.LEDGER_PATH)
        self.pos: OpenFly | None = None
        self.last_try: dict[str, float] = {}
        self.closed_today = 0

    # ------------------------------------------------------------- ledger
    def _row(self, event: str, fp: OpenFly, ts: float, pnl=None,
             reason: str = "") -> None:
        try:
            from core.position_manager import LEDGER_FIELDS
            new = not self.ledger.exists()
            with self.ledger.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, LEDGER_FIELDS)
                if new:
                    w.writeheader()
                w.writerow({
                    "ts": f"{ts:.3f}", "event": event, "index": fp.spec.index,
                    "symbol": (f"{fp.spec.wing_in_symbol}+"
                               f"{fp.spec.body_symbol}x2+"
                               f"{fp.spec.wing_out_symbol}"),
                    "direction": f"LONG_{fp.spec.side}_FLY",
                    "qty": fp.spec.lot * fp.lots,
                    "price": round(fp.debit, 2),
                    "value": round(fp.max_loss, 2),
                    "conviction": "", "win_prob": round(fp.pop, 3),
                    "pnl": ("" if pnl is None else round(pnl, 2)),
                    "costs": "", "reason": f"{fp.fly_id} {reason}".strip(),
                    "order_id": fp.fly_id})
        except Exception:                                 # noqa: BLE001
            pass

    # ------------------------------------------------------------- quotes
    @staticmethod
    def _legs_from(spec: FlySpec, quotes: dict):
        return (quotes.get(spec.wing_in_token) or {},
                quotes.get(spec.body_token) or {},
                quotes.get(spec.wing_out_token) or {})

    # --------------------------------------------------------------- open
    def try_open(self, *, ts: float, hm: str, spec: FlySpec,
                 quotes: dict, capital: float, mode: str) -> dict:
        """quotes: {token: {'bid','ask'}} for all THREE distinct strikes.
        Returns an event row (opened or the named refusal). BUY-ONLY."""
        base = {"ts": ts, "hm": hm, "index": spec.index, "side": spec.side,
                "body_k": spec.body_k, "wing_in_k": spec.wing_in_k,
                "wing_out_k": spec.wing_out_k}
        if self.pos is not None:
            return {**base, "skip": "book occupied"}
        if ts - self.last_try.get(spec.index, -1e9) \
                < config.SV_ATTEMPT_THROTTLE_S:
            return {**base, "skip": "throttled"}
        self.last_try[spec.index] = ts
        if self.risk is not None and getattr(self.risk, "halted", False):
            return {**base, "skip": "risk halted"}
        wi, bd, wo = self._legs_from(spec, quotes)
        debit, why = price_fly(wing_in=wi, body=bd, wing_out=wo,
                               wing_width=spec.wing_width)
        if debit is None:
            return {**base, "skip": why}
        lots = size_fly(debit, spec.wing_width, spec.lot, capital)
        if lots < 1:
            return {**base, "skip": f"unaffordable at {capital:.0f} "
                    f"(debit/lot {debit * spec.lot:.0f})"}
        open_q = {"wing_in_ask": float(wi["ask"]),
                  "wing_out_ask": float(wo["ask"]),
                  "body_bid": float(bd["bid"])}
        fp = OpenFly(spec=spec, fly_id=f"FL{int(ts)}{spec.index[:2]}",
                     debit=debit, lots=lots, open_ts=ts, open_hm=hm,
                     open_q=open_q,
                     pop=fly_pop_prior(debit, spec.wing_width), mode=mode)
        self.pos = fp
        if self.risk is not None:
            try:
                self.risk.register_entry(fp.max_loss)
            except Exception:                             # noqa: BLE001
                pass
        self._row("FLY_OPEN", fp, ts,
                  reason=f"debit {debit:.2f} pop~{fp.pop:.2f} "
                         f"lots {lots} mode {mode}")
        log_forward({"fly_id": fp.fly_id, "phase": "open", "entry_ts": ts,
                     "index": spec.index, "side": spec.side,
                     "body_k": spec.body_k, "wing_width": spec.wing_width,
                     "debit": round(debit, 2), "lots": lots,
                     "max_loss": round(fp.max_loss, 2), "mode": mode})
        return {**base, "opened": fp.fly_id, "debit": round(debit, 2),
                "lots": lots, "max_loss": round(fp.max_loss, 2),
                "pop": round(fp.pop, 2)}

    # ------------------------------------------------- mark (no side effects)
    def mark(self, *, ts: float, spot: float, quotes: dict) -> dict | None:
        """Live vitals WITHOUT touching the position — parity with
        SpreadBook.mark / PositionManager.status. Unrealized mark-to-market,
        live close credit, % of the way to the profit target and the max-loss
        floor, distance of spot from the body (the pin the fly wants)."""
        fp = self.pos
        if fp is None:
            return None
        wi, bd, wo = self._legs_from(fp.spec, quotes)
        wib, ba, wob = (float(wi.get("bid") or 0), float(bd.get("ask") or 0),
                        float(wo.get("bid") or 0))
        if wib > 0 and ba > 0 and wob > 0:
            cc = fly_close_credit(wing_in_bid=wib, body_ask=ba,
                                  wing_out_bid=wob)
            live = True
        else:                                    # dead book: worthless unwind
            cc = 0.0
            live = False
        q = fp.spec.lot * fp.lots
        unreal = (cc - fp.debit) * q             # gross mark (costs on close)
        maxv = fly_max_value(fp.spec.wing_width)
        # target = capture TP_FRAC of the debit→max-value journey
        tgt_credit = fp.debit + config.SV_FLY_TP_FRAC * (maxv - fp.debit)
        to_target = ((cc - fp.debit) / max(tgt_credit - fp.debit, 1e-9)) * 100
        # loss floor is total debit (cc → 0); how far down we are
        to_floor = ((fp.debit - cc) / max(fp.debit, 1e-9)) * 100
        dist_body = abs(spot - fp.spec.body_k)
        return {"fly_id": fp.fly_id, "index": fp.spec.index,
                "side": fp.spec.side, "body_k": fp.spec.body_k,
                "wing_in_k": fp.spec.wing_in_k,
                "wing_out_k": fp.spec.wing_out_k,
                "wing_in_symbol": fp.spec.wing_in_symbol,
                "body_symbol": fp.spec.body_symbol,
                "wing_out_symbol": fp.spec.wing_out_symbol,
                "debit": round(fp.debit, 2), "close_credit": round(cc, 2),
                "unreal": round(unreal, 2), "lots": fp.lots,
                "pop": round(fp.pop, 2),
                "to_target_pct": round(to_target, 0),
                "to_floor_pct": round(to_floor, 0),
                "dist_from_body": round(dist_body, 0),
                "held_s": int(ts - fp.open_ts), "live_quote": live,
                "mode": fp.mode, "max_loss": fp.max_loss}

    # ------------------------------------------------------- manage / close
    def manage(self, *, ts: float, hm: str, spot: float, quotes: dict,
               cascade_event: bool) -> dict | None:
        """Every-tick exit engine (buy-only; no live short unwind path).
        Exits on: cascade veto, hard-flat time, profit target hit, or the
        debit-fraction stop. Returns a close row when it exits."""
        fp = self.pos
        if fp is None:
            return None
        wi, bd, wo = self._legs_from(fp.spec, quotes)
        wib, ba, wob = (float(wi.get("bid") or 0), float(bd.get("ask") or 0),
                        float(wo.get("bid") or 0))
        why = None
        cc = None
        if cascade_event:
            why = "CASCADE_VETO"
        elif hm >= config.SV_CLOSE_HM:
            why = "TIME_FLAT"
        elif wib > 0 and ba > 0 and wob > 0:
            cc = fly_close_credit(wing_in_bid=wib, body_ask=ba,
                                  wing_out_bid=wob)
            maxv = fly_max_value(fp.spec.wing_width)
            tgt = fp.debit + config.SV_FLY_TP_FRAC * (maxv - fp.debit)
            if cc >= tgt:
                why = "TARGET"
            elif cc <= fp.debit * (1.0 - config.SV_FLY_SL_FRAC):
                why = "STOP"
        if why is None:
            return None
        # resolve the executable unwind (dead book → worthless, full debit loss)
        if not (wib > 0 and ba > 0 and wob > 0):
            wib = wob = 0.05
            ba = fp.spec.wing_width            # body worth ≈ width if pinned
        cc = fly_close_credit(wing_in_bid=wib, body_ask=ba, wing_out_bid=wob)
        close_q = {"wing_in_bid": wib, "wing_out_bid": wob, "body_ask": ba}
        pnl = fly_pnl(debit=fp.debit, close_credit=cc, lot=fp.spec.lot,
                      lots=fp.lots, open_q=fp.open_q, close_q=close_q)
        if self.risk is not None:
            try:
                self.risk.register_exit(fp.max_loss, pnl, "FLY", ts=ts)
            except Exception:                             # noqa: BLE001
                pass
        self._row("FLY_CLOSE", fp, ts, pnl=pnl, reason=why)
        log_forward({"fly_id": fp.fly_id, "phase": "close", "close_ts": ts,
                     "pnl": round(pnl, 2), "why": why,
                     "hold_s": int(ts - fp.open_ts)})
        row = {"fly_id": fp.fly_id, "index": fp.spec.index,
               "side": fp.spec.side, "why": why, "pnl": round(pnl, 2),
               "debit": round(fp.debit, 2), "close_credit": round(cc, 2),
               "hold_s": int(ts - fp.open_ts), "hm": hm, "mode": fp.mode}
        self.pos = None
        self.closed_today += 1
        return row