"""
APEX OMNI v9.3 — SHORT-VOL ENGINE (shared: harness ≡ brain)
============================================================
The other side of the trade this system PROVED exists. Every honest exam
converged on one number: long premium at these horizons wins ~22% after costs
(meta base_rate 0.2185 on 429 real triple-barrier labels; 7/7 walk-forward
losers). The counterparty of that condemned trade is the variance risk
premium — the most robustly documented edge in options (Carr–Wu RFS 2009;
Bakshi–Kapadia 2003; Bollerslev–Tauchen–Zhou 2009; Israelov–Nielsen;
Beckmeyer–Branger–Gayda 2023 on 0DTE buyers losing systematically).

REGIME DISCIPLINE (the part most retail short-vol blows up on): premium is
sold ONLY when the dealer book DAMPS moves — positive net gamma (Baltussen et
al. JFE 2021: mean reversion concentrates under long gamma), spot inside a
real wall corridor with rich IV rank — and NEVER while the cascade module's
negative-gamma machinery is active or cooling down. The cascade detector is
this engine's structural crash-veto: sell the calm, own the storm, zero
overlap by construction.

INSTRUMENT: defined-risk vertical credit spread. SHORT the tested WALL strike
(walls are where dealer hedging defends price in +gamma), LONG one step
further out. Max loss = (width − credit) × lot × lots, known at entry.
Executable pricing throughout: credit = short-leg BID − long-leg ASK at open;
unwind cost = short-leg ASK − long-leg BID; four real order legs of Zerodha
costs. Prespecified single spec — no strike/width/exit optimization; the
sensitivity grid in the harness is diagnostic only.

ONE state machine, imported verbatim by tools/shortvol_harness.py (grades
every historical opportunity on the vault) and by apex_main_v9 (runs it live,
certificate-staged) — the core/decision.py constitutional pattern. Staging is
identical to cascade: telemetry → PAPER-EXPLORE forward evidence →
certificate → (much later) live behind cert + the four locks. v9.3.0's
ceiling is PAPER: live spread ROUTING is deliberately unbuilt until a
certificate justifies it.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import config
from core.execution_engine import round_trip_costs

FORWARD_LOG = config.STATE_DIR / "shortvol_forward.jsonl"


# ------------------------------------------------------------------ lock
def sv_knob_hash() -> str:
    knobs = (config.CONFIG_HASH,
             config.SV_IVRANK_MIN, config.SV_NET_GEX_MIN,
             config.SV_WALL_BUFFER_STEPS, config.SV_CORRIDOR_MIN_STEPS,
             config.SV_DTE_MIN, config.SV_DTE_MAX, config.SV_AFTER_HM,
             config.SV_WIDTH_STEPS, config.SV_MIN_CREDIT_FRAC,
             config.SV_TP_FRAC, config.SV_SL_CREDIT_MULT,
             config.SV_TOUCH_EXIT, config.SV_CLOSE_HM,
             config.SV_ATTEMPT_THROTTLE_S, config.SV_POP_HAIRCUT,
             config.SV_RISK_PCT, config.MAX_ENTRY_SPREAD_PCT,
             # cascade veto geometry is part of the graded spec
             config.CASCADE_NET_GEX_MAX, config.CASCADE_COOLDOWN_S)
    return hashlib.sha1(repr(knobs).encode()).hexdigest()[:10]


def load_certificate() -> dict | None:
    """Valid, knob-matching shortvol certificate — else None (paper/telemetry
    staging applies). Fail-closed, exactly like the cascade lock."""
    try:
        c = json.loads(config.SHORTVOL_CERT_PATH.read_text())
    except Exception:                                     # noqa: BLE001
        return None
    if not (bool(c.get("ok"))
            and c.get("knob_hash") == sv_knob_hash()
            and (time.time() - float(c.get("ts", 0)))
            < config.EDGE_CERT_VALID_DAYS * 86400):
        return None
    return c


def shortvol_mode(cert: dict | None) -> str:
    """'certified' | 'paper-explore' | 'telemetry'. Paper-explore requires
    NOT live-armed — this tier is physically incapable of touching money.
    NOTE v9.3.0: even 'certified' cannot route live orders (routing unbuilt);
    the mode string still resolves so the staging ladder reads truthfully."""
    if not config.SHORTVOL_ENABLED:
        return "telemetry"
    if cert is not None:
        return "certified"
    if getattr(config, "SHORTVOL_PAPER_EXPLORE", False) \
            and not config.live_fire_armed():
        return "paper-explore"
    return "telemetry"


def log_forward(row: dict) -> None:
    """Append-only forward-evidence record (open AND close rows share a
    spread_id). Never raises into the trading loop."""
    try:
        FORWARD_LOG.parent.mkdir(exist_ok=True)
        with FORWARD_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:                                     # noqa: BLE001
        pass


# ------------------------------------------------------------------ gate
@dataclass
class GateVerdict:
    ok: bool
    side: str | None          # short-leg type: "CE" (bear call) | "PE" (bull put)
    reason: str
    corridor_steps: float = 0.0
    iv_rank: float | None = None
    net_gex: float | None = None


def evaluate_gate(*, hm: str, spot: float, mac: dict | None,
                  net_gex_now: float | None, dte: float | None,
                  strike_step: float, vix_bump: float,
                  cascade_blocked: bool) -> GateVerdict:
    """The prespecified sell-permission stack — pure, shared brain/harness.
    `net_gex_now` is the 1 Hz nowcast when fresh (preferred), else the
    radar's; `cascade_blocked` = zone-active OR inside cascade cooldown for
    this index (the structural crash-veto)."""
    if cascade_blocked:
        return GateVerdict(False, None, "cascade veto (zone/cooldown)")
    if vix_bump > 0:
        return GateVerdict(False, None, "VIX 5-min spike — never sell into it")
    if hm < config.SV_AFTER_HM:
        return GateVerdict(False, None, f"before {config.SV_AFTER_HM}")
    if hm >= config.SV_CLOSE_HM:
        return GateVerdict(False, None, "past hard-flat time")
    if not mac:
        return GateVerdict(False, None, "no macro snapshot")
    ivr = mac.get("iv_rank")
    if ivr is None:
        return GateVerdict(False, None, "iv_rank warming (needs 10d history)")
    if float(ivr) < config.SV_IVRANK_MIN:
        return GateVerdict(False, None,
                           f"iv_rank {float(ivr):.2f}<{config.SV_IVRANK_MIN}")
    gex = net_gex_now if net_gex_now is not None else mac.get("net_gex")
    if gex is None or float(gex) < config.SV_NET_GEX_MIN:
        return GateVerdict(False, None,
                           f"net_gex {0 if gex is None else gex:.2e} not long-"
                           f"gamma (≥{config.SV_NET_GEX_MIN:.0e})", 0.0,
                           float(ivr), None if gex is None else float(gex))
    cw, pw = mac.get("call_wall"), mac.get("put_wall")
    if not (cw and pw and cw > pw):
        return GateVerdict(False, None, "no wall corridor", 0.0, float(ivr),
                           float(gex))
    buf = config.SV_WALL_BUFFER_STEPS * strike_step
    corridor = (cw - pw) / strike_step
    if corridor < config.SV_CORRIDOR_MIN_STEPS:
        return GateVerdict(False, None,
                           f"corridor {corridor:.1f} steps < "
                           f"{config.SV_CORRIDOR_MIN_STEPS:.0f}",
                           corridor, float(ivr), float(gex))
    if not (pw + buf <= spot <= cw - buf):
        return GateVerdict(False, None, "spot outside corridor buffer",
                           corridor, float(ivr), float(gex))
    d = float(dte if dte is not None else 9.0)
    if not (config.SV_DTE_MIN <= d <= config.SV_DTE_MAX):
        return GateVerdict(False, None, f"dte {d:.2f} outside "
                           f"[{config.SV_DTE_MIN},{config.SV_DTE_MAX}]",
                           corridor, float(ivr), float(gex))
    # sell the TESTED wall: the nearer one is the magnet dealers defend
    side = "CE" if (cw - spot) <= (spot - pw) else "PE"
    return GateVerdict(True, side, "sell-permission granted",
                       corridor, float(ivr), float(gex))


# ------------------------------------------------------------------ build
@dataclass
class SpreadSpec:
    index: str
    side: str                 # short-leg type
    short_k: float
    long_k: float
    width: float
    lot: int
    short_symbol: str = ""
    long_symbol: str = ""
    short_token: int = 0
    long_token: int = 0
    exchange: str = "NFO"


def build_spread(index: str, side: str, strike_step: float,
                 call_wall: float, put_wall: float,
                 rungs: list[dict]) -> tuple[SpreadSpec | None, str]:
    """SHORT the wall strike, LONG one step further out, legs resolved from
    the same OTM ladder the long engine walks (mapper.hierarchy for `side`).
    Returns (spec, reason)."""
    wall = call_wall if side == "CE" else put_wall
    short_k = float(wall)
    long_k = short_k + (strike_step * config.SV_WIDTH_STEPS
                        if side == "CE" else
                        -strike_step * config.SV_WIDTH_STEPS)
    by_k = {float(r["strike"]): r for r in rungs}
    s, l = by_k.get(short_k), by_k.get(long_k)
    if s is None:
        return None, f"short leg {short_k:.0f} beyond harvested ladder"
    if l is None:
        return None, f"long leg {long_k:.0f} beyond harvested ladder"
    return SpreadSpec(index=index, side=side, short_k=short_k, long_k=long_k,
                      width=abs(long_k - short_k), lot=int(s["lot"]),
                      short_symbol=s["symbol"], long_symbol=l["symbol"],
                      short_token=int(s["token"]), long_token=int(l["token"]),
                      exchange=s.get("exchange", "NFO")), "ok"


def price_open(short_bid: float, short_ask: float,
               long_bid: float, long_ask: float,
               width: float) -> tuple[float | None, str]:
    """Executable opening credit: SELL short leg at its BID, BUY long leg at
    its ASK. Rejects one-sided books, illiquid legs, junk premium."""
    for b, a, name in ((short_bid, short_ask, "short"),
                       (long_bid, long_ask, "long")):
        if not (b and a) or b <= 0 or a <= 0:
            return None, f"{name} leg one-sided"
        mid = (a + b) / 2.0
        if (a - b) / max(mid, 0.05) > config.MAX_ENTRY_SPREAD_PCT:
            return None, f"{name} leg spread>cap"
    credit = short_bid - long_ask
    if credit < config.SV_MIN_CREDIT_FRAC * width:
        return None, (f"credit {credit:.2f} < "
                      f"{config.SV_MIN_CREDIT_FRAC:.0%} of width")
    return float(credit), "ok"


def close_cost(short_ask: float, long_bid: float) -> float:
    """Executable unwind: BUY back the short at its ASK, SELL the long at its
    BID. Conservative on both legs."""
    return float(short_ask - long_bid)


def spread_pnl(credit: float, close_c: float, lot: int, lots: int,
               open_short_bid: float, open_long_ask: float,
               close_short_ask: float, close_long_bid: float) -> float:
    """Realized after-cost ₹: gross (credit − unwind) minus FOUR real order
    legs of costs — short leg's round trip (sold first, bought back) and long
    leg's (bought first, sold back), each through the shared toll booth."""
    q = lot * lots
    gross = (credit - close_c) * q
    c_short = round_trip_costs(close_short_ask * q, open_short_bid * q)
    c_long = round_trip_costs(open_long_ask * q, close_long_bid * q)
    return float(gross - c_short - c_long)


def size_lots(credit: float, width: float, lot: int, capital: float) -> int:
    """Fixed-fractional: (width−credit)·lot·lots ≤ SV_RISK_PCT% of capital,
    also bounded by the global deployment cap. Kelly is deliberately NOT used
    here — fed the risk-neutral pop it is ≈0 by construction (the VRP edge IS
    true-p > risk-neutral-p); fractional risk is the literature-standard
    prespecification for systematic premium selling."""
    risk_per = (width - credit) * lot
    if risk_per <= 0:
        return 0
    budget = min(capital * config.SV_RISK_PCT / 100.0,
                 capital * config.MAX_KELLY_BUDGET_PCT)
    return int(budget // risk_per)


def pop_prior(credit: float, width: float) -> float:
    """Reporting prior only (NOT sizing): risk-neutral P(expire OTM) with a
    haircut, clamped to a sane band."""
    p = (1.0 - credit / max(width, 1e-9)) * config.SV_POP_HAIRCUT
    return float(min(max(p, 0.50), 0.90))


# ------------------------------------------------------------------ book
@dataclass
class OpenSpread:
    spec: SpreadSpec
    spread_id: str
    credit: float
    lots: int
    open_ts: float
    open_hm: str
    open_short_bid: float
    open_long_ask: float
    pop: float
    mode: str
    live: bool = False                        # v9.5: routed via spread_router
    max_loss: float = field(init=False)

    def __post_init__(self):
        self.max_loss = (self.spec.width - self.credit) \
            * self.spec.lot * self.lots


class SpreadBook:
    """ONE spread at a time, globally (capital truth). Mechanics only — the
    gate lives in evaluate_gate; the caller decides when to try. Writes
    SPREAD_OPEN/SPREAD_CLOSE rows into the SAME execution ledger (events the
    edge-auditor ignores by design — spread evidence certifies through its
    OWN harness, never contaminating the long book's certificate) and
    registers max-loss capital into the ONE RiskGovernor so a single
    drawdown constitution governs everything."""

    def __init__(self, risk=None, ledger_path=None, kite=None):
        self.risk = risk
        self.kite = kite                      # live routing handle (v9.5)
        self.ledger = Path(ledger_path or config.LEDGER_PATH)
        self.pos: OpenSpread | None = None
        self.last_try: dict[str, float] = {}
        self.closed_today = 0

    # ---------------- ledger
    def _row(self, event: str, sp: OpenSpread, ts: float, pnl=None,
             reason: str = "") -> None:
        try:
            from core.position_manager import LEDGER_FIELDS
            new = not self.ledger.exists()
            with self.ledger.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, LEDGER_FIELDS)
                if new:
                    w.writeheader()
                w.writerow({
                    "ts": f"{ts:.3f}", "event": event, "index": sp.spec.index,
                    "symbol": f"{sp.spec.short_symbol}/-{sp.spec.long_symbol}",
                    "direction": f"SHORT_{sp.spec.side}_SPREAD",
                    "qty": sp.spec.lot * sp.lots,
                    "price": round(sp.credit, 2),
                    "value": round(sp.max_loss, 2),
                    "conviction": "", "win_prob": round(sp.pop, 3),
                    "pnl": ("" if pnl is None else round(pnl, 2)),
                    "costs": "", "reason": f"{sp.spread_id} {reason}".strip(),
                    "order_id": sp.spread_id})
        except Exception:                                 # noqa: BLE001
            pass

    # ---------------- open
    def try_open(self, *, ts: float, hm: str, spec: SpreadSpec,
                 quotes: dict, capital: float, mode: str) -> dict:
        """quotes: {token: {'bid','ask'}} for both legs. Returns an event row
        (opened or the named refusal)."""
        base = {"ts": ts, "hm": hm, "index": spec.index, "side": spec.side,
                "short_k": spec.short_k, "long_k": spec.long_k}
        if self.pos is not None:
            return {**base, "skip": "book occupied"}
        if ts - self.last_try.get(spec.index, -1e9) \
                < config.SV_ATTEMPT_THROTTLE_S:
            return {**base, "skip": "throttled"}
        self.last_try[spec.index] = ts
        if self.risk is not None and getattr(self.risk, "halted", False):
            return {**base, "skip": "risk halted"}
        sq = quotes.get(spec.short_token) or {}
        lq = quotes.get(spec.long_token) or {}
        credit, why = price_open(float(sq.get("bid") or 0),
                                 float(sq.get("ask") or 0),
                                 float(lq.get("bid") or 0),
                                 float(lq.get("ask") or 0), spec.width)
        if credit is None:
            return {**base, "skip": why}
        lots = size_lots(credit, spec.width, spec.lot, capital)
        if lots < 1:
            return {**base, "skip": f"unaffordable at {capital:.0f} "
                    f"(risk/lot {(spec.width-credit)*spec.lot:.0f})"}
        # ---- v9.5 LIVE ROUTING (quintuple-locked; core/spread_router) --
        _live = False
        _sfill, _lfill = float(sq["bid"]), float(lq["ask"])
        if mode == "certified" and self.kite is not None:
            from core import spread_router as SR
            _ok, _why = SR.live_spread_allowed(load_certificate())
            if _ok:
                _r = SR.route_open(self.kite, spec, lots, quotes)
                if not _r.get("ok"):
                    return {**base, "skip": f"live route: {_r.get('why')}"}
                _live = True
                _sfill = float(_r["short_fill"])
                _lfill = float(_r["long_fill"])
                credit = _sfill - _lfill      # realized, not quoted
        sp = OpenSpread(spec=spec,
                        spread_id=f"SV{int(ts)}{spec.index[:2]}",
                        credit=credit, lots=lots, open_ts=ts, open_hm=hm,
                        open_short_bid=_sfill,
                        open_long_ask=_lfill,
                        pop=pop_prior(credit, spec.width), mode=mode,
                        live=_live)
        self.pos = sp
        if self.risk is not None:
            try:
                self.risk.register_entry(sp.max_loss)
            except Exception:                             # noqa: BLE001
                pass
        self._row("SPREAD_OPEN", sp, ts,
                  reason=f"credit {credit:.2f} pop~{sp.pop:.2f} "
                         f"lots {lots} mode {mode}")
        log_forward({"spread_id": sp.spread_id, "phase": "open",
                     "entry_ts": ts, "index": spec.index, "side": spec.side,
                     "short_k": spec.short_k, "long_k": spec.long_k,
                     "credit": round(credit, 2), "lots": lots,
                     "max_loss": round(sp.max_loss, 2), "mode": mode})
        return {**base, "opened": sp.spread_id, "credit": round(credit, 2),
                "lots": lots, "max_loss": round(sp.max_loss, 2),
                "pop": round(sp.pop, 2)}

    # ---------------- mark (telemetry only; NO side effects) ----------
    def mark(self, *, ts: float, spot: float, quotes: dict) -> dict | None:
        """Live vitals of the open spread WITHOUT touching it — the exact
        parity of PositionManager.status(): mark-to-market unrealized P&L,
        live close cost, distance to the TARGET and STOP thresholds manage()
        enforces, pop, and hold time. Uses the same leg quotes manage() reads
        this same tick, so it adds no API load. Returns None when flat."""
        sp = self.pos
        if sp is None:
            return None
        sq = quotes.get(sp.spec.short_token) or {}
        lq = quotes.get(sp.spec.long_token) or {}
        sa, lb = float(sq.get("ask") or 0), float(lq.get("bid") or 0)
        if sa > 0 and lb > 0:
            cc = close_cost(sa, lb)
            live = True
        else:                                   # dead book: intrinsic-worst
            cc = sp.credit * (1.0 + config.SV_SL_CREDIT_MULT)
            live = False
        unreal = spread_pnl(sp.credit, cc, sp.spec.lot, sp.lots,
                            sp.open_short_bid, sp.open_long_ask,
                            sa if live else sp.open_short_bid + sp.spec.width,
                            lb if live else max(sp.open_long_ask
                                                - sp.spec.width, 0.05))
        tgt_cc = sp.credit * (1.0 - config.SV_TP_FRAC)   # close cost at TARGET
        stp_cc = sp.credit * (1.0 + config.SV_SL_CREDIT_MULT)  # …at STOP
        # % of the credit→target and credit→stop journey the mark has traveled
        to_target = ((sp.credit - cc) / max(sp.credit - tgt_cc, 1e-9)) * 100
        to_stop = ((cc - sp.credit) / max(stp_cc - sp.credit, 1e-9)) * 100
        touch = ((sp.spec.side == "CE" and spot >= sp.spec.short_k)
                 or (sp.spec.side == "PE" and spot <= sp.spec.short_k))
        return {"spread_id": sp.spread_id, "index": sp.spec.index,
                "side": sp.spec.side, "short_k": sp.spec.short_k,
                "long_k": sp.spec.long_k,
                "short_symbol": sp.spec.short_symbol,
                "long_symbol": sp.spec.long_symbol,
                "credit": round(sp.credit, 2),
                "close_cost": round(cc, 2), "unreal": round(unreal, 2),
                "lots": sp.lots, "pop": round(sp.pop, 2),
                "to_target_pct": round(to_target, 0),
                "to_stop_pct": round(to_stop, 0),
                "held_s": int(ts - sp.open_ts), "live_quote": live,
                "touch": touch, "mode": sp.mode, "max_loss": sp.max_loss}

    # ---------------- manage / close
    def manage(self, *, ts: float, hm: str, spot: float, quotes: dict,
               cascade_event: bool) -> dict | None:
        """Every-tick exit engine. Returns a close row when it exits."""
        sp = self.pos
        if sp is None:
            return None
        sq = quotes.get(sp.spec.short_token) or {}
        lq = quotes.get(sp.spec.long_token) or {}
        sa, lb = float(sq.get("ask") or 0), float(lq.get("bid") or 0)
        why = None
        if cascade_event:
            why = "CASCADE_VETO"
        elif hm >= config.SV_CLOSE_HM:
            why = "TIME_FLAT"
        elif config.SV_TOUCH_EXIT and (
                (sp.spec.side == "CE" and spot >= sp.spec.short_k)
                or (sp.spec.side == "PE" and spot <= sp.spec.short_k)):
            why = "SHORT_STRIKE_TOUCH"
        elif sa > 0 and lb > 0:
            cc = close_cost(sa, lb)
            if cc <= sp.credit * (1.0 - config.SV_TP_FRAC):
                why = "TARGET"
            elif cc >= sp.credit * (1.0 + config.SV_SL_CREDIT_MULT):
                why = "STOP"
        if why is None:
            return None
        if not (sa > 0 and lb > 0):
            # forced exit with a dead book: mark at intrinsic-worst (max loss)
            sa = sp.open_short_bid + sp.spec.width
            lb = max(sp.open_long_ask - sp.spec.width, 0.05)
        if sp.live and self.kite is not None:
            from core import spread_router as SR
            _r = SR.route_close(self.kite, sp, quotes)
            if not _r.get("ok"):
                import logging
                logging.getLogger("shortvol").critical(
                    "LIVE spread close failed: %s - retrying next tick",
                    _r.get("why"))
                return None       # never fake-close a live book
            sa = float(_r["short_fill"])
            lb = float(_r["long_fill"])
        cc = close_cost(sa, lb)
        pnl = spread_pnl(sp.credit, cc, sp.spec.lot, sp.lots,
                         sp.open_short_bid, sp.open_long_ask, sa, lb)
        if self.risk is not None:
            try:
                self.risk.register_exit(sp.max_loss, pnl, "SPREAD", ts=ts)
            except Exception:                             # noqa: BLE001
                pass
        self._row("SPREAD_CLOSE", sp, ts, pnl=pnl, reason=why)
        log_forward({"spread_id": sp.spread_id, "phase": "close",
                     "close_ts": ts, "pnl": round(pnl, 2), "why": why,
                     "hold_s": int(ts - sp.open_ts)})
        row = {"spread_id": sp.spread_id, "index": sp.spec.index,
               "side": sp.spec.side, "why": why, "pnl": round(pnl, 2),
               "credit": round(sp.credit, 2), "close_cost": round(cc, 2),
               "hold_s": int(ts - sp.open_ts), "hm": hm, "mode": sp.mode}
        self.pos = None
        self.closed_today += 1
        return row