"""
APEX OMNI v9 — SCENARIO ENGINE
==============================
A synthetic—but microstructure-honest—trading day (09:15→15:30, one tick per
second) that the REAL v9 stack trades: the actual RiskGovernor, the actual
ExecutionEngine (paper, deterministic maker-fill model), the actual
PositionManager and the actual TrapShield. Nothing is mocked at the decision
layer — the simulator only replaces the exchange.

The generator controls, per second: the spot path (drift segments + noise +
event overlays), option IV (steps for crush/spike), spread multiplier
(liquidity pulls), absorption / aggressive-sell flags (trap fingerprints),
ΔOI behaviour (confirming vs non-confirming breaks), feed staleness, and GEX
walls. Premiums are repriced every second with Black-76 from (spot, IV,
time-to-expiry), so premium↔spot consistency — including theta bleed and the
gamma of cheap OTM wings — is automatic rather than assumed.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field

import numpy as np

import config
from core.quant_core import black76_price, black76_greeks
from core.risk_manager import RiskGovernor
from core.execution_engine import ExecutionEngine
from core.position_manager import PositionManager, LegQuote, TickContext

T0_SEC = 9 * 3600 + 15 * 60
N = 22500                                   # 09:15 → 15:30, 1 Hz
HIER_DEPTH = config.HIERARCHY_DEPTH          # sim == live, always
ATTEMPT_EVERY_S = config.ENTRY_ATTEMPT_THROTTLE_S


def hm_of(t: int) -> str:
    s = T0_SEC + t
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}"


class SimDay:
    def __init__(self, open_spot=24500.0, dte_days=2.3, base_iv=0.135,
                 base_spread_pct=0.012, lot=65, step=50, seed=7,
                 noise_pts_s=0.55, doi_per_s=40.0):
        self.rng = np.random.default_rng(seed)
        self.open = float(open_spot)
        self.dte0, self.lot, self.step = float(dte_days), int(lot), float(step)
        self.base_spread = float(base_spread_pct)
        self.drift = np.zeros(N)
        self.overlay = np.zeros(N)
        self.noise = self.rng.normal(0, noise_pts_s, N)
        self.iv = np.full(N, float(base_iv))
        self.spread_mult = np.ones(N)
        self.absorb = np.zeros(N, bool)
        self.sell_aggr = np.full(N, 0.5)
        self.doi = np.full(N, float(doi_per_s))
        self.stale = np.zeros(N, bool)
        self.put_wall = self.open - 60
        self.call_wall = self.open + 160
        self.atm0 = round(self.open / step) * step
        self._spot = None
        self._oi = None
        self._stale_age = None
        self._fro = None

    # ----------------------------------------------------- path authoring
    def trend(self, hm0: str, hm1: str, pts_per_min: float):
        a, b = self._idx(hm0), self._idx(hm1)
        self.drift[a:b] += pts_per_min / 60.0
        return self

    def gap(self, hm: str, pts: float, over_s: int = 15):
        """Air pocket: near-instant repricing of `pts` over `over_s` seconds
        (flash crash / V-rip). No flow fingerprints — just the move."""
        t = self._idx(hm)
        self.overlay[t:t + over_s] += np.linspace(0, pts, over_s)
        self.overlay[t + over_s:] += pts
        return self

    def stop_run(self, hm: str, depth_pts=30.0, down_s=15, hold_s=25,
                 recover_pts=None, recover_s=70, spread_mult=3.0,
                 iv_bump=0.004):
        """The institutional flush: fast spike down on pulled liquidity, big
        prints at the bid getting ABSORBED, ΔOI flat (closing, not committing),
        then the reclaim."""
        t = self._idx(hm)
        rec = depth_pts if recover_pts is None else recover_pts
        d, h, r = down_s, hold_s, recover_s
        self.overlay[t:t + d] += np.linspace(0, -depth_pts, d)
        self.overlay[t + d:t + d + h] += -depth_pts
        self.overlay[t + d + h:t + d + h + r] += np.linspace(-depth_pts,
                                                             -depth_pts + rec, r)
        self.overlay[t + d + h + r:] += (-depth_pts + rec)
        w = slice(t, t + d + h)
        self.spread_mult[w] = np.maximum(self.spread_mult[w], spread_mult)
        self.absorb[t + 3:t + d + h] = True
        self.sell_aggr[w] = 0.85
        self.doi[w] = 0.0                       # nobody NEW is committing
        self.iv[t:t + d + h + r] += iv_bump
        return self

    def breakdown(self, hm: str, depth_pts=85.0, over_s=150,
                  doi_per_s=900.0, spread_mult=1.6):
        """A GENUINE break: slower, sustained, sellers keep pressing, fresh
        OI piles onto the move, no absorption at the lows."""
        t = self._idx(hm)
        self.overlay[t:t + over_s] += np.linspace(0, -depth_pts, over_s)
        self.overlay[t + over_s:] += -depth_pts
        w = slice(t, t + over_s)
        self.sell_aggr[w] = 0.8
        self.absorb[w] = False
        self.doi[w] = doi_per_s
        self.spread_mult[w] = np.maximum(self.spread_mult[w], spread_mult)
        return self

    def iv_step(self, hm: str, new_iv: float, over_s: int = 5):
        t = self._idx(hm)
        self.iv[t:t + over_s] = np.linspace(self.iv[t], new_iv, over_s)
        self.iv[t + over_s:] = new_iv
        return self

    def widen_spread(self, hm0: str, hm1: str, mult: float):
        self.spread_mult[self._idx(hm0):self._idx(hm1)] *= mult
        return self

    def feed_stale(self, hm: str, dur_s: int):
        t = self._idx(hm)
        self.stale[t:t + dur_s] = True
        return self

    def walls(self, put=None, call=None):
        if put is not None: self.put_wall = put
        if call is not None: self.call_wall = call
        return self

    def _idx(self, hm: str) -> int:
        h, m = map(int, hm.split(":"))
        return max(0, min(N - 1, h * 3600 + m * 60 - T0_SEC))

    # ----------------------------------------------------- finalize / quotes
    def finalize(self):
        self._spot = self.open + np.cumsum(self.drift + self.noise) + self.overlay
        self._oi = 5_000_000 + np.cumsum(self.doi)
        age = np.zeros(N)
        run = 0
        self._fro = np.arange(N)
        for t in range(N):
            if self.stale[t]:
                run += 1
                self._fro[t] = self._fro[t - 1] if t else 0
            else:
                run = 0
            age[t] = run
        self._stale_age = age
        return self

    def spot(self, t): return float(self._spot[self._fro[t]])
    def true_spot(self, t): return float(self._spot[t])

    def T_years(self, t):
        return max(self.dte0 - t / N, 0.02) / 365.0

    def token(self, strike: float, is_call: bool) -> int:
        return int(10_000 + round(strike) * 2 + (0 if is_call else 1))

    def quote(self, t: int, strike: float, is_call: bool) -> dict:
        te = int(self._fro[t])
        S = float(self._spot[te])
        T = self.T_years(te)
        F = S * math.exp(config.RISK_FREE_RATE * T)
        mid = float(black76_price(F, strike, T, self.iv[te], is_call,
                                  config.RISK_FREE_RATE))
        mid = max(mid, 0.10)
        half = max(mid * self.base_spread * self.spread_mult[te] / 2, 0.05)
        return {"ltp": round(mid, 2), "bid": round(max(mid - half, 0.05), 2),
                "ask": round(mid + half, 2), "bid_qty": 600, "ask_qty": 600}

    def delta(self, t: int, strike: float, is_call: bool) -> float:
        te = int(self._fro[t])
        S = float(self._spot[te]); T = self.T_years(te)
        F = S * math.exp(config.RISK_FREE_RATE * T)
        return float(black76_greeks(F, strike, T, self.iv[te], is_call,
                                    config.RISK_FREE_RATE)["delta"])


@dataclass
class Signal:
    hm: str
    conviction: float           # signed; CE if >0
    win_prob: float = 0.72
    window_s: int = 300


@dataclass
class Scenario:
    name: str
    desc: str
    day: SimDay
    signals: list[Signal]
    pass_fn: object
    inject_rejects: int = 0
    cfg: dict | None = None   # per-scenario config overrides (applied+restored)
    meta_live: bool = False   # simulate a TRAINED meta-model live (feeds the
    #                           signal's win_prob into live_win_prob so the
    #                           model-shaped exit path is exercised)


@dataclass
class Result:
    name: str
    desc: str
    trades: list = field(default_factory=list)
    pnl: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    trap_holds: int = 0
    trap_confirmed: int = 0
    blocked: int = 0
    skipped: int = 0
    rejects: int = 0
    nofills: int = 0
    exit_reasons: list = field(default_factory=list)
    violations: list = field(default_factory=list)
    events: list = field(default_factory=list)
    ok: bool = False
    note: str = ""


def run_scenario(sc: Scenario) -> Result:
    _saved = {}
    if sc.cfg:
        for k, v in sc.cfg.items():
            _saved[k] = getattr(config, k)
            setattr(config, k, v)
    try:
        return _run_scenario_inner(sc)
    finally:
        for k, v in _saved.items():
            setattr(config, k, v)


def _run_scenario_inner(sc: Scenario) -> Result:
    day = sc.day.finalize()
    cur = {"t": 0.0}
    tokmap: dict[int, tuple] = {}

    def quote_fn(token):
        if token in tokmap:
            k, c = tokmap[token]
            return day.quote(int(cur["t"]), k, c)
        return {}

    def lookahead(token, horizon_s):
        if token not in tokmap:
            return None, None
        k, c = tokmap[token]
        t = int(cur["t"])
        asks, bids = [], []
        for dt_ in range(0, int(horizon_s) + 1):
            q = day.quote(min(t + dt_, N - 1), k, c)
            asks.append(q["ask"]); bids.append(q["bid"])
        return min(asks), max(bids)

    risk = RiskGovernor()
    eng = ExecutionEngine(kite=None, quote_fn=quote_fn,
                          clock=lambda: cur["t"])
    eng.lookahead_fn = lookahead
    eng.paper_fill_realism = False     # logic test: deterministic touch-fills
    eng.inject_rejects = sc.inject_rejects
    ledger = config.LOG_DIR / f"sim_{sc.name}.csv"
    if ledger.exists():
        ledger.unlink()
    pm = PositionManager("NIFTY", risk, eng, ledger_path=ledger)

    sig_windows = [(day._idx(s.hm), day._idx(s.hm) + s.window_s, s)
                   for s in sc.signals]
    last_attempt = -1e9
    halt_ts = None
    entries = {}

    for t in range(N):
        cur["t"] = float(t)
        risk.on_tick()
        if risk.halted and halt_ts is None:
            halt_ts = t
        active = next((s for a, b, s in sig_windows if a <= t < b), None)
        conv = active.conviction if active else 0.0
        wp = active.win_prob if active else 0.5
        te = int(day._fro[t])
        spot = day.spot(t)
        vel = spot - day.spot(t - 1) if t else 0.0
        oi_now = day._oi[te]
        oi_then = day._oi[max(te - 60, 0)]
        # When a scenario simulates a TRAINED model live (sc.meta_live), feed the
        # signal's win_prob into live_win_prob — exactly as the brain feeds the
        # position-direction meta P(win). None otherwise (bootstrap path).
        _wp_live = wp if getattr(sc, "meta_live", False) else None
        ctx = TickContext(
            ts=float(t), hm=hm_of(t), spot=spot, spot_velocity_1s=vel,
            data_age_s=float(day._stale_age[t]),
            atm_iv=float(day.iv[te]), minutes_to_close=(N - t) / 60.0,
            gex_put_wall=day.put_wall, gex_call_wall=day.call_wall,
            absorption=bool(day.absorb[te]),
            aggressive_sell_ratio=float(day.sell_aggr[te]),
            oi_delta_since=float((oi_now - oi_then) / max(oi_now, 1.0)),
            avg_spread_pct=day.base_spread, conviction=conv,
            live_win_prob=_wp_live)

        if pm.pos is not None:
            pm.manage(ctx, quote_fn(pm.pos.token))
        elif (active and abs(conv) >= config.ENTRY_CONVICTION
              and not risk.halted and t - last_attempt >= ATTEMPT_EVERY_S):
            last_attempt = t
            d = "CE" if conv > 0 else "PE"
            sgn = 1 if d == "CE" else -1
            atm_now = round(spot / day.step) * day.step
            hier = []
            for i in range(HIER_DEPTH):
                K = atm_now + sgn * i * day.step
                q = day.quote(t, K, d == "CE")
                tok = day.token(K, d == "CE")
                tokmap[tok] = (K, d == "CE")
                hier.append(LegQuote(
                    leg=f"{'+' if sgn > 0 else '-'}{i}", symbol=f"NIFTY{K:.0f}{d}",
                    exchange="NFO", token=tok, strike=K,
                    premium=(q["bid"] + q["ask"]) / 2, bid=q["bid"],
                    ask=q["ask"], bid_qty=q["bid_qty"], ask_qty=q["ask_qty"],
                    lot=day.lot, delta=day.delta(t, K, d == "CE"),
                    dte=day.dte0))
            pm.try_enter(ctx, d, conv, wp, hier)

    # ---------------------------------------------------------- summarize
    res = Result(sc.name, sc.desc, events=pm.events,
                 pnl=risk.realized_pnl, halted=risk.halted,
                 halt_reason=risk.halt_reason)
    for e in pm.events:
        ev = e["event"]
        if ev == "TRAP_HOLD": res.trap_holds += 1
        elif ev == "TRAP_CONFIRMED": res.trap_confirmed += 1
        elif ev == "BLOCKED": res.blocked += 1
        elif ev == "SKIP": res.skipped += 1
        elif ev == "REJECT": res.rejects += 1
        elif ev == "NOFILL": res.nofills += 1
        elif ev == "BUY_FILL":
            entries[e["symbol"]] = e
            if halt_ts is not None and float(e["ts"]) > halt_ts:
                res.violations.append("BUY after risk halt")
        elif ev == "SELL_FILL":
            b = entries.get(e["symbol"])
            if b:
                ep, xp = float(b["price"]), float(e["price"])
                loss_frac = (ep - xp) / ep
                reason = e["reason"]
                res.trades.append({"sym": e["symbol"], "entry": ep, "exit": xp,
                                   "pnl": float(e["pnl"]),
                                   "reason": reason,
                                   "t_in": float(b["ts"]), "t_out": float(e["ts"])})
                res.exit_reasons.append(reason)
                if reason.startswith(("STOP", "DISASTER")) and \
                        loss_frac > config.ABS_DISASTER_PCT + 0.08:
                    res.violations.append(
                        f"{e['symbol']} loss {loss_frac:.0%} beyond floor+tol")
    if pm.pos is not None:
        res.violations.append("position still open at session end")
    ok, note = sc.pass_fn(res)
    res.ok, res.note = ok, note
    return res
