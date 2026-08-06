"""
APEX OMNI v9.7.1 — COMMODITY BRAIN (parallel engine, heuristic-first)
=====================================================================
A complete, self-contained commodity trading engine that runs ALONGSIDE the
equity brain — never inside it. This mirrors exactly how the equity system began
before the forge trained anything: a transparent PHYSICS heuristic policy,
gated by calibration and risk, running paper-only until a commodity forge earns
the right to promote a trained model.

Why parallel and not merged into the equity brain
--------------------------------------------------
The equity brain's frame shape and action space are hardwired to the 6 equity
indices (INDEX_ORDER, ACTION_DIM=12), and the trained equity SAC/meta models
expect that exact 30×19 frame. Appending commodities would change ACTION_DIM,
break the frame, invalidate every trained equity model, and move CONFIG_HASH.
So commodities get their OWN frame, OWN policy, OWN governor — isolated, so they
can never destabilize the equity book, and each evolves independently. This is
the safe, correct architecture, not a compromise.

What is REAL here (no mockup)
-----------------------------
  • Feature nodes are computed by the EQUITY StateBuilder.leg_features — the
    exact same validated 19-feature code path, just over commodity instruments.
    No divergence.
  • The decision is the SAME transparent physics as the equity bootstrap policy
    (CE-vs-PE order-flow imbalance, dealer inventory, velocity, momentum),
    weighted by config.COMMODITY_HEURISTIC_W. Untrained, honest, inspectable.
  • Every entry is gated by (a) the scheduled-EVENT guard (blackout/settle), and
    (b) the TRADE-ELIGIBILITY gate (Track-A + Track-B calibration + operator
    opt-in). A commodity with no calibration simply never trades.
  • Dynamic stop/target come from the commodity's OWN calibrated volatility.

What is NOT here yet, by design
-------------------------------
  • No trained meta-labeler / SAC. Those require a commodity forge over weeks of
    harvested ticks — the commodity analog of the equity forge. Until then the
    engine runs heuristic-only, exactly as equity did pre-promotion. The hooks
    to load a promoted commodity model are present (load_commodity_meta) and
    return None until such an artifact exists.

Paper-only, always: LIVE_FIRE stays False AND no commodity trades unless it is
in COMMODITY_TRADABLE (empty by default) with both calibration tracks green.

  python core/commodity_brain.py        # self-test (synthetic market)
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                             # noqa: E402
from core.market_state import StateBuilder                 # noqa: E402
from core import calibration as CAL                        # noqa: E402
from core.event_engine import (CommodityEventEngine,       # noqa: E402
                               event_entry_gate)

import logging                                            # noqa: E402
log = logging.getLogger("commodity_brain")

_LEGS = ["spot", "atm_ce", "atm_pe", "otm_ce", "otm_pe"]


# --------------------------------------------------------------------------
# Commodity heuristic policy — same physics as the equity bootstrap policy,
# per commodity, with its own weights. Reads the 19-feature nodes.
# --------------------------------------------------------------------------
class CommodityHeuristicPolicy:
    """Transparent physics signal per commodity. Returns a dict {commodity:
    action in [-1,1]} where sign = direction (CE if >0), |value| = conviction.
    Identical structure to HeuristicPolicy but keyed by commodity name."""

    def predict(self, nodes_by_commodity: dict) -> dict:
        w_ofi, w_dlr, w_vel, w_mom = getattr(
            config, "COMMODITY_HEURISTIC_W", config.HEURISTIC_W)
        out = {}
        for name, nodes in nodes_by_commodity.items():
            spot, ce, pe = nodes[0], nodes[1], nodes[2]
            if not ce.any() and not pe.any():
                continue                       # no option legs streaming
            flow = (w_ofi * (ce[12] - pe[12]) / 4.0     # OFI z differential
                    + w_dlr * (ce[16] - pe[16])         # dealer inventory
                    + w_vel * (ce[5] - pe[5]))          # velocity differential
            mom = w_mom * float(spot[0])                # spot log-ret (×100)
            out[name] = math.tanh(flow + mom)
        return out


# --------------------------------------------------------------------------
# Promoted-model hook (returns None until a commodity forge exists)
# --------------------------------------------------------------------------
def load_commodity_meta():
    """Load a promoted commodity meta-labeler if one exists. Returns None until
    a commodity forge produces one — the engine then runs heuristic-only,
    exactly as the equity brain did before its first promotion."""
    try:
        import json
        p = config.MODEL_DIR / "commodity_meta.json"
        if p.exists():
            j = json.loads(p.read_text())
            _ah = j.get("config_hash")     # v9.9.1: same fail-closed rule
            _auc_c = j.get("auc_cal", j.get("auc"))
            _bar_c = float(getattr(config, "META_MIN_AUC", 0.52))
            if _auc_c is None or float(_auc_c) < _bar_c:   # v9.9.9: same
                log.error("commodity meta REJECTED: recorded AUC %s < %.2f "
                          "— never demonstrated ranking ability; heuristic "
                          "only.", "absent" if _auc_c is None
                          else f"{float(_auc_c):.4f}", _bar_c)
                return None
            if _ah and _ah != config.CONFIG_HASH:   # as the equity loader
                log.error("commodity meta REJECTED: trained under config "
                          "%s, running %s — heuristic-only until the "
                          "commodity forge re-trains.", _ah,
                          config.CONFIG_HASH)
                return None
            return j
    except Exception:                                          # noqa: BLE001
        pass
    return None


# --------------------------------------------------------------------------
# The commodity brain
# --------------------------------------------------------------------------
@dataclass
class CommodityDecision:
    commodity: str
    direction: str            # "CE" | "PE"
    conviction: float         # |signal| in [0,1]
    allowed: bool             # passed all gates?
    reason: str               # why (blocked reason or "entered")
    sl_pct: float = 0.0
    tp_pct: float = 0.0
    win_prob: float = 0.0     # meta P(win) when a promoted model exists,
    #                           else conviction — flows into the Kelly gate
    probe: bool = False       # v9.9: ambiguous EV zone → minimum-size entry
    meta_zone: str = ""       # v9.9: FULL | PROBE | VETO | MONITOR ("" legacy)


class CommodityBrain:
    """Runs the heuristic policy over harvested commodities, gated by the event
    guard and the calibration eligibility gate. Isolated from equity."""

    def __init__(self):
        self.sb = StateBuilder()              # reuse the validated feature path
        self.policy = CommodityHeuristicPolicy()
        self.events = CommodityEventEngine()
        self.meta = load_commodity_meta()     # None until the commodity forge runs
        self._commodities = list(getattr(config, "HARVEST_COMMODITIES", []))
        from collections import deque
        self._spot_hist = {c: deque(maxlen=600) for c in self._commodities}
        self._last_nodes: dict[str, np.ndarray] = {}
        if self.meta is None:
            log.info("commodity brain: heuristic-only (no promoted model yet) "
                     "— physics policy, calibration + event gated")
        else:
            log.info("commodity brain: PROMOTED meta active (%s, n=%s) — "
                     "meta P(win) feeds the Kelly gate",
                     self.meta.get("engine"), self.meta.get("n"))

    def _meta_x(self, name: str, nodes: np.ndarray, direction: str,
                now_ist) -> np.ndarray | None:
        """Build the FORGE-IDENTICAL x-vector:
        [spot_node, atm_ce_node, atm_pe_node, t/N, kaufman_er, capped_mom30,
        ±1 direction]. Returns P(win) or None (no model / scoring failed)."""
        if self.meta is None:
            return None
        try:
            import math as _m
            from core.meta_gbm import score_vec
            # commodity session window — SAME derivation as the forge's
            # _commodity_window (parity by construction, both config-driven)
            def _sod(hm):
                h, m = (int(x) for x in hm.split(":"))
                return h * 3600 + m * 60
            _t0 = _sod(getattr(config, "COMMODITY_SESSION_OPEN", "09:00"))
            _closes = [_sod(v.get("session_close", "23:30")) for v in
                       getattr(config, "COMMODITIES", {}).values()] or [56100]
            _N = max(_closes) - _t0
            h = list(self._spot_hist.get(name, []))
            er = 0.0
            if len(h) >= 30:
                a = np.asarray(h, float)
                churn = float(np.sum(np.abs(np.diff(a))))
                er = float((a[-1] - a[0]) / churn) if churn > 0 else 0.0
            f30 = (h[-1] / h[-31] - 1.0) if len(h) > 31 and h[-31] else 0.0
            t_sec = (now_ist.hour * 3600 + now_ist.minute * 60
                     + now_ist.second) - _t0
            t_frac = min(max(t_sec, 0), _N) / _N
            x = np.concatenate([nodes[0], nodes[1], nodes[2],
                                [t_frac, er,
                                 _m.copysign(min(abs(f30) * 100, 3), f30)
                                 if f30 else 0.0,
                                 1.0 if direction == "CE" else -1.0]]
                               ).astype(np.float32)
            return x
        except Exception as e:                                 # noqa: BLE001
            log.debug("meta x-build failed (%s)", e)
            return None

    def _meta_wp(self, name: str, nodes: np.ndarray, direction: str,
                 now_ist) -> float | None:
        x = self._meta_x(name, nodes, direction, now_ist)
        if x is None or self.meta is None:
            return None
        try:
            from core.meta_gbm import score_vec
            return score_vec(self.meta, x)
        except Exception as e:                                 # noqa: BLE001
            log.debug("meta scoring failed (%s) — using conviction", e)
            return None

    def _meta_interval(self, name: str, nodes: np.ndarray, direction: str,
                       now_ist):
        """v9.9: (p0, p1, p_merged, integrity) from the commodity artifact's
        Venn-Abers payload, or None (legacy point path)."""
        x = self._meta_x(name, nodes, direction, now_ist)
        if x is None or self.meta is None:
            return None
        try:
            from core.meta_gbm import score_interval
            return score_interval(self.meta, x)
        except Exception as e:                                 # noqa: BLE001
            log.debug("meta interval failed (%s)", e)
            return None

    def _nodes_for(self, name: str, ctx: dict, ts: float) -> np.ndarray:
        """Build the 5×19 node block for one commodity via the equity
        StateBuilder.leg_features (same validated computation)."""
        F = config.FEATURES_PER_NODE
        block = np.zeros((len(_LEGS), F), np.float32)
        spot_snap = ctx.get("spot") or {}
        spot = float(spot_snap.get("ltp") or 0.0)
        exp = ctx.get("expiry", "")
        dte = float(ctx.get("dte", 20.0))     # commodities are monthly
        T = float(ctx.get("T", max(dte, 0.05) / 365.0))
        block[0] = self.sb.leg_features(f"{name}:SPOT", spot_snap, index=name,
                                        expiry=exp, strike=0.0, opt_type="SPOT",
                                        spot=spot, T_years=T, dte=dte,
                                        is_weekly=False, ts=ts)
        for j, leg in enumerate(_LEGS[1:], start=1):
            info = (ctx.get("legs") or {}).get(leg)
            if not info or not info.get("snap"):
                continue
            block[j] = self.sb.leg_features(
                f"{name}:{leg}", info["snap"], index=name, expiry=exp,
                strike=float(info.get("strike") or 0.0),
                opt_type="CE" if leg.endswith("ce") else "PE",
                spot=spot, T_years=T, dte=dte, is_weekly=False, ts=ts)
        return block

    def decide(self, market: dict, now_ist) -> list:
        """market: {commodity: {spot, expiry, dte, T, legs:{...}}}. Returns a
        list of CommodityDecision — one per commodity with a live signal."""
        nodes = {}
        for name in self._commodities:
            ctx = market.get(name)
            if ctx:
                nodes[name] = self._nodes_for(name, ctx, now_ist.timestamp())
                self._last_nodes[name] = nodes[name]
                _sp = float((ctx.get("spot") or {}).get("ltp") or 0.0)
                if _sp > 0:
                    self._spot_hist[name].append(_sp)
        signals = self.policy.predict(nodes)

        decisions = []
        conv_bar = float(getattr(config, "COMMODITY_ENTRY_CONVICTION",
                                 config.ENTRY_CONVICTION))
        for name, sig in signals.items():
            direction = "CE" if sig > 0 else "PE"
            conviction = abs(float(sig))
            d = CommodityDecision(commodity=name, direction=direction,
                                  conviction=conviction, allowed=False,
                                  reason="")
            # gate 1: conviction bar
            if conviction < conv_bar:
                d.reason = f"conviction {conviction:.2f} < bar {conv_bar:.2f}"
                decisions.append(d)
                continue
            # gate 2: trade-eligibility (calibration + operator opt-in)
            elig, why = CAL.commodity_trade_eligible(name)
            if not elig:
                d.reason = why
                decisions.append(d)
                continue
            # gate 3: scheduled-event guard
            ev = self.events.evaluate(now_ist, name)
            allow_ev, ev_why = event_entry_gate(ev)
            if not allow_ev:
                d.reason = ev_why
                decisions.append(d)
                continue
            # passed all gates → compute dynamic stop/target from THIS
            # commodity's calibrated volatility
            ctx = market.get(name) or {}
            spot_snap = ctx.get("spot") or {}
            atm = (ctx.get("legs") or {}).get(
                "atm_ce" if direction == "CE" else "atm_pe") or {}
            entry_prem = float((atm.get("snap") or {}).get("ask")
                               or (atm.get("snap") or {}).get("ltp") or 0.0)
            lv = CAL.dynamic_stop_target(
                name, entry_premium=max(entry_prem, 0.01),
                delta=0.5, minutes_to_close=120.0, atm_iv=None)
            d.sl_pct, d.tp_pct = lv.sl_pct, lv.tp_pct
            _wp = self._meta_wp(name, self._last_nodes.get(name, np.zeros(
                (5, config.FEATURES_PER_NODE), np.float32)), direction, now_ist)
            d.win_prob = float(_wp) if _wp is not None else conviction
            # ---- v9.9 META-GATE v3 (commodities). Default "size_only":
            # a VA-capable artifact upgrades the SIZING probability to the
            # merged interval value and never vetoes (today's behavior).
            # "ev" (operator opt-in) adds the three-zone gate, with p*
            # from the SAME dynamic stop/target lv just computed. lot=1
            # here (real lot lives in the mapper, past this point), which
            # OVERSTATES flat brokerage per unit ⇒ p* conservative — the
            # safe direction for a gate.
            _ivl_c = self._meta_interval(
                name, self._last_nodes.get(name, np.zeros(
                    (5, config.FEATURES_PER_NODE), np.float32)),
                direction, now_ist)
            if _ivl_c is not None:
                _p0c, _p1c, _pmc, _intc = _ivl_c
                d.win_prob = float(_pmc)
                if getattr(config, "COMMODITY_META_GATE",
                           "size_only") == "ev" and entry_prem > 0:
                    from core import meta_gate as MGT
                    _ps_c = MGT.breakeven_p(
                        entry_prem, entry_prem * (1.0 + lv.tp_pct),
                        entry_prem * (1.0 - lv.sl_pct), 1)
                    _dec = MGT.decide(_p0c, _p1c, _ps_c,
                                      MGT.AdaptiveMargin("commodity").m,
                                      _intc)
                    d.meta_zone = _dec.zone
                    if not _dec.ok:
                        d.allowed = False
                        d.reason = _dec.reason
                        decisions.append(d)
                        continue
                    d.probe = bool(_dec.probe)
            d.allowed = True
            d.reason = (f"entered ({lv.source}"
                        + (f", meta wp {d.win_prob:.2f}" if _wp is not None
                           else "") + ")")
            decisions.append(d)
        return decisions


# --------------------------------------------------------------------------
# SELF-TEST (synthetic market; proves the gates without live data)
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import datetime as dt
    from zoneinfo import ZoneInfo
    logging.basicConfig(level=logging.INFO)
    _IST = ZoneInfo("Asia/Kolkata")

    brain = CommodityBrain()

    def _snap(ltp, bid, ask, bq=20, aq=20, oi=1000, vol=100):
        return {"ltp": ltp, "bid": bid, "ask": ask, "bid_qty": bq,
                "ask_qty": aq, "oi": oi, "vol_delta": vol}

    # a synthetic crude market with a bullish CE-vs-PE flow tilt
    market = {"CRUDEOIL": {
        "spot": _snap(6000, 5999, 6001), "expiry": "2026-08-19", "dte": 20,
        "T": 20 / 365.0,
        "legs": {
            "atm_ce": {"snap": _snap(120, 119, 121, bq=80, aq=20), "strike": 6000},
            "atm_pe": {"snap": _snap(118, 117, 119, bq=20, aq=80), "strike": 6000},
            "otm_ce": {"snap": _snap(80, 79, 81), "strike": 6050},
            "otm_pe": {"snap": _snap(78, 77, 79), "strike": 5950},
        }}}

    # a normal (non-event) time
    now = dt.datetime(2026, 7, 20, 14, 0, tzinfo=_IST)   # Monday afternoon
    print("\n=== decisions at a normal time (CRUDEOIL not in TRADABLE) ===")
    for d in brain.decide(market, now):
        print(f"  {d.commodity} {d.direction} conv={d.conviction:.2f} "
              f"allowed={d.allowed} — {d.reason}")

    # during an EIA blackout (Wed 19:45 IST) — even if it were eligible, blocked
    now_ev = dt.datetime(2026, 7, 22, 19, 45, tzinfo=_IST)
    print("\n=== decisions during EIA blackout (Wed 19:45 IST) ===")
    for d in brain.decide(market, now_ev):
        print(f"  {d.commodity} {d.direction} conv={d.conviction:.2f} "
              f"allowed={d.allowed} — {d.reason}")