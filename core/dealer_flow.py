"""
APEX OMNI v9.4 — DEALER-FLOW STATE ENGINE (Pillar 2 core)
=========================================================
Gamma was chapter one. Dealer books also bleed delta through TIME (charm) and
through VOL moves (vanna) — scheduled, mechanical hedging flows documented
across the demand-based option-pricing literature (Barbon–Buraschi 2021;
Ni–Pearson–Poteshman 2005; Baltussen et al. 2021; pinning: Avellaneda–Lipkin
2003, Golez–Jackwerth 2012). This module turns the radar's own per-contract
profile into a 1 Hz **dealer-flow vector** per index, with ZERO new API calls:

  • weight recovery — the gamma-nowcast trick, shared verbatim via
    gamma_nowcast.recover_profile(): w_i = sign_i·OI_i·lot from published GEX.
  • greeks — CENTRAL DIFFERENCES ON THE BLACK-76 PRICER'S OWN DELTA. No
    hand-derived charm/vanna formulas to get wrong: the pricer is the ground
    truth and the unit tests compare against it directly. Vanna is call≡put
    exactly (verified numerically); charm carries a small r-driven call/put
    asymmetry, so the option TYPE is inferred from sign(w) under the radar's
    DEFAULT dealer convention (+calls/−puts). If a custom participant_oi.json
    is installed, that inference is unsafe → the vector flags
    signs_inferred=False and charm is computed under the symmetric (call)
    approximation, error bounded by r·e^{-rT} ≈ 0.07/yr on delta — stated,
    not hidden.

Outputs per second:
  charm_flow_units_min — underlying units dealers must trade PER MINUTE as
      time passes, holding all else fixed (+ = dealer buying pressure);
      charm_flow_rs_min = × spot (₹ notional/min).
  vanna_units_volpt   — units dealers must trade per +1 IV point;
      vanna_rs_volpt = × spot.
  pin — per requested wall strike, that strike's share of total |gamma|
      exposure (0..1): the concentration that anchors pinning.

CONSTITUTION: telemetry + report ONLY in v9.4. Certified gate specs are never
widened silently — a pin-strength shortvol gate or a vanna cascade input is a
REGISTERED trial with its own harness pass (PROGRAM.md, Pillar 2 T3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import config
from core.quant_core import black76_greeks
from core.gamma_nowcast import recover_profile

_DT_MIN_Y = 1.0 / (365.0 * 24.0 * 60.0)      # one minute, in years
_DVOL = 0.01                                  # one vol point


@dataclass
class DealerFlowState:
    ts: float
    spot: float
    net_gex: float
    charm_flow_units_min: float
    charm_flow_rs_min: float
    vanna_units_volpt: float
    vanna_rs_volpt: float
    pin: dict = field(default_factory=dict)   # {strike: gamma-share 0..1}
    signs_inferred: bool = True
    snapshot_age_s: float = 0.0
    n_contracts: int = 0


class DealerFlow:
    """Per-index. Feed each fresh radar payload once (update_snapshot);
    call vector(spot, ts, walls=(...)) every decision second."""

    __slots__ = ("index", "_K", "_iv", "_w", "_is_call", "_ts0", "_spot0",
                 "_dte0", "_valid", "_signs_ok")

    def __init__(self, index: str):
        self.index = index
        self._valid = False
        self._ts0 = 0.0
        # custom participant signs make the call/put inference from sign(w)
        # unsafe — detect once (file the radar reads for custom signs)
        self._signs_ok = not (config.STATE_DIR / "participant_oi.json").exists()

    def update_snapshot(self, mac: dict | None) -> bool:
        if not mac:
            return self._valid
        if float(mac.get("ts") or 0.0) == self._ts0:
            return self._valid
        rec = recover_profile(mac)
        if rec is None:
            return self._valid
        self._K, self._iv, self._w, self._ts0, self._spot0, self._dte0 = rec
        # option type from dealer-sign convention (+call / −put by default)
        self._is_call = (self._w > 0) if self._signs_ok \
            else np.ones_like(self._w, bool)
        self._valid = True
        return True

    def vector(self, spot: float, ts: float,
               walls: tuple | None = None) -> DealerFlowState | None:
        """The 1 Hz dealer-flow state at (spot, ts). None when no usable or
        fresh snapshot (same MACRO_STALE_S discipline as the nowcast)."""
        if not self._valid or spot <= 0:
            return None
        age = ts - self._ts0
        if age > config.MACRO_STALE_S:
            return None
        dte = max(self._dte0 - max(age, 0.0) / 86400.0, 0.02)
        T = dte / 365.0
        r = config.RISK_FREE_RATE
        F = spot * math.exp(r * T)

        def _delta(Tq: float, ivq: np.ndarray) -> np.ndarray:
            Fq = spot * math.exp(r * Tq)
            gc = black76_greeks(Fq, self._K, Tq, ivq, True, r)["delta"]
            gp = black76_greeks(Fq, self._K, Tq, ivq, False, r)["delta"]
            return np.where(self._is_call, np.asarray(gc, float),
                            np.asarray(gp, float))

        d0 = _delta(T, self._iv)
        # charm: forward difference toward expiry, one minute — the actual
        # question ("what happens over the NEXT minute"), floor T for 0DTE
        T1 = max(T - _DT_MIN_Y, 1e-6)
        d_dt = _delta(T1, self._iv) - d0
        # vanna: central difference in vol, one point
        d_up = _delta(T, self._iv + _DVOL)
        d_dn = _delta(T, np.maximum(self._iv - _DVOL, 1e-4))
        d_dv = (d_up - d_dn) / 2.0

        fin = np.isfinite(d0) & np.isfinite(d_dt) & np.isfinite(d_dv)
        if int(fin.sum()) < 6:
            return None
        w, K = self._w[fin], self._K[fin]
        # dealer hedge H = Σ w·Δ  ⇒ required trade per minute / per vol point
        charm_units = float(np.sum(w * d_dt[fin]))
        vanna_units = float(np.sum(w * d_dv[fin]))
        # net-GEX (identical math to the nowcast, for one-call convenience)
        g = np.asarray(black76_greeks(F, K, T, self._iv[fin], True,
                                      r)["gamma"], float)
        gex_i = w * g * spot * spot * 0.01
        gfin = np.isfinite(gex_i)
        net_gex = float(gex_i[gfin].sum())
        # pin: each wall strike's share of total |gamma| exposure
        pin: dict = {}
        if walls:
            tot = float(np.sum(np.abs(w[gfin] * g[gfin])))
            for ks in walls:
                if ks is None:
                    continue
                m = gfin & (np.abs(K - float(ks)) < 1e-6)
                pin[float(ks)] = (float(np.sum(np.abs(w[m] * g[m]))) / tot
                                  if tot > 0 else 0.0)
        return DealerFlowState(
            ts=ts, spot=spot, net_gex=net_gex,
            charm_flow_units_min=charm_units,
            charm_flow_rs_min=charm_units * spot,
            vanna_units_volpt=vanna_units,
            vanna_rs_volpt=vanna_units * spot,
            pin=pin, signs_inferred=self._signs_ok,
            snapshot_age_s=age, n_contracts=int(fin.sum()))