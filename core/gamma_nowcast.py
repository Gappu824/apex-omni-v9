"""
APEX OMNI v9.2 — GAMMA NOWCAST (the "gex missed it" fix)
=========================================================
The macro radar sweeps a full chain every MACRO_LOOP_S (180 s) at Kite's
1-quote-req/s budget, so the gamma flip and net GEX the brain reads lag the
tape by 3–5 minutes — a full cascade's lifetime. This module closes that gap
WITHOUT a single new API call, using two facts:

  1. Black-76 gamma is IDENTICAL for calls and puts (∂²V/∂F² carries no
     option-type term), so per-contract gamma at any spot is computable from
     (K, iv, T) alone — no CE/PE flag needed.
  2. The radar's live JSON already publishes, per surviving contract, the
     strike K_i, the Newton-solved iv_i, AND the signed dealer GEX_i it
     computed at snapshot spot S₀. That triple lets us RECOVER the signed
     open-interest weight the radar used:

         w_i = GEX_i / ( γ_i(F₀) · S₀² · 0.01 )      [= sign_i · OI_i · lot]

     which is exactly the slow-moving quantity (positioning) — while the
     fast-moving quantity (gamma's dependence on spot) is closed-form.

Nowcast at any later second, spot S, elapsed Δt:

         T′  = max(dte₀ − Δt/86400, ~30 min) / 365          (theta decay)
         F′  = S · e^{r·T′}
         GEXᵢ′ = w_i · γ_i(F′) · S² · 0.01                    (per contract)

then the SAME per-strike aggregation, 1-2-3-2-1 smoothing and linear
zero-crossing the radar uses (reimplemented here as the canonical pure helper,
unit-tested for self-consistency: nowcast AT the snapshot's own spot must
reproduce the snapshot's own flip and net GEX to numerical precision).

Assumptions, named (the honest residuals between sweeps):
  • STICKY-STRIKE vol: iv_i held fixed at its snapshot value per strike until
    the next sweep (Derman 1999's regimes; the conservative practitioner
    default and the same convention the radar itself measures under). In a
    crash real skew steepens — under sticky-strike the nowcast slightly
    UNDERSTATES put gamma, i.e. the flip estimate errs conservative.
  • OI frozen between sweeps: NSE disseminates OI on a ~3-min cadence anyway,
    so this residual is bounded by the exchange, not by us.
  • Wing truncation: contracts that drift outside the radar's ±10% band since
    the snapshot aren't re-added; a validity flag trips when spot runs beyond
    the covered strike range.

Consumers: the live brain (feeds the cascade detector + heartbeat telemetry)
and tools/cascade_harness.py (identical class over archived snapshots, where
gex_json is present). Pure numpy; no torch, no kiteconnect.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import config
from core.quant_core import black76_greeks


def flip_from_profile(uniq_k: np.ndarray, net: np.ndarray):
    """Canonical smoothed zero-crossing of a per-strike net-GEX profile —
    byte-equivalent to the radar's assemble_snapshot logic (1-2-3-2-1 kernel
    when ≥5 strikes, first sign change, linear interpolation), factored pure
    so the nowcast, the radar's numbers and the harness can never diverge.
    Returns (flip, flip_width) or (None, None) when no crossing exists."""
    if len(uniq_k) >= 5:
        kern = np.array([1, 2, 3, 2, 1], float)
        kern /= kern.sum()
        sm = np.convolve(net, kern, mode="same")
    else:
        sm = net
    sgn = np.sign(sm)
    for i in range(len(sm) - 1):
        if sgn[i] != 0 and sgn[i + 1] != 0 and sgn[i] != sgn[i + 1]:
            x0, x1, y0, y1 = uniq_k[i], uniq_k[i + 1], sm[i], sm[i + 1]
            return (float(x0 - y0 * (x1 - x0) / (y1 - y0)),
                    float(x1 - x0))
    return None, None


@dataclass
class Nowcast:
    ts: float                 # nowcast time
    spot: float
    flip: float | None
    flip_width: float | None
    net_gex: float
    snapshot_ts: float        # the radar sweep this was projected from
    snapshot_age_s: float
    in_band: bool             # spot still inside the snapshot's strike range
    n_contracts: int


def recover_profile(mac: dict | None):
    """v9.4 shared recovery (used by GammaNowcast AND core/dealer_flow):
    from a radar payload carrying per-contract (K, iv, gex) at snapshot spot,
    recover the signed-OI weights w_i = gex_i / (γ_i(F₀)·S₀²·0.01). Returns
    (K, iv, w, ts0, spot0, dte0) or None when the payload is unusable."""
    if not mac:
        return None
    ts0 = float(mac.get("ts") or 0.0)
    K = np.asarray(mac.get("strikes") or [], float)
    iv = np.asarray(mac.get("iv") or [], float)
    gex = np.asarray(mac.get("gex") or [], float)
    spot0 = float(mac.get("spot") or 0.0)
    dte0 = float(mac.get("dte") or 0.0)
    if not (len(K) and len(K) == len(iv) == len(gex)
            and spot0 > 0 and dte0 > 0):
        return None
    T0 = max(dte0, 0.02) / 365.0
    F0 = spot0 * math.exp(config.RISK_FREE_RATE * T0)
    g0 = np.asarray(black76_greeks(F0, K, T0, iv, True,
                                   config.RISK_FREE_RATE)["gamma"], float)
    denom = g0 * spot0 * spot0 * 0.01
    ok = np.isfinite(denom) & (denom > 0) & np.isfinite(gex) \
        & np.isfinite(iv) & (iv > 0)
    if int(ok.sum()) < 6:
        return None
    return (K[ok], iv[ok], gex[ok] / denom[ok], ts0, spot0, dte0)


class GammaNowcast:
    """Per-index projector. Feed each fresh radar payload once
    (update_snapshot); call nowcast(spot, ts) every decision second."""

    __slots__ = ("index", "_ts0", "_spot0", "_dte0", "_K", "_iv", "_w",
                 "_kmin", "_kmax", "_valid")

    def __init__(self, index: str):
        self.index = index
        self._valid = False
        self._ts0 = 0.0

    # ------------------------------------------------------------ ingest
    def update_snapshot(self, mac: dict | None) -> bool:
        """Recover the signed-OI weights w_i from a radar payload (live JSON
        or archived row carrying per-contract 'gex'). Idempotent per ts.
        Returns True when the snapshot is usable. v9.4: recovery math lives
        in recover_profile(), shared with core/dealer_flow."""
        if not mac:
            return self._valid
        ts0 = float(mac.get("ts") or 0.0)
        if ts0 == self._ts0:                      # already ingested this sweep
            return self._valid
        rec = recover_profile(mac)
        if rec is None:
            return self._valid                    # keep last good snapshot
        self._K, self._iv, self._w, self._ts0, self._spot0, self._dte0 = rec
        self._kmin, self._kmax = float(self._K.min()), float(self._K.max())
        self._valid = True
        return True

    # ------------------------------------------------------------ project
    def nowcast(self, spot: float, ts: float) -> Nowcast | None:
        """Analytic flip / net-GEX at (spot, ts). None when no usable
        snapshot exists or the last one has gone stale past MACRO_STALE_S —
        a dead radar must not keep steering the cascade trigger."""
        if not self._valid or spot <= 0:
            return None
        age = ts - self._ts0
        if age > config.MACRO_STALE_S:
            return None
        dte = max(self._dte0 - max(age, 0.0) / 86400.0, 0.02)   # ≈30-min floor
        T = dte / 365.0
        F = spot * math.exp(config.RISK_FREE_RATE * T)
        g = np.asarray(black76_greeks(F, self._K, T, self._iv, True,
                                      config.RISK_FREE_RATE)["gamma"], float)
        gex = self._w * g * spot * spot * 0.01
        fin = np.isfinite(gex)
        if int(fin.sum()) < 6:
            return None
        Kf, gexf = self._K[fin], gex[fin]
        uniq = np.unique(Kf)
        net = np.array([gexf[Kf == k].sum() for k in uniq])
        flip, width = flip_from_profile(uniq, net)
        return Nowcast(ts=ts, spot=spot, flip=flip, flip_width=width,
                       net_gex=float(gexf.sum()), snapshot_ts=self._ts0,
                       snapshot_age_s=age,
                       in_band=bool(self._kmin <= spot <= self._kmax),
                       n_contracts=int(fin.sum()))