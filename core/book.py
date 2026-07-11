"""
APEX OMNI v9.6 — PORTFOLIO BOOK (net-greeks telemetry)
======================================================
The two engines — long-options cascade legs and short-vol spreads — read as
ONE book: per index, the net delta / gamma / vega / theta of everything
open, in rupees, at heartbeat cadence. Telemetry only: no gate consumes it
(a book-level risk gate would be a registered trial like everything else).

Conventions (₹, practitioner units):
  delta_units    Σ qty·Δ            (underlying units; qty is signed,
                                     lot-multiplied — short legs negative)
  delta_rs       delta_units × spot (₹ P&L per 100% move; divide by 100
                                     for per-1%)
  gamma_rs_1pct  Σ qty·γ·spot²·0.01 (Δ-units gained per 1% spot move,
                                     × spot ⇒ ₹ of new delta per 1%)
  vega_rs_volpt  Σ qty·ν·0.01       (₹ per +1 IV point)
  theta_rs_day   Σ qty·θ / 365      (₹ per calendar day)

Per-leg IV is solved from the leg's own quote mid (Black-76 Newton); when a
quote is missing the index ATM IV stands in and the output flags est_iv —
approximation stated, never silent. Time-to-expiry comes from the radar's
front-tenor dte (every open leg is front-expiry by construction of the
hierarchy and the spread spec).
"""
from __future__ import annotations

import math

from core.quant_core import black76_greeks, implied_vol_newton

import config


def leg_greeks(*, strike: float, is_call: bool, qty: int, spot: float,
               dte: float, iv: float, r: float = None) -> dict | None:
    """Signed greek contributions of one leg (qty already lot-multiplied)."""
    if not (spot > 0 and strike > 0 and dte > 0 and iv > 0 and qty != 0):
        return None
    r = config.RISK_FREE_RATE if r is None else r
    T = max(dte, 0.02) / 365.0
    F = spot * math.exp(r * T)
    g = black76_greeks(F, strike, T, iv, is_call, r)
    d, gm = float(g["delta"]), float(g["gamma"])
    ve, th = float(g["vega"]), float(g["theta"])
    return {"delta_units": qty * d,
            "gamma_units_1pct": qty * gm * spot * 0.01,
            "vega_rs_volpt": qty * ve * 0.01,
            "theta_rs_day": qty * th / 365.0}


def compute_book(legs: list[dict], ctx_by_index: dict) -> dict:
    """legs: [{index, strike, is_call, qty, mid|None}, …] (qty signed,
    lot-multiplied). ctx_by_index: {idx: {spot, dte, atm_iv}} from the live
    tape + freshest radar payload. Returns per-index aggregates + totals."""
    out: dict = {}
    for leg in legs:
        idx = leg["index"]
        ctx = ctx_by_index.get(idx) or {}
        spot = float(ctx.get("spot") or 0)
        dte = float(ctx.get("dte") or 0)
        if not (spot > 0 and dte > 0):
            continue
        r = config.RISK_FREE_RATE
        T = max(dte, 0.02) / 365.0
        F = spot * math.exp(r * T)
        iv, est = None, False
        mid = leg.get("mid")
        if mid and mid > 0:
            iv = implied_vol_newton(float(mid), F, float(leg["strike"]), T,
                                    bool(leg["is_call"]), r)
        if not iv or not (0.02 < iv < 3.0):
            iv, est = float(ctx.get("atm_iv") or 0) or None, True
        if not iv:
            continue
        lg = leg_greeks(strike=float(leg["strike"]),
                        is_call=bool(leg["is_call"]), qty=int(leg["qty"]),
                        spot=spot, dte=dte, iv=iv)
        if lg is None:
            continue
        b = out.setdefault(idx, {"delta_units": 0.0, "delta_rs": 0.0,
                                 "gamma_rs_1pct": 0.0, "vega_rs_volpt": 0.0,
                                 "theta_rs_day": 0.0, "n_legs": 0,
                                 "est_iv": False})
        b["delta_units"] += lg["delta_units"]
        b["gamma_rs_1pct"] += lg["gamma_units_1pct"] * spot
        b["vega_rs_volpt"] += lg["vega_rs_volpt"]
        b["theta_rs_day"] += lg["theta_rs_day"]
        b["n_legs"] += 1
        b["est_iv"] = b["est_iv"] or est
        b["delta_rs"] = b["delta_units"] * spot
    for b in out.values():
        for k in ("delta_units", "delta_rs", "gamma_rs_1pct",
                  "vega_rs_volpt", "theta_rs_day"):
            b[k] = round(b[k], 2)
    return out