"""
APEX OMNI v9.5 — VOL-SURFACE INTELLIGENCE (Pillar 3: SVI + event variance)
==========================================================================
• RAW-SVI slice fit (Gatheral–Jacquier 2014): total variance
      w(k) = a + b·(ρ(k−m) + √((k−m)² + σ²)),  k = ln(K/F)
  fitted per index per radar sweep from the PUBLISHED per-contract IVs.
  Optimizer is deterministic and derivative-free: for fixed (m,σ) the model
  is LINEAR in (a, bρ, b) via features [1, x, √(x²+σ²)] → constrained lstsq
  on a coarse (m,σ) grid, refined once around the winner. No scipy, no
  randomness, sub-millisecond at radar sizes.
• NO-ARBITRAGE check: Durrleman's condition g(k) ≥ 0 sampled across the fit
  band (butterfly arbitrage detector) — a slice that fails is flagged, never
  silently consumed.
• Outputs: params, rmse (vol pts), ATM skew dσ/dk, per-strike rich/cheap
  residuals — the raw material for the T3 "sell the rich side" gate trial.
• EVENT VARIANCE (Dubinsky–Johannes): with ATM total implied variance at two
  tenors TV₁=σ₁²T₁, TV₂=σ₂²T₂ and a scheduled event date τ:
      τ ∈ (T₁,T₂]:  diffusive rate v ≈ TV₁/T₁;  EV = TV₂ − TV₁ − v·(T₂−T₁)
      τ ≤ T₁      :  v ≈ (TV₂−TV₁)/(T₂−T₁);    EV = TV₁ − v·T₁
  EV is the market's priced event variance; √EV is the implied event move.
  Tenor-2 data comes from the radar's new term probe (macro_term_v9).
• EVENT CALENDAR: state/event_calendar.json — a USER-CURATED list of
  [{"date":"YYYY-MM-DD","label":"RBI"}, …]. Curated beats scraped: provenance
  is the point (PROGRAM.md negative space).

Telemetry + tools only in v9.5 — gate consumption is a registered T3 trial.
"""
from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np

import config

EVENT_CAL_PATH = config.STATE_DIR / "event_calendar.json"


# ------------------------------------------------------------------ SVI
def _linear_fit(x: np.ndarray, w: np.ndarray, sig: float):
    """For fixed (m already removed into x, σ): w ≈ a + c1·x + c2·√(x²+σ²).
    Returns (a, b, rho, sse) with b ≥ 0, |ρ| ≤ 0.999 enforced."""
    f2 = np.sqrt(x * x + sig * sig)
    A = np.column_stack([np.ones_like(x), x, f2])
    coef, *_ = np.linalg.lstsq(A, w, rcond=None)
    a, c1, b = float(coef[0]), float(coef[1]), float(coef[2])
    b = max(b, 1e-9)
    rho = float(np.clip(c1 / b, -0.999, 0.999))
    fit = a + b * (rho * x + f2)
    return a, b, rho, float(np.sum((fit - w) ** 2))


def fit_svi(K, iv, F: float, T: float) -> dict | None:
    """Fit one slice. Returns params + diagnostics, or None on thin input."""
    K = np.asarray(K, float)
    iv = np.asarray(iv, float)
    ok = np.isfinite(K) & np.isfinite(iv) & (K > 0) & (iv > 0.01)
    K, iv = K[ok], iv[ok]
    if K.size < 8 or F <= 0 or T <= 0:
        return None
    k = np.log(K / F)
    w = iv * iv * T
    span = float(k.max() - k.min())
    best = None
    m_grid = np.linspace(k.min(), k.max(), 7)
    s_grid = np.geomspace(max(span / 20, 1e-3), max(span, 5e-3), 7)
    for refine in range(2):
        for m in m_grid:
            for sg in s_grid:
                a, b, rho, sse = _linear_fit(k - m, w, sg)
                if best is None or sse < best[0]:
                    best = (sse, a, b, rho, float(m), float(sg))
        _, a, b, rho, m0, s0 = best
        m_grid = np.linspace(m0 - span / 6, m0 + span / 6, 5)
        s_grid = np.geomspace(max(s0 / 2, 1e-4), s0 * 2, 5)
    sse, a, b, rho, m, sg = best

    def w_of(kq):
        x = np.asarray(kq, float) - m
        return a + b * (rho * x + np.sqrt(x * x + sg * sg))

    wf = w_of(k)
    rmse_vol = float(np.sqrt(np.mean((np.sqrt(np.maximum(wf, 1e-12) / T)
                                      - iv) ** 2)))
    # Durrleman g(k) ≥ 0 on a dense band (numerical derivatives of w)
    kk = np.linspace(k.min(), k.max(), 121)
    h = max((kk[1] - kk[0]), 1e-5)
    wv = w_of(kk)
    wp = (w_of(kk + h) - w_of(kk - h)) / (2 * h)
    wpp = (w_of(kk + h) - 2 * wv + w_of(kk - h)) / (h * h)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = ((1 - kk * wp / (2 * wv)) ** 2
             - (wp * wp / 4) * (1 / wv + 0.25) + wpp / 2)
    arb_ok = bool(np.nanmin(g) > -1e-6)
    # ATM skew dσ/dk at k=0: σ=√(w/T) ⇒ dσ/dk = w′/(2σT)
    w0 = float(w_of(0.0))
    wp0 = float((w_of(h) - w_of(-h)) / (2 * h))
    atm_iv = math.sqrt(max(w0, 1e-12) / T)
    skew = wp0 / (2 * atm_iv * T)
    resid = np.sqrt(np.maximum(wf, 1e-12) / T) - iv     # fitted − market
    return {"a": a, "b": b, "rho": rho, "m": m, "sigma": sg,
            "rmse_vol": rmse_vol, "arb_ok": arb_ok,
            "atm_iv": atm_iv, "skew": float(skew),
            "n": int(K.size),
            "rich_cheap": {float(kx): float(rz)      # +ve ⇒ market CHEAP
                           for kx, rz in zip(K, resid)}}


# ------------------------------------------------ event variance (D–J)
def event_variance(T1: float, iv1: float, T2: float, iv2: float,
                   event_T: float | None) -> dict | None:
    """Extract the priced event variance from two ATM tenors. Times in
    YEARS; event_T = event date in years from now (None ⇒ no known event:
    returns the clean forward-variance term structure instead)."""
    if not (T2 > T1 > 0 and iv1 > 0 and iv2 > 0):
        return None
    tv1, tv2 = iv1 * iv1 * T1, iv2 * iv2 * T2
    fwd_var = (tv2 - tv1) / (T2 - T1)                 # annualized fwd rate
    out = {"tv1": tv1, "tv2": tv2, "fwd_var_rate": fwd_var,
           "fwd_iv": math.sqrt(max(fwd_var, 0.0))}
    if event_T is None:
        out.update({"event_var": None, "event_move_pct": None,
                    "branch": "no-event"})
        return out
    if T1 < event_T <= T2:
        v = tv1 / T1                                  # front tenor is clean
        ev = tv2 - tv1 - v * (T2 - T1)
        branch = "event-in-back"
    else:                                             # event inside T1
        v = max(fwd_var, 0.0)                         # back tail is clean
        ev = tv1 - v * T1
        branch = "event-in-front"
    ev = max(ev, 0.0)
    out.update({"event_var": ev,
                "event_move_pct": 100.0 * math.sqrt(ev),
                "diffusive_rate": v, "branch": branch})
    return out


# ------------------------------------------------------- event calendar
def load_calendar() -> list[dict]:
    try:
        rows = json.loads(EVENT_CAL_PATH.read_text())
        return sorted((r for r in rows if r.get("date")),
                      key=lambda r: r["date"])
    except Exception:                                 # noqa: BLE001
        return []


def next_event(after: dt.date | None = None) -> dict | None:
    after = after or dt.date.today()
    for r in load_calendar():
        try:
            if dt.date.fromisoformat(r["date"]) >= after:
                return r
        except ValueError:
            continue
    return None


def is_event_day(day: dt.date | None = None) -> dict | None:
    day = day or dt.date.today()
    for r in load_calendar():
        if r.get("date") == day.isoformat():
            return r
    return None