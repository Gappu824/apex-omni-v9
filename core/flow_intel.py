"""
APEX OMNI v9.9 — FLOW INTELLIGENCE (the sellers' footprint, read live)
======================================================================
You buy options; the writers ARE the market's structure. This module turns
each macro chain sweep into a nowcast of what the SELLERS just did, so the
buyer trades their forced hedging instead of against it:

  • ΔOI WRITER FLOW (₹-weighted): open interest only rises when new contracts
    are WRITTEN. Between consecutive sweeps, Δoi>0 at a strike is fresh supply
    there; premium×lot-weighting turns contracts into rupees of new short
    option inventory. Split by side: call-writing pins/caps from above,
    put-writing supports from below (Barbon–Buraschi dealer-flow mechanism,
    expressed as FLOW rather than level — the level is the GEX map the system
    already has). Sign convention: positive = net NEW writing since the last
    sweep; negative = writers covering (unpinning risk).
  • MAX PAIN + drift: the expiry-magnet strike minimizing total option value
    paid out; its DRIFT across sweeps shows the magnet being dragged by flow.
  • PCR (OI): put/call open-interest ratio of the harvested band — a
    positioning tilt, not a prophecy; consumed as a prior only.
  • PIN PRIOR: exp(−|spot−maxpain| / (κ·step-adjusted scale)) — a [0,1]
    pinning prior for the butterfly's body placement (Avellaneda–Lipkin
    pinning literature). Telemetry now; a registered spec before it gates.
  • FUTURES BASIS: front-future mid vs spot, annualized — carry/positioning
    read from an instrument you never trade (price-discovery literature:
    basis shocks lead spot volatility).

Everything is O(strikes) per sweep, stdlib+numpy only, and STATELESS to the
caller: the previous-sweep OI cache lives inside this module keyed by index,
age-gated (a stale previous sweep yields flow=None, never a fake number).
"""
from __future__ import annotations

import math
import time

import numpy as np

_PREV: dict[str, dict] = {}          # index → {"ts","spot","oi":{(K,cp):oi}}
_MP_PREV: dict[str, float] = {}      # index → last max_pain (for drift)
FLOW_MAX_GAP_S = 1200                # >20 min between sweeps → no Δ claimed


def max_pain(K: list[float], oi: list[float], is_call: list[bool],
             lot: int) -> float | None:
    """Strike minimizing total intrinsic payout to option HOLDERS at expiry
    (equivalently the writers' sweet spot). O(n²) on ≤ ~60 strikes — trivial."""
    ks = sorted(set(K))
    if len(ks) < 3:
        return None
    Ka = np.asarray(K, float)
    OI = np.asarray(oi, float)
    C = np.asarray(is_call, bool)
    best_k, best_v = None, None
    for s in ks:
        call_pay = float(np.sum(np.maximum(s - Ka[C], 0.0) * OI[C]))
        put_pay = float(np.sum(np.maximum(Ka[~C] - s, 0.0) * OI[~C]))
        v = (call_pay + put_pay) * lot
        if best_v is None or v < best_v:
            best_k, best_v = float(s), v
    return best_k


def pin_prior(spot: float, mp: float | None, strike_step: float,
              pcr: float | None) -> float | None:
    """[0,1] pinning prior: distance-decayed pull toward max pain, tilted by
    extreme PCR (positioning crowding weakens the pin). κ = 3 strike-steps."""
    if mp is None or spot <= 0 or strike_step <= 0:
        return None
    base = math.exp(-abs(spot - mp) / (3.0 * strike_step))
    if pcr is not None and pcr > 0:
        crowd = min(abs(math.log(pcr)), 1.0)     # |ln PCR| ≥1 ⇒ full tilt
        base *= (1.0 - 0.3 * crowd)
    return round(min(max(base, 0.0), 1.0), 4)


def futures_basis(spot: float, fut_bid: float, fut_ask: float,
                  dte_days: float) -> dict | None:
    """Annualized basis from a two-sided future book. None on a dead book."""
    if not (spot > 0 and fut_bid > 0 and fut_ask > 0 and dte_days > 0):
        return None
    mid = (fut_bid + fut_ask) / 2.0
    if (fut_ask - fut_bid) / mid > 0.005:        # >50 bps wide: don't trust
        return None
    b = (mid - spot) / spot
    return {"fut_mid": round(mid, 2), "basis_pct": round(100 * b, 4),
            "basis_ann_pct": round(100 * b * 365.0 / max(dte_days, 0.25), 2)}


def chain_flow(index: str, K: list[float], prem: list[float],
               oi: list[float], is_call: list[bool], lot: int,
               spot: float, strike_step: float,
               now: float | None = None) -> dict:
    """One call per macro sweep with the RAW aligned band arrays. Returns the
    full flow-intel dict for the snapshot payload; Δ-metrics are None until a
    fresh-enough previous sweep exists (honest cold start)."""
    now = time.time() if now is None else now
    Ka = np.asarray(K, float)
    OI = np.asarray(oi, float)
    PR = np.asarray(prem, float)
    C = np.asarray(is_call, bool)
    out: dict = {}
    # positioning level metrics (single-sweep)
    oi_c, oi_p = float(OI[C].sum()), float(OI[~C].sum())
    out["pcr_oi"] = round(oi_p / oi_c, 4) if oi_c > 0 else None
    mp = max_pain(K, oi, is_call, lot)
    out["max_pain"] = mp
    prev_mp = _MP_PREV.get(index)
    out["max_pain_drift"] = (round(mp - prev_mp, 2)
                             if (mp is not None and prev_mp is not None)
                             else None)
    if mp is not None:
        _MP_PREV[index] = mp
    out["pin_prior"] = pin_prior(spot, mp, strike_step, out["pcr_oi"])
    # Δ writer flow vs the previous sweep (₹ of NEW premium written)
    prev = _PREV.get(index)
    cur = {(float(k), bool(c)): float(o) for k, c, o in zip(Ka, C, OI)}
    flow = None
    if prev is not None and 0 < now - prev["ts"] <= FLOW_MAX_GAP_S:
        cw = pw = 0.0
        n_seen = 0
        for (k, c), o in cur.items():
            po = prev["oi"].get((k, c))
            if po is None:
                continue
            n_seen += 1
            d = o - po
            p_mid = float(PR[(Ka == k) & (C == c)][0])
            rs = d * lot * p_mid
            if c:
                cw += rs
            else:
                pw += rs
        if n_seen >= 6:
            flow = {"call_write_rs": round(cw, 0),
                    "put_write_rs": round(pw, 0),
                    "net_write_rs": round(cw + pw, 0),
                    "gap_s": round(now - prev["ts"], 1),
                    "strikes_matched": n_seen,
                    # positive: fresh put-writing BELOW minus call-writing
                    # ABOVE spot supports upward pin pressure; the classic
                    # writers-defend read.
                    "support_tilt_rs": round(
                        float(sum((o - prev["oi"].get((k, c), o)) * lot
                                  * PR[(Ka == k) & (C == c)][0]
                                  for (k, c), o in cur.items()
                                  if (k, c) in prev["oi"]
                                  and ((not c and k < spot)
                                       or (c and k > spot))
                                  ) -
                              sum((o - prev["oi"].get((k, c), o)) * lot
                                  * PR[(Ka == k) & (C == c)][0]
                                  for (k, c), o in cur.items()
                                  if (k, c) in prev["oi"]
                                  and ((c and k < spot)
                                       or (not c and k > spot)))), 0)}
    out["oi_flow"] = flow
    _PREV[index] = {"ts": now, "spot": spot, "oi": cur}
    return out