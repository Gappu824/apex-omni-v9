"""
APEX OMNI v9.5 — LIVE SPREAD ROUTER (Pillar 7 execution organ)
==============================================================
The code the shortvol certificate was always pointing at — complete,
defensive, and behind FIVE independent locks. It cannot fire unless ALL of:

  1. config.live_fire_armed()            (the four standing LIVE_FIRE locks)
  2. a valid shortvol certificate        (knob-hash-matched, living)
  3. graduation stage ∈ {micro_live, scaling}   (core/graduation.py)
  4. the operator token file exists      (state/ARM_LIVE_SPREADS)
  5. Kite's basket margin API confirms affordability at TRADING_CAPITAL

Routing discipline (defined-risk first, always):
  OPEN : margin check → SHORT leg (SELL) under an RCT arm; unfilled ⇒ cancel,
         abort clean. Filled ⇒ LONG leg (BUY); if the hedge cannot be
         completed, the short is IMMEDIATELY bought back at market
         (rollback) — a naked short never survives an error path.
  CLOSE: BUY the short back FIRST (risk off), then SELL the long.
Every leg runs through core/exec_rct.wrap_live_order — the execution RCT
begins measuring on the very first live lot.
"""
from __future__ import annotations

import json
import logging

import config
from core import exec_rct as RCT

log = logging.getLogger("spread_router")


def live_spread_allowed(cert: dict | None) -> tuple[bool, str]:
    if not config.live_fire_armed():
        return False, "LIVE_FIRE locks closed"
    if not (cert and cert.get("ok")):
        return False, "no valid shortvol certificate"
    try:
        grad = json.loads((config.STATE_DIR / "graduation.json").read_text())
        stage = grad["families"]["shortvol"]["stage"]
    except Exception:                                     # noqa: BLE001
        return False, "graduation state unavailable"
    if stage not in ("micro_live", "scaling"):
        return False, f"graduation stage '{stage}'"
    if not config.SPREAD_LIVE_TOKEN.exists():
        return False, "operator token absent (state/ARM_LIVE_SPREADS)"
    return True, "all five locks open"


def _margin_ok(kite, spec, lots: int) -> tuple[bool, str]:
    try:
        basket = [
            {"exchange": spec.exchange, "tradingsymbol": spec.short_symbol,
             "transaction_type": "SELL", "variety": "regular",
             "product": "NRML", "order_type": "MARKET",
             "quantity": spec.lot * lots},
            {"exchange": spec.exchange, "tradingsymbol": spec.long_symbol,
             "transaction_type": "BUY", "variety": "regular",
             "product": "NRML", "order_type": "MARKET",
             "quantity": spec.lot * lots}]
        m = kite.basket_order_margins(basket)
        need = float((m.get("final") or m.get("initial") or {})
                     .get("total") or 0)
        if need <= 0:
            return False, "margin api returned no figure"
        if need > config.TRADING_CAPITAL:
            return False, (f"margin ₹{need:.0f} > capital "
                           f"₹{config.TRADING_CAPITAL:.0f}")
        return True, f"margin ₹{need:.0f}"
    except Exception as e:                                # noqa: BLE001
        return False, f"margin api unavailable: {e}"      # fail-closed


def route_open(kite, spec, lots: int, quotes: dict) -> dict:
    """Returns {ok, short_fill, long_fill, arm_short, arm_long} or
    {ok:False, why} — with the rollback guarantee stated above."""
    ok, why = _margin_ok(kite, spec, lots)
    if not ok:
        return {"ok": False, "why": why}
    q = lots * spec.lot
    sq = quotes.get(spec.short_token) or {}
    lq = quotes.get(spec.long_token) or {}
    _sp_sq = None
    if sq.get("bid") and sq.get("ask"):
        _m_ = (float(sq["bid"]) + float(sq["ask"])) / 2
        _sp_sq = (float(sq["ask"]) - float(sq["bid"])) / max(_m_, 0.05)
    arm_s = RCT.assign(f"{spec.short_symbol}:{q}:open")
    r_s = RCT.wrap_live_order(kite, arm=arm_s, side="SELL",
                              exchange=spec.exchange,
                              symbol=spec.short_symbol, qty=q,
                              limit_px=float(sq.get("bid") or 0),
                              ref_px=float(sq.get("bid") or 0),
                              tag="spread_open_short",
                              spread_pct=_sp_sq)
    if not r_s.get("ok"):
        return {"ok": False, "why": f"short leg: {r_s.get('why')}"}
    _sp_lq = None
    if lq.get("bid") and lq.get("ask"):
        _m_ = (float(lq["bid"]) + float(lq["ask"])) / 2
        _sp_lq = (float(lq["ask"]) - float(lq["bid"])) / max(_m_, 0.05)
    arm_l = RCT.assign(f"{spec.long_symbol}:{q}:open")
    r_l = RCT.wrap_live_order(kite, arm=arm_l, side="BUY",
                              exchange=spec.exchange,
                              symbol=spec.long_symbol, qty=q,
                              limit_px=float(lq.get("ask") or 0),
                              ref_px=float(lq.get("ask") or 0),
                              tag="spread_open_long",
                              spread_pct=_sp_lq)
    if not r_l.get("ok"):
        # ROLLBACK — the short must not live unhedged for one more second
        log.critical("long leg failed (%s) — rolling back the short at "
                     "MARKET", r_l.get("why"))
        rb = RCT.wrap_live_order(kite, arm="CROSS", side="BUY",
                                 exchange=spec.exchange,
                                 symbol=spec.short_symbol, qty=q,
                                 limit_px=float(sq.get("ask") or 0),
                                 ref_px=float(sq.get("ask") or 0),
                                 tag="spread_rollback")
        return {"ok": False, "why": f"long leg failed; rollback "
                f"{'done' if rb.get('ok') else 'FAILED — FLATTEN MANUALLY'}"}
    return {"ok": True, "short_fill": r_s["fill_px"],
            "long_fill": r_l["fill_px"], "arm_short": r_s["arm"],
            "arm_long": r_l["arm"]}


def route_close(kite, sp, quotes: dict) -> dict:
    """Risk-first unwind: buy the SHORT back before selling the long."""
    q = sp.lots * sp.spec.lot
    sq = quotes.get(sp.spec.short_token) or {}
    lq = quotes.get(sp.spec.long_token) or {}
    r_s = RCT.wrap_live_order(kite, arm=RCT.assign(f"{sp.spread_id}:cs"),
                              side="BUY", exchange=sp.spec.exchange,
                              symbol=sp.spec.short_symbol, qty=q,
                              limit_px=float(sq.get("ask") or 0),
                              ref_px=float(sq.get("ask") or 0),
                              tag="spread_close_short")
    if not r_s.get("ok"):
        return {"ok": False, "why": f"close short: {r_s.get('why')}"}
    r_l = RCT.wrap_live_order(kite, arm=RCT.assign(f"{sp.spread_id}:cl"),
                              side="SELL", exchange=sp.spec.exchange,
                              symbol=sp.spec.long_symbol, qty=q,
                              limit_px=float(lq.get("bid") or 0),
                              ref_px=float(lq.get("bid") or 0),
                              tag="spread_close_long")
    if not r_l.get("ok"):
        return {"ok": False, "why": f"short bought back but long SELL "
                f"failed ({r_l.get('why')}) — long-only residue, benign "
                f"risk, retry next tick"}
    return {"ok": True, "short_fill": r_s["fill_px"],
            "long_fill": r_l["fill_px"]}