"""
APEX OMNI v9.5 — EXECUTION SCIENCE CORE (Pillar 6)
==================================================
The randomized controlled trial lives HERE, pre-installed for the first live
lot. Honesty first: paper fills are model output — randomizing them measures
the model, not the market — so the RCT arms bind to the LIVE order path only
(the spread router consumes them now; the single-leg engine at T4 arming).
Every live order is randomly assigned an arm:

    CROSS       — MARKET (today's assumption: pay the spread, fill now)
    LIMIT_FIRST — LIMIT at the touch, RCT_LIMIT_TIMEOUT_S to fill, then
                  cancel → MARKET (the classic maker-first policy)

Assignment is deterministic per order key (hash-seeded 50/50) so replays and
audits reproduce it. Every arm outcome — fill price vs decision price,
latency, whether the limit filled — appends to state/exec_rct.jsonl; the
analysis (tools/execution_report.py) is a real A/B on YOUR slippage
(Cont–Kukanov–Stoikov lineage for the eventual conditioned fill model).

FILL MODEL: logistic P(limit fills | spread%, |z|, side) via numpy IRLS —
REFUSES to fit below RCT_MIN_FIT labeled orders. A model on twelve points is
theater, and this program does not do theater.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np

import config

RCT_LOG = config.STATE_DIR / "exec_rct.jsonl"
ARMS = ("CROSS", "LIMIT_FIRST")


def assign(order_key: str) -> str:
    """Deterministic 50/50 arm per order key."""
    h = int(hashlib.sha1(order_key.encode()).hexdigest()[:8], 16)
    return ARMS[h & 1]


def log_row(row: dict) -> None:
    try:
        RCT_LOG.parent.mkdir(exist_ok=True)
        row = {"ts": time.time(), **row}
        with RCT_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:                                     # noqa: BLE001
        pass


def wrap_live_order(kite, *, arm: str, side: str, exchange: str,
                    symbol: str, qty: int, limit_px: float,
                    ref_px: float, tag: str = "",
                    spread_pct: float | None = None,
                    z: float | None = None) -> dict:
    """Place ONE live leg under an RCT arm. Returns
    {ok, fill_px, arm, latency_s, limit_filled, order_id} or {ok:False,...}.
    CROSS ⇒ MARKET. LIMIT_FIRST ⇒ LIMIT@limit_px for RCT_LIMIT_TIMEOUT_S,
    then cancel → MARKET. Slippage is logged vs ref_px (the decision quote).
    Fail-closed on any API doubt."""
    t0 = time.time()
    txn = kite.TRANSACTION_TYPE_BUY if side == "BUY" \
        else kite.TRANSACTION_TYPE_SELL

    def _fill_of(oid) -> float | None:
        try:
            for o in kite.order_history(oid):
                if o.get("status") == "COMPLETE":
                    return float(o.get("average_price") or 0) or None
        except Exception:                                 # noqa: BLE001
            return None
        return None

    def _market() -> dict:
        oid = kite.place_order(
            variety=kite.VARIETY_REGULAR, exchange=exchange,
            tradingsymbol=symbol, transaction_type=txn, quantity=qty,
            product=kite.PRODUCT_NRML, order_type=kite.ORDER_TYPE_MARKET)
        for _ in range(20):
            px = _fill_of(oid)
            if px:
                return {"ok": True, "fill_px": px, "order_id": oid}
            time.sleep(0.25)
        return {"ok": False, "why": "market order unconfirmed",
                "order_id": oid}

    try:
        limit_filled = False
        if arm == "LIMIT_FIRST":
            oid = kite.place_order(
                variety=kite.VARIETY_REGULAR, exchange=exchange,
                tradingsymbol=symbol, transaction_type=txn, quantity=qty,
                product=kite.PRODUCT_NRML, order_type=kite.ORDER_TYPE_LIMIT,
                price=round(limit_px, 2))
            deadline = time.time() + config.RCT_LIMIT_TIMEOUT_S
            px = None
            while time.time() < deadline:
                px = _fill_of(oid)
                if px:
                    limit_filled = True
                    break
                time.sleep(0.25)
            if not limit_filled:
                try:
                    kite.cancel_order(kite.VARIETY_REGULAR, oid)
                except Exception:                         # noqa: BLE001
                    pass
                res = _market()
            else:
                res = {"ok": True, "fill_px": px, "order_id": oid}
        else:
            res = _market()
    except Exception as e:                                # noqa: BLE001
        res = {"ok": False, "why": f"order api: {e}"}
    lat = time.time() - t0
    slip = ((res.get("fill_px", ref_px) - ref_px)
            * (1 if side == "BUY" else -1)) if res.get("ok") else None
    log_row({"arm": arm, "side": side, "symbol": symbol, "qty": qty,
             "spread_pct": spread_pct, "z": z,
             "ref_px": ref_px, "limit_px": limit_px,
             "fill_px": res.get("fill_px"), "slip": slip,
             "limit_filled": limit_filled, "latency_s": round(lat, 2),
             "ok": res.get("ok"), "why": res.get("why"), "tag": tag})
    return {**res, "arm": arm, "latency_s": lat, "limit_filled": limit_filled}


# ------------------------------------------------------------ fill model
def fit_fill_model(rows: list[dict] | None = None) -> dict:
    """Logistic P(limit fills) ~ [1, spread_pct, |z|] via IRLS. Refuses under
    RCT_MIN_FIT labeled LIMIT_FIRST orders — insufficiency is a verdict."""
    if rows is None:
        rows = []
        if RCT_LOG.exists():
            for line in RCT_LOG.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except Exception:                         # noqa: BLE001
                    pass
    lab = [r for r in rows if r.get("arm") == "LIMIT_FIRST"
           and r.get("ok") and r.get("limit_filled") is not None
           and r.get("spread_pct") is not None]
    if len(lab) < config.RCT_MIN_FIT:
        return {"ok": False,
                "why": f"{len(lab)} labeled orders < {config.RCT_MIN_FIT} — "
                       f"model refuses (no theater)"}
    X = np.array([[1.0, float(r["spread_pct"]), abs(float(r.get("z") or 0))]
                  for r in lab])
    y = np.array([1.0 if r["limit_filled"] else 0.0 for r in lab])
    # Ridge-regularized IRLS (MAP with a tiny Gaussian prior): keeps the
    # Hessian positive-definite under perfect separation — which a small
    # early live sample can genuinely produce — and bounds the predictor.
    beta = np.zeros(X.shape[1])
    for _ in range(30):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p) + 1e-6
        H = X.T @ (X * W[:, None]) + 1e-4 * np.eye(X.shape[1])
        beta = beta + np.linalg.solve(H, X.T @ (y - p))
    return {"ok": True, "beta": [float(b) for b in beta], "n": len(lab)}