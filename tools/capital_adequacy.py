"""
APEX OMNI v9.9.18 — CAPITAL ADEQUACY (can this equity express this strategy?)
============================================================================
A lot is indivisible. Half-Kelly sizes a bet as a FRACTION of equity, but the
smallest position available is one lot of the cheapest quoted rung. When that
lot costs more than the Kelly fraction allows, the trade is refused — and the
refusal says nothing about the signal. It would refuse a perfect signal too.

That is a CAPITAL condition, and it needs its own number:

    outlay(1 lot) = premium x lot
    half-Kelly funds it when   equity x KELLY_FRACTION x kelly >= outlay
    so the strategy needs      kelly >= outlay / (equity x KELLY_FRACTION)
    and, with  kelly = p - (1-p)/b,
                               p >= (kelly_req + 1/b) / (1 + 1/b)

`p_required` is the honest headline: the calibrated win probability the book
must believe before it may take ONE lot. If that exceeds what the calibration
ever produces, the index is unreachable by arithmetic — no amount of signal
work opens it, only more equity or a smaller contract.

Measured on the 2026-08-10..08-19 ledger this was 102 of 206 blocks (49.5%),
median shortfall 102x, and it is the binding reason the label certificate sits
at 3 trades of the 60 it needs.

    python tools/capital_adequacy.py [--equity 60000] [--b 1.5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                             # noqa: E402

config.setup_logging("capital_adequacy")
import logging                                            # noqa: E402
log = logging.getLogger("capital_adequacy")


def p_required(outlay: float, equity: float, b: float,
               kelly_fraction: float) -> tuple[float, float]:
    """(kelly_required, win_prob_required) to fund ONE lot under half-Kelly."""
    if equity <= 0 or kelly_fraction <= 0:
        return float("inf"), float("inf")
    k_req = outlay / (equity * kelly_fraction)
    inv_b = 1.0 / max(b, 1e-9)
    return k_req, (k_req + inv_b) / (1.0 + inv_b)


def assess(equity: float, b: float, premiums: dict[str, float]) -> list[dict]:
    kf = float(getattr(config, "KELLY_FRACTION", 0.5))
    mk = float(getattr(config, "MAX_KELLY_BUDGET_PCT", 0.8))
    out = []
    for index in getattr(config, "TRADABLE", []):
        spec = None
        try:
            from core.instruments import _spec_for
            spec = _spec_for(index) or {}
        except Exception:                                  # noqa: BLE001
            spec = {}
        lot = int(spec.get("lot_fallback") or 0)
        prem = float(premiums.get(index) or 0.0)
        if not lot or not prem:
            out.append({"index": index, "lot": lot, "premium": prem,
                        "note": "no lot/premium available — pass --premium"})
            continue
        outlay = prem * lot
        k_req, p_req = p_required(outlay, equity, b, kf)
        out.append({
            "index": index, "lot": lot, "premium": prem, "outlay": outlay,
            "pct_equity": 100.0 * outlay / equity,
            "kelly_required": k_req, "p_required": p_req,
            # the hard ceiling: even at kelly=1 the cap still binds
            "hard_blocked": bool(outlay > equity * mk),
            "equity_for_p50": (outlay / (max(0.5 - 0.5 / max(b, 1e-9), 1e-9) * kf)
                               if 0.5 - 0.5 / max(b, 1e-9) > 0 else float("inf")),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--equity", type=float,
                    default=float(getattr(config, "TRADING_CAPITAL", 60000)))
    ap.add_argument("--b", type=float, default=1.5,
                    help="reward/risk ratio tp_pct/sl_pct used by the sizer")
    ap.add_argument("--premium", type=str, default="",
                    help="INDEX=PREMIUM,... ATM premium per index")
    a = ap.parse_args()

    prem = {}
    for part in filter(None, a.premium.split(",")):
        k, _, v = part.partition("=")
        prem[k.strip().upper()] = float(v)
    if not prem:
        prem = {"NIFTY": 176.0, "BANKNIFTY": 420.0, "SENSEX": 300.0}
        log.info("no --premium given; using representative ATM premiums %s",
                 prem)

    kf = float(getattr(config, "KELLY_FRACTION", 0.5))
    log.info("=" * 72)
    log.info("CAPITAL ADEQUACY | equity Rs%s | KELLY_FRACTION %.2f | b %.2f",
             f"{a.equity:,.0f}", kf, a.b)
    log.info("  a lot is indivisible: the question is not 'is the signal good'")
    log.info("  but 'can half-Kelly fund ONE lot of the cheapest rung'")
    log.info("-" * 72)
    rows = assess(a.equity, a.b, prem)
    worst = None
    for r in rows:
        if r.get("note"):
            log.info("  %-10s %s", r["index"], r["note"])
            continue
        log.info("  %-10s lot %3d x Rs%-6.0f = Rs%-9.0f (%.1f%% of equity)",
                 r["index"], r["lot"], r["premium"], r["outlay"],
                 r["pct_equity"])
        log.info("             needs kelly >= %.3f  =>  win_prob >= %.3f%s",
                 r["kelly_required"], r["p_required"],
                 "   [HARD-BLOCKED by MAX_KELLY_BUDGET_PCT]"
                 if r["hard_blocked"] else "")
        log.info("             Kelly-consistent equity at p=0.50: Rs%.0f",
                 r["equity_for_p50"])
        if worst is None or r["p_required"] < worst["p_required"]:
            worst = r
    log.info("-" * 72)
    if worst:
        log.info("  most reachable index: %s at win_prob >= %.3f",
                 worst["index"], worst["p_required"])
        if worst["p_required"] > 0.60:
            log.warning("  NO index is reachable below win_prob 0.60. A "
                        "calibrated book rarely asserts that, so most "
                        "decisions will be refused on CAPITAL, not on "
                        "signal — and the block histogram will read as "
                        "selectivity when it is not.")
    log.info("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())