"""
APEX OMNI v9.5 — EPISTEMIC HEALTH (Pillar 5: the exam examines itself)
======================================================================
Two organs, one nightly run:

1) CALIBRATION MONITOR — every probability the system emits is scored
   against what then happened. Ledger BUY→SELL pairs (win_prob at entry) and
   SPREAD_OPEN→CLOSE pairs (pop prior) become reliability bins, a Brier
   score, and its Murphy decomposition (uncertainty − resolution +
   reliability; Gneiting–Raftery 2007 on proper scoring). "0.85" is hereby
   forced to mean 85%. → state/calibration.json + console table.

2) LIVING CERTIFICATES — a certificate is a claim about the PRESENT, not a
   trophy. For every family with ok=true, the last LC_WINDOW fills (backtest
   + forward, from the newest harness report + forward logs) are re-scored
   with the STATIONARY bootstrap (Politis–Romano — day-PnLs are dependent);
   if the rolling CI lower bound breaks below zero on ≥ LC_MIN_EVENTS, the
   certificate is REWRITTEN ok=false with a de-arm reason (the prior state
   preserved under `pre_dearm`). The brain's heartbeat cert-refresh then
   disarms the engine within a minute — no restart, no human in the loop.
   Edges die; the machine now notices first.

Run nightly after the harnesses:  python tools/epistemic_health.py
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                            # noqa: E402
from core.diagnostics import _atomic_write_json          # noqa: E402
from core.robust_stats import stationary_ci_lo           # noqa: E402

config.setup_logging("epistemic_health")
import logging                                           # noqa: E402
log = logging.getLogger("epistemic")


# ------------------------------------------------------------ calibration
def _forecast_outcomes() -> list[dict]:
    """(p, y, source) triples from the execution ledger: long options via
    BUY→SELL symbol pairing; spreads via SPREAD_OPEN→CLOSE order_id."""
    p = Path(config.LEDGER_PATH)
    if not p.exists():
        return []
    out, open_long, open_spr = [], {}, {}
    for r in csv.DictReader(p.open(encoding="utf-8")):
        ev, sym = r.get("event"), r.get("symbol") or ""
        if ev == "BUY_FILL":
            open_long[sym] = r
        elif ev == "SELL_FILL" and sym in open_long:
            b = open_long.pop(sym)
            try:
                out.append({"p": float(b.get("win_prob") or 0),
                            "y": 1.0 if float(r.get("pnl") or 0) > 0 else 0.0,
                            "src": "long"})
            except ValueError:
                pass
        elif ev == "SPREAD_OPEN":
            open_spr[r.get("order_id") or ""] = r
        elif ev == "SPREAD_CLOSE":
            b = open_spr.pop(r.get("order_id") or "", None)
            if b:
                try:
                    out.append({"p": float(b.get("win_prob") or 0),
                                "y": 1.0 if float(r.get("pnl") or 0) > 0
                                else 0.0, "src": "spread"})
                except ValueError:
                    pass
    return [o for o in out if 0.0 < o["p"] < 1.0]


def _murphy(rows: list[dict]) -> dict:
    """Brier + Murphy decomposition over 0.05-wide reliability bins."""
    if not rows:
        return {"n": 0}
    P = np.array([r["p"] for r in rows])
    Y = np.array([r["y"] for r in rows])
    n = len(P)
    brier = float(np.mean((P - Y) ** 2))
    ybar = float(Y.mean())
    unc = ybar * (1 - ybar)
    edges = np.arange(0.0, 1.0001, 0.05)
    bins, rel, res = [], 0.0, 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (P >= lo) & (P < hi)
        nk = int(m.sum())
        if nk == 0:
            continue
        pk, yk = float(P[m].mean()), float(Y[m].mean())
        rel += nk * (pk - yk) ** 2
        res += nk * (yk - ybar) ** 2
        if nk >= 5:
            bins.append({"bin": f"{lo:.2f}-{hi:.2f}", "n": nk,
                         "p_mean": round(pk, 3), "hit_rate": round(yk, 3),
                         "gap": round(pk - yk, 3)})
    return {"n": n, "brier": round(brier, 4), "base_rate": round(ybar, 4),
            "uncertainty": round(unc, 4), "reliability": round(rel / n, 4),
            "resolution": round(res / n, 4),
            "identity_check": round(unc - res / n + rel / n - brier, 6),
            "bins": bins}


# ------------------------------------------------------- living certificates
_FAMILIES = {
    "cascade": {"cert": lambda: config.CASCADE_CERT_PATH,
                "report_glob": "cascade_harness_report_*.json",
                "events_keys": ("backtest_events", "forward_events")},
    "shortvol": {"cert": lambda: config.SHORTVOL_CERT_PATH,
                 "report_glob": "shortvol_harness_report_*.json",
                 "events_keys": ("backtest_events", "forward_events")},
}


def _recent_fills(fam_cfg) -> list[dict]:
    reps = sorted(config.LOG_DIR.glob(fam_cfg["report_glob"]))
    if not reps:
        return []
    try:
        rep = json.loads(reps[-1].read_text())
    except Exception:                                     # noqa: BLE001
        return []
    fills = []
    for k in fam_cfg["events_keys"]:
        fills += [r for r in rep.get(k, []) if "pnl" in r]
    fills.sort(key=lambda r: (r.get("day", ""), r.get("hm", "")))
    return fills


def _living_check(family: str, fam_cfg) -> dict:
    cpath = fam_cfg["cert"]()
    try:
        cert = json.loads(cpath.read_text())
    except Exception:                                     # noqa: BLE001
        return {"family": family, "state": "no certificate"}
    fills = _recent_fills(fam_cfg)
    window = fills[-config.LC_WINDOW:]
    n = len(window)
    day_pnl: dict[str, float] = {}
    for r in window:
        day_pnl[r["day"]] = day_pnl.get(r["day"], 0.0) + float(r["pnl"])
    lo = stationary_ci_lo(list(day_pnl.values())) if n else None
    out = {"family": family, "cert_ok": bool(cert.get("ok")),
           "window_n": n, "window_days": len(day_pnl),
           "rolling_stat_ci_lo": (round(lo, 2) if lo is not None else None)}
    if cert.get("ok") and n >= config.LC_MIN_EVENTS and lo is not None \
            and lo < 0:
        dearmed = dict(cert)
        dearmed["ok"] = False
        dearmed["blocked_by"] = [
            f"LIVING-CERT DE-ARM {dt.date.today()}: rolling {n} fills over "
            f"{len(day_pnl)} days, stationary CI90 lower ₹{lo:.2f} < 0"]
        dearmed["pre_dearm"] = {k: cert.get(k) for k in
                                ("ok", "n_events", "mean_pnl", "ci_lo",
                                 "win_rate", "ts")}
        dearmed["ts"] = time.time()
        _atomic_write_json(cpath, dearmed)
        out["state"] = "DE-ARMED (edge decay)"
        log.warning("%s LIVING CERT DE-ARMED — rolling CI90 lo ₹%.2f over "
                    "%d fills; the brain disarms on its next heartbeat",
                    family, lo, n)
    else:
        out["state"] = ("healthy" if cert.get("ok") else
                        "not certified (nothing to de-arm)")
    return out


def main():
    cal_all = _forecast_outcomes()
    cal = {"all": _murphy(cal_all),
           "long": _murphy([r for r in cal_all if r["src"] == "long"]),
           "spread": _murphy([r for r in cal_all if r["src"] == "spread"]),
           "ts": time.time()}
    _atomic_write_json(config.STATE_DIR / "calibration.json", cal)
    a = cal["all"]
    if a.get("n"):
        log.info("CALIBRATION | n=%d | Brier %.4f = unc %.4f − res %.4f + "
                 "rel %.4f | base %.2f", a["n"], a["brier"],
                 a["uncertainty"], a["resolution"], a["reliability"],
                 a["base_rate"])
        for b in a.get("bins", []):
            log.info("  p∈%s  n=%-4d  said %.2f  did %.2f  gap %+.2f",
                     b["bin"], b["n"], b["p_mean"], b["hit_rate"], b["gap"])
    else:
        log.info("CALIBRATION | no scored forecasts in the ledger yet")
    health = [_living_check(f, c) for f, c in _FAMILIES.items()]
    _atomic_write_json(config.STATE_DIR / "cert_health.json",
                       {"families": health, "ts": time.time()})
    for h in health:
        log.info("LIVING CERT %-9s %s | window %s fills / %s days | "
                 "rolling CI lo %s", h["family"], h["state"],
                 h.get("window_n"), h.get("window_days"),
                 h.get("rolling_stat_ci_lo"))


if __name__ == "__main__":
    main()