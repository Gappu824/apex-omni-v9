"""
APEX OMNI v9.5.1 — EXECUTION REPORT + LIVE-FILL CERTIFICATE (Pillar 6/7 glue)
=============================================================================
The consumer existed before the producer — core/graduation.py's scaling
stage reads state/livefill_{family}.json; THIS tool writes it. Two jobs:

1) RCT A/B ANALYSIS of state/exec_rct.jsonl: per-arm slippage (mean, sd, n),
   Welch t between CROSS and LIMIT_FIRST, LIMIT_FIRST's realized fill rate,
   latency — the execution policy certified by experiment, not folklore.
   With zero live rows it says so and refuses conclusions.

2) LIVE-FILL CERTIFICATE, prespecified: for a family (rows tagged spread_* →
   shortvol; single-leg tags → cascade at T4 arming), ok = n ≥ 20 completed
   live legs AND mean ADVERSE slip ≤ 5% of mean reference premium. That is
   the reconciliation gate between paper's fill assumptions and the measured
   market — nothing scales past micro-live without it.

Run weekly (meaningful only once live lots exist):
    python tools/execution_report.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                            # noqa: E402
from core import exec_rct as RCT                         # noqa: E402
from core.diagnostics import _atomic_write_json          # noqa: E402

config.setup_logging("execution_report")
import logging                                           # noqa: E402
log = logging.getLogger("execution")

_FAMILY_TAGS = {"shortvol": ("spread_open_short", "spread_open_long",
                             "spread_close_short", "spread_close_long"),
                "cascade": ("single_open", "single_close")}


def _read_rows() -> list[dict]:
    if not RCT.RCT_LOG.exists():
        return []
    out = []
    for line in RCT.RCT_LOG.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:                                 # noqa: BLE001
            continue
    return out


def _welch_t(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    m1, m2 = np.mean(a), np.mean(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    den = math.sqrt(v1 / len(a) + v2 / len(b))
    return float((m1 - m2) / den) if den > 0 else None


def _analyze(rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if r.get("ok") and r.get("slip") is not None]
    per_arm = {}
    for arm in RCT.ARMS:
        s = [float(r["slip"]) for r in ok_rows if r.get("arm") == arm]
        lat = [float(r.get("latency_s") or 0) for r in ok_rows
               if r.get("arm") == arm]
        per_arm[arm] = {"n": len(s),
                        "mean_slip": round(float(np.mean(s)), 4) if s else None,
                        "sd_slip": (round(float(np.std(s, ddof=1)), 4)
                                    if len(s) > 1 else None),
                        "mean_latency_s": (round(float(np.mean(lat)), 2)
                                           if lat else None)}
    lf = [r for r in rows if r.get("arm") == "LIMIT_FIRST" and r.get("ok")]
    fill_rate = (round(sum(1 for r in lf if r.get("limit_filled")) / len(lf),
                       3) if lf else None)
    t = _welch_t([float(r["slip"]) for r in ok_rows
                  if r.get("arm") == "CROSS"],
                 [float(r["slip"]) for r in ok_rows
                  if r.get("arm") == "LIMIT_FIRST"])
    return {"n_rows": len(rows), "n_completed": len(ok_rows),
            "per_arm": per_arm, "limit_fill_rate": fill_rate,
            "welch_t_cross_vs_limit": (round(t, 3) if t is not None
                                       else None),
            "fill_model": RCT.fit_fill_model(rows)}


def _livefill(rows: list[dict], family: str) -> dict | None:
    """The prespecified reconciliation certificate. None when the family has
    produced no live rows yet (absence, not failure)."""
    tags = _FAMILY_TAGS[family]
    fam = [r for r in rows if (r.get("tag") or "").startswith(tags)
           if True] if False else \
          [r for r in rows if any((r.get("tag") or "").startswith(t)
                                  for t in tags)]
    if not fam:
        return None
    done = [r for r in fam if r.get("ok") and r.get("slip") is not None
            and r.get("ref_px")]
    n = len(done)
    adverse = [max(float(r["slip"]), 0.0) for r in done]
    refs = [float(r["ref_px"]) for r in done]
    slip_pct = (float(np.mean(adverse)) / max(float(np.mean(refs)), 1e-9)
                if done else None)
    reasons = []
    if n < 20:
        reasons.append(f"live legs {n} < 20")
    if slip_pct is not None and slip_pct > 0.05:
        reasons.append(f"mean adverse slip {100 * slip_pct:.1f}% of premium"
                       f" > 5%")
    cert = {"ok": not reasons, "family": family, "n_legs": n,
            "mean_adverse_slip_pct": (round(100 * slip_pct, 2)
                                      if slip_pct is not None else None),
            "criterion": "n≥20 live legs AND mean adverse slip ≤5% of "
                         "mean reference premium",
            "blocked_by": reasons or None, "ts": time.time()}
    _atomic_write_json(config.STATE_DIR / f"livefill_{family}.json", cert)
    return cert


def main():
    rows = _read_rows()
    rep = _analyze(rows)
    rep["ts"] = time.time()
    if not rows:
        log.info("no live RCT rows yet — the experiment begins at the first "
                 "live lot (graduation stage micro_live); no conclusions "
                 "drawn, no certificates written")
    else:
        for arm, a in rep["per_arm"].items():
            log.info("%-11s n=%-4s slip %s ± %s | latency %ss", arm, a["n"],
                     a["mean_slip"], a["sd_slip"], a["mean_latency_s"])
        log.info("LIMIT_FIRST fill rate %s | Welch t(CROSS−LIMIT) %s | "
                 "fill model: %s", rep["limit_fill_rate"],
                 rep["welch_t_cross_vs_limit"],
                 rep["fill_model"].get("ok") or rep["fill_model"].get("why"))
    for family in _FAMILY_TAGS:
        lf = _livefill(rows, family)
        if lf:
            rep[f"livefill_{family}"] = lf
            log.info("LIVE-FILL CERT %-9s %s | legs %d | adverse slip %s%% %s",
                     family, "OK ✓" if lf["ok"] else "blocked",
                     lf["n_legs"], lf["mean_adverse_slip_pct"],
                     f"← {'; '.join(lf['blocked_by'])}"
                     if lf["blocked_by"] else "")
    _atomic_write_json(config.STATE_DIR / "execution_report.json", rep)


if __name__ == "__main__":
    main()