"""
APEX OMNI v9.5 — CAPITAL GRADUATION PROTOCOL (Pillar 7)
=======================================================
Real money enters the same way every trade did: by passing an exam. Stages
per engine family, ALL transitions evidence-triggered, demotion automatic:

  research        — no valid certificate.
  paper_certified — certificate ok (blended, knob-hash-matched, living).
  micro_live      — paper_certified AND the OPERATOR arm file exists
                    (state/ARM_MICRO_{FAMILY} — a human touches a file; the
                    machine never promotes itself past paper alone) AND the
                    account is live-armed. Capital: GRAD_MICRO_CAPITAL, one
                    lot, purpose = manufacture the live-fill evidence.
  scaling         — micro_live AND a LIVE-FILL CERTIFICATE
                    (state/livefill_{family}.json ok=true: measured live
                    slippage reconciles with paper assumptions — written by
                    the execution analysis once ≥20 live RCT rows exist).
                    Capital ladder: base × GRAD_KELLY_FRAC × kelly(win-rate
                    lower bound, b=1 conservative) × Grossman–Zhou throttle
                    (1 − dd/GRAD_DD_MAX_PCT), floored at micro.

DEMOTION: a de-armed/expired certificate drops the family straight back to
research; a failed live-fill cert drops scaling → micro_live. LIVE_FIRE
remains a separate, prior lock — this module never touches it.
(MacLean–Thorp–Ziemba on fractional Kelly; Grossman–Zhou 1993 on drawdown.)

Report:  python -m core.graduation
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import config
from core.diagnostics import _atomic_write_json

STAGES = ("research", "paper_certified", "micro_live", "scaling")
_FAMILIES = {"cascade": lambda: config.CASCADE_CERT_PATH,
             "shortvol": lambda: config.SHORTVOL_CERT_PATH}


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text())
    except Exception:                                     # noqa: BLE001
        return None


def _kelly_capital(base: float, wr_lo: float, dd_pct: float) -> float:
    """Fractional Kelly at conservative b=1 (payoff symmetry assumed until
    the live book measures better) with a Grossman–Zhou drawdown throttle."""
    k = max(2.0 * wr_lo - 1.0, 0.0)
    gz = max(1.0 - dd_pct / config.GRAD_DD_MAX_PCT, 0.0)
    return max(base * config.GRAD_KELLY_FRAC * k * gz,
               config.GRAD_MICRO_CAPITAL if k > 0 else 0.0)


def evaluate(family: str) -> dict:
    cert = _load(_FAMILIES[family]())
    livefill = _load(config.STATE_DIR / f"livefill_{family}.json")
    arm = (config.STATE_DIR / f"ARM_MICRO_{family.upper()}").exists()
    cert_ok = bool(cert and cert.get("ok"))
    stage, why = "research", []
    if not cert_ok:
        why.append("no valid certificate"
                   if not cert else f"cert blocked: {cert.get('blocked_by')}")
    else:
        stage = "paper_certified"
        if not arm:
            why.append(f"operator arm file absent "
                       f"(state/ARM_MICRO_{family.upper()})")
        elif not config.live_fire_armed():
            why.append("account not live-armed (LIVE_FIRE locks)")
        else:
            stage = "micro_live"
            if not (livefill and livefill.get("ok")):
                why.append("live-fill certificate absent/failed "
                           "(≥20 live RCT rows must reconcile slippage)")
            else:
                stage = "scaling"
    wr_lo = float((cert or {}).get("win_rate_lo") or 0.0)
    dd = float((cert or {}).get("max_dd_pct") or 0.0)
    capital = {"research": 0.0, "paper_certified": 0.0,
               "micro_live": config.GRAD_MICRO_CAPITAL,
               "scaling": round(_kelly_capital(config.TRADING_CAPITAL,
                                               wr_lo, dd), 0)}[stage]
    return {"family": family, "stage": stage, "capital_rs": capital,
            "cert_ok": cert_ok, "win_rate_lo": wr_lo,
            "operator_armed": arm, "livefill_ok": bool(livefill
                                                       and livefill.get("ok")),
            "blocked_by": why or None, "ts": time.time()}


def write_state() -> dict:
    out = {"families": {f: evaluate(f) for f in _FAMILIES},
           "ts": time.time()}
    _atomic_write_json(config.STATE_DIR / "graduation.json", out)
    return out


if __name__ == "__main__":
    config.setup_logging("graduation")
    import logging
    log = logging.getLogger("graduation")
    st = write_state()
    for f, e in st["families"].items():
        log.info("%-9s stage=%-15s capital ₹%-9.0f cert=%s armed=%s "
                 "livefill=%s%s", f, e["stage"], e["capital_rs"],
                 e["cert_ok"], e["operator_armed"], e["livefill_ok"],
                 f"  ← {'; '.join(e['blocked_by'])}" if e["blocked_by"]
                 else "")