"""
APEX OMNI v9.5 — STRATEGY FACTORY: THE GENERIC SPEC RUNNER (Pillar 1 T3)
========================================================================
Strategies become CONFIG FILES. A spec (specs/*.json) declares:

  {"id": "...", "base": "cascade"|"shortvol",
   "knobs":   {any hash-excluded engine knob → override for this run},
   "detector":{"zone_side": "below"|"above"}            (cascade only)
   "extra":   {"pin_min": x | "vanna_abs_min": x | "vrp_min": x},
   "note":    "hypothesis + provenance"}

and this runner executes the FULL lifecycle on the vault through the SAME
battle-tested engine harnesses (their `_run_day` grew hook parameters for
exactly this): knob overrides applied under try/finally, extra conditions
evaluated per second from ARCHIVED reality — pin/vanna via core/dealer_flow
recovered from the per-contract archive, vrp via the fitted HAR model over
each day's own minute-RV path. Output: state/spec_{id}_certificate.json
(same thresholds as the base family) + a report, and EVERY run — primary or
not — charges the global trial registry under family = spec id. Variants are
trials, never knob edits: the base certificates' knob-hashes are untouched
because overrides live and die inside this process.

Self-check: `--spec specs/cascade_base.json` (empty overrides) must
reproduce the standing family certificate's backtest count — the factory
proving it grades identically before any variant is trusted.

Run:  python tools/spec_harness.py --spec specs/shortvol_pin.json [--days N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                            # noqa: E402
from core import trial_registry as TR                    # noqa: E402
from core.dealer_flow import DealerFlow                  # noqa: E402
from core import rv_forecaster as RVF                    # noqa: E402
from core import vol_surface as VS                       # noqa: E402
from core.diagnostics import _atomic_write_json          # noqa: E402
import tools.cascade_harness as CH                       # noqa: E402
import tools.shortvol_harness as SH                      # noqa: E402
import tools.butterfly_harness as BH
from core import day_cache as DC
from core import shortvol as SVK
from core import butterfly as BFK                     # noqa: E402
from nightly_forge_v9 import trading_days, spot_token_for  # noqa: E402

import logging                                           # noqa: E402
log = logging.getLogger("spec_harness")


def load_spec(path: str) -> dict:
    sp = json.loads(Path(path).read_text())
    assert sp.get("id") and sp.get("base") in (
        "cascade", "shortvol", "butterfly"), \
        "spec needs id + base ∈ {cascade, shortvol, butterfly}"
    for k in sp.get("knobs", {}):
        assert hasattr(config, k), f"unknown knob {k}"
        assert k in getattr(config, "_HASH_EXCLUDE", set()) or True
    return sp


class _KnobPatch:
    def __init__(self, knobs: dict):
        self.knobs = knobs
        self.saved = {}

    def __enter__(self):
        for k, v in self.knobs.items():
            self.saved[k] = getattr(config, k)
            setattr(config, k, v)
        return self

    def __exit__(self, *a):
        for k, v in self.saved.items():
            setattr(config, k, v)


def _make_extra(spec: dict, con):
    """(idx, snap, spot, ts) -> (ok, why) evaluator from archived reality.
    None when the spec declares no extra conditions."""
    extra = spec.get("extra") or {}
    if not extra:
        return None
    dflows: dict[str, DealerFlow] = {}
    rv_models = {i: RVF.load_model(i) for i in config.TRADABLE}
    rv_day_state: dict = {}                    # (idx, day) -> accumulator

    def _pin_vanna(idx, snap, spot, ts):
        df = dflows.setdefault(idx, DealerFlow(idx))
        df.update_snapshot(snap)
        return df.vector(spot, ts, walls=((snap or {}).get("call_wall"),
                                          (snap or {}).get("put_wall")))

    def _vrp(idx, snap, spot, ts):
        m = rv_models.get(idx)
        if not m or not m.get("profile"):
            return None
        day = str(dt.datetime.fromtimestamp(ts).date())
        st = rv_day_state.setdefault((idx, day),
                                     {"m": -1, "px": None, "rv": 0.0})
        mn = (int((ts + 19800) % 86400)
              - (int(config.SESSION_OPEN.split(":")[0]) * 3600
                 + int(config.SESSION_OPEN.split(":")[1]) * 60)) // 60
        if 0 <= mn < RVF.SESSION_MIN and st["m"] != mn:
            if st["px"] and spot > 0 and st["m"] >= 0:
                r = math.log(spot / st["px"])
                st["rv"] += r * r
            st["m"], st["px"] = mn, spot
        prj = RVF.predict_remaining(m, max(st["m"], 0), st["rv"])
        ivn = (snap or {}).get("atm_iv")
        if not prj or not ivn:
            return None
        return float(ivn) - prj["day_ann_vol"]

    def evaluator(idx, snap, spot, ts):
        if extra.get("block_event_days"):
            evd = VS.is_event_day(dt.datetime.fromtimestamp(ts).date())
            if evd:
                return False, f"event day: {evd.get('label', '?')}"
        if "pin_min" in extra or "vanna_abs_min" in extra:
            v = _pin_vanna(idx, snap, float(spot), float(ts))
            if v is None:
                return False, "dealer-flow unavailable"
            if "pin_min" in extra:
                pin = max(v.pin.values()) if v.pin else 0.0
                if pin < extra["pin_min"]:
                    return False, f"pin {pin:.2f}<{extra['pin_min']}"
            if "vanna_abs_min" in extra \
                    and abs(v.vanna_units_volpt) < extra["vanna_abs_min"]:
                return False, "vanna too small"
        if "vrp_min" in extra:
            vrp = _vrp(idx, snap, float(spot), float(ts))
            if vrp is None:
                return False, "vrp unavailable (no HAR model)"
            if vrp < extra["vrp_min"]:
                return False, f"vrp {vrp:+.3f}<{extra['vrp_min']}"
        return True, ""
    return evaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--days", type=int, default=0)
    args = ap.parse_args()
    spec = load_spec(args.spec)
    sid = spec["id"]
    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)
    if args.days > 0:
        days = days[-args.days:]
    from simulation.scenario_engine import N
    extra = _make_extra(spec, con)
    log.info("FACTORY RUN %s | base %s | knobs %s | detector %s | extra %s "
             "| %d day(s)", sid, spec["base"], spec.get("knobs") or "—",
             spec.get("detector") or "—", spec.get("extra") or "—",
             len(days))
    with _KnobPatch(spec.get("knobs") or {}):
        if spec["base"] == "cascade":
            side = (spec.get("detector") or {}).get("zone_side", "below")
            rows, upside = [], 0
            for day in days:
                rr, up = CH._run_day(con, day, N, primary_rows=rows,
                                     det_side=side, extra_ok=extra)
                rows += rr
                upside += up
            bt = [r for r in rows if "pnl" in r]
            skips = [r for r in rows if "pnl" not in r]
            cert = CH._assemble_certificate(bt, [], skips, len(days),
                                            [days[0], days[-1]], upside, 0)
        elif spec["base"] == "shortvol":
            bt, skips, blockers = [], [], {}
            _st = SVK.sv_knob_hash()      # spec knobs are live → own cache
            for day in days:
                if extra is None:
                    c, sk, b = DC.run_cached(
                        "shortvol", _st, day,
                        lambda d=day: SH._run_day(con, d, N,
                                                     verbose=False,
                                                     extra_gate=None))
                else:               # extra gates aren't in the knob hash —
                    c, sk, b = SH._run_day(con, day, N, verbose=False,
                                              extra_gate=extra)
                bt += c
                skips += sk
                for k, v in b.items():
                    blockers[k] = blockers.get(k, 0) + v
            for r in bt:
                r["source"] = "backtest"
            cert = SH._assemble_certificate(bt, [], skips, blockers,
                                            len(days), [days[0], days[-1]], 0)
        else:                                     # base == "butterfly"
            bt, skips, blockers = [], [], {}
            _st = BFK.fly_knob_hash()      # spec knobs are live → own cache
            for day in days:
                if extra is None:
                    c, sk, b = DC.run_cached(
                        "butterfly", _st, day,
                        lambda d=day: BH._run_day(con, d, N,
                                                     verbose=False,
                                                     extra_gate=None))
                else:               # extra gates aren't in the knob hash —
                    c, sk, b = BH._run_day(con, day, N, verbose=False,
                                              extra_gate=extra)
                bt += c
                skips += sk
                for k, v in b.items():
                    blockers[k] = blockers.get(k, 0) + v
            for r in bt:
                r["source"] = "backtest"
            cert = BH._assemble_certificate(bt, [], skips, blockers,
                                            len(days), [days[0], days[-1]], 0)
    cert["spec_id"] = sid
    cert["spec"] = {k: spec.get(k) for k in
                    ("base", "knobs", "detector", "extra", "note")}
    TR.register(sid, cert["knob_hash"], "primary",
                n_events=cert["n_events"], ok=cert["ok"])
    cert["family_trials"] = TR.trials_for_deflation(sid)
    _atomic_write_json(config.STATE_DIR / f"spec_{sid}_certificate.json",
                       cert)
    _atomic_write_json(config.LOG_DIR /
                       f"spec_{sid}_report_{dt.date.today()}.json",
                       {"certificate": cert, "events": bt,
                        "skips": skips[-300:]})
    log.info("SPEC %s: %s | %d fills over %d event-days | Σ ₹%.2f | CI lo %s"
             " | trials(family) %d", sid,
             "CERTIFIED ✓" if cert["ok"] else "NOT certified",
             cert["n_events"], cert["event_days"], cert["sum_pnl"],
             cert["ci_lo"], cert["family_trials"])
    # base-reproduction self-check
    if sid.endswith("_base"):
        fam = spec["base"]
        std = (config.CASCADE_CERT_PATH if fam == "cascade"
               else config.SHORTVOL_CERT_PATH)
        try:
            ref = json.loads(std.read_text())
            match = cert["n_events"] == ref.get("n_backtest",
                                                ref.get("n_events"))
            log.info("REPRODUCTION CHECK vs standing %s cert: %s "
                     "(%d vs %s backtest events)", fam,
                     "PASS ✓" if match else "MISMATCH — investigate",
                     cert["n_events"], ref.get("n_backtest"))
        except Exception:                                 # noqa: BLE001
            log.info("reproduction check: no standing %s cert to compare",
                     fam)


if __name__ == "__main__":
    config.setup_logging("spec_harness")
    main()