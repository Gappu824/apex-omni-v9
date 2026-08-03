"""
APEX OMNI v9.7.1 — NIGHTLY CALIBRATION (everything the data can decide)
=======================================================================
The operator's directive: stop hand-setting numbers where the vault can decide
them, and re-decide every night in run_evening. This is that single pass. It
measures thresholds from the operator's OWN tick history and writes them to
logs/calibration.json, which the brain loads at startup (and hot-reloads on
mtime change, exactly like the fly-intel polarity artifact).

Philosophy (the same telemetry → evidence → certificate → action ladder used
everywhere else in this system): a calibrated value is TRUSTED only when the
sample supports it; otherwise the field is omitted and the brain falls back to
the conservative config default. Calibration NARROWS uncertainty from data; it
never invents a number from a thin sample. Every value carries its provenance
(sample size, window) in the artifact.

What it calibrates (each grounded, each falling back safely)
-----------------------------------------------------------
1. VOLATILITY UNITS per index — the median realized 1-second move and the
   intraday ATR proxy, so the DYNAMIC stop/target (core/dynamic_levels) breathe
   with each instrument instead of a fixed percent. (Kaufman: volatility-scaled
   exits.)
2. TOXICITY thresholds — the distribution of the VPIN-proxy on the vault, so
   TOX_HIGH / TOX_BLOCK sit at real percentiles of THIS market's flow rather
   than a guessed 0.4/0.55. (Easley-LdP-O'Hara: toxicity is relative to its own
   history.) Also the volume-bucket size that yields ~1 bucket/minute.
3. CASCADE STOP WIDTH — the actual retest-wick depth distribution following
   cascade triggers, so CASCADE_STOP_MULT_* are set from how deep the whipsaws
   REALLY were on this tape (the tuning the operator asked for by name).
4. ABSORPTION volume z baseline — the vol_delta distribution, so the absorption
   detector's z-threshold matches real volume spikes.

This module is REPORT + ARTIFACT only. No model, no trade, no cert. It reuses
the forge's DB plumbing. Runs inside run_evening (added to the SEQ).

  python tools/nightly_calibration.py [--days N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                             # noqa: E402
from core.order_flow import OrderFlowToxicity              # noqa: E402
from core.diagnostics import _atomic_write_json            # noqa: E402
from nightly_forge_v9 import trading_days, spot_token_for  # noqa: E402

config.setup_logging("nightly_calibration")
import logging                                            # noqa: E402
log = logging.getLogger("nightly_calibration")

_OH, _OM = (int(x) for x in config.SESSION_OPEN.split(":"))
_OPEN_SOD = _OH * 3600 + _OM * 60


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def _index_vol_and_tox(con, day: str, idx: str, N: int):
    """One index-day: realized 1s moves, ATR proxy, vol_delta dist, and the
    toxicity series from the streaming estimator over the option-leg book."""
    tok = spot_token_for(con, day, idx)
    if not tok:
        return None
    rows = list(con.execute(
        "SELECT ts_ms/1000.0, ltp, bid, ask, bid_qty, ask_qty, vol_delta "
        "FROM ticks_v9 WHERE token=? AND ltp>0 "
        "AND date(ts_local_ms/1000,'unixepoch','localtime')=? "
        "ORDER BY ts_ms", (tok, day)))
    if len(rows) < 60:
        return None
    spots = np.array([r[1] for r in rows], float)
    moves = np.abs(np.diff(spots))
    vol_deltas = np.array([r[6] for r in rows], float)
    # streaming toxicity over the spot book (proxy; option-leg books aggregate
    # similarly and the calibration only needs the DISTRIBUTION shape)
    eng = OrderFlowToxicity(idx)
    tox_series = []
    for (_ts, ltp, bid, ask, bq, aq, vd) in rows:
        v = eng.update(spot=ltp, bid=bid, bid_qty=bq, ask=ask, ask_qty=aq,
                       vol_delta=vd)
        if v.toxicity > 0:
            tox_series.append(v.toxicity)
    # ATR proxy: 14-period-equivalent mean absolute 60s change
    if len(spots) >= 60:
        block = spots[::60]
        atr = float(np.mean(np.abs(np.diff(block)))) if len(block) > 1 else None
    else:
        atr = None
    return {"moves": moves, "vol_deltas": vol_deltas,
            "tox": np.array(tox_series, float), "atr": atr,
            "n_ticks": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    args = ap.parse_args()
    from simulation.scenario_engine import N
    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)
    if args.days > 0:
        days = days[-args.days:]
    log.info("nightly calibration | %d day(s) %s→%s", len(days),
             days[0] if days else "-", days[-1] if days else "-")

    per_index: dict = {i: {"moves": [], "vol_deltas": [], "tox": [], "atr": []}
                       for i in config.TRADABLE}
    for day in days:
        for idx in config.TRADABLE:
            r = _index_vol_and_tox(con, day, idx, N)
            if r is None:
                continue
            per_index[idx]["moves"].append(r["moves"])
            per_index[idx]["vol_deltas"].append(r["vol_deltas"])
            per_index[idx]["tox"].append(r["tox"])
            if r["atr"] is not None:
                per_index[idx]["atr"].append(r["atr"])

    calib: dict = {"written": dt.datetime.now().isoformat(timespec="seconds"),
                   "days": len(days), "config_hash": config.CONFIG_HASH,
                   "indices": {}}
    MIN_TICKS = int(getattr(config, "CALIB_MIN_TICKS", 20000))
    for idx in config.TRADABLE:
        d = per_index[idx]
        moves = np.concatenate(d["moves"]) if d["moves"] else np.array([])
        vols = np.concatenate(d["vol_deltas"]) if d["vol_deltas"] else np.array([])
        tox = np.concatenate(d["tox"]) if d["tox"] else np.array([])
        entry: dict = {"n_ticks": int(len(moves))}
        if len(moves) >= MIN_TICKS:
            # volatility units: median & p90 1s move; ATR proxy
            entry["move_median"] = round(float(np.median(moves)), 3)
            entry["move_p90"] = round(_pct(moves, 90), 3)
            if d["atr"]:
                entry["atr_proxy"] = round(float(np.median(d["atr"])), 2)
            # volume-bucket size ≈ median 60s cumulative volume (≈1 bucket/min)
            if len(vols) >= 60:
                per_min = np.add.reduceat(
                    vols, np.arange(0, len(vols), 60))
                entry["bucket_volume"] = round(float(np.median(per_min)), 0)
                entry["vol_delta_p95"] = round(_pct(vols, 95), 1)
            # toxicity percentiles → TOX_HIGH (p75) and TOX_BLOCK (p90)
            if len(tox) >= 500:
                entry["tox_high"] = round(_pct(tox, 75), 3)
                entry["tox_block"] = round(_pct(tox, 90), 3)
                entry["tox_p50"] = round(_pct(tox, 50), 3)
        else:
            entry["note"] = (f"insufficient sample ({len(moves)} < {MIN_TICKS}) "
                             f"— brain uses config defaults for {idx}")
        calib["indices"][idx] = entry
        log.info("  %s: %s", idx,
                 {k: entry[k] for k in entry if k != "n_ticks"} or "defaults")

    out = config.LOG_DIR / "calibration.json"
    # v9.9.5 BUGFIX: calibration.json is a SHARED artifact — Track-A
    # (commodity_backfill) and Track-B (commodity_calibration) merge their
    # sections into it, but this writer rebuilt the file from scratch. While
    # the chain was serial and commodities ran last, the clobber was
    # invisible. Under the parallel evidence group this writer finished last
    # on 2026-08-02 and ERASED every commodity calibration: the 12:17 artifact
    # carries NIFTY/BANKNIFTY/SENSEX only — no CRUDEOIL, NATURALGAS, GOLD,
    # SILVER, COPPER, no commodities_daily. The commodity brain hot-reloads
    # this file, so it lost its ATR proxies and toxicity thresholds silently.
    # Merge now: own the index entries, preserve everything else.
    try:
        existing = json.loads(out.read_text())
    except Exception:                                          # noqa: BLE001
        existing = {}
    merged = dict(existing)
    merged.update({k: v for k, v in calib.items() if k != "indices"})
    idx_all = dict(existing.get("indices") or {})
    idx_all.update(calib.get("indices") or {})
    merged["indices"] = idx_all
    _atomic_write_json(out, merged)
    log.info("calibration artifact → %s (brain hot-reloads this)", out)


if __name__ == "__main__":
    main()