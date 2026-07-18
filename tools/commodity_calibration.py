"""
APEX OMNI v9.7.1 — COMMODITY INTRADAY (TRACK-B) CALIBRATION
===========================================================
The second calibration track for commodities. Where Track-A (commodity_backfill)
measures DAILY-scale facts from historical futures candles, Track-B measures the
INTRADAY microstructure that ONLY exists in the live harvested vault — because
Kite has no historical intraday options data or minute depth (expired option
tokens are flushed). These are the numbers that decide whether an intraday
options engine can trade a commodity without being blind to its microstructure.

This tool deliberately REUSES the equity calibrator's measurement function
(`_index_vol_and_tox` from tools.nightly_calibration) so the commodity intraday
stats are computed by the EXACT same code path as the equities — no divergence.
It simply runs it over the harvested commodity names and writes the results into
the same logs/calibration.json (under each commodity, alongside the indices), so
the brain's existing readers (core.calibration.index_calib / tox_thresholds /
bucket_volume) pick them up unchanged.

Track-B calibrated fields per commodity (identical semantics to equity)
-----------------------------------------------------------------------
  • move_median / move_p90 — realized 1-second move (intraday vol unit)
  • atr_proxy              — intraday ATR proxy
  • bucket_volume          — volume-bucket size (~1 bucket/min) for VPIN
  • tox_high / tox_block   — toxicity percentiles (p75/p90) of THIS commodity's
    own flow (Easley-LdP-O'Hara: toxicity is relative to its own history)
  • vol_delta_p95          — volume-spike baseline for absorption
Trusted only when n_ticks ≥ COMMODITY_CALIB_MIN_TICKS; else omitted (the engine
treats the commodity as NOT intraday-calibrated → not trade-eligible).

REPORT + ARTIFACT only. The NUMBERS require weeks of harvested commodity ticks
— this tool is complete, but it reports honestly that a commodity is
uncalibrated until its vault is deep enough. That gate is the point.

  python tools/commodity_calibration.py [--days N]
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
from core.diagnostics import _atomic_write_json            # noqa: E402
from nightly_forge_v9 import trading_days                  # noqa: E402
# reuse the EXACT equity measurement path — no divergence
from tools.nightly_calibration import _index_vol_and_tox, _pct  # noqa: E402

config.setup_logging("commodity_calibration")
import logging                                            # noqa: E402
log = logging.getLogger("commodity_calibration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    args = ap.parse_args()
    from simulation.scenario_engine import N

    commodities = list(getattr(config, "HARVEST_COMMODITIES", []))
    if not commodities:
        log.info("HARVEST_COMMODITIES empty — nothing to calibrate")
        return

    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)
    if args.days > 0:
        days = days[-args.days:]
    log.info("commodity intraday (Track-B) calibration | %d day(s) %s→%s | %s",
             len(days), days[0] if days else "-", days[-1] if days else "-",
             ",".join(commodities))

    per: dict = {c: {"moves": [], "vol_deltas": [], "tox": [], "atr": []}
                 for c in commodities}
    for day in days:
        for c in commodities:
            r = _index_vol_and_tox(con, day, c, N)   # same code as equity
            if r is None:
                continue
            per[c]["moves"].append(r["moves"])
            per[c]["vol_deltas"].append(r["vol_deltas"])
            per[c]["tox"].append(r["tox"])
            if r["atr"] is not None:
                per[c]["atr"].append(r["atr"])

    # merge into the shared artifact (don't clobber equity or Track-A fields)
    path = config.LOG_DIR / "calibration.json"
    try:
        existing = json.loads(path.read_text())
    except Exception:                                          # noqa: BLE001
        existing = {}
    existing.setdefault("indices", {})       # commodities live alongside indices

    MIN_TICKS = int(getattr(config, "COMMODITY_CALIB_MIN_TICKS", 30000))
    for c in commodities:
        d = per[c]
        moves = np.concatenate(d["moves"]) if d["moves"] else np.array([])
        vols = np.concatenate(d["vol_deltas"]) if d["vol_deltas"] else np.array([])
        tox = np.concatenate(d["tox"]) if d["tox"] else np.array([])
        entry: dict = {"n_ticks": int(len(moves)), "track": "B-intraday"}
        if len(moves) >= MIN_TICKS:
            entry["move_median"] = round(float(np.median(moves)), 3)
            entry["move_p90"] = round(_pct(moves, 90), 3)
            if d["atr"]:
                entry["atr_proxy"] = round(float(np.median(d["atr"])), 2)
            if len(vols) >= 60:
                per_min = np.add.reduceat(vols, np.arange(0, len(vols), 60))
                entry["bucket_volume"] = round(float(np.median(per_min)), 0)
                entry["vol_delta_p95"] = round(_pct(vols, 95), 1)
            if len(tox) >= 500:
                entry["tox_high"] = round(_pct(tox, 75), 3)
                entry["tox_block"] = round(_pct(tox, 90), 3)
                entry["tox_p50"] = round(_pct(tox, 50), 3)
            entry["intraday_calibrated"] = True
        else:
            entry["intraday_calibrated"] = False
            entry["note"] = (f"insufficient intraday sample ({len(moves)} < "
                             f"{MIN_TICKS}) — {c} NOT trade-eligible until its "
                             f"vault is deeper; harvest more sessions")
        existing["indices"][c] = entry
        log.info("  %s: %s", c, {k: entry[k] for k in entry
                                 if k not in ("n_ticks",)} or "defaults")

    existing["commodity_calib_written"] = dt.datetime.now().isoformat(
        timespec="seconds")
    _atomic_write_json(path, existing)
    log.info("Track-B commodity calibration → %s (brain hot-reloads)", path)


if __name__ == "__main__":
    main()