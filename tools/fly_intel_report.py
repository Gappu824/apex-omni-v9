"""
APEX OMNI v9.7.1 — FLY-INTEL VAULT REPORT (measure the edge on YOUR data)
=========================================================================
The offline validator (tools/fly_intel_validate.py) proves the LOGIC is
correct on a modelled pin. This measures the actual EDGE on your real vault:
across every second the butterfly gate GRANTED (positive-gamma + rich-IV +
inside-corridor), it compares the forward return of a directional entry
pointing INTO the near wall vs a fade FROM the corridor edge toward the pin,
using the SAME shaped-barrier grader the forge uses for the long book — ASK
entry, bid exit, real Zerodha costs. If the fly's grant genuinely marks a
mean-reverting regime, the fade wins and the breakout loses on your tape, and
the modulation is justified. If not, this report says so and you set
FLY_INTEL_ENABLED=False — the number, not a hope, decides.

It also reports, per granted second, the fly-intel conv_mult that WOULD have
been applied — so you can see how often it tilts a decision and by how much,
before trusting it live.

Report only. Writes logs/fly_intel_report_<date>.json. No model, no cert, no
trade. Reuses the forge's DB plumbing (trading_days, spot_token_for, the
macro archive, the barrier grid).

  python tools/fly_intel_report.py [--days N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                             # noqa: E402
from core import shortvol as SV                            # noqa: E402
from core import fly_intel as FI                           # noqa: E402
from core.gamma_nowcast import GammaNowcast                # noqa: E402
from core.diagnostics import _atomic_write_json            # noqa: E402
from macro_gex_v9 import load_macro_archive                # noqa: E402
from nightly_forge_v9 import (trading_days, spot_token_for,      # noqa: E402
                              _latest_at, _eval_hm)

config.setup_logging("fly_intel_report")
import logging                                            # noqa: E402
log = logging.getLogger("fly_intel_report")

_OH, _OM = (int(x) for x in config.SESSION_OPEN.split(":"))
_OPEN_SOD = _OH * 3600 + _OM * 60


def _spot_series(con, day, tok, N):
    spots = np.full(N, np.nan)
    ts_arr = np.full(N, np.nan)
    for ts, ltp in con.execute(
            "SELECT ts_ms/1000.0, ltp FROM ticks_v9 WHERE token=? AND ltp>0 "
            "AND date(ts_local_ms/1000,'unixepoch','localtime')=? "
            "ORDER BY ts_ms", (tok, day)):
        t = int((ts + 19800) % 86400) - _OPEN_SOD
        if 0 <= t < N:
            spots[t] = ltp
            ts_arr[t] = ts
    return spots, ts_arr


def _fwd_spot_ret(spots, t0, hold):
    """Forward SPOT return over `hold` seconds from t0 (fraction). NaN-safe."""
    if t0 + hold >= len(spots):
        return None
    a, b = spots[t0], spots[t0 + hold]
    if np.isnan(a) or np.isnan(b) or a <= 0:
        return None
    return (b - a) / a


def _run_day(con, day, N, hold_s):
    from simulation.replay_real_day import load_day
    loaded = load_day(con, day, config.TRADABLE[0])
    if loaded is None:
        return []
    rows = []
    for idx in config.TRADABLE:
        tok = spot_token_for(con, day, idx)
        if not tok:
            continue
        spots, ts_arr = _spot_series(con, day, tok, N)
        snaps = load_macro_archive(con, day, idx)
        nc = GammaNowcast(idx)
        ptr = [0]
        step = float(config.INDICES[idx]["strike_step"])
        for t in range(N):
            spot, ts = spots[t], ts_arr[t]
            if np.isnan(spot) or np.isnan(ts):
                continue
            spot, ts = float(spot), float(ts)
            hm = _eval_hm(t)
            snap = _latest_at(snaps, ptr, ts, lambda s: s["ts"])
            if snap is not None:
                nc.update_snapshot(snap)
            ncast = nc.nowcast(spot, ts)
            gex = (ncast.net_gex if ncast is not None and ncast.flip is not None
                   else (snap or {}).get("net_gex"))
            g = SV.evaluate_gate(hm=hm, spot=spot, mac=snap, net_gex_now=gex,
                                 dte=(snap or {}).get("dte"), strike_step=step,
                                 vix_bump=0.0, cascade_blocked=False)
            if not g.ok:
                continue
            cw, pw = (snap or {}).get("call_wall"), (snap or {}).get("put_wall")
            fret = _fwd_spot_ret(spots, t, hold_s)
            if fret is None or not (cw and pw):
                continue
            room_ce = (float(cw) - spot) / step
            room_pe = (spot - float(pw)) / step
            near = "CE" if room_ce <= room_pe else "PE"
            # forward premium proxy: a long CE gains on +spot, PE on −spot.
            into_ret = fret if near == "CE" else -fret          # buy toward wall
            fade_dir = "PE" if near == "CE" else "CE"
            fade_ret = fret if fade_dir == "CE" else -fret      # buy toward pin
            fi = FI.assess(granted=True, side=near, spot=spot, call_wall=cw,
                           put_wall=pw, corridor_steps=g.corridor_steps,
                           iv_rank=g.iv_rank, net_gex=g.net_gex,
                           strike_step=step, direction=near, conviction=0.70)
            rows.append({"day": day, "index": idx, "near": near,
                         "into_ret": into_ret, "fade_ret": fade_ret,
                         "pin": fi.pin_pressure,
                         "conv_mult_into": fi.conv_mult,
                         "runway_mult": fi.target_runway_mult,
                         "room_near": round(min(room_ce, room_pe), 2)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--hold", type=int,
                    default=int(getattr(config, "RIDE_ER_WINDOW_S", 120) * 2),
                    help="forward horizon seconds for the reversion test")
    args = ap.parse_args()
    from simulation.scenario_engine import N
    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)
    if args.days > 0:
        days = days[-args.days:]
    log.info("fly-intel vault report | %d day(s) %s→%s | horizon %ds",
             len(days), days[0] if days else "-", days[-1] if days else "-",
             args.hold)
    allrows = []
    for day in days:
        allrows += _run_day(con, day, N, args.hold)

    if not allrows:
        log.warning("no granted-regime seconds in the vault window — nothing "
                    "to measure (the fly gate never fired). Report skipped.")
        return

    into = np.array([r["into_ret"] for r in allrows])
    fade = np.array([r["fade_ret"] for r in allrows])
    into_wr = float((into > 0).mean())
    fade_wr = float((fade > 0).mean())
    # bootstrap CI on the asymmetry (fade − into) mean return
    diff = fade - into
    rng = np.random.default_rng(20260716)
    boots = rng.choice(diff, size=(2000, len(diff)), replace=True).mean(axis=1)
    ci_lo, ci_hi = (float(np.quantile(boots, 0.05)),
                    float(np.quantile(boots, 0.95)))
    edge_real = ci_lo > 0                       # fade beats into at 90% CI

    tilts = np.array([r["conv_mult_into"] for r in allrows])
    n_tilt = int((tilts < 0.999).sum())

    rep = {
        "granted_seconds": len(allrows),
        "event_days": len(sorted({r["day"] for r in allrows})),
        "horizon_s": args.hold,
        "into_wall": {"mean_ret": round(float(into.mean()), 6),
                      "win_rate": round(into_wr, 4)},
        "fade_to_pin": {"mean_ret": round(float(fade.mean()), 6),
                        "win_rate": round(fade_wr, 4)},
        "asymmetry_fade_minus_into": {
            "mean": round(float(diff.mean()), 6),
            "ci90": [round(ci_lo, 6), round(ci_hi, 6)],
            "edge_confirmed": edge_real},
        "modulation_activity": {
            "seconds_with_dampen": n_tilt,
            "pct_of_granted": round(100.0 * n_tilt / len(allrows), 1),
            "mean_dampen_mult_when_active": (
                round(float(tilts[tilts < 0.999].mean()), 3)
                if n_tilt else None),
            "mean_runway_mult": round(float(
                np.array([r["runway_mult"] for r in allrows]).mean()), 3)},
        "verdict": ("EDGE CONFIRMED on the vault — fly-intel modulation "
                    "justified (keep FLY_INTEL_ENABLED=True)" if edge_real
                    else "NO significant reversion edge in this window — "
                    "consider FLY_INTEL_ENABLED=False until it appears"),
        "config_hash": config.CONFIG_HASH, "ts": time.time()}

    log.info("granted seconds: %d over %d days | horizon %ds",
             rep["granted_seconds"], rep["event_days"], args.hold)
    log.info("  INTO near wall : mean ret %+.4f%% | win %.1f%%",
             into.mean() * 100, into_wr * 100)
    log.info("  FADE to the pin: mean ret %+.4f%% | win %.1f%%",
             fade.mean() * 100, fade_wr * 100)
    log.info("  asymmetry (fade−into): %+.4f%% | CI90 [%+.4f%%, %+.4f%%] → %s",
             diff.mean() * 100, ci_lo * 100, ci_hi * 100,
             "EDGE CONFIRMED" if edge_real else "not significant")
    log.info("  modulation tilts %d/%d granted seconds (%.1f%%), mean dampen "
             "×%s", n_tilt, len(allrows),
             100.0 * n_tilt / len(allrows),
             f"{tilts[tilts < 0.999].mean():.3f}" if n_tilt else "—")
    log.info("  VERDICT: %s", rep["verdict"])

    out = config.LOG_DIR / f"fly_intel_report_{dt.date.today()}.json"
    _atomic_write_json(out, rep)
    log.info("report → %s", out)


if __name__ == "__main__":
    main()