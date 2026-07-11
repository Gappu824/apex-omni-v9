"""
APEX OMNI v9.2 — CASCADE HARNESS (falsification before a single live rupee)
============================================================================
Grades the gamma-cascade trigger on YOUR vault, event by event, as the EXACT
trade the brain would place — same detector bytes (core/cascade), same flip
source hierarchy (1 Hz analytic nowcast where the archive carries per-contract
GEX, right-continuous 3-min radar steps where it doesn't — i.e. history is
graded on the CONSERVATIVE information set; the live nowcast can only improve
timing), same strike ladder, ask entry, spread gate, FORGE_EVAL_CAPITAL Kelly
affordability, the standard shaped triple barrier with archived GEX walls, the
0-DTE-aware guillotine, real Zerodha costs, one position at a time.

Outputs:
  • console: every event with its verdict and realized ₹
  • logs/cascade_harness_report_<date>.json — full detail + sensitivity table
  • state/cascade_certificate.json — the LOCK. ok=true only when
        events ≥ CASCADE_CERT_MIN_EVENTS  across  event-days ≥ CASCADE_CERT_MIN_DAYS
        AND bootstrap CI (CASCADE_CERT_CI) lower bound of mean ₹/event > 0.
    Stamped with cascade_knob_hash(): tune any trigger/exit knob and the cert
    self-invalidates until this harness re-passes. The brain arms the live
    cascade path ONLY on a valid certificate (core/cascade.load_certificate).

Statistics discipline: the primary spec above is PRE-REGISTERED (one trial).
The sensitivity grid (z × gex-threshold, 9 cells) is DIAGNOSTIC ONLY and its
deflation note is printed with it — picking a better cell from the grid and
re-certifying is multiple testing, and the report says so in ink. Day-level
PSR is reported as an honesty label (event-days are few; see forge notes).

Run after any close:   python tools/cascade_harness.py [--days N]
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
from core import trial_registry as TR                   # noqa: E402
from core.cascade import (CascadeDetector, cascade_knob_hash)   # noqa: E402
from core.gamma_nowcast import GammaNowcast              # noqa: E402
from core.instruments import AsOfMapper                  # noqa: E402
from core.execution_engine import round_trip_costs       # noqa: E402
from core.diagnostics import _atomic_write_json          # noqa: E402
from macro_gex_v9 import load_macro_archive              # noqa: E402
from nightly_forge_v9 import (trading_days, spot_token_for,     # noqa: E402
                              _shaped_barriers, _hold_seconds,
                              _kelly_budget, _latest_at, _eval_hm, _psr)

config.setup_logging("cascade_harness")
import logging                                           # noqa: E402
log = logging.getLogger("cascade_harness")

_OH, _OM = (int(x) for x in config.SESSION_OPEN.split(":"))
_OPEN_SOD = _OH * 3600 + _OM * 60


def _spot_series(con, day: str, tok: int, N: int):
    """Per-session-second spot (forward-filled) + the epoch ts of each second,
    on the SAME exchange clock the archive and the forge use."""
    arr = np.full(N, np.nan)
    ts0 = np.full(N, np.nan)
    for ts, ltp in con.execute(
            "SELECT ts_ms/1000, ltp FROM ticks_v9 WHERE token=? AND ltp>0 AND "
            "date(ts_local_ms/1000,'unixepoch','localtime')=? ORDER BY ts_ms",
            (tok, day)):
        t = int((ts + 19800) % 86400) - _OPEN_SOD
        if 0 <= t < N:
            arr[t] = ltp
            ts0[t] = ts
    # forward-fill spot AND its epoch anchor (a silent second reuses the last
    # real tick's clock, exactly like the live ring)
    last = np.nan
    last_ts = np.nan
    for t in range(N):
        if np.isnan(arr[t]):
            arr[t] = last
            ts0[t] = last_ts
        else:
            last, last_ts = arr[t], ts0[t]
    return arr, ts0


def _flip_at(nc: GammaNowcast, snap: dict | None, spot: float, ts: float):
    """The flip-source hierarchy, identical to the live brain's: analytic
    nowcast when the snapshot carries per-contract GEX and is fresh, else the
    radar's own stepped numbers, else nothing (detector idles)."""
    if snap is None:
        return None, None, None, "none", 0.0
    if snap.get("gex"):
        nc.update_snapshot(snap)
        n = nc.nowcast(spot, ts)
        if n is not None:
            return (n.flip, n.flip_width, n.net_gex,
                    "nowcast", n.snapshot_age_s)
    age = ts - float(snap["ts"])
    if age > config.MACRO_STALE_S:
        return None, None, None, "stale", age
    return (snap.get("flip"), snap.get("flip_width"), snap.get("net_gex"),
            "radar", age)


def _grade_event(ev, mapper, ti, bidA, askA, last_tick, snap, N):
    """The DEPLOYABLE trade, graded: ladder walk → ask entry → shaped
    barriers on archived walls → bid exits → real costs. Returns a row dict
    (pnl present on a fill, else 'skip' reason)."""
    t = int((ev.ts + 19800) % 86400) - _OPEN_SOD
    budget = _kelly_budget(config.FORGE_EVAL_CAPITAL)
    pick = None
    skip = {"unharvested": 0, "stale>5s": 0, "one-sided": 0,
            "spread>cap": 0, "unaffordable": 0}
    for r in mapper.hierarchy(ev.index, ev.spot, ev.direction):
        k = ti.get(r["token"])
        if k is None:
            skip["unharvested"] += 1
            continue
        if t - last_tick.get(r["token"], -99) > 5:
            skip["stale>5s"] += 1
            continue
        b_, a_ = bidA[k, t], askA[k, t]
        if np.isnan(b_) or np.isnan(a_) or b_ <= 0 or a_ <= 0:
            skip["one-sided"] += 1
            continue
        mid = (b_ + a_) / 2.0
        if (a_ - b_) / max(mid, 0.05) > config.MAX_ENTRY_SPREAD_PCT:
            skip["spread>cap"] += 1
            continue
        if mid * r["lot"] > budget:
            skip["unaffordable"] += 1
            continue
        pick = (k, float(a_), int(r["lot"]), float(r["strike"]))
        break
    base = {"day": str(dt.datetime.fromtimestamp(ev.ts).date()),
            "hm": _eval_hm(t), **ev.as_dict()}
    if pick is None:
        base["skip"] = " ".join(f"{k}:{n}" for k, n in skip.items() if n) \
            or "empty ladder"
        return base
    k, e, lot, K = pick                                   # e = ASK ★
    dte = float(snap.get("dte") or 9.0) if snap else 9.0
    horizon = _hold_seconds(dte)
    mins_left = max((N - t) / 60.0, 1.0)
    T_ = max(dte, 0.02) / 365.0
    tp, sl = _shaped_barriers(e, ev.spot, K, T_, mins_left,
                              ev.direction == "CE",
                              (snap or {}).get("call_wall"),
                              (snap or {}).get("put_wall"))
    tp, sl = float(tp), float(sl)
    seg = bidA[k, t + 1:t + 1 + horizon]
    if seg.size == 0 or np.all(np.isnan(seg)):
        base["skip"] = "no forward bids"
        return base
    itp = int(np.argmax(seg >= tp)) if np.any(seg >= tp) else None
    isl = int(np.argmax(seg <= sl)) if np.any(seg <= sl) else None
    if itp is not None and (isl is None or itp < isl):
        exitp, off, how = float(tp), itp, "target"
    elif isl is not None:
        exitp, off, how = float(sl), isl, "stop"
    else:
        valid = np.nonzero(~np.isnan(seg))[0]
        exitp, off, how = float(seg[valid[-1]]), int(valid[-1]), "guillotine"
    pnl = (exitp - e) * lot - round_trip_costs(e * lot, exitp * lot)
    base.update({"strike": K, "lot": lot, "entry_ask": round(e, 2),
                 "exit": round(exitp, 2), "exit_how": how,
                 "hold_s": off + 1, "tp": round(tp, 2), "sl": round(sl, 2),
                 "pnl": round(float(pnl), 2), "exit_t": t + off + 1})
    return base


def _run_day(con, day: str, N: int, primary_rows: list | None,
             det_side: str = "below", extra_ok=None):
    """One pass over one day: build spot/flip series, run the shared
    detector at 1 Hz, grade each event, ONE open position at a time.
    Returns (event rows, upside_zone_candidate_seconds) — the latter counts
    seconds where net GEX is beyond threshold and the impulse qualifies but
    spot sits ABOVE flip+hyst: the mirrored configuration the current zone
    (spot<flip) deliberately excludes. DIAGNOSTIC ONLY — if these accumulate,
    an above-flip variant earns its own prespecified harness pass; widening
    the live trigger without one would be silent multiple testing."""
    from simulation.replay_real_day import load_day
    loaded = load_day(con, day, config.TRADABLE[0])
    if loaded is None:
        return [], 0
    _stok, by_sec, ti, bidA, askA = loaded
    mapper = AsOfMapper(dt.date.fromisoformat(day))
    rows = []
    upside = 0
    open_until = -1
    state = {}
    for idx in config.TRADABLE:
        tok = spot_token_for(con, day, idx)
        if not tok:
            continue
        spots, ts_arr = _spot_series(con, day, tok, N)
        snaps = load_macro_archive(con, day, idx)
        state[idx] = {"spots": spots, "ts": ts_arr, "snaps": snaps,
                      "ptr": [0], "det": CascadeDetector(idx, zone_side=det_side),
                      "nc": GammaNowcast(idx),
                      "step": float(config.INDICES[idx]["strike_step"])}
    last_tick: dict[int, int] = {}
    for t in range(N):
        for tok in by_sec.get(t, {}):
            last_tick[tok] = t
        for idx, st in state.items():
            spot = st["spots"][t]
            ts = st["ts"][t]
            if np.isnan(spot) or np.isnan(ts):
                continue
            snap = _latest_at(st["snaps"], st["ptr"], ts,
                              lambda s: s["ts"])
            flip, width, gex, src, age = _flip_at(st["nc"], snap,
                                                  float(spot), float(ts))
            ev = st["det"].update(ts=float(ts), day=day, spot=float(spot),
                                  flip=flip, flip_width=width, net_gex=gex,
                                  strike_step=st["step"],
                                  flip_source=src, flip_age_s=float(age))
            _z = st["det"].last_z
            if (ev is None and _z is not None and flip is not None
                    and gex is not None
                    and abs(_z) >= config.CASCADE_VEL_Z
                    and gex <= config.CASCADE_NET_GEX_MAX):
                _hy = max(config.CASCADE_HYST_MULT * float(width or 0.0),
                          st["step"])
                if spot > flip + _hy:
                    upside += 1                   # mirrored config — diagnostic
            if ev is None:
                continue
            if extra_ok is not None:
                _xok, _xwhy = extra_ok(idx, snap, float(spot), float(ts))
                if not _xok:
                    rows.append({"day": day, "hm": _eval_hm(t),
                                 **ev.as_dict(), "skip": f"extra: {_xwhy}"})
                    continue
            if t < open_until:                    # one position at a time
                row = {"day": day, "hm": _eval_hm(t), **ev.as_dict(),
                       "skip": "position open"}
            else:
                row = _grade_event(ev, mapper, ti, bidA, askA,
                                   last_tick, snap, N)
                if "pnl" in row:
                    open_until = row["exit_t"]
            rows.append(row)
            if primary_rows is not None:
                mark = (f"₹{row['pnl']:+.2f} via {row['exit_how']}"
                        if "pnl" in row else f"SKIP {row['skip']}")
                log.info("EVENT %s %s %s %-4s %-11s z=%+.2f flip %.0f "
                         "spot %.0f [%s %.0fs] → %s", day, row["hm"], idx,
                         ev.direction, ev.kind, ev.z, ev.flip, ev.spot,
                         ev.flip_source, ev.flip_age_s, mark)
    return rows, upside


def _wilson_lo(wins: int, n: int, ci: float) -> float:
    """Wilson score lower bound on the win rate — the certificate's sizing
    probability is the pessimistic one, never the point estimate."""
    if n == 0:
        return 0.0
    from statistics import NormalDist
    z = NormalDist().inv_cdf(0.5 + ci / 2.0)
    p = wins / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max((ctr - rad) / den, 0.0)


def _bootstrap_ci_lo(pnls: list[float], ci: float, n_boot: int) -> float:
    r = np.asarray(pnls, float)
    rng = np.random.default_rng(20260708)                # reproducible
    means = rng.choice(r, size=(n_boot, len(r)), replace=True).mean(axis=1)
    return float(np.quantile(means, (1 - ci) / 2.0))


# ==========================================================================
# FORWARD EVIDENCE (v9.2.1) — the paper-explore tier's harvest. The brain
# appends a join record (symbol + entry_ts) at every cascade fill; here we
# pair it against the execution ledger's BUY_FILL/SELL_FILL rows and blend
# realized forward ₹ into the certificate. Forward fills are out-of-sample
# BY CONSTRUCTION (the trigger fired on data the detector had never seen and
# executed through the real order path) — the strongest evidence class; the
# 2026-07-09 run proved the backtest tier alone cannot certify on
# pre-widening vault history.
# ==========================================================================
_FWD_JOIN_TOL_S = 180.0


def _closed_pairs(ledger_path) -> dict[str, list[tuple[float, float, float]]]:
    """symbol → [(buy_ts, sell_ts, pnl_after_costs)], BUY→SELL paired the
    same way core.edge_audit does (open-interest map keyed on symbol)."""
    import csv
    p = Path(ledger_path)
    out: dict[str, list] = {}
    if not p.exists():
        return out
    opens: dict[str, dict] = {}
    for r in csv.DictReader(p.open(encoding="utf-8")):
        ev = r.get("event")
        sym = r.get("symbol") or ""
        if ev == "BUY_FILL":
            opens[sym] = r
        elif ev == "SELL_FILL" and sym in opens:
            b = opens.pop(sym)
            try:
                out.setdefault(sym, []).append(
                    (float(b.get("ts") or 0), float(r.get("ts") or 0),
                     float(r.get("pnl") or 0)))
            except ValueError:
                continue
    return out


def _forward_fills():
    """(matched_rows, pending). Matched: forward-log entries whose ledger
    BUY_FILL sits within ±_FWD_JOIN_TOL_S of the recorded entry_ts, with the
    paired SELL's realized after-cost ₹. Pending: entered but not yet closed
    in the ledger (position still open, or today's session unfinished)."""
    from core.cascade import FORWARD_LOG
    if not FORWARD_LOG.exists():
        return [], []
    pairs = _closed_pairs(config.LEDGER_PATH)
    used: set[tuple[str, float]] = set()
    matched, pending = [], []
    for line in FORWARD_LOG.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except Exception:                                 # noqa: BLE001
            continue
        sym, ets = e.get("symbol") or "", float(e.get("entry_ts") or 0)
        hit = None
        for (bts, sts, pnl) in pairs.get(sym, []):
            if abs(bts - ets) <= _FWD_JOIN_TOL_S and (sym, bts) not in used:
                hit = (bts, sts, pnl)
                break
        if hit is None:
            pending.append(e)
            continue
        used.add((sym, hit[0]))
        matched.append({
            "source": "forward", "symbol": sym,
            "day": str(dt.datetime.fromtimestamp(hit[0]).date()),
            "hm": dt.datetime.fromtimestamp(hit[0]).strftime("%H:%M"),
            "index": e.get("index"), "direction": e.get("direction"),
            "kind": e.get("kind"), "z": e.get("z"), "mode": e.get("mode"),
            "pnl": round(float(hit[2]), 2),
            "hold_s": int(max(hit[1] - hit[0], 0))})
    return matched, pending


def _assemble_certificate(bt_fills: list[dict], fw_fills: list[dict],
                          skips: list[dict], days_scanned: int,
                          data_span: list[str], upside_s: int,
                          fw_pending: int) -> dict:
    """The blended verdict, factored pure for unit testing. Thresholds apply
    to the UNION of backtest and forward fills; forward rows carry full
    weight (they are the higher evidence class)."""
    fills = bt_fills + fw_fills
    pnls = [r["pnl"] for r in fills]
    ev_days = sorted({r["day"] for r in fills})
    wins = sum(1 for p in pnls if p > 0)
    n = len(pnls)
    mean = float(np.mean(pnls)) if n else 0.0
    ci_lo = _bootstrap_ci_lo(pnls, config.CASCADE_CERT_CI,
                             config.EDGE_BOOTSTRAP_N) if n >= 5 else None
    day_pnl: dict[str, float] = {}
    for r in fills:
        day_pnl[r["day"]] = day_pnl.get(r["day"], 0.0) + r["pnl"]
    psr = _psr(list(day_pnl.values())) if len(day_pnl) >= 3 else None
    wr = wins / n if n else 0.0
    wr_lo = _wilson_lo(wins, n, config.CASCADE_CERT_CI)
    reasons = []
    if n < config.CASCADE_CERT_MIN_EVENTS:
        reasons.append(f"events {n} < {config.CASCADE_CERT_MIN_EVENTS}")
    if len(ev_days) < config.CASCADE_CERT_MIN_DAYS:
        reasons.append(f"event-days {len(ev_days)} < "
                       f"{config.CASCADE_CERT_MIN_DAYS}")
    if ci_lo is None or ci_lo <= 0:
        reasons.append(f"bootstrap CI{int(config.CASCADE_CERT_CI*100)} "
                       f"lower {ci_lo} ≤ 0")
    return {"ok": not reasons, "blocked_by": reasons or None,
            "n_events": n, "n_backtest": len(bt_fills),
            "n_forward": len(fw_fills), "forward_pending": fw_pending,
            "event_days": len(ev_days), "days_scanned": days_scanned,
            "skipped_triggers": len(skips),
            "mean_pnl": round(mean, 2),
            "sum_pnl": round(float(np.sum(pnls)), 2) if n else 0.0,
            "ci_lo": round(ci_lo, 2) if ci_lo is not None else None,
            "ci_level": config.CASCADE_CERT_CI,
            "win_rate": round(wr, 4), "win_rate_lo": round(wr_lo, 4),
            "psr_day": (round(psr["psr"], 3) if psr else None),
            "per_kind": {k: sum(1 for r in fills if r.get("kind") == k)
                         for k in ("flip_break", "zone_impulse")},
            "per_dir": {d: sum(1 for r in fills if r.get("direction") == d)
                        for d in ("PE", "CE")},
            "upside_zone_candidate_s": int(upside_s),
            "paper_explore": bool(getattr(config, "CASCADE_PAPER_EXPLORE",
                                          False)),
            "eval_capital": config.FORGE_EVAL_CAPITAL,
            "knob_hash": cascade_knob_hash(),
            "config_hash": config.CONFIG_HASH,
            "data_span": data_span, "ts": time.time()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0,
                    help="only the last N vault days (0 = all)")
    args = ap.parse_args()
    from simulation.scenario_engine import N
    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)
    if args.days > 0:
        days = days[-args.days:]
    log.info("cascade harness | %d day(s) %s → %s | knob %s | eval ₹%.0f",
             len(days), days[0], days[-1], cascade_knob_hash(),
             config.FORGE_EVAL_CAPITAL)

    all_rows: list[dict] = []
    upside_total = 0
    for day in days:
        rr, up = _run_day(con, day, N, primary_rows=all_rows)
        all_rows += rr
        upside_total += up

    bt_fills = [r for r in all_rows if "pnl" in r]
    for r in bt_fills:
        r["source"] = "backtest"
    skips = [r for r in all_rows if "pnl" not in r]
    fw_fills, fw_pending = _forward_fills()
    if fw_fills:
        log.info("forward evidence: %d realized paper cascade fill(s) "
                 "blended (Σ ₹%.2f), %d pending in the ledger",
                 len(fw_fills), sum(r["pnl"] for r in fw_fills),
                 len(fw_pending))
    elif fw_pending:
        log.info("forward evidence: %d entry(ies) logged, none closed in "
                 "the ledger yet", len(fw_pending))

    cert = _assemble_certificate(bt_fills, fw_fills, skips, len(days),
                                 [days[0], days[-1]], upside_total,
                                 len(fw_pending))
    TR.register("cascade", cascade_knob_hash(), "primary",
                n_events=cert["n_events"], ok=cert["ok"])
    _atomic_write_json(config.CASCADE_CERT_PATH, cert)

    # ---- SENSITIVITY (diagnostic ONLY — 9 trials; deflation applies) -------
    sens = []
    for z in (1.5, 2.0, 2.5):
        for gm in (0.5, 1.0, 2.0):
            oz, og = config.CASCADE_VEL_Z, config.CASCADE_NET_GEX_MAX
            config.CASCADE_VEL_Z = z
            config.CASCADE_NET_GEX_MAX = og * gm
            try:
                rr = []
                for day in days:
                    _r, _ = _run_day(con, day, N, primary_rows=None)
                    rr += _r
                pf = [r["pnl"] for r in rr if "pnl" in r]
                TR.register("cascade",
                            f"{cascade_knob_hash()}:z{z}g{gm}",
                            "sensitivity", events=len(pf))
                sens.append({"z": z, "gex_mult": gm, "events": len(pf),
                             "sum": round(float(np.sum(pf)), 2) if pf else 0.0,
                             "mean": round(float(np.mean(pf)), 2) if pf else None})
            finally:
                config.CASCADE_VEL_Z, config.CASCADE_NET_GEX_MAX = oz, og

    cert["family_trials"] = TR.trials_for_deflation("cascade")
    _atomic_write_json(config.CASCADE_CERT_PATH, cert)         # re-stamp with the trial count
    report = {"certificate": cert,
              "backtest_events": all_rows,
              "forward_events": fw_fills,
              "forward_pending": fw_pending,
              "sensitivity_DIAGNOSTIC_ONLY": {
                  "note": "9 trials, backtest tier only — any cell chosen "
                          "from this grid and re-certified is multiple "
                          "testing; deflate accordingly (Bailey–LdP).",
                  "grid": sens},
              "upside_zone_note": (
                  "upside_zone_candidate_s counts seconds with net GEX "
                  "beyond threshold and a qualifying impulse while spot sat "
                  "ABOVE flip+hyst — the configuration the current zone "
                  "excludes. If material, an above-flip variant deserves its "
                  "OWN prespecified harness pass; do not widen the live "
                  "trigger without one.")}
    rpath = config.LOG_DIR / \
        f"cascade_harness_report_{dt.date.today()}.json"
    _atomic_write_json(rpath, report)

    log.info("─" * 76)
    log.info("VERDICT: %s | %d fills (%d backtest + %d forward, %d pending) "
             "over %d event-days / %d scanned | Σ ₹%.2f | mean ₹%.2f | CI%d "
             "lo %s | win %.0f%% (lo %.0f%%) | PSR(day) %s | upside-zone %ds",
             "CERTIFIED ✓" if cert["ok"] else "NOT certified",
             cert["n_events"], cert["n_backtest"], cert["n_forward"],
             cert["forward_pending"], cert["event_days"], len(days),
             cert["sum_pnl"], cert["mean_pnl"],
             int(config.CASCADE_CERT_CI * 100), cert["ci_lo"],
             100 * cert["win_rate"], 100 * cert["win_rate_lo"],
             cert["psr_day"], cert["upside_zone_candidate_s"])
    if cert["blocked_by"]:
        for r in cert["blocked_by"]:
            log.info("  blocked_by: %s", r)
    log.info("certificate → %s | report → %s", config.CASCADE_CERT_PATH,
             rpath)


if __name__ == "__main__":
    main()