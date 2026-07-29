"""
APEX OMNI v9.7 — BUTTERFLY HARNESS (falsification before a single rupee)
========================================================================
Grades the BUY-ONLY long-butterfly engine on YOUR vault, opportunity by
opportunity, as the EXACT structure the brain would place — same gate bytes
(core.shortvol.evaluate_gate — the signal is shared with the retired credit
spread), same wall-anchored construction (core.butterfly.build_fly: BUY inner
wing / SELL 2× body at the wall / BUY outer wing), executable DEBIT (wings at
ASK, body at BID), the same FlyBook exit engine (TP/SL on unwind credit, hard
time-flat, cascade veto), FOUR real order legs of Zerodha costs, ONE fly at a
time. Risk is the debit paid — capped, no naked leg.

Information set mirrors live exactly: 1 Hz analytic gamma nowcast where the
archive carries per-contract GEX (since 2026-07-09), the radar's 3-minute
steps before that; the cascade detector runs alongside purely as the
structural VETO. Honest limitation in ink: the vault has no VIX stream, so
the VIX-spike veto cannot be applied historically (its correlated cascade
veto IS) — forward paper evidence carries the full gate; the certificate
blends both.

The forward log is SHARED with shortvol (state/shortvol_forward.jsonl) but
fly rows carry fly_id and spread rows carry spread_id — this harness pairs
ONLY on fly_id, cleanly ignoring any legacy spread rows.

Outputs: console verdict, logs/butterfly_harness_report_<date>.json, and
state/butterfly_certificate.json — pass = ≥SV_CERT_MIN_EVENTS fills over
≥SV_CERT_MIN_DAYS event-days AND bootstrap CI lower bound of mean ₹/event > 0
at SV_CERT_CI. Knob-hash-stamped (fly_knob_hash), fail-closed, forward-blended.

Run after any close:   python tools/butterfly_harness.py [--days N]
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
from core import trial_registry as TR
from core import day_cache as DC
from core import stat_bootstrap as SB                    # noqa: E402
from core import shortvol as SV                           # noqa: E402
from core import butterfly as BF                          # noqa: E402
from core.cascade import CascadeDetector                  # noqa: E402
from core.gamma_nowcast import GammaNowcast               # noqa: E402
from core.instruments import AsOfMapper                   # noqa: E402
from core.diagnostics import _atomic_write_json           # noqa: E402
from macro_gex_v9 import load_macro_archive               # noqa: E402
from nightly_forge_v9 import (trading_days, spot_token_for,     # noqa: E402
                              _latest_at, _eval_hm, _psr)

config.setup_logging("butterfly_harness")
import logging                                           # noqa: E402
log = logging.getLogger("butterfly_harness")

_OH, _OM = (int(x) for x in config.SESSION_OPEN.split(":"))
_OPEN_SOD = _OH * 3600 + _OM * 60


def _spot_series(con, day: str, tok: int, N: int):
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


def _gex_at(nc: GammaNowcast, snap: dict | None, spot: float, ts: float):
    if snap is not None:
        nc.update_snapshot(snap)
    ncast = nc.nowcast(spot, ts)
    if ncast is not None and ncast.flip is not None:
        return ncast.net_gex, ncast.flip, ncast.flip_width
    if snap is not None:
        return (snap.get("net_gex"), snap.get("flip"),
                snap.get("flip_width") or 0.0)
    return None, None, 0.0


def _butterfly_day_worker(args):
    """One day, own process, own read-only SQLite handle.

    Structurally identical to the shortvol worker: _run_day returns
    (closes, skips, blockers) matching DC.run_cached's contract, with no
    cross-day state. verbose stays True — every log line embeds its day and
    short-line appends are atomic, so workers interleave without loss.
    """
    day, stamp, N = args
    import sqlite3 as _sq
    con = _sq.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    try:
        return DC.run_cached("butterfly", stamp, day,
                             lambda: _run_day(con, day, N, verbose=True))
    finally:
        try:
            con.close()
        except Exception:                                  # noqa: BLE001
            pass


def _run_day(con, day: str, N: int, verbose: bool, extra_gate=None):
    """Returns (close_rows, skip_rows, gate_blockers dict) for the fly."""
    from simulation.replay_real_day import load_day
    loaded = load_day(con, day, config.TRADABLE[0])
    if loaded is None:
        return [], [], {}
    _stok, by_sec, ti, bidA, askA = loaded
    mapper = AsOfMapper(dt.date.fromisoformat(day))
    book = BF.FlyBook(risk=None)              # raw event economics
    closes, skips = [], []
    blockers: dict[str, int] = {}
    state = {}
    for idx in config.TRADABLE:
        tok = spot_token_for(con, day, idx)
        if not tok:
            continue
        spots, ts_arr = _spot_series(con, day, tok, N)
        state[idx] = {"spots": spots, "ts": ts_arr,
                      "snaps": load_macro_archive(con, day, idx),
                      "ptr": [0], "nc": GammaNowcast(idx),
                      "det": CascadeDetector(idx),
                      "blocked_until": -1e18,
                      "step": float(config.INDICES[idx]["strike_step"])}
    last_tick: dict[int, int] = {}

    def leg_quotes(spec, t):
        """Three distinct strikes → executable {token: {bid,ask}}."""
        q = {}
        for tokn in (spec.wing_in_token, spec.body_token, spec.wing_out_token):
            k = ti.get(tokn)
            fresh = k is not None and t - last_tick.get(tokn, -99) <= 5
            b = bidA[k, t] if fresh else np.nan
            a = askA[k, t] if fresh else np.nan
            q[tokn] = {"bid": (0.0 if np.isnan(b) else float(b)),
                       "ask": (0.0 if np.isnan(a) else float(a))}
        return q

    for t in range(N):
        for tokn in by_sec.get(t, {}):
            last_tick[tokn] = t
        for idx, st in state.items():
            spot, ts = st["spots"][t], st["ts"][t]
            if np.isnan(spot) or np.isnan(ts):
                continue
            spot, ts = float(spot), float(ts)
            hm = _eval_hm(t)
            snap = _latest_at(st["snaps"], st["ptr"], ts, lambda s: s["ts"])
            gex, flip, width = _gex_at(st["nc"], snap, spot, ts)
            ev = st["det"].update(ts=ts, day=day, spot=spot, flip=flip,
                                  flip_width=width, net_gex=gex,
                                  strike_step=st["step"],
                                  flip_source="replay", flip_age_s=0.0)
            if ev is not None:
                st["blocked_until"] = ts + config.CASCADE_COOLDOWN_S
            hyst = max(config.CASCADE_HYST_MULT * float(width or 0.0),
                       st["step"])
            in_zone = (flip is not None and gex is not None
                       and spot < flip - hyst
                       and gex <= config.CASCADE_NET_GEX_MAX)
            blocked = in_zone or ts < st["blocked_until"]
            # ---- manage the open fly every second --------------------------
            if book.pos is not None and book.pos.spec.index == idx:
                row = book.manage(ts=ts, hm=hm, spot=spot,
                                  quotes=leg_quotes(book.pos.spec, t),
                                  cascade_event=(ev is not None or blocked))
                if row is not None:
                    row["day"] = day
                    closes.append(row)
                    if verbose:
                        log.info("CLOSE %s %s %s %s %-14s ₹%+.2f (debit "
                                 "%.2f→credit %.2f, %ds)", day, row["hm"],
                                 idx, row["side"], row["why"], row["pnl"],
                                 row["debit"], row["close_credit"],
                                 row["hold_s"])
                continue
            if book.pos is not None:
                continue                       # one fly globally
            # ---- gate (shared with shortvol) + attempt ---------------------
            g = SV.evaluate_gate(hm=hm, spot=spot, mac=snap, net_gex_now=gex,
                                 dte=(snap or {}).get("dte"),
                                 strike_step=st["step"], vix_bump=0.0,
                                 cascade_blocked=blocked)
            if not g.ok:
                blockers[g.reason.split("(")[0].strip()[:40]] = \
                    blockers.get(g.reason.split("(")[0].strip()[:40], 0) + 1
                continue
            if extra_gate is not None:
                _xok, _xwhy = extra_gate(idx, snap, spot, ts)
                if not _xok:
                    blockers[f"extra: {_xwhy}"[:40]] = \
                        blockers.get(f"extra: {_xwhy}"[:40], 0) + 1
                    continue
            if ts - book.last_try.get(idx, -1e9) \
                    < config.SV_ATTEMPT_THROTTLE_S:
                continue
            rungs = mapper.hierarchy(idx, spot, g.side)
            spec, why = BF.build_fly(idx, g.side, st["step"],
                                     (snap or {}).get("call_wall") or 0,
                                     (snap or {}).get("put_wall") or 0, rungs)
            if spec is None:
                book.last_try[idx] = ts
                skips.append({"day": day, "hm": hm, "index": idx,
                              "side": g.side, "skip": why})
                continue
            r = book.try_open(ts=ts, hm=hm, spec=spec,
                              quotes=leg_quotes(spec, t),
                              capital=config.FORGE_EVAL_CAPITAL,
                              mode="backtest")
            if "opened" in r:
                if verbose:
                    log.info("OPEN  %s %s %s %s body %.0f ±%.0f debit %.2f "
                             "lots %d (ivr %.2f gex %.1e)", day, hm, idx,
                             g.side, spec.body_k, spec.wing_width,
                             r["debit"], r["lots"], g.iv_rank or 0,
                             g.net_gex or 0)
            elif r.get("skip") not in ("throttled", "book occupied"):
                skips.append({"day": day, **{k: r[k] for k in
                                             ("hm", "index", "side", "skip")
                                             if k in r}})
    # EOD: force-flat any survivor at the last quotes
    if book.pos is not None:
        idx = book.pos.spec.index
        st = state[idx]
        t_last = N - 1
        row = book.manage(
            ts=float(st["ts"][t_last] if not np.isnan(st["ts"][t_last])
                     else time.time()),
            hm="15:29", spot=float(st["spots"][t_last] or 0),
            quotes=leg_quotes(book.pos.spec, t_last), cascade_event=False)
        if row is not None:
            row["day"] = day
            closes.append(row)
    return closes, skips, blockers


def _wilson_lo(wins: int, n: int, ci: float) -> float:
    if n == 0:
        return 0.0
    from statistics import NormalDist
    z = NormalDist().inv_cdf(0.5 + ci / 2.0)
    p = wins / n
    den = 1 + z * z / n
    ctr = p + z * z / (2 * n)
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max((ctr - rad) / den, 0.0)


def _boot_lo(pnls, ci, n_boot):
    r = np.asarray(pnls, float)
    rng = np.random.default_rng(20260709)
    means = rng.choice(r, size=(n_boot, len(r)), replace=True).mean(axis=1)
    return float(np.quantile(means, (1 - ci) / 2.0))


def _forward_fills():
    """Pair the SHARED forward log by fly_id (spread rows are ignored).

    v9.7.1 censoring honesty: a why=DISPLACED close is a PORTFOLIO decision
    by core/displacement.py — the fly was cut for a better trade, not by its
    own exit policy. Grading the fly's certificate on those truncated rows
    would blame (or credit) the fly for the governor's choices, so they are
    EXCLUDED from the blend and returned separately for the report.

    Returns (fills, pending_opens, displaced_rows)."""
    if not SV.FORWARD_LOG.exists():
        return [], [], []
    opens, fills, displaced = {}, [], []
    for line in SV.FORWARD_LOG.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except Exception:                                 # noqa: BLE001
            continue
        fid = e.get("fly_id")
        if fid is None:                       # legacy spread row — skip
            continue
        if e.get("phase") == "open":
            opens[fid] = e
        elif e.get("phase") == "close" and fid in opens:
            o = opens.pop(fid)
            row = {
                "source": "forward", "fly_id": fid,
                "day": str(dt.datetime.fromtimestamp(
                    float(e.get("close_ts") or 0)).date()),
                "index": o.get("index"), "side": o.get("side"),
                "why": e.get("why"), "pnl": float(e.get("pnl") or 0),
                "debit": o.get("debit"), "hold_s": e.get("hold_s"),
                "mode": o.get("mode")}
            if str(e.get("why") or "").startswith("DISPLACED"):
                displaced.append(row)
            else:
                fills.append(row)
    return fills, list(opens.values()), displaced


def _assemble_certificate(bt, fw, skips, blockers, days_scanned,
                          data_span, fw_pending) -> dict:
    fills = bt + fw
    pnls = [r["pnl"] for r in fills]
    ev_days = sorted({r["day"] for r in fills})
    wins = sum(1 for p in pnls if p > 0)
    n = len(pnls)
    mean = float(np.mean(pnls)) if n else 0.0
    ci_lo = _boot_lo(pnls, config.SV_CERT_CI,
                     config.EDGE_BOOTSTRAP_N) if n >= 5 else None
    day_pnl: dict[str, float] = {}
    for r in fills:
        day_pnl[r["day"]] = day_pnl.get(r["day"], 0.0) + r["pnl"]
    psr = _psr(list(day_pnl.values())) if len(day_pnl) >= 3 else None
    reasons = []
    if n < config.SV_CERT_MIN_EVENTS:
        reasons.append(f"events {n} < {config.SV_CERT_MIN_EVENTS}")
    if len(ev_days) < config.SV_CERT_MIN_DAYS:
        reasons.append(f"event-days {len(ev_days)} < {config.SV_CERT_MIN_DAYS}")
    if ci_lo is None or ci_lo <= 0:
        reasons.append(f"bootstrap CI{int(config.SV_CERT_CI*100)} lower "
                       f"{ci_lo} ≤ 0")
    per_why: dict[str, int] = {}
    for r in fills:
        per_why[r.get("why") or "?"] = per_why.get(r.get("why") or "?", 0) + 1
    return {"ok": not reasons, "blocked_by": reasons or None,
            "n_events": n, "n_backtest": len(bt), "n_forward": len(fw),
            "forward_pending": fw_pending,
            "event_days": len(ev_days), "days_scanned": days_scanned,
            "skipped_attempts": len(skips),
            "gate_blockers_top": dict(sorted(blockers.items(),
                                             key=lambda kv: -kv[1])[:8]),
            "mean_pnl": round(mean, 2),
            "sum_pnl": round(float(np.sum(pnls)), 2) if n else 0.0,
            "ci_lo": round(ci_lo, 2) if ci_lo is not None else None,
            "stationary_ci_lo": (round(SB.stat_boot_lo(
                pnls, config.SV_CERT_CI, config.EDGE_BOOTSTRAP_N), 2)
                if len(pnls) >= 5 else None),
            "ci_level": config.SV_CERT_CI,
            "win_rate": round(wins / n, 4) if n else 0.0,
            "win_rate_lo": round(_wilson_lo(wins, n, config.SV_CERT_CI), 4),
            "psr_day": (round(psr["psr"], 3) if psr else None),
            "per_side": {s: sum(1 for r in fills if r.get("side") == s)
                         for s in ("CE", "PE")},
            "per_exit": per_why,
            "instrument": "long_debit_butterfly (buy-only; risk = debit)",
            "vix_veto_note": "not applicable in replay (no VIX stream) — "
                             "forward evidence carries the full gate",
            "eval_capital": config.FORGE_EVAL_CAPITAL,
            "knob_hash": BF.fly_knob_hash(),
            "config_hash": config.CONFIG_HASH,
            "data_span": data_span, "ts": time.time()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0)
    args = ap.parse_args()
    from simulation.scenario_engine import N
    con = sqlite3.connect(config.DB_PATH)
    days = trading_days(con)
    if args.days > 0:
        days = days[-args.days:]
    if getattr(config, "HARNESS_MAX_DAYS", 0) > 0:
        days = days[-config.HARNESS_MAX_DAYS:]   # v10.2 trailing window
    log.info("butterfly harness | %d day(s) %s → %s | knob %s | eval ₹%.0f",
             len(days), days[0], days[-1], BF.fly_knob_hash(),
             config.FORGE_EVAL_CAPITAL)
    bt, skips, blockers = [], [], {}
    _stamp = BF.fly_knob_hash()
    # Independent days; aggregation is concat + counter-merge, both
    # order-insensitive, and map_days returns in DAY ORDER.
    from core.parallel_days import map_days
    _res = map_days(_butterfly_day_worker, [(d, _stamp, N) for d in days],
                    desc="butterfly day")
    for _day, _out in zip(days, _res):
        if _out is None:
            log.warning("  %s: no result (worker failed) — day omitted", _day)
            continue
        c, sk, b = _out
        bt += c
        skips += sk
        for k, v in b.items():
            blockers[k] = blockers.get(k, 0) + v
    for r in bt:
        r["source"] = "backtest"
    fw, pending, displaced = _forward_fills()
    if fw or pending or displaced:
        log.info("forward evidence: %d closed paper fly(s) blended "
                 "(Σ ₹%.2f), %d still open, %d DISPLACED (censored — "
                 "excluded from the certificate)", len(fw),
                 sum(r["pnl"] for r in fw), len(pending), len(displaced))
    cert = _assemble_certificate(bt, fw, skips, blockers, len(days),
                                 [days[0], days[-1]], len(pending))
    cert["forward_displaced"] = len(displaced)   # censored, NOT blended
    TR.register("butterfly", BF.fly_knob_hash(), "primary",
                n_events=cert["n_events"], ok=cert["ok"])
    _atomic_write_json(config.FLY_CERT_PATH, cert)

    # sensitivity — DIAGNOSTIC ONLY (9 trials; deflation applies)
    sens = []
    for ivr in (0.50, 0.60, 0.70):
        for tp in (0.40, 0.50, 0.60):
            o1, o2 = config.SV_IVRANK_MIN, config.SV_FLY_TP_FRAC
            config.SV_IVRANK_MIN, config.SV_FLY_TP_FRAC = ivr, tp
            try:
                rr = []
                _st2 = BF.fly_knob_hash()   # knob-patched → own cache
                for day in days:
                    c, _, _ = DC.run_cached("butterfly", _st2, day,
                                            lambda d=day: _run_day(
                                                con, d, N, verbose=False))
                    rr += c
                pf = [r["pnl"] for r in rr]
                TR.register("butterfly",
                            f"{BF.fly_knob_hash()}:i{ivr}t{tp}",
                            "sensitivity", events=len(pf))
                sens.append({"ivrank_min": ivr, "tp_frac": tp,
                             "events": len(pf),
                             "sum": round(float(np.sum(pf)), 2) if pf else 0.0,
                             "mean": round(float(np.mean(pf)), 2)
                             if pf else None})
            finally:
                config.SV_IVRANK_MIN, config.SV_FLY_TP_FRAC = o1, o2

    cert["family_trials"] = TR.trials_for_deflation("butterfly")
    _atomic_write_json(config.FLY_CERT_PATH, cert)          # re-stamp
    report = {"certificate": cert, "backtest_events": bt,
              "skipped_attempts": skips[-400:],
              "forward_events": fw, "forward_pending": pending,
              "forward_displaced_CENSORED": displaced,
              "sensitivity_DIAGNOSTIC_ONLY": {
                  "note": "9 trials, backtest tier only — choosing a cell and "
                          "re-certifying is multiple testing (Bailey–LdP).",
                  "grid": sens}}
    rpath = config.LOG_DIR / f"butterfly_harness_report_{dt.date.today()}.json"
    _atomic_write_json(rpath, report)

    log.info("─" * 76)
    log.info("VERDICT: %s | %d fills (%d bt + %d fwd, %d open) over %d "
             "event-days / %d scanned | Σ ₹%.2f | mean ₹%.2f | CI%d lo %s | "
             "win %.0f%% (lo %.0f%%) | PSR(day) %s",
             "CERTIFIED ✓" if cert["ok"] else "NOT certified",
             cert["n_events"], cert["n_backtest"], cert["n_forward"],
             cert["forward_pending"], cert["event_days"], len(days),
             cert["sum_pnl"], cert["mean_pnl"], int(config.SV_CERT_CI * 100),
             cert["ci_lo"], 100 * cert["win_rate"],
             100 * cert["win_rate_lo"], cert["psr_day"])
    if cert["blocked_by"]:
        for r in cert["blocked_by"]:
            log.info("  blocked_by: %s", r)
    log.info("top gate blockers: %s", cert["gate_blockers_top"])
    log.info("certificate → %s | report → %s", config.FLY_CERT_PATH, rpath)


if __name__ == "__main__":
    main()