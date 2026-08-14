"""
APEX OMNI — TRAINING & CALIBRATION RUN
======================================
    python run_training.py [--plan] [--force] [--only STEP] [--jobs N]

`run_evening.py` is the full nightly: harnesses, certificates, reports,
research, the analyst brief. On 2026-08-03 it took 8.9 hours, and 4.3 of
those were `fast_lane_report` and the rv/toxicity reports — evidence
tools that produce CERTIFICATES, not inputs the brain reads tomorrow.

This file runs only what actually changes how the system behaves at the
next open:

    calibration artifacts  the brain hot-reloads (ATR proxies, toxicity
                           thresholds, commodity eligibility)
    trained models         the forge's meta, its promotion decision
    regime gates           the post-auction / CAS readiness certificates

Everything else — cascade, shortvol and butterfly harnesses, rv_skill,
rvnet, toxicity, fast_lane, epistemic_health, the research stack, Gemma
— is deliberately absent. Run `run_evening.py` when you want the
evidence; run this when you want the system ready.

FRESHNESS, NOT BLIND REPETITION
-------------------------------
The second idea here matters more than the first. Every step declares
what it PRODUCES and what that product DEPENDS on. Before running, the
chain asks whether the artifact already reflects the current inputs: the
right CONFIG_HASH, and a vault day-count that has not moved. If it does,
the step is skipped with its reason printed.

That makes the common cases cheap in the way they should be:

  • nothing changed since the last run  → minutes
  • one new trading day landed          → only what depends on that day
  • CONFIG_HASH rotated                 → everything re-derives, correctly

`--force` runs everything regardless. Freshness is an optimisation, and
an optimisation you cannot switch off is a liability.

ORDER IS DEPENDENCY, NOT PREFERENCE
-----------------------------------
    prime      day caches — everything downstream reads them
    calibrate  indices, then commodity Track-A, then Track-B. SERIAL by
               necessity: all three merge into ONE calibration.json, and
               2026-08-02 showed what concurrent writers do to it.
    train      commodity forge, then the main forge (trains + promotes)
    verify     the meta-gate replay: the rupee verdict on what was just
               trained, against the freshly rebuilt caches
    regime     post-auction calibration: counts CAS sessions and opens
               the new regimes when their evidence arrives

Logging matches run_evening: live output, per-step logs under
logs/evening/, a master log, and a closing table. Exit code is the count
of failed steps.
"""
import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config                                              # noqa: E402

LOG_DIR = Path("logs/evening")
TIMINGS = Path("state/evening_timings.json")
_PRINT_LOCK = threading.Lock()
_MASTER = None


# ----------------------------------------------------------------- output
def emit(line: str, prefix: str = "") -> None:
    txt = (f"[{prefix}] {line}" if prefix else line).rstrip("\n")
    with _PRINT_LOCK:
        print(txt, flush=True)
        if _MASTER is not None:
            _MASTER.write(txt + "\n")
            _MASTER.flush()


# ------------------------------------------------------------- freshness
def _vault_days() -> int:
    """Trading days present in the vault — the input every calibration and
    training artifact is really a function of."""
    try:
        import sqlite3
        from nightly_forge_v9 import trading_days
        con = sqlite3.connect(str(config.DB_PATH))
        try:
            return len(trading_days(con))
        finally:
            con.close()
    except Exception:                                      # noqa: BLE001
        return -1


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:                                      # noqa: BLE001
        return {}


def fresh_index_calibration(days: int) -> tuple[bool, str]:
    j = _json(config.LOG_DIR / "calibration.json")
    if j.get("config_hash") != config.CONFIG_HASH:
        return False, "config hash moved"
    if int(j.get("days", -1)) != days:
        return False, f"vault has {days} day(s), artifact has {j.get('days')}"
    if not all(i in (j.get("indices") or {}) for i in config.TRADABLE):
        return False, "an index is missing from the artifact"
    return True, f"calibration.json already covers {days} day(s)"


def fresh_commodity_daily(days: int) -> tuple[bool, str]:
    j = _json(config.LOG_DIR / "calibration.json")
    cd = j.get("commodities_daily") or {}
    if j.get("config_hash") != config.CONFIG_HASH or not cd:
        return False, "no Track-A block for this config"
    # Track-A is a 5-year daily pull; it is stale only once a day has ended
    ends = {v.get("window", ["", ""])[1] for v in cd.values()}
    if not ends or max(ends) < (dt.date.today()
                                - dt.timedelta(days=1)).isoformat():
        return False, f"Track-A window ends {max(ends) if ends else '?'}"
    return True, f"Track-A daily current to {max(ends)}"


def fresh_commodity_intraday(days: int) -> tuple[bool, str]:
    j = _json(config.LOG_DIR / "calibration.json")
    if j.get("config_hash") != config.CONFIG_HASH:
        return False, "config hash moved"
    idx = j.get("indices") or {}
    names = list(getattr(config, "COMMODITIES", {}) or {})
    if names and not all(n in idx for n in names):
        return False, "a commodity is missing its Track-B block"
    if not j.get("commodity_calib_written"):
        return False, "Track-B never written"
    return True, "Track-B intraday present for every commodity"


def fresh_forge(days: int) -> tuple[bool, str]:
    p = config.LOG_DIR / f"forge_report_{dt.date.today().isoformat()}.json"
    j = _json(p)
    if not j:
        return False, "no forge report today"
    if j.get("config_hash") != config.CONFIG_HASH:
        return False, "config hash moved"
    if len((j.get("days") or {}).get("all") or []) != days:
        return False, "vault day-count moved since the report"
    return True, f"forge already ran today over {days} day(s)"


# ------------------------------------------------------------------ plan
# (name, script, parallel_ok, freshness_fn | None)

def _day_is_complete() -> tuple[bool, str]:
    """Is today's tape finished? WARNING ONLY — never blocks.

    This matters more here than in run_evening: run_training exists to
    produce the artifacts the brain reads at the NEXT OPEN. Fitting
    calibration or the meta to a half-session is not an evidence problem,
    it is tomorrow's trading. But the operator, not the script, decides —
    a mid-afternoon re-run after a fix is entirely legitimate.
    """
    import datetime as _dt
    import sqlite3 as _sq
    try:
        closes = ["15:40"] + [str(v.get("session_close", "23:55"))
                              for v in (getattr(config, "COMMODITIES", {})
                                        or {}).values()]
        last_hm = max(closes)
        h, m = (int(x) for x in last_hm.split(":")[:2])
        now = _dt.datetime.now()
        today = now.date().isoformat()
        con = _sq.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT MAX(ts_local_ms)/1000 FROM ticks_v9 WHERE "
                "date(ts_local_ms/1000,'unixepoch','localtime')=?",
                (today,)).fetchone()
        finally:
            con.close()
        if not row or not row[0]:
            return True, f"no ticks for {today} — rest day or catch-up run"
        newest = _dt.datetime.fromtimestamp(float(row[0]))
        need = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if newest >= need - _dt.timedelta(minutes=5):
            return True, f"{today} complete (newest tick {newest:%H:%M:%S})"
        return (False, f"{today} is INCOMPLETE — newest tick "
                       f"{newest:%H:%M:%S}, last session closes {last_hm}. "
                       f"Artifacts fitted now will carry a truncated "
                       f"session into tomorrow's open.")
    except Exception as e:                                 # noqa: BLE001
        return True, f"completeness check unavailable ({e})"



def _readiness() -> tuple[bool, list[str]]:
    """Can the brain trade tomorrow on what is on disk RIGHT NOW?

    run_training exists to answer yes. Every artifact below is one the
    brain reads at the next open; anything missing or stale means it opens
    on something older than it should, and nothing else in the system
    would tell you.
    """
    import datetime as _dt
    import json as _json
    out, ok = [], True
    now = _dt.datetime.now()

    def _age_days(p):
        try:
            return (now - _dt.datetime.fromtimestamp(
                p.stat().st_mtime)).total_seconds() / 86400.0
        except OSError:
            return None

    checks = [("calibration", config.LOG_DIR / "calibration.json", 3.0),
              ("meta model", getattr(config, "META_MODEL_PATH", None), 8.0),
              ("model manifest", getattr(config, "MODEL_MANIFEST", None),
               8.0)]
    for name, path, max_age in checks:
        if path is None:
            continue
        a = _age_days(Path(path))
        if a is None:
            out.append(f"MISSING  {name} ({path})")
            ok = False
        elif a > max_age:
            out.append(f"STALE    {name} — {a:.1f}d old (limit {max_age:.0f}d)")
            ok = False
        else:
            out.append(f"ok       {name} — {a:.1f}d old")

    # the calibration artifact must match THIS config world
    try:
        cj = _json.loads((config.LOG_DIR / "calibration.json").read_text())
        if cj.get("config_hash") != config.CONFIG_HASH:
            out.append(f"MISMATCH calibration was written under "
                       f"{cj.get('config_hash')} != {config.CONFIG_HASH}")
            ok = False
        else:
            out.append(f"ok       calibration hash matches "
                       f"({cj.get('days')} day(s))")
    except Exception:                                      # noqa: BLE001
        pass

    # certificates are informational: absent is the SAFE state, never an error
    try:
        from core import label_certificate as LC
        out.append(f"info     training target: {LC.active_label()}")
    except Exception:                                      # noqa: BLE001
        pass
    try:
        from core import cascade as CS
        out.append("info     cascade cert: "
                   + ("present" if CS.load_certificate() else
                      "none — cascade stays paper-explore"))
    except Exception:                                      # noqa: BLE001
        pass
    return ok, out


STEPS = [
    ("prime",       "tools/prime_day_caches.py",        False, None),
    ("idx_calib",   "tools/nightly_calibration.py",     False,
     fresh_index_calibration),
    ("cmdty_daily", "tools/commodity_backfill.py",      False,
     fresh_commodity_daily),
    ("cmdty_intra", "tools/commodity_calibration.py",   False,
     fresh_commodity_intraday),
    ("cmdty_forge", "nightly_commodity_forge.py",       False, None),
    ("forge",       "run_nightly.py",                   False, fresh_forge),
    ("verify",      "tools/meta_gate_replay.py",        False, None),
    ("regime",      "tools/post_auction_calibrate.py",  False, None),
    # v9.9.21: the training TARGET is itself a trained decision. This runs
    # after the forge, judges whether a shadow-derived label would have
    # produced a better model on the matrix the forge just used, and the
    # NEXT run's forge reads core.label_certificate.active_label(). It
    # prints the month countdown every night from the first night, so a
    # coverage problem surfaces in week one rather than week four.
    # v9.9.25: self-arming. nightly_forge publishes the payoff matrix on
    # every run; this reads it and decides for itself whether there is
    # enough evidence to say anything. Below the bar it prints a countdown;
    # above it, it measures and only fits if the measurement clears. No
    # switch to flip, no date to remember.
    ("payoff",      "tools/payoff_study.py",            False, None),
    ("episode",     "tools/episode_study.py",           False, None),
    ("seq",         "tools/seq_study.py",               False, None),
    # v9.9.33: the 2x2 A/B of the day plan and range gate. Also in the
    # weekly chain; here too so a nightly operator sees the evidence
    # accumulate instead of waiting for Saturday. Idempotent and cached —
    # it reads day caches the forge already built.
    ("gate_ab",     "tools/gate_ab_study.py",           False, None),
    ("label_cert",  "tools/label_cert_report.py --issue", False, None),
]


def _run(script: str) -> tuple[str, int, float]:
    t0 = time.time()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # v9.9.21: a step may carry arguments. Popen was handed the whole
    # string as ONE argv element, so "tools/x.py --issue" would have died
    # with "can't open file 'tools/x.py --issue'". Bare script names are
    # byte-identical through shlex.split.
    argv = shlex.split(script)
    lf = LOG_DIR / (Path(argv[0]).stem + ".log")
    with lf.open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen([sys.executable, *argv],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace",
                                bufsize=1)
        for line in proc.stdout:
            f.write(line)
            emit(line)
        rc = proc.wait()
    return script, rc, time.time() - t0


def _timings() -> dict:
    try:
        return json.loads(TIMINGS.read_text(encoding="utf-8"))
    except Exception:                                      # noqa: BLE001
        return {}


def _save_timings(results, prev: dict) -> None:
    out = dict(prev)
    for script, _rc, dt_ in results:
        k = Path(script).stem
        out[k] = round(0.4 * float(dt_) + 0.6 * float(out.get(k, dt_)), 1)
    try:
        TIMINGS.parent.mkdir(parents=True, exist_ok=True)
        tmp = TIMINGS.with_name(f"{TIMINGS.stem}.{os.getpid()}.tmp.json")
        tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
        os.replace(tmp, TIMINGS)
    except Exception as e:                                 # noqa: BLE001
        emit(f"[warn] could not persist timings: {e}")


def main() -> int:
    global _MASTER
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true",
                    help="show what would run, and why, then exit")
    ap.add_argument("--force", action="store_true",
                    help="ignore freshness and run every step")
    ap.add_argument("--only", type=str, default="",
                    help="run a single step by name")
    a = ap.parse_args()

    _ok, _why = _day_is_complete()
    print(("[preflight] WARNING: " if not _ok else "[preflight] ") + _why,
          flush=True)
    print("[cadence] this is the DAILY chain. run_evening.py is weekly.",
          flush=True)

    days = _vault_days()
    known = _timings()
    if days < 0:
        # The vault could not be read. Freshness is an optimisation and an
        # optimisation that guesses is a liability — if we cannot see the
        # inputs, we cannot claim an artifact reflects them. Run everything.
        emit("[warn] vault day-count unreadable — freshness disabled, "
             "running every step")
        a.force = True
    plan = []
    for name, script, _par, fresh in STEPS:
        if a.only and name != a.only:
            continue
        why, skip = "", False
        if fresh and not a.force:
            try:
                skip, why = fresh(days)
            except Exception as e:                         # noqa: BLE001
                skip, why = False, f"freshness check failed ({e})"
        est = known.get(Path(script).stem)
        plan.append((name, script, skip, why, est))

    if a.plan:
        print(f"vault: {days} trading day(s) | config {config.CONFIG_HASH}")
        tot = 0.0
        for name, script, skip, why, est in plan:
            mark = "SKIP" if skip else "RUN "
            if not skip and est:
                tot += est
            print(f"  {mark} {name:12s} {Path(script).name:30s} "
                  f"{('~%.0f min' % (est / 60)) if est else '  ?':>9s}"
                  f"   {why}")
        print(f"  estimated run time: ~{tot / 60:.0f} min "
              f"(run_evening's full chain has been 8-11 h)")
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _MASTER = (LOG_DIR / "_training.log").open("a", encoding="utf-8",
                                               errors="replace")
    emit(f"\n══════ TRAINING RUN {time.strftime('%Y-%m-%d %H:%M:%S')} | "
         f"vault {days} day(s) | config {config.CONFIG_HASH}"
         f"{' | FORCED' if a.force else ''} ══════")
    results, fails, t0 = [], 0, time.time()
    for name, script, skip, why, _est in plan:
        if skip:
            emit(f"───── {name}: SKIPPED — {why} ─────")
            continue
        emit(f"\n═════ {name} ({script}) ═════")
        s_, rc, dt_ = _run(script)
        emit(f"═════ {name} rc={rc} ({dt_:.0f}s) "
             f"[log: logs/evening/{Path(script).stem}.log] ═════")
        results.append((s_, rc, dt_))
        fails += 1 if rc else 0
    emit(f"\n══════ training run complete | {fails} failed step(s) | "
         f"{(time.time() - t0) / 60:.1f} min ══════")
    if results:
        emit(f"{'step':30s} {'rc':>3s} {'mins':>7s}")
        for s_, rc, dt_ in sorted(results, key=lambda r: -r[2]):
            emit(f"{Path(s_).stem:30s} {rc:3d} {dt_ / 60:7.1f}"
                 + ("   <<< FAILED" if rc else ""))
    skipped = [n for n, _s, sk, _w, _e in plan if sk]
    if skipped:
        emit(f"skipped as already current: {', '.join(skipped)}"
             f"   (--force to run anyway)")
    emit("evidence tools (harnesses, rv, toxicity, fast_lane, research, "
         "analyst) were NOT run — use run_evening.py for those")
    _save_timings(results, known)
    if _MASTER is not None:
        _MASTER.close()
    _rok, _lines = _readiness()
    print("\n" + "=" * 62)
    print("READY TO TRADE" if _rok else "NOT READY — see below")
    for _l in _lines:
        print("  " + _l)
    print("=" * 62)
    return fails


if __name__ == "__main__":
    raise SystemExit(main())