"""
APEX OMNI v9.9.2 — THE EVENING RITUAL, ONE COMMAND, PARALLEL BY DEPENDENCY
==========================================================================
    python run_evening.py [--serial] [--jobs N] [--plan]

The chain used to be one serial pipe of 16 tools; on 2026-07-29 it ran
~12 h because a config-hash rotation invalidated every raw day cache and
the first tool paid 31 sqlite replays inline. v9.9.2 restructures the
evening around three facts that were always true:

  1. Raw day caches are shared by everything → build them ONCE, first,
     in parallel (tools/prime_day_caches.py). After that stage every
     tool cache-hits (and the cache stamp itself now fingerprints only
     the 9 build-path constants, so decision-knob tuning never triggers
     a rebuild again).
  2. The evidence tools are independent of each other → they run
     CONCURRENTLY in dependency groups. The only true edges are kept:
     nightly_calibration BEFORE toxicity/fast-lane (they read tonight's
     thresholds); registry-writing harnesses BEFORE epistemic_health;
     reports BEFORE gemma_analyst (it summarises them); everything
     BEFORE run_nightly; meta_gate_replay LAST (it examines the freshly
     promoted artifact on the freshly built caches).
  3. Heavy solo stages (Gemma on the GPU, the two forges) keep the whole
     machine to themselves.

Per-step stdout/stderr streams to logs/evening/<step>.log so parallel
output never interleaves; the console shows one start line and one
"rc=N (Ns)" line per step — same contract as before. A failing step is
reported and its GROUP continues; exit code is the count of failed
steps — 0 is a clean evening. --serial reproduces the legacy one-at-a-
time behaviour with identical ordering.
"""
import argparse
import json
import os
import shlex
import subprocess
import threading
import sys
import time
from pathlib import Path

# ---- dependency groups (order within a group is start order only) ----
GROUPS = [
    ("prime",     False, ["tools/prime_day_caches.py"]),
    ("evidence",  True,  ["tools/cascade_harness.py",
                          "tools/shortvol_harness.py",
                          "tools/butterfly_harness.py",
                          "tools/rv_skill_report.py",
                          "tools/fly_intel_report.py"]),
    # v9.9.5: the three calibration writers share ONE artifact
    # (logs/calibration.json). They now merge rather than overwrite, but
    # concurrent read-modify-write is still a race — so they also run
    # SERIALLY, indices first, then Track-A daily, then Track-B intraday.
    # This is the group that feeds the commodity brain; it is not a place
    # for speed. (2026-08-02: a parallel run silently erased every
    # commodity calibration.)
    ("calibrate", False, ["tools/nightly_calibration.py",
                          "tools/commodity_backfill.py",
                          "tools/commodity_calibration.py"]),
    ("derived",   True,  ["tools/rvnet_train.py",
                          "tools/toxicity_report.py",     # reads tonight's
                          "tools/fast_lane_report.py",    #   thresholds
                          "tools/epistemic_health.py"]),  # reads registry
    ("forge",     False, ["nightly_commodity_forge.py",
                          "run_nightly.py"]),
    ("verdict",   False, ["tools/meta_gate_replay.py"]),   # needs the forge
    # v9.9.4: research runs on the freshly rebuilt caches. Both tools
    # gate themselves on core.capability_ladder — at STAGE BLIND they
    # log why and exit in seconds, so this group costs nothing until the
    # vault can actually answer the questions it asks.
    ("discovery", True,  ["tools/horizon_sweep.py",
                          "tools/feature_discovery.py",
                          # the A/B verdict: for every feature group the
                          # system carries, did it make ranking and rupees
                          # better, worse, or is the vault still too thin
                          # to say? Paired on real days, FDR-corrected.
                          "tools/ab_ablation.py",
                          # v9.9.10: what did each REAL fill leave on the
                          # table, and which exit rule would have left
                          # less? MFE/MAE capture + paired counterfactual
                          # exit policies over the whole session.
                          # v9.9.13: rewritten. Correct FIFO reconstruction
                          # (short legs and re-entries are visible again),
                          # the DATE-AWARE session window (15:40 after the
                          # reform), a staleness mask so a pruned leg reads
                          # as dead instead of flat, and --promote, which
                          # closes the loop that has been open since
                          # v9.9.10: the verdict now goes through
                          # core.exit_policy_store's ladder gate instead of
                          # into a JSON nobody reads.
                          "tools/trade_potential.py --promote",
                          # v9.9.12: measures the 15:35-15:40 window and
                          # OPENS it once a week of data proves the median
                          # move can pay the spread at 2:1 odds.
                          "tools/post_auction_calibrate.py",
                          # v9.9.14: the ENTRY-side twin of trade_potential.
                          # Sweeps the pre-registered conviction-bar grid as
                          # a POLICY (one book, real cooldown/throttle/
                          # curfew/affordability, one shared exit) rather
                          # than grading blocked signals in isolation, and
                          # judges it with a Westfall-Young max-statistic
                          # because the grid is a nested correlated ladder.
                          # Leads every report with the capacity arithmetic:
                          # 1 concurrent position + 60min guillotine + 180s
                          # cooldown caps the session at ~5 trades, so the
                          # bar chooses WHICH slots fill, never HOW MANY.
                          # Promotion is OFF by default.
                          "tools/entry_bar_study.py",
                          # v9.9.25: also here, so a weekly-only operator
                          # still gets the R-target measurement even if
                          # run_training is not on a nightly schedule.
                          # Idempotent — it reads a published matrix and
                          # writes a report; running it twice costs seconds.
                          "tools/payoff_study.py",
                          "tools/episode_study.py",
                          "tools/seq_study.py",
                          # v9.9.29: 2x2 factorial A/B of the day plan and
                          # the range gate on the real tape, paired by
                          # session. Both gates ship OFF; this is what tells
                          # you what they WOULD have done before either is
                          # armed. Weekly, not nightly: it replays every
                          # session four times and changes no artifact the
                          # brain reads tomorrow.
                          "tools/gate_ab_study.py",
                          "tools/label_cert_report.py"]),
    # v9.9.3: the analyst runs LAST — its brief can now digest the forge
    # verdicts and the ₹ exam, and its model no longer squats in RAM ahead
    # of the heaviest stages.
    ("analyst",   False, ["tools/gemma_analyst.py"]),      # GPU — solo
]

LOG_DIR = Path("logs/evening")
TIMINGS = Path("state/evening_timings.json")



def _day_is_complete(force: bool = False) -> tuple[bool, str]:
    """Refuse to analyse a session that has not finished yet.

    There was no guard here at all. The chain would run happily at 16:00,
    read a vault whose MCX legs stop at whatever second the operator hit
    enter, and produce calibrations, a forge retrain and a bar sweep over a
    truncated day — every one of them silently wrong, none of them raising.
    Equity closes 15:40; SILVER runs to 23:55. The whole tape matters.
    """
    import datetime as _dt
    import sqlite3 as _sq
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import config
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
            return (True, f"no ticks for {today} — treating as a rest day / "
                          f"catch-up run")
        newest = _dt.datetime.fromtimestamp(float(row[0]))
        need = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if newest >= need - _dt.timedelta(minutes=5):
            return (True, f"newest tick {newest:%H:%M:%S} ≥ last close "
                          f"{last_hm} — the day is complete")
        return (False, f"{today} is INCOMPLETE — newest tick {newest:%H:%M:%S}, but "
                       f"the last session closes {last_hm}. Anything that "
                       f"fits today's data (calibration, the forge retrain, "
                       f"the bar sweep) will fit a truncated session, and "
                       f"none of them would raise.")
    except Exception as e:                                 # noqa: BLE001
        return (True, f"completeness check unavailable ({e}) — proceeding")



_CADENCE_STAMP = "evening_last_run.json"


def _cadence_note(force: bool) -> str:
    """run_evening is the WEEKLY chain; run_training is the daily one.

    The split is not stylistic. run_training produces the artifacts the
    brain reads at the next open (calibration, the meta, the regime gates)
    and finishes in hours. run_evening adds fifteen more steps of evidence
    and research — harnesses, sweeps, certificates — that change no
    decision tomorrow morning. Running the heavy chain nightly costs a
    working day per week and buys nothing the brain uses; running the light
    one weekly means trading Tuesday through Friday on Monday's model, with
    nothing that warns you.
    """
    import datetime as _dt
    import json as _json
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        import config
        p = config.STATE_DIR / _CADENCE_STAMP
        now = _dt.datetime.now()
        prev = None
        if p.exists():
            try:
                prev = _dt.datetime.fromisoformat(
                    _json.loads(p.read_text())["last"])
            except Exception:                              # noqa: BLE001
                prev = None
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps({"last": now.isoformat()}))
        if prev is None:
            return "first recorded run — weekly cadence starts now"
        age = (now - prev).total_seconds() / 86400.0
        if age < 5.0 and not force:
            return (f"last full run was {age:.1f} day(s) ago. This is the "
                    f"WEEKLY chain; run_training.py is the daily one. "
                    f"Running anyway — most steps will be served from cache.")
        return f"last full run {age:.1f} day(s) ago — on cadence"
    except Exception as e:                                 # noqa: BLE001
        return f"cadence stamp unavailable ({e})"


def _load_timings() -> dict:
    try:
        return json.loads(TIMINGS.read_text(encoding="utf-8"))
    except Exception:                                      # noqa: BLE001
        return {}


def _save_timings(results, prev: dict) -> None:
    """Exponentially-smoothed per-step durations. Smoothing (α=0.4) keeps
    one anomalous night from re-ordering everything, while still tracking
    a tool that genuinely got faster."""
    out = dict(prev)
    for script, _rc, dt in results:
        k = Path(shlex.split(script)[0]).stem
        out[k] = round(0.4 * float(dt) + 0.6 * float(out.get(k, dt)), 1)
    try:
        TIMINGS.parent.mkdir(parents=True, exist_ok=True)
        tmp = TIMINGS.with_name(f"{TIMINGS.stem}.{os.getpid()}.tmp.json")
        tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
        os.replace(tmp, TIMINGS)
    except Exception as e:                                 # noqa: BLE001
        print(f"[warn] could not persist step timings: {e}", flush=True)


def _lpt_order(steps: list[str], timings: dict) -> list[str]:
    """Longest-Processing-Time-first (Graham 1969).

    v9.9.8. On 2026-08-03 the derived group ran [rvnet 3449s, toxicity
    4697s, epistemic 0s, fast_lane 11982s] two-at-a-time in LIST order.
    fast_lane — the longest job by a factor of three — was picked up
    LAST, at t=3449, so the group ended at 15431s instead of the 11982s
    its own critical path required: 57 minutes of two-slot machine idling
    behind one long tail. Scheduling the longest job first is the classic
    fix and carries a proven makespan bound of (4/3 − 1/3m) × optimal.
    A step with no history sorts FIRST (assumed long), so a newly added
    tool is never starved at the end of a group.
    """
    return sorted(steps,
                  key=lambda s_: -float(timings.get(Path(shlex.split(s_)[0]).stem,
                                                    float("inf"))))
_PRINT_LOCK = threading.Lock()
_MASTER = None          # every console line also lands here


def emit(line: str, prefix: str = "") -> None:
    """The ONLY console writer. Holds a lock for the whole write so lines
    from concurrently-running tools can never interleave mid-line, and
    mirrors everything into logs/evening/_evening.log — one file that is
    the complete story of the night, in order."""
    txt = (f"[{prefix}] {line}" if prefix else line).rstrip("\n")
    with _PRINT_LOCK:
        print(txt, flush=True)
        if _MASTER is not None:
            _MASTER.write(txt + "\n")
            _MASTER.flush()


def _inner_share(jobs: int) -> str:
    """Day-workers each concurrent tool may open.

    v9.9.3 FIX: this used to divide the RAM-capped base measured ONCE in
    the parent at chain start — on 2026-07-31 the parent read low RAM,
    handed every child a share of 1, and the whole night ran serial even
    after memory freed (evidence 5.1 h, fast_lane 5.9 h at 1 worker). The
    share now divides the CPU-ONLY base; each child's pool then applies
    its OWN live RAM cap at the moment it actually opens (parallel_days
    measures per pool). Parent's momentary reading can no longer pin the
    night."""
    import os as _os
    cpu = _os.cpu_count() or 2
    cpu_base = max(1, min(6, cpu // 2))
    return str(max(1, cpu_base // max(1, jobs)))


def _run_step(script: str, live: bool = True, inner: str | None = None
              ) -> tuple[str, int, float]:
    """EVERY step streams live. Solo stages print bare lines; steps inside
    a parallel group are prefixed with the tool name, and `emit` serialises
    whole lines, so eight concurrent tools remain readable and each still
    gets its own clean, unprefixed logs/evening/<step>.log."""
    t0 = time.time()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # v9.9.13: a step may now carry arguments ("tools/x.py --promote").
    # Popen was handed the whole string as ONE argv element, so any step
    # with a flag died with "can't open file 'tools/x.py --promote'".
    # shlex.split makes every step in every group argument-capable, and
    # leaves bare script names byte-identical to before.
    argv = shlex.split(script)
    lf = LOG_DIR / (Path(argv[0]).stem + ".log")
    env = dict(os.environ)
    if inner:                       # parallel group: hand out a share
        env["APEX_INNER_WORKERS"] = inner
    else:                           # solo stage: full machine, as before
        env.pop("APEX_INNER_WORKERS", None)
    tag = Path(argv[0]).stem if inner else ""
    with lf.open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen([sys.executable, *argv], env=env,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8",
                                errors="replace", bufsize=1)
        for line in proc.stdout:
            f.write(line)                 # clean per-step record
            emit(line, tag)               # prefixed, serialised console
        rc = proc.wait()
    return script, rc, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", action="store_true",
                    help="legacy one-at-a-time execution, same order")
    ap.add_argument("--jobs", type=int, default=2,
                    help="max concurrent tools inside a parallel group "
                         "(default 2; each still gets its own day-worker "
                         "pool, sized so the machine-wide total is capped)")
    ap.add_argument("--strict", action="store_true",
                    help="refuse to run on an incomplete tape (for "
                         "unattended/scheduled runs, where nobody is "
                         "reading the warning)")
    ap.add_argument("--force", action="store_true",
                    help="run even if today's tape is incomplete (catch-up "
                         "or a deliberate partial-day analysis)")
    ap.add_argument("--plan", action="store_true",
                    help="print the group plan and exit")
    a = ap.parse_args()
    if a.plan:
        for name, par, steps in GROUPS:
            mode = f"parallel x{min(a.jobs, len(steps))}" if (
                par and not a.serial) else "serial"
            print(f"[{name:8s}] {mode:12s} {', '.join(steps)}")
        return 0
    # v9.9.22: WARN, do not block. The first version of this exited 2 on a
    # partial tape, which was the wrong call: run_evening is overwhelmingly
    # EVIDENCE tooling over the 38 complete days already in the vault, and
    # one incomplete day at the end does not corrupt them — it only adds a
    # short sample the operator should know about. Refusing to start also
    # breaks the legitimate case of re-running mid-afternoon after a fix.
    # The information is what matters; the decision is the operator's.
    # `--strict` restores the hard block for unattended/scheduled runs,
    # where nobody is reading the warning.
    print(f"[cadence] {_cadence_note(getattr(a, 'force', False))}")
    ok, why = _day_is_complete(getattr(a, "force", False))
    if not ok:
        print("[preflight] WARNING: " + why)
        if getattr(a, "strict", False) and not getattr(a, "force", False):
            print("[preflight] --strict is set — refusing to run on a "
                  "partial tape. Drop --strict, or pass --force.")
            return 2
        print("[preflight] proceeding anyway. The 38 complete day(s) in the "
              "vault are unaffected; only today's short sample is in "
              "question. Use --days N on the individual tools if you want "
              "it excluded.")
    else:
        print(f"[preflight] {why}")
    global _MASTER
    _timings = _load_timings()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _MASTER = (LOG_DIR / "_evening.log").open("a", encoding="utf-8",
                                              errors="replace")
    emit(f"\n══════ EVENING RUN {time.strftime('%Y-%m-%d %H:%M:%S')} | "
         f"jobs={a.jobs} serial={a.serial} ══════")
    results: list[tuple[str, int, float]] = []
    fails, t_all = 0, time.time()
    for name, par, steps in GROUPS:
        gt = time.time()
        emit(f"\n═════ group {name} "
             f"({'parallel' if par and not a.serial else 'serial'}, "
             f"{len(steps)} step(s)) ═════")
        if par and not a.serial and len(steps) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            share = _inner_share(a.jobs)
            steps = _lpt_order(steps, _timings)
            _known = [f"{Path(x).stem} {_timings[Path(x).stem] / 60:.0f}m"
                      for x in steps if Path(x).stem in _timings]
            emit(f"      {a.jobs} tool(s) at a time, {share} day-worker(s) "
                 f"each (machine-wide RAM cap respected)")
            emit(f"      longest-first order: "
                 f"{', '.join(_known) if _known else 'no history yet'}")
            with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
                futs = {}
                for s_ in steps:
                    emit(f"----- {s_} started -----")
                    futs[ex.submit(_run_step, s_, True, share)] = s_
                for fu in as_completed(futs):
                    script, rc, dt = fu.result()
                    emit(f"===== {script} rc={rc} ({dt:.0f}s) "
                         f"[log: logs/evening/{Path(script).stem}.log] "
                         f"=====")
                    results.append((script, rc, dt))
                    fails += 1 if rc != 0 else 0
        else:
            for s in steps:
                emit(f"===== {s} (live) =====")
                script, rc, dt = _run_step(s, True, None)
                emit(f"===== {script} rc={rc} ({dt:.0f}s) "
                     f"[log: logs/evening/{Path(script).stem}.log] =====")
                results.append((script, rc, dt))
                fails += 1 if rc != 0 else 0
        emit(f"───── group {name} done in {time.time() - gt:.0f}s ─────")
    # ---- the night on one screen: what ran, how long, where its log is
    emit(f"\n═════ evening complete | {fails} failed step(s) | "
         f"{(time.time() - t_all) / 60:.1f} min ═════")
    emit(f"{'step':32s} {'rc':>3s} {'mins':>7s}   log")
    for script, rc, dt in sorted(results, key=lambda r: -r[2]):
        stem = Path(script).stem
        emit(f"{stem:32s} {rc:3d} {dt / 60:7.1f}   "
             f"logs/evening/{stem}.log"
             + ("   <<< FAILED" if rc else ""))
    _save_timings(results, _timings)
    emit(f"master log: logs/evening/_evening.log")
    if _MASTER is not None:
        _MASTER.close()
    return fails


if __name__ == "__main__":
    raise SystemExit(main())