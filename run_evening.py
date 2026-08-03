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
import os
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
                          "tools/ab_ablation.py"]),
    # v9.9.3: the analyst runs LAST — its brief can now digest the forge
    # verdicts and the ₹ exam, and its model no longer squats in RAM ahead
    # of the heaviest stages.
    ("analyst",   False, ["tools/gemma_analyst.py"]),      # GPU — solo
]

LOG_DIR = Path("logs/evening")
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
    lf = LOG_DIR / (Path(script).stem + ".log")
    env = dict(os.environ)
    if inner:                       # parallel group: hand out a share
        env["APEX_INNER_WORKERS"] = inner
    else:                           # solo stage: full machine, as before
        env.pop("APEX_INNER_WORKERS", None)
    tag = Path(script).stem if inner else ""
    with lf.open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen([sys.executable, script], env=env,
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
    ap.add_argument("--plan", action="store_true",
                    help="print the group plan and exit")
    a = ap.parse_args()
    if a.plan:
        for name, par, steps in GROUPS:
            mode = f"parallel x{min(a.jobs, len(steps))}" if (
                par and not a.serial) else "serial"
            print(f"[{name:8s}] {mode:12s} {', '.join(steps)}")
        return 0
    global _MASTER
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
            emit(f"      {a.jobs} tool(s) at a time, {share} day-worker(s) "
                 f"each (machine-wide RAM cap respected)")
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
    emit(f"master log: logs/evening/_evening.log")
    if _MASTER is not None:
        _MASTER.close()
    return fails


if __name__ == "__main__":
    raise SystemExit(main())