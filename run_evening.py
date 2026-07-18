"""
APEX OMNI v9.7.1 — THE EVENING RITUAL, ONE COMMAND
==================================================
    python run_evening.py

Runs the nightly chain in order, each step logged and timed; a failing step
is reported and the chain CONTINUES to the next tool (evidence tools are
independent), except run_nightly which always runs last. Exit code is the
count of failed steps — 0 is a clean evening.

Ordering note (v9.7.1): nightly_calibration runs BEFORE toxicity_report.
toxicity_report reads the calibrated toxicity thresholds (tox_high/tox_block)
to decide which entries it counts as blocked vs allowed, so it must see
TONIGHT's freshly-measured thresholds — otherwise the lift it reports lags the
calibration by a day. calibration measures raw-series percentiles (independent
of the block/allow decision), so calibration→report is the correct dependency
direction. Both tools default their args, so the order is a pure dependency fix.
"""
import subprocess
import sys
import time

SEQ = ["tools/cascade_harness.py", "tools/shortvol_harness.py",
       "tools/butterfly_harness.py",
       "tools/rv_skill_report.py", "tools/rvnet_train.py",
       "tools/fly_intel_report.py",
       "tools/nightly_calibration.py",     # calibrate FIRST (writes thresholds)
       "tools/commodity_backfill.py",      # commodity Track-A (daily futures)
       "tools/commodity_calibration.py",   # commodity Track-B (intraday vault)
       "tools/toxicity_report.py",         # then measure lift under them
       "tools/fast_lane_report.py",        # fast-lane vs 45-min edge on the vault
       "tools/epistemic_health.py",
       "tools/gemma_analyst.py",           # offline Gemma brief over events+reports
       "nightly_commodity_forge.py",   # commodity meta (data-gated)
       "run_nightly.py"]

fails = 0
for script in SEQ:
    t0 = time.time()
    print(f"\n===== {script} =====", flush=True)
    rc = subprocess.call([sys.executable, script])
    print(f"===== {script} rc={rc} ({time.time() - t0:.0f}s) =====",
          flush=True)
    fails += 1 if rc != 0 else 0
sys.exit(fails)