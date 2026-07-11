"""
APEX OMNI v9.6 — THE EVENING RITUAL, ONE COMMAND
================================================
    python run_evening.py

Runs the nightly chain in order, each step logged and timed; a failing step
is reported and the chain CONTINUES to the next tool (evidence tools are
independent), except run_nightly which always runs last. Exit code is the
count of failed steps — 0 is a clean evening.
"""
import subprocess
import sys
import time

SEQ = ["tools/cascade_harness.py", "tools/shortvol_harness.py",
       "tools/rv_skill_report.py", "tools/rvnet_train.py",
       "tools/epistemic_health.py", "run_nightly.py"]

fails = 0
for script in SEQ:
    t0 = time.time()
    print(f"\n===== {script} =====", flush=True)
    rc = subprocess.call([sys.executable, script])
    print(f"===== {script} rc={rc} ({time.time() - t0:.0f}s) =====",
          flush=True)
    fails += 1 if rc != 0 else 0
sys.exit(fails)