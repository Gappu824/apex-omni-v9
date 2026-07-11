"""
APEX OMNI v9.6 — THE WEEKLY EXAM, ONE COMMAND
=============================================
    python run_weekly.py

Stress worlds + CPCV, every spec in specs/ through the factory, the
execution report (live-fill certificates when live rows exist), the
graduation ladder, and finally the deterministic weekly digest. Independent
steps; failures are counted, not fatal. Run it on a weekend.
"""
import subprocess
import sys
import time
from pathlib import Path

steps = [["tools/stress_exam.py"]]
steps += [["tools/spec_harness.py", "--spec", str(p)]
          for p in sorted(Path("specs").glob("*.json"))]
steps += [["tools/execution_report.py"], ["-m", "core.graduation"],
          ["tools/weekly_digest.py"]]

fails = 0
for step in steps:
    t0 = time.time()
    print(f"\n===== {' '.join(step)} =====", flush=True)
    rc = subprocess.call([sys.executable] + step)
    print(f"===== rc={rc} ({time.time() - t0:.0f}s) =====", flush=True)
    fails += 1 if rc != 0 else 0
sys.exit(fails)