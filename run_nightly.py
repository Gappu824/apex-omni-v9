"""
APEX OMNI v9 — NIGHTLY ORCHESTRATOR
===================================
One command after the close:

    python run_nightly.py

Sequence (each step logged to logs/nightly_<date>.log):
  1. JUDGE   — analyzer verdict over today's ledger; refreshes
               calibration_table.json (tomorrow's Kelly mapping).
  2. MINE    — scan today's harvested ticks for stop-runs / air pockets /
               feed gaps; append them to the regression suite.
  3. REGRESS — run the full scenario suite (16 core + everything mined).
               Writes state/sim_gate.json.
  4. FORGE   — retrain on real ticks (point-in-time replay, cost-aware
               reward); promote ONLY if the held-out day beats the incumbent
               AND the regression gate is green. Needs torch/SB3 and ≥2
               harvested days — until then it tells you so and exits cleanly.
"""
from __future__ import annotations
import logging
import traceback

import config

config.setup_logging("nightly")
log = logging.getLogger("nightly")


def step(n, title):
    log.info("—" * 24 + f" {n}. {title} " + "—" * 24)


def main():
    step(1, "JUDGE (analyzer verdict → calibration table)")
    try:
        import live_trade_analyzer_v9 as judge
        judge.verdict()
    except Exception:                                      # noqa: BLE001
        log.error("verdict failed:\n%s", traceback.format_exc())

    step(2, "EDGE AUDIT (bootstrap proof-of-edge → live-arming certificate)")
    try:
        from core.edge_audit import audit_and_certify, progress_line
        res = audit_and_certify()
        line = progress_line(res)
        log.info(line)                       # prominent in the nightly log
        (config.STATE_DIR / "edge_progress.txt").write_text(
            line + "\n", encoding="utf-8")     # one-glance file (UTF-8: emoji)
    except Exception:                                      # noqa: BLE001
        log.error("edge audit failed:\n%s", traceback.format_exc())

    step(3, "MINE (today's ticks → new regression scenarios)")
    try:
        from simulation.scenario_miner import mine
        log.info("%d new scenario(s) mined", mine())
    except Exception:                                      # noqa: BLE001
        log.error("miner failed:\n%s", traceback.format_exc())

    step(4, "REGRESS (core 16 + mined — writes the promotion gate)")
    suite_green = False
    try:
        from simulation.run_simulation import main as sim_main
        suite_green = (sim_main() == 0)
    except Exception:                                      # noqa: BLE001
        log.error("regression suite crashed:\n%s", traceback.format_exc())
    log.info("regression gate: %s", "GREEN" if suite_green else "RED")

    step(5, "FORGE (train on real ticks; gated promotion)")
    try:
        import nightly_forge_v9 as forge
        forge.main()
    except SystemExit as e:
        log.info("forge: %s", e)
    except Exception:                                      # noqa: BLE001
        log.error("forge failed:\n%s", traceback.format_exc())

    log.info("nightly complete. Sleep well; the verdict file doesn't lie.")


if __name__ == "__main__":
    main()