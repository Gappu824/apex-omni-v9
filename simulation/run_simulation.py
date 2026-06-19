"""
APEX OMNI v9 — RUN ALL SCENARIOS
================================
    python simulation/run_simulation.py
Pure numpy — no broker, no torch, no network. The decision stack under test
is the REAL one (RiskGovernor / ExecutionEngine / PositionManager /
TrapShield); only the exchange is synthetic. Writes SIM_REPORT.md.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                     # noqa: E402
from simulation.scenario_engine import run_scenario   # noqa: E402
from simulation import scenarios                  # noqa: E402

import logging
logging.basicConfig(level=logging.WARNING, format=config.LOG_FORMAT)


def main():
    t0 = time.time()
    results = []
    from simulation.scenario_miner import load_discovered
    specs = [build() for build in scenarios.ALL]
    mined = load_discovered()
    for sc in specs + mined:
        r = run_scenario(sc)
        results.append((sc, r))
        tag = "⛏" if sc.name.startswith("mined_") else " "
        flag = "✅ PASS" if r.ok else "❌ FAIL"
        print(f"{flag}{tag} {sc.name:<28} ₹{r.pnl:>+9.2f}  "
              f"trades={len(r.trades)} traps={r.trap_holds} "
              f"blocked={r.blocked + r.skipped}  | {r.note}")
    npass = sum(1 for _, r in results if r.ok)
    print("-" * 100)
    print(f"{npass}/{len(results)} scenarios PASS "
          f"({len(mined)} mined from live data)  "
          f"({time.time()-t0:.1f}s, capital ₹{config.TRADING_CAPITAL:,.0f}, "
          f"LIVE_FIRE={config.LIVE_FIRE})")

    rep = Path(__file__).resolve().parents[1] / "SIM_REPORT.md"
    with rep.open("w", encoding="utf-8") as f:
        f.write("# Apex Omni v9 — Intraday Scenario Simulation Report\n\n")
        f.write(f"Capital ₹{config.TRADING_CAPITAL:,.0f} · LIVE_FIRE="
                f"{config.LIVE_FIRE} (paper) · {npass}/{len(results)} PASS · "
                f"one synthetic 09:15–15:30 day per scenario, 1-second ticks, "
                f"Black-76 repriced premiums, Zerodha+statutory costs on "
                f"every fill.\n\n")
        f.write("| # | Scenario | Verdict | PnL (₹) | Trades | Trap holds | "
                "Blocked/Skipped | Exit reasons |\n|--|--|--|--|--|--|--|--|\n")
        for i, (sc, r) in enumerate(results, 1):
            f.write(f"| {i} | `{sc.name}` | "
                    f"{'**PASS**' if r.ok else '**FAIL**'} | {r.pnl:+.2f} | "
                    f"{len(r.trades)} | {r.trap_holds} | "
                    f"{r.blocked + r.skipped} | "
                    f"{', '.join(sorted(set(x.split(' ')[0] for x in r.exit_reasons))) or '—'} |\n")
        f.write("\n")
        for i, (sc, r) in enumerate(results, 1):
            f.write(f"## {i}. {sc.name} — "
                    f"{'PASS ✅' if r.ok else 'FAIL ❌'}\n\n{sc.desc}.\n\n"
                    f"**Outcome:** {r.note}\n\n")
            if r.halted:
                f.write(f"*Day halted:* {r.halt_reason}\n\n")
            if r.trades:
                f.write("| entry ₹ | exit ₹ | PnL ₹ | held (s) | exit reason "
                        "|\n|--|--|--|--|--|\n")
                for t in r.trades:
                    f.write(f"| {t['entry']:.2f} | {t['exit']:.2f} | "
                            f"{t['pnl']:+.2f} | {t['t_out']-t['t_in']:.0f} | "
                            f"{t['reason']} |\n")
                f.write("\n")
            if r.violations:
                f.write("**Invariant violations:** " +
                        "; ".join(r.violations) + "\n\n")
    print(f"report → {rep}")
    import json as _json
    (config.STATE_DIR / "sim_gate.json").write_text(_json.dumps(
        {"ts": time.time(), "pass": npass, "total": len(results),
         "mined": len(mined)}))
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
