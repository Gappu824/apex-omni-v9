"""
APEX OMNI v9.6 — WEEKLY DIGEST (deterministic, by constitution)
===============================================================
One markdown page assembled purely from the machine's own state files — no
generated prose, no narrative risk (PROGRAM.md negative space: deterministic
digests only). Sections: certificates & living health, graduation ladder,
trial-registry census, forge's latest verdict + 7-day gate attribution,
RV-forecast skill & RV-Net verdicts, calibration, latest stress exam,
supervisor uptime. Every section renders honestly when its file is absent:
"no data yet" is a finding.

    python tools/weekly_digest.py     →  logs/digest_YYYY-Wnn.md + console
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                            # noqa: E402
from core import trial_registry as TR                    # noqa: E402

config.setup_logging("weekly_digest")
import logging                                           # noqa: E402
log = logging.getLogger("digest")


def _j(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:                                     # noqa: BLE001
        return None


def _latest(glob: str) -> dict | None:
    files = sorted(config.LOG_DIR.glob(glob))
    return _j(files[-1]) if files else None


def _fmt(x, spec=",.2f", none="—"):
    return format(x, spec) if isinstance(x, (int, float)) else none


def build() -> str:
    L = [f"# APEX OMNI — weekly digest, {dt.date.today().isoformat()} "
         f"(ISO week {dt.date.today().isocalendar()[1]})",
         f"CONFIG_HASH `{config.CONFIG_HASH}` · LIVE_FIRE="
         f"{config.LIVE_FIRE} · capital ₹{config.TRADING_CAPITAL:,.0f}", ""]

    L.append("## Certificates & living health")
    for fam, p in (("cascade", config.CASCADE_CERT_PATH),
                   ("shortvol", config.SHORTVOL_CERT_PATH)):
        c = _j(p)
        if not c:
            L.append(f"- **{fam}**: no certificate yet")
            continue
        st = c.get("stress") or {}
        L.append(f"- **{fam}**: {'OK ✓' if c.get('ok') else 'not certified'}"
                 f" · events {c.get('n_events')} · Σ₹{_fmt(c.get('sum_pnl'))}"
                 f" · CI₉₀lo ₹{_fmt(c.get('ci_lo'))}"
                 f" · stress-lo ₹{_fmt(st.get('stationary_ci_lo'))}"
                 f" · trials {c.get('family_trials', '—')}"
                 + (f" · blocked: {'; '.join(c['blocked_by'][:2])}"
                    if c.get("blocked_by") else ""))
    ch = _j(config.STATE_DIR / "cert_health.json")
    if ch:
        for h in ch.get("families", []):
            L.append(f"  - living[{h['family']}]: {h.get('state')} "
                     f"(window {h.get('window_n')} fills, rolling-lo "
                     f"₹{_fmt(h.get('rolling_stat_ci_lo'))})")
    L.append("")

    L.append("## Graduation ladder")
    g = _j(config.STATE_DIR / "graduation.json")
    if g:
        for fam, e in g.get("families", {}).items():
            L.append(f"- **{fam}** → `{e.get('stage')}` · capital "
                     f"₹{_fmt(e.get('capital_rs'), ',.0f')}"
                     + (f" · blocked: {'; '.join(e['blocked_by'][:2])}"
                        if e.get("blocked_by") else ""))
    else:
        L.append("- not evaluated yet (`python -m core.graduation`)")
    L.append("")

    L.append("## Trial registry (program-wide looks at the data)")
    cs = TR.counts()
    L.append("- " + (" · ".join(f"{k}: {v}" for k, v in sorted(cs.items()))
                     if cs else "empty") + f" · **total {TR.total()}**")
    L.append("")

    L.append("## Forge (latest nightly)")
    f = _latest("forge_report_*.json")
    if f:
        L.append(f"- candidate `{f.get('candidate', '—')}` · promoted: "
                 f"{f.get('promoted', '—')} · WF psr {_fmt(f.get('psr'))} "
                 f"dsr {_fmt(f.get('dsr'))} · meta n "
                 f"{(f.get('meta') or {}).get('n', '—')}")
        ga_lines = (config.STATE_DIR / "gate_attribution.jsonl")
        if ga_lines.exists():
            agg: dict = {}
            rows = ga_lines.read_text().splitlines()[-7:]
            for line in rows:
                try:
                    for gname, a in json.loads(line)["attribution"].items():
                        d = agg.setdefault(gname, {"n": 0, "sum": 0.0})
                        d["n"] += a.get("n", 0)
                        d["sum"] += a.get("sum", 0.0)
                except Exception:                         # noqa: BLE001
                    continue
            for gname, a in sorted(agg.items(),
                                   key=lambda kv: kv[1]["sum"]):
                L.append(f"  - gate `{gname}` (last {len(rows)}d): "
                         f"n {a['n']} · refused ₹{a['sum']:+,.0f}")
    else:
        L.append("- no forge report found")
    L.append("")

    L.append("## RV forecasting")
    rv = _j(config.STATE_DIR / "rv_skill_certificate.json")
    if rv:
        for idx, v in (rv.get("per_index") or {}).items():
            L.append(f"- {idx}: {'SKILL ✓' if v.get('ok') else 'no skill yet'}"
                     f" · eval {v.get('eval_days')}d · QLIKE HAR "
                     f"{_fmt(v.get('qlike_har'), '.4f')} vs RW "
                     f"{_fmt(v.get('qlike_rw'), '.4f')}")
    else:
        L.append("- no skill certificate yet")
    nv = _j(config.STATE_DIR / "rvnet_verdict.json")
    if nv:
        for idx, v in (nv.get("per_index") or {}).items():
            L.append(f"  - rvnet[{idx}]: "
                     f"{'BEATS HAR ✓' if v.get('beats_har') else 'refused'}"
                     f" ({v.get('eval_days')}d)")
    L.append("")

    L.append("## Calibration")
    cal = (_j(config.STATE_DIR / "calibration.json") or {}).get("all") or {}
    L.append(f"- n {cal.get('n', 0)} · Brier {_fmt(cal.get('brier'), '.4f')}"
             f" = unc {_fmt(cal.get('uncertainty'), '.4f')} − res "
             f"{_fmt(cal.get('resolution'), '.4f')} + rel "
             f"{_fmt(cal.get('reliability'), '.4f')}"
             if cal.get("n") else "- no scored forecasts yet")
    L.append("")

    L.append("## Stress exam (latest)")
    sx = _latest("stress_report_*.json")
    if sx:
        for w in (sx.get("stitched_worlds") or [])[:4]:
            L.append(f"- world {w.get('morning')}→{w.get('afternoon')}: "
                     f"cascade ₹{_fmt(w.get('cascade_pnl'), '+,.0f')} "
                     f"({w.get('cascade_fills')} fills) · shortvol "
                     f"₹{_fmt(w.get('shortvol_pnl'), '+,.0f')}")
        for idx, r in (sx.get("cpcv_rv") or {}).items():
            L.append(f"  - CPCV(HAR)[{idx}]: {r}")
    else:
        L.append("- no stress report yet")
    L.append("")

    sup = _j(config.STATE_DIR / "supervisor_status.json")
    if sup:
        kids = ", ".join(f"{k['script'].split('_')[0]}:"
                         f"{'up' if k['alive'] else 'down'}"
                         f"({k['restarts_1h']}r)"
                         for k in sup.get("children", []))
        L.append(f"## Supervisor\n- last heartbeat "
                 f"{dt.datetime.fromtimestamp(sup['ts']):%H:%M} · {kids}")
    return "\n".join(L) + "\n"


def main():
    text = build()
    wk = dt.date.today().isocalendar()
    out = config.LOG_DIR / f"digest_{wk[0]}-W{wk[1]:02d}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    log.info("digest → %s", out)


if __name__ == "__main__":
    main()