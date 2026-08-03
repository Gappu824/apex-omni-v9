"""
APEX OMNI v9.7.1 — GEMMA NIGHTLY ANALYST (the offline reasoning layer)
======================================================================
Integrates Gemma 4 (E4B, via Ollama, Q4_K_M) as the system's NIGHTLY ANALYST —
a local reasoning layer on the EVIDENCE side of the system. It runs once per
night inside run_evening, reads the deterministic artifacts the rest of the
system produces, and writes a structured digest + a human-readable brief.

Where Gemma adds real value (and where it must NOT go)
------------------------------------------------------
USED HERE (offline, nightly, advisory):
  • Reason over the commodity EVENT CALENDAR (core/event_engine) — which
    scheduled releases land tomorrow, their severity, which commodities they
    hit — and turn it into an operator-facing risk brief.
  • Read the night's own reports (calibration.json, toxicity_report,
    fast_lane_report, epistemic_health) and synthesize a plain-language "state
    of the system" the operator can act on.
  • DRAFT (never auto-apply) candidate EVENT_OVERRIDES for dated one-offs
    (OPEC/FOMC/CPI/NFP) it can infer, for the operator to review.

NEVER (by design):
  • The per-second decision/entry path. An 8GB-class LLM cannot make a sub-
    second options decision, and putting it there would wreck latency and
    determinism. Gemma is the analyst, not the trader. The trading engine's
    decisions remain the deterministic, tested policy path.

Hard engineering constraints (RTX 4060, 8GB)
--------------------------------------------
  • Model: gemma-4-e4b (Q4_K_M) — the 8GB "daily driver" tier. Configurable.
  • Context kept small (num_ctx ≈ 4096) so the KV cache fits alongside the
    weights and the operator's other GPU work.
  • FAIL-SAFE: if Ollama is not running, the model isn't pulled, or the call
    errors/times out, this module logs and returns None. The nightly ritual
    and every trading decision proceed EXACTLY as if this layer didn't exist.
    Gemma is strictly additive; it can never block or break the system.
  • The model runs, emits its digest, and the process exits — Ollama unloads
    the model, freeing VRAM. No resident GPU footprint during trading hours.

  python tools/gemma_analyst.py [--dry]   # --dry prints the prompt, no Ollama
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                             # noqa: E402
from core.event_engine import CommodityEventEngine        # noqa: E402
from zoneinfo import ZoneInfo                              # noqa: E402

config.setup_logging("gemma_analyst")
import logging                                            # noqa: E402
log = logging.getLogger("gemma_analyst")

_IST = ZoneInfo("Asia/Kolkata")


# --------------------------------------------------------------------------
# Ollama client — minimal, fail-safe, stdlib only (no new dependency)
# --------------------------------------------------------------------------
def _resolve_model(host: str, configured: str) -> str | None:
    """Ask Ollama what is ACTUALLY installed (/api/tags) instead of trusting a
    guessed tag. Returns: the configured model if installed; else any installed
    gemma* model (largest first) with a clear log; else None with the exact
    remedy. This is what fixes the 404 ('model not found') without tag-guessing."""
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags",
                                    timeout=10) as r:
            tags = [m.get("name", "") for m in
                    json.loads(r.read().decode()).get("models", [])]
    except Exception as e:                                     # noqa: BLE001
        log.warning("Ollama unreachable for model discovery (%s)", e)
        return None
    if configured in tags or f"{configured}:latest" in tags:
        return configured
    # v9.7.1 (RE-APPLIED 2026-07-23): reverse-alpha picked 'gemma:latest'
    # (Gemma-1 2B) over the installed 'gemma4:e2b' because ':' > '4' in ASCII —
    # exactly what happened tonight. Prefer the installed tag sharing the
    # LONGEST prefix with the configured name, so any gemma4* beats gemma:*.
    # (This fix was lost once when a later edit was applied to a freshly-reset
    # file; it is restored here with the digest change intact.)
    def _pref(t):
        n = 0
        for a, b in zip(t.lower(), configured.lower()):
            if a != b:
                break
            n += 1
        return (n, t)
    gemmas = sorted([t for t in tags if "gemma" in t.lower()],
                    key=_pref, reverse=True)
    if gemmas:
        log.warning("configured model '%s' is NOT installed; using installed "
                    "'%s' instead (set GEMMA_MODEL to pin it). Installed: %s",
                    configured, gemmas[0], ", ".join(tags) or "(none)")
        return gemmas[0]
    log.warning("no gemma model installed in Ollama (installed: %s). Remedy: "
                "run `ollama list`, then `ollama pull <a gemma tag>` and set "
                "GEMMA_MODEL to that exact tag. Analyst SKIPPED; system "
                "unaffected.", ", ".join(tags) or "(none)")
    return None


def _ollama_generate(prompt: str, *, model: str, host: str, num_ctx: int,
                     timeout: float) -> str | None:
    """Call Ollama's /api/generate. Returns text, or None on ANY failure —
    the caller treats None as 'analyst unavailable, proceed normally'."""
    url = f"{host.rstrip('/')}/api/generate"
    body = json.dumps({
        "model": model, "prompt": prompt, "stream": False,
        # v9.9.3: unload immediately after the brief. Ollama's default
        # keep_alive pins the model in RAM for minutes; on 2026-08-01 that
        # residue left 3.7 GB free and the RAM-aware pool sizer collapsed
        # the commodity forge to ONE worker.
        "keep_alive": 0,
        "options": {"num_ctx": num_ctx, "temperature": 0.2},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response")
    except (urllib.error.URLError, TimeoutError, ConnectionError,
            json.JSONDecodeError, OSError) as e:
        log.warning("Ollama unavailable (%s) — nightly analyst SKIPPED; the "
                    "system runs normally without it", e)
        return None


# --------------------------------------------------------------------------
# Gather the deterministic evidence the analyst reasons over
# --------------------------------------------------------------------------
def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:                                          # noqa: BLE001
        return None


def _latest(prefix: str) -> dict | None:
    """Most recent logs/<prefix>_<date>.json, if any."""
    try:
        files = sorted(config.LOG_DIR.glob(f"{prefix}_*.json"))
        return _read_json(files[-1]) if files else None
    except Exception:                                          # noqa: BLE001
        return None


# v9.9.4 ------------------------------------------------------- research
# The MENU. Gemma may only propose from this fixed list of transforms.
# An open-ended "suggest a feature" would launder unbounded multiple
# testing through a language model: unfalsifiable, uncountable, and
# uncorrectable. A closed menu keeps the hypothesis space finite, so
# every proposal can be pre-registered and FDR-corrected like any other.
HYPOTHESIS_MENU = [
    "horizon:test_shorter_holds",
    "horizon:test_longer_holds",
    "feature:interact_regime_with_flow",
    "feature:interact_iv_rank_with_gex",
    "feature:add_peer_dispersion_lag",
    "feature:add_time_since_regime_flip",
    "label:test_asymmetric_barriers",
    "label:test_vol_scaled_barriers",
    "data:collect_more_days_no_change",
]


def _ladder_now() -> dict | None:
    """Tonight's resolving power, computed from the real sample cache —
    the number that decides whether any research is even meaningful."""
    try:
        import sqlite3
        import numpy as np
        from core import capability_ladder as CL
        from nightly_forge_v9 import trading_days, _gen_meta_samples_cached
        con = sqlite3.connect(str(config.DB_PATH))
        try:
            Y, W, SD = [], [], []
            for d in trading_days(con):
                try:
                    x_, y_, w_, _r, _ret, _e = _gen_meta_samples_cached(con, d)
                except Exception:                          # noqa: BLE001
                    continue
                if not x_:
                    continue
                Y += list(y_); W += list(w_); SD += [d] * len(x_)
        finally:
            con.close()
        if not Y:
            return None
        cap = CL.assess(np.asarray(Y, float), np.asarray(W, float),
                        np.asarray(SD))
        return cap.as_dict()
    except Exception as e:                                 # noqa: BLE001
        log.debug("ladder snapshot unavailable (%s)", e)
        return None


def preregister_hypotheses(text: str) -> list[str]:
    """Extract menu items Gemma named and PRE-REGISTER them as trials.
    Registering before testing is the whole point: a hypothesis that
    costs nothing to propose would otherwise cost nothing to be wrong
    about. Here every proposal permanently raises the bar that any
    future claim must clear."""
    picked = [h for h in HYPOTHESIS_MENU if h in (text or "")]
    if not picked:
        return []
    try:
        from core.trial_registry import register
        for h in picked:
            register(family="gemma_hypothesis", spec_id=h,
                     kind="pre_registered_llm")
        q = config.STATE_DIR / "hypothesis_queue.jsonl"
        q.parent.mkdir(parents=True, exist_ok=True)
        with q.open("a", encoding="utf-8") as f:
            for h in picked:
                f.write(json.dumps({"ts": time.time(), "hypothesis": h,
                                    "config_hash": config.CONFIG_HASH,
                                    "source": "gemma"}) + "\n")
        log.info("pre-registered %d Gemma hypothesis/es: %s",
                 len(picked), ", ".join(picked))
    except Exception as e:                                 # noqa: BLE001
        log.warning("hypothesis pre-registration failed (%s)", e)
    return picked


def gather_context() -> dict:
    """All the evidence, as plain data. Every field degrades to None/[] so the
    brief is honest about what's actually present."""
    now = dt.datetime.now(_IST)
    eng = CommodityEventEngine()
    # tomorrow's events per harvested commodity
    events = []
    for c in getattr(config, "HARVEST_COMMODITIES", []):
        nxt = eng.next_event(now, c)
        if nxt:
            name, ts = nxt
            events.append({"commodity": c, "event": name,
                           "when_ist": ts.isoformat(timespec="minutes"),
                           "hours_away": round((ts - now).total_seconds() / 3600, 1)})
    return {
        "as_of_ist": now.isoformat(timespec="minutes"),
        "config_hash": config.CONFIG_HASH,
        "upcoming_events": events,
        "calibration": _read_json(config.LOG_DIR / "calibration.json"),
        "toxicity_report": _latest("toxicity_report"),
        # v9.9.4 research evidence — the ladder, the horizon sweep and the
        # feature screen. Gemma reads these to WRITE about them and to
        # propose next hypotheses; it never selects a horizon or a feature.
        # Selection is BH-FDR + repetition, in the tools themselves.
        "capability_ladder": _ladder_now(),
        "horizon_sweep": _latest("horizon_sweep"),
        "feature_discovery": _latest("feature_discovery"),
        "fast_lane_report": _latest("fast_lane_report"),
        "commodities_harvested": list(getattr(config, "HARVEST_COMMODITIES", [])),
        "commodity_trading_enabled": bool(getattr(config, "COMMODITY_TRADABLE", [])),
    }


# --------------------------------------------------------------------------
# Prompt — tightly scoped, asks for STRUCTURED output
# --------------------------------------------------------------------------
_SYSTEM = """You are the nightly risk analyst for a paper-mode options trading \
system (Indian markets: NIFTY/SENSEX equity indices, plus MCX commodities in \
DATA-COLLECTION only — no commodity trading yet). You are given tonight's \
deterministic reports and the scheduled-event calendar. Produce a concise, \
factual brief for the human operator. Do not invent numbers not in the data. \
Do not give trade signals. Flag scheduled-event risk for tomorrow, summarize \
what the reports say, and note anything that needs the operator's attention. \
Be terse and specific. If a report is missing, say so plainly.

You are ALSO the research scribe. The system measures its own statistical \
resolving power ("capability_ladder"): mde_auc is the smallest ranking \
ability the current vault can tell apart from chance, stage says what \
research is currently meaningful, days_to_promote_power is how many more \
trading days are needed. Rules you must follow: a model AUC near 0.50 when \
mde_auc is far above the promotion bar means UNDERPOWERED, not "no edge" — \
say so. Never call a result significant unless the report marks it \
survives_fdr. Never recommend adopting a horizon or a feature; selection is \
statistical and automatic. You may PROPOSE next hypotheses, but ONLY by \
quoting exact strings from the menu you are given."""


def build_prompt(ctx: dict) -> str:
    return (f"{_SYSTEM}\n\nTONIGHT'S DATA (JSON):\n"
            f"{json.dumps(ctx, indent=2, default=str)}\n\n"
            "Write the brief in these sections, each 1-4 short lines:\n"
            "1. SCHEDULED-EVENT RISK TOMORROW (which commodities, when in IST, "
            "severity; if none in the next 24h, say so)\n"
            "2. CALIBRATION STATE (what the vault has calibrated; if thin/absent, "
            "say the system is on config defaults)\n"
            "3. FILTER/EDGE REPORTS (toxicity and fast-lane verdicts if present)\n"
            "4. OPERATOR ATTENTION (anything notable; else 'nothing pressing')\n"
            "5. RESEARCH STATE (the ladder: stage, mde_auc vs the promotion "
            "bar, days_to_promote_power; what the horizon sweep and feature "
            "screen concluded, and whether anything survived FDR)\n"
            "6. NEXT HYPOTHESES (1-3 items quoted EXACTLY from this menu, "
            "one per line, each with a one-line reason. If the ladder stage "
            "is BLIND or nothing is testable yet, quote only "
            "'data:collect_more_days_no_change'.)\n"
            f"MENU: {HYPOTHESIS_MENU}\n"
            "Keep it under 320 words. Facts only.")


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true",
                    help="print the prompt and exit (no Ollama call)")
    args = ap.parse_args()

    if not bool(getattr(config, "GEMMA_ANALYST_ENABLED", True)):
        log.info("Gemma analyst disabled by config — skipping")
        return

    ctx = gather_context()
    prompt = build_prompt(ctx)

    if args.dry:
        print(prompt)
        return

    model = getattr(config, "GEMMA_MODEL", "gemma-4-e4b")
    host = getattr(config, "OLLAMA_HOST", "http://127.0.0.1:11434")
    num_ctx = int(getattr(config, "GEMMA_NUM_CTX", 4096))
    timeout = float(getattr(config, "GEMMA_TIMEOUT_S", 120))

    log.info("nightly analyst | model=%s ctx=%d | %d upcoming event(s)",
             model, num_ctx, len(ctx["upcoming_events"]))
    resolved = _resolve_model(host, model)
    brief = None
    if resolved is not None:
        brief = _ollama_generate(prompt, model=resolved, host=host,
                                 num_ctx=num_ctx, timeout=timeout)

    # AUDIT (2026-07-22 logs): the digest recorded the CONFIGURED model, so
    # gemma_digest.json + the brief header both claimed "gemma4:e4b" on a night
    # the resolver had substituted the installed "gemma4:e2b". Record what
    # ACTUALLY wrote the brief (and keep the requested name for provenance).
    out = {"generated_ist": ctx["as_of_ist"],
           "model": resolved or model, "model_requested": model,
           "context": ctx, "brief": brief,
           "available": brief is not None}
    # structured digest (always written, even if the model was unavailable)
    # encoding pinned: these files carry ₹ / — / ✓ and Windows would otherwise
    # write them in the legacy codepage (the ? seen in nightly_brief .md).
    (config.LOG_DIR / "gemma_digest.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")

    if brief is None:
        log.info("analyst unavailable — digest written with brief=null; "
                 "the system is unaffected")
        return
    # human-readable brief
    brief_path = config.LOG_DIR / f"nightly_brief_{dt.date.today()}.md"
    brief_path.write_text(
        f"# Apex Omni nightly brief — {ctx['as_of_ist']}\n\n"
        f"*(Gemma {resolved or model}, local, advisory — not a trade "
        f"signal)*\n\n{brief}\n", encoding="utf-8")
    preregister_hypotheses(brief or "")
    log.info("nightly brief → %s", brief_path)
    print(brief)


if __name__ == "__main__":
    main()