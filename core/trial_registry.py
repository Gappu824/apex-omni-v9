"""
APEX OMNI v9.4 — GLOBAL TRIAL REGISTRY (Pillar 1 foundation)
============================================================
Every hypothesis this machine ever examines — nightly forge candidates,
harness primary runs, every sensitivity cell — is recorded here, append-only,
forever. Deflated performance statistics (Bailey–López de Prado 2014;
Harvey–Liu–Zhu 2016) are only honest when the trial count is the TRUE number
of looks taken at the data; per-tool counters undercount by construction.
This registry makes selection bias arithmetically impossible to hide from:
one file, one number, charged against every future claim.

Storage: state/trial_registry.jsonl — one JSON object per line:
  {ts, family, spec_id, kind, config_hash, meta...}
    family: "forge" | "cascade" | "shortvol" | "rv" | future families
    kind:   "candidate" (nightly model) | "primary" (prespecified spec run)
            | "sensitivity" (diagnostic grid cell) | "backfill"

Deflation policy (documented, prespecified): trials_for_deflation(family) =
count of ALL kinds within the family, because a sensitivity cell is a look
at the same exam family even when labeled diagnostic. total() is reported
alongside for the program-wide picture. Idempotent forge_history backfill
migrates the pre-registry era once.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import config

REGISTRY_PATH = config.STATE_DIR / "trial_registry.jsonl"


def _read() -> list[dict]:
    if not REGISTRY_PATH.exists():
        return []
    out = []
    for line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:                                 # noqa: BLE001
            continue                                      # torn line: skip
    return out


def register(family: str, spec_id: str, kind: str = "primary",
             **meta) -> None:
    """Append one trial. Never raises into a caller's run."""
    try:
        REGISTRY_PATH.parent.mkdir(exist_ok=True)
        row = {"ts": time.time(), "family": family, "spec_id": str(spec_id),
               "kind": kind, "config_hash": config.CONFIG_HASH}
        row.update({k: v for k, v in meta.items()
                    if isinstance(v, (str, int, float, bool)) or v is None})
        with REGISTRY_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:                                     # noqa: BLE001
        pass


def ensure_forge_backfill() -> int:
    """One-time migration: each pre-registry forge_history line becomes a
    'backfill' candidate trial, so the forge's deflation never resets.
    Idempotent (guarded by a marker row). Returns rows added."""
    rows = _read()
    if any(r.get("kind") == "backfill_marker" and r.get("family") == "forge"
           for r in rows):
        return 0
    hist = config.MODEL_DIR / "forge_history.jsonl"
    n = 0
    if hist.exists():
        for line in hist.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                h = json.loads(line)
            except Exception:                             # noqa: BLE001
                continue
            register("forge", h.get("ver", "pre-registry"), "backfill",
                     day=h.get("day"))
            n += 1
    register("forge", "-", "backfill_marker", migrated=n)
    return n


def counts(family: str | None = None) -> dict:
    """{kind: n} for one family, or {family: n_total} for all."""
    rows = _read()
    if family is not None:
        out: dict[str, int] = {}
        for r in rows:
            if r.get("family") == family and r.get("kind") != "backfill_marker":
                out[r.get("kind", "?")] = out.get(r.get("kind", "?"), 0) + 1
        return out
    out2: dict[str, int] = {}
    for r in rows:
        if r.get("kind") == "backfill_marker":
            continue
        out2[r.get("family", "?")] = out2.get(r.get("family", "?"), 0) + 1
    return out2


def trials_for_deflation(family: str) -> int:
    """The honest denominator: every look within the family, ≥1."""
    return max(sum(counts(family).values()), 1)


def total() -> int:
    return sum(counts().values())