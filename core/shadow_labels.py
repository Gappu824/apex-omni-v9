"""
SHADOW LABELS — the shadow teaches the ENTRY model, it never opens a trade
==========================================================================
THE ONE INVARIANT
-----------------
A position is opened by the gated system and by nothing else: cascade, the
meta probability, the conviction bar, the persistence filter, the risk
governor. The shadow book has no engine handle, no order path and no risk
call, and core/decision.py, core/meta_gate.py and core/cascade.py contain
zero references to it — verified by tools/shadow_audit.py, which fails if
that ever stops being true. The shadow is a MEASUREMENT that feeds TRAINING.
It is never an input to a live decision.

WHAT THE SHADOW ACTUALLY CONTRIBUTES — AND WHAT IT CANNOT
---------------------------------------------------------
It is tempting to read "learn from the shadow" as "more trades to train
on". It is not, and believing so would poison the model:

  THE SHADOW ONLY OBSERVES TRADES THE GATE ALREADY TOOK. Every shadow row
  is conditioned on entry. Fitting an entry model to them estimates
  P(outcome | entered), not P(outcome | signal). That is the reject-
  inference problem from credit scoring, and no amount of shadow data
  fixes it — the rejected signals are simply not in the sample. The
  entry-side counterfactual (core/entry_counterfactual.py) is the tool for
  that question; this module is not.

What the shadow DOES fix is the LABEL, and that is worth more than it
sounds. Today's label is "did the realised trade make money?" — which is
contaminated by the exit. 2026-08-10 trade 3 is the clean example: NIFTY
24900CE labelled a LOSS at -Rs126.99 because MAX_HOLD_THETA fired at 60
minutes, while lock_5pct on the identical path came out at -Rs0.07. The
entry was not the problem. The old label teaches the model that this
signal loses; it teaches a lie about entries.

So the shadow label asks a different question: WAS THERE ANYTHING HERE TO
COLLECT? A trade is positive if the best pre-registered exit policy would
have made money on it. Entry quality and exit quality stop being confounded.

FIVE GUARDS, EACH LOAD-BEARING
------------------------------
1. HORIZON-MATCHED, NOT ORACLE. The label is the best policy in
   config.SHADOW_POLICIES — exits that could actually be adopted, because
   core/exit_policy_store.py can promote them. Labelling on raw MFE-to-the-
   bell would train entries on potential the exit stack cannot capture:
   the model would learn to love signals that pay at 14:30 while the theta
   guillotine cuts at 12:00. That is a worse model, not a better one.
2. COVERAGE FLOOR. A trade whose feed died is a trade whose MFE is
   fiction. 2026-08-10's one closed shadow had 22% coverage. Anything
   under SHADOW_MIN_COVERAGE is EXCLUDED from labelling and counted.
3. PRE-REGISTERED. The label spec is fixed in config and carried in the
   artifact as a hash. You cannot fit three labels and keep the one that
   backtests best — that is the trial registry's whole purpose, applied to
   the target instead of the features.
4. LEAKAGE IS ALLOWED IN y AND FORBIDDEN IN X. The label uses post-entry
   information; that is what a label is. The FEATURE vector must remain
   strictly entry-time. This module returns labels only and never touches
   a feature.
5. A NEW LABEL IS A NEW MODEL. Switching the target does not get a free
   pass because it sounds better. It must beat the incumbent label
   head-to-head through core/capability_ladder.py — paired by day, FDR
   corrected, above MDE — exactly as any other promotion.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

import config

log = logging.getLogger("shadow_labels")

# Pre-registered label specs. `realized` is the incumbent — the target the
# meta model is trained on today. Anything else must earn its place.
LABEL_SPECS = ("realized", "best_policy", "any_policy_positive")


@dataclass
class LabelledTrade:
    shadow_id: str
    day: str
    index: str
    symbol: str
    token: int
    entry_ts: float
    coverage: float
    realized_pnl: float
    best_policy: str
    best_pnl: float
    n_positive_policies: int
    y_realized: int
    y_best_policy: int
    y_any_policy_positive: int
    left_on_table: float

    def y(self, spec: str) -> int:
        return {"realized": self.y_realized,
                "best_policy": self.y_best_policy,
                "any_policy_positive": self.y_any_policy_positive}[spec]

    def as_dict(self) -> dict:
        return asdict(self)


def label_spec_hash(spec: str) -> str:
    """Stamp the target so an artifact can never be silently relabelled.

    The policy family is part of the identity: 'best_policy' means nothing
    without knowing which policies were candidates.
    """
    body = json.dumps({"spec": spec,
                       "policies": list(getattr(config, "SHADOW_POLICIES",
                                                ())),
                       "min_coverage": float(getattr(
                           config, "SHADOW_MIN_COVERAGE", 0.60))},
                      sort_keys=True)
    return hashlib.sha1(body.encode()).hexdigest()[:10]


def read_shadow_ledger(path: Path | str | None = None
                       ) -> tuple[list[LabelledTrade], dict]:
    """Turn SHADOW_CLOSE rows into labelled trades.

    SHADOW_ABANDONED rows are counted, never labelled: an abandoned shadow
    is a trade whose path we stopped watching, and guessing at its
    potential is exactly the flat-line error in a new costume.
    """
    p = Path(path or getattr(config, "SHADOW_LEDGER_PATH",
                             config.LOG_DIR / "shadow_ledger_v9.csv"))
    stats = {"closed": 0, "abandoned": 0, "thin_coverage": 0,
             "unparsable": 0, "labelled": 0}
    out: list[LabelledTrade] = []
    if not p.exists():
        log.info("no shadow ledger at %s — nothing to label", p)
        return out, stats

    min_cov = float(getattr(config, "SHADOW_MIN_COVERAGE", 0.60))
    import datetime as dt
    try:
        with p.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                ev = str(r.get("event") or "")
                if ev == "SHADOW_ABANDONED":
                    stats["abandoned"] += 1
                    continue
                if ev != "SHADOW_CLOSE":
                    continue
                stats["closed"] += 1
                try:
                    pol = json.loads(r.get("policies") or "{}")
                    cov = float(r.get("coverage") or 0.0)
                    ets = float(r.get("entry_ts") or 0.0)
                except (ValueError, TypeError):
                    stats["unparsable"] += 1
                    continue
                if not pol or "as_traded" not in pol:
                    stats["unparsable"] += 1
                    continue
                if cov < min_cov:
                    stats["thin_coverage"] += 1
                    continue

                real = float(pol["as_traded"].get("pnl") or 0.0)
                # Candidates exclude as_traded (the baseline) and any exit
                # that fired on a dead feed — a stale exit price is not an
                # outcome, it is the last thing we happened to see.
                cands = {k: float(v.get("pnl") or 0.0)
                         for k, v in pol.items()
                         if k != "as_traded" and not v.get("stale")}
                if not cands:
                    stats["unparsable"] += 1
                    continue
                best_k = max(cands, key=lambda k: cands[k])
                best_v = cands[best_k]
                n_pos = sum(1 for v in cands.values() if v > 0)

                out.append(LabelledTrade(
                    shadow_id=str(r.get("shadow_id") or ""),
                    day=(dt.datetime.fromtimestamp(ets).date().isoformat()
                         if ets else ""),
                    index=str(r.get("index") or ""),
                    symbol=str(r.get("symbol") or ""),
                    token=int(float(r.get("token") or 0)),
                    entry_ts=ets, coverage=cov, realized_pnl=real,
                    best_policy=best_k, best_pnl=best_v,
                    n_positive_policies=n_pos,
                    y_realized=int(real > 0),
                    y_best_policy=int(best_v > 0),
                    y_any_policy_positive=int(n_pos > 0),
                    left_on_table=best_v - real))
                stats["labelled"] += 1
    except OSError as e:                                   # noqa: BLE001
        log.error("could not read shadow ledger %s (%s)", p, e)
    return out, stats


def report(trades: list[LabelledTrade], stats: dict,
           logger: logging.Logger | None = None) -> dict:
    """State plainly what the shadow does and does not license."""
    lg = logger or log
    lg.info("shadow ledger → %d closed, %d labelled, %d thin coverage "
            "(<%.0f%%), %d abandoned, %d unparsable",
            stats["closed"], stats["labelled"], stats["thin_coverage"],
            100 * float(getattr(config, "SHADOW_MIN_COVERAGE", 0.6)),
            stats["abandoned"], stats["unparsable"])
    if not trades:
        lg.info("nothing labelled — the incumbent target stands")
        return {"n": 0}

    y_r = np.array([t.y_realized for t in trades], float)
    y_b = np.array([t.y_best_policy for t in trades], float)
    flip = int(np.sum((y_r == 0) & (y_b == 1)))
    lot = float(np.sum([t.left_on_table for t in trades]))

    lg.info("label agreement: realized %d/%d positive | best_policy %d/%d "
            "| %d trade(s) FLIP from loss to win once the exit is held "
            "constant", int(y_r.sum()), len(trades), int(y_b.sum()),
            len(trades), flip)
    lg.info("Rs %,.0f total left on the table across labelled trades — that "
            "is the exit-noise the incumbent label was charging to the "
            "ENTRY.", lot) if False else lg.info(
        "Rs %s total left on the table across labelled trades — that is "
        "exit noise the incumbent label was charging to the ENTRY.",
        f"{lot:,.0f}")
    lg.warning("SELECTION: every row here is a trade the gate ALREADY TOOK. "
               "These labels de-noise the target; they do NOT enlarge the "
               "sample and they say nothing about signals that were "
               "blocked. Fitting an entry model to them estimates "
               "P(outcome | entered). For P(outcome | signal), use "
               "tools/entry_bar_study.py.")
    return {"n": len(trades), "flips": flip, "left_on_table": lot,
            "pos_realized": int(y_r.sum()), "pos_best": int(y_b.sum()),
            "spec_hash": {s: label_spec_hash(s) for s in LABEL_SPECS}}


def label_map(trades: list[LabelledTrade], spec: str | None = None
              ) -> dict[str, int]:
    """shadow_id → y under the ACTIVE, pre-registered spec.

    The forge joins this onto its own training rows by shadow_id (the
    execution ledger carries the column). A trade with no shadow — feed
    died, snapshot abandoned, or a session that ran before the book was
    armed — is simply absent, and the caller must keep the incumbent label
    for it rather than invent one.
    """
    spec = str(spec or getattr(config, "META_LABEL_SPEC", "realized"))
    if spec not in LABEL_SPECS:
        raise ValueError(
            f"META_LABEL_SPEC={spec!r} is not pre-registered. Add it to "
            f"shadow_labels.LABEL_SPECS deliberately — picking a target "
            f"after seeing its result is how a backtest gets chosen "
            f"instead of a model.")
    return {t.shadow_id: t.y(spec) for t in trades if t.shadow_id}


def active_spec() -> tuple[str, str]:
    spec = str(getattr(config, "META_LABEL_SPEC", "realized"))
    return spec, label_spec_hash(spec)