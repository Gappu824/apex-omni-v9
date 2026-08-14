"""
LABEL CERTIFICATE — the training target changes only on earned evidence
=======================================================================
core/shadow_labels.py can compute a better training target: instead of
"did the trade as EXITED make money", ask "was there anything here to
collect". 2026-08-10 trade 3 is why that matters — NIFTY 24900CE labelled
a LOSS at -Rs126.99 because MAX_HOLD_THETA fired at 60 minutes, while
lock_5pct on the identical path came out at -Rs0.07. The incumbent label
charges exit noise to the ENTRY and teaches the model a lie about signals.

Knowing that is not permission to switch. The training target is the most
load-bearing choice in the whole stack: change it and every downstream
number — the meta probability, the EV gate, p*, the sizing — is measured
against a different question. So it moves the way cascade moves: behind a
CERTIFICATE, fail-closed, on the house pattern (core/cascade.py:224).

WHAT THE CERTIFICATE ASSERTS
----------------------------
Not "the label looks cleaner". A head-to-head MODEL comparison:

  1. A MONTH OF EVIDENCE. >= LABEL_CERT_MIN_SESSIONS distinct sessions and
     >= LABEL_CERT_MIN_TRADES labelled trades, every one of them at or
     above SHADOW_MIN_COVERAGE. A month is not a superstition: the
     regime mix rotates on roughly that scale (2026-08-07 was CHOP-dominant
     at 11 504s; 2026-08-10 was HIGH_VOL at 12 500s), and a target fitted
     inside one regime is a target fitted to one month of weather.
  2. COVERAGE HEALTH ACROSS THE WINDOW, not in aggregate. A month whose
     shadows averaged 60% because half were 100% and half were 20% is not
     a month of evidence. Every session must clear the floor.
  3. THE ECONOMIC CRITERION, NEVER THE LABEL ITSELF. This is the trap the
     certificate exists to close. A model trained on best_policy will
     always score better AGAINST best_policy — that comparison is
     circular and would certify anything. Both candidates are judged on
     the SAME realised economic outcome under the SAME exit stack, paired
     by day, through core/capability_ladder.py.
  4. LADDER + FDR + MDE + HOLDOUT, exactly as every other promotion here.
     The last LABEL_CERT_HOLDOUT of sessions is withheld and must agree in
     SIGN, so a target that only wins on the oldest data is refused.

FAIL-CLOSED, LIKE THE OTHERS
----------------------------
`active_label()` returns the incumbent ("realized") unless a certificate
is present, `ok`, matching CONFIG_HASH *and* the label-spec hash, and
inside its validity window. Any doubt — missing file, bad JSON, stale
timestamp, rotated hash — resolves to the incumbent. The certificate can
only ever be a reason to change; its absence is never a reason to.

AND IT EXPIRES. LABEL_CERT_VALID_DAYS is deliberately longer than the
7-day EDGE_CERT_VALID_DAYS — a training target is not a live detector and
should not thrash weekly — but it is finite. A target earned in July does
not authorise itself in October.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

import config
from core.shadow_labels import LABEL_SPECS, label_spec_hash

log = logging.getLogger("label_certificate")

INCUMBENT = "realized"


def _path() -> Path:
    return Path(getattr(config, "LABEL_CERT_PATH",
                        config.STATE_DIR / "label_certificate.json"))


# ------------------------------------------------------------- evaluation
def evaluate(per_trade: list, candidate: str = "best_policy",
             X=None, feat_names=None) -> dict:
    """Head-to-head: does `candidate` beat the incumbent as a TARGET?

    per_trade rows are LabelledTrade objects from core.shadow_labels.
    The economic criterion is `realized_pnl` — what the account actually
    booked. Both targets are scored on it. Never on their own label.
    """
    from core import capability_ladder as CL

    if candidate not in LABEL_SPECS:
        return {"ok": False, "reason": f"{candidate!r} is not pre-registered"}
    if candidate == INCUMBENT:
        return {"ok": False, "reason": "candidate is the incumbent"}

    min_cov = float(getattr(config, "SHADOW_MIN_COVERAGE", 0.60))
    rows = [t for t in per_trade if float(t.coverage) >= min_cov]
    excluded = len(per_trade) - len(rows)
    days = sorted({t.day for t in rows if t.day})

    min_sessions = int(getattr(config, "LABEL_CERT_MIN_SESSIONS", 20))
    min_trades = int(getattr(config, "LABEL_CERT_MIN_TRADES", 60))

    # ---- per-session coverage health: aggregate averages hide a bad half
    by_day: dict[str, list] = {}
    for t in rows:
        by_day.setdefault(t.day, []).append(t)
    thin_days = sorted(d for d, v in by_day.items()
                       if float(np.mean([x.coverage for x in v])) < min_cov)

    gates = {
        "enough_sessions": len(days) >= min_sessions,
        "enough_trades": len(rows) >= min_trades,
        "every_session_covered": not thin_days,
    }

    if not rows or len(days) < 3:
        return {"ok": False, "reason": "insufficient labelled evidence",
                "n_trades": len(rows), "n_days": len(days),
                "n_excluded_thin": excluded, "gates": gates,
                "candidate": candidate, "ts": time.time(),
                "config_hash": config.CONFIG_HASH}

    # ---- THE ECONOMIC CRITERION, AND WHY IT NEEDS A MODEL.
    #
    # The obvious comparison is tautological and I built it before I caught
    # it: "mean realised P&L among trades the target calls positive" is
    # rigged, because y_realized IS the criterion thresholded
    # (y_realized = realized_pnl > 0). Selecting on it and then scoring it
    # selects on the outcome itself, the incumbent wins by construction,
    # and the gate can never certify anything — it would have LOOKED strict
    # while being broken. On synthetic data with a genuine +Rs260 edge
    # injected, that form still returned -16.74/session.
    #
    # A label is only meaningful through the MODEL it trains. So the
    # comparison is: fit the same learner, on the same features, with the
    # same purged day-folds, changing ONLY the target; then score each
    # model's OUT-OF-FOLD gate decisions against realised P&L. Neither
    # model has seen the fold it is judged on, and neither target appears
    # in the criterion.
    #
    # That requires the forge's feature matrix. This module will not fake
    # it: with no X, the honest answer is "not evaluable", never a verdict.
    if X is None:
        return {"ok": False,
                "reason": "no feature matrix supplied — a label can only be "
                          "compared through the model it trains. Call "
                          "evaluate(rows, X=..., feat_names=...) from the "
                          "forge, where the training matrix exists.",
                "n_trades": len(rows), "n_days": len(days),
                "n_excluded_thin": excluded, "gates": gates,
                "candidate": candidate, "ts": time.time(),
                "config_hash": config.CONFIG_HASH}

    Xa = np.asarray(X, float)
    if Xa.shape[0] != len(rows):
        return {"ok": False,
                "reason": f"feature matrix has {Xa.shape[0]} row(s) for "
                          f"{len(rows)} labelled trade(s) — the join is "
                          f"wrong and a mismatched fit would be worse than "
                          f"no answer",
                "gates": gates, "candidate": candidate, "ts": time.time(),
                "config_hash": config.CONFIG_HASH}

    day_arr = np.array([t.day for t in rows])
    pnl = np.array([t.realized_pnl for t in rows], float)
    y_inc = np.array([t.y(INCUMBENT) for t in rows], float)
    y_can = np.array([t.y(candidate) for t in rows], float)

    oof_inc = _oof_scores(Xa, y_inc, day_arr)
    oof_can = _oof_scores(Xa, y_can, day_arr)
    if oof_inc is None or oof_can is None:
        return {"ok": False, "reason": "out-of-fold fit failed",
                "gates": gates, "candidate": candidate, "ts": time.time(),
                "config_hash": config.CONFIG_HASH}

    # Each model admits its top-quantile scores; both admit the SAME COUNT,
    # so the comparison isolates WHICH trades are chosen, not how many.
    per_day_delta: dict[str, float] = {}
    for d in sorted(set(day_arr)):
        m = day_arr == d
        if m.sum() < 2:
            continue
        k = max(int(round(m.sum() * float(getattr(
            config, "LABEL_CERT_ADMIT_FRAC", 0.5)))), 1)
        pick_i = np.argsort(-oof_inc[m])[:k]
        pick_c = np.argsort(-oof_can[m])[:k]
        per_day_delta[d] = float(np.mean(pnl[m][pick_c])) - \
                           float(np.mean(pnl[m][pick_i]))

    if len(per_day_delta) < 3:
        return {"ok": False, "reason": "too few sessions with both targets "
                                       "selecting anything",
                "n_trades": len(rows), "n_days": len(days),
                "gates": gates, "candidate": candidate,
                "ts": time.time(), "config_hash": config.CONFIG_HASH}

    ordered = sorted(per_day_delta)
    hold_frac = float(getattr(config, "LABEL_CERT_HOLDOUT", 0.30))
    n_hold = max(int(round(len(ordered) * hold_frac)), 1)
    train_d = ordered[:len(ordered) - n_hold]
    hold_d = ordered[len(ordered) - n_hold:]

    st = CL.paired_test({d: per_day_delta[d] for d in train_d})
    ho = float(np.mean([per_day_delta[d] for d in hold_d])) if hold_d \
        else float("nan")

    y = np.array([1.0 if t.realized_pnl > 0 else 0.0 for t in rows])
    dd = np.array([t.day for t in rows])
    cap = CL.assess(y, np.ones(len(rows)), dd)
    stage = str(getattr(config, "LABEL_CERT_LADDER_STAGE", "PROMOTE"))
    gates[f"ladder_comparative_{stage}"] = bool(
        cap.allows_comparative(stage))

    mde = float(st.get("mde", float("inf")))
    ci = st.get("ci90") or (None, None)
    checks = {
        "mean_positive": float(st["mean"]) > 0,
        "above_mde": float(st["mean"]) >
                     float(getattr(config, "LABEL_CERT_MDE_MULT", 1.0)) * mde,
        "ci_lower_clears": ci[0] is not None and float(ci[0]) > 0.0,
        "significant": float(st.get("p", 1.0)) <=
                       float(getattr(config, "LABEL_CERT_ALPHA", 0.05)),
        "holdout_agrees": bool(np.isfinite(ho) and ho > 0),
    }

    ok = all(gates.values()) and all(checks.values())
    return {"ok": ok, "candidate": candidate, "incumbent": INCUMBENT,
            "n_trades": len(rows), "n_days": len(days),
            "n_excluded_thin": excluded, "thin_days": thin_days,
            "mean_delta_rs_per_day": round(float(st["mean"]), 2),
            "ci90": ci, "p": round(float(st.get("p", 1.0)), 5),
            "mde": round(mde, 2), "holdout_mean": (round(ho, 2)
                                                   if np.isfinite(ho)
                                                   else None),
            "holdout_days": hold_d, "gates": gates, "checks": checks,
            "ladder": cap.as_dict() if hasattr(cap, "as_dict") else {},
            "label_spec_hash": label_spec_hash(candidate),
            "config_hash": config.CONFIG_HASH, "ts": time.time()}


def _oof_scores(X: np.ndarray, y: np.ndarray, days: np.ndarray):
    """Out-of-fold scores under PURGED DAY folds.

    Days are the exchangeable unit — trades inside one session share the
    tape, so a random split would leak. Whole days are held out, exactly as
    core/meta_gbm.py does, and the learner is a plain logistic fit: the
    question is which TARGET is better, and a heavier learner would let
    model capacity confound the answer.
    """
    try:
        uniq = sorted(set(days.tolist()))
        if len(uniq) < 4 or len(set(y.tolist())) < 2:
            return None
        n_fold = min(5, len(uniq))
        folds = [set(uniq[i::n_fold]) for i in range(n_fold)]
        out = np.full(len(y), np.nan)
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
        Xz = np.hstack([Xz, np.ones((len(Xz), 1))])
        for f in folds:
            te = np.array([d in f for d in days])
            tr = ~te
            if tr.sum() < 20 or len(set(y[tr].tolist())) < 2:
                continue
            w = np.zeros(Xz.shape[1])
            for _ in range(60):                     # IRLS, ridge-stabilised
                p = 1.0 / (1.0 + np.exp(-np.clip(Xz[tr] @ w, -30, 30)))
                g = Xz[tr].T @ (y[tr] - p) - 1e-3 * w
                H = Xz[tr].T @ (Xz[tr] * (p * (1 - p))[:, None]) \
                    + 1e-3 * np.eye(Xz.shape[1])
                try:
                    w = w + np.linalg.solve(H, g)
                except np.linalg.LinAlgError:
                    break
            out[te] = Xz[te] @ w
        return None if np.isnan(out).any() else out
    except Exception as e:                                 # noqa: BLE001
        log.warning("out-of-fold fit failed (%s)", e)
        return None


def report(v: dict, logger: logging.Logger | None = None) -> None:
    lg = logger or log
    need_s = int(getattr(config, "LABEL_CERT_MIN_SESSIONS", 20))
    lg.info("LABEL CERTIFICATE — candidate %r vs incumbent %r",
            v.get("candidate"), INCUMBENT)
    lg.info("  evidence: %d labelled trade(s) over %d session(s) "
            "(need %d sessions) | %d excluded below the coverage floor",
            v.get("n_trades", 0), v.get("n_days", 0), need_s,
            v.get("n_excluded_thin", 0))
    if v.get("thin_days"):
        lg.warning("  %d session(s) averaged below the coverage floor and "
                   "the window is therefore NOT a month of evidence: %s",
                   len(v["thin_days"]), v["thin_days"][:5])
    if "mean_delta_rs_per_day" in v:
        lg.info("  economic criterion (realised Rs/day, paired by session, "
                "NEVER the label itself): %+.2f | p=%.4f | MDE %.2f | "
                "holdout %s", v["mean_delta_rs_per_day"], v.get("p", 1.0),
                v.get("mde", float("nan")), v.get("holdout_mean"))
    for k, okk in {**v.get("gates", {}), **v.get("checks", {})}.items():
        lg.info("  %-34s %s", k, "PASS" if okk else "FAIL")
    if not v.get("ok"):
        lg.info("  NOT CERTIFIED — the incumbent target %r stands. %s",
                INCUMBENT, v.get("reason", ""))


# -------------------------------------------------------------- issuance
def issue(v: dict, dry_run: bool = False) -> bool:
    """Write the certificate if and only if the evidence earned it."""
    if not v.get("ok"):
        log.info("no certificate issued — %r remains the training target",
                 INCUMBENT)
        return False
    if dry_run:
        log.info("DRY RUN — %r would be certified", v["candidate"])
        return False
    body = {"ok": True, "label": v["candidate"], "incumbent": INCUMBENT,
            "ts": time.time(), "day": dt.date.today().isoformat(),
            "config_hash": config.CONFIG_HASH,
            "label_spec_hash": v["label_spec_hash"],
            "valid_days": float(getattr(config, "LABEL_CERT_VALID_DAYS", 45)),
            "evidence": {k: v.get(k) for k in
                         ("n_trades", "n_days", "mean_delta_rs_per_day",
                          "ci90", "p", "mde", "holdout_mean", "holdout_days",
                          "gates", "checks", "ladder")}}
    try:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.json")
        tmp.write_text(json.dumps(body, indent=1, default=float),
                       encoding="utf-8")
        os.replace(tmp, p)
        log.info("LABEL CERTIFIED: training target -> %r "
                 "(+Rs%.2f/session over %d session(s), p=%.4f). Valid %.0f "
                 "days; voided by any CONFIG_HASH change.",
                 v["candidate"], v["mean_delta_rs_per_day"], v["n_days"],
                 v.get("p", 1.0), body["valid_days"])
        return True
    except Exception as e:                                 # noqa: BLE001
        log.warning("certificate write failed (%s) — incumbent stands", e)
        return False


def load_certificate() -> dict | None:
    """The valid, matching certificate — or None. Fail-closed on any doubt,
    exactly like cascade.load_certificate and live_fire_armed."""
    try:
        c = json.loads(_path().read_text(encoding="utf-8"))
    except Exception:                                      # noqa: BLE001
        return None
    try:
        if not bool(c.get("ok")):
            return None
        if c.get("config_hash") != config.CONFIG_HASH:
            return None
        if c.get("label") not in LABEL_SPECS:
            return None
        if c.get("label_spec_hash") != label_spec_hash(str(c["label"])):
            return None                    # the policy family moved
        valid = float(c.get("valid_days",
                            getattr(config, "LABEL_CERT_VALID_DAYS", 45)))
        if (time.time() - float(c.get("ts", 0))) >= valid * 86400:
            return None
        return c
    except Exception:                                      # noqa: BLE001
        return None


def active_label() -> str:
    """The certified training target, or the incumbent.

    THIS is what the forge must call. config.META_LABEL_SPEC alone is not
    authority — an operator editing a string should not be able to change
    what the model is fitted to.
    """
    c = load_certificate()
    return str(c["label"]) if c else INCUMBENT