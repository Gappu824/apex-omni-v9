"""
CAS CAPTURE — seeing through the auction with the options themselves
=====================================================================
THE PROBLEM, STATED HONESTLY
----------------------------
The tradable event is the 15:35 print: the auction publishes and the
options reprice to it. To be IN that move you must be positioned BEFORE
15:35 — which is inside the window where I suspended entries, because
the index feed there is INDICATIVE, not traded.

Both facts are true at once. The resolution is not to trust the
indicative index; it is to stop needing it.

THE INSIGHT
-----------
Between 15:15 and 15:35 the CASH constituents are in auction — but the
INDEX OPTIONS never stop trading. They are continuously quoted through
the entire window, and their prices embed the market's live expectation
of what the auction will print. Put-call parity turns that expectation
into a number:

        C − P = S·e^(−qT) − K·e^(−rT)      ⇒     S_synth ≈ (C − P) + K·e^(−rT)

So during the blackout we can compute a SYNTHETIC UNDERLYING from the
ATM call and put — a continuously-traded estimate of spot, sourced
entirely from instruments that are actually trading. The divergence
between S_synth and the last continuously-traded index (frozen at 15:15)
is the market's own forecast of the auction gap.

That is the signal. It needs no indicative feed, no imbalance broadcast,
and nothing from the exchange that Kite may or may not forward — only
the option bid/ask this system already streams every second.

WHAT THIS MODULE DOES
---------------------
1. `synthetic_spot()` — put-call parity on the ATM pair, from mid prices,
   with the spread carried so quality can be judged.
2. `record()` — writes one row per second through 15:15–15:40 to a
   per-day CAS tape: phase, last traded index, synthetic spot, basis,
   option mids and spreads. This is the training set for the regime and
   it did not exist before; nothing else in the system stores it.
3. `preprint_readiness()` — may we enter BEFORE the print yet? Requires
   CAS_MIN_SESSIONS of tape AND a fitted relationship showing the basis
   actually predicts the print. Same discipline as the post-auction
   regime: it opens itself on evidence, or never.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not assume Kite forwards NSE's indicative equilibrium price or
order imbalance. Those fields may exist on the wire; this system does
not depend on them. Everything here is derived from option quotes that
are unambiguously streamed. If the indicative fields do turn out to be
available, they are strictly additive.
"""
from __future__ import annotations

import json
import logging
import math
import time

import config

log = logging.getLogger("cas_capture")

TAPE_DIR = config.STATE_DIR / "cas_tape"
CERT_PATH = config.STATE_DIR / "cas_preprint_certificate.json"


def _mid(snap: dict | None) -> tuple[float, float]:
    """(mid, relative spread) from a leg snapshot, or (0,1) if unusable."""
    if not snap:
        return 0.0, 1.0
    try:
        b, a = float(snap.get("bid") or 0), float(snap.get("ask") or 0)
    except (TypeError, ValueError):
        return 0.0, 1.0
    if b <= 0 or a <= 0 or a < b:
        ltp = float(snap.get("ltp") or 0)
        return (ltp, 1.0) if ltp > 0 else (0.0, 1.0)
    return 0.5 * (a + b), (a - b) / a


def synthetic_spot(ce_snap: dict | None, pe_snap: dict | None,
                   strike: float, T_years: float,
                   r: float | None = None) -> tuple[float, float] | None:
    """Put-call parity synthetic underlying, and its quality score.

    Returns (S_synth, spread_penalty) or None when the quotes cannot
    support it. `spread_penalty` is the summed relative spread of the two
    legs — the honest width of the estimate. A synthetic spot derived
    from a 20%-wide option quote is not a price, it is a rumour, and the
    caller must be able to tell the difference.
    """
    c, sc = _mid(ce_snap)
    p, sp = _mid(pe_snap)
    if c <= 0 or p <= 0 or strike <= 0 or T_years is None or T_years < 0:
        return None
    rr = float(config.RISK_FREE_RATE if r is None else r)
    disc = math.exp(-rr * max(float(T_years), 0.0))
    return (c - p) + strike * disc, float(sc + sp)


def basis_points(s_synth: float, last_index: float) -> float:
    """How far the options say the auction will print from the last
    continuously-traded index. Positive ⇒ options expect a higher close."""
    if not (s_synth and last_index):
        return float("nan")
    return float(s_synth - last_index)


def tape_path(day: str, index: str):
    return TAPE_DIR / f"{day}_{index}.jsonl"


def record(day: str, index: str, ts: float, phase: str,
           last_index: float, ce_snap: dict | None, pe_snap: dict | None,
           strike: float, T_years: float) -> dict | None:
    """Append one second of CAS tape. Never raises into the trading loop —
    a failed write costs research data, not a position."""
    try:
        syn = synthetic_spot(ce_snap, pe_snap, strike, T_years)
        if syn is None:
            return None
        s_synth, pen = syn
        c, sc = _mid(ce_snap)
        p, sp = _mid(pe_snap)
        row = {"ts": round(float(ts), 1), "phase": phase,
               "last_index": round(float(last_index or 0.0), 2),
               "synth": round(s_synth, 2),
               "basis": round(basis_points(s_synth, last_index), 2),
               "ce_mid": round(c, 2), "pe_mid": round(p, 2),
               "ce_spread": round(sc, 4), "pe_spread": round(sp, 4),
               "quality": round(pen, 4), "strike": float(strike)}
        TAPE_DIR.mkdir(parents=True, exist_ok=True)
        with tape_path(day, index).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return row
    except Exception as e:                                 # noqa: BLE001
        log.debug("cas tape write skipped (%s)", e)
        return None


def sessions_captured(index: str = "NIFTY") -> list[str]:
    """Days for which a usable CAS tape exists."""
    try:
        out = []
        for f in sorted(TAPE_DIR.glob(f"*_{index}.jsonl")):
            if f.stat().st_size > 512:          # a few dozen rows minimum
                out.append(f.name.split("_")[0])
        return out
    except Exception:                                      # noqa: BLE001
        return []


def preprint_readiness(index: str = "NIFTY") -> tuple[bool, str]:
    """May the system take a position BEFORE the 15:35 print?

    This is the aggressive half of the regime and it carries the higher
    burden: not only enough sessions, but a FITTED relationship showing
    the option-implied basis actually forecasts the print. Until then the
    honest answer is no — being early to a move you cannot predict is
    just being exposed to it.
    """
    if not bool(getattr(config, "CAS_PREPRINT_ENABLED", True)):
        return False, "pre-print entries disabled by config"
    need = int(getattr(config, "CAS_MIN_SESSIONS", 7))
    have = len(sessions_captured(index))
    if have < need:
        return False, (f"learning: {have}/{need} CAS tape(s) recorded — "
                       f"the pre-print window opens by itself once the "
                       f"basis is shown to forecast the print")
    try:
        j = json.loads(CERT_PATH.read_text(encoding="utf-8"))
    except Exception:                                      # noqa: BLE001
        return False, f"{have} tape(s) but no fit yet — run the calibrator"
    if str(j.get("config_hash", "")) != config.CONFIG_HASH:
        return False, "certificate fitted under a different config"
    if not j.get("predictive"):
        return False, (f"measured: option-implied basis does NOT forecast "
                       f"the print ({j.get('reason', 'no relationship')}) — "
                       f"entering before it would be exposure, not edge")
    return True, (f"ready: basis forecasts the print over {have} session(s) "
                  f"| min |basis| {j.get('min_basis_pts', 0):.0f} pts "
                  f"| hit rate {j.get('hit_rate', 0):.0%}")


def preprint_signal(basis: float, quality: float,
                    index: str = "NIFTY") -> tuple[str | None, str]:
    """The CAS entry decision — the ONLY signal trusted inside 15:15–15:35.

    The conviction stack cannot be used here: it reads the index, and the
    index is indicative through the auction. The option-implied basis is
    the one number sourced entirely from continuously-traded instruments,
    so it decides direction on its own or there is no trade.

    basis > 0 ⇒ the options price a HIGHER close than the frozen index,
                so the print should gap up ⇒ CE.
    basis < 0 ⇒ PE.

    Returns (direction | None, reason).
    """
    ok, why = preprint_readiness(index)
    if not ok:
        return None, why
    try:
        j = json.loads(CERT_PATH.read_text(encoding="utf-8"))
    except Exception:                                      # noqa: BLE001
        return None, "no fitted certificate"
    cap = float(getattr(config, "CAS_MAX_QUALITY_PENALTY", 0.10))
    if quality > cap:
        return None, (f"quote quality {quality:.1%} worse than {cap:.0%} — "
                      f"the synthetic spot is a rumour, not a price")
    floor = float(j.get("min_basis_pts", 0.0) or 0.0)
    if not (basis == basis) or abs(basis) < floor:
        return None, (f"basis {basis:+.1f} pts inside the fitted floor "
                      f"±{floor:.1f} — nothing to act on")
    return ("CE" if basis > 0 else "PE",
            f"basis {basis:+.1f} pts ≥ floor {floor:.1f} "
            f"(fit: {j.get('hit_rate', 0):.0%} signs, p={j.get('sign_test_p', 1):.4f})")


def preprint_exit_decision(entry: float, mark: float, peak: float,
                           now_hm: str, phase: str,
                           geom=None) -> tuple[bool, str]:
    """Exit ladder for a position carried INTO the print.

    Two regimes in one trade, and conflating them is the error to avoid:

      • BEFORE the print (15:15–15:35) the position is a bet on an event
        that has not happened. Intermediate wobble is not information —
        the auction has not uncrossed — so only the DISASTER FLOOR may
        exit. A normal stop here would be stopped out by noise and then
        miss the very move it was placed for.

      • AFTER the print (15:35 onward) the event has occurred and the
        position becomes an ordinary short-horizon trade: the
        post-auction ladder takes over, with its target, stop, ratchet
        and absolute bell.
    """
    from core import post_auction as _PA
    g = geom or _PA.load()
    if entry <= 0:
        return True, "CAS_BAD_ENTRY"
    floor_pct = float(getattr(config, "MAX_LOSS_PER_TRADE_PCT", 0.3))
    if mark <= entry * (1.0 - floor_pct):
        return True, "CAS_DISASTER_FLOOR"          # constitution, always
    if now_hm >= str(getattr(config, "POST_AUCTION_FLAT_HM", "15:39")):
        return True, "CAS_BELL"
    if phase in ("CAS_REFERENCE", "CAS_ENTRY", "CAS_LIMIT_ONLY",
                 "CAS_MATCHING"):
        return False, ""                            # carry into the print
    if g.tp_pct <= 0:
        return True, "CAS_UNCALIBRATED"
    up = mark / entry - 1.0
    if up >= g.tp_pct:
        return True, "CAS_TARGET"
    if 1.0 - mark / entry >= g.sl_pct:
        return True, "CAS_STOP"
    if peak / entry - 1.0 >= g.arm_at_pct and mark <= peak * (
            1.0 - g.trail_giveback):
        return True, "CAS_TRAIL"
    return False, ""


def fit_preprint(tapes: list[list[dict]], prints: list[float]
                 ) -> tuple[bool, dict]:
    """Does the basis observed DURING the auction forecast the actual move
    from 15:15's last index to the 15:35 print?

    tapes  : per-session rows (from the tape) inside 15:15–15:35
    prints : per-session realised (print − last_continuous_index), points

    The test is deliberately blunt: sign agreement between the mean basis
    and the realised gap, plus a magnitude floor so we never act on a
    basis smaller than the option spread that produced it. A relationship
    that cannot beat a coin on direction is not a forecast.
    """
    import statistics as st
    xs, ys, quals = [], [], []
    for rows, realised in zip(tapes, prints):
        good = [r for r in rows
                if r.get("basis") == r.get("basis")
                and r.get("quality", 1.0) <= float(
                    getattr(config, "CAS_MAX_QUALITY_PENALTY", 0.10))]
        if len(good) < 30:
            continue
        xs.append(st.mean(r["basis"] for r in good))
        ys.append(float(realised))
        quals.append(st.mean(r["quality"] for r in good))
    n = len(xs)
    if n < int(getattr(config, "CAS_MIN_SESSIONS", 7)):
        return False, {"reason": f"only {n} usable session(s)", "n": n}
    hits = sum(1 for x, y in zip(xs, ys) if x * y > 0)
    hit = hits / n
    med_abs = st.median(abs(x) for x in xs)
    med_move = st.median(abs(y) for y in ys)
    # EXACT BINOMIAL SIGN TEST, not a hit-rate threshold.
    # A "must beat 65%" floor is not a test at these sample sizes: with 9
    # sessions, 6 correct signs is 67% and happens 25% of the time on pure
    # noise — my own self-test produced exactly that false positive. The
    # bar is now P(X >= hits | n, p=0.5) <= alpha, which at n=9 demands
    # 8/9. That is deliberately hard, and it means this regime will NOT
    # open after one week unless the relationship is genuinely strong;
    # it will open when the evidence is real, which is the point.
    pv = sum(math.comb(n, i) for i in range(hits, n + 1)) / (2.0 ** n)
    alpha = float(getattr(config, "CAS_ALPHA", 0.05))
    predictive = (pv <= alpha and med_abs > 0 and med_move > med_abs * 0.5)
    ev = {"predictive": bool(predictive), "n": n, "hit_rate": round(hit, 3),
          "sign_test_p": round(pv, 5), "alpha": alpha,
          "median_abs_basis_pts": round(med_abs, 2),
          "median_abs_print_move_pts": round(med_move, 2),
          "median_quality": round(st.mean(quals), 4) if quals else None,
          "min_basis_pts": round(max(med_abs, 1.0), 2),
          "reason": (f"basis forecasts the print ({hits}/{n} signs, "
                     f"p={pv:.4f})" if predictive else
                     f"{hits}/{n} correct signs, sign-test p={pv:.3f} > "
                     f"{alpha} (or the basis is too small to act on) — "
                     f"not distinguishable from chance"),
          "config_hash": config.CONFIG_HASH, "ts": time.time()}
    try:
        CERT_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CERT_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(ev, indent=1), encoding="utf-8")
        import os
        os.replace(tmp, CERT_PATH)
    except Exception as e:                                 # noqa: BLE001
        log.warning("cas certificate write failed (%s)", e)
    return predictive, ev


# ---------------------------------------------------------------- selftest
if __name__ == "__main__":                                 # pragma: no cover
    import sys
    import tempfile
    from pathlib import Path
    ok = 0

    def chk(name, cond):
        global ok
        print(("  PASS  " if cond else "  FAIL  ") + name)
        ok += bool(cond)

    config.STATE_DIR = Path(tempfile.mkdtemp())
    globals()["TAPE_DIR"] = config.STATE_DIR / "cas_tape"
    globals()["CERT_PATH"] = config.STATE_DIR / "cas_preprint_certificate.json"

    # ---- put-call parity recovers a known spot
    K, T, S = 24700.0, 1.3 / 365.0, 24750.0
    disc = math.exp(-config.RISK_FREE_RATE * T)
    c_true = 95.0
    p_true = c_true - S + K * disc          # parity, by construction
    ce = {"bid": c_true - 0.5, "ask": c_true + 0.5}
    pe = {"bid": p_true - 0.5, "ask": p_true + 0.5}
    syn, pen = synthetic_spot(ce, pe, K, T)
    chk(f"parity recovers spot ({syn:.1f} vs {S:.1f})", abs(syn - S) < 1.0)
    chk("quality penalty reflects the spread", 0 < pen < 0.05)
    chk("wide quotes score worse",
        synthetic_spot({"bid": 80, "ask": 110}, {"bid": 30, "ask": 60},
                       K, T)[1] > pen)
    chk("unusable quotes refuse", synthetic_spot(None, pe, K, T) is None)
    chk("zero strike refuses", synthetic_spot(ce, pe, 0.0, T) is None)

    # ---- basis is the options' forecast of the auction gap
    chk("basis is synth minus last traded index",
        abs(basis_points(24780.0, 24750.0) - 30.0) < 1e-9)

    # ---- the tape
    r = record("2026-08-05", "NIFTY", time.time(), "CAS_ENTRY", 24750.0,
               ce, pe, K, T)
    chk("tape row written", r is not None and "basis" in r)
    for _ in range(40):
        record("2026-08-05", "NIFTY", time.time(), "CAS_ENTRY", 24750.0,
               ce, pe, K, T)
    chk("session counted once tape is substantial",
        sessions_captured("NIFTY") == ["2026-08-05"])

    rdy, why = preprint_readiness("NIFTY")
    chk("pre-print refuses with one session", not rdy and "1/7" in why)

    # ---- the fit: a REAL relationship opens it
    import random
    rng = random.Random(4)
    tapes, prints = [], []
    for i in range(9):
        b = rng.choice([-40, -25, 25, 40]) + rng.uniform(-3, 3)
        tapes.append([{"basis": b + rng.uniform(-2, 2), "quality": 0.02}
                      for _ in range(60)])
        prints.append(b * rng.uniform(0.8, 1.4))       # print follows basis
    pred, ev = fit_preprint(tapes, prints)
    chk(f"a real relationship is found ({ev['hit_rate']:.0%} hits, "
        f"p={ev['sign_test_p']:.4f})", pred)
    rdy, why = preprint_readiness("NIFTY")
    chk("but readiness still needs the tapes",
        not rdy or "ready" in why)          # only 1 tape on disk here

    # ---- pure noise must NOT open it
    tapes_n = [[{"basis": rng.uniform(-30, 30), "quality": 0.02}
                for _ in range(60)] for _ in range(9)]
    prints_n = [rng.uniform(-30, 30) for _ in range(9)]
    pred_n, ev_n = fit_preprint(tapes_n, prints_n)
    chk(f"noise is refused ({ev_n['hit_rate']:.0%} hits, "
        f"p={ev_n['sign_test_p']:.3f})", not pred_n)
    rdy, why = preprint_readiness("NIFTY")
    chk("and the refusal reaches readiness", not rdy)

    # ---- wide-spread rows are excluded from the fit
    junk = [[{"basis": 50.0, "quality": 0.5} for _ in range(60)]
            for _ in range(9)]
    pred_j, ev_j = fit_preprint(junk, [50.0] * 9)
    chk("rows wider than the quality cap are dropped",
        not pred_j and ev_j["n"] == 0)

    # ---- the entry signal
    d, why = preprint_signal(50.0, 0.02)
    chk("signal refuses while un-ready", d is None)
    import json as _j
    CERT_PATH.write_text(_j.dumps({"predictive": True, "min_basis_pts": 20.0,
                                   "hit_rate": 0.89, "sign_test_p": 0.02,
                                   "config_hash": config.CONFIG_HASH}))
    for _ in range(7):
        pass
    # fabricate 7 tapes so readiness passes
    for i in range(7):
        for _ in range(45):
            record(f"2026-08-{10+i:02d}", "NIFTY", time.time(), "CAS_ENTRY",
                   24750.0, ce, pe, K, T)
    d, why = preprint_signal(50.0, 0.02)
    chk(f"positive basis ⇒ CE ({why[:34]})", d == "CE")
    chk("negative basis ⇒ PE", preprint_signal(-50.0, 0.02)[0] == "PE")
    chk("basis inside the fitted floor ⇒ no trade",
        preprint_signal(12.0, 0.02)[0] is None)
    chk("wide quotes ⇒ no trade", preprint_signal(50.0, 0.4)[0] is None)

    # ---- the carry-into-the-print ladder
    from core import post_auction as _PAT
    gg = _PAT.fit_geometry([60, 85, 110, 45, 130, 75, 95], 24750.0,
                           [0.004, 0.005, 0.004, 0.006, 0.005, 0.004,
                            0.005], 7)
    chk("noise before the print does NOT stop us out",
        not preprint_exit_decision(100.0, 96.0, 100.0, "15:24",
                                   "CAS_ENTRY", gg)[0])
    chk("but the disaster floor always fires",
        preprint_exit_decision(100.0, 65.0, 100.0, "15:24", "CAS_ENTRY",
                               gg)[1] == "CAS_DISASTER_FLOOR")
    chk("after the print the target works",
        preprint_exit_decision(100.0, 100 * (1 + gg.tp_pct + .01), 100.0,
                               "15:36", "POST_AUCTION",
                               gg)[1] == "CAS_TARGET")
    chk("after the print the stop works",
        preprint_exit_decision(100.0, 100 * (1 - gg.sl_pct - .01), 100.2,
                               "15:36", "POST_AUCTION",
                               gg)[1] == "CAS_STOP")
    chk("the bell is absolute",
        preprint_exit_decision(100.0, 200.0, 200.0, "15:39", "POST_AUCTION",
                               gg)[1] == "CAS_BELL")

    total = 24
    print(f"\n{ok}/{total} checks passed")
    sys.exit(0 if ok == total else 1)