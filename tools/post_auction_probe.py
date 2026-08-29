"""
APEX OMNI v9.9.18 — POST-AUCTION FEED PROBE (feed silence, or harvester carry?)
==============================================================================
post_auction_calibrate refuses 15:35-15:40 on every session it has seen: seven
report "300 real tick(s) but range is EXACTLY 0.00", one reports zero. It is
right to refuse — a disseminated index does not hold one value for five
minutes — but refusing is not diagnosing, and the two possible causes need
opposite fixes:

  A. THE FEED IS SILENT through the window and the harvester carries the last
     value forward, writing it once per second and marking it fresh. Fix:
     stop writing carried rows, or stop flagging them fresh. Harvester-side.

  B. THE FEED PUBLISHES but the vault stores it wrong (wrong token, a stale
     snapshot, a write path that dedupes to one row). Fix: harvester write
     path or token mapping. Also harvester-side, but a different bug.

They are separated by EVIDENCE THIS TOOL READS DIRECTLY, and by a CONTROL
WINDOW. Statistics from the suspect window alone prove nothing — 1 row/sec
with one distinct value looks damning until you find the same signature at
14:00, at which point the harvester writes on a timer ALL DAY and `fresh` is
meaningless everywhere, not just at the close. So every metric below is
reported for 15:35-15:40 AND for a mid-session control, and it is the
DIFFERENCE that carries the diagnosis.

Reads the vault read-only. Writes nothing. Run it on your own box:

    python tools/post_auction_probe.py --days 8
    python tools/post_auction_probe.py --day 2026-08-06 --verbose
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                        # noqa: E402
import config                                             # noqa: E402

config.setup_logging("post_auction_probe")
import logging                                            # noqa: E402
log = logging.getLogger("post_auction_probe")

CAS = ("15:35", "15:40")
CONTROL = ("14:00", "14:05")


def _sod(hm: str) -> int:
    h, m = (int(x) for x in hm.split(":"))
    return h * 3600 + m * 60


def _rows(con, day: str, token: int, a: str, b: str):
    q = ("SELECT ts_local_ms/1000.0 AS t, ltp, bid, ask FROM ticks_v9 "
         "WHERE token=? AND date(ts_local_ms/1000,'unixepoch','localtime')=? "
         "ORDER BY ts_local_ms")
    out = []
    for t, ltp, bid, ask in con.execute(q, (int(token), day)):
        sod = int((t + 19800) % 86400)
        if _sod(a) <= sod < _sod(b):
            out.append((t, ltp, bid, ask))
    return out


def _describe(rows, label: str) -> dict:
    if not rows:
        return {"label": label, "n": 0}
    ts = np.array([r[0] for r in rows], float)
    ltp = np.array([r[1] if r[1] is not None else np.nan for r in rows], float)
    fin = ltp[np.isfinite(ltp) & (ltp > 0)]
    dt = np.diff(ts) if ts.size > 1 else np.array([0.0])
    return {
        "label": label, "n": len(rows),
        "distinct_ltp": int(np.unique(fin).size) if fin.size else 0,
        "range": float(fin.max() - fin.min()) if fin.size else 0.0,
        "median_gap_s": float(np.median(dt)) if dt.size else 0.0,
        # the signature of a TIMER: every gap identical, to the millisecond
        "gap_is_constant": bool(dt.size > 3 and float(np.std(dt)) < 1e-3),
        "rows_per_sec": (len(rows) / max(ts.max() - ts.min(), 1e-9)
                         if ts.size > 1 else 0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--day", type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    try:
        days = ([a.day] if a.day else
                [r[0] for r in con.execute(
                    "SELECT DISTINCT date(ts_local_ms/1000,'unixepoch',"
                    "'localtime') d FROM ticks_v9 ORDER BY d DESC LIMIT ?",
                    (a.days,))][::-1])
        log.info("=" * 76)
        log.info("POST-AUCTION FEED PROBE | CAS %s-%s vs control %s-%s",
                 CAS[0], CAS[1], CONTROL[0], CONTROL[1])
        log.info("=" * 76)
        verdicts = []
        for day in days:
            toks = [r for r in con.execute(
                "SELECT name, token FROM spot_tokens WHERE snap_date=?",
                (day,))]
            if not toks:
                log.info("  %s: no spot_tokens snapshot — skipped", day)
                continue
            for name, tok in toks:
                cas = _describe(_rows(con, day, tok, *CAS), "CAS")
                ctl = _describe(_rows(con, day, tok, *CONTROL), "control")
                if not ctl["n"]:
                    continue
                log.info("  %s %-10s CAS n=%-4d distinct=%-4d range=%-8.2f "
                         "gap=%.2fs%s", day, name, cas["n"],
                         cas.get("distinct_ltp", 0), cas.get("range", 0.0),
                         cas.get("median_gap_s", 0.0),
                         "  [CONSTANT GAP]" if cas.get("gap_is_constant")
                         else "")
                log.info("  %s %-10s CTL n=%-4d distinct=%-4d range=%-8.2f "
                         "gap=%.2fs%s", " " * 10, " " * 10, ctl["n"],
                         ctl["distinct_ltp"], ctl["range"],
                         ctl["median_gap_s"],
                         "  [CONSTANT GAP]" if ctl["gap_is_constant"] else "")
                # ---- the discriminator
                if cas["n"] == 0:
                    v = "A-silent (no rows written at all)"
                elif cas.get("distinct_ltp", 0) <= 1 and ctl["distinct_ltp"] > 1:
                    v = ("A-carry (rows exist, ONE value, while the control "
                         "window varies normally)")
                elif cas.get("gap_is_constant") and not ctl["gap_is_constant"]:
                    v = ("A-carry (CAS arrivals are on a timer, control "
                         "arrivals are event-driven)")
                elif cas.get("distinct_ltp", 0) <= 1 and ctl["distinct_ltp"] <= 1:
                    v = ("NEITHER — the control window is degenerate too. The "
                         "harvester writes on a timer ALL DAY; `fresh` is "
                         "meaningless everywhere, not just at the close")
                else:
                    v = "B-or-healthy (CAS varies — re-check the consumer)"
                log.info("       -> %s", v)
                verdicts.append(v.split()[0])
        log.info("-" * 76)
        if verdicts:
            from collections import Counter
            for k, n in Counter(verdicts).most_common():
                log.info("  %-10s %d session-token(s)", k, n)
            log.info("")
            log.info("  A-*  => harvester-side: stop writing carried rows into")
            log.info("          the CAS window, or stop flagging them fresh.")
            log.info("          post_auction_calibrate is already correct to")
            log.info("          refuse them; the data must stop arriving.")
            log.info("  B-*  => the rows are real; the consumer or the token")
            log.info("          mapping is at fault, not the feed.")
            log.info("  NEITHER => the timer behaviour is global. Fix `fresh`")
            log.info("          semantics before trusting ANY freshness gate.")
        else:
            log.info("  no comparable session/token pairs found")
        log.info("=" * 76)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())