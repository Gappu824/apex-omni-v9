"""
APEX OMNI v9.7.1 — VAULT MAINTENANCE (the disk-full remedy)
===========================================================
The vault lives at config.DB_PATH (default: <repo>/data on C:). Five harvested
commodities + equity chains grow it fast, and a full disk kills the evening
chain ("database or disk is full") and can even break logging. This tool is the
operational fix, in order of preference:

  1. REPORT what is actually consuming space (db + wal/shm, logs, state,
     models, per-day tick counts, free space on the drive).
  2. --prune-logs N        delete rotated *.log* files and DATED report jsons
                           older than N days. Never touches calibration.json,
                           the vault, models, or state.
  3. --vacuum-into PATH    rebuild the vault compactly at PATH (another drive).
                           Reads need no free space on the source drive. Then:
                           set  $env:APEX_DATA_DIR = <that folder>  (config now
                           honors it) and restart — everything follows DB_PATH.
  4. --prune-ticks-before YYYY-MM-DD [--confirm]
                           LAST RESORT. Dry-run by default; refuses to leave
                           fewer than RETAIN_MIN_DAYS of ticks. Ticks are the
                           training data — deleting them is the opposite of
                           "improve by harvesting more"; prefer (3).

  python tools/vault_maintenance.py                 # report
  python tools/vault_maintenance.py --prune-logs 14
  python tools/vault_maintenance.py --vacuum-into D:/apex_data/arjun_tick_vault_v9.db
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config                                             # noqa: E402

config.setup_logging("vault_maint")
import logging                                            # noqa: E402
log = logging.getLogger("vault_maint")

RETAIN_MIN_DAYS = 30


def _mb(p: Path) -> float:
    try:
        return p.stat().st_size / 1e6
    except Exception:                                          # noqa: BLE001
        return 0.0


def _dir_mb(d: Path) -> float:
    return sum(_mb(f) for f in d.rglob("*") if f.is_file()) if d.exists() else 0.0


def report():
    db = Path(config.DB_PATH)
    print(f"\n=== VAULT MAINTENANCE REPORT ({dt.date.today()}) ===")
    total, used, free = shutil.disk_usage(db.parent if db.parent.exists()
                                          else Path.cwd())
    print(f"drive [{db.parent}]: free {free/1e9:.1f} GB of {total/1e9:.1f} GB")
    print(f"vault  {db.name}: {_mb(db):.0f} MB"
          f" (+wal {_mb(db.with_name(db.name+'-wal')):.0f} MB,"
          f" +shm {_mb(db.with_name(db.name+'-shm')):.0f} MB)")
    for label, d in (("logs", config.LOG_DIR), ("state", config.STATE_DIR),
                     ("models", config.MODEL_DIR)):
        print(f"{label:6s} {d}: {_dir_mb(Path(d)):.0f} MB")
    if db.exists():
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            rows = con.execute(
                "SELECT date(ts_local_ms/1000,'unixepoch','localtime') d,"
                " COUNT(*) FROM ticks_v9 GROUP BY d ORDER BY d").fetchall()
            if rows:
                print(f"tick days: {len(rows)} ({rows[0][0]} → {rows[-1][0]}),"
                      f" total {sum(r[1] for r in rows):,} ticks")
                for d_, n in rows[-5:]:
                    print(f"    {d_}: {n:,}")
        except Exception as e:                                 # noqa: BLE001
            print(f"tick summary unavailable: {e}")
    print("remedy order: --prune-logs → --vacuum-into <bigger drive> "
          "(+ APEX_DATA_DIR) → --prune-ticks-before (last resort)\n")


def prune_logs(days: int):
    cutoff = dt.datetime.now() - dt.timedelta(days=days)
    freed, n = 0.0, 0
    for f in Path(config.LOG_DIR).glob("*"):
        if not f.is_file() or f.name == "calibration.json":
            continue
        rotated = ".log" in f.name and not f.name.endswith(".log")
        dated = any(ch.isdigit() for ch in f.stem[-10:]) and \
            f.suffix in (".json", ".md", ".log")
        if (rotated or dated) and \
                dt.datetime.fromtimestamp(f.stat().st_mtime) < cutoff:
            freed += _mb(f)
            n += 1
            f.unlink()
    log.info("pruned %d log/report file(s) older than %dd — freed %.0f MB",
             n, days, freed)


def vacuum_into(target: str):
    tgt = Path(target)
    tgt.parent.mkdir(parents=True, exist_ok=True)
    src = Path(config.DB_PATH)
    log.info("VACUUM INTO %s (source %.0f MB) — this reads the source and "
             "writes a compact copy; source drive needs no free space", tgt,
             _mb(src))
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    con.execute("VACUUM INTO ?", (str(tgt),))
    con.close()
    log.info("done: %s = %.0f MB. Now set APEX_DATA_DIR to %s (PowerShell: "
             "$env:APEX_DATA_DIR='%s'; add to your profile), ensure the file "
             "name matches %s, and restart the stack.", tgt.name, _mb(tgt),
             tgt.parent, tgt.parent, src.name)


def prune_ticks(before: str, confirm: bool):
    con = sqlite3.connect(config.DB_PATH)
    days = [r[0] for r in con.execute(
        "SELECT DISTINCT date(ts_local_ms/1000,'unixepoch','localtime') d "
        "FROM ticks_v9 ORDER BY d")]
    doomed = [d for d in days if d < before]
    kept = len(days) - len(doomed)
    if kept < RETAIN_MIN_DAYS:
        log.error("REFUSED: would leave %d day(s) < retention floor %d. "
                  "Ticks are the training data — use --vacuum-into to move "
                  "the vault instead.", kept, RETAIN_MIN_DAYS)
        return
    n = con.execute(
        "SELECT COUNT(*) FROM ticks_v9 WHERE "
        "date(ts_local_ms/1000,'unixepoch','localtime') < ?",
        (before,)).fetchone()[0]
    if not confirm:
        log.info("DRY-RUN: would delete %s ticks across %d day(s) (< %s), "
                 "keeping %d day(s). Re-run with --confirm to execute, then "
                 "run --vacuum-into to reclaim the space.", f"{n:,}",
                 len(doomed), before, kept)
        return
    con.execute("DELETE FROM ticks_v9 WHERE "
                "date(ts_local_ms/1000,'unixepoch','localtime') < ?", (before,))
    con.commit()
    log.info("deleted %s ticks (< %s). Space returns to the OS only after "
             "VACUUM — run --vacuum-into next.", f"{n:,}", before)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prune-logs", type=int, metavar="DAYS")
    ap.add_argument("--vacuum-into", metavar="PATH")
    ap.add_argument("--prune-ticks-before", metavar="YYYY-MM-DD")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    if a.prune_logs:
        prune_logs(a.prune_logs)
    if a.vacuum_into:
        vacuum_into(a.vacuum_into)
    if a.prune_ticks_before:
        prune_ticks(a.prune_ticks_before, a.confirm)
    if not any([a.prune_logs, a.vacuum_into, a.prune_ticks_before]):
        report()