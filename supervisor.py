"""
APEX OMNI v9.6 — MARKET-HOURS SUPERVISOR (the end of three-restart days)
========================================================================
One command owns the trading day:

    python supervisor.py

It launches the three market processes (data_harvester_v9, macro_gex_v9,
apex_main_v9) shortly before the open, watches them at 5 s cadence, restarts
any that die with exponential backoff (5→10→20→40→60 s, max 6 restarts per
process per hour — beyond that it stops trying and says so loudly: a token
or exchange problem needs a human, not a retry storm), tears everything
down cleanly at the close, and writes state/supervisor_status.json each
minute so the morning read shows exactly what happened unattended. It never
touches the evening ritual (run_evening.py) and it assumes get_token was run
— an auth-dead brain will hit the restart ceiling and be reported, not
hidden. Ctrl-C = orderly SIGINT to children, then exit.
"""
from __future__ import annotations

import datetime as dt
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import config

PROCS = ["data_harvester_v9.py", "macro_gex_v9.py", "apex_main_v9.py"]
PRE_OPEN_MIN = 5           # start this many minutes before the session open
POST_CLOSE_MIN = 5         # stop this many minutes after the close
MAX_RESTARTS_HOUR = 6


def _hm_to_sod(hm: str) -> int:
    h, m = (int(x) for x in hm.split(":"))
    return h * 3600 + m * 60


def _in_session(now: dt.datetime) -> bool:
    if now.weekday() >= 5:
        return False
    sod = now.hour * 3600 + now.minute * 60 + now.second
    return (_hm_to_sod(config.SESSION_OPEN) - PRE_OPEN_MIN * 60 <= sod
            <= _hm_to_sod(config.SESSION_CLOSE) + POST_CLOSE_MIN * 60)


def _next_backoff(n_recent: int) -> int:
    return min(5 * (2 ** max(n_recent - 1, 0)), 60)


class Child:
    def __init__(self, script: str):
        self.script = script
        self.proc: subprocess.Popen | None = None
        self.restarts: list[float] = []
        self.gave_up = False

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        self.proc = subprocess.Popen([sys.executable, self.script],
                                     cwd=str(Path(__file__).parent))
        print(f"[supervisor] started {self.script} pid={self.proc.pid}",
              flush=True)

    def stop(self):
        if self.alive():
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def tick(self):
        if self.gave_up or self.alive():
            return
        now = time.time()
        self.restarts = [t for t in self.restarts if now - t < 3600]
        if self.proc is not None:                     # it died
            if len(self.restarts) >= MAX_RESTARTS_HOUR:
                self.gave_up = True
                print(f"[supervisor] {self.script} exceeded "
                      f"{MAX_RESTARTS_HOUR} restarts/hour — GIVING UP; "
                      f"this needs a human (token? exchange?)", flush=True)
                return
            wait = _next_backoff(len(self.restarts) + 1)
            print(f"[supervisor] {self.script} died (rc="
                  f"{self.proc.returncode}) — restart in {wait}s", flush=True)
            time.sleep(wait)
            self.restarts.append(time.time())
        self.start()


def main():
    kids = [Child(s) for s in PROCS]
    running = False
    last_status = 0.0
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    print("[supervisor] up — waiting for the session window", flush=True)
    while not stop["flag"]:
        if _in_session(dt.datetime.now()):
            if not running:
                print("[supervisor] session window open — launching",
                      flush=True)
                for k in kids:
                    k.start()
                    time.sleep(2)
                running = True
            for k in kids:
                k.tick()
        elif running:
            print("[supervisor] session over — clean teardown", flush=True)
            for k in reversed(kids):
                k.stop()
            for k in kids:
                k.restarts, k.gave_up = [], False
            running = False
        if time.time() - last_status > 60:
            try:
                (config.STATE_DIR / "supervisor_status.json").write_text(
                    json.dumps({"ts": time.time(), "running": running,
                                "children": [{"script": k.script,
                                              "alive": k.alive(),
                                              "restarts_1h": len(k.restarts),
                                              "gave_up": k.gave_up}
                                             for k in kids]}))
            except Exception:                          # noqa: BLE001
                pass
            last_status = time.time()
        time.sleep(5)
    for k in reversed(kids):
        k.stop()
    print("[supervisor] down", flush=True)


if __name__ == "__main__":
    main()