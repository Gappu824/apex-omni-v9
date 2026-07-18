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
import shutil
import subprocess
import sys
import time
from pathlib import Path

import config

PRE_OPEN_MIN = 5           # start this many minutes before a window opens
POST_CLOSE_MIN = 5         # stop this many minutes after a window closes
MAX_RESTARTS_HOUR = 6


def _hm_to_sod(hm: str) -> int:
    h, m = (int(x) for x in hm.split(":"))
    return h * 3600 + m * 60


def _equity_window() -> tuple[int, int]:
    return (_hm_to_sod(config.SESSION_OPEN) - PRE_OPEN_MIN * 60,
            _hm_to_sod(config.SESSION_CLOSE) + POST_CLOSE_MIN * 60)


def _mcx_window() -> tuple[int, int]:
    """Full commodity window: MCX open → the LATEST commodity close (SILVER
    23:55), buffered, capped at 23:59 so we never cross midnight."""
    opens = [_hm_to_sod(getattr(config, "COMMODITY_SESSION_OPEN", "09:00")),
             _hm_to_sod(config.SESSION_OPEN)]
    closes = [_hm_to_sod(v.get("session_close", "23:30"))
              for v in getattr(config, "COMMODITIES", {}).values()] or \
             [_hm_to_sod(config.SESSION_CLOSE)]
    return (min(opens) - PRE_OPEN_MIN * 60,
            min(max(closes) + POST_CLOSE_MIN * 60, 23 * 3600 + 59 * 60))


def _window_for(script: str) -> tuple[int, int]:
    """Per-process session window. The harvester and the commodity brain run
    the full MCX day when evening capture is on (operator decision — evening
    microstructure is a different regime, captured deliberately); the equity
    brain and macro_gex keep the equity window. Master switch reverts all."""
    if bool(getattr(config, "EVENING_CAPTURE_ENABLED", False)) and \
            script in ("data_harvester_v9.py", "apex_commodity_main.py"):
        return _mcx_window()
    return _equity_window()


def _active(sod: int, window: tuple[int, int]) -> bool:
    return window[0] <= sod <= window[1]


def _proc_log(script: str) -> Path:
    return Path(config.LOG_DIR) / f"proc_{Path(script).stem}.log"


def _tab_cmd(script: str, log_path: Path, platform: str = sys.platform,
             wt: str | None = None) -> list[str] | None:
    """The Windows Terminal command that opens ONE viewer tab live-tailing this
    child's log. Pure + testable. Returns None off-Windows or if wt.exe is
    absent (children still run + log; you just tail the files yourself).
    The tab is a VIEWER: the supervisor keeps direct Popen control of the
    child (restart/backoff/teardown intact); closing a tab kills nothing, and
    a restarted child streams into the same tab seamlessly."""
    if not bool(getattr(config, "SUPERVISOR_TABS", True)):
        return None
    if platform != "win32":
        return None
    wt = wt if wt is not None else shutil.which("wt")
    if not wt:
        return None
    return [wt, "-w", "0", "new-tab", "--title", Path(script).stem,
            "powershell", "-NoExit", "-Command",
            f"Get-Content -Path '{log_path}' -Wait -Tail 40"]


# (script, window) — windows resolved at import; config-driven
PROCS = ["data_harvester_v9.py", "macro_gex_v9.py", "apex_main_v9.py",
         "apex_commodity_main.py"]


def _next_backoff(n_recent: int) -> int:
    return min(5 * (2 ** max(n_recent - 1, 0)), 60)


class Child:
    def __init__(self, script: str):
        self.script = script
        self.window = _window_for(script)
        self.proc: subprocess.Popen | None = None
        self.restarts: list[float] = []
        self.gave_up = False
        self.was_active = False
        self.log_fh = None
        self.tab_opened = False

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        log_path = _proc_log(self.script)
        try:                      # rotate an oversized log (viewer tab re-tails)
            if log_path.exists() and log_path.stat().st_size > 50e6:
                log_path.replace(log_path.with_suffix(".log.old"))
        except Exception:                                  # noqa: BLE001
            pass
        self.log_fh = open(log_path, "a", buffering=1, encoding="utf-8",
                           errors="replace")
        self.proc = subprocess.Popen([sys.executable, "-u", self.script],
                                     cwd=str(Path(__file__).parent),
                                     stdout=self.log_fh,
                                     stderr=subprocess.STDOUT)
        print(f"[supervisor] started {self.script} pid={self.proc.pid} "
              f"→ {log_path.name}", flush=True)
        if not self.tab_opened:
            cmd = _tab_cmd(self.script, log_path)
            if cmd:
                try:
                    subprocess.Popen(cmd)
                    print(f"[supervisor] viewer tab opened for {self.script}",
                          flush=True)
                except Exception as e:                     # noqa: BLE001
                    print(f"[supervisor] tab open failed ({e}) — logs still "
                          f"at {log_path}", flush=True)
            self.tab_opened = True    # one tab per supervisor run, not per restart

    def stop(self):
        if self.alive():
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        if self.log_fh:
            try:
                self.log_fh.close()
            except Exception:                              # noqa: BLE001
                pass
            self.log_fh = None

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
    last_status = 0.0
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    print("[supervisor] up — waiting for the session window", flush=True)
    while not stop["flag"]:
        now = dt.datetime.now()
        weekday_ok = now.weekday() < 5
        sod = now.hour * 3600 + now.minute * 60 + now.second
        for k in kids:
            act = weekday_ok and _active(sod, k.window)
            if act:
                if not k.was_active:
                    print(f"[supervisor] window open for {k.script}",
                          flush=True)
                    k.was_active = True
                k.tick()
            elif k.was_active:
                print(f"[supervisor] window over for {k.script} — teardown",
                      flush=True)
                k.stop()
                k.restarts, k.gave_up, k.was_active = [], False, False
        if time.time() - last_status > 60:
            try:
                (config.STATE_DIR / "supervisor_status.json").write_text(
                    json.dumps({"ts": time.time(),
                                "children": [{"script": k.script,
                                              "window": list(k.window),
                                              "active": k.was_active,
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