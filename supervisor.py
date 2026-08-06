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
hidden. Ctrl-C = orderly stop of children (CTRL_BREAK on Windows via their
own process group, SIGINT on POSIX), then exit — with terminate()/taskkill
fallbacks so a child can never be left holding the ring mmap.
"""
from __future__ import annotations

import datetime as dt
import json
import os
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
RETRY_GIVEUP_S = 900       # re-attempt a given-up child every 15 min


def _hm_to_sod(hm: str) -> int:
    h, m = (int(x) for x in hm.split(":"))
    return h * 3600 + m * 60


def _equity_window() -> tuple[int, int]:
    """v9.9.12 — THE DATA-CAPTURE FIX.

    This read the legacy 15:30 constant, so from 2026-08-03 every equity
    process was being shut down TEN MINUTES BEFORE THE MARKET CLOSED. The
    2026-08-03 brain report proves it: `session_complete` stamped at
    10:00:00Z — 15:30 IST — while index options traded on until 15:40.
    The vault therefore holds ZERO seconds of the closing auction, the
    uncrossing, or the post-auction window: the system could not have
    traded those moves because it was not even watching them, and no
    amount of research can recover data that was never harvested.

    The window now follows core.session_calendar for TODAY, per index, so
    it is 15:30 + buffer on pre-reform days and 15:40 + buffer from the
    reform onward — and it will follow BSE automatically the day
    BSE_FOLLOWS_NSE_CAS is turned on.
    """
    import datetime as _dt
    from core import session_calendar as _SC
    today = _dt.date.today()
    close = max(_hm_to_sod(_SC.session_close_hm(today, i))
                for i in (list(getattr(config, "TRADABLE", [])) or ["NIFTY"]))
    return (_hm_to_sod(config.SESSION_OPEN) - PRE_OPEN_MIN * 60,
            close + POST_CLOSE_MIN * 60)


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
        self.gaveup_at = 0.0
        self.was_active = False
        self.done = False        # clean self-exit for THIS window (reset next open)
        self.retry_at = 0.0       # SUP-F4: non-blocking backoff deadline
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
        # SUP-F1: on Windows a plain Popen cannot receive SIGINT — stop() used
        # to raise, leaving the child alive holding the ring mmap (the root of
        # the weekend "stale process" that crashed the next writer). Launch in
        # a NEW PROCESS GROUP so we can deliver CTRL_BREAK_EVENT for a graceful
        # stop, with terminate()/taskkill as escalating fallbacks.
        _kw = {}
        if os.name == "nt":
            _kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.proc = subprocess.Popen([sys.executable, "-u", self.script],
                                     cwd=str(Path(__file__).parent),
                                     stdout=self.log_fh,
                                     stderr=subprocess.STDOUT, **_kw)
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
        p = self.proc
        if p is not None and p.poll() is None:
            try:
                if os.name == "nt":
                    p.send_signal(signal.CTRL_BREAK_EVENT)   # group we created
                else:
                    p.send_signal(signal.SIGINT)
            except (ValueError, OSError, AttributeError):
                try:
                    p.terminate()
                except Exception:                          # noqa: BLE001
                    pass
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                    p.wait(timeout=5)
                except Exception:                          # noqa: BLE001
                    if os.name == "nt":                    # last resort
                        subprocess.run(["taskkill", "/F", "/T", "/PID",
                                        str(p.pid)], capture_output=True)
        self.proc = None
        if self.log_fh:
            try:
                self.log_fh.close()
            except Exception:                              # noqa: BLE001
                pass
            self.log_fh = None

    def tick(self):
        if self.gave_up or self.done or self.alive():
            return
        now = time.time()
        if now < self.retry_at:          # SUP-F4/F5: backoff is a TIMESTAMP the
            return                        # loop re-checks — never a blocking
        #                                  sleep that freezes other children or
        #                                  deafens Ctrl-C.
        self.restarts = [t for t in self.restarts if now - t < 3600]
        if self.proc is not None:                     # it exited
            rc = self.proc.returncode
            # SUP restart-storm: a CLEAN exit (rc==0) inside the window means
            # the child finished its own work — apex_main self-exits at
            # SESSION_CLOSE while its window runs 5 min longer. A finished
            # child is not a fault: mark done for THIS window (next window-open
            # resets it), don't restart, don't count it. Only rc!=0 is a crash.
            if rc == 0:
                # SUP-F6: a clean exit near the window END is normal (self-exit
                # at close). A clean exit with lots of window LEFT is suspicious
                # — flag it loudly, but still don't hammer-restart (rc==0 means
                # it chose to stop; a restart loop wouldn't help).
                secs_left = self.window[1] - (
                    dt.datetime.now().hour * 3600
                    + dt.datetime.now().minute * 60)
                tag = ("(work done for this window)" if secs_left < 600
                       else f"⚠ {secs_left // 60} min of window REMAIN — "
                            f"check {_proc_log(self.script).name}")
                print(f"[supervisor] {self.script} exited cleanly {tag} — "
                      f"not restarting", flush=True)
                self.done = True
                return
            if len(self.restarts) >= MAX_RESTARTS_HOUR:
                self.gave_up = True
                self.gaveup_at = now
                print(f"[supervisor] {self.script} exceeded "
                      f"{MAX_RESTARTS_HOUR} restarts/hour — GIVING UP; "
                      f"this needs a human (token? exchange?)", flush=True)
                return
            wait = _next_backoff(len(self.restarts) + 1)
            self.retry_at = now + wait
            self.restarts.append(now)
            print(f"[supervisor] {self.script} CRASHED (rc="
                  f"{rc}) — restart in {wait}s", flush=True)
            return                        # come back after retry_at; no sleep
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
                    k.done = False        # new window → a finished child runs again
                # SUP-F2: a child that hit the restart ceiling is retried once
                # every RETRY_GIVEUP_S (default 900s) — if a human fixed the
                # token mid-session it recovers on its own instead of staying
                # dead until the window closes.
                if k.gave_up and (now.timestamp() - k.gaveup_at
                                  >= RETRY_GIVEUP_S):
                    print(f"[supervisor] retrying {k.script} after give-up "
                          f"cooldown (a fix may have landed)", flush=True)
                    k.gave_up = False
                    k.restarts = []
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
                                              "in_window": k.was_active,
                                              "alive": k.alive(),
                                              "done": k.done,
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