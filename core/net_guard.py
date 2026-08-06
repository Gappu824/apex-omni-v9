"""
NET GUARD — a name lookup must never freeze the trading loop
=============================================================
2026-07-29, 10:20 IST: the brain went totally silent for 36 minutes with
a position open. No traceback, no exit — the process was alive and
blocked. Earlier that morning the log carries `getaddrinfo failed`. That
is the signature.

WHY A TIMEOUT DID NOT SAVE IT
-----------------------------
`socket.settimeout` and every HTTP-library timeout apply to CONNECT and
READ. Name resolution happens BEFORE either: `getaddrinfo()` is a
blocking call into the platform resolver (glibc / Windows DNS Client),
and it honours no Python-level timeout at all. When the resolver stalls
— captive portal, VPN flap, an unreachable DNS server with a long
retry — the calling thread parks in C. On a single-threaded 1 Hz loop
that means the market keeps moving and nothing is watching the trade.

THE FIX, IN THREE LAYERS
------------------------
1. PIN THE NAMES. Resolve the small, fixed set of hosts this system
   talks to (Kite API/ticker) once at startup and cache the answer.
   Serving from cache means the steady state never calls the resolver at
   all — the hang cannot happen on the hot path.

2. STALE-ON-ERROR. If a later refresh fails, keep serving the last known
   good address instead of propagating the failure. An IP that worked
   sixty seconds ago is far better evidence than an exception, and DNS
   blips outlive their usefulness in milliseconds. (This is the standard
   "stale-while-revalidate" resilience pattern; it is why a cached
   resolver survives outages that kill an uncached one.)

3. BOUND THE UNAVOIDABLE. The very first lookup, and any cache miss,
   runs in a daemon thread with a hard deadline. If the resolver has not
   answered by then the caller gets an exception it can handle, while
   the stuck thread is abandoned to the OS. A leaked thread is a cheap
   price; a frozen trading loop is not.

Plus a WATCHDOG that is independent of all of the above: the main loop
stamps a heartbeat each pass, and a monitor thread shouts if the stamp
stops advancing. It cannot unblock C code — nothing in Python can — but
it converts an invisible 36-minute silence into a timestamped ERROR
naming the phase the loop died in, which is the difference between
"something happened" and a diagnosis.

Nothing here changes what the system trades. It changes whether the
system keeps breathing while the network misbehaves.
"""
from __future__ import annotations

import logging
import socket
import threading
import time

import config

log = logging.getLogger("net_guard")

_ORIG_GETADDRINFO = socket.getaddrinfo
_CACHE: dict[tuple, tuple[float, list]] = {}
_LOCK = threading.Lock()
_INSTALLED = False


def call_with_deadline(fn, timeout_s: float, name: str = "call"):
    """Run `fn()` in a daemon thread and give up after `timeout_s`.

    The worker cannot be killed — Python has no safe thread abort — but
    the CALLER is freed, which is the property that matters: the trading
    loop continues while an abandoned thread waits on the OS. Raises
    TimeoutError on expiry, re-raises the callee's exception otherwise.
    """
    box: dict = {}

    def _run():
        try:
            box["v"] = fn()
        except BaseException as e:                         # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=_run, name=f"deadline-{name}", daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TimeoutError(f"{name} exceeded {timeout_s:.1f}s — abandoned "
                           f"(thread leaked deliberately; the loop goes on)")
    if "e" in box:
        raise box["e"]
    return box.get("v")


def _cached_getaddrinfo(host, port, *args, **kwargs):
    """Drop-in for socket.getaddrinfo: cache hit → instant; miss →
    deadline-bounded lookup; failure → last known good, if we have one."""
    key = (host, port) + tuple(args)
    ttl = float(getattr(config, "DNS_CACHE_TTL_S", 300))
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(key)
    if hit and now - hit[0] < ttl:
        return hit[1]
    try:
        res = call_with_deadline(
            lambda: _ORIG_GETADDRINFO(host, port, *args, **kwargs),
            float(getattr(config, "DNS_TIMEOUT_S", 5.0)),
            f"getaddrinfo({host})")
        with _LOCK:
            _CACHE[key] = (now, res)
        return res
    except BaseException as e:                             # noqa: BLE001
        if hit:
            log.warning("DNS lookup for %s failed/stalled (%s) — serving "
                        "the last known good address (%.0fs old). The "
                        "session continues.", host, e, now - hit[0])
            with _LOCK:
                _CACHE[key] = (now - ttl * 0.5, hit[1])   # retry sooner
            return hit[1]
        raise


def install(hosts: list[str] | None = None) -> None:
    """Install the cache and pre-resolve the hosts this system uses.
    Idempotent; safe to call from any entrypoint."""
    global _INSTALLED
    if _INSTALLED or not bool(getattr(config, "NET_GUARD_ENABLED", True)):
        return
    to = float(getattr(config, "SOCKET_DEFAULT_TIMEOUT_S", 15.0))
    if socket.getdefaulttimeout() is None:
        socket.setdefaulttimeout(to)      # backstop for connect/read
    socket.getaddrinfo = _cached_getaddrinfo
    _INSTALLED = True
    warmed, failed = [], []
    for h in (hosts or list(getattr(config, "NET_GUARD_HOSTS", []))):
        try:
            _cached_getaddrinfo(h, 443, socket.AF_INET, socket.SOCK_STREAM)
            warmed.append(h)
        except Exception:                                  # noqa: BLE001
            failed.append(h)
    log.info("net guard installed | socket timeout %.0fs | DNS cache TTL "
             "%.0fs | warmed %s%s", to,
             float(getattr(config, "DNS_CACHE_TTL_S", 300)),
             ", ".join(warmed) or "none",
             f" | UNRESOLVED {', '.join(failed)}" if failed else "")


# ------------------------------------------------------------- watchdog
class LoopWatchdog:
    """Turns an invisible stall into a timestamped ERROR.

    The loop calls beat("phase") each pass; a monitor thread compares the
    stamp against wall time. It cannot interrupt blocked C code, but it
    names the phase and the duration, so a freeze is diagnosable from the
    log alone instead of inferred from silence.
    """

    def __init__(self, warn_s: float | None = None, name: str = "main"):
        self.warn_s = float(warn_s if warn_s is not None
                            else getattr(config, "LOOP_STALL_WARN_S", 90))
        self.name = name
        self._ts = time.time()
        self._phase = "start"
        self._stop = threading.Event()
        self._warned_at = 0.0
        self._t: threading.Thread | None = None

    def beat(self, phase: str = "") -> None:
        self._ts = time.time()
        if phase:
            self._phase = phase

    def start(self) -> "LoopWatchdog":
        def _mon():
            while not self._stop.wait(5.0):
                stalled = time.time() - self._ts
                if stalled >= self.warn_s and stalled - self._warned_at >= \
                        self.warn_s:
                    self._warned_at = stalled
                    log.error("LOOP STALLED: %s loop has not advanced for "
                              "%.0fs (last phase: %s). The process is alive "
                              "but blocked — most likely a network/name "
                              "resolution call. Any open position is "
                              "UNMANAGED right now.",
                              self.name, stalled, self._phase)
        self._t = threading.Thread(target=_mon, name=f"watchdog-{self.name}",
                                   daemon=True)
        self._t.start()
        log.info("loop watchdog armed (%s, warn after %.0fs)",
                 self.name, self.warn_s)
        return self

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------- selftest
if __name__ == "__main__":                                 # pragma: no cover
    import sys
    ok = 0

    def chk(name, cond):
        global ok
        print(("  PASS  " if cond else "  FAIL  ") + name)
        ok += bool(cond)

    # ---- deadline frees the caller from a hung call
    t0 = time.time()
    try:
        call_with_deadline(lambda: time.sleep(30), 0.4, "hang")
        freed = False
    except TimeoutError:
        freed = True
    dt_ = time.time() - t0
    chk("hung call raises TimeoutError", freed)
    chk(f"caller freed in {dt_:.2f}s, not 30s", dt_ < 2.0)
    chk("normal call returns its value",
        call_with_deadline(lambda: 7, 2.0, "ok") == 7)
    try:
        call_with_deadline(lambda: 1 / 0, 2.0, "boom")
        raised = False
    except ZeroDivisionError:
        raised = True
    chk("callee exceptions propagate unchanged", raised)

    # ---- DNS cache: hit is instant, and a stalled refresh serves stale
    _CACHE.clear()
    calls = {"n": 0}

    def fake_gai(host, port, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return [("fam", "type", 6, "", ("10.0.0.1", port))]
        time.sleep(30)                       # the resolver hangs, forever

    globals()["_ORIG_GETADDRINFO"] = fake_gai
    config.DNS_CACHE_TTL_S = 0.2
    first = _cached_getaddrinfo("api.kite.trade", 443)
    chk("first lookup resolves", first[0][4][0] == "10.0.0.1")
    t0 = time.time()
    second = _cached_getaddrinfo("api.kite.trade", 443)
    chk("cache hit is instant", time.time() - t0 < 0.05 and calls["n"] == 1)
    time.sleep(0.25)                                       # expire the TTL
    t0 = time.time()
    third = _cached_getaddrinfo("api.kite.trade", 443)
    took = time.time() - t0
    chk("stalled refresh serves the last known good address",
        third[0][4][0] == "10.0.0.1")
    chk(f"and returns in {took:.1f}s, bounded by the deadline",
        took < float(getattr(config, "DNS_TIMEOUT_S", 5.0)) + 1.5)

    # ---- an unknown host with no history still raises (no silent fiction)
    try:
        _cached_getaddrinfo("never.seen.invalid", 443)
        bad = True
    except BaseException:                                  # noqa: BLE001
        bad = False
    chk("unknown host with no cache raises rather than inventing one",
        not bad)

    # ---- watchdog notices a stall and names the phase
    seen = []

    class _Cap(logging.Handler):
        def emit(self, r):
            seen.append(r.getMessage())

    log.addHandler(_Cap())
    log.setLevel(logging.ERROR)
    w = LoopWatchdog(warn_s=0.5, name="test").start()
    w.beat("quote fetch")
    time.sleep(6.5)                        # monitor ticks every 5s
    w.stop()
    chk("watchdog fires on a stall", any("LOOP STALLED" in m for m in seen))
    chk("watchdog names the last phase",
        any("quote fetch" in m for m in seen))

    total = 11
    print(f"\n{ok}/{total} checks passed")
    sys.exit(0 if ok == total else 1)