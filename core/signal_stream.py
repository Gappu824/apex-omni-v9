"""
SIGNAL STREAM — replay each session once, not once per study
=============================================================
THE 30-HOUR PROBLEM
--------------------
The 2026-08-14 evening chain took 1804 minutes. The discovery group alone:

    entry_bar_study    59 494 s   (16.5 h)
    gate_ab_study      46 448 s   (12.9 h, and SERIAL)
    horizon_sweep      33 944 s   ( 9.4 h, then crashed)
    trade_potential     4 795 s

Every one of those replays THE SAME 40 SESSIONS. And the replay is not
cheap for the reason people usually assume: the day caches are already
warm (`prime_day_caches: 0 stale`), so the tick arrays load instantly.
The cost is the per-second decision loop — HeuristicPolicy().predict once
per second per index, 22 500 x 3 = 67 500 forward passes per session,
repeated in full by every study that wants the signal stream.

Three studies asking the identical question of the identical tape and
each paying for the answer separately.

WHAT THIS CACHES, AND WHAT IT DELIBERATELY DOES NOT
-----------------------------------------------------
CACHED: the SIGNAL STREAM — (t, index, conviction, win_prob, spot) for
every second the replayer emitted. That is the expensive, deterministic
part: given the same day cache and the same policy, it is the same
answer every time.

NOT CACHED: quotes, the chain walk, or anything a study prices against.
Those come from the day arrays, which are already cached upstream and are
cheap to slice. Caching them here would duplicate `load_day`'s job and
create a second copy that could drift from it — and a study pricing fills
off a stale private copy is a far worse failure than a slow study.

INVALIDATION IS THE WHOLE DESIGN
---------------------------------
A cached signal stream that outlives the policy that produced it is a
study of a system that no longer exists. The stamp therefore binds:

    CONFIG_HASH      the feature world
    decision_stamp   the policy path (the forge's own marker)
    data_stamp       the vault contents
    STREAM_VERSION   this file's own schema

Any change to any of them and the stream is rebuilt. This is the same
discipline that failed on 2026-08-14 when the meta-sample cache was NOT
bumped after the publication changed shape — every day cache-hit, the
replay never ran, and the matrix shipped without the fields it was
supposed to carry. The lesson taken here: a cache key must cover
everything the cached value depends on, and the version marker must be
bumped by the same commit that changes the shape.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

import config

log = logging.getLogger("signal_stream")

STREAM_VERSION = "s1"


@dataclass
class Stream:
    """One session's emitted signals, in emission order."""
    day: str
    t: np.ndarray           # session-second
    idx: np.ndarray         # index name per row
    conv: np.ndarray
    wp: np.ndarray
    spot: np.ndarray
    n_sec: int              # last second the replay reached
    # v9.9.42: the replayer's per-second SIDE EFFECTS. `last_tick` and
    # `spot_hist` are built INSIDE _Replayer.run (lines 519, 559) — they are
    # not set up in __init__. Caching only the decisions and skipping the
    # loop therefore left last_tick EMPTY, so
    #     stale = t - last_tick.get(tok, -1e9) > SHADOW_MAX_STALE_S
    # was True for every token, every quote read as dead, _affordable found
    # no rung, and gate_ab_study returned 0 trades in all four arms across
    # all 40 sessions while reporting "indistinguishable". A cache that
    # restores state must restore ALL of it; a partial restore is worse
    # than no cache, because it fails silently and looks like a result.
    tick_tok: np.ndarray = None      # token
    tick_last: np.ndarray = None     # its final real-tick second
    spot_idx: np.ndarray = None      # index name per spot series
    spot_series: np.ndarray = None   # (n_index, n_sec+1) per-second spot

    def __len__(self) -> int:
        return int(self.t.size)

    def restore_into(self, rep) -> None:
        """Put the replayer back in the state a full run() would have left.

        Without this a cache hit yields a replayer with an empty last_tick
        and empty spot_hist — quotes read stale, chains find no rung, and
        every study silently measures nothing.
        """
        from collections import deque
        try:
            rep.last_tick = {int(k): int(v) for k, v in
                             zip(self.tick_tok, self.tick_last)}
            for k, name in enumerate(self.spot_idx):
                ser = np.asarray(self.spot_series[k], float)
                ser = ser[np.isfinite(ser)]
                rep.spot_hist[str(name)] = deque(ser.tolist(), maxlen=1800)
        except Exception as e:                             # noqa: BLE001
            log.warning("stream restore failed (%s) — the caller will see a "
                        "replayer with no tick state and should rebuild", e)
            raise

    def iter_signals(self):
        for i in range(self.t.size):
            yield (int(self.t[i]), str(self.idx[i]), float(self.conv[i]),
                   float(self.wp[i]), float(self.spot[i]))


def _dir():
    p = config.STATE_DIR / "signal_stream"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stamp(day: str | None = None) -> str:
    """Everything THIS DAY's stream depends on — and nothing else.

    The first version included nightly_forge._data_stamp(), which is
    VAULT-WIDE. Adding one session therefore invalidated every stream:
    2026-08-20 reported "41 session(s) in vault | 0 already cached | 41 to
    build" and spent 365 minutes rebuilding streams that had not changed.
    That is not a cache — it is a rebuild with extra steps, and it would
    have cost six hours EVERY evening as the vault grew.

    A stream for 2026-06-16 depends on that day's ticks and the decision
    path. It has no dependency on whether 2026-08-19 exists. So the key is
    the day's own cache stamp plus the decision path, and a new session
    leaves the previous 40 untouched.
    """
    try:
        from nightly_forge_v9 import _decision_stamp
        dec = _decision_stamp()
    except Exception:                                      # noqa: BLE001
        dec = "?"
    d = ""
    if day:
        try:
            # the day cache's own stamp — changes only if THIS day's ticks
            # or the cache format change
            from core import day_cache as _DC
            d = str(_DC.day_stamp(day)) if hasattr(_DC, "day_stamp") else ""
        except Exception:                                  # noqa: BLE001
            d = ""
        if not d:
            # fall back to the day's tick-file mtime+size via the vault row
            # count for that date; still per-day, never vault-wide
            d = _day_fingerprint(day)
    return f"{config.CONFIG_HASH}:{dec}:{d}:{STREAM_VERSION}"


def _day_fingerprint(day: str) -> str:
    """A cheap, INDEXED, per-day fingerprint of the session's ticks.

    Two things this must not do. It must not scan: an earlier draft used
    `date(ts_local_ms/1000,'unixepoch','localtime')=?`, which no index
    covers, so fingerprinting 41 sessions meant 41 full scans of a 4.6M-row
    table just to decide whether a cache was warm. And it must not collapse
    to a constant on failure: if every day shares a fingerprint, a
    corrupted stream for one session looks valid for all of them.

    So it bounds on ts_ms — which idx_ticks_ts covers — and falls back to
    the day string itself, which is at least unique per day.
    """
    import datetime as _dt
    import sqlite3
    try:
        mid = _dt.datetime.combine(_dt.date.fromisoformat(day),
                                   _dt.time(0, 0)).timestamp()
        lo, hi = int(mid * 1000), int((mid + 86400) * 1000)
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        try:
            r = con.execute(
                "SELECT COUNT(*), MAX(ts_ms) FROM ticks_v9 "
                "WHERE ts_ms >= ? AND ts_ms < ?", (lo, hi)).fetchone()
        finally:
            con.close()
        if r and r[0]:
            return f"{int(r[0])}-{int(r[1] or 0)}"
    except Exception:                                      # noqa: BLE001
        pass
    return f"nodb-{day}"


def _path(day: str):
    return _dir() / f"stream_{day}.npz"


def load(day: str) -> Stream | None:
    try:
        p = _path(day)
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=False)
        if str(z["stamp"]) != _stamp(day):
            return None
        if "tick_tok" not in z.files:
            log.info("stream for %s predates the side-effect fix — "
                     "rebuilding rather than restoring a replayer whose "
                     "last_tick would be empty", day)
            return None
        return Stream(day=day, t=z["t"], idx=z["idx"], conv=z["conv"],
                      wp=z["wp"], spot=z["spot"], n_sec=int(z["n_sec"]),
                      tick_tok=z["tick_tok"], tick_last=z["tick_last"],
                      spot_idx=z["spot_idx"], spot_series=z["spot_series"])
    except Exception as e:                                 # noqa: BLE001
        log.debug("stream load failed for %s (%s)", day, e)
        return None


def save(s: Stream) -> None:
    try:
        p = _path(s.day)
        # np.savez appends .npz to a name that lacks it — the 2026-08-05
        # WinError 5 that broke every atomic publication. The name already
        # ends in .npz, so os.replace sees the file it expects.
        tmp = p.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, stamp=np.asarray(_stamp(s.day)), t=s.t,
                            idx=s.idx, conv=s.conv, wp=s.wp, spot=s.spot,
                            n_sec=np.asarray(s.n_sec),
                            tick_tok=s.tick_tok, tick_last=s.tick_last,
                            spot_idx=s.spot_idx, spot_series=s.spot_series)
        os.replace(tmp, p)
    except Exception as e:                                 # noqa: BLE001
        log.warning("stream save failed for %s (%s) — the next study will "
                    "re-replay this session", s.day, e)


def build(con, day: str, decide, meta, cal, actions_fn=None,
          force: bool = False) -> tuple[Stream | None, object]:
    """Return (stream, replayer). The replayer is handed back because a
    study still needs its quote/chain arrays; only the DECISION LOOP is
    served from cache.

    On a hit the replayer is constructed but never `run()`, which is the
    entire saving: the arrays load from the warm day cache in milliseconds
    while the 67 500 policy forward passes are skipped.
    """
    from nightly_forge_v9 import _Replayer

    rep = _Replayer(con, day, meta, cal)
    if not getattr(rep, "ok", False):
        return None, rep

    if not force:
        s = load(day)
        if s is not None and len(s):
            s.restore_into(rep)
            log.debug("stream HIT %s: %d signal(s), %d token tick-state, "
                      "%d spot series restored", day, len(s),
                      int(s.tick_tok.size), int(s.spot_idx.size))
            return s, rep

    ts, ix, cv, wp, sp = [], [], [], [], []
    # capture the spot series per index as the loop walks it, so a cache
    # hit can rebuild spot_hist without re-running the decision loop
    spot_by_idx: dict = {i: [] for i in config.TRADABLE}

    def _hook(idx, ctx):
        ts.append(int(ctx["t"]))
        ix.append(str(idx))
        cv.append(float(ctx.get("conv") or 0.0))
        wp.append(float(ctx.get("wp") or 0.0))
        sp.append(float(ctx.get("spot") or 0.0))

    last = 0
    for ev in rep.run(decide, on_signal=_hook, actions_fn=actions_fn):
        if ev and ev[0] == "sec":
            last = int(ev[1])
            for _i in config.TRADABLE:
                _sh = rep.spot_hist.get(_i)
                spot_by_idx[_i].append(float(_sh[-1]) if _sh else np.nan)

    _toks = sorted(rep.last_tick)
    _idxs = list(config.TRADABLE)
    _w = max(len(v) for v in spot_by_idx.values()) if spot_by_idx else 0
    _ser = np.full((len(_idxs), _w), np.nan, np.float32)
    for _k, _i in enumerate(_idxs):
        _v = spot_by_idx.get(_i) or []
        _ser[_k, :len(_v)] = _v
    s = Stream(day=day, t=np.asarray(ts, np.int32),
               idx=np.asarray(ix), conv=np.asarray(cv, np.float32),
               wp=np.asarray(wp, np.float32),
               spot=np.asarray(sp, np.float32), n_sec=int(last),
               tick_tok=np.asarray(_toks, np.int64),
               tick_last=np.asarray([rep.last_tick[k] for k in _toks],
                                    np.int32),
               spot_idx=np.asarray(_idxs), spot_series=_ser)
    save(s)
    log.info("stream BUILT %s: %d signal(s) over %ds — cached for every "
             "study in this chain", day, len(s), last)
    return s, rep


def purge_stale() -> int:
    """Drop streams from a previous stamp. Cheap housekeeping; without it
    the directory grows one file per day per config world."""
    n = 0
    try:
        for p in _dir().glob("stream_*.npz"):
            try:
                _d = p.stem.replace("stream_", "")
                if str(np.load(p, allow_pickle=False)["stamp"]) != _stamp(_d):
                    p.unlink()
                    n += 1
            except Exception:                              # noqa: BLE001
                p.unlink(missing_ok=True)
                n += 1
    except Exception as e:                                 # noqa: BLE001
        log.debug("purge failed (%s)", e)
    if n:
        log.info("purged %d stale signal stream(s)", n)
    return n