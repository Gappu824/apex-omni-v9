"""
APEX OMNI v9 — IPC CORE (rebuilds the missing apex_ipc_core.BinaryRingBuffer)
=============================================================================
Cross-platform (Windows laptop friendly) single-writer / multi-reader shared
state over a memory-mapped file. Seqlock semantics: the writer bumps an odd
sequence number, writes payload, bumps to even; readers retry if they catch
an odd sequence or a torn read (CRC mismatch). Freshness is FIRST-CLASS: the
header carries the write timestamp and readers get (state, age_seconds), so
the live brain's watchdog (audit §8 leap) has ground truth, not hope.

v8's harvester re-serialized the whole dict every tick; v9 keeps that simple
model (it is fine at ~1 Hz × a few hundred tokens) but pickles with protocol
5 and writes once per tick into a fixed slot — no growth, no GC churn.
"""
from __future__ import annotations
import mmap
import os
import pickle
import struct
import time
import zlib
from pathlib import Path

_HDR = struct.Struct("<QdII")          # seq, ts, length, crc32
DEFAULT_SIZE = 8 * 1024 * 1024         # 8 MB slot — plenty for 6-index state


class BinaryRingBuffer:
    def __init__(self, path: str | Path = None, size: int = DEFAULT_SIZE,
                 writer: bool = False):
        import config
        self.path = Path(path or config.RING_BUFFER_PATH)
        self.size = size
        total = _HDR.size + size
        if writer or not self.path.exists() or self.path.stat().st_size < total:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "wb") as f:
                f.truncate(total)
        self._f = open(self.path, "r+b")
        self._mm = mmap.mmap(self._f.fileno(), total)
        self.writer = writer
        if writer:
            self._seq = 0
            self._write_header(0, time.time(), 0, 0)

    # ------------------------------------------------------------ writer
    def _write_header(self, seq, ts, length, crc):
        self._mm[: _HDR.size] = _HDR.pack(seq, ts, length, crc)

    def write_state(self, state: dict, ts: float | None = None):
        if not self.writer:
            raise RuntimeError("read-only handle")
        payload = pickle.dumps(state, protocol=5)
        if len(payload) > self.size:
            raise ValueError(f"state {len(payload)}B exceeds slot {self.size}B")
        ts = ts if ts is not None else time.time()
        self._seq += 1                                   # odd: write in flight
        self._write_header(self._seq, ts, len(payload),
                           zlib.crc32(payload))
        self._mm[_HDR.size:_HDR.size + len(payload)] = payload
        self._seq += 1                                   # even: stable
        self._write_header(self._seq, ts, len(payload),
                           zlib.crc32(payload))
        self._mm.flush(0, _HDR.size + len(payload))

    # ------------------------------------------------------------ reader
    def read_state(self, retries: int = 3) -> tuple[dict | None, float]:
        """Returns (state_dict | None, age_seconds)."""
        for _ in range(retries):
            seq1, ts, length, crc = _HDR.unpack(self._mm[: _HDR.size])
            if seq1 == 0 or seq1 % 2 == 1 or length == 0:
                time.sleep(0.001)
                continue
            payload = bytes(self._mm[_HDR.size:_HDR.size + length])
            seq2 = _HDR.unpack(self._mm[: _HDR.size])[0]
            if seq1 != seq2 or zlib.crc32(payload) != crc:
                continue                                  # torn read — retry
            try:
                return pickle.loads(payload), max(time.time() - ts, 0.0)
            except Exception:                             # noqa: BLE001
                continue
        return None, float("inf")

    def close(self):
        try:
            self._mm.close(); self._f.close()
        except Exception:                                 # noqa: BLE001
            pass
