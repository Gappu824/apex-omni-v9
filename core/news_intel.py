"""
NEWS INTEL — an LLM in the trading path, made auditable
========================================================
THE PROBLEM WITH PUTTING GEMMA IN THE ENTRY PATH
-------------------------------------------------
An LLM call inside a live decision is non-deterministic, unversioned and
unreplayable. The forge could never reproduce the decision, the ladder
could never certify it, and `entry_counterfactual` could never A/B it —
so the one component with the least evidence behind it would be the one
component immune to the evidence machinery. Every other input in this
system has to clear a paired, FDR-corrected, MDE-floored test. A news
score that cannot be replayed cannot even be tested.

THE FIX: SCORE ONCE, PERSIST THE SCALAR, REPLAY THE FILE
---------------------------------------------------------
The model runs ONCE per session, before the commit window, and writes a
small immutable record:

    state/news/news_2026-08-12.json
      { "day", "score", "confidence", "per_index", "model",
        "prompt_version", "input_hash", "ts", "config_hash", "rationale" }

Everything downstream — the day plan's commit tilt, the meta feature
column, the forge's replay of that session — reads THAT FILE. The live
path and the replay path consume identical bytes, so a backtest of a
news-tilted decision is exact rather than approximate. This is the same
discipline `macro_gex` already follows: compute live, archive the result,
and let the forge replay the archive rather than re-deriving it.

FIVE PROPERTIES THAT KEEP IT HONEST
------------------------------------
1. BOUNDED. `score` is clamped to [-1, +1] and the day plan caps its
   weight at 0.35 (DAYPLAN_NEWS_WEIGHT). It TILTS a ranking among
   candidates that already cleared the conviction bar. It cannot promote
   a candidate that failed the bar and it cannot veto one that passed.
2. FAIL-CLOSED TO NEUTRAL. No model, no network, malformed JSON, timeout
   — every failure path returns 0.0, which is exactly "no opinion". A
   news outage must never stop trading, and must never be mistaken for
   bearishness.
3. PROMPT-VERSIONED. PROMPT_VERSION is part of the record and of the
   cache key. Editing the prompt is a model change: it invalidates the
   stored history rather than silently mixing two different instruments
   in one time series. The same rule CONFIG_HASH enforces everywhere
   else.
4. INPUT-HASHED. The same headlines on the same day produce the same
   cached score without a second call. A re-run cannot quietly redraw
   the number, and a DIFFERENT input set on the same day is visible as a
   hash change rather than silently overwriting.
5. TELEMETRY FIRST. NEWS_ENABLED is False by default and
   NEWS_FEED_META is False. The score is recorded for weeks before it is
   allowed to influence anything, so its IC can be measured on real
   sessions — through the same per-day IC + BH + MDE discipline that
   killed the directional hypothesis in July.

WHAT THE SCORE MEANS
--------------------
+1 strongly supports upside, -1 strongly supports downside, 0 no
directional information. `confidence` is separate and multiplies nothing
automatically — a high-magnitude score with low confidence is exactly the
case where the bounded weight matters most.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, asdict

import config

log = logging.getLogger("news_intel")

# Bump this whenever the prompt changes. It is part of the cache key AND
# the stored record: a prompt edit must invalidate history, not silently
# splice a new instrument into an old time series.
PROMPT_VERSION = "v1"

NEUTRAL = 0.0


@dataclass
class NewsScore:
    day: str
    score: float = NEUTRAL          # [-1, +1]; + = supports upside
    confidence: float = 0.0         # [0, 1]
    per_index: dict = None
    model: str = ""
    prompt_version: str = PROMPT_VERSION
    input_hash: str = ""
    rationale: str = ""
    ts: float = 0.0
    config_hash: str = ""
    source: str = "none"            # live | cache | neutral_fallback

    def __post_init__(self):
        if self.per_index is None:
            self.per_index = {}
        self.score = max(min(float(self.score), 1.0), -1.0)
        self.confidence = max(min(float(self.confidence), 1.0), 0.0)

    def for_index(self, index: str) -> float:
        """Per-index score, falling back to the market-wide one. Bounded
        the same way, because a per-index value from a malformed model
        response must not escape the clamp."""
        try:
            v = float(self.per_index.get(index, self.score))
        except (TypeError, ValueError):
            v = self.score
        return max(min(v, 1.0), -1.0)

    def as_dict(self) -> dict:
        return asdict(self)


def _dir() -> "os.PathLike":
    p = config.STATE_DIR / "news"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _path(day: str):
    return _dir() / f"news_{day}.json"


def input_hash(items) -> str:
    body = json.dumps(sorted(str(x) for x in (items or [])), sort_keys=True)
    return hashlib.sha1(
        (body + "|" + PROMPT_VERSION).encode()).hexdigest()[:12]


def load(day: str | None = None) -> NewsScore:
    """Read the persisted score. THE ONLY read path — live and replay both
    come through here, so they cannot diverge."""
    day = day or dt.date.today().isoformat()
    try:
        p = _path(day)
        if not p.exists():
            return NewsScore(day=day, source="neutral_fallback")
        b = json.loads(p.read_text(encoding="utf-8"))
        if b.get("prompt_version") != PROMPT_VERSION:
            log.info("news %s was scored under prompt %s (now %s) — treated "
                     "as NEUTRAL. A prompt change is a model change; mixing "
                     "them in one series would be mixing two instruments.",
                     day, b.get("prompt_version"), PROMPT_VERSION)
            return NewsScore(day=day, source="neutral_fallback")
        if b.get("config_hash") and b["config_hash"] != config.CONFIG_HASH:
            log.info("news %s was scored under CONFIG_HASH %s — kept, since "
                     "headlines do not depend on the feature world",
                     day, b.get("config_hash"))
        b.pop("source", None)
        return NewsScore(source="cache", **b)
    except Exception as e:                                 # noqa: BLE001
        log.warning("news read failed for %s (%s) — NEUTRAL", day, e)
        return NewsScore(day=day, source="neutral_fallback")


def _persist(s: NewsScore) -> None:
    try:
        p = _path(s.day)
        tmp = p.with_name(f"{p.stem}.{os.getpid()}.tmp.json")
        body = s.as_dict()
        body.pop("source", None)
        tmp.write_text(json.dumps(body, indent=1, default=float),
                       encoding="utf-8")
        os.replace(tmp, p)
        log.info("news %s persisted: score %+.2f conf %.2f (%s) -> %s",
                 s.day, s.score, s.confidence, s.model, p.name)
    except Exception as e:                                 # noqa: BLE001
        log.warning("news persist failed (%s) — the score will not be "
                    "replayable and is therefore NOT usable downstream", e)


def score_day(headlines, day: str | None = None,
              force: bool = False) -> NewsScore:
    """Run the model once for `day` and persist. Total — never raises.

    Called pre-open by the daily chain. If a valid record already exists
    for the same inputs, the model is NOT re-run: a second call must not
    be able to redraw the number.
    """
    day = day or dt.date.today().isoformat()
    ih = input_hash(headlines)
    if not force:
        cur = load(day)
        if cur.source == "cache" and cur.input_hash == ih:
            log.info("news %s already scored for these inputs (%s) — reusing",
                     day, ih)
            return cur

    if not bool(getattr(config, "NEWS_ENABLED", False)):
        log.info("NEWS_ENABLED is False — recording NEUTRAL. The score is "
                 "collected as telemetry for weeks before it is allowed to "
                 "influence anything, so its IC can be measured first.")
        s = NewsScore(day=day, score=NEUTRAL, input_hash=ih,
                      ts=time.time(), config_hash=config.CONFIG_HASH,
                      model="disabled", source="neutral_fallback")
        _persist(s)
        return s

    raw = None
    model = str(getattr(config, "NEWS_MODEL", "gemma"))
    try:
        from tools import gemma_analyst as GA           # the house LLM path
        fn = getattr(GA, "score_news", None)
        if fn is None:
            log.warning("tools.gemma_analyst has no score_news() — NEUTRAL. "
                        "Wire it there rather than opening a second LLM "
                        "path; one call site is one thing to audit.")
        else:
            raw = fn(headlines, prompt_version=PROMPT_VERSION)
    except Exception as e:                                 # noqa: BLE001
        log.warning("news model call failed (%s) — NEUTRAL. A news outage "
                    "must never stop trading and must never be mistaken "
                    "for bearishness.", e)

    if not isinstance(raw, dict):
        s = NewsScore(day=day, score=NEUTRAL, input_hash=ih, ts=time.time(),
                      config_hash=config.CONFIG_HASH, model=model,
                      source="neutral_fallback")
        _persist(s)
        return s

    s = NewsScore(day=day, score=float(raw.get("score") or 0.0),
                  confidence=float(raw.get("confidence") or 0.0),
                  per_index=dict(raw.get("per_index") or {}),
                  model=model, input_hash=ih,
                  rationale=str(raw.get("rationale") or "")[:600],
                  ts=time.time(), config_hash=config.CONFIG_HASH,
                  source="live")
    _persist(s)
    return s


def tilt_for(index: str, day: str | None = None) -> float:
    """What the day plan should use. Zero unless news is BOTH enabled and
    allowed to influence the commit — two switches, because recording a
    score and acting on it are different decisions."""
    if not bool(getattr(config, "NEWS_ENABLED", False)):
        return 0.0
    if not bool(getattr(config, "NEWS_TILT_COMMIT", False)):
        return 0.0
    return load(day).for_index(index)


def meta_feature(index: str, day: str | None = None) -> float:
    """The feature column for the meta model.

    Gated separately from the commit tilt: a score good enough to record
    is not automatically good enough to train on. Adding it to X changes
    the feature world, so NEWS_FEED_META must also rotate CONFIG_HASH —
    which is why, unlike the other news constants, it is NOT in
    _HASH_EXCLUDE.
    """
    if not bool(getattr(config, "NEWS_FEED_META", False)):
        return 0.0
    # ENFORCED INVALIDATION. A news column changes the feature world, so the
    # day caches and the trained model must be rebuilt — but the flag itself
    # is hash-excluded (declaring it at False would otherwise have rotated
    # the hash and forced a 12-hour rebuild for a switch nobody turned on).
    # So the rebuild is enforced here instead of hoped for: the column is
    # refused until FEATURE_WORLD carries the matching marker, which IS
    # hash-included, so setting it rotates the hash exactly once, when the
    # column actually enters X.
    want = str(getattr(config, "NEWS_FEED_META_WORLD", "fw_news_v1"))
    have = str(getattr(config, "FEATURE_WORLD", ""))
    if want not in have:
        log.error("NEWS_FEED_META is True but FEATURE_WORLD (%r) does not "
                  "carry %r. A news column changes X; serving it without "
                  "rotating CONFIG_HASH would train the meta on a feature "
                  "the day caches were never built with — train/serve skew "
                  "by construction. Returning 0.0.", have, want)
        return 0.0
    return load(day).for_index(index)


def history(days: int = 60) -> list[NewsScore]:
    """Every persisted score, newest last — the series an IC study reads."""
    out = []
    try:
        for p in sorted(_dir().glob("news_*.json"))[-int(days):]:
            d = p.stem.replace("news_", "")
            s = load(d)
            if s.source == "cache":
                out.append(s)
    except Exception as e:                                 # noqa: BLE001
        log.debug("news history failed (%s)", e)
    return out