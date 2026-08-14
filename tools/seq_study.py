"""
SEQ STUDY — train the CNN-GRU on the vault, and judge it against controls
=========================================================================
    python tools/seq_study.py [--json out.json]

Wired into both chains. Reads the payoff matrix, builds the approach
windows, and runs three things in order:

  1. the real model, out-of-fold on purged day-folds with an embargo
  2. the SHUFFLED-LABEL control — the same pipeline on labels permuted
     within each session. This vault's overfitting floor.
  3. the HELD-OUT MONTH — sessions removed before anything was fitted.

The verdict reads the control line FIRST. An IC of +0.20 against a control
of +0.19 is a measurement of the pipeline's own capacity, not of the
market, and at n≈141 episodes that is the likelier outcome.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
config.setup_logging("seq_study")
import logging
log = logging.getLogger("seq_study")
from core import seq_model as SM
from core.episode_ranker import build_episodes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, default="")
    a = ap.parse_args()

    mp = config.STATE_DIR / "meta_train_matrix.npz"
    if not mp.exists():
        log.info("no payoff matrix at %s — self-arms when the forge "
                 "publishes it", mp)
        return 0
    try:
        z = np.load(mp, allow_pickle=False)
        ch = str(z["config_hash"][0]) if "config_hash" in z.files else ""
        if ch and ch != config.CONFIG_HASH:
            log.warning("matrix built under CONFIG_HASH %s != %s — stopping",
                        ch, config.CONFIG_HASH)
            return 0
        X = np.asarray(z["X"], float)
        r = np.asarray(z["ret"], float) / np.where(
            np.asarray(z["risk"], float) > 0,
            np.asarray(z["risk"], float), np.nan)
        days = np.array([str(d) for d in z["day"]])
        ts = (np.asarray(z["t"], float) if "t" in z.files
              else np.arange(len(r), dtype=float))
    except Exception as e:                                 # noqa: BLE001
        log.error("could not read the payoff matrix (%s)", e)
        return 0

    ok = np.isfinite(r)
    keep = np.where(ok)[0]
    # pass the ORIGINAL row indices through, so every episode knows which
    # matrix row it came from and the window join below is explicit
    eps, st = build_episodes(days[ok], ts[ok], r[ok], X[ok], rows=keep)
    if len(eps) < SM.MIN_EPISODES:
        log.info("EVIDENCE: %d/%d episode(s) — not yet measurable",
                 len(eps), SM.MIN_EPISODES)
        return 0

    # The approach window. The published matrix is snapshot features, so
    # until the forge emits per-second tape the "sequence" is a single
    # step and the CNN-GRU degenerates to an MLP. Said plainly rather than
    # dressed up: this is the honest current limit.
    Xg = np.vstack([e.x for e in eps])
    if "seq" in z.files:
        seq_all = np.asarray(z["seq"], np.float32)
        idx = np.array([e.row for e in eps], int)
        if idx.max(initial=-1) >= seq_all.shape[0]:
            log.error("window array has %d row(s) but an episode points at "
                      "row %d — the forge published seq and X out of step. "
                      "Refusing rather than joining windows to the wrong "
                      "labels.", seq_all.shape[0], int(idx.max()))
            return 0
        Xs = seq_all[idx]            # EXPLICIT join, never positional
        # assert the join is sane: the snapshot we carry must equal the
        # snapshot at that row. Cheap, and it catches a mis-publication
        # before any number is computed.
        if not np.allclose(np.nan_to_num(X[idx]), np.nan_to_num(Xg),
                           atol=1e-5):
            log.error("row-index join failed its own consistency check — "
                      "X[row] does not match the episode's stored features. "
                      "Stopping.")
            return 0
        live = float(np.mean(np.abs(Xs).sum(axis=(1, 2)) > 0))
        ch = ([str(c) for c in z["seq_channels"]]
              if "seq_channels" in z.files else [])
        log.info("approach windows: %s over %d step(s), %d channel(s) %s | "
                 "%.0f%% non-empty", Xs.shape, Xs.shape[1], Xs.shape[2],
                 ch, 100 * live)
        if live < 0.5:
            log.warning("more than half the windows are EMPTY — most signals "
                        "fire before enough tape exists. The sequence model "
                        "would be fitted on the later-session subset, which "
                        "is not the population the snapshot model sees.")
    else:
        Xs = Xg[:, None, :]
        log.warning("the matrix carries SNAPSHOTS, not tape. With a 1-step "
                    "window the CNN-GRU is an MLP and its whole advantage — "
                    "reading the APPROACH — is unavailable. Re-run the forge "
                    "with the window publication in place.")
    data = SM.SeqData(Xs, Xg, np.array([e.r for e in eps], float),
                      np.array([e.day for e in eps]))

    v = SM.evaluate(data, SM.make_learner,
                    n_control=int(getattr(config, "SEQ_CONTROL_RUNS", 5)))
    SM.report(v, log)

    out = {"ts": time.time(), "config_hash": config.CONFIG_HASH,
           "episodes": len(eps), "build_stats": st, "verdict": v}
    p = Path(a.json) if a.json else (
        config.LOG_DIR / f"seq_study_{time.strftime('%Y-%m-%d')}.json")
    try:
        p.write_text(json.dumps(out, indent=1, default=float))
        log.info("report → %s", p)
    except Exception as e:                                 # noqa: BLE001
        log.warning("report write failed (%s)", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())