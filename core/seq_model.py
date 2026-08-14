"""
SEQUENCE MODEL — a CNN-GRU on the approach, and the controls that judge it
==========================================================================
WHAT IS ACTUALLY NEW HERE
--------------------------
Every model this system has fitted so far reads a hand-summarised SNAPSHOT:
sixty-odd numbers describing the instant a signal fired. A sequence encoder
reads the APPROACH — the shape of the last few minutes of tape leading into
that instant. Two signals with identical snapshots can arrive on completely
different paths (a grind up versus a spike and fade), and the snapshot
throws that away by construction.

That is the one place a neural network has a real advantage over the
gradient-boosted trees already in the stack, and it is an advantage in
REPRESENTATION, not in capacity. It does not require more sessions to
exploit — which matters, because more sessions are exactly what this vault
does not have.

  Conv1d (dilated, causal)  local shape: bursts, compressions, reversals
  GRU                       longer dependency across the window
  attention pool            which part of the approach mattered
  three heads               rank / magnitude / direction, shared trunk

Multi-task on purpose: one target on 141 episodes is a thin gradient
signal. Three related targets over a shared trunk extract more from the
same sample, which is the standard remedy when n is fixed and small.

THE HONEST PART: WHY THIS PROBABLY STILL FAILS, AND HOW YOU WILL KNOW
----------------------------------------------------------------------
n_eff on this vault is 146-476. A CNN-GRU has tens of thousands of
parameters. It will fit the training folds beautifully and that will mean
nothing. So the architecture is not the deliverable — the CONTROLS are:

1. SHUFFLED-LABEL CONTROL. The identical pipeline, identical folds,
   identical early stopping, trained on labels permuted WITHIN each session
   (preserving day structure and the per-day label distribution, destroying
   only the signal→outcome link). Whatever it scores is this pipeline's
   overfitting floor on this sample. If the real model scores 0.58 and the
   shuffled control scores 0.57, the honest reading is that 0.57 of that is
   the machinery flattering itself.

   Nothing else in this system does this, and at n=141 it is the single
   most informative number available.

2. HELD-OUT MONTH. The most recent SEQ_HOLDOUT_SESSIONS sessions are
   removed before anything is fitted — not a CV fold, not touched by
   feature selection, not touched by early stopping. The model must beat
   the incumbent THERE, on days it has never influenced in any way. This
   is what the operator asked for and it is the right bar: a first
   clearance on cross-validation is when a holdout is worth more than a
   deployment.

3. THE SAME GATES AS EVERYTHING ELSE. Purged day-folds with embargo,
   day-clustered significance, explicit MDE, and refusal by default.

PLUGGABLE LEARNER, ON PURPOSE
------------------------------
The discipline above is independent of the model class, so it is written
against a Learner interface. The default is the torch CNN-GRU (CUDA when
available — this runs on an RTX 4060). When torch is absent the pipeline
falls back to a ridge learner and SAYS SO: a fallback that silently
pretends to be the deep model would make every number here a lie.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

import config

log = logging.getLogger("seq_model")

WINDOW_S = 300          # seconds of approach fed to the encoder
MIN_SESSIONS = 25
MIN_EPISODES = 120


# ------------------------------------------------------------- learners
class Learner:
    """Fit on (Xseq, Xg, y) and score. One interface, two implementations,
    so the controls below judge the DISCIPLINE regardless of model class."""

    name = "base"

    def fit(self, Xs, Xg, y, days, seed: int = 0):
        raise NotImplementedError

    def score(self, Xs, Xg, mc: int | None = None):
        raise NotImplementedError


class RidgeLearner(Learner):
    """Fallback. Mean-pools the sequence and fits a ridge — genuinely weak,
    and named so in every report. It exists so the fold logic, the shuffled
    control and the holdout gate remain testable without torch."""

    name = "ridge_fallback"

    def __init__(self, lam: float = 10.0):
        self.lam, self.w = lam, None

    @staticmethod
    def _flat(Xs, Xg):
        a = np.nanmean(Xs, axis=1)
        b = np.nanstd(Xs, axis=1)
        last = Xs[:, -1, :]
        z = np.hstack([a, b, last, Xg])
        return np.nan_to_num(np.hstack([z, np.ones((len(z), 1))]))

    def fit(self, Xs, Xg, y, days, seed: int = 0):
        Z = self._flat(Xs, Xg)
        A = Z.T @ Z + self.lam * np.eye(Z.shape[1])
        try:
            self.w = np.linalg.solve(A, Z.T @ y)
        except np.linalg.LinAlgError:
            self.w = np.zeros(Z.shape[1])
        return self

    def score(self, Xs, Xg, mc: int | None = None):
        s = self._flat(Xs, Xg) @ self.w
        # a linear fit has no dropout and therefore no epistemic spread to
        # report. Returning zeros would CLAIM certainty it has not earned,
        # so it returns NaN and the caller must treat it as "unknown".
        return (s, np.full_like(s, np.nan)) if mc else s


class CNNGRULearner(Learner):
    """Dilated causal CNN -> GRU -> attention pool -> three heads.

    Small on purpose. At 141 episodes the binding risk is memorisation, so
    the width is chosen to keep the parameter count in the low tens of
    thousands and dropout is heavy. The dilations (1,2,4) give a receptive
    field that spans the window without stacking depth.
    """

    name = "cnn_gru"

    def __init__(self, d: int = 24, dropout: float = 0.3, epochs: int = 120,
                 lr: float = 2e-3, patience: int = 15, ensemble: int = 5):
        self.d, self.dropout, self.epochs = d, dropout, epochs
        self.lr, self.patience, self.ensemble = lr, patience, ensemble
        self.models, self.dev = [], "cpu"

    def _build(self, n_ch: int, n_g: int):
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s, d, n_ch, n_g, p):
                super().__init__()
                s.conv = nn.Sequential(
                    nn.Conv1d(n_ch, d, 5, padding=4, dilation=1), nn.GELU(),
                    nn.Conv1d(d, d, 5, padding=8, dilation=2), nn.GELU(),
                    nn.Conv1d(d, d, 5, padding=16, dilation=4), nn.GELU())
                s.drop = nn.Dropout(p)
                s.gru = nn.GRU(d, d, batch_first=True)
                s.att = nn.Linear(d, 1)
                s.trunk = nn.Sequential(nn.Linear(d + n_g, d), nn.GELU(),
                                        nn.Dropout(p))
                s.h_rank = nn.Linear(d, 1)
                s.h_mag = nn.Linear(d, 1)
                s.h_dir = nn.Linear(d, 1)

            def forward(s, xs, xg):
                z = s.conv(xs.transpose(1, 2))
                z = z[:, :, :xs.shape[1]].transpose(1, 2)   # causal trim
                z = s.drop(z)
                h, _ = s.gru(z)
                a = torch.softmax(s.att(h).squeeze(-1), dim=1).unsqueeze(-1)
                pooled = (h * a).sum(1)
                t = s.trunk(torch.cat([pooled, xg], dim=1))
                return s.h_rank(t).squeeze(-1), s.h_mag(t).squeeze(-1), \
                    s.h_dir(t).squeeze(-1)

        return Net(self.d, n_ch, n_g, self.dropout)

    def fit(self, Xs, Xg, y, days, seed: int = 0):
        import torch
        import torch.nn as nn
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        xs = torch.tensor(np.nan_to_num(Xs), dtype=torch.float32,
                          device=self.dev)
        xg = torch.tensor(np.nan_to_num(Xg), dtype=torch.float32,
                          device=self.dev)
        yt = torch.tensor(y, dtype=torch.float32, device=self.dev)
        sgn = (yt > 0).float()
        # a small internal split for early stopping — BY DAY, never by row,
        # or adjacent seconds land on both sides and stopping is meaningless
        uniq = sorted(set(days.tolist()))
        rng = np.random.default_rng(seed)
        vald = set(rng.choice(uniq, max(2, len(uniq) // 6), replace=False))
        vm = torch.tensor(np.array([d in vald for d in days]),
                          device=self.dev)
        tm = ~vm
        self.models = []
        for k in range(self.ensemble):
            torch.manual_seed(seed * 100 + k)
            net = self._build(Xs.shape[2], Xg.shape[1]).to(self.dev)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr,
                                    weight_decay=1e-2)
            best, bad, best_state = 1e18, 0, None
            for _ in range(self.epochs):
                net.train()
                opt.zero_grad()
                r, m, dh = net(xs[tm], xg[tm])
                loss = (_pairwise_rank(r, yt[tm], days[tm.cpu().numpy()])
                        + nn.functional.huber_loss(m, yt[tm])
                        + nn.functional.binary_cross_entropy_with_logits(
                            dh, sgn[tm]))
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                net.eval()
                with torch.no_grad():
                    rv, mv, _ = net(xs[vm], xg[vm])
                    v = float(nn.functional.huber_loss(mv, yt[vm]))
                if v < best - 1e-5:
                    best, bad = v, 0
                    best_state = {kk: vv.detach().clone()
                                  for kk, vv in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
            if best_state:
                net.load_state_dict(best_state)
            self.models.append(net)
        return self

    def score(self, Xs, Xg, mc: int | None = None
              ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Mean score, and — when `mc` > 0 — the EPISTEMIC spread.

        Ensemble variance answers "do differently-seeded fits agree?".
        MC-dropout answers a different question: "is THIS input in a region
        the network is confident about?" Dropout is left ON at inference and
        the forward pass repeated; the spread of those passes is the model's
        own uncertainty about that specific sample (Gal & Ghahramani 2016).

        This is the piece that was missing when the promoted meta served
        0.204 for 14 655 evaluations. A near-constant point estimate and a
        genuine "I have no opinion here" are indistinguishable in a single
        number, and the gate treated the first as if it were information.
        With a spread the gate can refuse an uncertain sample instead of
        acting on a confident-looking constant.
        """
        import torch
        xs = torch.tensor(np.nan_to_num(Xs), dtype=torch.float32,
                          device=self.dev)
        xg = torch.tensor(np.nan_to_num(Xg), dtype=torch.float32,
                          device=self.dev)
        passes = []
        n_mc = int(mc or 0)
        for net in self.models:
            if n_mc > 0:
                net.train()          # dropout ON — the whole point
                for m in net.modules():
                    if isinstance(m, torch.nn.GRU):
                        m.eval()     # ...but not the recurrent state
                with torch.no_grad():
                    for _ in range(n_mc):
                        r, _, _ = net(xs, xg)
                        passes.append(r.cpu().numpy())
            else:
                net.eval()
                with torch.no_grad():
                    r, _, _ = net(xs, xg)
                passes.append(r.cpu().numpy())
        A = np.stack(passes)
        return (A.mean(0), A.std(0)) if n_mc > 0 else A.mean(0)


def _pairwise_rank(scores, y, days):
    """Within-session pairwise logistic ranking loss.

    Within-session because that is the comparison the book makes: one slot,
    today's candidates. Comparing a Tuesday signal to a Thursday one would
    teach an ordering the book never uses and let a regime shift masquerade
    as signal.
    """
    import torch
    total, n = 0.0, 0
    for d in np.unique(days):
        m = np.where(days == d)[0]
        if m.size < 2:
            continue
        idx = torch.tensor(m, device=scores.device)
        s, t = scores[idx], y[idx]
        ds = s.unsqueeze(0) - s.unsqueeze(1)
        dt = t.unsqueeze(0) - t.unsqueeze(1)
        w = (dt.abs() > 1e-9).float()
        if float(w.sum()) == 0:
            continue
        lab = (dt > 0).float()
        total = total + (torch.nn.functional.
                         binary_cross_entropy_with_logits(
                             ds, lab, weight=w, reduction="sum"))
        n += int(w.sum())
    return total / max(n, 1)


_TORCH_WARNED = {"n": 0}


def make_learner() -> Learner:
    try:
        import torch                                       # noqa: F401
        return CNNGRULearner()
    except Exception:                                      # noqa: BLE001
        # ONCE. make_learner() is called per fold and per control run — 27
        # times in a 5-fold, 4-control study — and an un-throttled warning
        # buries the report it is warning about.
        _TORCH_WARNED["n"] += 1
        if _TORCH_WARNED["n"] > 1:
            return RidgeLearner()
        log.warning("torch is not installed — falling back to a RIDGE "
                    "learner. Every number below describes THAT model, not "
                    "a CNN-GRU. A fallback that silently claimed to be the "
                    "deep model would make this whole report a lie. "
                    "`pip install torch` to run the real thing on the 4060.")
        return RidgeLearner()


# ------------------------------------------------------------ the study
@dataclass
class SeqData:
    Xs: np.ndarray          # (n, T, C) approach windows
    Xg: np.ndarray          # (n, G) snapshot context
    y: np.ndarray           # R = P&L / risk
    days: np.ndarray
    feat: list = field(default_factory=list)


def _folds(uniq, n_fold, embargo):
    out = []
    for i in range(n_fold):
        te = set(uniq[i::n_fold])
        emb = set()
        for j, d in enumerate(uniq):
            if d in te:
                for o in range(-embargo, embargo + 1):
                    if 0 <= j + o < len(uniq):
                        emb.add(uniq[j + o])
        out.append((te, emb))
    return out


def run_cv(data: SeqData, learner_fn, n_fold: int = 5, embargo: int = 1,
           seed: int = 0) -> np.ndarray:
    """Out-of-fold scores under purged day-folds."""
    uniq = sorted(set(data.days.tolist()))
    oof = np.full(len(data.y), np.nan)
    for te, emb in _folds(uniq, n_fold, embargo):
        tem = np.array([d in te for d in data.days])
        trm = np.array([d not in emb for d in data.days])
        if trm.sum() < max(20, len(data.y) // 4) or tem.sum() < 4:
            continue
        lr = learner_fn()
        lr.fit(data.Xs[trm], data.Xg[trm], data.y[trm], data.days[trm],
               seed=seed)
        oof[tem] = lr.score(data.Xs[tem], data.Xg[tem])
    return oof


def shuffle_within_day(y: np.ndarray, days: np.ndarray,
                       seed: int = 0) -> np.ndarray:
    """Permute labels WITHIN each session.

    Within-day, not global: it preserves the day structure and each
    session's label distribution, and destroys only the signal→outcome
    link. A global shuffle would also destroy between-day variation and
    make the control easier than the real problem, which would understate
    the overfitting floor — the opposite of what a control is for.
    """
    rng = np.random.default_rng(seed)
    out = y.copy()
    for d in np.unique(days):
        m = np.where(days == d)[0]
        if m.size > 1:
            out[m] = y[rng.permutation(m)]
    return out


def deflated_sharpe(returns: np.ndarray, n_trials: int) -> dict:
    """Bailey & Lopez de Prado (2014) deflated Sharpe ratio.

    Selecting the best of N trials inflates the winner's Sharpe even when
    every trial is noise. This system has run 25 pre-registered trials
    (forge_report.trials_for_deflation) and every study added here raises
    that count — so an undeflated Sharpe on the holdout would flatter the
    LAST thing tried simply because it was tried last.

    Returns the probability the true Sharpe exceeds zero AFTER accounting
    for selection, skew and kurtosis.
    """
    from math import erf, sqrt, log
    x = np.asarray(returns, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8 or np.std(x, ddof=1) == 0:
        return {"sr": float("nan"), "dsr": float("nan"), "n": int(n)}
    sr = float(np.mean(x) / np.std(x, ddof=1))
    g1 = float(np.mean((x - x.mean()) ** 3) / (np.std(x) ** 3 + 1e-12))
    g2 = float(np.mean((x - x.mean()) ** 4) / (np.std(x) ** 4 + 1e-12))
    N = max(int(n_trials), 1)
    # expected max Sharpe under the null across N independent trials
    e = 0.5772156649
    z1 = _ppf(1 - 1.0 / max(N, 2))
    z2 = _ppf(1 - 1.0 / (max(N, 2) * 2.718281828))
    sr0 = (1 - e) * z1 + e * z2
    denom = sqrt(max(1e-12, 1 - g1 * sr + (g2 - 1) / 4.0 * sr * sr))
    stat = (sr - sr0) * sqrt(max(n - 1, 1)) / denom
    dsr = 0.5 * (1 + erf(stat / sqrt(2)))
    return {"sr": round(sr, 4), "sr0_expected_max": round(float(sr0), 4),
            "dsr": round(float(dsr), 4), "n": int(n), "n_trials": N,
            "skew": round(g1, 3), "kurt": round(g2, 3)}


def _ppf(q: float) -> float:
    """Inverse normal CDF (Acklam). Good to ~1e-9, no scipy dependency."""
    from math import sqrt, log
    q = min(max(float(q), 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    pl, ph = 0.02425, 1 - 0.02425
    if q < pl:
        t = sqrt(-2 * log(q))
        return (((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5]) / \
               ((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1)
    if q > ph:
        t = sqrt(-2 * log(1 - q))
        return -(((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5]) / \
               ((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1)
    t = q - 0.5
    r = t * t
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*t / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _per_day_ic(oof, y, days) -> dict:
    out = {}
    for d in np.unique(days):
        m = (days == d) & ~np.isnan(oof)
        if m.sum() >= 3 and np.std(oof[m]) > 0 and np.std(y[m]) > 0:
            out[str(d)] = float(np.corrcoef(
                np.argsort(np.argsort(oof[m])).astype(float),
                np.argsort(np.argsort(y[m])).astype(float))[0, 1])
    return out


def evaluate(data: SeqData, learner_fn, n_control: int = 5,
             seed: int = 0) -> dict:
    """Real model, shuffled control, and the holdout month."""
    from core import capability_ladder as CL

    uniq = sorted(set(data.days.tolist()))
    n_hold = int(getattr(config, "SEQ_HOLDOUT_SESSIONS", 22))
    if len(uniq) < MIN_SESSIONS + 5:
        return {"ok": False, "reason": f"{len(uniq)} session(s) — the "
                                       f"holdout alone wants {n_hold}"}
    # ADAPTIVE HOLDOUT. A fixed 22-session holdout on a 34-session vault
    # leaves 12 to fit on, the purged folds then gut the training sets, and
    # the study reports "too few scored sessions" — a starved fit dressed
    # up as a data problem. The holdout SHRINKS rather than starving the
    # development set, and the report states the size it actually got so a
    # thin holdout is never mistaken for a strong one.
    n_hold_eff = int(max(0, min(n_hold, len(uniq) - MIN_SESSIONS)))
    hold = set(uniq[-n_hold_eff:]) if n_hold_eff >= 5 else set()
    if n_hold_eff < n_hold:
        log.warning("holdout reduced %d -> %d session(s): %d in the vault "
                    "and the fit needs %d. A holdout this size is a weak "
                    "test, not a strong one.", n_hold, n_hold_eff,
                    len(uniq), MIN_SESSIONS)
    devm = np.array([d not in hold for d in data.days])
    dev = SeqData(data.Xs[devm], data.Xg[devm], data.y[devm],
                  data.days[devm], data.feat)

    oof = run_cv(dev, learner_fn, seed=seed)
    ic_real = _per_day_ic(oof, dev.y, dev.days)
    if len(ic_real) < 5:
        return {"ok": False, "reason": "too few scored sessions"}
    st_real = CL.paired_test(ic_real)

    # ---- the control: identical pipeline, permuted labels
    ctrl = []
    for c in range(int(n_control)):
        ys = shuffle_within_day(dev.y, dev.days, seed=1000 + c)
        d2 = SeqData(dev.Xs, dev.Xg, ys, dev.days, dev.feat)
        o2 = run_cv(d2, learner_fn, seed=seed)
        ic2 = _per_day_ic(o2, ys, d2.days)
        if ic2:
            ctrl.append(float(np.mean(list(ic2.values()))))
    ctrl_mean = float(np.mean(ctrl)) if ctrl else 0.0
    ctrl_sd = float(np.std(ctrl)) if len(ctrl) > 1 else 0.0
    margin = st_real["mean"] - ctrl_mean
    # z of the real IC against the SHUFFLED distribution — the honest
    # comparison. Against zero is the wrong null: this pipeline does not
    # score zero on noise.
    z_ctrl = (margin / ctrl_sd) if ctrl_sd > 1e-9 else float("nan")

    # ---- the holdout month: fitted on dev only, judged on days never seen
    hold_ic, hold_n = float("nan"), 0
    mc_mean = mc_frac = float("nan")
    dsr = {"sr": float("nan"), "dsr": float("nan"), "n": 0}
    if hold:
        lr = learner_fn()
        lr.fit(dev.Xs, dev.Xg, dev.y, dev.days, seed=seed)
        hm = ~devm
        _mc = int(getattr(config, "SEQ_MC_PASSES", 20))
        _out = lr.score(data.Xs[hm], data.Xg[hm], mc=_mc)
        sc, unc = _out if isinstance(_out, tuple) else (_out, None)
        hic = _per_day_ic(sc, data.y[hm], data.days[hm])
        hold_n = len(hic)
        if hic:
            hold_ic = float(np.mean(list(hic.values())))
        # EPISTEMIC SPREAD. NaN from the ridge fallback is honest: a linear
        # fit has no dropout and therefore no uncertainty to report.
        if unc is not None and np.isfinite(unc).any():
            mc_mean = float(np.nanmean(unc))
            mc_frac = float(np.nanmean(unc > np.nanquantile(unc, 0.75)))
        else:
            mc_mean, mc_frac = float("nan"), float("nan")
        # DEFLATED SHARPE on the holdout, in the book's own units: the R of
        # the top-ranked episode per session, which is what one slot buys.
        top_r = []
        for d0 in np.unique(data.days[hm]):
            m0 = data.days[hm] == d0
            if m0.sum() >= 2:
                top_r.append(float(data.y[hm][m0][int(np.argmax(sc[m0]))]))
        dsr = deflated_sharpe(np.asarray(top_r),
                              int(getattr(config, "SEQ_N_TRIALS", 25)))

    checks = {
        "ic_positive": st_real["mean"] > 0,
        "ic_above_mde": st_real["mean"] > float(st_real.get(
            "mde", float("inf"))),
        "beats_shuffled_control": margin > 0,
        "margin_exceeds_control_noise": bool(
            np.isfinite(z_ctrl) and z_ctrl >= 2.0),
        "holdout_positive": bool(np.isfinite(hold_ic) and hold_ic > 0),
        "holdout_scored": hold_n >= 5,
        # Deflated for the 25+ pre-registered trials this system has already
        # run. An undeflated Sharpe would flatter whatever was tried LAST.
        "deflated_sharpe_clears": bool(
            np.isfinite(dsr.get("dsr", float("nan")))
            and dsr["dsr"] >= float(getattr(config, "SEQ_MIN_DSR", 0.95))),
    }
    return {"ok": all(checks.values()),
            "learner": learner_fn().name,
            "n_dev_sessions": len(ic_real), "n_episodes": int(len(dev.y)),
            "ic_real": st_real["mean"], "ic_p": st_real.get("p", 1.0),
            "ic_mde": st_real.get("mde", float("nan")),
            "ctrl_mean": ctrl_mean, "ctrl_sd": ctrl_sd,
            "margin_over_control": margin, "z_vs_control": z_ctrl,
            "holdout_ic": hold_ic, "holdout_sessions": hold_n,
            "holdout_days": sorted(hold),
            "holdout_requested": int(n_hold),
            "mc_uncertainty_mean": mc_mean, "mc_high_frac": mc_frac,
            "deflated": dsr, "checks": checks,
            "config_hash": config.CONFIG_HASH}


def report(v: dict, logger=None) -> None:
    lg = logger or log
    if "ic_real" not in v:
        lg.info("sequence model: %s", v.get("reason"))
        return
    lg.info("SEQUENCE MODEL (%s) | %d episode(s), %d dev session(s)",
            v["learner"], v["n_episodes"], v["n_dev_sessions"])
    if v.get("learner") == "ridge_fallback":
        lg.warning("  THIS IS THE FALLBACK. It MEAN-POOLS the window, which "
                   "destroys the temporal shape the CNN-GRU exists to read — "
                   "so a NEGATIVE result here is uninformative about the "
                   "real model. Install torch before drawing any conclusion.")
    lg.info("  within-session IC (real)     %+.4f | p %.4f | MDE %.4f",
            v["ic_real"], v["ic_p"], v["ic_mde"])
    lg.info("  SHUFFLED-LABEL CONTROL       %+.4f +/- %.4f  <- this "
            "pipeline's overfitting floor on this sample",
            v["ctrl_mean"], v["ctrl_sd"])
    lg.info("  margin over control          %+.4f  (z = %.2f)",
            v["margin_over_control"], v["z_vs_control"])
    lg.info("  HELD-OUT MONTH IC            %+.4f over %d session(s) never "
            "touched by fitting, folds or early stopping",
            v["holdout_ic"], v["holdout_sessions"])
    d = v.get("deflated", {})
    lg.info("  DEFLATED SHARPE (top-1 R)    SR %s vs expected-max-under-null "
            "%s over %s trial(s) -> DSR %s",
            d.get("sr"), d.get("sr0_expected_max"), d.get("n_trials"),
            d.get("dsr"))
    if np.isfinite(v.get("mc_uncertainty_mean", float("nan"))):
        lg.info("  MC-DROPOUT spread            %.4f mean | %.0f%% of holdout "
                "samples in the top uncertainty quartile — the gate can "
                "refuse THOSE instead of acting on a confident-looking "
                "constant", v["mc_uncertainty_mean"],
                100 * v.get("mc_high_frac", float("nan")))
    else:
        lg.info("  MC-DROPOUT spread            n/a (the ridge fallback has "
                "no dropout; reporting 0 would claim certainty it has not "
                "earned)")
    for k, ok in v.get("checks", {}).items():
        lg.info("  %-30s %s", k, "PASS" if ok else "FAIL")
    if not v.get("ok"):
        lg.info("NOT PROMOTED. Read the control line before the IC line: a "
                "model that scores +0.20 against a control of +0.19 has "
                "measured its own capacity, not the market.")