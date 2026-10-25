"""Session-level detectors for the v2 protocol.

Every detector maps a session to a sequence of online scores s_1..s_T, where
s_t may depend only on exchanges 1..t (causal); the session score is
max_t s_t, so an alarm at threshold theta fires at the first t with
s_t >= theta.

Per-exchange evidence (computed upstream):
  g_user[t], g_resp[t]  stateless Llama Guard 3 scores (prompt / response)
  g_ctx[t]              context-aware Llama Guard 3 score (full history)
  S1[t], S2[t], S3[t]   CRA reference signals (run_cranet.py definitions)
  g_goal[t], g_goalresp[t]  guard scores of the goal reconstructed from
                        exchanges 1..t, alone and with the latest reply
                        (run_goal_features.py; only in --goal evaluations)
  g_seq[t]              guard score of the concatenated user turns 1..t
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHA, BETA, GAMMA = 0.35, 0.45, 0.20   # published CRA fusion weights
EWMA_LAMBDA = 0.5                       # accumulator smoothing
CUSUM_REF = 0.5                         # reference level for the excess sum

CHANNELS = {
    "L": ["g_user", "g_resp"],          # local (single-exchange) guard evidence
    "C": ["S1", "S2", "S3"],            # CRA reference state signals
    "X": ["g_ctx"],                     # context-aware guard evidence
    "G": ["g_goal", "g_goalresp"],      # goal-inference evidence
}


def evidence(rec: dict, groups: str) -> np.ndarray:
    """(T, d) matrix of the requested evidence channels."""
    cols = [c for g in groups for c in CHANNELS[g]]
    return np.array([rec[c] for c in cols], dtype=float).T


# ─── fixed (training-free) detectors ─────────────────────────────────────────

def score_length(rec: dict) -> np.ndarray:
    return np.arange(1, rec["T"] + 1, dtype=float)


def score_convex(rec: dict) -> np.ndarray:
    return ALPHA * np.array(rec["S1"]) + BETA * np.array(rec["S2"]) + GAMMA * np.array(rec["S3"])


def score_guard_turn(rec: dict) -> np.ndarray:
    return np.maximum(rec["g_user"], rec["g_resp"])


def score_guard_resp(rec: dict) -> np.ndarray:
    return np.array(rec["g_resp"], dtype=float)


def score_guard_ctx(rec: dict) -> np.ndarray:
    return np.array(rec["g_ctx"], dtype=float)


def score_thrd(rec: dict) -> np.ndarray:
    """THRD (Ma et al., 2026, arXiv:2606.01738): Temporal Hierarchical Risk
    Detection combining turn-level risk R_t = 1 + 4 g_t^u in [1, 5],
    historical context risk J_h = g_t^c, and response risk J_r = g_t^r via
    F_t = R_t + (1 - (R_t - 1)/4) * sqrt(0.5 * J_h^2 + 0.5 * J_r^2) plus a
    monotonic 3-turn escalation bonus delta = 0.25."""
    gu = np.asarray(rec["g_user"], dtype=float)
    gr = np.asarray(rec["g_resp"], dtype=float)
    gc = np.asarray(rec["g_ctx"], dtype=float)
    R = 1.0 + 4.0 * gu
    lam = 1.0 - (R - 1.0) / 4.0
    omega = np.sqrt(0.5 * gc ** 2 + 0.5 * gr ** 2)
    F = R + lam * omega
    F_adj = F.copy()
    for t in range(2, len(F)):
        if F[t - 2] <= F[t - 1] <= F[t]:
            F_adj[t] += 0.25
    return (F_adj - 1.0) / 4.25


FIXED = {
    "Length": score_length,
    "CRA-Convex": score_convex,
    "Guard-Turn": score_guard_turn,
    "Guard-Resp": score_guard_resp,
    "Guard-Context": score_guard_ctx,
    "THRD": score_thrd,
}


def score_seq(rec: dict) -> np.ndarray:
    return np.array(rec["g_seq"], dtype=float)


def score_goal_resp(rec: dict) -> np.ndarray:
    return np.array(rec["g_goalresp"], dtype=float)


def score_goal_turn(rec: dict) -> np.ndarray:
    return np.maximum(rec["g_goal"], rec["g_goalresp"])


GOAL_FIXED = {
    "Seq-Monitor": score_seq,
    "Goal-Resp": score_goal_resp,
    "Goal-Turn": score_goal_turn,
}


# ─── learned detectors ───────────────────────────────────────────────────────

def accumulate(E: np.ndarray, with_time: bool = True) -> np.ndarray:
    """Causal accumulator state h_t for every prefix (Eqs. in Sec. III).

    For each evidence channel x: current value x_t, running max, EWMA,
    running mean, excess sum sum_{k<=t} max(0, x_k - CUSUM_REF), and the
    one-step change x_t - x_{t-1}. Optionally log(1 + t).
    """
    T, d = E.shape
    H = []
    m = np.full(d, -np.inf)
    a = np.zeros(d)
    s = np.zeros(d)
    c = np.zeros(d)
    prev = E[0]
    for t in range(T):
        x = E[t]
        m = np.maximum(m, x)
        a = x if t == 0 else EWMA_LAMBDA * x + (1 - EWMA_LAMBDA) * a
        s = s + x
        c = c + np.maximum(0.0, x - CUSUM_REF)
        delta = x - prev
        prev = x
        row = [x, m, a, s / (t + 1), c, delta]
        if with_time:
            row.append(np.array([math.log1p(t + 1)]))
        H.append(np.concatenate(row))
    return np.array(H)


def _gbm():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          class_weight="balanced", random_state=0)


def _lr():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))


class TurnLR:
    """Stateless learned baseline: one classifier applied to each exchange's
    local evidence in isolation (multiple-instance training: every exchange
    inherits its session label); the session score is the max over turns."""

    def __init__(self, groups: str = "L", head=None):
        self.groups = groups
        self.head = head or _lr

    def fit(self, recs):
        X = np.vstack([evidence(r, self.groups) for r in recs])
        y = np.concatenate([[r["label"]] * r["T"] for r in recs])
        self.m = self.head().fit(X, y)
        return self

    def turn_scores(self, rec):
        return self.m.predict_proba(evidence(rec, self.groups))[:, 1]


class NoisyOR(TurnLR):
    """Turn-LR probabilities aggregated by the noisy-OR rule,
    s_t = sum_{k<=t} -log(1 - p_k): the Bayes-optimal session score when the
    label is local and exchanges are conditionally independent given their
    evidence (Proposition 1). Monotone in t, so max_t s_t = s_T."""

    def turn_scores(self, rec):
        p = np.clip(super().turn_scores(rec), 1e-6, 1 - 1e-6)
        return np.cumsum(-np.log1p(-p))


class CRAAcc:
    """Accumulator detector: logistic model on the causal accumulator state
    h_t, applied online to every prefix state h_t.

    prefix=False: fitted on end-of-session states h_T only (as published).
    prefix=True:  fitted on every prefix state h_t with the session label,
                  each session weighted equally (weights 1/T), so the model
                  is trained on the same prefix states it scores online and
                  learns P(session becomes positive | h_t)."""

    def __init__(self, groups: str = "LCX", with_time: bool = True, head=None, prefix: bool = False):
        self.groups = groups
        self.with_time = with_time
        self.head = head or _lr
        self.prefix = prefix

    def fit(self, recs):
        if self.prefix:
            H = [accumulate(evidence(r, self.groups), self.with_time) for r in recs]
            X = np.vstack(H)
            y = np.concatenate([[r["label"]] * len(h) for r, h in zip(recs, H)])
            w = np.concatenate([[1.0 / len(h)] * len(h) for h in H])
            m = self.head()
            if hasattr(m, "steps"):
                self.m = m.fit(X, y, **{m.steps[-1][0] + "__sample_weight": w})
            else:
                self.m = m.fit(X, y, sample_weight=w)
            return self
        X = np.vstack([accumulate(evidence(r, self.groups), self.with_time)[-1] for r in recs])
        y = np.array([r["label"] for r in recs])
        self.m = self.head().fit(X, y)
        return self

    def turn_scores(self, rec):
        return self.m.predict_proba(accumulate(evidence(rec, self.groups), self.with_time))[:, 1]

    def coefficients(self) -> dict:
        lr = self.m[-1]
        return {"coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}


class GRUDetector:
    """Recurrent learned state. ``prior=True`` reproduces the published
    CRA-Net input ([S1, S2, S3, t_norm, n_norm], lambda = 0.05 length GRL);
    note that n_norm encodes the full session length and is therefore not
    causal. ``prior=False`` feeds causal evidence [L, C, X, log(1+t)]."""

    def __init__(self, prior: bool, epochs: int = 40, seed: int = 0):
        self.prior = prior
        self.epochs = epochs
        self.seed = seed

    def _feat(self, rec):
        T = rec["T"]
        t = np.arange(T)
        if self.prior:
            n_norm = math.log(1 + T) / math.log(21)
            t_norm = np.log1p(t) / math.log(1 + max(T, 1))
            return np.column_stack([rec["S1"], rec["S2"], rec["S3"], t_norm,
                                    np.full(T, n_norm)])
        return np.column_stack([evidence(rec, "LCX"), np.log1p(t + 1) / math.log(9)])

    def fit(self, recs):
        import torch
        import torch.nn as nn
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import run_cranet as rc
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        feats = [self._feat(r) for r in recs]
        d = feats[0].shape[1]
        L = max(f.shape[0] for f in feats)
        X = np.zeros((len(recs), L, d), dtype=np.float32)
        W = np.zeros((len(recs), L), dtype=np.float32)
        for i, f in enumerate(feats):
            X[i, :len(f)] = f
            W[i, :len(f)] = ((np.arange(1, len(f) + 1)) / len(f)) ** rc.RHO
        y = np.array([r["label"] for r in recs], dtype=np.float32)
        lengths = np.array([len(f) for f in feats])
        lam = 0.05 if self.prior else 0.0
        self.model = rc.CRANet(input_dim=d, hidden=64, n_layers=2, lam=lam)
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        bce = nn.BCELoss(reduction="none")
        mse = nn.MSELoss()
        Xt, yt, Wt, Lt = map(torch.tensor, (X, y, W, lengths))
        n = len(recs)
        for _ in range(self.epochs):
            self.model.train()
            perm = torch.randperm(n)
            for b in range(0, n, 64):
                idx = perm[b:b + 64]
                risk, len_pred = self.model(Xt[idx], Lt[idx])
                yy = yt[idx].unsqueeze(1).expand_as(risk)
                loss = (bce(risk, yy) * Wt[idx]).sum() / (Wt[idx].sum() + 1e-8)
                if lam > 0:
                    target = torch.log(1 + Lt[idx].float()) / math.log(21)
                    loss = loss + mse(len_pred, target)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
        self.model.eval()
        return self

    def turn_scores(self, rec):
        import torch
        f = torch.tensor(self._feat(rec), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            risk, _ = self.model(f, None)
        return risk[0].numpy()


def learned_factories(seed: int) -> dict:
    """Name -> zero-arg constructor for every learned detector."""
    return {
        "Turn-LR": lambda: TurnLR("L"),
        "CRA-Net (prior)": lambda: GRUDetector(prior=True, seed=seed),
        "GRU-State": lambda: GRUDetector(prior=False, seed=seed),
        "CRA-Acc[C]": lambda: CRAAcc("C"),
        "CRA-Acc[L]": lambda: CRAAcc("L"),
        "CRA-Acc[L+C]": lambda: CRAAcc("LC"),
        "CRA-Acc": lambda: CRAAcc("LCX"),
        "CRA-Acc (no t)": lambda: CRAAcc("LCX", with_time=False),
        "Turn-GBM": lambda: TurnLR("L", head=_gbm),
        "CRA-Acc-GBM[L]": lambda: CRAAcc("L", head=_gbm),
        "Turn-LR-NOR": lambda: NoisyOR("L"),
        "CRA-Acc-P[L]": lambda: CRAAcc("L", prefix=True),
        "CRA-Acc-P": lambda: CRAAcc("LCX", prefix=True),
        "CRA-Acc-P-GBM[L]": lambda: CRAAcc("L", head=_gbm, prefix=True),
    }


def goal_factories(seed: int) -> dict:
    """Learned detectors that add goal-inference evidence (--goal runs)."""
    return {
        "Goal-LR": lambda: TurnLR("LG"),
        "Goal-GBM": lambda: TurnLR("LG", head=_gbm),
        "CRA-Acc[L+G]": lambda: CRAAcc("LG"),
        "CRA-Acc-P[L+G]": lambda: CRAAcc("LG", prefix=True),
    }


def registry(goal: bool = False):
    """(fixed detectors, learned-factory function) for one evaluation setup;
    goal=True appends the goal-evidence detectors after the base ones."""
    fixed = {**FIXED, **(GOAL_FIXED if goal else {})}

    def learned(seed: int) -> dict:
        return {**learned_factories(seed), **(goal_factories(seed) if goal else {})}

    return fixed, learned
