#!/usr/bin/env python3
"""
CRA-Net evaluation on length-matched corpus.

Trains CRA-Net (lambda=0, 0.05) on all 1800 CoSafe sessions, then evaluates
on a length-matched test corpus:
  - Positive class:  750 CoSafe 3-turn gradual-escalation sessions
  - Negative class:  216 ShareGPT 3-user-turn benign sessions

Both classes have exactly 3 user turns, so length-only AUROC ~ 0.50.
This tests whether CRA-Net can discriminate by content rather than length.

Output: experiments/results/cranet_length_matched.json
"""

from __future__ import annotations
import json, math, sys, time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR  = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CACHE_PATH      = RESULTS_DIR / "cosafe_per_turn_signals.json"
SHAREGPT_CACHE  = RESULTS_DIR / "sharegpt_3turn_signals.json"
OUT_PATH        = RESULTS_DIR / "cranet_length_matched.json"

# ─── Hyper-parameters (match run_cranet.py) ───────────────────────────────────
S2_NORM    = 10.0
WINDOW     = 6
EPOCHS     = 80          # slightly more epochs for full-data training
LR         = 1e-3
BATCH      = 32
GRU_H      = 128
GRU_LAYERS = 2
RHO        = 1.0
LAMBDAS    = [0.0, 0.05]
SEED       = 42
N_BOOT     = 2000
MAX_SHAREGPT_SESSIONS = 950   # load enough to find 216 3-turn sessions

torch.manual_seed(SEED)
np.random.seed(SEED)

SENSITIVE_ENTS = {"PERSON","ORG","GPE","LOC","MONEY","CARDINAL","PRODUCT","LAW","WEAPON"}
UNSAFE_KEYWORDS = [
    "ignore previous","disregard","jailbreak","no restrictions",
    "pretend you are","roleplay as","do anything now","unrestricted",
    "bypass safety","forget your guidelines","step by step synthesis",
    "exploit code","without redaction","how to make","how to build",
]
REFUSAL_CUES = [
    "cannot","can't","unable to","not able","i refuse","decline",
    "against policy","not permitted","not allowed","unable to assist",
]


def _cosine_sim(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0


def keyword_judge(text: str) -> float:
    s = text.lower()
    unsafe  = sum(1 for kw in UNSAFE_KEYWORDS if kw in s)
    refusal = sum(1 for c  in REFUSAL_CUES   if c  in s)
    return min(1.0, unsafe * 0.25 + refusal * 0.15)


def ner_score(nlp, text: str) -> int:
    import re
    doc  = nlp(text)
    hits = sum(1 for ent in doc.ents if ent.label_ in SENSITIVE_ENTS)
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):                   hits += 2
    if re.search(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b', text):      hits += 1
    if re.search(r'\b(ssn|social security|credit card|password|api key)\b', text, re.I): hits += 2
    return hits


def extract_signals_from_turns(turns: list, sbert, nlp) -> list | None:
    """Extract per-turn [S1, S2, S3, t_norm, n_norm] for a ShareGPT session."""
    user_turns  = [t for t in turns if t.get("from") == "human"]
    asst_turns  = [t for t in turns if t.get("from") == "gpt"]
    n_user = len(user_turns)
    if n_user == 0:
        return None

    anchor_text = user_turns[0].get("value", "")
    anchor_emb  = sbert.encode(anchor_text, show_progress_bar=False)
    cumulative_user = ""
    cumulative_sens = 0.0
    refusal_hist: list[float] = []
    feat_seq = []

    for ti, ut in enumerate(user_turns):
        cumulative_user += " " + ut.get("value", "")
        cum_emb = sbert.encode(cumulative_user.strip(), show_progress_bar=False)
        S1 = max(0.0, min(1.0, 1.0 - _cosine_sim(anchor_emb, cum_emb)))

        asst_text = asst_turns[ti].get("value", "") if ti < len(asst_turns) else ""
        cumulative_sens += ner_score(nlp, asst_text)
        S2 = max(0.0, min(1.0, cumulative_sens / S2_NORM))

        win = "\n".join(
            f"{'User' if u.get('from')=='human' else 'Asst'}: {u.get('value','')}"
            for u in user_turns[max(0, ti-WINDOW):ti+1]
        )
        raw_s3 = keyword_judge(win)
        refusal_hist.append(raw_s3)
        slope = refusal_hist[-1] - refusal_hist[-2] if len(refusal_hist) >= 2 else 0.0
        S3 = max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-10.0 * slope))))

        t_norm = math.log(1 + ti) / math.log(1 + max(n_user, 1))
        n_norm = math.log(1 + n_user) / math.log(1 + 20)
        feat_seq.append([S1, S2, S3, t_norm, n_norm])

    return feat_seq


def load_sharegpt_3turn_signals(sbert, nlp) -> list:
    """Load 216 ShareGPT 3-user-turn benign sessions and extract signals."""
    if SHAREGPT_CACHE.exists():
        print("[ShareGPT] Loading cached 3-turn signals...")
        with open(SHAREGPT_CACHE) as f:
            return json.load(f)

    print("[ShareGPT] Extracting signals from ShareGPT 3-turn benign sessions...")
    from datasets import load_dataset
    ds = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)

    sessions = []
    seen = 0
    for row in ds:
        if seen >= MAX_SHAREGPT_SESSIONS:
            break
        seen += 1
        convs = row.get("conversations", [])
        user_turns = [t for t in convs if t.get("from") == "human"]
        if len(user_turns) != 3:
            continue
        # Safety filter
        all_text = " ".join(t.get("value", "") for t in convs).lower()
        if any(kw in all_text for kw in UNSAFE_KEYWORDS):
            continue

        feats = extract_signals_from_turns(convs, sbert, nlp)
        if feats is not None and len(feats) == 3:
            sessions.append({"label": 0, "features": feats, "n_turns": 3})

    print(f"[ShareGPT] Extracted signals for {len(sessions)} 3-turn benign sessions")
    with open(SHAREGPT_CACHE, "w") as f:
        json.dump(sessions, f)
    return sessions


# ─── Gradient reversal & CRA-Net ─────────────────────────────────────────────

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None

def grad_reverse(x, lam=0.05):
    return GradReverse.apply(x, lam)


class CRANet(nn.Module):
    def __init__(self, input_dim=5, hidden=GRU_H, n_layers=GRU_LAYERS, lam=0.05):
        super().__init__()
        self.lam = lam
        self.gru = nn.GRU(input_dim, hidden, n_layers, batch_first=True, dropout=0.2)
        self.classifier  = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())
        self.length_head = nn.Linear(hidden, 1)

    def forward(self, x, lengths=None):
        out, _ = self.gru(x)
        risk    = self.classifier(out).squeeze(-1)
        len_pred = None
        if lengths is not None:
            B   = out.size(0)
            idx = (lengths - 1).clamp(min=0).long()
            last_h   = out[torch.arange(B), idx]
            rev_h    = grad_reverse(last_h, self.lam)
            len_pred = self.length_head(rev_h).squeeze(-1)
        return risk, len_pred


class SessionDataset(Dataset):
    def __init__(self, sessions: list, max_len: int = 20):
        self.sessions = sessions
        self.max_len  = max_len

    def __len__(self):  return len(self.sessions)

    def __getitem__(self, idx):
        s      = self.sessions[idx]
        feats  = s["features"]
        n      = min(len(feats), self.max_len)
        x      = torch.zeros(self.max_len, 5)
        x[:n]  = torch.tensor(feats[:n], dtype=torch.float32)
        label  = torch.tensor(s["label"], dtype=torch.float32)
        length = torch.tensor(n, dtype=torch.long)
        return x, label, length


def _collate(batch):
    xs, labels, lengths = zip(*batch)
    return torch.stack(xs), torch.stack(labels), torch.stack(lengths)


def train_model(train_sessions: list, lam: float) -> CRANet:
    model = CRANet(lam=lam)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    ds    = SessionDataset(train_sessions)
    dl    = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=_collate)

    for ep in range(EPOCHS):
        model.train()
        for x, y, lengths in dl:
            opt.zero_grad()
            T = x.size(1)
            t_indices = torch.arange(1, T + 1, dtype=torch.float32)
            n_turns   = lengths.float()

            risk, len_pred = model(x, lengths)

            # Turn-weighted BCE
            w_t = (t_indices.unsqueeze(0) / n_turns.unsqueeze(1)) ** RHO
            bce = -(y.unsqueeze(1) * torch.log(risk + 1e-8)
                    + (1 - y.unsqueeze(1)) * torch.log(1 - risk + 1e-8))
            # Mask padding
            mask = (t_indices.unsqueeze(0) <= n_turns.unsqueeze(1)).float()
            clf_loss = (bce * w_t * mask).sum() / (mask.sum() + 1e-8)

            # Length adversarial loss
            if len_pred is not None and lam > 0:
                log_len  = torch.log1p(n_turns.float())
                adv_loss = ((len_pred - log_len) ** 2).mean()
                loss = clf_loss + adv_loss
            else:
                loss = clf_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    return model


def score_sessions(model: CRANet, sessions: list) -> list[float]:
    """Return session-level score = max predicted risk across turns."""
    model.eval()
    ds = SessionDataset(sessions)
    dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=_collate)
    scores = []
    with torch.no_grad():
        for x, y, lengths in dl:
            risk, _ = model(x)
            # Mask padding
            T = x.size(1)
            t_idx = torch.arange(1, T + 1).unsqueeze(0)
            mask  = (t_idx <= lengths.unsqueeze(1)).float()
            risk  = risk * mask + (1 - mask) * (-1.0)
            scores.extend(risk.max(dim=1).values.tolist())
    return scores


def _auc(y_true, y_score) -> float:
    pairs   = sorted(zip(y_score, y_true), reverse=True)
    n_pos   = sum(y_true)
    n_neg   = len(y_true) - n_pos
    if not n_pos or not n_neg: return float("nan")
    tp = fp = val = 0
    for _, yl in pairs:
        if yl: tp += 1
        else:  fp += 1; val += tp
    return val / (n_pos * n_neg)


def _boot_ci(y_true, y_score, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    yt, ys = np.array(y_true), np.array(y_score)
    samples = [_auc(yt[idx].tolist(), ys[idx].tolist())
               for idx in (rng.integers(0, len(yt), len(yt)) for _ in range(n))]
    samples = [s for s in samples if not math.isnan(s)]
    lo, hi  = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def compute_sfpr(y_true, y_score, tpr=0.90):
    pos_scores = sorted(s for s, l in zip(y_score, y_true) if l)
    k          = int(math.ceil((1 - tpr) * len(pos_scores)))
    thresh     = pos_scores[k] if k < len(pos_scores) else pos_scores[-1]
    neg_exc    = sum(1 for s, l in zip(y_score, y_true) if l == 0 and s >= thresh)
    n_neg      = sum(1 for l in y_true if l == 0)
    return neg_exc / n_neg if n_neg else 0.0, thresh


def length_only_auroc(pos_sessions, neg_sessions):
    y_true  = [1]*len(pos_sessions) + [0]*len(neg_sessions)
    y_score = [s["n_turns"] for s in pos_sessions + neg_sessions]
    return _auc(y_true, y_score)


def main():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    print("[LM] Loading CoSafe per-turn signals from cache...")
    with open(CACHE_PATH) as f:
        cosafe_data = json.load(f)

    cosafe_pos = [s for s in cosafe_data if s["label"] == 1]  # 750 sessions, 3-turn
    print(f"[LM] CoSafe positive: {len(cosafe_pos)} sessions")

    # Load models
    print("[LM] Loading sentence encoder...")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    print("[LM] Loading spaCy...")
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    # Load ShareGPT 3-turn benign sessions
    sharegpt_neg = load_sharegpt_3turn_signals(sbert, nlp)
    print(f"[LM] ShareGPT negative (3-turn, benign): {len(sharegpt_neg)} sessions")

    # Length-only baseline for sanity check
    lo_auroc = length_only_auroc(cosafe_pos, sharegpt_neg)
    print(f"[LM] Length-only AUROC on length-matched corpus: {lo_auroc:.4f} (expected ~0.50)")

    # Build length-matched test corpus labels
    lm_sessions = cosafe_pos + sharegpt_neg
    y_true_lm   = [s["label"] for s in lm_sessions]

    results = {"length_only_auroc": round(lo_auroc, 4)}

    for lam in LAMBDAS:
        print(f"\n[LM] Training CRA-Net λ={lam} on all 1800 CoSafe sessions...")
        t0    = time.time()
        model = train_model(cosafe_data, lam)
        print(f"[LM] Training done in {time.time()-t0:.1f}s")

        # Score on CoSafe-only (sanity check: should be ~1.0000)
        cosafe_scores = score_sessions(model, cosafe_data)
        y_cosafe = [s["label"] for s in cosafe_data]
        auroc_cosafe = _auc(y_cosafe, cosafe_scores)
        print(f"[LM] CoSafe AUROC (sanity check): {auroc_cosafe:.4f}")

        # Score on length-matched corpus
        lm_scores = score_sessions(model, lm_sessions)
        auroc_lm  = _auc(y_true_lm, lm_scores)
        sfpr_lm, thr_lm = compute_sfpr(y_true_lm, lm_scores)
        ci_lo, ci_hi = _boot_ci(y_true_lm, lm_scores)

        print(f"[LM] λ={lam}: AUROC(LM) = {auroc_lm:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]  sFPR = {sfpr_lm:.4f}")

        # TTD on CoSafe positive sessions
        cosafe_pos_scores_per_turn = []
        with torch.no_grad():
            ds = SessionDataset(cosafe_pos)
            dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=_collate)
            for x, y, lengths in dl:
                risk, _ = model(x)
                cosafe_pos_scores_per_turn.extend(risk.tolist())

        # Find threshold from CoSafe full eval
        full_scores = score_sessions(model, cosafe_data)
        _, thresh = compute_sfpr([s["label"] for s in cosafe_data], full_scores)

        ttd_turns = []
        for turn_scores, sess in zip(cosafe_pos_scores_per_turn, cosafe_pos):
            n = sess["n_turns"]
            turn_scores_trimmed = turn_scores[:n]
            detected = next((t for t, sc in enumerate(turn_scores_trimmed) if sc >= thresh), n)
            turns_before_apex = (n - 1) - detected
            ttd_turns.append(max(0, turns_before_apex))
        mean_ttd = sum(ttd_turns) / len(ttd_turns) if ttd_turns else float("nan")

        results[f"lambda_{lam}"] = {
            "lam": lam,
            "auroc_cosafe_sanity": round(auroc_cosafe, 4),
            "auroc_length_matched": round(auroc_lm, 4),
            "auroc_lm_ci_lo": round(ci_lo, 4),
            "auroc_lm_ci_hi": round(ci_hi, 4),
            "sfpr_length_matched": round(sfpr_lm, 4),
            "threshold": round(thr_lm, 6),
            "mean_ttd_cosafe": round(mean_ttd, 2),
            "n_pos": len(cosafe_pos),
            "n_neg": len(sharegpt_neg),
        }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n[LM] Results:")
    print(json.dumps(results, indent=2))
    print(f"\n[LM] Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
