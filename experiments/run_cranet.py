#!/usr/bin/env python3
"""
CRA-Net: 2-layer GRU learned trajectory fusion with length regularization.

Pipeline:
  1. Load CoSafe (reuses run_cra_cosafe.py loaders) and extract per-turn
     [S1, S2, S3, turn_log_norm, n_turns_log_norm] feature vectors.
  2. Train CRA-Net with 5-fold stratified CV.
     - Length regularization via gradient-reversal adversarial head (λ=0.05).
     - Turn-weighted BCE loss (rho=1.0).
  3. Report session-level AUROC (TM = max prediction across turns),
     sFPR @ TPR=0.90, mean TTD.
  4. Also evaluate variant without length regularization (λ=0).

Output: experiments/results/cranet_results.json
"""

from __future__ import annotations
import json, math, sys, time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(SCRIPT_DIR))

# ─── Hyper-parameters ────────────────────────────────────────────────────────
ALPHA, BETA, GAMMA = 0.35, 0.45, 0.20
S2_NORM = 10.0
WINDOW  = 6
N_FOLDS = 5
EPOCHS  = 60
LR      = 1e-3
BATCH   = 32
GRU_H   = 128
GRU_LAYERS = 2
RHO     = 1.0          # turn-weight exponent
LAMBDAS = [0.0, 0.05]  # length-reg strengths to evaluate
SEED    = 42
N_BOOT  = 2000
torch.manual_seed(SEED)
np.random.seed(SEED)

SENSITIVE_ENTS = {"PERSON","ORG","GPE","LOC","MONEY","CARDINAL","PRODUCT","LAW","WEAPON"}

# ─── Signal extraction ────────────────────────────────────────────────────────

def _cosine_sim(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na and nb else 0.0

KEYWORD_UNSAFE = [
    "ignore previous instructions","disregard","jailbreak","no restrictions",
    "pretend you are","roleplay as","do anything now","unrestricted",
    "bypass safety","forget your guidelines","hypothetically speaking let's",
    "step by step synthesis","exploit code","without redaction",
]
REFUSAL_CUES = [
    "cannot","can't","unable to","not able","i refuse","decline",
    "against policy","not permitted","not allowed","unable to assist",
]

def keyword_judge(text: str) -> float:
    s = text.lower()
    unsafe_hits = sum(1 for kw in KEYWORD_UNSAFE if kw in s)
    refusal_hits = sum(1 for c in REFUSAL_CUES if c in s)
    return min(1.0, unsafe_hits * 0.25 + refusal_hits * 0.15)

def ner_score(nlp, text: str) -> float:
    import re
    doc = nlp(text)
    hits = sum(1 for ent in doc.ents if ent.label_ in SENSITIVE_ENTS)
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text): hits += 2
    if re.search(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b', text): hits += 1
    if re.search(r'\b(ssn|social security|credit card|password|api key)\b', text, re.I): hits += 2
    return hits


def extract_signals(
    sessions: List[dict],
    sbert,
    nlp,
    s3_mode: str = "keyword",
    s3_clf=None,
) -> List[dict]:
    """Return list of {label, features: [[S1,S2,S3,t_norm,n_norm], ...], n_turns, onset}.

    s3_mode: 'keyword' (default) or 'classifier' (DistilBERT refusal head).
    """
    if s3_mode == "classifier" and s3_clf is None:
        from s3_refusal import ClassifierS3
        s3_clf = ClassifierS3()
    out = []
    for sess in sessions:
        turns = sess["turns"]
        user_turns = [t for t in turns if t["role"] == "user"]
        n_user = len(user_turns)
        if n_user == 0:
            continue

        anchor_emb = sbert.encode(user_turns[0]["content"], show_progress_bar=False)
        cumulative_user = ""
        cumulative_sens = 0.0
        refusal_hist: List[float] = []

        feat_seq = []
        for ti, ut in enumerate(user_turns):
            cumulative_user += " " + ut["content"]

            # S1
            cum_emb = sbert.encode(cumulative_user.strip(), show_progress_bar=False)
            S1 = max(0.0, min(1.0, 1.0 - _cosine_sim(anchor_emb, cum_emb)))

            # S2: find assistant reply
            asst_text = ""
            ui = turns.index(ut)
            for j in range(ui + 1, min(ui + 3, len(turns))):
                if turns[j]["role"] == "assistant":
                    asst_text = turns[j]["content"]
                    break
            cumulative_sens += ner_score(nlp, asst_text)
            S2 = max(0.0, min(1.0, cumulative_sens / S2_NORM))

            # S3
            window_start = max(0, ti - WINDOW)
            win_turns_flat = []
            u_idx = 0
            for t in turns:
                if t["role"] == "user":
                    if window_start <= u_idx <= ti:
                        win_turns_flat.append(t)
                    u_idx += 1
                else:
                    if window_start <= u_idx - 1 <= ti:
                        win_turns_flat.append(t)
            window_text = "\n".join(
                f"{'User' if t['role']=='user' else 'Asst'}: {t['content']}"
                for t in win_turns_flat
            )
            if s3_mode == "classifier" and s3_clf is not None:
                S3 = s3_clf.s3_turn(turns, ti, WINDOW)
            else:
                raw_s3 = keyword_judge(window_text)
                refusal_hist.append(raw_s3)
                if len(refusal_hist) >= 2:
                    slope = refusal_hist[-1] - refusal_hist[-2]
                else:
                    slope = 0.0
                S3 = max(0.0, min(1.0, 1.0 / (1.0 + math.exp(-10.0 * slope))))

            # Turn index features
            t_norm = math.log(1 + ti) / math.log(1 + max(n_user, 1))
            n_norm = math.log(1 + n_user) / math.log(1 + 20)

            feat_seq.append([S1, S2, S3, t_norm, n_norm])

        out.append({
            "label":    sess["label"],
            "features": feat_seq,
            "n_turns":  n_user,
            "onset":    sess.get("onset_turn", 0),
        })
    return out


# ─── Gradient reversal layer ──────────────────────────────────────────────────

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


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Deep CORAL (Sun & Saenko 2016): align second-order statistics."""
    d = source.size(1)
    ns, nt = source.size(0), target.size(0)
    if ns < 2 or nt < 2:
        return source.new_tensor(0.0)
    xm = source - source.mean(0, keepdim=True)
    xc = target - target.mean(0, keepdim=True)
    cs = (xm.t() @ xm) / (ns - 1)
    ct = (xc.t() @ xc) / (nt - 1)
    return ((cs - ct) ** 2).sum() / (4.0 * d * d)


def batch_coral_loss(last_h: torch.Tensor, fam: torch.Tensor) -> torch.Tensor:
    """Sum pairwise CORAL losses across distinct family ids in a minibatch."""
    unique = fam.unique(sorted=True)
    if unique.numel() < 2:
        return last_h.new_tensor(0.0)
    total = last_h.new_tensor(0.0)
    pairs = 0
    for i, fi in enumerate(unique):
        hi = last_h[fam == fi]
        if hi.size(0) < 2:
            continue
        for fj in unique[i + 1:]:
            hj = last_h[fam == fj]
            if hj.size(0) < 2:
                continue
            total = total + coral_loss(hi, hj)
            pairs += 1
    if pairs == 0:
        return last_h.new_tensor(0.0)
    return total / pairs


# ─── CRA-Net model ────────────────────────────────────────────────────────────

class CRANet(nn.Module):
    def __init__(self, input_dim=5, hidden=GRU_H, n_layers=GRU_LAYERS, lam=0.05):
        super().__init__()
        self.lam = lam
        self.gru = nn.GRU(input_dim, hidden, n_layers, batch_first=True, dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        # Adversarial length predictor head
        self.length_head = nn.Linear(hidden, 1)

    def forward(self, x, lengths=None):
        """
        x: (B, T, d) padded sequence
        Returns: risk_scores (B, T), length_pred (B,)
        """
        out, _ = self.gru(x)  # (B, T, H)
        risk = self.classifier(out).squeeze(-1)  # (B, T)
        # Length adversarial head: uses gradient reversal
        if lengths is not None:
            # Take last actual hidden state per sequence
            idx = (lengths - 1).clamp(min=0).long()
            B = out.size(0)
            last_h = out[torch.arange(B), idx]  # (B, H)
            rev_h  = grad_reverse(last_h, self.lam)
            len_pred = self.length_head(rev_h).squeeze(-1)  # (B,)
            return risk, len_pred
        return risk, None


# ─── CRA-Net DA: family-adversarial variant ──────────────────────────────────

class CRANetDA(nn.Module):
    """CRA-Net + length GRL + family GRL.

    Adds a domain-adversarial (Ganin & Lempitsky 2015) head that predicts the
    threat family from the encoder's final hidden state. Its gradient is
    reversed with weight `lam_fam`, pushing the encoder toward family-invariant
    representations. The intended effect is to reduce per-family template
    memorization and improve leave-one-family-out (LOFO) generalization.

    The length head from CRANet is preserved (length GRL with weight `lam`).
    """

    def __init__(self, input_dim=5, hidden=GRU_H, n_layers=GRU_LAYERS,
                 lam=0.05, lam_fam=0.3, n_families=2):
        super().__init__()
        self.lam = lam
        self.lam_fam = lam_fam
        self.n_families = n_families
        self.gru = nn.GRU(input_dim, hidden, n_layers, batch_first=True,
                          dropout=0.2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.length_head = nn.Linear(hidden, 1)
        self.family_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden // 2, n_families),
        )

    def forward(self, x, lengths=None):
        """x: (B, T, d). Returns risk (B, T), len_pred (B,)|None,
        fam_logits (B, n_families)|None."""
        out, _ = self.gru(x)
        risk = self.classifier(out).squeeze(-1)
        if lengths is None:
            return risk, None, None
        idx = (lengths - 1).clamp(min=0).long()
        B = out.size(0)
        last_h = out[torch.arange(B), idx]
        len_pred = None
        fam_logits = None
        if self.lam > 0:
            rev_h_len = grad_reverse(last_h, self.lam)
            len_pred = self.length_head(rev_h_len).squeeze(-1)
        if self.lam_fam > 0:
            rev_h_fam = grad_reverse(last_h, self.lam_fam)
            fam_logits = self.family_head(rev_h_fam)
        return risk, len_pred, fam_logits


# ─── Dataset ──────────────────────────────────────────────────────────────────

class SessionDataset(Dataset):
    def __init__(self, records: List[dict], max_len: int):
        self.records = records
        self.max_len = max_len

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        feat = np.array(r["features"], dtype=np.float32)
        T = len(feat)
        pad = np.zeros((self.max_len, feat.shape[1]), dtype=np.float32)
        pad[:T] = feat
        label = float(r["label"])
        turn_weights = np.array(
            [(t / max(T, 1)) ** RHO for t in range(1, T + 1)], dtype=np.float32
        )
        tw_pad = np.zeros(self.max_len, dtype=np.float32)
        tw_pad[:T] = turn_weights
        return (
            torch.from_numpy(pad),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(T, dtype=torch.long),
            torch.from_numpy(tw_pad),
        )


# ─── Metrics ─────────────────────────────────────────────────────────────────

def _auc(y_true, y_score) -> float:
    pairs = sorted(zip(y_score, y_true), reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    tp = fp = 0
    auc = 0.0
    prev_fp = 0
    for _, yl in pairs:
        if yl == 1:
            tp += 1
        else:
            fp += 1
            auc += tp
    return auc / (n_pos * n_neg)


def compute_sfpr(y_true, y_score, tpr_target=0.90):
    pos_scores = sorted([s for s, l in zip(y_score, y_true) if l == 1])
    k = int(math.ceil((1 - tpr_target) * len(pos_scores)))
    if not pos_scores:
        return 0.0, 0.0
    thresh = pos_scores[k] if k < len(pos_scores) else pos_scores[-1]
    neg_exc = sum(1 for s, l in zip(y_score, y_true) if l == 0 and s >= thresh)
    n_neg = sum(1 for l in y_true if l == 0)
    sfpr = neg_exc / n_neg if n_neg else 0.0
    return sfpr, thresh


def compute_ttd(records, thresh) -> float:
    ttds = []
    for r in records:
        if r["label"] != 1:
            continue
        onset = r["onset"]
        n = r["n_turns"]
        # find first turn where prediction crosses thresh
        preds = r.get("preds", [])
        for ti, p in enumerate(preds):
            if p >= thresh:
                ttd = max(0, n - (ti + 1))
                ttds.append(ttd)
                break
    return float(np.mean(ttds)) if ttds else 0.0


def bootstrap_auc(y_true, y_score, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap CI for AUROC.

    Resamples sessions with replacement and recomputes AUROC. Returns
    (mean, p2.5, p97.5). NaN-safe.
    """
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)
    n = len(y_true_arr)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true_arr[idx].tolist()
        ys = y_score_arr[idx].tolist()
        a = _auc(yt, ys)
        if not math.isnan(a):
            vals.append(a)
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return (float(np.mean(vals)),
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def bootstrap_sfpr(y_true, y_score, tpr_target: float = 0.90,
                   n_boot: int = N_BOOT, seed: int = SEED):
    """Percentile bootstrap CI for sFPR @ TPR=tpr_target."""
    rng = np.random.default_rng(seed)
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)
    n = len(y_true_arr)
    if n == 0:
        return float("nan"), float("nan")
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true_arr[idx].tolist()
        ys = y_score_arr[idx].tolist()
        if sum(yt) == 0 or sum(1 for v in yt if v == 0) == 0:
            continue
        s, _ = compute_sfpr(yt, ys, tpr_target=tpr_target)
        vals.append(s)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_ttd(records, thresh: float, n_boot: int = N_BOOT,
                  seed: int = SEED):
    """Percentile bootstrap CI for mean TTD on positive sessions only."""
    pos_recs = [r for r in (records or []) if r["label"] == 1 and r.get("preds")]
    n = len(pos_recs)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [pos_recs[i] for i in idx]
        ttds = []
        for r in sample:
            preds = r.get("preds", [])
            nt = r["n_turns"]
            for ti, p in enumerate(preds):
                if p >= thresh:
                    ttds.append(max(0, nt - (ti + 1)))
                    break
        if ttds:
            vals.append(float(np.mean(ttds)))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ─── Training loop ────────────────────────────────────────────────────────────

def train_fold(train_recs, val_recs, max_len, lam, device):
    model = CRANet(input_dim=5, hidden=GRU_H, n_layers=GRU_LAYERS, lam=lam).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    train_ds = SessionDataset(train_recs, max_len)
    train_dl  = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

    bce = nn.BCELoss(reduction="none")
    mse = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        for x, y, lengths, tw in train_dl:
            x, y, tw = x.to(device), y.to(device), tw.to(device)
            risk, len_pred = model(x, lengths)
            # Turn-weighted BCE loss (broadcast session label to all turns)
            y_exp = y.unsqueeze(1).expand_as(risk)
            loss_cls = (bce(risk, y_exp) * tw).sum() / (tw.sum() + 1e-8)
            # Length adversarial loss
            if lam > 0 and len_pred is not None:
                log_len = torch.log(1 + lengths.float()) / math.log(21)
                loss_len = mse(len_pred.to(device), log_len.to(device))
            else:
                loss_len = torch.tensor(0.0)
            loss = loss_cls + loss_len
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    # Evaluation on val set
    model.eval()
    val_ds = SessionDataset(val_recs, max_len)
    val_dl  = DataLoader(val_ds, batch_size=BATCH)
    all_preds, all_labels, per_rec = [], [], []
    with torch.no_grad():
        bi = 0
        for x, y, lengths, _ in val_dl:
            risk, _ = model(x)
            B = x.size(0)
            for i in range(B):
                T = lengths[i].item()
                preds_i = risk[i, :T].cpu().numpy().tolist()
                sess_score = float(max(preds_i)) if preds_i else 0.0
                all_preds.append(sess_score)
                all_labels.append(int(y[i].item()))
                val_recs[bi + i]["preds"] = preds_i
            bi += B

    return model, all_preds, all_labels, val_recs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CRANet] Device: {device}")

    # ── Load CoSafe ──────────────────────────────────────────────────────────
    print("[CRANet] Loading CoSafe dataset...")
    # import loader from run_cra_cosafe
    import importlib.util
    spec = importlib.util.spec_from_file_location("rcc", SCRIPT_DIR / "run_cra_cosafe.py")
    rcc  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rcc)
    sessions = rcc.load_cosafe()
    print(f"[CRANet] Loaded {len(sessions)} sessions "
          f"({sum(s['label'] for s in sessions)} pos / {sum(1-s['label'] for s in sessions)} neg)")

    # ── Load models ───────────────────────────────────────────────────────────
    print("[CRANet] Loading sentence-transformer and spaCy...")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    nlp   = rcc._load_spacy()

    # ── Extract per-turn signals ──────────────────────────────────────────────
    signals_path = RESULTS_DIR / "cosafe_per_turn_signals.json"
    if signals_path.exists():
        print(f"[CRANet] Loading cached signals from {signals_path}")
        with open(signals_path) as f:
            records = json.load(f)
    else:
        print("[CRANet] Extracting per-turn signals (this takes a few minutes)...")
        t0 = time.time()
        records = extract_signals(sessions, sbert, nlp)
        with open(signals_path, "w") as f:
            json.dump(records, f)
        print(f"[CRANet] Signals extracted in {time.time()-t0:.1f}s, saved to {signals_path}")

    labels = [r["label"] for r in records]
    n_turns_list = [r["n_turns"] for r in records]
    max_len = max(n_turns_list)
    print(f"[CRANet] {len(records)} sessions, max_turns={max_len}, "
          f"pos={sum(labels)}, neg={len(labels)-sum(labels)}")

    # ── 5-fold CV for each λ ─────────────────────────────────────────────────
    results = {}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(records, labels))

    for lam in LAMBDAS:
        tag = f"cranet_lam{lam}"
        print(f"\n[CRANet] Training {tag} ...")
        fold_preds, fold_labels, fold_recs_val = [], [], []

        for fi, (train_idx, val_idx) in enumerate(splits):
            print(f"  Fold {fi+1}/{N_FOLDS} ...", end=" ", flush=True)
            train_recs = [records[i] for i in train_idx]
            val_recs   = [dict(records[i]) for i in val_idx]  # copy for preds storage
            t0 = time.time()
            _, preds, lbls, val_recs_out = train_fold(train_recs, val_recs, max_len, lam, device)
            print(f"done in {time.time()-t0:.1f}s  fold-AUROC={_auc(lbls, preds):.4f}")
            fold_preds.extend(preds)
            fold_labels.extend(lbls)
            fold_recs_val.extend(val_recs_out)

        auroc     = _auc(fold_labels, fold_preds)
        sfpr, thr = compute_sfpr(fold_labels, fold_preds)
        mean_ttd  = compute_ttd(fold_recs_val, thr)
        auc_m, auc_lo, auc_hi = bootstrap_auc(fold_labels, fold_preds)

        results[tag] = {
            "lambda": lam,
            "n_folds": N_FOLDS,
            "n_sessions": len(records),
            "n_pos": sum(labels),
            "n_neg": len(labels) - sum(labels),
            "auroc_turn_max": round(auroc, 4),
            "auroc_tm_ci_lo": round(auc_lo, 4),
            "auroc_tm_ci_hi": round(auc_hi, 4),
            "sfpr_at_tpr90":  round(sfpr, 4),
            "threshold":      round(thr, 6),
            "mean_ttd":       round(mean_ttd, 4),
        }
        print(f"  [{tag}] AUROC={auroc:.4f} [{auc_lo:.4f},{auc_hi:.4f}]  "
              f"sFPR={sfpr:.4f}  TTD={mean_ttd:.2f}")

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / "cranet_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[CRANet] Results saved to {out_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
