#!/usr/bin/env python3
"""End-to-end text-trained transformer and DeepContext evaluation (v2 protocol).

Addresses Weakness 3 (fine-tuning a transformer directly on conversation text)
and evaluates DeepContext (Albrethsen et al., 2026, arXiv:2602.16935) under the
exact v2 grouped cross-validation, held-out threshold calibration (5% FPR), and
cluster-bootstrap CI protocol across all corpora:

  1. FT-Turn      Stateless: all-MiniLM-L6-v2 fine-tuned end-to-end (all six
                  layers, mean pooling, linear head) on single-exchange text
                  (u_t, r_t), each exchange labeled with its session label;
                  session score = max_t s_t.
  2. FT-Context   Stateful: the same model fine-tuned end-to-end on the causal
                  conversation prefix (u_1, r_1, ..., u_t, r_t), truncated
                  from the left to 256 tokens so the latest turns are kept;
                  session score = max_t s_t.
  3. DeepContext  Re-implementation of the DeepContext design (Albrethsen et
                  al., 2026) over frozen MiniLM layer-4 features: task-
                  attention pooling, 3-layer GRU, residual current-turn
                  shortcut, focal loss. Not the authors' model or weights.

Single fold shuffle (seed 0); sessions are subsampled per corpus for compute
(MAX_SESSIONS_PER_CORPUS). actorattack_llama is MT-ActorAttack restricted to
the Llama 3.2 target (the DeepSeek-R1 transcripts contain empty replies).

Output:
  experiments/results/v2/text_models.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DERIVED as D, RESULTS, index_jsonl, read_jsonl  # noqa: E402
from detectors import CRAAcc  # noqa: E402
from run_v2_eval import FPR_TARGET, N_BOOT, auroc, boot_ci, calibrate  # noqa: E402

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CORPORA = ["hh_rt", "bad", "prosocial", "crabench", "actorattack_llama", "hh_long"]
MAX_SESSIONS_PER_CORPUS = {
    "hh_rt": 1500,       # stratified group-preserving subsample for tractable 5-fold text fine-tuning
    "bad": 1200,
    "prosocial": 1200,
    "crabench": 1000,
    "actorattack_llama": 410,
    "hh_long": 92,
    "crabench_long": 500,
}
FT_MAX_TURN_TOKENS = 192
FT_MAX_CTX_TOKENS = 256
FT_EPOCHS = 2
FT_LR = 3e-5


def load_corpus_sessions(corpus: str) -> list[dict]:
    llama_only = corpus == "actorattack_llama"
    src = "actorattack" if llama_only else corpus
    sess = index_jsonl(D / f"{src}.jsonl")
    guard = index_jsonl(D / f"guard_{src}.jsonl")
    cra = index_jsonl(D / f"cra_{src}.jsonl")
    recs = []
    for sid, s in sess.items():
        if sid not in guard or sid not in cra:
            continue
        if llama_only and not s["meta"]["target"].startswith("llama"):
            continue
        g, c = guard[sid], cra[sid]
        S = np.array(c["S"])
        recs.append(
            {
                "id": sid,
                "corpus": corpus,
                "label": int(s["label"]),
                "group": s["group"],
                "onset": s["onset"],
                "T": len(s["turns"]),
                "turns": s["turns"],
                "meta": s["meta"],
                "g_user": g["g_user"],
                "g_resp": g["g_resp"],
                "g_ctx": g["g_ctx"],
                "S1": S[:, 0].tolist(),
                "S2": S[:, 1].tolist(),
                "S3": S[:, 2].tolist(),
            }
        )
    max_n = MAX_SESSIONS_PER_CORPUS.get(corpus, len(recs))
    if len(recs) > max_n:
        # Stratified group-preserving subsample (seed 0)
        rng = np.random.default_rng(0)
        by_g = {}
        for r in recs:
            by_g.setdefault(r["group"], []).append(r)
        groups = sorted(by_g.keys())
        rng.shuffle(groups)
        picked = []
        pos_target = max_n // 2
        neg_target = max_n - pos_target
        n_pos, n_neg = 0, 0
        for g in groups:
            for r in by_g[g]:
                if r["label"] == 1 and n_pos < pos_target:
                    picked.append(r)
                    n_pos += 1
                elif r["label"] == 0 and n_neg < neg_target:
                    picked.append(r)
                    n_neg += 1
                if len(picked) >= max_n:
                    break
            if len(picked) >= max_n:
                break
        recs = picked
    return recs


def extract_frozen_stem_features(
    recs: list[dict], tok, base_model, device: str, max_turn_len: int = 192, max_ctx_len: int = 384
):
    """Run layers 0..3 of the transformer once to produce intermediate sequence
    features (mean, max, first-token, and attention-quartiles) + cached layer-4
    inputs so that fine-tuning the top transformer blocks + Task-Attention +
    DeepContext GRU across 5 folds runs in seconds per fold on MPS."""
    turn_texts = []
    ctx_texts = []
    map_idx = []
    for i, r in enumerate(recs):
        hist_parts = []
        for t_i, t in enumerate(r["turns"]):
            u = (t["user"] or "").strip()[:600]
            a = (t["assistant"] or "").strip()[:600]
            cur = f"User: {u}\nAssistant: {a}"
            hist_parts.append(cur)
            # Keep tail of conversation history up to ~2400 chars (~512 tokens)
            ctx = "\n".join(hist_parts)[-2400:]
            turn_texts.append(cur)
            ctx_texts.append(ctx)
            map_idx.append((i, t_i))

    def encode_stem(texts: list[str], max_len: int, batch_size: int = 32) -> np.ndarray:
        """Return (N, 4, d) token-segment tensor per text from transformer layer 4
        output so the trainable Task-Attention + Transformer-top block operates
        over sequence positions."""
        out_segs = []
        for b in range(0, len(texts), batch_size):
            chunk = texts[b : b + batch_size]
            enc = tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                outputs = base_model(**enc, output_hidden_states=True)
                # Use hidden state after layer 4 (index 4) and final layer (index 6)
                h4 = outputs.hidden_states[4].float()  # (B, L, d)
                mask = enc["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
                B, L, d = h4.shape
                # Partition active tokens into 4 sequential sub-segments + CLS + mean + max
                cls_tok = h4[:, 0, :]
                mean_tok = (h4 * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                max_tok = (h4.masked_fill(mask == 0, -1e4)).max(dim=1).values
                # First half vs second half of sequence (captures prompt vs response / early vs late context)
                half = max(1, L // 2)
                m1 = mask[:, :half, :]
                m2 = mask[:, half:, :]
                h_first = (h4[:, :half, :] * m1).sum(dim=1) / m1.sum(dim=1).clamp(min=1.0)
                h_second = (h4[:, half:, :] * m2).sum(dim=1) / m2.sum(dim=1).clamp(min=1.0)
                # Stack 5 sequence summary tokens: [CLS, first_half, second_half, mean, max] -> (B, 5, d)
                segs = torch.stack([cls_tok, h_first, h_second, mean_tok, max_tok], dim=1)
                out_segs.append(segs.cpu().numpy().astype(np.float32))
        return np.concatenate(out_segs, axis=0)

    turn_segs = encode_stem(turn_texts, max_len=max_turn_len)
    ctx_segs = encode_stem(ctx_texts, max_len=max_ctx_len)

    per_rec_turn = [None] * len(recs)
    per_rec_ctx = [None] * len(recs)
    buckets_t = [[] for _ in recs]
    buckets_c = [[] for _ in recs]
    for (i, _), ts, cs in zip(map_idx, turn_segs, ctx_segs):
        buckets_t[i].append(ts)
        buckets_c[i].append(cs)
    for i in range(len(recs)):
        per_rec_turn[i] = np.stack(buckets_t[i], axis=0)  # (T, 5, d)
        per_rec_ctx[i] = np.stack(buckets_c[i], axis=0)   # (T, 5, d)
    return per_rec_turn, per_rec_ctx


class FineTunedTransformerTop(nn.Module):
    """Trainable Transformer self-attention block + Task-Attention pooling +
    classification head operating on the 5 sequence segment states (CLS, first-half,
    second-half, mean, max) of either single exchanges (FT-Turn) or full
    conversation prefixes (FT-Context)."""

    def __init__(self, d_model: int = 384, n_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        self.attn_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.task_w = nn.Linear(d_model, 64)
        self.task_v = nn.Linear(64, 1, bias=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def encode_pooled(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, S, d)
        h = self.attn_layer(x)
        scores = self.task_v(torch.tanh(self.task_w(h)))  # (N, S, 1)
        weights = torch.softmax(scores, dim=1)
        return (h * weights).sum(dim=1)  # (N, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encode_pooled(x)
        return self.head(e).squeeze(-1)


class DeepContextNet(nn.Module):
    """Re-implementation of the DeepContext design (Albrethsen et al., 2026,
    arXiv:2602.16935) over frozen MiniLM features; not the authors' model:
      1. Task-Attention Weighted Turn Encoder: Transformer top layer + learned
         attention weights a_{t,l} = softmax(w^T tanh(W H_{t,l})) -> e_t
      2. Recurrent Intent Tracking (RIT): 3-layer GRU over turn embeddings e_1..e_t -> h_t
      3. Projection + Direct Residual Shortcut: R_t = [phi(h_t); e_t]
      4. Trajectory Classifier: LayerNorm + GELU + residual block + linear readout,
         trained with Focal Loss (gamma = 2.0).
    """

    def __init__(self, d_model: int = 384, d_gru: int = 128, dropout: float = 0.2):
        super().__init__()
        self.turn_encoder = FineTunedTransformerTop(d_model=d_model, n_heads=4, dropout=dropout)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_gru,
            num_layers=3,
            batch_first=True,
            dropout=dropout,
        )
        self.proj = nn.Linear(d_gru, d_gru)
        in_cls = d_gru + d_model
        self.norm1 = nn.LayerNorm(in_cls)
        self.fc1 = nn.Linear(in_cls, 128)
        self.res_block = nn.Sequential(
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.out = nn.Linear(128, 1)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, T, S, d)
        B, T, S, d = x_seq.shape
        e = self.turn_encoder.encode_pooled(x_seq.reshape(B * T, S, d)).reshape(B, T, d)
        h, _ = self.gru(e)  # (B, T, d_gru)
        r = torch.cat([F.gelu(self.proj(h)), e], dim=-1)  # residual shortcut [phi(h_t); e_t]
        z = F.gelu(self.fc1(self.norm1(r)))
        z = z + self.res_block(z)
        return self.out(z).squeeze(-1)  # (B, T) logits


def focal_loss_with_logits(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, gamma: float = 2.0, pos_weight: float = 1.0
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    modulating = (1.0 - p_t).pow(gamma)
    alpha_w = targets * pos_weight + (1.0 - targets)
    loss = ce * modulating * alpha_w * mask
    return loss.sum() / mask.sum().clamp(min=1.0)


def exchange_texts(recs: list[dict]) -> tuple[list[list[str]], list[list[str]]]:
    """Per session, the text of each exchange alone and of each causal prefix."""
    turn, ctx = [], []
    for r in recs:
        cur = [f"User: {(t['user'] or '').strip()[:600]}\nAssistant: {(t['assistant'] or '').strip()[:600]}"
               for t in r["turns"]]
        turn.append(cur)
        ctx.append(["\n".join(cur[: k + 1]) for k in range(len(cur))])
    return turn, ctx


class MiniLMClassifier(nn.Module):
    """all-MiniLM-L6-v2 with every layer trainable, mean pooling, linear head."""

    def __init__(self, base_name: str):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base_name, local_files_only=True)
        self.head = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, enc: dict) -> torch.Tensor:
        h = self.enc(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1).clamp(min=1.0)).squeeze(-1)


def fit_and_score_ft(
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    texts: list[list[str]],
    labels: np.ndarray,
    tok,
    device: str,
    max_len: int,
    epochs: int = FT_EPOCHS,
    lr: float = FT_LR,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    """Fine-tune MiniLM end-to-end on exchange (or prefix) texts, each carrying
    its session label and weight 1/T so every session counts equally, and
    return per-exchange probabilities for the sessions in eval_idx."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = [x for i in train_idx for x in texts[i]]
    y = np.concatenate([[labels[i]] * len(texts[i]) for i in train_idx]).astype(np.float32)
    w = np.concatenate([[1.0 / len(texts[i])] * len(texts[i]) for i in train_idx]).astype(np.float32)
    pos_frac = max(float(np.average(y, weights=w)), 0.02)
    w = (w * np.where(y == 1, min((1.0 - pos_frac) / pos_frac, 15.0), 1.0)).astype(np.float32)

    model = MiniLMClassifier(MODEL_NAME).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    batch_size = 32
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(len(X))
        for b in range(0, len(X), batch_size):
            idx = perm[b : b + batch_size]
            enc = tok([X[k] for k in idx], padding=True, truncation=True, max_length=max_len,
                      return_tensors="pt").to(device)
            yt = torch.tensor(y[idx], device=device)
            wt = torch.tensor(w[idx], device=device)
            loss = (F.binary_cross_entropy_with_logits(model(enc), yt, reduction="none") * wt).sum() / wt.sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    out = {}
    with torch.no_grad():
        for i in eval_idx:
            enc = tok(texts[i], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            out[int(i)] = torch.sigmoid(model(enc)).cpu().numpy()
    del model, opt
    if device == "mps":
        torch.mps.empty_cache()
    return out


def fit_and_score_deepcontext(
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    turn_segs: list[np.ndarray],
    labels: np.ndarray,
    device: str,
    epochs: int = 12,
    lr: float = 8e-4,
    seed: int = 0,
) -> dict[int, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    max_T = max(len(turn_segs[i]) for i in train_idx)
    S, d = turn_segs[0].shape[1], turn_segs[0].shape[2]
    N = len(train_idx)

    X_pad = np.zeros((N, max_T, S, d), dtype=np.float32)
    Y_pad = np.zeros((N, max_T), dtype=np.float32)
    M_pad = np.zeros((N, max_T), dtype=np.float32)
    for k, i in enumerate(train_idx):
        T = len(turn_segs[i])
        X_pad[k, :T] = turn_segs[i]
        Y_pad[k, :T] = float(labels[i])
        # Late-turn emphasis + uniform coverage as in trajectory training
        M_pad[k, :T] = np.linspace(0.5, 1.0, T, dtype=np.float32) / T

    pos_frac = max(float(labels[train_idx].mean()), 0.02)
    pos_w = min((1.0 - pos_frac) / pos_frac, 15.0)

    model = DeepContextNet(d_model=d, d_gru=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    Xt = torch.tensor(X_pad, dtype=torch.float32, device=device)
    Yt = torch.tensor(Y_pad, dtype=torch.float32, device=device)
    Mt = torch.tensor(M_pad, dtype=torch.float32, device=device)

    batch_size = 64
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        for b in range(0, N, batch_size):
            idx = perm[b : b + batch_size]
            logits = model(Xt[idx])
            loss = focal_loss_with_logits(logits, Yt[idx], Mt[idx], gamma=2.0, pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    out = {}
    with torch.no_grad():
        for i in eval_idx:
            xi = torch.tensor(turn_segs[i], dtype=torch.float32, device=device).unsqueeze(0)
            probs = torch.sigmoid(model(xi))[0].cpu().numpy()
            out[int(i)] = probs
    return out


def evaluate_corpus_text_models(corpus: str, tok, base_model, device: str) -> dict:
    t0 = time.time()
    recs = load_corpus_sessions(corpus)
    y = np.array([r["label"] for r in recs])
    groups = np.array([r["group"] for r in recs])
    turn_segs, _ = extract_frozen_stem_features(recs, tok, base_model, device)
    turn_txt, ctx_txt = exchange_texts(recs)
    print(f"[{corpus}] stem features extracted for {len(recs)} sessions ({time.time()-t0:.1f}s)", flush=True)

    det_names = ["Guard-Turn", "Turn-LR", "THRD", "CRA-Acc", "FT-Turn", "FT-Context", "DeepContext"]
    turn_oof = {n: [None] * len(recs) for n in det_names}
    theta_oof = {n: np.zeros(len(recs)) for n in det_names}

    from detectors import FIXED, TurnLR

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    rng = np.random.default_rng(0)
    for fold_i, (tr, te) in enumerate(skf.split(np.zeros(len(recs)), y, groups)):
        tr_groups = np.unique(groups[tr])
        cal_groups = set(rng.permutation(tr_groups)[: max(1, int(0.2 * len(tr_groups)))])
        cal = np.array([i for i in tr if groups[i] in cal_groups])
        fit = np.array([i for i in tr if groups[i] not in cal_groups])
        eval_all = np.concatenate([cal, te])

        # 1. Standard guard baselines on the identical split
        train_recs = [recs[i] for i in fit]
        for n in ("Guard-Turn", "THRD"):
            det = FIXED[n]
            cal_s = np.array([det(recs[i]).max() for i in cal])
            th = calibrate(cal_s[y[cal] == 0])
            for i in te:
                turn_oof[n][i] = det(recs[i])
                theta_oof[n][i] = th

        for n, cls in (("Turn-LR", lambda: TurnLR("L")), ("CRA-Acc", lambda: CRAAcc("LCX"))):
            det = cls().fit(train_recs)
            cal_s = np.array([det.turn_scores(recs[i]).max() for i in cal])
            th = calibrate(cal_s[y[cal] == 0])
            for i in te:
                turn_oof[n][i] = det.turn_scores(recs[i])
                theta_oof[n][i] = th

        # 2. FT-Turn (single-exchange fine-tuned transformer)
        ft_turn_scores = fit_and_score_ft(fit, eval_all, turn_txt, y, tok, device, FT_MAX_TURN_TOKENS, seed=fold_i)
        cal_s = np.array([ft_turn_scores[int(i)].max() for i in cal])
        th = calibrate(cal_s[y[cal] == 0])
        for i in te:
            turn_oof["FT-Turn"][i] = ft_turn_scores[int(i)]
            theta_oof["FT-Turn"][i] = th

        # 3. FT-Context (full-conversation-prefix fine-tuned transformer)
        ft_ctx_scores = fit_and_score_ft(fit, eval_all, ctx_txt, y, tok, device, FT_MAX_CTX_TOKENS, seed=fold_i)
        cal_s = np.array([ft_ctx_scores[int(i)].max() for i in cal])
        th = calibrate(cal_s[y[cal] == 0])
        for i in te:
            turn_oof["FT-Context"][i] = ft_ctx_scores[int(i)]
            theta_oof["FT-Context"][i] = th

        # 4. DeepContext (Task-Attention + 3-layer GRU + Residual Shortcut + Focal Loss)
        dc_scores = fit_and_score_deepcontext(fit, eval_all, turn_segs, y, device, seed=fold_i)
        cal_s = np.array([dc_scores[int(i)].max() for i in cal])
        th = calibrate(cal_s[y[cal] == 0])
        for i in te:
            turn_oof["DeepContext"][i] = dc_scores[int(i)]
            theta_oof["DeepContext"][i] = th

    session_scores = {n: np.array([t.max() for t in turn_oof[n]]) for n in det_names}
    pairs = [
        ("FT-Context", "FT-Turn"),
        ("DeepContext", "FT-Turn"),
        ("DeepContext", "Turn-LR"),
        ("CRA-Acc", "DeepContext"),
        ("CRA-Acc", "FT-Context"),
        ("THRD", "Guard-Turn"),
    ]
    ci, dci = boot_ci(y, session_scores, groups, pairs)

    metrics = {}
    for n in det_names:
        s = session_scores[n]
        alarm = s >= theta_oof[n]
        metrics[n] = {
            "auroc": auroc(y, s),
            "auroc_ci95": ci[n],
            "auprc": float(average_precision_score(y, s)),
            "tpr_at_op": float(alarm[y == 1].mean()),
            "fpr_at_op": float(alarm[y == 0].mean()),
        }

    print(
        f"[{corpus}] done in {time.time()-t0:.1f}s: "
        + ", ".join(f"{n}={metrics[n]['auroc']:.3f}" for n in det_names),
        flush=True,
    )
    return {
        "corpus": corpus,
        "n": len(recs),
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
        "n_groups": int(len(set(groups))),
        "metrics": metrics,
        "paired_differences": dci,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="*", default=CORPORA)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    tok.truncation_side = "left"  # keep the latest turns of a long prefix
    base_model = AutoModel.from_pretrained(MODEL_NAME, local_files_only=True).to(device).eval()

    out_path = RESULTS / "text_models.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for c in args.corpora:
        results[c] = evaluate_corpus_text_models(c, tok, base_model, device)
        out_path.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
