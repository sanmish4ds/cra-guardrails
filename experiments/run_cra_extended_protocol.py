#!/usr/bin/env python3
"""Extended CRA-Bench v0.1 evaluation:

  (A) Per-family AUROC breakdown on the standard 60/20/20 stratified split
      (same as run_cra_primary_protocol.py) for every baseline and every
      CRA-Net variant. Reveals whether the headline number is carried by one
      easy family or spread across all three.

  (B) Leave-One-Family-Out (LOFO) generalization: for each held-out family F,
      train CRA-Net on the other two families (and calibrate the operating
      threshold on a held-out val from those two), then score the complete F
      partition. This is the fairest test of trajectory features vs. template
      memorization.

  (C) Benign multi-turn false-alarm rate on ShareGPT, scored with the
      CRA-Bench-calibrated theta from the full CRA-Net (lambda=0.05) trained
      on the standard split. Closes the loop that the CoSafe-calibrated theta
      left at FPR = 1.0.

Output: experiments/results/cra_extended_protocol.json
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
BENCH_PATH = PROJECT_ROOT / "data" / "cra_bench_v01" / "sessions.jsonl"
OUT_PATH = RESULTS_DIR / "cra_extended_protocol.json"

SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2
LAMBDA_GRL = 0.05
EPOCHS = 50
LR = 1e-3
BATCH = 32
GRU_H = 128
GRU_LAYERS = 2
N_BOOT = 1000
N_BENIGN_TARGET = 1000
# Family-adversarial weight on the standard split. The default is chosen so
# the 3-family regime is a Pareto win; 5-family regimes do better with the
# softer 0.1 setting (see Section "Sweeping lam_fam"). Override via CLI.
LAM_FAM_STD_DEFAULT = 0.3
LAM_FAM_LOFO_DEFAULT = 0.5
LAM_FAM_STD = LAM_FAM_STD_DEFAULT
LAM_FAM_LOFO = LAM_FAM_LOFO_DEFAULT
LAM_CORAL_LOFO_DEFAULT = 1.0
LAM_CORAL_LOFO = LAM_CORAL_LOFO_DEFAULT

KNOWN_FAMILIES = ("fragmentation", "conditioning", "aggregation",
                  "persona", "stuffing")
# Auto-detected at runtime from the loaded bench (positive families only).
POSITIVE_FAMILIES: tuple[str, ...] = ()
BENIGN_FAMILIES: tuple[str, ...] = ()
ALL_FAMILIES: tuple[str, ...] = ()


def autodetect_families(sessions: list[dict]) -> tuple[str, ...]:
    """Read unique positive-family names from the loaded sessions, preserving
    the canonical order in KNOWN_FAMILIES so reports are deterministic."""
    seen = {s["family"] for s in sessions if s["label"] == 1}
    return tuple(f for f in KNOWN_FAMILIES if f in seen) + \
           tuple(sorted(f for f in seen if f not in KNOWN_FAMILIES))

# ─── helpers shared with primary protocol ────────────────────────────────────

def _load_mod(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_bench(path: Path) -> list[dict]:
    sessions = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sessions.append({
                "id": row["session_id"],
                "label": int(row["label"]),
                "turns": row["turns"],
                "onset_turn": int(row.get("onset_turn", 0)),
                "cra_type": row.get("cra_type", "bench"),
                "family": row.get("cra_type", "bench").replace("_benign_twin", ""),
                "meta": row,
            })
    return sessions


def stratified_split(sessions: list[dict], seed: int = SEED,
                     train_frac: float = TRAIN_FRAC,
                     val_frac: float = VAL_FRAC):
    rng = np.random.default_rng(seed)
    pos = [s for s in sessions if s["label"] == 1]
    neg = [s for s in sessions if s["label"] == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def _split(lst):
        n = len(lst)
        n_tr = int(n * train_frac)
        n_va = int(n * val_frac)
        return lst[:n_tr], lst[n_tr:n_tr + n_va], lst[n_tr + n_va:]

    tr_p, va_p, te_p = _split(pos)
    tr_n, va_n, te_n = _split(neg)
    train = tr_p + tr_n
    val = va_p + va_n
    test = te_p + te_n
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ─── feature scorers (copied / aligned with primary protocol) ────────────────
ALPHA, BETA, GAMMA = 0.35, 0.45, 0.20


def convex_session_score(rec: dict, ablate: str | None = None) -> float:
    scores = []
    for row in rec["features"]:
        s1, s2, s3 = row[0], row[1], row[2]
        if ablate == "S1":
            s1 = 0.0
        elif ablate == "S2":
            s2 = 0.0
        elif ablate == "S3":
            s3 = 0.0
        scores.append(ALPHA * s1 + BETA * s2 + GAMMA * s3)
    return float(max(scores)) if scores else 0.0


def turn_max_s1(rec: dict) -> float:
    return float(max(row[0] for row in rec["features"])) if rec["features"] else 0.0


def sliding_ema_s1(rec: dict, alpha: float = 0.5) -> float:
    ema = 0.0
    peak = 0.0
    for row in rec["features"]:
        s1 = row[0]
        ema = alpha * s1 + (1 - alpha) * ema
        peak = max(peak, ema)
    return peak


def fit_mlp(train_recs: list[dict], epochs: int = 200, lr: float = 0.05):
    dim = len(train_recs[0]["features"][0])
    rng = np.random.default_rng(SEED)
    W = rng.normal(0, 0.1, size=dim)
    b = 0.0
    for _ in range(epochs):
        for r in train_recs:
            X = np.array(r["features"], dtype=np.float64)
            if len(X) == 0:
                continue
            h = X.max(axis=0)
            z = float(np.dot(W, h) + b)
            p = 1.0 / (1.0 + math.exp(-z))
            y = float(r["label"])
            err = p - y
            W -= lr * err * h
            b -= lr * err
    return {"W": W, "b": b}


def mlp_score(rec: dict, weights: dict) -> float:
    X = np.array(rec["features"], dtype=np.float64)
    if len(X) == 0:
        return 0.0
    h = X.max(axis=0)
    z = float(np.dot(weights["W"], h) + weights["b"])
    return 1.0 / (1.0 + math.exp(-z))


# ─── CRA-Net training wrapper ────────────────────────────────────────────────

def train_cranet(rcn, train_recs, val_recs, device, lam: float = LAMBDA_GRL,
                 input_dim: int = 5, epochs: int = EPOCHS):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    max_len = max(r["n_turns"] for r in train_recs + val_recs)

    model = rcn.CRANet(input_dim=input_dim, hidden=GRU_H,
                       n_layers=GRU_LAYERS, lam=lam).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    train_ds = rcn.SessionDataset(train_recs, max_len)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    bce = nn.BCELoss(reduction="none")
    mse = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for x, y, lengths, tw in train_dl:
            x, y, tw = x.to(device), y.to(device), tw.to(device)
            risk, len_pred = model(x, lengths)
            y_exp = y.unsqueeze(1).expand_as(risk)
            loss_cls = (bce(risk, y_exp) * tw).sum() / (tw.sum() + 1e-8)
            if lam > 0 and len_pred is not None:
                log_len = torch.log(1 + lengths.float()) / math.log(21)
                loss_len = mse(len_pred.to(device), log_len.to(device))
            else:
                loss_len = torch.tensor(0.0, device=device)
            loss = loss_cls + loss_len
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    return model, max_len


def predict_cranet(model, rcn, recs, device, max_len):
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    out_recs = [dict(r) for r in recs]
    ds = rcn.SessionDataset(out_recs, max_len)
    dl = DataLoader(ds, batch_size=BATCH)
    scores = []
    bi = 0
    with torch.no_grad():
        for x, y, lengths, _ in dl:
            out = model(x.to(device), lengths)
            risk = out[0]  # works for both CRANet (2-tuple) and CRANetDA (3-tuple)
            B = x.size(0)
            for i in range(B):
                T = lengths[i].item()
                preds_i = risk[i, :T].cpu().numpy().tolist()
                scores.append(float(max(preds_i)) if preds_i else 0.0)
                out_recs[bi + i]["preds"] = preds_i
            bi += B
    return scores, out_recs


# ─── CRA-Net DA (family-adversarial) training wrapper ────────────────────────

def train_cranet_da(rcn, train_recs, val_recs, device,
                    lam: float = LAMBDA_GRL,
                    lam_fam: float = 0.3,
                    input_dim: int = 5,
                    epochs: int = EPOCHS):
    """Train CRA-Net with both length GRL (weight `lam`) and family GRL
    (weight `lam_fam`). `train_recs` must each carry an integer `family_idx`
    field in [0, n_families). Returns (model, max_len)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import numpy as np

    max_len = max(r["n_turns"] for r in train_recs + val_recs)

    fam_ids = sorted({int(r["family_idx"]) for r in train_recs})
    n_fam = max(fam_ids) + 1 if fam_ids else 2
    assert min(fam_ids) >= 0, "family_idx must be in [0, n_families)"

    model = rcn.CRANetDA(input_dim=input_dim, hidden=GRU_H,
                         n_layers=GRU_LAYERS, lam=lam, lam_fam=lam_fam,
                         n_families=n_fam).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    bce = nn.BCELoss(reduction="none")
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    # ── custom dataset that also returns family_idx
    class _SessionDSWithFam(Dataset):
        def __init__(self, recs, max_len):
            self.recs = recs
            self.max_len = max_len

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, idx):
            r = self.recs[idx]
            feat = np.array(r["features"], dtype=np.float32)
            T = len(feat)
            pad = np.zeros((self.max_len, feat.shape[1]), dtype=np.float32)
            pad[:T] = feat
            label = float(r["label"])
            tw = np.zeros(self.max_len, dtype=np.float32)
            for t in range(T):
                tw[t] = ((t + 1) / max(T, 1)) ** rcn.RHO
            fam = int(r["family_idx"])
            return (
                torch.from_numpy(pad),
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(T, dtype=torch.long),
                torch.from_numpy(tw),
                torch.tensor(fam, dtype=torch.long),
            )

    train_ds = _SessionDSWithFam(train_recs, max_len)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

    for _ in range(epochs):
        model.train()
        for x, y, lengths, tw, fam in train_dl:
            x = x.to(device); y = y.to(device); tw = tw.to(device); fam = fam.to(device)
            risk, len_pred, fam_logits = model(x, lengths)
            y_exp = y.unsqueeze(1).expand_as(risk)
            loss_cls = (bce(risk, y_exp) * tw).sum() / (tw.sum() + 1e-8)
            if lam > 0 and len_pred is not None:
                log_len = torch.log(1 + lengths.float()) / math.log(21)
                loss_len = mse(len_pred.to(device), log_len.to(device))
            else:
                loss_len = torch.tensor(0.0, device=device)
            if lam_fam > 0 and fam_logits is not None:
                loss_fam = ce(fam_logits, fam)
            else:
                loss_fam = torch.tensor(0.0, device=device)
            loss = loss_cls + loss_len + loss_fam
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    return model, max_len


def train_cranet_coral(rcn, train_recs, val_recs, device,
                       lam: float = LAMBDA_GRL,
                       lam_coral: float = LAM_CORAL_LOFO_DEFAULT,
                       input_dim: int = 5,
                       epochs: int = EPOCHS):
    """CRA-Net + Deep CORAL (Sun & Saenko 2016) on encoder last states."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import numpy as np

    max_len = max(r["n_turns"] for r in train_recs + val_recs)
    model = rcn.CRANet(input_dim=input_dim, hidden=GRU_H,
                       n_layers=GRU_LAYERS, lam=lam).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    bce = nn.BCELoss(reduction="none")
    mse = nn.MSELoss()

    class _SessionDSWithFam(Dataset):
        def __init__(self, recs, max_len):
            self.recs = recs
            self.max_len = max_len

        def __len__(self):
            return len(self.recs)

        def __getitem__(self, idx):
            r = self.recs[idx]
            feat = np.array(r["features"], dtype=np.float32)
            T = len(feat)
            pad = np.zeros((self.max_len, feat.shape[1]), dtype=np.float32)
            pad[:T] = feat
            label = float(r["label"])
            tw = np.zeros(self.max_len, dtype=np.float32)
            for t in range(T):
                tw[t] = ((t + 1) / max(T, 1)) ** rcn.RHO
            fam = int(r["family_idx"])
            return (
                torch.from_numpy(pad),
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(T, dtype=torch.long),
                torch.from_numpy(tw),
                torch.tensor(fam, dtype=torch.long),
            )

    train_ds = _SessionDSWithFam(train_recs, max_len)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)

    for _ in range(epochs):
        model.train()
        for x, y, lengths, tw, fam in train_dl:
            x = x.to(device)
            y = y.to(device)
            tw = tw.to(device)
            fam = fam.to(device)
            risk, len_pred = model(x, lengths)
            y_exp = y.unsqueeze(1).expand_as(risk)
            loss_cls = (bce(risk, y_exp) * tw).sum() / (tw.sum() + 1e-8)
            if lam > 0 and len_pred is not None:
                log_len = torch.log(1 + lengths.float()) / math.log(21)
                loss_len = mse(len_pred.to(device), log_len.to(device))
            else:
                loss_len = torch.tensor(0.0, device=device)
            out, _ = model.gru(x)
            idx = (lengths - 1).clamp(min=0).long()
            B = out.size(0)
            last_h = out[torch.arange(B, device=device), idx]
            loss_coral = rcn.batch_coral_loss(last_h, fam)
            loss = loss_cls + loss_len + lam_coral * loss_coral
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    return model, max_len


def assign_family_idx(records: list[dict], training_families: list[str]) -> list[dict]:
    """Annotate each record with `family_idx` mapping the (positive) family
    name to a contiguous integer. Benign-twin records use the same index as
    their positive counterpart (so the family head learns to ignore label and
    focus on family identity)."""
    fmap = {f: i for i, f in enumerate(training_families)}
    out = []
    for r in records:
        fam = r.get("family") or ""
        fam_clean = fam.replace("_benign_twin", "")
        if fam_clean not in fmap:
            continue  # records from other families are excluded
        r = dict(r)
        r["family_idx"] = int(fmap[fam_clean])
        out.append(r)
    return out


# ─── evaluation helpers ──────────────────────────────────────────────────────

def eval_block(y_true, y_score, rcn, recs_for_ttd=None):
    auroc = rcn._auc(y_true, y_score)
    sfpr, thr = rcn.compute_sfpr(y_true, y_score)
    ttd = rcn.compute_ttd(recs_for_ttd, thr) if recs_for_ttd else None
    _, auc_lo, auc_hi = rcn.bootstrap_auc(y_true, y_score, n_boot=N_BOOT, seed=SEED)
    sfpr_lo, sfpr_hi = rcn.bootstrap_sfpr(y_true, y_score, tpr_target=0.90,
                                          n_boot=N_BOOT, seed=SEED)
    if recs_for_ttd:
        ttd_lo, ttd_hi = rcn.bootstrap_ttd(recs_for_ttd, thr,
                                           n_boot=N_BOOT, seed=SEED)
    else:
        ttd_lo, ttd_hi = float("nan"), float("nan")
    return {
        "n": len(y_true),
        "n_pos": int(sum(y_true)),
        "n_neg": int(len(y_true) - sum(y_true)),
        "auroc": None if math.isnan(auroc) else round(auroc, 4),
        "auroc_ci95": [round(auc_lo, 4), round(auc_hi, 4)],
        "sfpr_at_tpr90": round(sfpr, 4),
        "sfpr_ci95": [round(sfpr_lo, 4), round(sfpr_hi, 4)],
        "threshold": round(float(thr), 6),
        "mean_ttd": None if ttd is None else round(ttd, 4),
        "ttd_ci95": [
            None if math.isnan(ttd_lo) else round(ttd_lo, 4),
            None if math.isnan(ttd_hi) else round(ttd_hi, 4),
        ],
    }


def per_family_auroc(records: list[dict], scores: list[float]) -> dict:
    """Group (record, score) by family and compute AUROC within each family
    pool, where the pool is (positives of family X) U (benign twins of family X).
    Returns {family: {n, n_pos, n_neg, auroc, auroc_ci95}}."""
    import math
    out: dict[str, dict] = {}
    by_fam: dict[str, list[tuple[float, int]]] = {}
    for r, s in zip(records, scores):
        fam = r.get("family") or r.get("meta", {}).get("cra_type", "")\
            .replace("_benign_twin", "")
        by_fam.setdefault(fam, []).append((float(s), int(r["label"])))
    rng = np.random.default_rng(SEED)
    for fam, pairs in sorted(by_fam.items()):
        ys = [p[1] for p in pairs]
        ss = [p[0] for p in pairs]
        n_pos = sum(ys)
        n_neg = len(ys) - n_pos
        if n_pos == 0 or n_neg == 0:
            out[fam] = {"n": len(ys), "n_pos": n_pos, "n_neg": n_neg,
                        "auroc": None, "auroc_ci95": [None, None]}
            continue
        a = _local_auc(ys, ss)
        boot = []
        arr_y = np.asarray(ys)
        arr_s = np.asarray(ss, dtype=float)
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(ys), size=len(ys))
            yb = arr_y[idx].tolist()
            sb = arr_s[idx].tolist()
            if sum(yb) == 0 or sum(yb) == len(yb):
                continue
            boot.append(_local_auc(yb, sb))
        if boot:
            lo = float(np.percentile(boot, 2.5))
            hi = float(np.percentile(boot, 97.5))
        else:
            lo, hi = float("nan"), float("nan")
        out[fam] = {
            "n": len(ys),
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "auroc": round(a, 4) if not math.isnan(a) else None,
            "auroc_ci95": [None if math.isnan(lo) else round(lo, 4),
                            None if math.isnan(hi) else round(hi, 4)],
        }
    return out


def _local_auc(y_true, y_score) -> float:
    pairs = sorted(zip(y_score, y_true), reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = 0
    auc = 0.0
    for _, yl in pairs:
        if yl == 1:
            tp += 1
        else:
            auc += tp
    return auc / (n_pos * n_neg)


# ─── benign FPR ──────────────────────────────────────────────────────────────

def load_benign_sharegpt(n_target: int, rcc) -> list[dict]:
    bfp = _load_mod("bfp", "run_benign_fpr_sharegpt.py")
    return bfp.load_sharegpt(n_target, seed=SEED + 7)


def main() -> int:
    import torch
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / ".env.local", override=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, default=BENCH_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH,
                    help="Where to write the results JSON.")
    ap.add_argument("--skip-benign", action="store_true")
    ap.add_argument("--benign-n", type=int, default=N_BENIGN_TARGET)
    ap.add_argument("--lam-fam-std", type=float, default=LAM_FAM_STD_DEFAULT,
                    help="Family-adversarial weight on the standard split.")
    ap.add_argument("--lam-fam-lofo", type=float, default=LAM_FAM_LOFO_DEFAULT,
                    help="Family-adversarial weight in the LOFO loop.")
    ap.add_argument("--lam-coral-lofo", type=float, default=LAM_CORAL_LOFO_DEFAULT,
                    help="Deep CORAL weight in the LOFO loop.")
    ap.add_argument("--s3-mode", choices=("keyword", "classifier"), default="keyword",
                    help="S3 refusal signal: keyword proxy or DistilBERT classifier.")
    ap.add_argument("--fast", action="store_true",
                    help="15 epochs, 100 bootstrap samples, skip LOFO & benign FPR, signal cache.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--skip-lofo", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    global LAM_FAM_STD, LAM_FAM_LOFO, LAM_CORAL_LOFO, EPOCHS, N_BOOT
    LAM_FAM_STD = args.lam_fam_std
    LAM_FAM_LOFO = args.lam_fam_lofo
    LAM_CORAL_LOFO = args.lam_coral_lofo
    if args.fast:
        args.skip_benign = True
        args.skip_lofo = True
        EPOCHS = args.epochs or 15
        N_BOOT = args.n_boot or 100
    else:
        if args.epochs is not None:
            EPOCHS = args.epochs
        if args.n_boot is not None:
            N_BOOT = args.n_boot

    rcc = _load_mod("rcc", "run_cra_cosafe.py")
    rcn = _load_mod("rcn", "run_cranet.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("[Ext] Loading CRA-Bench v0.1 ...")
    sessions = load_bench(args.bench)

    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    nlp = rcc._load_spacy()

    s3_clf = None
    if args.s3_mode == "classifier":
        from s3_refusal import ClassifierS3
        s3_clf = ClassifierS3()
        if not s3_clf.available():
            raise SystemExit(
                f"[Ext] --s3-mode classifier requires {s3_clf.model_dir} "
                "(run experiments/train_s3_refusal_classifier.py first)."
            )
    use_cache = not args.no_cache
    if use_cache:
        from signal_cache import load_or_extract
        print(f"[Ext] Signals (S3={args.s3_mode}, cache=on) ...")
        all_records = load_or_extract(
            args.bench, sessions, rcn, sbert, nlp,
            s3_mode=args.s3_mode, s3_clf=s3_clf)
    else:
        print(f"[Ext] Extracting per-turn signals (S3={args.s3_mode}) ...")
        all_records = rcn.extract_signals(
            sessions, sbert, nlp, s3_mode=args.s3_mode, s3_clf=s3_clf)
    for r, s in zip(all_records, sessions):
        r["session_id"] = s["id"]
        r["family"] = s["family"]
        r["cra_type"] = s["cra_type"]

    by_id = {r["session_id"]: r for r in all_records}

    # Detect families dynamically so the same script supports 3, 4, or 5
    # families without code edits.
    global POSITIVE_FAMILIES, BENIGN_FAMILIES, ALL_FAMILIES
    POSITIVE_FAMILIES = autodetect_families(sessions)
    BENIGN_FAMILIES = tuple(f + "_benign_twin" for f in POSITIVE_FAMILIES)
    ALL_FAMILIES = POSITIVE_FAMILIES + BENIGN_FAMILIES
    print(f"[Ext] families detected: {POSITIVE_FAMILIES}")

    bench_versions = sorted({s["meta"].get("bench_version", "v0.1")
                              for s in sessions})
    results: dict[str, Any] = {
        "protocol": "cra_extended_v1",
        "bench_path": str(args.bench),
        "bench_versions": bench_versions,
        "s3_mode": args.s3_mode,
        "fast_mode": args.fast,
        "epochs": EPOCHS,
        "n_boot": N_BOOT,
        "n_sessions_total": len(sessions),
        "families": list(POSITIVE_FAMILIES),
    }

    # ── A. Standard split with per-family breakdown ──────────────────────────
    print("[Ext-A] Standard 60/20/20 split + per-family AUROC")
    train_s, val_s, test_s = stratified_split(sessions)
    train_r = [by_id[s["id"]] for s in train_s]
    val_r = [by_id[s["id"]] for s in val_s]
    test_r = [by_id[s["id"]] for s in test_s]

    y_te = [r["label"] for r in test_r]
    methods: dict[str, list[float]] = {
        "CRA-convex":             [convex_session_score(r) for r in test_r],
        "Turn-max S1":            [turn_max_s1(r) for r in test_r],
        "Sliding-window S1":      [sliding_ema_s1(r) for r in test_r],
    }
    methods["CRA-convex \\ S1"] = [convex_session_score(r, "S1") for r in test_r]
    methods["CRA-convex \\ S2"] = [convex_session_score(r, "S2") for r in test_r]
    methods["CRA-convex \\ S3"] = [convex_session_score(r, "S3") for r in test_r]

    mlp_w = fit_mlp(train_r)
    methods["Feature-MLP"] = [mlp_score(r, mlp_w) for r in test_r]

    # Train CRA-Net variants on the standard split
    print("  Training CRA-Net (lambda=0.05) ...")
    model_full, max_len = train_cranet(rcn, train_r, val_r, device,
                                       lam=LAMBDA_GRL, input_dim=5)
    _, val_out_full = predict_cranet(model_full, rcn, val_r, device, max_len)
    val_scores_full = [max(r["preds"]) if r["preds"] else 0.0 for r in val_out_full]
    val_labels_full = [r["label"] for r in val_out_full]
    _, theta_full = rcn.compute_sfpr(val_labels_full, val_scores_full)
    test_scores_full, test_out_full = predict_cranet(
        model_full, rcn, test_r, device, max_len)
    methods["CRA-Net (lambda=0.05)"] = test_scores_full

    print("  Training CRA-Net no-GRL (lambda=0) ...")
    model_nogrl, _ = train_cranet(rcn, train_r, val_r, device, lam=0.0,
                                  input_dim=5)
    test_scores_nogrl, _ = predict_cranet(model_nogrl, rcn, test_r, device, max_len)
    methods["CRA-Net no GRL (lambda=0)"] = test_scores_nogrl

    print("  Training CRA-Net drift-only (no S2/S3) ...")
    train_drift = [{**r, "features": [[row[0], 0.0, 0.0] for row in r["features"]]}
                   for r in train_r]
    val_drift = [{**r, "features": [[row[0], 0.0, 0.0] for row in r["features"]]}
                 for r in val_r]
    test_drift = [{**r, "features": [[row[0], 0.0, 0.0] for row in r["features"]]}
                  for r in test_r]
    model_d, _ = train_cranet(rcn, train_drift, val_drift, device,
                              lam=LAMBDA_GRL, input_dim=3)
    test_scores_d, _ = predict_cranet(model_d, rcn, test_drift, device, max_len)
    methods["CRA-Net no S2/S3"] = test_scores_d

    # CRA-Net DA: family-adversarial (3 families on standard split)
    print("  Training CRA-Net DA (length GRL + family GRL) ...")
    train_da_r = assign_family_idx(train_r, list(POSITIVE_FAMILIES))
    val_da_r = assign_family_idx(val_r, list(POSITIVE_FAMILIES))
    model_da, _ = train_cranet_da(rcn, train_da_r, val_da_r, device,
                                  lam=LAMBDA_GRL, lam_fam=LAM_FAM_STD,
                                  input_dim=5)
    _, val_out_da = predict_cranet(model_da, rcn, val_r, device, max_len)
    val_scores_da = [max(r["preds"]) if r["preds"] else 0.0 for r in val_out_da]
    val_labels_da = [r["label"] for r in val_out_da]
    _, theta_da = rcn.compute_sfpr(val_labels_da, val_scores_da)
    test_scores_da, test_out_da = predict_cranet(
        model_da, rcn, test_r, device, max_len)
    methods["CRA-Net DA (lambda=0.05, lam_fam=0.3)"] = test_scores_da

    overall: dict[str, Any] = {}
    per_family: dict[str, Any] = {}
    for name, sc in methods.items():
        if name == "CRA-Net (lambda=0.05)":
            overall[name] = eval_block(y_te, sc, rcn, recs_for_ttd=test_out_full)
        elif name == "CRA-Net DA (lambda=0.05, lam_fam=0.3)":
            overall[name] = eval_block(y_te, sc, rcn, recs_for_ttd=test_out_da)
        else:
            overall[name] = eval_block(y_te, sc, rcn)
        per_family[name] = per_family_auroc(test_r, sc)

    results["standard_split"] = {
        "splits": {"train": len(train_r), "val": len(val_r), "test": len(test_r)},
        "calibrated_theta_cranet_full": float(theta_full),
        "calibrated_theta_cranet_da": float(theta_da),
        "lam_fam_std": LAM_FAM_STD,
        "overall_test": overall,
        "per_family_test_auroc": per_family,
    }

    # ── B. Leave-One-Family-Out ──────────────────────────────────────────────
    if args.skip_lofo:
        print("[Ext-B] LOFO skipped (--skip-lofo / --fast)")
    else:
        print("[Ext-B] Leave-One-Family-Out (LOFO) - vanilla CRA-Net + CRA-Net DA + CORAL")
    lofo: dict[str, Any] = {}
    lofo_da: dict[str, Any] = {}
    lofo_coral: dict[str, Any] = {}
    if not args.skip_lofo:
        for held in POSITIVE_FAMILIES:
            print(f"  Held-out family: {held}")
            in_pool = [s for s in sessions if s["family"] != held]
            out_pool = [s for s in sessions if s["family"] == held]
            training_families = [f for f in POSITIVE_FAMILIES if f != held]

            in_train, in_val, _ = stratified_split(in_pool, seed=SEED,
                                                    train_frac=0.8, val_frac=0.2)
            tr_r = [by_id[s["id"]] for s in in_train]
            va_r = [by_id[s["id"]] for s in in_val]
            te_r = [by_id[s["id"]] for s in out_pool]

            model_l, max_len_l = train_cranet(rcn, tr_r, va_r, device,
                                              lam=LAMBDA_GRL, input_dim=5)
            _, va_out = predict_cranet(model_l, rcn, va_r, device, max_len_l)
            va_scores = [max(r["preds"]) if r["preds"] else 0.0 for r in va_out]
            va_labels = [r["label"] for r in va_out]
            _, theta_l = rcn.compute_sfpr(va_labels, va_scores)
            te_scores, te_out = predict_cranet(model_l, rcn, te_r, device, max_len_l)
            te_labels = [r["label"] for r in te_r]
            block = eval_block(te_labels, te_scores, rcn, recs_for_ttd=te_out)
            lofo[held] = {
                "n_train": len(tr_r),
                "n_val": len(va_r),
                "n_test": len(te_r),
                "calibrated_theta": float(theta_l),
                "metrics": block,
            }

            tr_da = assign_family_idx(tr_r, training_families)
            va_da = assign_family_idx(va_r, training_families)
            model_da_l, _ = train_cranet_da(rcn, tr_da, va_da, device,
                                            lam=LAMBDA_GRL,
                                            lam_fam=LAM_FAM_LOFO,
                                            input_dim=5)
            _, va_out_da = predict_cranet(model_da_l, rcn, va_r, device, max_len_l)
            va_scores_da = [max(r["preds"]) if r["preds"] else 0.0
                            for r in va_out_da]
            _, theta_da_l = rcn.compute_sfpr(va_labels, va_scores_da)
            te_scores_da, te_out_da = predict_cranet(
                model_da_l, rcn, te_r, device, max_len_l)
            block_da = eval_block(te_labels, te_scores_da, rcn,
                                  recs_for_ttd=te_out_da)
            lofo_da[held] = {
                "n_train": len(tr_da),
                "n_val": len(va_da),
                "n_test": len(te_r),
                "lam_fam": LAM_FAM_LOFO,
                "calibrated_theta": float(theta_da_l),
                "metrics": block_da,
            }

            tr_coral = assign_family_idx(tr_r, training_families)
            va_coral = assign_family_idx(va_r, training_families)
            model_coral_l, max_len_c = train_cranet_coral(
                rcn, tr_coral, va_coral, device,
                lam=LAMBDA_GRL, lam_coral=LAM_CORAL_LOFO, input_dim=5)
            _, va_out_coral = predict_cranet(
                model_coral_l, rcn, va_r, device, max_len_c)
            va_scores_coral = [max(r["preds"]) if r["preds"] else 0.0
                               for r in va_out_coral]
            _, theta_coral_l = rcn.compute_sfpr(va_labels, va_scores_coral)
            te_scores_coral, te_out_coral = predict_cranet(
                model_coral_l, rcn, te_r, device, max_len_c)
            block_coral = eval_block(te_labels, te_scores_coral, rcn,
                                     recs_for_ttd=te_out_coral)
            lofo_coral[held] = {
                "n_train": len(tr_coral),
                "n_val": len(va_coral),
                "n_test": len(te_r),
                "lam_coral": LAM_CORAL_LOFO,
                "calibrated_theta": float(theta_coral_l),
                "metrics": block_coral,
            }
    results["lofo_cranet"] = lofo
    results["lofo_cranet_da"] = lofo_da
    results["lofo_cranet_coral"] = lofo_coral

    # ── C. Benign FPR with CRA-Net at CRA-Bench-calibrated threshold ─────────
    if not args.skip_benign:
        print(f"[Ext-C] Benign FPR on ShareGPT (target n={args.benign_n}) ...")
        try:
            benign = load_benign_sharegpt(min(args.benign_n, 1200), rcc)
        except Exception as e:
            print(f"  Failed to load ShareGPT ({e}); skipping benign FPR")
            benign = []

        if benign:
            print(f"  Extracting signals on {len(benign)} benign sessions ...")
            benign_recs_all = rcn.extract_signals(benign, sbert, nlp)
            for r, s in zip(benign_recs_all, benign):
                r["session_id"] = s.get("id", "")
            n_before = len(benign_recs_all)
            benign_recs = [r for r in benign_recs_all
                           if 2 <= r["n_turns"] <= 16]
            print(f"  After length filter [2..16 user turns]: "
                  f"{len(benign_recs)} / {n_before}")

            if benign_recs:
                convex_scores = [convex_session_score(r) for r in benign_recs]
                benign_max_len = max(
                    max_len, max(r["n_turns"] for r in benign_recs)
                )
                cranet_scores, _ = predict_cranet(
                    model_full, rcn, benign_recs, device, benign_max_len
                )
                cranet_da_scores, _ = predict_cranet(
                    model_da, rcn, benign_recs, device, benign_max_len
                )

                theta = float(theta_full)
                theta_da_op = float(theta_da)
                convex_exceed = sum(1 for s in convex_scores if s >= theta)
                cranet_exceed = sum(1 for s in cranet_scores if s >= theta)
                cranet_da_exceed = sum(1 for s in cranet_da_scores
                                        if s >= theta_da_op)

                # ─── Benign-anchored calibration ────────────────────────────
                # Pick theta_benign so that benign FPR <= budget; then report
                # TPR on the CRA-Bench test split at theta_benign. This is
                # the deployment-side "FPR-budget" recipe.
                def _benign_anchored(benign_scores, fpr_budget, te_scores, y_te):
                    sorted_b = sorted(benign_scores)
                    k = max(0, int(math.ceil(len(sorted_b) * (1 - fpr_budget))) - 1)
                    theta_b = sorted_b[k] if sorted_b else 1.0
                    n_pos = sum(y_te)
                    if n_pos == 0:
                        return float(theta_b), 0.0
                    tp = sum(1 for s, y in zip(te_scores, y_te)
                             if y == 1 and s >= theta_b)
                    return float(theta_b), tp / n_pos

                test_scores_full_bench, _ = predict_cranet(
                    model_full, rcn, test_r, device, benign_max_len)
                test_scores_da_bench, _ = predict_cranet(
                    model_da, rcn, test_r, device, benign_max_len)

                ba_results = {}
                for budget in (0.05, 0.01):
                    th_f, tpr_f = _benign_anchored(
                        cranet_scores, budget, test_scores_full_bench, y_te)
                    th_d, tpr_d = _benign_anchored(
                        cranet_da_scores, budget, test_scores_da_bench, y_te)
                    ba_results[f"fpr_budget_{budget:.2f}"] = {
                        "cranet_full": {"theta_benign": round(th_f, 4),
                                        "tpr_on_bench": round(tpr_f, 4)},
                        "cranet_da":   {"theta_benign": round(th_d, 4),
                                        "tpr_on_bench": round(tpr_d, 4)},
                    }

                results["benign_fpr"] = {
                    "n_benign": len(benign_recs),
                    "n_benign_before_filter": int(n_before),
                    "length_filter": "2..16 user turns",
                    "theta_cranet_full_val": theta,
                    "theta_cranet_da_val": theta_da_op,
                    "convex": {
                        "exceed_count": int(convex_exceed),
                        "benign_fpr": round(convex_exceed / len(convex_scores), 4),
                        "mean_score": round(float(np.mean(convex_scores)), 4),
                        "max_score": round(float(np.max(convex_scores)), 4),
                    },
                    "cranet_full": {
                        "exceed_count": int(cranet_exceed),
                        "benign_fpr": round(cranet_exceed / len(cranet_scores), 4),
                        "mean_score": round(float(np.mean(cranet_scores)), 4),
                        "max_score": round(float(np.max(cranet_scores)), 4),
                    },
                    "cranet_da": {
                        "exceed_count": int(cranet_da_exceed),
                        "benign_fpr": round(cranet_da_exceed / len(cranet_da_scores), 4),
                        "mean_score": round(float(np.mean(cranet_da_scores)), 4),
                        "max_score": round(float(np.max(cranet_da_scores)), 4),
                    },
                    "note": (
                        "FPR for each CRA-Net variant is the fraction of benign "
                        "ShareGPT sessions whose max-turn CRA-Net score exceeds "
                        "that variant's own CRA-Bench-calibrated threshold "
                        "(TPR=0.90 on the mixed validation split). Convex is "
                        "reported at the CRA-Net full threshold for comparison."
                    ),
                    "benign_anchored_calibration": ba_results,
                }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Ext] Wrote {args.out}")
    print(json.dumps(results, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
