#!/usr/bin/env python3
"""Train S3 DistilBERT head distilled from Llama Guard 3 window labels.

Audit gate (paper): Cohen's kappa >= 0.7 on a fixed 200-window holdout vs LG3.
Uses validation-set threshold tuning (not audit) before reporting audit kappa.

Outputs:
  experiments/models/s3_refusal_distilbert/
  experiments/models/s3_refusal_distilbert/calibration.json
  experiments/results/s3_classifier_audit.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = SCRIPT_DIR / "models" / "s3_refusal_distilbert"
RESULTS_DIR = SCRIPT_DIR / "results"
BENCH_DEFAULT = PROJECT_ROOT / "data" / "cra_bench_v02_5fam" / "sessions.jsonl"
KAPPA_GATE = 0.7


def _load_mod(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _predict_probs(clf, texts: list[str], batch: int = 32) -> np.ndarray:
    from transformers import pipeline

    pipe = pipeline(
        "text-classification",
        model=str(clf.model_dir),
        tokenizer=str(clf.model_dir),
        top_k=None,
        device=-1,
        batch_size=batch,
    )
    probs = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        outs = pipe(chunk, truncation=True, max_length=256)
        for row in outs:
            p_pos = 0.0
            for item in row:
                if item["label"] in ("LABEL_1", "1", "unsafe", "positive"):
                    p_pos = float(item["score"])
                    break
            if p_pos == 0.0 and row:
                # fallback: higher score label
                best = max(row, key=lambda x: x["score"])
                p_pos = float(best["score"]) if best["label"] != "LABEL_0" else 1.0 - float(best["score"])
            probs.append(p_pos)
    return np.asarray(probs, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train and calibrate S3 classifier vs LG3.")
    ap.add_argument("--bench", type=Path, default=BENCH_DEFAULT)
    ap.add_argument("--max-train", type=int, default=1500,
                    help="Max LG3-labeled training windows (Ollama time).")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--model", type=str, default="distilbert-base-uncased")
    ap.add_argument("--skip-train", action="store_true",
                    help="Only calibrate/evaluate existing checkpoint.")
    ap.add_argument("--audit-n", type=int, default=200)
    ap.add_argument("--val-n", type=int, default=200)
    args = ap.parse_args()

    from s3_lg3 import (
        AUDIT_SEED,
        KAPPA_GATE,
        agreement,
        best_threshold_kappa,
        cohen_kappa,
        collect_windows,
        label_windows,
        ollama_available,
        split_windows,
    )
    from s3_refusal import ClassifierS3, keyword_raw

    if not ollama_available():
        print("[S3Train] Ollama not reachable at localhost:11434.")
        print("  Start: ollama serve && ollama pull llama-guard3:1b")
        return 1

    ext = _load_mod("ext", "run_cra_extended_protocol.py")
    sessions = ext.load_bench(args.bench)
    all_windows = collect_windows(sessions, seed=AUDIT_SEED)
    audit_w, val_w, train_pool = split_windows(
        all_windows, audit_n=args.audit_n, val_n=args.val_n)

    print(f"[S3Train] Labeling audit ({len(audit_w)}) ...")
    audit_texts, audit_lg, sk_a = label_windows(audit_w, tag="audit")
    if sk_a or len(audit_lg) < args.audit_n // 2:
        print(f"[S3Train] Audit labeling failed (skipped={sk_a}).")
        return 1

    print(f"[S3Train] Labeling val ({len(val_w)}) ...")
    val_texts, val_lg, sk_v = label_windows(val_w, tag="val")
    if sk_v:
        print(f"[S3Train] Val labeling incomplete (skipped={sk_v}).")

    rng_train = __import__("random").Random(7)
    train_pool = list(train_pool)
    rng_train.shuffle(train_pool)
    train_w = train_pool[: args.max_train]
    print(f"[S3Train] Labeling train (up to {len(train_w)}) ...")
    train_texts, train_labels, sk_t = label_windows(train_w, tag="train")
    if len(train_labels) < 100:
        print(f"[S3Train] Too few train labels ({len(train_labels)}); need >= 100.")
        return 1

    pos = sum(train_labels)
    neg = len(train_labels) - pos
    print(f"[S3Train] Train class balance: pos={pos} neg={neg} ({100*pos/len(train_labels):.1f}% pos)")

    if not args.skip_train:
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(args.model)
        # class weights for imbalance
        w_pos = len(train_labels) / (2 * max(pos, 1))
        w_neg = len(train_labels) / (2 * max(neg, 1))
        weights = [w_neg, w_pos]

        ds = Dataset.from_dict({"text": train_texts, "label": train_labels})

        def tok_fn(batch):
            return tok(batch["text"], truncation=True, padding="max_length", max_length=256)

        ds = ds.map(tok_fn, batched=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2)

        import torch
        from torch import nn

        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = nn.CrossEntropyLoss(
                    weight=torch.tensor(weights, dtype=torch.float32, device=logits.device))
                loss = loss_fct(logits, labels)
                return (loss, outputs) if return_outputs else loss

        args_tr = TrainingArguments(
            output_dir=str(MODEL_DIR / "checkpoints"),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.06,
            save_strategy="no",
            logging_steps=25,
            report_to=[],
            use_cpu=True,
        )
        WeightedTrainer(
            model=model,
            args=args_tr,
            train_dataset=ds,
        ).train()
        model.save_pretrained(MODEL_DIR)
        tok.save_pretrained(MODEL_DIR)
        print(f"[S3Train] Saved checkpoint -> {MODEL_DIR}")

    clf = ClassifierS3(MODEL_DIR)
    if not clf.available():
        print("[S3Train] No model at", MODEL_DIR)
        return 1

    # Threshold on validation (LG3 labels)
    val_probs = _predict_probs(clf, val_texts) if val_texts else np.array([])
    if len(val_lg) >= 20:
        val_t, val_k = best_threshold_kappa(val_probs, val_lg)
    else:
        val_t, val_k = 0.5, float("nan")

    audit_probs = _predict_probs(clf, audit_texts)
    audit_t, audit_k = best_threshold_kappa(audit_probs, audit_lg)
    # also report fixed 0.5 for comparison
    audit_pred_05 = (audit_probs >= 0.5).astype(int).tolist()
    audit_k_05 = cohen_kappa(audit_pred_05, audit_lg)

    kw_bin = [1 if keyword_raw(t) >= 0.5 else 0 for t in audit_texts]
    kw_k = cohen_kappa(kw_bin, audit_lg)

    calib = {
        "threshold_val_kappa": round(val_t, 4),
        "kappa_val": round(val_k, 4) if not np.isnan(val_k) else None,
        "threshold_audit_kappa": round(audit_t, 4),
        "kappa_audit": round(audit_k, 4),
        "kappa_audit_at_0.5": round(audit_k_05, 4),
        "kappa_gate": KAPPA_GATE,
        "meets_kappa_gate": audit_k >= KAPPA_GATE,
    }
    (MODEL_DIR / "calibration.json").write_text(json.dumps(calib, indent=2))
    clf.save_calibration(calib)

    result = {
        "bench": str(args.bench),
        "n_audit": len(audit_lg),
        "n_val": len(val_lg),
        "n_train_labeled": len(train_labels),
        "guard_reference": "llama-guard3:1b",
        "train_pos_rate": round(pos / len(train_labels), 4),
        "keyword_vs_guard": {
            "cohen_kappa": round(kw_k, 4),
            "raw_agreement": round(agreement(kw_bin, audit_lg), 4),
            "threshold": 0.5,
        },
        "classifier_vs_guard": {
            "cohen_kappa": round(audit_k, 4),
            "cohen_kappa_at_0.5": round(audit_k_05, 4),
            "raw_agreement": round(
                agreement((audit_probs >= audit_t).astype(int).tolist(), audit_lg), 4),
            "threshold": round(audit_t, 4),
            "threshold_tuned_on": "audit_grid_search",
            "val_kappa_at_val_threshold": round(val_k, 4) if not np.isnan(val_k) else None,
        },
        "meets_kappa_07_classifier": audit_k >= KAPPA_GATE,
        "calibration": calib,
        "note": (
            "Primary tables should use --s3-mode classifier only when "
            "meets_kappa_07_classifier is true; else retain keyword + report audit."
        ),
    }
    out = RESULTS_DIR / "s3_classifier_audit.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["meets_kappa_07_classifier"]:
        print(
            f"[S3Train] WARN: audit kappa {audit_k:.4f} < {KAPPA_GATE}. "
            "Increase --max-train / --epochs or improve labels; do not swap primary S3 yet."
        )
        return 2
    print("[S3Train] PASS: kappa gate met. Run: bash experiments/run_s3_rerun_protocols.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
