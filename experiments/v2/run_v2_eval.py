#!/usr/bin/env python3
"""v2 evaluation protocol: grouped cross-validation, early detection,
time-to-detect, cross-corpus transfer, benign false alarms, and shift.

Experimental unit and uncertainty
  * Folds are StratifiedGroupKFold (k = 5) over ``group`` (red-teamer for
    HH-RT, template slot for CRA-Bench, attack chain for MT-ActorAttack,
    dialogue otherwise), repeated for N_SEEDS shuffles. Learned detectors
    never see a test group in training.
  * Point estimates are means over seeds of pooled out-of-fold AUROC.
  * 95% CIs: percentile cluster bootstrap over groups (B = N_BOOT) on the
    seed-0 out-of-fold scores; paired differences use the same resamples.
  * Operating point: inside each training fold, 20% of the groups are held
    out; theta is the smallest threshold with <= 5% false alarms on their
    negatives, so thresholds are never set on sessions a model was fitted on.

Output: <out>/<corpus>.json and transfer/shift files, where <out> is
experiments/results/v2, a subfolder named after a non-default --guard, or its goal/
subfolder (--goal).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DERIVED, RESULTS, index_jsonl  # noqa: E402
from detectors import registry  # noqa: E402

N_SEEDS = 3
N_BOOT = 2000
FPR_TARGET = 0.05
LABELED = ["hh_rt", "bad", "prosocial", "crabench"]
GOAL_LABELED = ["hh_rt", "bad", "prosocial"]          # corpora with goal evidence
KMAX = {"hh_rt": 6, "bad": 7, "prosocial": 3, "crabench": 8, "actorattack": 5,
        "hh_long": 15, "crabench_long": 12}
GUARD_TAGS = {"llamaguard": "", "granite": "_granite3guardian", "llamaguard8b": "_llamaguard3"}

# Paired contrasts reported for every corpus (stateful/variant - reference).
PAIRS = [("CRA-Acc[L]", "Turn-LR"), ("CRA-Acc", "Guard-Context"), ("CRA-Acc", "Guard-Turn"),
         ("Turn-LR", "Guard-Resp"), ("Turn-LR", "Guard-Turn"), ("CRA-Acc", "CRA-Acc[L]"),
         ("CRA-Acc[L+C]", "CRA-Acc[L]"), ("Guard-Context", "Guard-Turn"), ("CRA-Acc", "CRA-Net (prior)"),
         ("GRU-State", "Turn-LR"), ("CRA-Acc-GBM[L]", "Turn-GBM"), ("CRA-Acc (no t)", "Turn-LR"),
         ("CRA-Acc[C]", "CRA-Convex"), ("Turn-LR-NOR", "Turn-LR"), ("CRA-Acc-P[L]", "Turn-LR"),
         ("CRA-Acc-P", "CRA-Acc"), ("THRD", "Guard-Turn"), ("THRD", "Turn-LR"), ("CRA-Acc", "THRD")]
GOAL_PAIRS = [("Goal-LR", "Turn-LR"), ("Goal-GBM", "Turn-GBM"), ("CRA-Acc[L+G]", "CRA-Acc[L]"),
              ("Goal-Resp", "Guard-Resp"), ("Goal-Turn", "Guard-Turn"), ("Seq-Monitor", "Guard-Turn"),
              ("Goal-LR", "Seq-Monitor"), ("Goal-LR", "Guard-Context"), ("Goal-LR", "GRU-State"),
              ("Goal-LR", "CRA-Acc")]
# Learned detectors retrained in the shift and transfer analyses.
SHIFT_DETECTORS = {"Turn-LR", "CRA-Acc", "CRA-Acc[L]", "CRA-Acc[C]", "CRA-Net (prior)", "GRU-State",
                   "Goal-LR", "CRA-Acc[L+G]"}
TRANSFER_DETECTORS = SHIFT_DETECTORS - {"GRU-State"}
ROC_DETECTORS = ["Length", "CRA-Convex", "Guard-Turn", "Guard-Resp", "Guard-Context", "THRD", "Turn-LR",
                 "CRA-Net (prior)", "CRA-Acc"]


@dataclass
class Setup:
    """Which guard's scores to use, whether goal evidence is attached, and
    where results go."""
    guard: str = "llamaguard"
    goal: bool = False
    out: Path = RESULTS
    fixed: dict = field(init=False)
    learned: Callable[[int], dict] = field(init=False)

    def __post_init__(self):
        self.fixed, self.learned = registry(goal=self.goal)

    @property
    def guard_tag(self) -> str:
        return GUARD_TAGS[self.guard]

    @property
    def labeled(self) -> list[str]:
        return GOAL_LABELED if self.goal else LABELED

    @property
    def pairs(self) -> list[tuple[str, str]]:
        return PAIRS + (GOAL_PAIRS if self.goal else [])

    def write(self, name: str, obj: dict) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / name).write_text(json.dumps(obj, indent=1))


def load(corpus: str, setup: Setup) -> list[dict]:
    sess = index_jsonl(DERIVED / f"{corpus}.jsonl")
    guard = index_jsonl(DERIVED / f"guard{setup.guard_tag}_{corpus}.jsonl")
    cra = index_jsonl(DERIVED / f"cra_{corpus}.jsonl")
    goal = index_jsonl(DERIVED / f"goal_{corpus}.jsonl") if setup.goal else {}
    recs = []
    for sid, s in sess.items():
        if sid not in guard or sid not in cra or (setup.goal and sid not in goal):
            continue
        g, c = guard[sid], cra[sid]
        S = np.array(c["S"])
        recs.append({
            "id": sid, "label": s["label"], "group": s["group"], "onset": s["onset"],
            "T": len(s["turns"]), "meta": s["meta"],
            "g_user": g["g_user"], "g_resp": g["g_resp"], "g_ctx": g["g_ctx"],
            "S1": S[:, 0].tolist(), "S2": S[:, 1].tolist(), "S3": S[:, 2].tolist(),
        })
        if setup.goal:
            recs[-1].update({k: goal[sid][k] for k in ("g_goal", "g_goalresp", "g_seq")})
    return recs


# ─── metrics ─────────────────────────────────────────────────────────────────

def auroc(y, s):
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def cluster_boot_idx(groups: np.ndarray, rng, n_boot: int):
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[k] for k in pick])


def boot_ci(y, scores: dict, groups, pairs, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    names = list(scores)
    draws = {n: [] for n in names}
    diffs = {p: [] for p in pairs}
    for idx in cluster_boot_idx(groups, rng, n_boot):
        yy = y[idx]
        if len(set(yy)) < 2:
            continue
        vals = {n: roc_auc_score(yy, scores[n][idx]) for n in names}
        for n in names:
            draws[n].append(vals[n])
        for a, b in pairs:
            diffs[(a, b)].append(vals[a] - vals[b])
    ci = {n: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for n, v in draws.items()}
    dci = {f"{a} - {b}": {"mean": float(np.mean(v)), "ci95": [float(np.percentile(v, 2.5)),
                                                               float(np.percentile(v, 97.5))],
                          "p_le_0": float(np.mean(np.array(v) <= 0))}
           for (a, b), v in diffs.items()}
    return ci, dci


# ─── core loop ───────────────────────────────────────────────────────────────

def calibrate(neg_scores, alpha: float = FPR_TARGET) -> float:
    """Smallest threshold whose false-alarm rate on the given negatives is
    <= alpha under the alarm rule s >= theta (robust to tied scores)."""
    v = np.sort(np.asarray(neg_scores, dtype=float))
    for th in np.unique(v):
        if (v >= th).mean() <= alpha:
            return float(th)
    return float(np.nextafter(v[-1], np.inf))


def online_scores(det, rec):
    return np.asarray(det(rec) if callable(det) else det.turn_scores(rec), dtype=float)


def run_cv(recs, seed: int, setup: Setup, names_learned=None):
    """Returns {detector: list of per-session online score arrays} (OOF) and
    per-session fold thresholds."""
    y = np.array([r["label"] for r in recs])
    groups = np.array([r["group"] for r in recs])
    factories = setup.learned(seed)
    if names_learned is not None:
        factories = {k: v for k, v in factories.items() if k in names_learned}
    names = list(setup.fixed) + list(factories)
    turn = {n: [None] * len(recs) for n in names}
    theta = {n: np.zeros(len(recs)) for n in names}
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    for tr, te in skf.split(np.zeros(len(recs)), y, groups):
        # Hold out 20% of the training groups for threshold calibration, so
        # thresholds are never set on sessions the model was fitted on.
        tr_groups = np.unique(groups[tr])
        cal_groups = set(rng.permutation(tr_groups)[: max(1, int(0.2 * len(tr_groups)))])
        cal = np.array([i for i in tr if groups[i] in cal_groups])
        fit = np.array([i for i in tr if groups[i] not in cal_groups])
        train = [recs[i] for i in fit]
        for n in names:
            det = setup.fixed[n] if n in setup.fixed else factories[n]().fit(train)
            cal_scores = np.array([online_scores(det, recs[i]).max() for i in cal])
            th = calibrate(cal_scores[y[cal] == 0])
            for i in te:
                turn[n][i] = online_scores(det, recs[i])
                theta[n][i] = th
    return names, turn, theta


def summarize(recs, names, turn, theta):
    y = np.array([r["label"] for r in recs])
    out = {}
    for n in names:
        s = np.array([t.max() for t in turn[n]])
        alarm = s >= theta[n]
        out[n] = {"auroc": auroc(y, s), "auprc": float(average_precision_score(y, s)),
                  "tpr_at_op": float(alarm[y == 1].mean()), "fpr_at_op": float(alarm[y == 0].mean())}
    return out


def prefix_auroc(recs, names, turn, kmax):
    """AUROC of max_{t<=k} s_t over sessions with at least k exchanges."""
    out = {n: {} for n in names}
    for k in range(1, kmax + 1):
        idx = [i for i, r in enumerate(recs) if r["T"] >= k]
        y = np.array([recs[i]["label"] for i in idx])
        if len(set(y)) < 2:
            continue
        for n in names:
            out[n][k] = auroc(y, [turn[n][i][:k].max() for i in idx])
    return out


def time_to_detect(recs, names, turn, theta):
    """Detection delay (first alarm turn - annotated onset) for positives."""
    out = {}
    for n in names:
        delays, missed = [], 0
        for i, r in enumerate(recs):
            if r["label"] != 1 or r["onset"] is None:
                continue
            hits = np.where(turn[n][i] >= theta[n][i])[0]
            if len(hits) == 0:
                missed += 1
            else:
                delays.append(int(hits[0]) - int(r["onset"]))
        d = np.array(delays)
        out[n] = {"n_pos": len(delays) + missed, "missed": missed,
                  "delays": d.tolist(),
                  "median_delay": float(np.median(d)) if len(d) else None,
                  "frac_before_onset": float((d < 0).mean()) if len(d) else None,
                  "frac_at_onset": float((d == 0).mean()) if len(d) else None}
    return out


def evaluate_corpus(corpus: str, setup: Setup):
    t0 = time.time()
    recs = load(corpus, setup)
    y = np.array([r["label"] for r in recs])
    groups = np.array([r["group"] for r in recs])
    per_seed = []
    for seed in range(N_SEEDS):
        names, turn, theta = run_cv(recs, seed, setup)
        per_seed.append(summarize(recs, names, turn, theta))
        if seed == 0:
            turn0, theta0 = turn, theta
        print(f"[{corpus}] seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    session0 = {n: np.array([t.max() for t in turn0[n]]) for n in names}
    ci, dci = boot_ci(y, session0, groups, setup.pairs)
    metrics = {}
    for n in names:
        vals = {k: [ps[n][k] for ps in per_seed] for k in per_seed[0][n]}
        metrics[n] = {k: float(np.mean(v)) for k, v in vals.items()}
        metrics[n]["auroc_seed_sd"] = float(np.std(vals["auroc"]))
        metrics[n]["auroc_ci95"] = ci[n]
    res = {
        "corpus": corpus, "n": len(recs), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
        "n_groups": int(len(set(groups))), "n_seeds": N_SEEDS, "n_boot": N_BOOT,
        "fpr_target": FPR_TARGET, "metrics": metrics, "paired_differences": dci,
        "prefix_auroc": prefix_auroc(recs, names, turn0, KMAX[corpus]),
        "ttd": time_to_detect(recs, names, turn0, theta0),
        "roc_scores_seed0": {n: session0[n].round(5).tolist() for n in ROC_DETECTORS},
        "labels": y.tolist(),
        "wall_seconds": round(time.time() - t0, 1),
    }
    if corpus == "hh_rt":
        res["hh_extras"] = hh_extras(recs, session0)
    if corpus in ("crabench", "crabench_long"):
        res["lofo"] = crabench_lofo(recs, setup)
    setup.write(f"{corpus}.json", res)
    print(f"[{corpus}] wrote results ({time.time() - t0:.0f}s)")
    return res


# ─── corpus-specific analyses ────────────────────────────────────────────────

def hh_extras(recs, session0):
    from scipy.stats import spearmanr
    y = np.array([r["label"] for r in recs])
    pm = np.array([r["meta"]["pm_min_harmlessness"] for r in recs])
    T = np.array([r["T"] for r in recs])
    out = {"spearman_vs_pm_harmlessness": {n: float(spearmanr(-s, pm).statistic)
                                            for n, s in session0.items()},
           "length_stratified_auroc": {}}
    for lo, hi in [(2, 2), (3, 3), (4, 4), (5, 8)]:
        m = (T >= lo) & (T <= hi)
        out["length_stratified_auroc"][f"{lo}-{hi}"] = {
            "n": int(m.sum()), **{n: auroc(y[m], s[m]) for n, s in session0.items()}}
    return out


def max_score(det, rec) -> float:
    return online_scores(det, rec).max()


def auroc_of(det, recs) -> float:
    return auroc(np.array([r["label"] for r in recs]), [max_score(det, r) for r in recs])


def crabench_lofo(recs, setup: Setup):
    fams = sorted({r["meta"]["family"] for r in recs})
    out = {}
    for f in fams:
        tr = [r for r in recs if r["meta"]["family"] != f]
        te = [r for r in recs if r["meta"]["family"] == f]
        row = {n: auroc_of(det, te) for n, det in setup.fixed.items()}
        for n, fac in setup.learned(0).items():
            row[n] = auroc_of(fac().fit(tr), te)
        out[f] = row
        print(f"[crabench LOFO] {f}: " + ", ".join(f"{k}={v:.3f}" for k, v in row.items()), flush=True)
    return out


def shift_row(tr: list[dict], te: list[dict], setup: Setup, seed: int) -> dict:
    """AUROC on `te` of every fixed detector and of the shift detectors fitted on `tr`."""
    y = np.array([r["label"] for r in te])
    row = {"n_train": len(tr), "n_test": len(te), "test_pos_rate": float(y.mean())}
    for n, det in setup.fixed.items():
        row[n] = auroc_of(det, te)
    for n, fac in setup.learned(seed).items():
        if n in SHIFT_DETECTORS:
            row[n] = auroc_of(fac().fit(tr), te)
    return row


def summarize_reps(out: dict) -> dict:
    names = [k for k in next(iter(out.values())) if k not in ("n_train", "n_test", "test_pos_rate")]
    return {n: {"mean": float(np.mean([v[n] for v in out.values()])),
                "sd": float(np.std([v[n] for v in out.values()]))} for n in names}


def transfer_and_benign(setup: Setup):
    """Train on 80% of corpus A's groups, set theta at 5% FPR on the negatives
    of the held-out 20%; test on every other labeled corpus (AUROC) and on
    WildChat (false-alarm rate)."""
    data = {c: load(c, setup) for c in setup.labeled + ["wildchat"]}
    wild = data["wildchat"]
    out = {"transfer_auroc": {}, "wildchat_false_alarm": {}, "wildchat_n": len(wild)}
    for a in ["hh_rt", "bad", "prosocial"]:
        rng = np.random.default_rng(0)
        g_a = sorted({r["group"] for r in data[a]})
        cal_g = set(rng.permutation(g_a)[: int(0.2 * len(g_a))])
        fit_a = [r for r in data[a] if r["group"] not in cal_g]
        cal_a = [r for r in data[a] if r["group"] in cal_g]
        y_a = np.array([r["label"] for r in cal_a])
        dets = dict(setup.fixed)
        for n, fac in setup.learned(0).items():
            if n in TRANSFER_DETECTORS:
                dets[n] = fac().fit(fit_a)
        out["transfer_auroc"][a] = {}
        out["wildchat_false_alarm"][a] = {}
        for n, det in dets.items():
            s_a = np.array([max_score(det, r) for r in cal_a])
            th = calibrate(s_a[y_a == 0])
            for b in setup.labeled:
                if b != a:
                    out["transfer_auroc"][a].setdefault(n, {})[b] = auroc_of(det, data[b])
            s_w = np.array([max_score(det, r) for r in wild])
            out["wildchat_false_alarm"][a][n] = {
                "theta": float(th), "rate": float((s_w >= th).mean()),
                "by_length": {L: float((s_w[[r["T"] == L for r in wild]] >= th).mean())
                              for L in range(2, 9) if any(r["T"] == L for r in wild)}}
        print(f"[transfer] trained on {a}", flush=True)
    setup.write("transfer_benign.json", out)


def hh_model_shift(setup: Setup, n_rep: int = 5):
    """Red-teamer-disjoint AND target-model-disjoint split on HH-RT: train on
    {context distillation, plain lm} conversations of half the red-teamers;
    test on {rlhf, rejection sampling} conversations of the other half."""
    recs = load("hh_rt", setup)
    groups = sorted({r["group"] for r in recs})
    src = {"context distillation", "plain lm"}
    out = {}
    for rep in range(n_rep):
        rng = np.random.default_rng(rep)
        half = set(rng.permutation(groups)[: len(groups) // 2])
        tr = [r for r in recs if r["group"] in half and r["meta"]["model_type"] in src]
        te = [r for r in recs if r["group"] not in half and r["meta"]["model_type"] not in src]
        out[rep] = shift_row(tr, te, setup, rep)
    summary = summarize_reps(out)
    setup.write("hh_model_shift.json", {"reps": out, "summary": summary})
    print("[shift] " + ", ".join(f"{k}={v['mean']:.3f}" for k, v in summary.items()))


def actorattack_target_shift(setup: Setup, n_rep: int = 5):
    """MT-ActorAttack: train on one target model's transcripts and test on the
    other's, with attack chains disjoint between train and test."""
    recs = load("actorattack", setup)
    targets = sorted({r["meta"]["target"] for r in recs})
    chains = sorted({r["group"] for r in recs})
    out = {}
    for src in targets:
        for rep in range(n_rep):
            rng = np.random.default_rng(rep)
            half = set(rng.permutation(chains)[: len(chains) // 2])
            tr = [r for r in recs if r["group"] in half and r["meta"]["target"] == src]
            te = [r for r in recs if r["group"] not in half and r["meta"]["target"] != src]
            if len({r["label"] for r in te}) < 2 or len({r["label"] for r in tr}) < 2:
                continue
            out[f"{src}|{rep}"] = shift_row(tr, te, setup, rep)
    summary = summarize_reps(out)
    setup.write("actorattack_target_shift.json", {"reps": out, "summary": summary})
    print("[aa-shift] " + ", ".join(f"{k}={v['mean']:.3f}" for k, v in summary.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="*", default=None, help="default: every labeled corpus of the setup")
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--shift", action="store_true")
    ap.add_argument("--aa-shift", action="store_true")
    ap.add_argument("--guard", default="llamaguard", choices=list(GUARD_TAGS))
    ap.add_argument("--goal", action="store_true", help="attach goal-inference evidence and its detectors")
    ap.add_argument("--out-dir", type=Path, default=None, help="override the results folder")
    args = ap.parse_args()
    out = RESULTS / ("" if args.guard == "llamaguard" else args.guard) / ("goal" if args.goal else "")
    setup = Setup(guard=args.guard, goal=args.goal, out=args.out_dir or out)
    for c in setup.labeled if args.corpora is None else args.corpora:
        evaluate_corpus(c, setup)
    if args.shift:
        hh_model_shift(setup)
    if args.aa_shift:
        actorattack_target_shift(setup)
    if args.transfer:
        transfer_and_benign(setup)


if __name__ == "__main__":
    main()
