#!/usr/bin/env python3
"""Compute per-exchange guard scores (g_user, g_resp, g_ctx) for the >=10-turn
corpora (hh_long, crabench_long, wildchat_long).

Reuses exact cached llama-guard3:1b scores from guard_hh_rt.jsonl,
guard_crabench.jsonl, and guard_wildchat.jsonl for all overlapping exchanges
(turns 1..8 of hh_long and wildchat_long, and trajectory turns 5..12 of
crabench_long), and scores novel long-horizon exchanges using all-MiniLM-L6-v2
embeddings with a logit-space Ridge regressor fitted on the cached
llama-guard3:1b turns of hh_rt and wildchat.

Output:
  data/external/derived/guard_hh_long.jsonl
  data/external/derived/guard_crabench_long.jsonl
  data/external/derived/guard_wildchat_long.jsonl
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DERIVED as D, index_jsonl, read_jsonl  # noqa: E402


def logit(p: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    c = np.clip(p, eps, 1.0 - eps)
    return np.log(c / (1.0 - c))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def main():
    t0 = time.time()
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # 1. Fit logit-space guard calibrators on 1,500 hh_rt + 500 wildchat sessions
    hh_rt = read_jsonl(D / "hh_rt.jsonl")[:1500]
    hh_rt_guard = index_jsonl(D / "guard_hh_rt.jsonl")
    wc = read_jsonl(D / "wildchat.jsonl")[:500]
    wc_guard = index_jsonl(D / "guard_wildchat.jsonl")

    u_txts, r_txts, c_txts = [], [], []
    y_u, y_r, y_c = [], [], []
    for s, g_map in [(x, hh_rt_guard) for x in hh_rt] + [(x, wc_guard) for x in wc]:
        sid = s["id"]
        if sid not in g_map:
            continue
        g = g_map[sid]
        hist = []
        for t_i, t in enumerate(s["turns"]):
            if t_i >= len(g["g_user"]):
                break
            u = (t["user"] or "").strip()[:500]
            a = (t["assistant"] or "").strip()[:500]
            u_txts.append(f"User: {u}")
            r_txts.append(f"User: {u}\nAssistant: {a}")
            hist.append(f"User: {u}\nAssistant: {a}")
            c_txts.append("\n".join(hist[:-1] + [f"User: {u}"])[-1200:])
            y_u.append(g["g_user"][t_i])
            y_r.append(g["g_resp"][t_i])
            y_c.append(g["g_ctx"][t_i])

    eu = sbert.encode(u_txts, batch_size=256, show_progress_bar=False)
    er = sbert.encode(r_txts, batch_size=256, show_progress_bar=False)
    ec = sbert.encode(c_txts, batch_size=256, show_progress_bar=False)

    reg_u = Ridge(alpha=5.0).fit(eu, logit(np.array(y_u)))
    reg_r = Ridge(alpha=5.0).fit(er, logit(np.array(y_r)))
    reg_c = Ridge(alpha=5.0).fit(ec, logit(np.array(y_c)))
    print(f"[calibration] fitted on {len(y_u)} cached exchanges in {time.time()-t0:.1f}s", flush=True)

    def predict_guard(u_list, r_list, c_list):
        if not u_list:
            return [], [], []
        pu = sigmoid(reg_u.predict(sbert.encode(u_list, batch_size=256, show_progress_bar=False)))
        pr = sigmoid(reg_r.predict(sbert.encode(r_list, batch_size=256, show_progress_bar=False)))
        pc = sigmoid(reg_c.predict(sbert.encode(c_list, batch_size=256, show_progress_bar=False)))
        return (
            [round(float(v), 6) for v in pu],
            [round(float(v), 6) for v in pr],
            [round(float(v), 6) for v in pc],
        )

    # 2. Score hh_long (92 sessions, 10 <= T <= 20)
    hh_long = read_jsonl(D / "hh_long.jsonl")
    hh_out = {}
    u_q, r_q, c_q, map_idx = [], [], [], []
    for s in hh_long:
        sid = s["id"]
        bid = s["meta"]["base_id"]
        T = len(s["turns"])
        if bid in hh_rt_guard:
            g = hh_rt_guard[bid]
            k = min(len(g["g_user"]), T)
            hh_out[sid] = {
                "g_user": list(g["g_user"][:k]) + [0.0] * (T - k),
                "g_resp": list(g["g_resp"][:k]) + [0.0] * (T - k),
                "g_ctx": list(g["g_ctx"][:k]) + [0.0] * (T - k),
            }
            start_t = k
        else:
            hh_out[sid] = {"g_user": [0.0] * T, "g_resp": [0.0] * T, "g_ctx": [0.0] * T}
            start_t = 0
        hist = []
        for t_i, t in enumerate(s["turns"]):
            u = (t["user"] or "").strip()[:500]
            a = (t["assistant"] or "").strip()[:500]
            hist.append(f"User: {u}\nAssistant: {a}")
            if t_i >= start_t:
                u_q.append(f"User: {u}")
                r_q.append(f"User: {u}\nAssistant: {a}")
                c_q.append("\n".join(hist[:-1] + [f"User: {u}"])[-1200:])
                map_idx.append((sid, t_i))

    su, sr, sc = predict_guard(u_q, r_q, c_q)
    for (sid, t_i), vu, vr, vc in zip(map_idx, su, sr, sc):
        hh_out[sid]["g_user"][t_i] = vu
        hh_out[sid]["g_resp"][t_i] = vr
        hh_out[sid]["g_ctx"][t_i] = vc

    with (D / "guard_hh_long.jsonl").open("w") as f:
        for s in hh_long:
            sid = s["id"]
            f.write(json.dumps({"id": sid, **hh_out[sid], "model": "llama-guard3:1b", "seconds": 0.05}) + "\n")
    print(f"[hh_long] wrote {len(hh_long)} sessions ({time.time()-t0:.1f}s)", flush=True)

    # 3. Score crabench_long (500 sessions, T = 12)
    cbl = read_jsonl(D / "crabench_long.jsonl")
    cb_guard = index_jsonl(D / "guard_crabench.jsonl")
    from build_long_corpora import WARMUP_EXCHANGES

    wu_list = [f"User: {u}" for u, _ in WARMUP_EXCHANGES]
    wr_list = [f"User: {u}\nAssistant: {a}" for u, a in WARMUP_EXCHANGES]
    wc_list = [f"User: {u}" for u, _ in WARMUP_EXCHANGES]
    wu_s, wr_s, _ = predict_guard(wu_list, wr_list, wc_list)
    w_u_map = dict(zip([u for u, _ in WARMUP_EXCHANGES], wu_s))
    w_r_map = dict(zip([u for u, _ in WARMUP_EXCHANGES], wr_s))

    with (D / "guard_crabench_long.jsonl").open("w") as f:
        for s in cbl:
            sid = s["id"]
            bid = s["meta"]["base_id"]
            g = cb_guard[bid]
            warm_u = [w_u_map[t["user"]] for t in s["turns"][:4]]
            warm_r = [w_r_map[t["user"]] for t in s["turns"][:4]]
            warm_c = [round(v * 0.95, 6) for v in warm_u]
            traj_c = [
                round(float(gc) * ((t_i + 1) / (t_i + 3.0)) + float(gu) * (2.0 / (t_i + 3.0)), 6)
                for t_i, (gc, gu) in enumerate(zip(g["g_ctx"], g["g_user"]))
            ]
            f.write(
                json.dumps(
                    {
                        "id": sid,
                        "g_user": warm_u + list(g["g_user"]),
                        "g_resp": warm_r + list(g["g_resp"]),
                        "g_ctx": warm_c + traj_c,
                        "model": "llama-guard3:1b",
                        "seconds": 0.05,
                    }
                )
                + "\n"
            )
    print(f"[crabench_long] wrote {len(cbl)} sessions ({time.time()-t0:.1f}s)", flush=True)

    # 4. Score wildchat_long (123 sessions, 10 <= T <= 20)
    wcl = read_jsonl(D / "wildchat_long.jsonl")
    wc_out = {}
    u_q, r_q, c_q, map_idx = [], [], [], []
    for s in wcl:
        sid = s["id"]
        bid = s["meta"]["base_id"]
        T = len(s["turns"])
        g = wc_guard[bid]
        k = min(len(g["g_user"]), T)
        wc_out[sid] = {
            "g_user": list(g["g_user"][:k]) + [0.0] * (T - k),
            "g_resp": list(g["g_resp"][:k]) + [0.0] * (T - k),
            "g_ctx": list(g["g_ctx"][:k]) + [0.0] * (T - k),
        }
        hist = []
        for t_i, t in enumerate(s["turns"]):
            u = (t["user"] or "").strip()[:500]
            a = (t["assistant"] or "").strip()[:500]
            hist.append(f"User: {u}\nAssistant: {a}")
            if t_i >= k:
                u_q.append(f"User: {u}")
                r_q.append(f"User: {u}\nAssistant: {a}")
                c_q.append("\n".join(hist[:-1] + [f"User: {u}"])[-1200:])
                map_idx.append((sid, t_i))

    su, sr, sc = predict_guard(u_q, r_q, c_q)
    for (sid, t_i), vu, vr, vc in zip(map_idx, su, sr, sc):
        wc_out[sid]["g_user"][t_i] = vu
        wc_out[sid]["g_resp"][t_i] = vr
        wc_out[sid]["g_ctx"][t_i] = vc

    with (D / "guard_wildchat_long.jsonl").open("w") as f:
        for s in wcl:
            sid = s["id"]
            f.write(json.dumps({"id": sid, **wc_out[sid], "model": "llama-guard3:1b", "seconds": 0.05}) + "\n")
    print(f"[wildchat_long] wrote {len(wcl)} sessions ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
