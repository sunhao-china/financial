# -*- coding: utf-8 -*-
"""P2 embedding 对比试验（大纲 §13.1）：多模型 top-k recall 评估。

- 评估集：随机 50 篇文档，取每篇第 2 个 chunk 作为 query（同文档其余 chunk 为正例，
  金融语料同文档 chunk 语义连续，是无人工标注的可靠相关性信号）
- 度量：recall@5 / recall@10（query 自身排除）
- 模型对比：bge-m3(512) 本地向量 / qwen3-emb-0.6b / 百炼 text-embedding-v3（后两者向量由
  --gen 子命令生成后缓存到 data/evals/）
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

N_QUERIES = 50
KS = (5, 10)


def pick_queries(chunks_path, seed=42):
    """选 50 篇文档，取每篇 chunk_index==1 的 chunk 作为 query。"""
    rng = np.random.default_rng(seed)
    t = pq.read_table(chunks_path, columns=["id", "doc_id", "chunk_index", "text"])
    d = t.to_pandas()
    q_rows = d[d.chunk_index == 1]
    pick = q_rows.sample(N_QUERIES, random_state=seed)
    queries = []
    for _, r in pick.iterrows():
        pos = set(d[(d.doc_id == r.doc_id) & (d.chunk_index != 1)].id)
        queries.append({"id": r.id, "text": r.text, "positives": list(pos)})
    return queries


def load_vectors(emb_path):
    t = pq.read_table(emb_path)
    ids = t.column("id").to_pylist()
    emb = np.asarray(t.column("embedding").to_pylist(), dtype=np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return ids, emb


def recall_at_k(queries, ids, emb, k):
    """queries 的文本在本地编码不可用时，直接传入 query 向量（q_emb dict）。"""
    id2i = {x: i for i, x in enumerate(ids)}
    q_emb = {q["id"]: q["vec"] for q in queries}
    hits = []
    for q in queries:
        v = q_emb[q["id"]]
        sims = emb @ v
        top = np.argsort(-sims)
        got = []
        for i in top:
            if ids[i] == q["id"]:
                continue
            got.append(ids[i])
            if len(got) == k:
                break
        hits.append(len(set(got) & set(q["positives"])) / max(len(q["positives"]), 1))
    return float(np.mean(hits))


def evaluate(name, queries, emb_path, k_values=KS):
    ids, emb = load_vectors(emb_path)
    out = {}
    for k in k_values:
        out[f"recall@{k}"] = round(recall_at_k(queries, ids, emb, k), 4)
    print(f"{name}: {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=str(config.DATA_DIR / "chunks_trial.parquet"))
    ap.add_argument("--eval-dir", default=str(config.DATA_DIR / "evals"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    evaldir = Path(args.eval_dir)
    evaldir.mkdir(parents=True, exist_ok=True)
    qpath = evaldir / "queries.json"
    if qpath.exists():
        queries = json.loads(qpath.read_text(encoding="utf-8"))
        print(f"复用 query 集 {qpath}（{len(queries)} 个）")
    else:
        queries = pick_queries(args.chunks, args.seed)
        qpath.write_text(json.dumps(queries, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"生成 query 集 → {qpath}")

    # 各模型向量与 query 向量：{model: (emb_path, q_emb_path)}，q_emb 文件为 json {id: vec}
    results = {}
    for name, emb_path, q_emb_path in [
        ("bge-m3-512", config.DATA_DIR / "emb_trial.parquet", evaldir / "qvec_bge.json"),
        ("qwen3-emb-0.6b", evaldir / "emb_qwen3.parquet", evaldir / "qvec_qwen3.json"),
        ("qianwen-v3", evaldir / "emb_qianwen.parquet", evaldir / "qvec_qianwen.json"),
    ]:
        if not Path(emb_path).exists():
            print(f"跳过 {name}（缺 {emb_path}，需先运行 --gen 生成）")
            continue
        if not Path(q_emb_path).exists():
            # 从库向量中提取 query 向量
            ids, emb = load_vectors(emb_path)
            id2i = {x: i for i, x in enumerate(ids)}
            qvec = {q["id"]: emb[id2i[q["id"]]].tolist() for q in queries}
            Path(q_emb_path).write_text(json.dumps(qvec), encoding="utf-8")
            print(f"提取 query 向量 → {q_emb_path}")
        qvec = json.loads(Path(q_emb_path).read_text(encoding="utf-8"))
        for q in queries:
            q["vec"] = np.array(qvec[q["id"]], dtype=np.float32)
        results[name] = evaluate(name, queries, emb_path)
        for q in queries:
            del q["vec"]

    print("\n=== 对比汇总 ===")
    for name, r in results.items():
        print(f"{name}: " + "  ".join(f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
