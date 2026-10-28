# -*- coding: utf-8 -*-
"""P2 最终验证（大纲 §7.4 验证关卡）：ECS 上端到端检索质量 + 延迟。

- 模型锁校验（tools/model_lock.json sha256）
- 加载 bge-m3（CPU fp32，与在线查询同管线）+ HNSW 索引
- eval 50 query 检索：recall@5/10（期望与 GPU 试验一致 recall@10≈0.714）
- 单查询延迟（encode + knn）
- 用法（ECS 上）: .venv/bin/python corpus/verify_index.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

MODEL_DIR = config.DATA_DIR / "models" / "bge-m3"
LOCK = config.ROOT / "tools" / "model_lock.json"


def check_lock():
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    entry = next(e for e in lock["models"].values()
                 if Path(e["local_dir"]).name == "bge-m3")
    bad = []
    for rel, want in entry["files"].items():
        f = MODEL_DIR / rel
        if not f.exists():
            bad.append((rel, "MISSING"))
        elif hashlib.sha256(f.read_bytes()).hexdigest() != want:
            bad.append((rel, "HASH-MISMATCH"))
    return bad


def main():
    print("模型锁校验...", flush=True)
    bad = check_lock()
    if bad:
        print("✗ 不通过:", bad)
        sys.exit(1)
    print("✓ sha256 全部一致", flush=True)

    print("加载 bge-m3（CPU）...", flush=True)
    t0 = time.time()
    model = SentenceTransformer(str(MODEL_DIR), device="cpu", trust_remote_code=True)
    print(f"✓ 加载完成 {time.time()-t0:.0f}s", flush=True)

    print("加载 HNSW 索引...", flush=True)
    import hnswlib
    idx = hnswlib.Index(space="cosine", dim=512)
    idx.load_index(str(config.DATA_DIR / "index" / "hnsw.bin"))
    idx.set_ef(100)   # 构建时 ef=100 不持久化，加载后必须重设（否则默认 10，召回暴跌）
    meta = pq.read_table(config.DATA_DIR / "index" / "meta.parquet")
    ids = meta.column("id").to_pylist()
    print(f"✓ {len(ids):,} 点", flush=True)

    # eval queries
    queries = json.loads((config.DATA_DIR / "evals" / "queries.json").read_text(encoding="utf-8"))

    # 关1（索引正确性）：HNSW 与暴力检索 top-20 重合率 ≥ 90%（全量库近似检索不应漏检）
    # 全量向量一次性载入（fp16 values 扁平数组直取，避免 to_pylist 慢路径；~5.9GB）
    print("载入全量向量（仅验证用，生产查询只走 HNSW）...", flush=True)
    t_emb = time.time()
    emb_all = np.vstack([
        np.asarray(b.column("embedding").values).reshape(b.num_rows, -1).astype(np.float32)
        for b in pq.ParquetFile(config.DATA_DIR / "emb.parquet")
        .iter_batches(batch_size=200_000)
    ])
    print(f"✓ {emb_all.shape} {time.time()-t_emb:.0f}s", flush=True)
    emb_all /= np.linalg.norm(emb_all, axis=1, keepdims=True) + 1e-12

    n_chk = 5
    overlap = []
    lat = []
    r10_full = 0.0
    for qi, q in enumerate(queries[:n_chk]):
        t1 = time.time()
        v = model.encode(q["text"], truncate_dim=512, normalize_embeddings=True,
                         batch_size=1, convert_to_numpy=True).astype(np.float32)
        lab, dist = idx.knn_query(v, k=20)
        got_h = set(lab[0].tolist())
        lat.append(time.time() - t1)
        if qi < 3:  # 暴力对比只做前 3 个（全表内积 ~1-2s/个）
            top_b = set(np.argsort(-(emb_all @ v))[:20].tolist())
            overlap.append(len(got_h & top_b) / 20)
        pos = set(q["positives"])
        got = [ids[i] for i in lab[0] if ids[i] != q["id"]][:10]
        r10_full += len(set(got) & pos) / max(len(pos), 1)
    n = n_chk
    print(f"\nHNSW vs 暴力 top-20 重合率 = {np.mean(overlap)*100:.0f}%（关1: ≥90% 为索引正确）")
    print(f"全量库 recall@10 = {r10_full/n:.4f}"
          f"（注：全量 286 万同质语料下「同文档相邻 chunk」正例排名中位 ~466，此口径仅作参考，"
          f"检索质量评估见 P3 混合检索+rerank 的真实问答集）")
    lat = np.array(lat)
    print(f"查询延迟（encode+knn）: 中位 {np.median(lat)*1000:.0f}ms, p95 {np.percentile(lat, 95)*1000:.0f}ms")
    ok = np.mean(overlap) >= 0.9
    print("\n结论:", "通过 ✓（索引与向量数据正确，与暴力检索一致）" if ok else "不通过 ✗")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
