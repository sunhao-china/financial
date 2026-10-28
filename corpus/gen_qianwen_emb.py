# -*- coding: utf-8 -*-
"""百炼 text-embedding-v3 对比向量生成（OpenAI 兼容端点，dimensions=512）。

- 输入：data/chunks_trial.parquet + data/evals/queries.json
- 输出：data/evals/emb_qianwen.parquet + data/evals/qvec_qianwen.json
- 断点续传：每 10 批写 checkpoint
- 费用：19k chunks × ~500 字 ≈ 950 万 tokens ≈ ¥5（¥0.0005/1k tokens）
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

BASE = f"{config.DASHSCOPE_BASE}/compatible-mode/v1"
API = f"{BASE}/embeddings"
DIM = 512
REQ_BATCH = 10          # text-embedding-v3 单请求最多 10 条


def embed_batch(sess, texts, retries=5):
    payload = {"model": "text-embedding-v3", "input": texts,
               "dimensions": DIM, "encoding_format": "float"}
    for attempt in range(retries):
        try:
            r = sess.post(API, json=payload,
                          headers={"Authorization": f"Bearer {config.DASHSCOPE_API_KEY}"},
                          timeout=60)
            if r.status_code == 200:
                data = r.json()["data"]
                data.sort(key=lambda x: x["index"])
                return np.array([d["embedding"] for d in data], dtype=np.float32)
            print(f"  HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"  异常: {e}")
        time.sleep(3 * (attempt + 1))
    raise RuntimeError("重试耗尽")


def main():
    out_dir = config.DATA_DIR / "evals"
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / "emb_qianwen.ckpt"
    skip_rows = json.loads(ckpt.read_text())["done"] if ckpt.exists() else 0

    sess = requests.Session()
    pf = pq.ParquetFile(config.DATA_DIR / "chunks_trial.parquet")
    n_total = pf.metadata.num_rows
    texts = []
    ids_all = []
    writer = None
    done = 0
    t0 = time.time()

    for batch in pf.iter_batches(batch_size=REQ_BATCH):
        t = pa.Table.from_batches([batch])
        if skip_rows:
            s = min(skip_rows, t.num_rows)
            t = t.slice(s)
            skip_rows -= s
        if t.num_rows == 0:
            continue
        texts.extend(t.column("text").to_pylist())
        ids_all.extend(t.column("id").to_pylist())
        if len(texts) < REQ_BATCH:
            continue
        emb = embed_batch(sess, texts)
        emb = emb.astype(np.float16).reshape(len(texts), -1)
        col = pa.FixedSizeListArray.from_arrays(
            pa.array(emb.reshape(-1), type=pa.float16()), DIM)
        out = pa.table({"id": pa.array(ids_all, type=pa.string()), "embedding": col})
        if writer is None:
            writer = pq.ParquetWriter(out_dir / "emb_qianwen.parquet", out.schema)
        writer.write_table(out)
        texts, ids_all = [], []
        done += REQ_BATCH
        ckpt.write_text(json.dumps({"done": done}))
        rate = done / max(time.time() - t0, 1)
        print(f"\r{done:,}/{n_total:,}  {rate:.1f} chunks/s", end="", flush=True)
    if texts:  # 尾部
        emb = embed_batch(sess, texts)
        col = pa.FixedSizeListArray.from_arrays(
            pa.array(emb.astype(np.float16).reshape(-1), type=pa.float16()), DIM)
        out = pa.table({"id": pa.array(ids_all, type=pa.string()), "embedding": col})
        if writer is None:
            writer = pq.ParquetWriter(out_dir / "emb_qianwen.parquet", out.schema)
        writer.write_table(out)
        done += len(texts)
        ckpt.write_text(json.dumps({"done": done}))
    if writer:
        writer.close()
    print(f"\n完成 {done:,} chunks，耗时 {(time.time()-t0)/60:.1f} 分钟")

    # queries
    queries = json.loads((out_dir / "queries.json").read_text(encoding="utf-8"))
    qtexts = [q["text"] for q in queries]
    qemb = []
    for i in range(0, len(qtexts), REQ_BATCH):
        qemb.append(embed_batch(sess, qtexts[i:i + REQ_BATCH]))
    qemb = np.vstack(qemb)
    qvec = {q["id"]: [float(x) for x in v] for q, v in zip(queries, qemb)}
    (out_dir / "qvec_qianwen.json").write_text(json.dumps(qvec), encoding="utf-8")
    print(f"queries 完成 → {out_dir / 'qvec_qianwen.json'}")


if __name__ == "__main__":
    main()
