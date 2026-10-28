# -*- coding: utf-8 -*-
"""P2 向量索引构建（大纲 §8）：emb.parquet → HNSW 索引 + id 映射。

- 512 维 fp16（bge-m3 MRL），hnswlib cosine（向量已归一化，cosine == 内积）
- 内存 ≈ 向量 2.9GB + 图 ~1GB（286 万点，M=16）→ 远低于 32G 预算
- 产物：data/index/hnsw.bin（hnswlib 索引）+ data/index/meta.parquet（id 映射与检索元数据）
- 用法: python corpus/build_index.py [emb parquet] [--dim 512] [--limit N 试验用]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

M = 16                  # HNSW 连接数（召回/内存平衡）
EF_CONSTRUCTION = 200   # 建索引搜索宽度（越大质量越高，慢）
EF_SEARCH = 100         # 查询时默认 ef


def build(emb_path, out_dir, dim=512, limit=0):
    import hnswlib
    pf = pq.ParquetFile(emb_path)
    n_total = pf.metadata.num_rows if not limit else min(pf.metadata.num_rows, limit)
    idx = hnswlib.Index(space="cosine", dim=dim)
    idx.init_index(max_elements=n_total, M=M, ef_construction=EF_CONSTRUCTION)
    idx.set_num_threads(8)

    ids_all = []
    labels = []
    t0 = time.time()
    n = 0
    for batch in pf.iter_batches(batch_size=10_000):
        t = pa.Table.from_batches([batch])
        if limit and n + t.num_rows > limit:
            t = t.slice(0, limit - n)
        if t.num_rows == 0:
            break
        emb = np.asarray(t.column("embedding").to_pylist(), dtype=np.float32)
        # FixedSizeListArray to_pylist 给出 (n, dim) list；展平校验
        if emb.ndim == 1:
            emb = emb.reshape(t.num_rows, -1)
        ids = t.column("id").to_pylist()
        lab = np.arange(n, n + t.num_rows)
        idx.add_items(emb, lab)
        ids_all.extend(ids)
        labels.append(lab)
        n += t.num_rows
        print(f"\r{n:,}/{n_total:,}  {(time.time()-t0):.0f}s", end="", flush=True)
        if limit and n >= limit:
            break

    idx.set_ef(EF_SEARCH)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx.save_index(str(out_dir / "hnsw.bin"))
    meta = pa.table({"label": pa.array(np.concatenate(labels), type=pa.int64()),
                     "id": pa.array(ids_all, type=pa.string())})
    pq.write_table(meta, out_dir / "meta.parquet")
    print(f"\n索引完成：{n:,} 点 → {out_dir}/hnsw.bin + meta.parquet，"
          f"耗时 {(time.time()-t0)/60:.1f} 分钟")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emb", nargs="?", default=str(config.DATA_DIR / "emb.parquet"))
    ap.add_argument("--out", default=str(config.DATA_DIR / "index"))
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    build(args.emb, args.out, args.dim, args.limit)


if __name__ == "__main__":
    main()
