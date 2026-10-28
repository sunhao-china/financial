# -*- coding: utf-8 -*-
"""AutoDL GPU 上批量 embedding（sentence-transformers，与在线查询同一管线）。

一致性约定（tools/model_lock.json 锁定）：
- 模型文件 sha256 与 model_lock.json 一致
- sentence-transformers 6.1.0 + torch fp16 GPU 推理
- bge-m3 输出 truncate_dim=512（MRL 截断）+ normalize
- 在线阶段（ECS CPU）用同一模型文件 + 同一库 + 同一 truncate_dim/normalize

- 输入：chunks parquet（text + id 列）
- 输出：embedding parquet（id + embedding float16，512 维）
- 断点续跑：--checkpoint 每 N 批落盘一次，重启后跳过已处理行
- 用法（AutoDL 上）：
    /root/miniconda3/envs/emb/bin/python embed_all.py chunks.parquet out.parquet \
        --model-dir /root/autodl-tmp/models/bge-m3 --dim 512 --checkpoint ckpt.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

BATCH = 256             # 每批 chunk 数（4090 fp16 实测 726/s 的批大小）
CHECKPOINT_EVERY = 20   # 每 20 批落盘断点


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("out")
    ap.add_argument("--model-dir", default="/root/autodl-tmp/models/bge-m3")
    ap.add_argument("--dim", type=int, default=512, help="MRL 截断维度")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 行（试验用）")
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()

    print(f"加载模型 {args.model_dir}（fp16, cuda）...", flush=True)
    model = SentenceTransformer(args.model_dir, device="cuda",
                                trust_remote_code=True,
                                model_kwargs={"torch_dtype": torch.float16})
    print(f"加载完成。输出 {args.dim} 维 + 归一化", flush=True)

    pf = pq.ParquetFile(args.chunks)
    n_total = pf.metadata.num_rows
    if args.limit:
        n_total = min(n_total, args.limit)
    skip_rows = 0   # 启动时待跳过（断点续跑），与运行中完成计数 done 分离
    if args.checkpoint and Path(args.checkpoint).exists():
        skip_rows = json.loads(Path(args.checkpoint).read_text())["done"]
        print(f"断点续跑：跳过前 {skip_rows:,} 行")

    writer = None
    done = 0
    t0 = time.time()
    texts, ids = [], []
    def flush_batch():
        nonlocal done, writer
        b_texts, b_ids = texts[:args.batch], ids[:args.batch]
        del texts[:args.batch], ids[:args.batch]
        emb = model.encode(b_texts, truncate_dim=args.dim,
                           normalize_embeddings=True, batch_size=args.batch,
                           convert_to_numpy=True)
        emb = emb.astype(np.float16).reshape(len(b_texts), -1)
        emb_col = pa.FixedSizeListArray.from_arrays(
            pa.array(emb.reshape(-1), type=pa.float16()), list_size=args.dim)
        out = pa.table({"id": pa.array(b_ids, type=pa.string()),
                        "embedding": emb_col})
        if writer is None:
            writer = pq.ParquetWriter(args.out, out.schema)
        writer.write_table(out)
        done += len(b_ids)
        if args.checkpoint and done % (args.batch * CHECKPOINT_EVERY) == 0:
            Path(args.checkpoint).write_text(json.dumps({"done": done}))
        rate = done / max(time.time() - t0, 1)
        print(f"\r{done:,}/{n_total:,}  {rate:.0f} chunks/s", end="", flush=True)

    for batch in pf.iter_batches(batch_size=args.batch):
        t = pa.Table.from_batches([batch])
        if skip_rows:
            s = min(skip_rows, t.num_rows)
            t = t.slice(s)
            skip_rows -= s
        if t.num_rows == 0:
            continue
        texts.extend(t.column("text").to_pylist())
        ids.extend(t.column("id").to_pylist())
        while len(texts) >= args.batch:
            flush_batch()
            if done >= n_total:
                break
        if done >= n_total:
            break
    if texts:  # 尾部不足一批
        flush_batch()
    if writer:
        writer.close()
    print(f"\n完成 {done:,} chunks → {args.out}，耗时 {(time.time()-t0)/60:.1f} 分钟")
    if args.checkpoint:
        Path(args.checkpoint).write_text(json.dumps({"done": done, "finished": True}))


if __name__ == "__main__":
    main()
