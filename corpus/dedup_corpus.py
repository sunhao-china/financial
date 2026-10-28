# -*- coding: utf-8 -*-
"""P0-3 去重（大纲 §7.3）三级流水：

1. 精确去重：原文 md5
2. 规范去重：空白折叠后 md5（转载最常见的差异就是空白/换行）
3. 近重复：字符 4-gram MinHash + 精确 Jaccard ≥ 0.8（见 jaccard.py 注释：
   simhash 在主题聚集的金融语料上无法分离真转载与同主题异文，已弃用）

近重复流程：minhash 仅用于排序（相似文档排在一起），窗口 ±100 内
两两比对精确 Jaccard，并查集合并，保留首次出现（文件顺序即优先级）。
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
sys.path.insert(0, str(Path(__file__).parent))
from jaccard import normalize, shingle_set, minhash_params, minhash_sketch, jaccard


def exact_dedup_mask(texts) -> np.ndarray:
    """精确重复（md5 相同）只保留首次出现。"""
    import hashlib
    seen = set()
    keep = np.ones(len(texts), dtype=bool)
    for i, t in tqdm(enumerate(texts), total=len(texts), desc="md5 精确去重"):
        h = hashlib.md5(t.encode("utf-8")).digest()
        if h in seen:
            keep[i] = False
        else:
            seen.add(h)
    return keep


def norm_dedup_mask(texts) -> np.ndarray:
    """空白折叠后的 md5 去重（转载文最常见的差异是空白/换行）。"""
    import hashlib
    seen = set()
    keep = np.ones(len(texts), dtype=bool)
    for i, t in tqdm(enumerate(texts), total=len(texts), desc="规范去重"):
        h = hashlib.md5(normalize(t).encode("utf-8")).digest()
        if h in seen:
            keep[i] = False
        else:
            seen.add(h)
    return keep


def near_dedup_mask(texts) -> np.ndarray:
    """精确 Jaccard ≥ JACCARD_MIN 合并到首次出现的文档。"""
    n = len(texts)
    print("4-gram 集合构建…")
    sets = [shingle_set(normalize(t)) for t in tqdm(texts, desc="4-gram 哈希")]
    print("minhash 计算…")
    a, b = minhash_params(config.MINHASH_PERM)
    sk = np.zeros((n, config.MINHASH_PERM), dtype=np.uint32)
    for i, s in enumerate(tqdm(sets, desc="minhash")):
        if s.size:
            sk[i] = minhash_sketch(s, a, b)
        else:
            sk[i] = 0xFFFFFFFF  # 空集永不匹配

    order = np.lexsort([sk[:, k] for k in range(config.JACCARD_SORT_KEYS - 1, -1, -1)])
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merged = 0
    w = config.JACCARD_WINDOW
    for i in tqdm(range(n), desc="窗口比对"):
        oi = order[i]
        si = sets[oi]
        for j in range(i + 1, min(n, i + w + 1)):
            oj = order[j]
            if jaccard(si, sets[oj]) >= config.JACCARD_MIN:
                ri, rj = find(oi), find(oj)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)  # 保留更小索引（更早文件）
                    merged += 1
    keep = np.array([find(i) == i for i in range(n)])
    print(f"Jaccard ≥ {config.JACCARD_MIN} 合并 {merged:,} 对")
    return keep


def dedup():
    n0_table = pq.read_table(config.FILTERED_PARQUET, columns=["text"])
    texts = n0_table.column("text").to_pylist()
    del n0_table  # 释放 text 列内存，后面还要装 4-gram 集合
    n0 = len(texts)

    # 1. 精确 md5
    keep_md5 = exact_dedup_mask(texts)
    print(f"精确去重: {n0:,} → {int(keep_md5.sum()):,}（移除 {n0 - int(keep_md5.sum()):,}）")

    # 2. 空白规范 md5
    idx = np.where(keep_md5)[0]
    keep_norm = norm_dedup_mask([texts[i] for i in idx])
    keep = np.zeros(n0, dtype=bool)
    keep[idx[keep_norm]] = True
    print(f"规范去重: {int(keep_md5.sum()):,} → {int(keep.sum()):,}（移除 {int(keep_md5.sum()) - int(keep.sum()):,}）")

    # 3. 近重复（Jaccard）
    idx = np.where(keep)[0]
    keep_near = near_dedup_mask([texts[i] for i in idx])
    keep[:] = False
    keep[idx[keep_near]] = True
    n1 = int(keep.sum())
    print(f"近似去重: {int(keep_norm.sum()):,} → {n1:,}（移除 {int(keep_norm.sum()) - n1:,}）")

    # mask 落盘，最终写出放到子进程：本进程内存已被 4-gram 集合占满，
    # 再读 2.5GB 全表会 OOM（实测 ArrowMemoryError）
    np.save(config.DATA_DIR / "dedup_keep.npy", keep)
    print(f"合计: {n0:,} → {n1:,} ({n1/n0*100:.1f}%)")
    subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--apply-mask"], check=True)


def apply_mask():
    """子进程模式：读 mask 应用过滤，分批流式读（5 万行/批）避免 OOM。"""
    keep = np.load(config.DATA_DIR / "dedup_keep.npy")
    pf = pq.ParquetFile(config.FILTERED_PARQUET)
    n_rows = pf.metadata.num_rows
    writer = None
    written = 0
    pos = 0
    for batch in pf.iter_batches(batch_size=50_000):
        m = keep[pos:pos + batch.num_rows]
        pos += batch.num_rows
        out = pa.Table.from_batches([batch]).filter(pa.array(m))
        if out.num_rows == 0:
            continue
        if writer is None:
            writer = pq.ParquetWriter(config.DEDUPED_PARQUET, out.schema)
        writer.write_table(out)
        written += out.num_rows
    if writer:
        writer.close()
    (config.DATA_DIR / "dedup_keep.npy").unlink()
    print(f"合计: {n_rows:,} → {written:,} ({written/n_rows*100:.1f}%)")
    print(f"已写入 {config.DEDUPED_PARQUET}")


if __name__ == "__main__":
    if "--apply-mask" in sys.argv:
        apply_mask()
    else:
        dedup()
