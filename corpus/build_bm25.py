# -*- coding: utf-8 -*-
"""P3 BM25 倒排索引构建（大纲 §5.2）：chunks.parquet → jieba 分词倒排索引。

单遍低内存方案：每个 worker 扫描自己区间一次，维护局部词表 dict（token→tid_local），
产出 (tid_local, doc, tf) 三元组 + 排序后词表；主进程 heapq.merge 流式合并各 worker
已排序词表为全局词表（避免 np.unique 大对象数组的内存峰值——旧两遍方案在词表合并时
23.8GB 触发 OOM 被杀），再向量化换算全局 gid 并散落到 postings。

产物（data/index/bm25/）：
    terms.npy         全局词表（字典序；查询 np.searchsorted 二分）
    df.npy            int32 文档频率
    offsets.npy       int64 词在 postings 的偏移（length = V+1）
    postings_doc.bin  int32 文档号（= chunk 行号 = HNSW label，与 meta.parquet 对齐）
    postings_tf.bin   uint8 词频
    doc_lens.npy      int32 每 chunk 词数
    params.json       {k1, b, n_docs, avgdl}

用法: python corpus/build_bm25.py [--limit N 试验] [--workers 6]
"""
import argparse
import heapq
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

OUT_DIR = config.DATA_DIR / "index" / "bm25"
CHUNKS = config.DATA_DIR / "chunks.parquet"
DICT_FILE = config.DATA_DIR / "fin_dict.txt"
STOP_FILE = config.DATA_DIR / "stopwords.txt"

K1, B = 1.5, 0.75
_TOKEN_OK = re.compile(r"[\w一-鿿]")


def _init():
    import jieba
    if DICT_FILE.exists():
        jieba.load_userdict(str(DICT_FILE))
    stop = set(l.strip() for l in STOP_FILE.read_text(encoding="utf-8").splitlines()
               if l.strip())
    return jieba, stop


def tokenize(jieba, stop, text):
    return [t for t in jieba.lcut_for_search(text)
            if t not in stop and len(t) <= 32 and _TOKEN_OK.search(t)]


def _range_batches(pf, s, e, batch_size=20_000):
    """迭代 (行号起始, text 列表)：只取 [s, e) 区间的行（仍需全表扫描）。"""
    row0 = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        n = batch.num_rows
        lo, hi = max(s - row0, 0), min(e - row0, n)
        if lo < hi:
            yield row0 + lo, batch.column("text").slice(lo, hi - lo).to_pylist()
        row0 += n
        if row0 >= e:
            break


def _worker(args):
    """本区间单遍扫描 → npz：排序词表 + (tid_local, doc, tf) 三元组 + doc_lens。"""
    s, e, tmp, wid = args
    jieba, stop = _init()
    pf = pq.ParquetFile(str(CHUNKS))
    vocab = {}                      # token → tid_local
    gids, docs, tfs, lens = [], [], [], []
    t0 = time.time()
    n = 0
    for row, texts in _range_batches(pf, s, e):
        for off, t in enumerate(texts):
            toks = tokenize(jieba, stop, t)
            lens.append(len(toks))
            if not toks:
                continue
            ids = [vocab.setdefault(tk, len(vocab)) for tk in toks]
            u, c = np.unique(np.array(ids, np.int32), return_counts=True)
            gids.append(u)
            docs.append(np.full(len(u), row + off, np.int32))
            tfs.append(c.astype(np.uint8))
        n += len(texts)
        if n % 100_000 < 20_000:
            print(f"  worker {wid} {n:,} {time.time()-t0:.0f}s", flush=True)
    gids = np.concatenate(gids)
    docs = np.concatenate(docs)
    tfs = np.concatenate(tfs)
    # 词表按字典序排序；df_w（文档频率）按 tid_local 序
    vocab_arr = np.array(list(vocab.keys()), dtype=object)
    order = np.argsort(vocab_arr, kind="stable")
    df_w = np.bincount(gids, minlength=len(vocab_arr)).astype(np.int32)
    np.savez(tmp, vocab_sorted=vocab_arr[order], tid_of_sorted=order.astype(np.int32),
             df_w=df_w, gids=gids, docs=docs, tfs=tfs, lens=np.array(lens, np.int32))
    return tmp, len(gids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    pf = pq.ParquetFile(str(CHUNKS))
    N = pf.metadata.num_rows if not args.limit else min(pf.metadata.num_rows, args.limit)
    W = args.workers
    ranges = [(i * N // W, (i + 1) * N // W) for i in range(W)]
    ranges = [(s, e) for s, e in ranges if s < e]

    t0 = time.time()
    print(f"单遍扫描（{N:,} chunks / {len(ranges)} workers）...", flush=True)
    tmp = str(OUT_DIR / "_tmp")
    Path(tmp).mkdir(parents=True, exist_ok=True)
    from multiprocessing import Pool
    with Pool(len(ranges)) as pool:
        res = pool.map(_worker, [(s, e, f"{tmp}/w{i}.npz", i)
                                 for i, (s, e) in enumerate(ranges)])
    total_postings = sum(r[1] for r in res)
    print(f"三元组 {total_postings:,} {time.time()-t0:.0f}s", flush=True)

    # 流式 k 路合并局部词表（已排序）→ 全局词表 + 全局 df
    print("合并词表...", flush=True)
    data = [np.load(f, allow_pickle=True) for f, _ in res]
    iters = [iter(zip(d["vocab_sorted"].tolist(), [i] * len(d["tid_of_sorted"]),
                      d["df_w"][d["tid_of_sorted"]].tolist()))
             for i, d in enumerate(data)]
    terms, dfs, last, acc = [], [], None, 0
    for term, _, dfv in heapq.merge(*iters):
        if term != last:
            if last is not None:
                terms.append(last)
                dfs.append(acc)
            last, acc = term, 0
        acc += int(dfv)
    if last is not None:
        terms.append(last)
        dfs.append(acc)
    gterms = np.array(terms, dtype=object)
    df = np.array(dfs, np.int32)
    print(f"全局词表 {len(gterms):,} 词 {time.time()-t0:.0f}s", flush=True)

    # 每个 worker：tid_local → 全局 gid → 散落 postings
    # 修复 2026-10-17：旧版 idx = pos[g] 对同词重复出现时产生重复下标，
    # 花式赋值同一下标只保留最后一次写入，其余槽位保持 0（doc=0/tf=0 垃圾 postings），
    # 导致每词每 worker 实际只写入 1 条倒排，BM25 检索近零召回。
    # 新版：按 gid 排序后以「词内累计偏移」生成互不重复的下标。
    print("散落 postings...", flush=True)
    offsets = np.concatenate([[0], np.cumsum(df.astype(np.int64))]).astype(np.int64)
    pdoc = np.empty(total_postings, np.int32)
    ptf = np.empty(total_postings, np.uint8)
    pos = offsets[:-1].copy()
    for d in data:
        gpos = np.searchsorted(gterms, d["vocab_sorted"])
        gid_map = np.empty(len(d["df_w"]), np.int32)
        gid_map[d["tid_of_sorted"]] = gpos.astype(np.int32)
        g = gid_map[d["gids"]]
        order = np.argsort(g, kind="stable")          # 按全局 gid 排序
        gs = g[order]
        counts = np.bincount(gs, minlength=len(gterms))   # 本 worker 各词 postings 数
        c0 = np.concatenate([[0], np.cumsum(counts)])[:-1]  # 各词在本 worker 排序序列中的起始位置
        idx = pos[gs] + (np.arange(len(gs)) - c0[gs])  # 词内偏移，保证 idx 互不重复
        pdoc[idx] = d["docs"][order]
        ptf[idx] = d["tfs"][order]
        pos += counts
    lens = np.concatenate([d["lens"] for d in data])
    avgdl = float(lens.mean())

    # 槽位完整性硬校验（写文件前）：每个 postings 槽位都必须写入（tf>0、doc 在界内）
    assert np.all(ptf > 0), "存在未写入的 postings 槽位（tf=0）"
    assert int(pdoc.min()) >= 0 and int(pdoc.max()) < N, "postings 文档号越界"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "terms.npy", gterms)
    np.save(OUT_DIR / "df.npy", df)
    np.save(OUT_DIR / "offsets.npy", offsets)
    pdoc.tofile(OUT_DIR / "postings_doc.bin")
    ptf.tofile(OUT_DIR / "postings_tf.bin")
    np.save(OUT_DIR / "doc_lens.npy", lens)
    (OUT_DIR / "params.json").write_text(json.dumps(
        {"k1": K1, "b": B, "n_docs": int(N), "avgdl": avgdl}, ensure_ascii=False),
        encoding="utf-8")

    # postings 抽样自检（流式读文本，避免整列 take 触发 string 2GB offset 溢出）。
    # 只抽 df≥5 的词且只读桶内 postings——df<5 的词读满 5 条会越界串到相邻词的桶，造成假阳性失败。
    rng2 = np.random.default_rng(7)
    candidates = np.flatnonzero(df >= 5)
    terms_idx = candidates[rng2.integers(0, len(candidates), size=100)]
    wanted = {int(x) for i in terms_idx for x in pdoc[offsets[i]:offsets[i] + 5]}
    found, row0 = {}, 0
    for batch in pq.ParquetFile(str(CHUNKS)).iter_batches(batch_size=20_000, columns=["text"]):
        for j in range(batch.num_rows):
            if row0 + j in wanted:
                found[row0 + j] = batch.column("text")[j].as_py()
        row0 += batch.num_rows
        if len(found) == len(wanted):
            break
    fails = sum(1 for i in terms_idx
                for j in range(5)
                if gterms[i] not in found[int(pdoc[offsets[i] + j])])
    print(f"✓ postings 自检：{len(terms_idx)} 词 × 5 篇，含词失败 {fails}")

    # 一致性自检：chunks 行号 ↔ meta.parquet label 对齐（RRF 融合的前提）
    meta = pq.read_table(config.DATA_DIR / "index" / "meta.parquet")
    assert len(meta) >= N, f"meta {len(meta)} < chunks {N}"
    ids_chunks = pq.read_table(CHUNKS, columns=["id"]).column(0).to_pylist()
    ids_meta = meta.column("id").to_pylist()
    rng = np.random.default_rng(42)
    for i in rng.integers(0, N, size=200):
        if ids_chunks[i] != ids_meta[i]:
            raise SystemExit(f"行号↔label 不一致: row {i}")
    print("✓ 行号↔label 对齐自检通过（200 采样）")

    print(f"✓ BM25 索引完成：{N:,} 文档 / {len(gterms):,} 词 / {total_postings:,} postings "
          f"/ avgdl={avgdl:.1f} / 总耗时 {(time.time()-t0)/60:.1f} 分钟 → {OUT_DIR}")
    sys.exit(0)


if __name__ == "__main__":
    main()
