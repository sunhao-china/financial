# -*- coding: utf-8 -*-
"""BM25 索引完整性诊断：直接读 postings，验证桶内文档是否真的包含该词。"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from corpus.bm25 import BM25Index

bm = BM25Index()
chunks = pq.read_table(config.DATA_DIR / "chunks.parquet", columns=["text"])
texts = chunks.column("text").to_pylist()


def postings_check(term, n=5):
    i = int(np.searchsorted(bm.terms, term))
    ok = i < len(bm.terms) and bm.terms[i] == term
    print(f"\nterm={term!r} in_vocab={ok}")
    if not ok:
        return
    df = int(bm.df[i])
    o0, o1 = int(bm.offsets[i]), int(bm.offsets[i + 1])
    print(f"  df={df} postings={o1 - o0} (期望 df==postings 且每篇都含词)")
    docs = bm.pdoc[o0:o0 + n]
    tfs = bm.ptf[o0:o0 + n]
    for d, tf in zip(docs, tfs):
        contains = term in texts[int(d)]
        print(f"  doc={int(d)} tf={int(tf)} 含词={contains} {texts[int(d)][:60]!r}")
    # 抽样 1000 篇统计含词率
    if df > 1000:
        idx = np.random.default_rng(1).integers(0, df, 1000)
        samp = bm.pdoc[o0 + idx]
        rate = np.mean([term in texts[int(d)] for d in samp])
        print(f"  随机 1000 篇含词率: {rate:.3f}")


for t in ["巴基斯坦", "GDP", "年均增长", "综合保税区", "进出口", "摩根大通", "降准"]:
    postings_check(t)
