# -*- coding: utf-8 -*-
"""BM25 查询（大纲 §5.2）：加载 build_bm25.py 产出的倒排索引，返回 top-k (label, score)。

用法:
    bm = BM25Index("/data/financial/data/index/bm25")
    bm.search("综合保税区进出口值", k=50)  # -> [(label, score), ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from corpus.build_bm25 import _init, tokenize


class BM25Index:
    def __init__(self, index_dir: Path | str = None):
        d = Path(index_dir) if index_dir else config.DATA_DIR / "index" / "bm25"
        self.jieba, self.stop = _init()
        self.terms = np.load(d / "terms.npy", allow_pickle=True)          # 字典序词表
        self.df = np.load(d / "df.npy")                                    # int32
        self.offsets = np.load(d / "offsets.npy")                          # int64, V+1
        self.pdoc = np.memmap(d / "postings_doc.bin", dtype=np.int32, mode="r")
        self.ptf = np.memmap(d / "postings_tf.bin", dtype=np.uint8, mode="r")
        self.doc_lens = np.load(d / "doc_lens.npy")                        # int32
        p = json.loads((d / "params.json").read_text(encoding="utf-8"))
        self.k1, self.b, self.n, self.avgdl = p["k1"], p["b"], p["n_docs"], p["avgdl"]

    def search(self, query: str, k: int = 50) -> list:
        """BM25 top-k，返回 [(label, score)] 降序（score>0）。"""
        toks = set(tokenize(self.jieba, self.stop, query))
        if not toks:
            return []
        score = np.zeros(self.n, np.float32)
        N, k1, b, avgdl = self.n, self.k1, self.b, self.avgdl
        for t in toks:
            i = np.searchsorted(self.terms, t)
            if i >= len(self.terms) or self.terms[i] != t:
                continue
            df = int(self.df[i])
            o0, o1 = int(self.offsets[i]), int(self.offsets[i + 1])
            docs = self.pdoc[o0:o1]
            tfs = self.ptf[o0:o1]
            idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
            denom = tfs.astype(np.float32) + k1 * (1 - b + b * self.doc_lens[docs] / avgdl)
            score[docs] += idf * tfs * (k1 + 1) / denom
        if k >= self.n:
            top = np.flatnonzero(score > 0)
            top = top[np.argsort(-score[top])]
        else:
            top = np.argpartition(score, -k)[-k:]
            top = top[np.argsort(-score[top])]
        return [(int(i), float(score[i])) for i in top if score[i] > 0][:k]


if __name__ == "__main__":
    bm = BM25Index()
    for q in sys.argv[1:] or ["综合保税区进出口值", "央行降准", "人民币汇率破7"]:
        t0 = time.time()
        hits = bm.search(q, 50)
        print(f"{q}: {len(hits)} 条 {time.time()-t0:.3f}s  top3={hits[:3]}")
