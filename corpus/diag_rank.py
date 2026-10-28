# -*- coding: utf-8 -*-
"""失败题排名追踪：正例 chunk 在向量/BM25 两路 top-1000 的排名 + 源文档切分实况。

诊断「匹配层断裂」的具体位置，决定语料重整理方向（重切 / 元数据富化 / 查询侧）。
用法（ECS）: .venv/bin/python corpus/diag_rank.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from corpus.retrieval import HybridRetriever

FAILED = ["q004", "q031", "q050", "q025", "q065", "q087", "q091", "q060"]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    qa = {q["id"]: q for q in json.loads(
        (config.DATA_DIR / "evals" / "qa_set.json").read_text(encoding="utf-8"))}
    ret = HybridRetriever()
    for qid in FAILED:
        q = qa[qid]
        pos = ret.labels_for_docs(q["source_doc_ids"])
        print(f"\n[{qid} {q['category']}] {q['question'][:50]}", flush=True)
        print(f"  正例 {len(pos)} chunks（源文档 {len(q['source_doc_ids'])} 篇）", flush=True)
        # 向量路：原问句 top-1000
        emb = ret.model.encode([q["question"]], truncate_dim=512,
                               normalize_embeddings=True, convert_to_numpy=True)[0]
        labs, _ = ret.idx.knn_query(emb.astype(np.float32), k=1000)
        ranks = [r for r, lab in enumerate(labs[0].tolist(), 1) if lab in pos]
        vrank = min(ranks) if ranks else None
        # BM25 路：原问句 top-1000
        bres = [lab for lab, s in ret.bm25.search(q["question"], 1000)]
        branks = [r for r, lab in enumerate(bres, 1) if lab in pos]
        brank = min(branks) if branks else None
        print(f"  向量路首正例排名: {vrank or '>1000 无'} | BM25路首正例排名: {brank or '>1000 无'}",
              flush=True)
        # 源文档切分实况：正例 chunk 逐个看（判断证据是否被切碎）
        for lab in sorted(pos)[:8]:
            t = ret.texts[lab].replace("\n", " ")
            print(f"    chunk[{lab}] ({len(t)}字): {t[:90]}", flush=True)


if __name__ == "__main__":
    main()
