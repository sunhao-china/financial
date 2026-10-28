# -*- coding: utf-8 -*-
"""BM25 诊断：词项命中率 + top 命中文本抽查（定位 eval 中 BM25 近零 recall 的原因）。"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from corpus.bm25 import BM25Index
from corpus.build_bm25 import _init, tokenize

bm = BM25Index()
jieba, stop = _init()
print(f"index: {bm.n:,} docs, {len(bm.terms):,} terms, avgdl={bm.avgdl:.1f}", flush=True)

chunks = pq.read_table(config.DATA_DIR / "chunks.parquet", columns=["id", "doc_id", "text"])
ids = chunks.column("id").to_pylist()
doc_ids = chunks.column("doc_id").to_pylist()
texts = chunks.column("text").to_pylist()

qa = __import__("json").loads((config.DATA_DIR / "evals" / "qa_set.json").read_text(encoding="utf-8"))
qs = [(q["id"], q["question"], q["source_doc_ids"]) for q in qa
      if q["category"] == "fact_qa"][:3]

for qid, q, src in qs:
    print(f"\n=== {qid}: {q}  (源文档 {src})", flush=True)
    toks = tokenize(jieba, stop, q)
    in_vocab = [tk for tk in toks
                if (i := int(np.searchsorted(bm.terms, tk))) < len(bm.terms) and bm.terms[i] == tk]
    print(f" 查询词 {len(toks)} 个，命中词表 {len(in_vocab)} 个: {in_vocab}", flush=True)
    hits = bm.search(q, 10)
    print(f" hits={len(hits)}", flush=True)
    for lab, s in hits[:5]:
        hit_src = "✔" if doc_ids[lab] in src else "✘"
        print(f"   {hit_src} {s:7.2f} {ids[lab][:12]} doc={doc_ids[lab][:14]} {texts[lab][:50]!r}", flush=True)

# 直接从源文档取一个 12 字短语当查询（排除问题质量因素）
print("\n=== 源文档短语直接查询 ===", flush=True)
clean = pq.read_table(config.CLEANED_PARQUET, columns=["_id", "text"])
ctext = {str(i): t for i, t in zip(clean.column("_id").to_pylist(),
                                   clean.column("text").to_pylist())}
qid, q, src = qs[0]
doc = ctext[src[0]]
phrase = doc[100:112]
print(f" phrase={phrase!r}  doc={src[0]}", flush=True)
hits = bm.search(phrase, 10)
for lab, s in hits[:5]:
    hit_src = "✔" if doc_ids[lab] in src else "✘"
    print(f"   {hit_src} {s:7.2f} {ids[lab][:12]} {texts[lab][:50]!r}", flush=True)
