# -*- coding: utf-8 -*-
"""P3 试点 A/B：抽查 12 题（跨四类）对比 hybrid_rerank(旧 50/30 等权) vs hybrid_rerank_deep(新 300/100 加权1:3)。

用法（ECS）: .venv/bin/python corpus/pilot_ab.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from corpus.retrieval import HybridRetriever

# 跨类别抽查；重灾题（变体霸占诊断：q031=189、q065=115、q050=42、q025=30）重点覆盖
PICK = {
    "fact_qa": ["q001", "q004"],
    "analysis": ["q031", "q024", "q050", "q025"],
    "multi_doc": ["q052", "q055"],
    "comparison": ["q065", "q087", "q091", "q060"],
}

CONFIGS = {
    "sum-old(50/30,1:1)": dict(expand=True, rerank=True, bm25_top=50, rerank_top=30,
                               rrf_w=(1.0, 1.0), vec_raw=False),
    "sum(300/100,1:3)": dict(expand=True, rerank=True),
    "max(300/100,1:1)": dict(expand=True, rerank=True, rrf_mode="max", rrf_w=(1.0, 1.0)),
    "max(300/100,1:1,无扩展)": dict(expand=False, rerank=True, rrf_mode="max",
                                    rrf_w=(1.0, 1.0)),
}


def score(hits, positives):
    ranks = [h.rank for h in hits if h.label in positives]
    n_pos = max(len(positives), 1)
    return {"r@10": sum(1 for h in hits[:10] if h.label in positives) / n_pos,
            "r@20": sum(1 for h in hits[:20] if h.label in positives) / n_pos,
            "hit@10": 1.0 if ranks and min(ranks) <= 10 else 0.0,
            "hit@20": 1.0 if ranks and min(ranks) <= 20 else 0.0,
            "top_rank": min(ranks) if ranks else None}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    qa = {q["id"]: q for q in json.loads(
        (config.DATA_DIR / "evals" / "qa_set.json").read_text(encoding="utf-8"))}
    ret = HybridRetriever()
    t0 = time.time()
    agg = {c: {m: [] for m in ["r@10", "r@20", "hit@10", "hit@20"]} for c in CONFIGS}
    for cat, ids in PICK.items():
        for qid in ids:
            q = qa[qid]
            positives = ret.labels_for_docs(q["source_doc_ids"])
            if not positives:
                print(f"{qid} 正例为空，跳过", flush=True)
                continue
            print(f"[{qid} {cat}] {q['question'][:40]}", flush=True)
            for name, cfg in CONFIGS.items():
                t1 = time.time()
                hits = ret.retrieve(q["question"], n=20, **cfg)
                m = score(hits, positives)
                for k in ("r@10", "r@20", "hit@10", "hit@20"):
                    agg[name][k].append(m[k])
                print(f"   {name:16s} r@10={m['r@10']:.2f} r@20={m['r@20']:.2f} "
                      f"hit@10={m['hit@10']:.0f} hit@20={m['hit@20']:.0f} "
                      f"首正例排名={m['top_rank']} ({time.time()-t1:.0f}s)", flush=True)
    print("=== 合计 ===", flush=True)
    for name, ms in agg.items():
        print(f"  {name:16s} " + " ".join(f"{k}={sum(v)/len(v):.2f}" for k, v in ms.items()),
              flush=True)
    print(f"总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
