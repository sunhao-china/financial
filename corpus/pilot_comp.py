# -*- coding: utf-8 -*-
"""comparison 分解生产路径试点：11 道 comparison 题对比
retrieve(decomp=False) 生产现状 vs retrieve(decomp=True) 分解分支，
两者共用同一份 LLM 扩展结果（s2 显式传入），隔离变体随机性。

用法（ECS）: .venv/bin/python corpus/pilot_comp.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from corpus.retrieval import HybridRetriever


def score(hits, positives):
    ranks = [h.rank for h in hits if h.label in positives]
    n_pos = max(len(positives), 1)
    return {"r@10": sum(1 for h in hits[:10] if h.label in positives) / n_pos,
            "r@20": sum(1 for h in hits[:20] if h.label in positives) / n_pos,
            "hit@10": 1.0 if ranks and min(ranks) <= 10 else 0.0,
            "top_rank": min(ranks) if ranks else None}


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    qa = json.loads((config.DATA_DIR / "evals" / "qa_set.json").read_text(encoding="utf-8"))
    comp = [q for q in qa if q["category"] == "comparison" and q["id"] != "q065"]
    ret = HybridRetriever()
    t0 = time.time()
    agg = {c: {m: [] for m in ("r@10", "r@20", "hit@10")} for c in ("prod", "decomp")}
    for q in comp:
        positives = ret.labels_for_docs(q["source_doc_ids"])
        print(f"\n[{q['id']}] {q['question'][:44]}", flush=True)
        s2 = ret.expand(q["question"])
        for name, kw in (("prod", {}), ("decomp", {"decomp": True})):
            t1 = time.time()
            hits = ret.retrieve(q["question"], s2=s2, rerank=True, n=20, **kw)
            m = score(hits, positives)
            for k in ("r@10", "r@20", "hit@10"):
                agg[name][k].append(m[k])
            print(f"   {name:6s} r@10={m['r@10']:.2f} r@20={m['r@20']:.2f} "
                  f"hit@10={m['hit@10']:.0f} 首正例排名={m['top_rank']} "
                  f"({time.time()-t1:.0f}s)", flush=True)
    print("\n=== 合计（11 题）===", flush=True)
    for name, ms in agg.items():
        print(f"  {name:6s} " + " ".join(f"{k}={sum(v)/len(v):.2f}" for k, v in ms.items()),
              flush=True)
    print(f"总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
