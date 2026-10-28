# -*- coding: utf-8 -*-
"""P3 comparison 类查询分解第三轮：并联融合验证（生产设计预演）。

设计（基于前两轮证据）：
  - 单主题 chunk 被混合问句 rerank 埋掉（q060/q064）→ 需要分侧 rerank
  - 跨主题 chunk 靠混合问句 rerank 才捞得动（q088）→ 需要全局 rerank
  - 两者互补 → 并联：每侧独立检索+rerank 各取 top-10，全池再按原问句 rerank
    取 top-10，三路合并去重成 20。

本轮用真实 LLM 扩展变体当「侧」（union-llm），对比手工分侧（union-hand）与
当前生产（prod），验证生产设计的预期收益。

用法（ECS）: .venv/bin/python corpus/diag_comp3.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from corpus.retrieval import HybridRetriever

# 11 道真 comparison 题的手工分侧（q065 为评估集噪声题，跳过）
HAND_SIDES = {
    "q087": ["G20财长会 改革多边开发银行 新融资机制", "一带一路 沿线国家 进出口 增长9.8%"],
    "q063": ["证监会 跨境审计监管合作 中概股", "香港 国际金融中心 财务造假 监管"],
    "q088": ["国务院 提高上市公司质量的意见", "国企改革三年行动 提升上市公司质量"],
    "q057": ["香港 一带一路 角色定位", "人民币国际化 香港离岸人民币市场 作用"],
    "q064": ["财政赤字率安排 加力提效", "预算管理制度改革 稳字当头 稳中求进"],
    "q061": ["国企改革 双百行动 激发企业活力", "美国 独立董事制度 企业活力"],
    "q089": ["2016年 公募基金 A股 走势 判断", "2020年 疫情 私募基金经理 应对逻辑"],
    "q090": ["朱佩江 中关村区块链产业联盟", "国际货币基金组织前副总裁 泡沫 预期 态度"],
    "q060": ["科创板 支持科技创新", "新修订证券法 支持科技创新"],
    "q091": ["高盛 2023年 美国经济 预测", "瑞银 2023年 中国经济 预测"],
    "q062": ["广东省外商投资权益保护条例", "国务院办公厅 外贸保稳提质 意见"],
}


def score(hits, positives):
    ranks = [h.rank for h in hits if h.label in positives]
    n_pos = max(len(positives), 1)
    return {"r@10": sum(1 for h in hits[:10] if h.label in positives) / n_pos,
            "r@20": sum(1 for h in hits[:20] if h.label in positives) / n_pos,
            "hit@10": 1.0 if ranks and min(ranks) <= 10 else 0.0,
            "top_rank": min(ranks) if ranks else None}


def rerank_pool(ret, query, fused, n, positives=None):
    """对融合池 rerank 取 top-n；返回 (hits, 池内正例首排名)。"""
    from llm_client import client
    pr = min((i + 1 for i, (lab, _) in enumerate(fused) if lab in positives),
             default=None) if positives else None
    cand = ret._make_hits(fused, 100, "decomp")
    try:
        scored = client.rerank(query, [h.text for h in cand])
        top = [cand[r["index"]] for r in scored[:n]]
        for r, h in zip(scored[:n], top):
            h.score, h.rank = r["score"], 1 + scored.index(r)
        return top, pr
    except Exception:
        return cand[:n], pr


def union_retrieve(ret, question, sides, positives):
    """并联：每侧独立检索+分侧 rerank top-10 + 全池原问句 rerank top-10 → 合并 20。"""
    out, pools = [], []
    for s in sides:
        fused = ret.filter_meta(ret.fuse_rrf([ret.vector_search([s]),
                                              ret.bm25_search([s], k=300)],
                                             mode="max"))[:100]
        hits, pr = rerank_pool(ret, s, fused, 10, positives)
        out.extend(hits)
        pools.append(pr)
    lists = []
    for s in sides:
        lists.append(ret.vector_search([s]))
        lists.append(ret.bm25_search([s], k=300))
    gfused = ret.filter_meta(ret.fuse_rrf(lists, mode="max"))[:100]
    ghits, gpr = rerank_pool(ret, question, gfused, 10, positives)
    out.extend(ghits)
    best = {}                           # label → SearchHit（跨路去重保留最高 rerank 分）
    for h in out:
        if h.label not in best or h.score > best[h.label].score:
            best[h.label] = h
    hits = sorted(best.values(), key=lambda x: -x.score)[:20]
    for i, h in enumerate(hits, 1):
        h.rank = i
    return hits, pools + [gpr]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    qa = {q["id"]: q for q in json.loads(
        (config.DATA_DIR / "evals" / "qa_set.json").read_text(encoding="utf-8"))}
    ret = HybridRetriever()
    t0 = time.time()
    agg = {c: {m: [] for m in ("r@10", "r@20", "hit@10")} for c in
           ("prod", "union-llm", "union-hand")}
    for qid, hand_sides in HAND_SIDES.items():
        q = qa[qid]
        positives = ret.labels_for_docs(q["source_doc_ids"])
        if not positives:
            print(f"{qid} 正例为空，跳过", flush=True)
            continue
        print(f"\n[{qid}] {q['question'][:44]}", flush=True)
        s2 = ret.expand(q["question"])
        llm_sides = [x for x in (s2.queries or []) if x.strip()]
        print(f"   LLM 变体({len(llm_sides)}): {llm_sides}", flush=True)

        for name in ("prod", "union-llm", "union-hand"):
            t1 = time.time()
            if name == "prod":
                hits = ret.retrieve(q["question"], s2=s2, rerank=True, n=20)
                pool = ""
            else:
                sides = llm_sides if name == "union-llm" else hand_sides
                hits, pool_info = union_retrieve(ret, q["question"], sides, positives)
                pool = " 池内正例#" + "/".join(str(p) if p else "-" for p in pool_info)
            m = score(hits, positives)
            for k in ("r@10", "r@20", "hit@10"):
                agg[name][k].append(m[k])
            print(f"   {name:12s} r@10={m['r@10']:.2f} r@20={m['r@20']:.2f} "
                  f"hit@10={m['hit@10']:.0f} 首正例排名={m['top_rank']}{pool} "
                  f"({time.time()-t1:.0f}s)", flush=True)

    print("\n=== 合计（11 题）===", flush=True)
    for name, ms in agg.items():
        print(f"  {name:12s} " + " ".join(f"{k}={sum(v)/len(v):.2f}" for k, v in ms.items()),
              flush=True)
    print(f"总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
