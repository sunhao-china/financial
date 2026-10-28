# -*- coding: utf-8 -*-
"""P3 comparison 类查询分解诊断：跨主题问句拆成单侧子问题分别检索，
验证「分侧检索能把正例捞进候选池，而融合问句捞不动」假设。

四个配置对比：
  prod           当前生产（LLM 扩展 + max-RRF + 原问句 rerank）
  decomp-max     分侧子查询全列表 max-RRF 融合 + 原问句 rerank
  decomp-quota   每侧融合后取 top-50 去重合并 + 原问句 rerank
  decomp-perside 每侧自融 top-100 + 该侧子查询单独 rerank top-10 + 跨侧合并 20

用法（ECS）: .venv/bin/python corpus/diag_comp.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from corpus.retrieval import HybridRetriever

# 11 道真 comparison 题的手工分侧（q065 为评估集噪声题，跳过）
SIDES = {
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


def decomp_retrieve(ret, question, sides, mode="quota", positives=None):
    """分侧检索：每侧 vector+BM25 → 融合 → rerank。

    mode="max":      所有侧的全部列表（向量+BM25）直接 max-RRF 融合 + 原问句 rerank。
    mode="quota":    每侧先自融取 top-50，再跨侧去重合并 + 原问句 rerank。
    mode="perside":  每侧自融 top-100 后**用该侧子查询单独 rerank**，每侧取 top-10，
                     跨侧去重合并成 20（排序按 rerank 分）——验证「混合问句 rerank
                     埋单侧证据」假设。
    返回 (hits, pool_info)：pool_info 为每侧正例在精排池中的最高排名（诊断用）。
    """
    from llm_client import client
    if mode == "perside":
        out, pool_info = [], []
        for s in sides:
            fused = ret.filter_meta(ret.fuse_rrf([ret.vector_search([s]),
                                                  ret.bm25_search([s], k=300)],
                                                 mode="max"))[:100]
            pr = min((i + 1 for i, (lab, _) in enumerate(fused) if lab in positives),
                     default=None) if positives else None
            pool_info.append(pr)
            cand = ret._make_hits(fused, 100, "decomp")
            try:
                scored = client.rerank(s, [h.text for h in cand])
                for r in scored[:10]:
                    h = cand[r["index"]]
                    h.score, h.rank = r["score"], 1 + scored.index(r)
                    out.append(h)
            except Exception:
                out.extend(cand[:10])
        best = {}                           # 跨侧去重保留最高分
        for h in out:
            if h.label not in best or h.score > best[h.label].score:
                best[h.label] = h
        hits = sorted(best.values(), key=lambda x: -x.score)[:20]
        for i, h in enumerate(hits, 1):
            h.rank = i
        return hits, pool_info

    if mode == "quota":
        acc = {}
        for s in sides:
            fused = ret.filter_meta(ret.fuse_rrf([ret.vector_search([s]),
                                                  ret.bm25_search([s], k=300)],
                                                 mode="max"))[:50]
            for lab, sc in fused:
                if sc > acc.get(lab, 0.0):
                    acc[lab] = sc
        fused = sorted(acc.items(), key=lambda x: -x[1])
    else:
        lists = []
        for s in sides:
            lists.append(ret.vector_search([s]))
            lists.append(ret.bm25_search([s], k=300))
        fused = ret.filter_meta(ret.fuse_rrf(lists, mode="max"))
    cand = ret._make_hits(fused, 100, "decomp")
    try:
        scored = client.rerank(question, [h.text for h in cand])
        top = [cand[r["index"]] for r in scored[:20]]
        for r, h in zip(scored[:20], top):
            h.score, h.rank = r["score"], 1 + scored.index(r)
        return top, []
    except Exception:
        return ret._make_hits(fused, 20, "decomp"), []


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    qa = {q["id"]: q for q in json.loads(
        (config.DATA_DIR / "evals" / "qa_set.json").read_text(encoding="utf-8"))}
    ret = HybridRetriever()
    t0 = time.time()
    agg = {c: {m: [] for m in ("r@10", "r@20", "hit@10")} for c in
           ("prod", "decomp-max", "decomp-quota", "decomp-perside")}
    for qid, sides in SIDES.items():
        q = qa[qid]
        positives = ret.labels_for_docs(q["source_doc_ids"])
        if not positives:
            print(f"{qid} 正例为空，跳过", flush=True)
            continue
        print(f"\n[{qid}] {q['question'][:50]}", flush=True)

        s2 = ret.expand(q["question"])
        print(f"   LLM 变体: {s2.queries}", flush=True)
        print(f"   LLM 关键词: {s2.keywords}", flush=True)

        for s in sides:
            vlab = [lab for lab, _, _ in ret.vector_search([s])]
            blab = [lab for lab, _, _ in ret.bm25_search([s], k=300)]
            vr = min((i + 1 for i, l in enumerate(vlab) if l in positives), default=None)
            br = min((i + 1 for i, l in enumerate(blab) if l in positives), default=None)
            print(f"   侧[{s[:24]}] 向量首正例#{vr}  BM25首正例#{br}", flush=True)

        for name in ("prod", "decomp-max", "decomp-quota", "decomp-perside"):
            t1 = time.time()
            if name == "prod":
                hits = ret.retrieve(q["question"], s2=s2, rerank=True, n=20)
                pool = ""
            else:
                hits, pool_info = decomp_retrieve(ret, q["question"], sides,
                                                  mode=name.split("-")[1],
                                                  positives=positives)
                pool = (" 池内正例#" + "/".join(str(p) if p else "-" for p in pool_info)
                        if name == "decomp-perside" else "")
            m = score(hits, positives)
            for k in ("r@10", "r@20", "hit@10"):
                agg[name][k].append(m[k])
            print(f"   {name:14s} r@10={m['r@10']:.2f} r@20={m['r@20']:.2f} "
                  f"hit@10={m['hit@10']:.0f} 首正例排名={m['top_rank']}{pool} "
                  f"({time.time()-t1:.0f}s)", flush=True)

    print("\n=== 合计（11 题）===", flush=True)
    for name, ms in agg.items():
        print(f"  {name:12s} " + " ".join(f"{k}={sum(v)/len(v):.2f}" for k, v in ms.items()),
              flush=True)
    print(f"总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
