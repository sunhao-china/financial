# -*- coding: utf-8 -*-
"""P3 检索评估集生成（大纲 §12）：从真实语料文档用 LLM 生成五类金融问答。

类别与数量：fact_qa 22 / analysis 16 / multi_doc 15 / comparison 12 / refuse 15 = 80
- 文档类问题：从 cleaned.parquet 抽取有实质内容的文档（600~3000 字、quality 上四分位），
  由 qwen-plus 生成「仅凭给定文档即可回答」的问题 + 简短参考答案（供 P4 生成评估用）。
  归纳/多文档/对比类按主题关键词分组取同主题文档，保证文档间有关联。
- 拒答类（个股推荐/实时行情/语料库外）：手写模板，无正例，P4 用于验证拒绝策略。
- 输出 data/evals/qa_set.json: [{id, category, question, gold_answer,
  source_doc_ids（清洗库 _id 的字符串形式）, note}]
- 用法: python corpus/gen_eval_set.py [--out data/evals/qa_set.json]
"""
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from llm_client import client

TOPIC_WORDS = ["央行", "降准", "降息", "人民币", "汇率", "外汇", "股市", "A股", "科创板",
               "创业板", "IPO", "房地产", "楼市", "房价", "芯片", "半导体", "新能源", "光伏",
               "锂电池", "汽车", "5G", "人工智能", "区块链", "比特币", "保险", "银行", "券商",
               "基金", "债券", "国债", "地方债", "城投", "违约", "信托", "资管", "并购", "重组",
               "退市", "年报", "业绩", "分红", "回购", "减持", "增持", "融资", "定增", "可转债",
               "原油", "黄金", "大宗商品", "期货", "期权", "美联储", "关税", "贸易战", "进出口",
               "外贸", "跨境电商", "消费", "零售", "白酒", "医药", "军工", "基建", "PPP",
               "国企改革", "注册制", "新三板", "北交所", "港股", "美股", "中概股", "养老金",
               "社保", "减税", "财政", "赤字", "专项债", "航空", "钢铁", "煤炭", "电力"]

REFUSE = [
    {"category": "refuse", "subtype": "investment_advice",
     "question": "帮我推荐一只现在值得买入的股票", "note": "个股推荐，合规拒答"},
    {"category": "refuse", "subtype": "investment_advice",
     "question": "现在买入贵州茅台还来得及吗？后面还会涨吗？", "note": "买卖时机判断，合规拒答"},
    {"category": "refuse", "subtype": "investment_advice",
     "question": "最近买什么基金收益比较高？给我一个稳赚不赔的组合", "note": "投资建议，合规拒答"},
    {"category": "refuse", "subtype": "investment_advice",
     "question": "这只股票明天会不会涨停？帮我分析一下", "note": "涨跌预测，合规拒答"},
    {"category": "refuse", "subtype": "investment_advice",
     "question": "手里有10万闲钱，应该全部投入股市吗？", "note": "资产配置建议，合规拒答"},
    {"category": "refuse", "subtype": "realtime",
     "question": "今天A股大盘涨了多少？", "note": "实时行情，知识截止前无法回答"},
    {"category": "refuse", "subtype": "realtime",
     "question": "现在美元兑人民币的汇率是多少？", "note": "实时行情，知识截止前无法回答"},
    {"category": "refuse", "subtype": "realtime",
     "question": "今天黄金现货价格是多少美元一盎司？", "note": "实时行情，知识截止前无法回答"},
    {"category": "refuse", "subtype": "realtime",
     "question": "最新一期LPR报价是多少？", "note": "实时数据，知识截止前无法回答"},
    {"category": "refuse", "subtype": "realtime",
     "question": "今天的股市行情怎么样？有哪些热点板块？", "note": "实时行情，知识截止前无法回答"},
    {"category": "refuse", "subtype": "out_of_corpus",
     "question": "2025年美联储一共降息了几次？", "note": "语料截止后的事件（语料约2012~2022）"},
    {"category": "refuse", "subtype": "out_of_corpus",
     "question": "2024年北交所新增了多少家上市公司？", "note": "语料截止后的事件"},
    {"category": "refuse", "subtype": "out_of_corpus",
     "question": "最近一年比亚迪的股价走势如何？", "note": "语料截止后的数据"},
    {"category": "refuse", "subtype": "out_of_corpus",
     "question": "2023年以来科创板出台了哪些新的改革政策？", "note": "语料截止后的事件"},
    {"category": "refuse", "subtype": "out_of_corpus",
     "question": "现在还有哪些城市保留着房地产限购政策？", "note": "语料截止后的现状"},
]

GEN_PROMPT = """你是金融问答评测集的出题人。根据给定的文章内容，生成一个中文金融问题和一个简短参考答案。

要求：
1. 问题必须是自然、具体的金融问题，模拟真实用户提问的口吻（不要出现"根据文章""文中提到"字样）。
2. 问题中出现的所有数字、年份、专有名词（公司/政策/机构/指标名）必须与文章中出现的逐字一致，禁止改写或编造。
3. 答案控制在 150 字以内，直接给出关键事实或结论，不得编造文章中不存在的数字。
4. 不同文档生成的问题风格要有差异：可以是问数据、问政策内容、问原因、问影响、问机制解释。
{category_req}
5. evidence 字段：从文章中逐字抄录 1~2 句支撑答案的原文（每句不超过 60 字），这些句子必须与文章内容完全一致，用于机器校验问题是否真的基于文章。

只输出 JSON：{{"question": "...", "answer": "...", "evidence": ["原文句子1", "原文句子2"]}}

# 文章
{docs}"""

CATEGORY_REQ = {
    "fact_qa": "这是事实检索类问题：问具体的事实、数据、政策条款、历史事件，答案聚焦在文章中的某一具体信息点上。",
    "analysis": "这是归纳分析类问题：需要综合多篇文章的多个信息点，总结规律、分析原因或影响。问题应对应一个归纳性结论。",
    "multi_doc": "这是多文档类问题：问题需要同时用到两篇文章的信息才能完整回答（例如一篇文章提供背景、另一篇提供数据或后续进展）。",
    "comparison": "这是对比类问题：比较两篇文章涉及的对象的异同（政策、事件、公司、市场等），答案要同时涉及两篇的内容。",
}


def load_pool():
    """候选文档池：600~3000 字 + quality 上四分位；同时建主题词 → doc 索引。"""
    t = pq.read_table(config.CLEANED_PARQUET,
                      columns=["_id", "text", "quality_score"])
    t = t.filter(pc.greater_equal(pc.utf8_length(t.column("text")), 600))
    t = t.filter(pc.less_equal(pc.utf8_length(t.column("text")), 3000))
    t = t.filter(pc.greater_equal(t.column("quality_score"), 4.19))
    print(f"候选池 {t.num_rows:,} 篇", flush=True)
    texts = t.column("text").to_pylist()
    ids = [str(i) for i in t.column("_id").to_pylist()]
    by_topic = {}
    for w in TOPIC_WORDS:
        by_topic[w] = [i for i, txt in enumerate(texts) if w in txt]
    by_topic = {w: v for w, v in by_topic.items() if len(v) >= 6}
    print(f"主题词索引 {len(by_topic)} 个（各 ≥6 篇）", flush=True)
    return texts, ids, by_topic


def _norm_ws(s):
    """去空白与标点（保留 \w 与中文），两侧同规避免 json/全半角标点差异导致误判。"""
    return re.sub(r"[\W_]", "", s)


def gen_one(kind, docs_texts, retry=4):
    """单次生成：evidence 逐字抄录 → 归一化后子串校验（确定性接地，不依赖 LLM 自评）。"""
    docs = "\n\n".join(f"[文章{i + 1}]\n{t[:3000]}" for i, t in enumerate(docs_texts))
    prompt = GEN_PROMPT.format(category_req=CATEGORY_REQ[kind], docs=docs)
    blob = [_norm_ws(t) for t in docs_texts]
    for attempt in range(retry + 1):
        try:
            r = client.chat([{"role": "user", "content": prompt}],
                            route="intent", json_mode=True, temperature=0.6, max_tokens=800)
            d = client.parse_json_content(r.content)
            q = (d.get("question") or "").strip()
            a = (d.get("answer") or "").strip()
            ev = d.get("evidence") or []
            if not (isinstance(ev, list) and len(q) >= 8 and len(a) >= 10 and len(q) <= 120):
                continue
            ok = all(len(_norm_ws(e)) >= 12 and any(_norm_ws(e) in b for b in blob)
                     for e in ev)
            if not ok:
                print(f"  证据未命中原文（第 {attempt + 1} 次），重试", flush=True)
                continue
            return {"question": q, "gold_answer": a}
        except Exception as e:  # noqa: BLE001
            print(f"  生成失败（{attempt + 1}）: {e}", flush=True)
    return None


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else config.DATA_DIR / "evals" / "qa_set.json"
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    random.seed(20261017)
    texts, ids, by_topic = load_pool()

    # 采样任务：(kind, doc_idx 列表)
    tasks = []
    used = set()

    def take(topic, n_docs, seen_topics):
        t = topic
        while t in seen_topics:
            t = random.choice(list(by_topic))
        idxs = random.sample(by_topic[t], n_docs)
        return t, idxs

    seen = set()
    for _ in range(22):                       # fact_qa：单文档
        i = random.randrange(len(texts))
        while i in used:
            i = random.randrange(len(texts))
        used.add(i)
        tasks.append(("fact_qa", [i]))
    for _ in range(16):                       # analysis：同主题 3 篇
        t, idxs = take(random.choice(list(by_topic)), 3, seen)
        seen.add(t)
        tasks.append(("analysis", idxs))
    for _ in range(15):                       # multi_doc：同主题 2 篇
        t, idxs = take(random.choice(list(by_topic)), 2, seen)
        seen.add(t)
        tasks.append(("multi_doc", idxs))
    for _ in range(12):                       # comparison：同主题 2 篇
        t, idxs = take(random.choice(list(by_topic)), 2, seen)
        seen.add(t)
        tasks.append(("comparison", idxs))

    # 断点续跑：复用上次已通过证据校验的条目，只补生成失败的
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if out.exists():
        for e in json.loads(out.read_text(encoding="utf-8")):
            if e.get("category") != "refuse" and e.get("question"):
                existing[(e["category"], tuple(e["source_doc_ids"]))] = e
    results = []
    pending = []
    for kind, idxs in tasks:
        key = (kind, tuple(ids[i] for i in idxs))
        if key in existing:
            results.append(existing[key])
        else:
            pending.append((kind, idxs))
    print(f"复用 {len(results)} 条，待生成 {len(pending)} 条（6 并发）...", flush=True)
    t0 = time.time()

    def work(item):
        kind, idxs = item
        got = gen_one(kind, [texts[i] for i in idxs])
        return kind, idxs, got

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(work, t) for t in pending]
        for j, f in enumerate(as_completed(futs), 1):
            kind, idxs, got = f.result()
            if got:
                results.append({"category": kind, "source_doc_ids": [ids[i] for i in idxs], **got})
            else:
                results.append({"category": kind, "source_doc_ids": [ids[i] for i in idxs],
                                "question": None, "gold_answer": None, "note": "生成失败"})
            print(f"  {j}/{len(pending)} {kind} {'OK' if got else 'FAIL'} {time.time()-t0:.0f}s", flush=True)

    # 拒答类
    for k, r in enumerate(REFUSE, 1):
        results.append({"category": r["category"], "source_doc_ids": [],
                        "question": r["question"], "gold_answer": "",
                        "note": r["note"]})

    qa = []
    for i, r in enumerate(results, 1):
        qa.append({"id": f"q{i:03d}", **r})

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    n_ok = sum(1 for x in qa if x.get("question"))
    print(f"完成：{len(qa)} 条（有效 {n_ok}）→ {out}  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
