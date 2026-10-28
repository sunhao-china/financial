# -*- coding: utf-8 -*-
"""Langfuse 数据集注册 + 评测轮次（2026-10-22，ECS 上运行）。

把 P3/P4 评估集（data/evals/qa_set.json，79 条）注册为 Langfuse 数据集，
并支持对数据集跑「轮次」评测（真实 RAG 管线 + 检索命中 + LLM-as-judge 打分），
每轮在 Langfuse UI 的 Datasets / Experiments 页可见、可跨轮对比。

模式：
  .venv/bin/python tools/langfuse_eval.py --register            注册/同步数据集（幂等）
  .venv/bin/python tools/langfuse_eval.py --run <run_name>      跑一轮（默认全量 79 题）
      [--limit N] [--shuffle] [--category X] [--no-judge] [--concurrency N]

设计要点：
- 任务在进程内跑真实管线（HybridRetriever + SearchAgent，P4 eval_generate 同款构造），
  不经过 HTTP 服务 → 不污染生产 MySQL 会话表、不吃登录验证码；
  MemorySessionStore 使 S7 快照不落 Redis。
- agent.run 自带的 Langfuse trace 也会进 UI（session_id=该轮名、user=langfuse-eval，
  与线上真实流量区分）——每题的完整 RAG 瀑布图（意图/检索/生成/验证各 span）可查。
- run_experiment 一次调用 = UI 里的一轮（dataset run）；每题自动建 trace 挂到该轮，
  输出 + 打分（retrieval_hit / retrieval_mrr / answer_judge / refuse_ok + 轮级聚合）。
- 打分口径与 P4 一致：answer_judge 复用 prompts/judge.py + run_judge（qwen-plus，
  verdict=consistent/inconsistent）；refuse 类用话术/诚实声明规则判定。
- 每轮 run_name 建议带配置语义（如 prod-20261022、model-flash），UI 按轮名对比。
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402  （config.py 自带 /data/financial/.env 加载）

from corpus.agent import SearchAgent  # noqa: E402
from corpus.eval_generate import REFUSAL_MARKERS, run_judge  # noqa: E402
from corpus.retrieval import HybridRetriever  # noqa: E402
from corpus.session_store import MemorySessionStore  # noqa: E402

DATASET_NAME = "financial-rag-eval"
QA_SET = config.DATA_DIR / "evals" / "qa_set.json"
MANIFEST = config.ROOT / "tasks" / "lf_eval_registry.json"
QA_URL = "http://8.137.174.74:8001"  # 仅打印提示用（服务地址见 §14）


def lf_client():
    """Langfuse 客户端（env 来自 .env；SDK 缺失/未启用时直接报错——本工具必须开启）。"""
    if not (config.LANGFUSE_ENABLED and config.LANGFUSE_PUBLIC_KEY):
        raise SystemExit("LANGFUSE_ENABLED/PUBLIC_KEY 未配置（查 /data/financial/.env）")
    from langfuse import Langfuse  # noqa: PLC0415 — 可选依赖
    return Langfuse(public_key=config.LANGFUSE_PUBLIC_KEY,
                    secret_key=config.LANGFUSE_SECRET_KEY,
                    host=config.LANGFUSE_HOST)


# ---------- 注册 ----------

def load_manifest() -> set:
    if MANIFEST.exists():
        return set(json.loads(MANIFEST.read_text(encoding="utf-8")).get("item_ids", []))
    return set()


def save_manifest(ids: set):
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(
        {"dataset": DATASET_NAME, "item_ids": sorted(ids)}, ensure_ascii=False, indent=2),
        encoding="utf-8")


def register() -> int:
    lf = lf_client()
    qa = json.loads(QA_SET.read_text(encoding="utf-8"))
    print(f"评估集: {QA_SET} 共 {len(qa)} 条", flush=True)

    try:
        ds = lf.get_dataset(DATASET_NAME)
        server_ids = {it.id for it in ds.items}
    except Exception:  # noqa: BLE001 — 数据集不存在 → 新建
        lf.create_dataset(
            name=DATASET_NAME,
            description="金融 RAG 端到端评估集（P3/P4，qa_set.json）：79 条五类意图 "
                        "（fact_qa/analysis/multi_doc/comparison/refuse），"
                        "gold_answer 为参考答案、source_doc_ids 为检索金标准。")
        server_ids = set()

    known = load_manifest() | server_ids
    todo = [q for q in qa if q["id"] not in known]
    print(f"已注册 {len(known)} 条，待注册 {len(todo)} 条", flush=True)
    for q in todo:
        try:
            lf.create_dataset_item(
                dataset_name=DATASET_NAME,
                id=q["id"],
                input=q["question"],
                expected_output=q.get("gold_answer") or "",
                metadata={"qid": q["id"], "category": q["category"],
                          "source_doc_ids": q.get("source_doc_ids") or []})
            known.add(q["id"])
            save_manifest(known)   # 逐条落盘：中断可续
            print(f"  + {q['id']} [{q['category']}] {q['question'][:30]}…", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {q['id']} 注册失败: {e}", flush=True)
    save_manifest(known)
    print(f"完成：数据集 {DATASET_NAME} 共 {len(known)} 条 "
          f"→ Langfuse UI Datasets 页查看", flush=True)
    return 0


# ---------- 跑轮次 ----------

def retrieval_scores(retrieved, gold_ids):
    """top-20 检索结果 vs 金标准 doc 列表 → (hit 0/1, MRR)；无金标准返回 (None, None)。"""
    if not gold_ids:
        return None, None
    gold = {str(g) for g in gold_ids}
    for i, did in enumerate(retrieved[:20], 1):
        if str(did) in gold:
            return 1, round(1.0 / i, 4)
    return 0, 0.0


def build_task(ret, run_name: str):
    """每题的运行函数（run_experiment 并发调用；进程内跑真实管线）。"""
    def task(*, item, **kw):
        qid = item.id
        q = item.input
        gold = (item.expected_output or "").strip()
        meta = item.metadata or {}
        print(f"[{qid}] {q[:40]}…", flush=True)
        try:
            t0 = time.time()
            agent = SearchAgent(ret, store=MemorySessionStore())
            ans = agent.run(q, session_id=f"lf-eval-{run_name}", user="langfuse-eval")
            dt = time.time() - t0
            answer = (ans.answer or ans.fallback or "").strip()
            retrieved = [h.doc_id for h in (ans.chunks or [])]
            hit, mrr = retrieval_scores(retrieved, meta.get("source_doc_ids") or [])
            print(f"  [{qid}] intent={ans.intent} 用时{dt:.0f}s cost={ans.cost} "
                  f"hit={hit} mrr={mrr}", flush=True)
            return {
                "qid": qid,
                "question": q,
                "gold_answer": gold[:300],
                "answer": answer[:1200],
                "intent": ans.intent,
                "refused": bool(ans.fallback),
                "retrieved_top10": [str(d) for d in retrieved[:10]],
                "hit_gold": hit,
                "mrr": mrr,
                "cost_tokens": ans.cost,
                "latency_s": round(dt, 1),
                "timings_ms": ans.timings,
                "verification": (ans.verification or {}).get("verdict"),
                "query_id": ans.query_id,
            }
        except Exception as e:  # noqa: BLE001 — 单题失败不中断整轮（UI 该 item 显示 error）
            print(f"  [{qid}] ✗ {type(e).__name__}: {e}", flush=True)
            return {"qid": qid, "question": q, "gold_answer": gold[:300],
                    "error": f"{type(e).__name__}: {e}"[:300]}
    return task


def ev_retrieval(*, input, output, expected_output, metadata, **kw):
    """检索打分：hit（金标准 doc 是否进 top-20）+ MRR。refuse 类无金标准 → 不记。"""
    if output.get("hit_gold") is None:
        return []
    return [
        {"name": "retrieval_hit", "value": float(output["hit_gold"])},
        {"name": "retrieval_mrr", "value": float(output["mrr"] or 0.0)},
    ]


def ev_answer(*, input, output, expected_output, metadata, **kw):
    """答案打分：refuse 类走话术/诚实声明规则（P4 同口径）；其余 LLM-as-judge。"""
    cat = (metadata or {}).get("category")
    answer = output.get("answer") or ""
    if cat == "refuse":
        if output.get("refused"):
            return [{"name": "refuse_ok", "value": 1.0, "comment": "话术拒答"}]
        for mk in REFUSAL_MARKERS:
            if mk in answer:
                return [{"name": "refuse_ok", "value": 1.0,
                         "comment": f"诚实声明（{mk}）"}]
        return [{"name": "refuse_ok", "value": 0.0,
                 "comment": f"给出实质答案：{answer[:80]}"}]
    if output.get("error"):
        return []
    d = run_judge({"question": input, "gold_answer": expected_output or ""}, answer)
    if not d.get("parse_ok"):
        return [{"name": "answer_judge", "value": 0.0,
                 "comment": f"judge 失败: {d.get('err', '')[:200]}"}]
    ok = d["verdict"] == "consistent"
    comment = "、".join(d.get("matched", [])[:2])[:300] or "关键信息不一致"
    return [{"name": "answer_judge", "value": 1.0 if ok else 0.0, "comment": comment}]


def _mean(vals):
    return round(sum(vals) / len(vals), 4) if vals else None


def ev_run_summary(*, item_results, **kw):
    """轮级聚合（UI 轮次对比用的列）：命中率/MRR/一致性率/拒答正确率/平均耗时成本。"""
    outs = [r.output for r in item_results]
    n = len(outs)
    hit = [o["hit_gold"] for o in outs if o.get("hit_gold") is not None]
    mrr = [o["mrr"] for o in outs if o.get("mrr") is not None]
    ans = [o for o in outs if (o.get("answer") or "").strip()]

    def _sum(name):
        vals = [e.value for r in item_results for e in r.evaluations if e.name == name]
        return len(vals), _mean(vals)

    nj, jr = _sum("answer_judge")
    nf, fr = _sum("refuse_ok")
    evs = [
        {"name": "n_items", "value": float(n)},
        {"name": "answer_rate", "value": round(len(ans) / n, 4) if n else 0.0},
        {"name": "retrieval_hit_rate", "value": _mean(hit) or 0.0,
         "comment": f"{sum(hit)}/{len(hit)}"},
        {"name": "retrieval_mrr_avg", "value": _mean(mrr) or 0.0},
        {"name": "answer_judge_rate", "value": jr or 0.0,
         "comment": f"{int(sum(e.value for r in item_results for e in r.evaluations if e.name == 'answer_judge'))}/{nj}"},
        {"name": "refuse_ok_rate", "value": fr or 0.0,
         "comment": f"{int(sum(e.value for r in item_results for e in r.evaluations if e.name == 'refuse_ok'))}/{nf}"},
        {"name": "avg_latency_s", "value": _mean([o.get("latency_s") or 0 for o in outs]) or 0.0},
        {"name": "avg_cost_tokens", "value": float(_mean([o.get("cost_tokens") or 0 for o in outs]) or 0.0)},
    ]
    return evs


def print_item_table(result) -> None:
    print("\n—— 逐题结果 ——")
    for r in result.item_results:
        o = r.output or {}
        evs = {e.name: e.value for e in r.evaluations}
        line = f"{o.get('qid', r.item.id):>6} {o.get('intent', '-'):<17} " \
               f"hit={evs.get('retrieval_hit', '-')} mrr={evs.get('retrieval_mrr', '-')} " \
               f"judge={evs.get('answer_judge', '-')} refuse={evs.get('refuse_ok', '-')} " \
               f"{o.get('latency_s', '-')}s {o.get('cost_tokens', '-')}tok"
        if o.get("error"):
            line += f"  ✗ {o['error'][:60]}"
        print(line)
    if getattr(result, "run_evaluations", None):
        print("\n—— 轮级聚合 ——")
        for e in result.run_evaluations:
            print(f"  {e.name} = {e.value}"
                  + (f"（{e.comment}）" if getattr(e, "comment", None) else ""))
    print(f"\n共 {len(result.item_results)} 题。数据集轮次页: {result.dataset_run_url}")


def run_round(args) -> int:
    lf = lf_client()
    ds = lf.get_dataset(DATASET_NAME)
    items = list(ds.items)
    print(f"数据集 {DATASET_NAME}: {len(items)} 条", flush=True)
    if args.category:
        items = [it for it in items
                 if (it.metadata or {}).get("category") == args.category]
        print(f"按类别过滤 [{args.category}]: {len(items)} 条", flush=True)
    if args.shuffle:
        random.Random(20261022).shuffle(items)
    if args.limit:
        items = items[: args.limit]
        print(f"截取前 {len(items)} 条（{'乱序抽样' if args.shuffle else '原序'}）", flush=True)

    print("构建检索管线（HybridRetriever：BM25 + Qdrant + bge-m3，约半分钟）…", flush=True)
    ret = HybridRetriever()

    evaluators = [ev_retrieval]
    if not args.no_judge:
        evaluators.append(ev_answer)
    print(f"开始轮次 [{args.run_name}]：{len(items)} 题，并发 {args.concurrency}，"
          f"judge={'开' if not args.no_judge else '关'}，预计 20~60 分钟（全量）\n", flush=True)
    result = lf.run_experiment(
        name=DATASET_NAME,
        run_name=args.run_name,
        description="金融 RAG 端到端评测轮次（进程内真实管线，P4 同款构造与打分口径）",
        data=items,
        task=build_task(ret, args.run_name),
        evaluators=evaluators,
        run_evaluators=[ev_run_summary],
        max_concurrency=args.concurrency,
        metadata={"round": args.run_name, "dataset": DATASET_NAME,
                  "pipeline": "hybrid_rerank_max 生产配置",
                  "source": "tools/langfuse_eval.py"},
    )
    print_item_table(result)
    return 0


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Langfuse 数据集注册 + 评测轮次")
    ap.add_argument("--register", action="store_true", help="注册/同步数据集（幂等）")
    ap.add_argument("--run", dest="run_name", metavar="RUN_NAME",
                    help="跑一轮评测（run 名，UI 按此对比）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（配合 --shuffle 抽样）")
    ap.add_argument("--shuffle", action="store_true", help="乱序（固定种子，可复现）")
    ap.add_argument("--category", default="", help="只跑某类：fact_qa/analysis/multi_doc/comparison/refuse")
    ap.add_argument("--no-judge", action="store_true", help="跳过 LLM-as-judge（省时间省 token）")
    ap.add_argument("--concurrency", type=int, default=3, help="并发数（默认 3，服务信号量同款上限 4）")
    args = ap.parse_args()
    if args.register:
        return register()
    if args.run_name:
        return run_round(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
