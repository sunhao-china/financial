# -*- coding: utf-8 -*-
"""P4 生成路由对照实验（§3「评测后可配置翻转」）：deepseek-v4-pro vs deepseek-flash。

对深推理类题目（analysis/comparison），复用同一检索上下文（S6），分别用
v4-pro（当前默认）与 flash 生成，对比：
  - 生成耗时（秒）
  - 答案质量：LLM-as-judge 一致性（vs gold_answer）
  - 验证关 verdict（数字溯源能力）
结论用途：flash 质量持平/接近 → analysis/comparison 默认路由切 flash（延迟降 3~5 倍）。

用法: python corpus/bench_generate_route.py [--n 8] [--only analysis|comparison|multi_doc]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from corpus import generation
from corpus.retrieval import HybridRetriever, S2
from corpus.agent import AgentContext, SearchAgent
from corpus.verify_gate import verify_answer
from llm_client import client
from prompts.judge import build_judge_messages


def pick_questions(n, only):
    qa = json.loads((config.DATA_DIR / "evals" / "qa_set.json").read_text(encoding="utf-8"))
    cats = {"analysis", "comparison", "multi_doc"} if not only else {only}
    return [it for it in qa if it["category"] in cats][:n]


def run_one(agent, item, route):
    """检索到 S6（与 agent.run 同链路：expand → retrieve），然后指定 route 生成 + 验证 + judge。"""
    ctx = AgentContext(question=item["question"])
    agent._expand(ctx)
    if ctx.expanded.intent not in ("fact_qa", "analysis", "comparison", "multi_doc"):
        return {"id": item["id"], "route": route, "error": f"intent={ctx.expanded.intent}"}
    agent._retrieve(ctx, ctx.expanded)
    s6 = agent._assemble(ctx)
    t0 = time.time()
    gen = generation.generate(s6, intent=item["category"], route=route)
    dt = time.time() - t0
    vr = verify_answer(item["question"], gen.answer, s6.references) if gen.answer.strip() else None
    judge = None
    if gen.answer.strip():
        try:
            r = client.chat(build_judge_messages(item["question"], gen.answer,
                                                 item.get("gold_answer", "")),
                            route="self_check", json_mode=True, max_tokens=800)
            judge = client.parse_json_content(r.content).get("verdict")
        except Exception as e:  # noqa: BLE001
            judge = f"error:{e}"
    return {"id": item["id"], "category": item["category"], "route": route,
            "seconds": round(dt, 1), "tokens": gen.cost_tokens.get("total_tokens", 0),
            "answer_len": len(gen.answer), "verify": vr.verdict if vr else "empty",
            "judge": judge, "answer": gen.answer[:400]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    items = pick_questions(args.n, args.only)
    print(f"对照 {len(items)} 题 × 2 路由（v4-pro vs flash）", flush=True)
    ret = HybridRetriever()
    agent = SearchAgent(ret)

    rows = []
    for it in items:
        for route in ("generate_complex", "generate_fact"):
            try:
                rec = run_one(agent, it, route)
            except Exception as e:  # noqa: BLE001
                rec = {"id": it["id"], "route": route, "error": str(e)[:150]}
            rows.append(rec)
            print(f"  {rec.get('id')} {route} {rec.get('seconds', '?')}s "
                  f"verify={rec.get('verify', '-')} judge={rec.get('judge', '-')}", flush=True)

    out = config.REPORT_DIR / "bench_route.json"
    out.write_text(json.dumps({"items": items, "rows": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    # 汇总
    by = {}
    for r in rows:
        k = (r.get("route") or "?")
        d = by.setdefault(k, {"n": 0, "sec": 0.0, "consistent": 0, "judged": 0,
                              "pass": 0, "empty": 0})
        d["n"] += 1
        d["sec"] += r.get("seconds", 0)
        if r.get("judge") == "consistent":
            d["consistent"] += 1
        if r.get("judge") in ("consistent", "inconsistent"):
            d["judged"] += 1
        if r.get("verify") == "pass":
            d["pass"] += 1
        if r.get("verify") == "empty":
            d["empty"] += 1
    print("\n汇总：")
    for k, d in by.items():
        print(f"  {k}: 平均 {d['sec']/d['n']:.1f}s | judge 一致 {d['consistent']}/{d['judged']} "
              f"| 验证 pass {d['pass']}/{d['n']} | 空答案 {d['empty']}")


if __name__ == "__main__":
    main()
