# -*- coding: utf-8 -*-
"""P4 生成评测（大纲 §12）：全链路问答（检索+生成+验证关）+ LLM-as-judge。

评测对象：data/evals/qa_set.json 80 条（65 文档类 + 15 拒答类）。
指标：
  - 拒答准确率：refuse 类必须走拒答话术或「现有资料中未提及」诚实声明（不得给实质答案）
  - 可回答率：文档类中给出带引用实质答案的比例
  - 验证关分布：pass / downgrade / fail + 重生成改善率（fail → 复验通过）
  - 引用合法性：答案无越界引用编号的比例（规则硬校验）
  - 证据覆盖：答案引用的 chunk 至少 1 个来自 gold doc（source_doc_ids）的比例
  - 正确性（LLM-as-judge）：生成答案 vs gold_answer 关键信息一致性
  - Agent 专项：补检触发率、轮数分布
  - 成本/耗时：平均 token 与秒数，按环节拆分（S7 timings/cost）
断点续跑：结果逐条 append tasks/p4_eval.jsonl，重跑跳过已完成 id。
报告：data/reports/p4_eval.json + p4_eval.md。
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from corpus.agent import SearchAgent
from corpus.retrieval import HybridRetriever
from corpus.session_store import MemorySessionStore
from llm_client import client
from prompts import remote
from prompts.judge import build_judge_messages

TASK_FILE = Path(__file__).parent.parent / "tasks" / "p4_eval.jsonl"
REPORT_JSON = config.REPORT_DIR / "p4_eval.json"
REPORT_MD = config.REPORT_DIR / "p4_eval.md"
CITE_PAT = re.compile(r"\[(\d+(?:,\d+)*)\]")

REFUSAL_MARKERS = ("现有资料中未提及", "无法提供", "不构成", "不能提供", "未找到", "建议查阅",
                   "无法确认", "不足以确认")


def load_done() -> dict:
    done = {}
    if TASK_FILE.exists():
        for line in TASK_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["id"]] = rec
    return done


def check_invalid_cites(answer: str, max_ref: int) -> list:
    """答案中越界引用编号（规则级，验证关硬校验的同款逻辑）。"""
    return sorted({int(n) for m in CITE_PAT.finditer(answer)
                   for n in m.group(1).split(",") if int(n) > max_ref})


def refusal_verdict(item: dict, ans) -> tuple:
    """refuse 类判定：拒答话术 / 诚实声明 → ok；给出实质答案 → bad。"""
    if ans.fallback:
        return True, "话术拒答"
    a = ans.answer or ""
    for mk in REFUSAL_MARKERS:
        if mk in a:
            return True, f"诚实声明（{mk}）"
    return False, f"给出了实质答案：{a[:80]}"


def run_judge(item: dict, answer: str) -> dict:
    """LLM-as-judge：生成答案 vs gold_answer 关键信息一致性（qwen-plus）。"""
    prompt_client = remote.get_prompt("judge")
    for attempt in range(config.JSON_RETRY + 1):
        try:
            r = client.chat(build_judge_messages(item["question"], answer,
                                                 item.get("gold_answer", "")),
                            route="self_check", json_mode=True, max_tokens=800,
                            prompt_client=prompt_client)
            d = client.parse_json_content(r.content)
            return {"verdict": d.get("verdict", "?"),
                    "matched": d.get("matched", []),
                    "mismatched": d.get("mismatched", []),
                    "parse_ok": True, "cost": r.usage.total_tokens}
        except Exception as e:  # noqa: BLE001
            if attempt >= config.JSON_RETRY:
                return {"verdict": "error", "parse_ok": False, "err": str(e)[:200]}
    return {"verdict": "error", "parse_ok": False}


def build_record(item: dict, ans, snap, dt: float) -> dict:
    rec = {
        "id": item["id"], "category": item["category"],
        "question": item["question"],
        "intent": ans.intent, "rounds": ans.rounds,
        "fallback_kind": "refuse" if ans.intent in ("realtime", "investment_advice", "chitchat")
        else ("verify_fail" if ans.fallback and "确认" in ans.fallback
              else ("llm_down" if ans.fallback else "")),
        "has_answer": bool(ans.answer) and not ans.fallback,
        "answer": (ans.answer or ans.fallback)[:1200],
        "cost_tokens": ans.cost, "seconds": round(dt, 1),
    }
    if snap:
        rec["query_id"] = snap.query_id
        rec["status"] = snap.status
        rec["verification"] = snap.verification.model_dump() if snap.verification else None
        rec["timings_ms"] = snap.timings_ms
        rec["n_retrieval_rounds"] = len(snap.retrieval)
        rec["citations"] = snap.citations
        # 证据覆盖：引用的 chunk 是否命中 gold doc
        cited_docs = set()
        for h in ans.chunks:
            if h.id in set(snap.citations):
                cited_docs.add(h.doc_id)
        gold = set(item.get("source_doc_ids", []))
        rec["cites_gold_doc"] = bool(cited_docs & gold)
        rec["cited_docs"] = sorted(cited_docs)
        rec["gold_docs"] = sorted(gold)
    rec["invalid_cites"] = check_invalid_cites(
        ans.answer or ans.fallback or "", len(ans.context.get("references", [])))
    n_gen = sum(1 for s in ans.path if s["step"] == "generate")
    n_verify = sum(1 for s in ans.path if s["step"] == "verify")
    rec["n_generate_calls"] = n_gen
    rec["n_verify_calls"] = n_verify
    rec["regen_improved"] = (n_gen == 2 and n_verify == 2
                             and rec.get("verification") and rec["verification"]["verdict"]
                             in ("pass", "downgrade"))
    if item["category"] == "refuse":
        ok, why = refusal_verdict(item, ans)
        rec["refusal_ok"] = ok
        rec["refusal_why"] = why
    elif rec["has_answer"]:
        rec["judge"] = run_judge(item, ans.answer)
    return rec


def report(done: dict):
    items = sorted(done.values(), key=lambda x: x["id"])
    refuse = [r for r in items if r["category"] == "refuse"]
    docs = [r for r in items if r["category"] != "refuse"]

    def pct(n, d):
        return f"{n}/{d} ({100 * n / d:.1f}%)" if d else "-"

    # 拒答准确率
    refuse_ok = sum(1 for r in refuse if r.get("refusal_ok"))
    # 可回答率
    answered = [r for r in docs if r.get("has_answer")]
    # 验证关分布（按最终 verdict）
    verdicts = {}
    regen_improved = 0
    regen_total = 0
    for r in docs:
        v = (r.get("verification") or {}).get("verdict")
        if v:
            verdicts[v] = verdicts.get(v, 0) + 1
        if r.get("n_generate_calls", 0) >= 2:
            regen_total += 1
            if r.get("regen_improved"):
                regen_improved += 1
    # 引用合法性
    legal = sum(1 for r in answered if not r.get("invalid_cites"))
    # 证据覆盖（有引用的）
    with_cites = [r for r in answered if r.get("citations")]
    gold_hit = sum(1 for r in with_cites if r.get("cites_gold_doc"))
    # judge
    judged = [r for r in answered if r.get("judge", {}).get("parse_ok")]
    consistent = sum(1 for r in judged if r["judge"]["verdict"] == "consistent")
    judge_err = sum(1 for r in answered if not r.get("judge", {}).get("parse_ok"))
    # Agent 专项
    slow = sum(1 for r in docs if r.get("rounds", 0) > 0)
    # 成本
    tok = [r.get("cost_tokens", 0) for r in items]
    secs = [r.get("seconds", 0) for r in items]
    gen_ms = [r["timings_ms"].get("generate", 0) for r in docs if r.get("timings_ms")]
    ver_ms = [r["timings_ms"].get("verify", 0) for r in docs if r.get("timings_ms")]

    out = {
        "n_total": len(items), "n_doc": len(docs), "n_refuse": len(refuse),
        "refuse_accuracy": pct(refuse_ok, len(refuse)),
        "answer_rate": pct(len(answered), len(docs)),
        "verification": verdicts,
        "regen": {"total_fail_first": regen_total, "improved_after_regen": regen_improved},
        "cite_legality": pct(legal, len(answered)),
        "evidence_coverage": pct(gold_hit, len(with_cites)),
        "judge_consistent": pct(consistent, len(judged)),
        "judge_errors": judge_err,
        "agent_slow_path_rate": pct(slow, len(docs)),
        "cost": {"total_tokens": sum(tok), "avg_tokens": sum(tok) / len(items) if items else 0,
                 "total_seconds": round(sum(secs), 0),
                 "avg_seconds": round(sum(secs) / len(items), 1) if items else 0,
                 "avg_generate_ms": round(sum(gen_ms) / len(gen_ms)) if gen_ms else 0,
                 "avg_verify_ms": round(sum(ver_ms) / len(ver_ms)) if ver_ms else 0},
        "refuse_details": [{r["id"]: (r.get("refusal_ok"), r.get("refusal_why", ""))}
                           for r in refuse],
    }
    REPORT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))

    lines = ["# P4 生成与评测报告", "",
             f"- 评测集：{len(items)} 条（{len(docs)} 文档类 + {len(refuse)} 拒答类），全链路（检索+生成+验证关）",
             f"- 拒答准确率：**{out['refuse_accuracy']}**",
             f"- 可回答率（文档类给出实质答案）：**{out['answer_rate']}**",
             f"- 验证关分布：pass {verdicts.get('pass', 0)} / downgrade {verdicts.get('downgrade', 0)} / fail {verdicts.get('fail', 0)}",
             f"- 重生成改善：fail 首验 {regen_total} 条，复验通过 {regen_improved} 条",
             f"- 引用合法性（无越界编号）：**{out['cite_legality']}**",
             f"- 证据覆盖（引用命中 gold doc）：**{out['evidence_coverage']}**",
             f"- LLM-as-judge 一致性：**{out['judge_consistent']}**（judge 失败 {judge_err} 条）",
             f"- Agent 慢路径（补检）触发率：{out['agent_slow_path_rate']}",
             f"- 成本：总 {out['cost']['total_tokens']:,} tokens / {out['cost']['total_seconds']}s，"
             f"平均 {out['cost']['avg_tokens']:.0f} tokens / {out['cost']['avg_seconds']:.1f}s，"
             f"生成环节均值 {out['cost']['avg_generate_ms']}ms / 验证 {out['cost']['avg_verify_ms']}ms", "",
             "## 拒答类明细", ""]
    for r in refuse:
        lines.append(f"- {r['id']} {'✅' if r.get('refusal_ok') else '❌'} {r['question'][:50]} → {r.get('refusal_why', '')}")
    lines += ["", "## 文档类失败/降级明细", ""]
    bad = [r for r in docs if not r.get("has_answer") or r.get("judge", {}).get("verdict") == "inconsistent"]
    if bad:
        for r in bad:
            lines.append(f"- {r['id']} [{r['category']}] verdict={ (r.get('verification') or {}).get('verdict') } "
                         f"judge={r.get('judge', {}).get('verdict', '-')} 覆盖gold={'是' if r.get('cites_gold_doc') else '否'} "
                         f"→ {r['question'][:60]}")
            for mm in (r.get("judge", {}) or {}).get("mismatched", [])[:3]:
                lines.append(f"    - mismatched: {mm[:120]}")
    else:
        lines.append("（无）")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    qa = json.loads((config.DATA_DIR / "evals" / "qa_set.json").read_text(encoding="utf-8"))
    done = load_done()
    todo = [it for it in qa if it["id"] not in done]
    print(f"评测集 {len(qa)} 条，已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    if not todo:
        print("全部完成，直接出报告。", flush=True)
    else:
        t0 = time.time()
        ret = HybridRetriever()
        store = MemorySessionStore()
        agent = SearchAgent(ret, store=store)
        print(f"模型加载 {time.time()-t0:.0f}s", flush=True)
        TASK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TASK_FILE, "a", encoding="utf-8") as f:
            for i, item in enumerate(todo):
                t1 = time.time()
                try:
                    ans = agent.run(item["question"], session_id="eval_p4", verify=True)
                    snap = store.get(ans.query_id) if ans.query_id else None
                    rec = build_record(item, ans, snap, time.time() - t1)
                except Exception as e:  # noqa: BLE001
                    rec = {"id": item["id"], "category": item["category"],
                           "question": item["question"], "error": str(e)[:300]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[item["id"]] = rec
                tag = ""
                if item["category"] == "refuse":
                    tag = "拒答" + ("✅" if rec.get("refusal_ok") else "❌")
                elif rec.get("has_answer"):
                    tag = f"答 judge={rec.get('judge', {}).get('verdict', '-')} "
                    tag += f"verdict={(rec.get('verification') or {}).get('verdict', '-')}"
                else:
                    tag = f"降级({rec.get('fallback_kind', '?')})"
                print(f"[{i+1}/{len(todo)}] {item['id']} {tag} "
                      f"{rec.get('cost_tokens', 0)}tok {rec.get('seconds', 0)}s", flush=True)
        print(f"\n全部跑完 {time.time()-t0:.0f}s，快照已存会话记忆。", flush=True)

    report(done)


if __name__ == "__main__":
    main()
