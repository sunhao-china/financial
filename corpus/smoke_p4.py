# -*- coding: utf-8 -*-
"""P4 Agent 冒烟测试：真实生成链路 + 验证关 + 拒答路径 + S7 会话记忆。

覆盖场景：
  1 fact_qa 快路径（真生成 + 验证关）
  2 guard 实时拒答（今天大盘）
  3 投资建议拒答（买入个股）
  4 时效超界（2025 年问题 → 诚实声明「现有资料中未提及」）
  5 S7 快照写入会话记忆并落盘
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from corpus.retrieval import HybridRetriever
from corpus.agent import SearchAgent
from corpus.session_store import MemorySessionStore

QUESTIONS = [
    "截至2020年末，邮储银行金融精准扶贫贷款余额是多少？",            # fact_qa：快路径+生成+验证
    "今天A股大盘涨了多少？",                                        # guard 实时拒答
    "现在买入贵州茅台还来得及吗？",                                  # 投资建议拒答
    "2025年美联储一共降息了几次？",                                  # 时效超界：诚实声明
]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    ret = HybridRetriever()
    store = MemorySessionStore()
    agent = SearchAgent(ret, store=store)
    print(f"模型加载 {time.time()-t0:.0f}s", flush=True)
    for q in QUESTIONS:
        t1 = time.time()
        a = agent.run(q, session_id="smoke_p4")
        print(f"\n{'='*70}\nQ: {q}\n"
              f"intent={a.intent} rounds={a.rounds} cost={a.cost} tokens "
              f"fallback={'是' if a.fallback else '否'} 耗时={time.time()-t1:.1f}s "
              f"query_id={a.query_id}", flush=True)
        print("路径:", flush=True)
        for s in a.path:
            print(f"  {s}", flush=True)
        out = a.answer or a.fallback
        print(f"回复:\n{out[:500]}", flush=True)
        if a.query_id:
            snap = store.get(a.query_id)
            if snap:
                print(f"S7: status={snap.status} rounds={len(snap.retrieval)} "
                      f"verification={snap.verification.verdict if snap.verification else '-'} "
                      f"citations={len(snap.citations)} timings={snap.timings_ms}", flush=True)
    dump = Path("tasks") / "smoke_p4_s7.jsonl"
    dump.parent.mkdir(parents=True, exist_ok=True)
    store.dump(dump)
    print(f"\n冒烟完成 {time.time()-t0:.0f}s；S7 快照 {len(store)} 条 → {dump}", flush=True)


if __name__ == "__main__":
    main()
