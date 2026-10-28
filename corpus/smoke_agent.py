# -*- coding: utf-8 -*-
"""P3 Agent 冒烟测试：3 个问题验证快路径/慢路径/钩子（预算、轮数上限、查询去重）。"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from corpus.retrieval import HybridRetriever
from corpus.agent import SearchAgent

QUESTIONS = [
    "截至2020年末，邮储银行金融精准扶贫贷款余额是多少？",          # fact_qa：快路径
    "为什么奇富科技在2023年5月8日发布一季报后股价一度跌至50.65港元/股新低，但6月后又一路飙涨？",  # multi_doc：可能触发慢路径
    "今天A股大盘涨了多少？",                                        # realtime：queries 空，验证降级路径
]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t0 = time.time()
    ret = HybridRetriever()
    agent = SearchAgent(ret)
    print(f"模型加载 {time.time()-t0:.0f}s", flush=True)
    for q in QUESTIONS:
        t1 = time.time()
        a = agent.run(q)
        print(f"\n{'='*70}\nQ: {a.question}\n"
              f"intent={a.intent} rounds={a.rounds} cost={a.cost} tokens "
              f"fallback={'是' if a.fallback else '否'} stub={a.generated_by_stub} "
              f"耗时={time.time()-t1:.1f}s", flush=True)
        print(f"chunks: {len(a.chunks)} 条，top1 来源={a.chunks[0].source if a.chunks else '-'} "
              f"score={a.chunks[0].score:.3f} {a.chunks[0].text[:50]!r}" if a.chunks else "无片段")
        print("路径:", flush=True)
        for s in a.path:
            print(f"  {s}", flush=True)
        if a.fallback:
            print(f"降级话术: {a.fallback[:120]}", flush=True)
        else:
            print(f"回答(stub): {a.answer[:150]}", flush=True)
    print(f"\n冒烟测试完成 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
