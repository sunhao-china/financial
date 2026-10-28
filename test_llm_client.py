# -*- coding: utf-8 -*-
"""llm_client.py 冒烟测试：chat / chat_stream / embed / rerank 四个接口。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from llm_client import LLMClient


def main():
    c = LLMClient()

    # 1. 非流式对话（事实类路由 → deepseek-flash）
    r = c.chat(
        [{"role": "user", "content": "用一句话回答：什么是综合保税区？"}],
        route="generate_fact",
        max_tokens=1500,
    )
    print(f"[1 chat] {r.model}: {r.content[:60]!r}")
    print(f"    usage: {r.usage.to_dict()}")

    # 2. 流式对话
    print("[2 stream]", end=" ")
    parts = []
    for ev in c.chat_stream(
        [{"role": "user", "content": "用一句话回答：什么是GDP？"}],
        route="generate_fact",
        max_tokens=1500,
    ):
        if ev["type"] == "reasoning":
            parts.append(f"(推理{len(ev['text'])}字)")
        elif ev["type"] == "content":
            parts.append(ev["text"])
        elif ev["type"] == "usage":
            print("".join(parts)[:60], "| usage:", ev["usage"].to_dict())

    # 3. embedding
    vecs = c.embed(["综合保税区进出口值", "今天天气不错"])
    print(f"[3 embed] {len(vecs)} 条 × {len(vecs[0])} 维")

    # 4. rerank
    hits = c.rerank("综合保税区进出口值", ["全国综合保税区实现进出口值5.9万亿元", "今天天气晴朗适合散步"])
    print(f"[4 rerank] top1: {hits[0]}")


if __name__ == "__main__":
    main()
