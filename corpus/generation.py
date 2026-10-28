# -*- coding: utf-8 -*-
"""P4 生成链路（大纲 §11）：S6 上下文 → DeepSeek 生成 → 引用解析。

按意图路由（§3/§11）：fact_qa → deepseek-flash（快一倍）；analysis/comparison → deepseek-v4-pro（深推理）。
流式输出 reasoning/content 增量（reasoning_content 字段），P5 SSE 直接复用。
引用解析：答案中 [n] / [1][2] / [1,2] 编号 → 越界编号规则级抓出（验证关硬校验的输入）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import config
from llm_client import client
from prompts import remote
from prompts.generate import build_generate_messages
from schemas.models import S6Context, S6Reference

CITE_PAT = re.compile(r"\[(\d+(?:,\d+)*)\]")
DELTA_CB = Callable[[str], None]

# 生成 max_tokens（§3 实测坑：深推理模型 reasoning 会占预算，需覆盖推理+正文）
MAX_TOKENS = {"generate_fact": 4096, "generate_complex": 8192}


@dataclass
class Generated:
    answer: str = ""
    reasoning: str = ""
    citations: List[str] = field(default_factory=list)          # 引用到的 chunk_id
    cited_ref_nos: List[int] = field(default_factory=list)      # 出现过的合法编号
    invalid_ref_nos: List[int] = field(default_factory=list)    # 越界编号（规则硬校验用）
    model: str = ""
    finish_reason: str = ""
    cost_tokens: dict = field(default_factory=dict)             # prompt/completion/reasoning/total


def route_for_intent(intent: str) -> str:
    """§11 + P4 对照实验结论（2026-10-18）：默认全走 deepseek-flash。

    8 题深推理对照：flash judge 一致性 6/8 与 v4-pro 持平、平均 14.6s vs 68.7s
    （快 4.7 倍）、tokens 减半；flash 验证 pass 4/8 vs 6/8（数字溯源略弱，
    验证关 fail 时 agent 用 flash 重生成 1 次，仍 fail 走 VERIFY_FAIL 话术）。"""
    return "generate_fact"


def parse_citations(answer: str, references: List[S6Reference]):
    """解析答案引用编号 → (合法引用 chunk_id 列表, 越界编号列表)。"""
    by_no = {r.ref_no: r.chunk_id for r in references}
    cited, invalid = [], []
    seen = set()
    for m in CITE_PAT.finditer(answer):
        for num in m.group(1).split(","):
            n = int(num)
            if n in by_no:
                if n not in seen:
                    seen.add(n)
                    cited.append(by_no[n])
            elif n not in invalid:
                invalid.append(n)
    return cited, sorted(invalid)


def generate(context: S6Context, intent: str = "fact_qa", stream: bool = False,
             on_delta: Optional[DELTA_CB] = None,
             on_reasoning: Optional[DELTA_CB] = None,
             route: Optional[str] = None) -> Generated:
    """S6 → 生成。on_delta 流式回调收到正文增量（§14 SSE 的 delta 事件源）；
    on_reasoning 收到推理增量（SSE reasoning 事件源，P5）。
    route 缺省按意图路由（§11）；显式传入可覆盖（消融/降级链用）。"""
    msgs = build_generate_messages(context.question, context.references,
                                   context.history, intent)
    route = route or route_for_intent(intent)
    prompt_client = remote.get_prompt("generate")
    g = Generated(model=config.MODELS[route]["model"])
    if stream:
        for ev in client.chat_stream(msgs, route=route, prompt_client=prompt_client):
            if ev["type"] == "reasoning":
                g.reasoning += ev["text"]
                if on_reasoning:
                    on_reasoning(ev["text"])
            elif ev["type"] == "content":
                g.answer += ev["text"]
                if on_delta:
                    on_delta(ev["text"])
            elif ev["type"] == "usage":
                g.cost_tokens = ev["usage"].to_dict()
    else:
        r = client.chat(msgs, route=route, max_tokens=MAX_TOKENS[route],
                        prompt_client=prompt_client)
        g.answer = r.content
        g.model = r.model
        g.finish_reason = r.finish_reason
        g.cost_tokens = r.usage.to_dict()
        if not g.answer.strip():
            # §3 实测坑：推理模型 reasoning 耗尽 max_tokens → content 空。
            # 加倍预算重试一次，仍空则返回空答案由调用方降级（§13.2）。
            r = client.chat(msgs, route=route, max_tokens=MAX_TOKENS[route] * 2,
                            prompt_client=prompt_client)
            g.answer = r.content
            g.model = r.model
            g.finish_reason = r.finish_reason
            g.cost_tokens = r.usage.to_dict()
    g.citations, g.invalid_ref_nos = parse_citations(g.answer, context.references)
    g.cited_ref_nos = [r.ref_no for r in context.references
                       if r.chunk_id in set(g.citations)]
    return g
