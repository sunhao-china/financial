# -*- coding: utf-8 -*-
"""LLM 结构化输出的解析与兜底（大纲 §9.1/§13.2）。

- extract_json：从 LLM 回复提取 JSON 对象（容忍代码围栏、前后杂文）
- structured_call：一次调用 + pydantic 校验；解析/校验失败重试 1 次
  （把上次输出原样附回对话要求重发，§13.2），仍失败返回 fallback
- fallback_s2/s3/s4：规则兜底（§13.2：意图默认 fact_qa、扩展用原问题；
  自检兜底视为充分走快路径；校验兜底视为 fail 走降级话术，pre_reply 不放行）
"""
import json
import re
from typing import Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

import config
from llm_client import ChatResult, client
from schemas.models import S2IntentExpand, S3SelfCheck, S4Verify, SV

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(content: str) -> dict:
    """提取回复中最外层 JSON 对象。"""
    text = content.strip()
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("回复中未找到 JSON 对象")
    return json.loads(text[start:end + 1])


def structured_call(
    messages: list,
    *,
    route: str,
    model_cls: Type[T],
    fallback: Optional[T] = None,
    max_tokens: int = 2048,
    temperature: float = 0.1,
    timeout: float = 60,
) -> Tuple[T, Optional[ChatResult]]:
    """结构化调用（json_mode），解析/校验失败重试 1 次（§13.2），仍失败返回 fallback。

    返回 (模型实例, 最后一次调用结果)；fallback 为 None 且重试耗尽时抛出原异常。
    """
    last_result, last_err = None, None
    msgs = list(messages)
    for _ in range(config.JSON_RETRY + 1):
        last_result = client.chat(msgs, route=route, json_mode=True,
                                  max_tokens=max_tokens, temperature=temperature, timeout=timeout)
        try:
            return model_cls.model_validate(extract_json(last_result.content)), last_result
        except ValueError as e:
            last_err = e
            msgs = msgs + [
                {"role": "assistant", "content": last_result.content},
                {"role": "user", "content": "你的输出不是合法 JSON 或不符合结构要求。"
                                           "请严格按给定结构只输出一个 JSON 对象，不要任何其他内容。"},
            ]
    if fallback is not None:
        return fallback, last_result
    raise last_err


def fallback_s2(question: str) -> S2IntentExpand:
    """S2 兜底（§13.2）：意图默认 fact_qa，扩展用原问题。"""
    return S2IntentExpand(schema_version=SV, intent="fact_qa", confidence=0.0,
                          entities=[], queries=[question], keywords=[])


def fallback_s3() -> S3SelfCheck:
    """S3 兜底：自检解析失败视为充分，直接用现有结果生成（避免无意义补检）。"""
    return S3SelfCheck(schema_version=SV, sufficient=True, missing=[], followup_queries=[])


def fallback_s4() -> S4Verify:
    """S4 兜底：校验解析失败视为 fail，pre_reply 钩子不放行，走降级话术（§6.3）。"""
    return S4Verify(schema_version=SV, verdict="fail",
                    unverified_claims=["校验环节异常，无法确认数字溯源"],
                    single_source_items=[], notes="解析失败兜底")
