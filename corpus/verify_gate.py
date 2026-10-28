# -*- coding: utf-8 -*-
"""P4 来源验证关（大纲 §6.3）：生成答案后、出站回复前的强制闸门。

三道检查（qwen-plus 一次调用，S4 结构）：
  1 数字/日期溯源：答案中每个金额/比例/年份/日期必须能在资料原文定位
  2 多来源一致性：关键结论至少 2 个不同 doc_id 支撑，单源须标注
  3 引用编号核对：答案 [n] 必须对应存在的资料编号
处理策略（§6.3）：pass → 正常回复；downgrade → 附可靠性说明后回复；
fail → 重生成 1 次 → 仍失败 → VERIFY_FAIL 降级话术 + 相关片段。

硬校验前置（零成本规则）：越界引用编号先抓出，直接判 fail，不浪费一次 LLM 调用。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import config
from llm_client import client
from prompts import remote
from prompts.verify import build_verify_messages
from schemas.models import S4Verify, S6Reference

CITE_PAT = re.compile(r"\[(\d+(?:,\d+)*)\]")
DOWNGRADE_NOTE = ("\n\n（可靠性提示：本回答部分结论仅来自单一资料来源，"
                  "可靠性有限，建议结合其他渠道核实。）")


@dataclass
class VerifyResult:
    verdict: str = "pass"                       # pass / downgrade / fail
    unverified_claims: List[str] = field(default_factory=list)
    single_source_items: List[str] = field(default_factory=list)
    invalid_ref_nos: List[int] = field(default_factory=list)   # 硬校验抓出的越界编号
    notes: str = ""
    model: str = ""
    cost_tokens: dict = field(default_factory=dict)
    parse_ok: bool = True


def verify_answer(question: str, answer: str, references: List[S6Reference]) -> VerifyResult:
    """S4 校验。返回 verdict ∈ {pass, downgrade, fail}。"""
    max_no = max((r.ref_no for r in references), default=0)
    hard_invalid = sorted({int(n) for m in CITE_PAT.finditer(answer)
                           for n in m.group(1).split(",") if int(n) > max_no})
    if hard_invalid:
        return VerifyResult(verdict="fail", invalid_ref_nos=hard_invalid,
                            notes=f"引用编号越界 {hard_invalid}（规则硬校验，未调用 LLM）")
    d = None
    prompt_client = remote.get_prompt("verify")
    for attempt in range(config.JSON_RETRY + 1):
        try:
            r = client.chat(build_verify_messages(question, answer, references),
                            route="verify", json_mode=True, max_tokens=1200,
                            prompt_client=prompt_client)
            d = client.parse_json_content(r.content)
            s4 = S4Verify.model_validate(d)
            return VerifyResult(verdict=s4.verdict,
                                unverified_claims=s4.unverified_claims,
                                single_source_items=s4.single_source_items,
                                notes=s4.notes, model=r.model,
                                cost_tokens=r.usage.to_dict(), parse_ok=True)
        except Exception as e:  # noqa: BLE001 — JSON 解析失败重试 1 次（§13.2）
            if attempt >= config.JSON_RETRY:
                return VerifyResult(verdict="fail", parse_ok=False,
                                    notes=f"校验调用失败：{e}")
    return VerifyResult(verdict="fail", parse_ok=False)
