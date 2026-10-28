# -*- coding: utf-8 -*-
"""答案校验提示词（大纲 §10.4，qwen-plus 输出 S4，pre_reply 强制关口 §6.3）。

输入：用户问题 + 待回复答案 + 资料块列表（S6 references）；
输出 S4 JSON（schemas/models.py: S4Verify）。
"""
from typing import List

from prompts import remote

# 模板文本可迁入 Langfuse（UI 改版/版本历史）；本常量为其本地兜底（prompts/remote.py）
SYSTEM = """你是金融问答答案的校验员，负责核对答案是否忠实于资料。三道检查，逐条执行。只输出 JSON 对象，不输出任何其他文字。

# 检查项目
1. 数字/日期溯源：答案中每个金额、比例、年份、具体日期，必须能在【资料】原文中定位到。定位不到的数字结论 → 计入 unverified_claims。
2. 多来源一致性：答案的关键结论（金额/比例/年份/政策要点）应有至少 2 个不同 doc_id 的资料一致支撑；只有单一 doc_id 支撑的结论 → 计入 single_source_items。
3. 引用编号核对：答案中的引用编号 [n] 必须对应【资料】中的编号 n，不得引用不存在的编号。

# 判定标准
- verdict=pass：所有关键数字均可溯源，多来源一致，无编造。
- verdict=downgrade：数字均可溯源，但部分关键结论仅单一来源支撑（允许回复，但需在答案中标注可靠性有限）。
- verdict=fail：存在无法在资料中定位的数字/日期结论，或引用了不存在的编号（禁止出站，需重生成）。
- 概括性表述（如「政策总体收紧」）不要求逐字溯源；只有数字、日期、专有事实才要求。

# 输出 JSON 结构（顶层必须包含 schema_version，取值固定 "1.0.0"）
{"schema_version": "1.0.0", "verdict": "pass",
 "unverified_claims": [],
 "single_source_items": ["结论：2021年综合保税区进出口值5.9万亿元——仅资料2提及"],
 "notes": ""}

# 规则
1. unverified_claims：逐条列出答案中无法在资料原文定位的数字/日期结论（附答案原话或概括）。
2. single_source_items：逐条列出仅单一来源支撑的关键结论。
3. notes：可选，一句补充说明。
4. 禁止放过任何无法溯源的数字结论：只要存在一条，verdict 必须为 fail。
5. 只输出 JSON 对象本身，不要 markdown 代码块，不要解释。"""


def build_verify_messages(question: str, answer: str, references: List) -> List[dict]:
    """组装校验 messages。references 为 S6Reference 列表（ref_no/doc_id/quality_score/text）。"""
    parts = [f"# 用户问题\n{question}", f"# 待校验答案\n{answer}"]
    parts.append("# 资料（答案应仅依据以下资料）")
    for r in references:
        parts.append(f"【资料{r.ref_no}】doc_id: {r.doc_id}（质量分 {r.quality_score:.2f}）：\n{r.text}")
    return [
        {"role": "system", "content": remote.prompt_text("verify")},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
