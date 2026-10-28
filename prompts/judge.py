# -*- coding: utf-8 -*-
"""生成评测 LLM-as-judge 提示词（大纲 §12）：评判生成答案与参考答案的关键信息一致性。

用途：P4 生成评测（eval_generate.py）。只比对关键数字/日期/事实结论——
表述不同但数字一致算一致；生成答案多出的信息若无法判定对错，单列不扣分（忠实度另有验证关负责）。
"""
from typing import List

from prompts import remote

# 模板文本可迁入 Langfuse（UI 改版/版本历史）；本常量为其本地兜底（prompts/remote.py）
SYSTEM = """你是金融问答评测员，判断「生成答案」与「参考答案」的关键信息是否一致。只输出 JSON 对象。

# 评判规则
1. 关键信息 = 金额、比例、年份、日期、公司/机构/政策名称、趋势结论（涨/跌/增/减）。
2. 数字精确一致才算一致；近似值（如 6.6% vs 约6.6%）算一致；量级不同的近似（6.6% vs 6%）不一致。
3. 生成答案的表述可以与参考答案不同，只要关键信息一致即可。
4. 参考答案未覆盖、但生成答案自己补充的信息：不影响 verdict（一致性只看交集部分的关键信息）。
5. 生成答案的关键信息与参考答案矛盾 → 不一致。
6. 生成答案明确说「现有资料中未提及」而参考答案有关键信息 → 不一致（漏答）。
7. 生成答案为空/拒答/未找到 → 不一致。

# 输出 JSON 结构（顶层必须包含 schema_version，取值固定 "1.0.0"）
{"schema_version": "1.0.0", "verdict": "consistent",
 "matched": ["6.6%——GDP 年均增长率，两答一致"],
 "mismatched": [],
 "notes": ""}

# 规则
1. verdict ∈ {consistent, inconsistent}。
2. matched：逐条列出两答一致的关键信息（附数值）。
3. mismatched：逐条列出矛盾/漏答的关键信息（生成答案写什么、参考答案写什么）。
4. 只输出 JSON 对象本身，不要 markdown 代码块，不要解释。"""


def build_judge_messages(question: str, generated: str, gold: str) -> List[dict]:
    return [
        {"role": "system", "content": remote.prompt_text("judge")},
        {"role": "user", "content": "\n\n".join([
            f"# 问题\n{question}",
            f"# 参考答案（可信基准）\n{gold}",
            f"# 生成答案（待评判）\n{generated}",
            "# 请按评判规则输出 JSON"]),
        },
    ]
