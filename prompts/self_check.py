# -*- coding: utf-8 -*-
"""信息自检提示词（大纲 §10.3，qwen-plus 输出 S3，Agent 循环慢路径判据）。

输入：原始问题 + 已尝试查询 + 当前检索片段（S5 列表）；
输出 S3 JSON（schemas/models.py: S3SelfCheck）。
"""
from typing import List

from prompts import remote

# 模板文本可迁入 Langfuse（UI 改版/版本历史）；本常量为其本地兜底（prompts/remote.py）
SYSTEM = """你是 RAG 检索质量自检器。给定用户问题与当前检索到的资料片段，判断这些片段是否足以回答该问题。只输出 JSON 对象，不输出任何其他文字。

# 判断标准（宁宽勿严）
- 只要片段中存在与问题直接相关的信息，就判 sufficient=true，避免无意义补检。
- 只有关键信息明显缺失（例如问的是某年的数据、片段全是其他年份；问的是某公司、片段全是无关公司；问的是某个指标、片段只字未提）才判 sufficient=false。
- 问题中每个关键约束（时间、对象、指标）都要在片段中有对应信息才算完整覆盖。

# 输出 JSON 结构（顶层必须包含 schema_version，取值固定 "1.0.0"）
{"schema_version": "1.0.0", "sufficient": false,
 "missing": ["缺少2021年的数据，片段均为2019年"],
 "followup_queries": ["2021年综合保税区进出口数据"]}

# 规则
1. sufficient=false 时：missing 逐条列出缺失的信息点；followup_queries 给出 1~2 个针对缺失点的补充检索查询。
2. followup_queries 最多 2 个，且不得与「已尝试查询」中任何一个重复（语义相近也算重复）。
3. sufficient=true 时：missing 与 followup_queries 输出空数组。
4. 只输出 JSON 对象本身，不要 markdown 代码块，不要解释。"""


def build_self_check_messages(question: str, tried_queries: List[str], chunks: List) -> List[dict]:
    """组装自检 messages。chunks 为 S5 条目列表（chunk_id/doc_id/text）。"""
    parts = [f"# 用户问题\n{question}", "# 已尝试查询\n" + "\n".join(tried_queries)]
    parts.append("# 当前检索到的资料片段")
    if not chunks:
        parts.append("（本次检索未返回任何片段）")
    for i, c in enumerate(chunks, 1):
        # 兼容 SearchHit（id/doc_id）与 dict/S5 条目（chunk_id）
        cid = getattr(c, "chunk_id", None) or getattr(c, "id", "")
        parts.append(f"【片段{i}】{cid}（doc_id: {getattr(c, 'doc_id', '')}）：\n{getattr(c, 'text', '')}")
    return [
        {"role": "system", "content": remote.prompt_text("self_check")},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
