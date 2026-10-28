# -*- coding: utf-8 -*-
"""意图识别+查询扩展提示词（大纲 §10.1，qwen-plus 一次调用输出 S2，§4 示例）。

输入：对话历史 + 当前问题；输出 S2 JSON（schemas/models.py: S2IntentExpand）。
联调/回归：tests/test_p1_live.py 的 intent 阶段。
"""
import json
from typing import List, Optional

from prompts import remote

# 模板文本可迁入 Langfuse（UI 改版/版本历史）；本常量为其本地兜底（prompts/remote.py）
SYSTEM = """你是金融问答系统的意图识别与查询扩展模块。只做两件事：判断用户问题的意图类别，并为检索生成查询变体与关键词。只输出 JSON 对象，不输出任何其他文字。

# 意图类别（六选一）
- fact_qa：事实检索类——问具体事实、数据、政策内容、历史事件（例：2021年综合保税区进出口值是多少）
- analysis：归纳分析类——需要综合多篇资料总结规律、分析原因与影响（例：总结央企海外并购的主要问题）
- comparison：对比类——比较两个对象、政策或事件的异同（例：人民币破7与破8的影响有何不同）
- realtime：时效性类——问今天/当前/最新的行情数据、实时信息（例：今天A股涨了多少）
- investment_advice：个股推荐/投资建议类——问该买什么、该不该买、会不会涨（例：买什么股票好）
- chitchat：闲聊/寒暄/与金融无关的话题（例：你好、谢谢）

# 输出 JSON 结构（顶层必须包含 schema_version，取值固定 "1.0.0"）
{"schema_version": "1.0.0", "intent": "fact_qa", "confidence": 0.95,
 "entities": ["综合保税区", "2021年", "进出口值"],
 "queries": ["2021年全国综合保税区进出口值是多少", "2021年综合保税区进出口数据", "综合保税区进出口额统计"],
 "keywords": ["综合保税区", "进出口值", "2021"]}

# 规则
1. queries：3~5 个检索变体，从不同角度表述（换说法、拆子问题、补同义背景），互不重复，保留关键数字与专名。仅当 intent 为 realtime / investment_advice / chitchat 时，queries 与 keywords 输出空数组。
2. keywords：只输出用于关键词检索的金融实体词——公司名、机构名、政策名、指标名、行业术语，2~8 个，不含虚词。
3. entities：问题涉及的关键实体（公司/政策/指标/时间），1~6 个。
4. confidence：0~1 置信度。跨类别判断把握不大时取低值（如 <0.7）。
5. 时效边界：问题明确涉及「今天/当前/最新/实时」的行情数据 → realtime；明确要求推荐个股、询问买卖时机 → investment_advice。
6. 只输出 JSON 对象本身，不要 markdown 代码块，不要解释。"""

# 每类意图 1 个示例（§10.1：few-shot，示例取自真实语料问题风格）
FEWSHOT = [
    ("2021年全国综合保税区进出口值是多少？",
     {"schema_version": "1.0.0", "intent": "fact_qa", "confidence": 0.97,
      "entities": ["综合保税区", "2021年", "进出口值"],
      "queries": ["2021年全国综合保税区进出口值是多少", "2021年综合保税区进出口数据", "综合保税区进出口额统计"],
      "keywords": ["综合保税区", "进出口值", "2021"]}),
    ("总结央企海外并购中存在的主要问题",
     {"schema_version": "1.0.0", "intent": "analysis", "confidence": 0.93,
      "entities": ["央企", "海外并购"],
      "queries": ["央企海外并购面临的问题与风险", "央企海外并购失败案例原因", "央企海外并购监管审查"],
      "keywords": ["央企", "海外并购", "风险", "审查"]}),
    ("人民币破7和破8的影响有什么不同？",
     {"schema_version": "1.0.0", "intent": "comparison", "confidence": 0.92,
      "entities": ["人民币汇率", "破7", "破8"],
      "queries": ["人民币汇率破7的影响", "人民币汇率破8意味着什么", "人民币贬值对经济的影响"],
      "keywords": ["人民币汇率", "破7", "破8"]}),
    ("今天A股大盘涨了多少？",
     {"schema_version": "1.0.0", "intent": "realtime", "confidence": 0.99,
      "entities": ["A股"], "queries": [], "keywords": []}),
    ("现在买什么股票比较好？",
     {"schema_version": "1.0.0", "intent": "investment_advice", "confidence": 0.98,
      "entities": [], "queries": [], "keywords": []}),
    ("你好，你是谁？",
     {"schema_version": "1.0.0", "intent": "chitchat", "confidence": 0.99,
      "entities": [], "queries": [], "keywords": []}),
]


def build_intent_messages(question: str, history: Optional[List[dict]] = None,
                          max_history_turns: int = 3) -> List[dict]:
    """组装意图识别+查询扩展的 messages（system + few-shot + 历史 + 当前问题）。"""
    parts = ["# 示例（输入问题 → 输出 JSON）"]
    for q, a in FEWSHOT:
        parts.append(f"问题：{q}\n输出：{json.dumps(a, ensure_ascii=False)}")
    if history:
        turns = [m for m in history if m.get("role") in ("user", "assistant")][-max_history_turns * 2:]
        parts.append("# 对话历史（用于指代消解，可能为空）")
        for m in turns:
            parts.append(f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}")
    parts.append(f"# 当前问题\n{question}")
    return [
        {"role": "system", "content": remote.prompt_text("intent_expand")},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
