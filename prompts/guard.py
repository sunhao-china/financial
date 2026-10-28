# -*- coding: utf-8 -*-
"""意图兜底正则（大纲 §4：关键词正则先行拦截，LLM 置信度低时默认走 RAG）。

只做高置信拦截：realtime（时效性）与 investment_advice（买卖建议）两类
需要话术拒答的意图；其余一律放行给 LLM 判断，宁检索不漏。
"""
import re

ADVICE_PAT = re.compile(
    r"买什么|该不该买|能不能买|值得买|买入|卖出|加仓|减仓|满仓|清仓|止损|止盈|建仓"
    r"|继续持有|还能持有|该不该持有"          # 「持有」裸词太宽（持有比例/长期持有是中性事实询问）
    r"|推荐.{0,6}(股|基金)|(股|基金).{0,6}推荐"
)
REALTIME_PAT = re.compile(
    r"今天|今日|现在|最新|实时|昨日收盘|今日收盘|大盘.{0,4}(涨|跌)|(涨|跌).{0,4}多少"
    # 删「当前/刚刚/刚才」：政策分析题常带背景描述（当前外贸压力），误拦率高，交给 LLM intent
)


def quick_guard(question: str):
    """返回 "realtime" / "investment_advice" / None。

    买卖类优先于时间类（「现在买入茅台」是投资建议而非行情查询）。
    """
    q = question.strip()
    if ADVICE_PAT.search(q):
        return "investment_advice"
    if REALTIME_PAT.search(q):
        return "realtime"
    return None
