# -*- coding: utf-8 -*-
"""Langfuse 远端提示词（2026-10-22，§13.5 追踪层同款降级哲学）。

5 套 LLM 模板文本迁入 Langfuse Prompts（UI 改版、版本历史、回滚），
本地 prompts/*.py 的 SYSTEM 常量保留为兜底默认——远端不可用时行为与迁移前完全一致。

机制：
- get_prompt(name)：复用 llm_client 的 Langfuse 客户端（惰性、单例），
  label="production"；SDK 自带内存缓存（cache_ttl_seconds=300 → 热路径最多每 5 分钟
  一次远端拉取，零额外延迟）与 fetch 失败兜底。任何异常 → 返回 None。
- prompt_text(name)：远端 production 版文本；不可用/空文本 → 本地 SYSTEM 默认。
- 名字约定：Langfuse 中为 financial-rag/<name>，name ∈ intent_expand/generate/
  self_check/verify/judge（与 prompts/*.py 模块一一对应）。
- FIN_LANGFUSE_PROMPTS=0 整体关闭（纯本地，等价迁移前）。

版本联动（§13.5）：调用方把 get_prompt() 返回的 prompt 客户端经
client.chat(prompt_client=...) 传给 span——UI 每条 LLM span 显示所用提示词版本，
评测轮次可精确对比「哪版提示词产出哪个答案」。
"""
from __future__ import annotations

import os

import config

PREFIX = "financial-rag/"
CACHE_TTL_SEC = 300        # UI 改完提示词最多 5 分钟生效（改大减拉取、改小提时效）
FETCH_TIMEOUT_SEC = 3.0    # 单次远端拉取超时（超时/失败走兜底，问答零影响）
NAMES = ("intent_expand", "generate", "self_check", "verify", "judge")


def _default_text(name: str) -> str:
    """本地兜底文本 = prompts/<name>.py 的 SYSTEM 常量（即迁移前行为）。"""
    mod = __import__(f"prompts.{name}", fromlist=["SYSTEM"])
    return mod.SYSTEM


def get_prompt(name: str):
    """Langfuse 提示词客户端（production 标签，带缓存）；不可用 → None。"""
    if not (config.LANGFUSE_ENABLED and config.LANGFUSE_PUBLIC_KEY
            and os.environ.get("FIN_LANGFUSE_PROMPTS", "1") == "1"):
        return None
    try:
        from llm_client import _lf  # noqa: PLC0415 — 复用同一客户端（同进程单例）
        lf = _lf()
        if lf is None:
            return None
        return lf.get_prompt(
            f"{PREFIX}{name}", label="production",
            cache_ttl_seconds=CACHE_TTL_SEC, fetch_timeout_seconds=FETCH_TIMEOUT_SEC)
    except Exception:  # noqa: BLE001 — SDK 缺失/网络失败 → 本地兜底
        return None


def prompt_text(name: str) -> str:
    """模板文本：远端 production 版；不可用/为空 → 本地 SYSTEM 默认。"""
    p = get_prompt(name)
    text = getattr(p, "prompt", "") if p is not None else ""
    if text and text.strip():
        return text
    return _default_text(name)
