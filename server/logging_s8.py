# -*- coding: utf-8 -*-
"""S8 链路事件日志（§13.5）：JSONL 落盘（10MB 轮转/30 天）+ 可选 stdout。

emit 前经 S8Event 构造强制校验（§9.2），校验失败的事件拒绝落盘。
loguru enqueue=True 后台线程写盘，不阻塞请求线程；S8Logger 单例（服务进程内唯一）。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import ValidationError

from schemas.models import S7Cost, S8Event


class S8Logger:
    def __init__(self, path: Optional[Path] = None, stdout: bool = True):
        logger.remove()   # 去掉默认 handler（本进程内 loguru 只服务 S8）
        fmt = "{message}"
        if stdout:
            logger.add(sys.stdout, format=fmt, level="INFO")
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            logger.add(str(path), format=fmt, level="INFO",
                       rotation="10 MB", retention=30, encoding="utf-8", enqueue=True)

    def emit(self, query_id: str, step: str, level: str = "info",
             payload: Optional[dict] = None, latency_ms: Optional[int] = None,
             cost_tokens: Optional[S7Cost] = None) -> None:
        try:
            ev = S8Event(
                ts=datetime.now().astimezone().isoformat(timespec="milliseconds"),
                query_id=query_id, step=step, level=level,
                payload=payload or {}, latency_ms=latency_ms, cost_tokens=cost_tokens)
        except ValidationError:
            return   # §9.2：校验失败拒绝落盘（装配错误当场暴露在 stdout 之外）
        logger.info(json.dumps(ev.model_dump(), ensure_ascii=False))
