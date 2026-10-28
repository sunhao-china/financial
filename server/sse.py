# -*- coding: utf-8 -*-
"""SSE 桥接（大纲 §14）：daemon 线程跑同步 agent.run → Queue 桥接 → 同步生成器 drain。

执行模型（P5 核心决策，workers=1 硬约束）：
- `def` 路由内起 daemon 线程执行同步 agent.run（LLM 调用 IO 密集，严禁进 async 上下文）
- 线程 emit → Queue(SSE_QUEUE_SIZE) → 同步生成器 drain（starlette 放线程池）
- delta/reasoning 溢出丢弃计数；meta/sources/done/error 必达（独立 deque，数量 ≤4/请求）
- 心跳：SSE_HEARTBEAT_SEC 空窗发 `: ping`；客户端断开置 closed，后续 emit 静默，
  agent 线程照常跑完（快照/历史照常写）
- 事件时序：meta（expand 后）→ sources（assemble 后）→ reasoning*/delta* 交错 → done；
  验证 fail 重生成走非流式，客户端静默等 done；done.answer 为最终权威（§14）
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Iterator, List, Optional

from schemas.models import (S5Hit, S7Cost, S7Verification, S9Delta, S9Done, S9Error,
                            S9Meta, S9Sources, make_sse)

SSE_QUEUE_SIZE = 256            # 流式增量队列容量（溢出丢弃计数）
MUST_DELIVER = {"meta", "sources", "done", "error"}   # 必达事件（独立 deque，不丢）
_STOP = object()                # drain 终止哨兵


class SSEHub:
    """单请求 SSE 事件中枢：worker 线程 emit，drain 生成器消费。"""

    def __init__(self, query_id: str):
        self.query_id = query_id
        self._q: queue.Queue = queue.Queue(SSE_QUEUE_SIZE)
        self._must: deque = deque()             # 必达事件，保持 emit 顺序
        self.closed = threading.Event()         # 客户端断开
        self.finished = threading.Event()       # worker 线程结束（drain 终止判据）
        self.drop_count = 0                     # delta/reasoning 溢出丢弃计数
        self._lock = threading.Lock()

    def emit(self, event: str, payload) -> None:
        """线程安全 emit。客户端已断开则静默；必达事件永不丢弃。"""
        if self.closed.is_set():
            return
        if event in MUST_DELIVER:
            with self._lock:
                self._must.append((event, payload))
        else:
            try:
                self._q.put_nowait((event, payload))
            except queue.Full:
                with self._lock:
                    self.drop_count += 1

    def drain(self, heartbeat_sec: float) -> Iterator[str]:
        """同步生成器：必达 → 流式增量，保序；空窗发心跳；finished 且清空后结束。"""
        idle_start = None
        while True:
            item = self._next()
            if item is _STOP:
                return
            if item is None:
                now = time.monotonic()
                if idle_start is None:
                    idle_start = now
                elif now - idle_start >= heartbeat_sec:
                    idle_start = now
                    yield ": ping\n\n"
                time.sleep(0.05)
                continue
            idle_start = None
            event, payload = item
            yield self.render(event, payload)

    def _next(self):
        """优先级（保证 §14 时序契约 meta→sources→delta*→done）：
        1) meta/sources 立即送出（emit 顺序上先于一切 delta）
        2) 流式增量优先（done 不能抢跑）
        3) done/error：worker 已结束（此后无新 delta）且增量队列清空后才送出"""
        with self._lock:
            if self._must and self._must[0][0] in ("meta", "sources"):
                return self._must.popleft()
        try:
            return self._q.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._must and self._must[0][0] in ("done", "error") and self.finished.is_set():
                return self._must.popleft()
        if self.finished.is_set():
            return _STOP
        return None

    def render(self, event: str, payload) -> str:
        """payload（pydantic 模型）→ S9 信封校验 → SSE data 帧（§9.2 双向校验）。"""
        ev = make_sse(event, self.query_id, payload,
                      ts=datetime.now().astimezone().isoformat(timespec="milliseconds"))
        return f"data: {ev.model_dump_json()}\n\n"


def to_s5hits(hits: List, ret) -> List[S5Hit]:
    """SearchHit → S5Hit（§9.3 统一出口）。BM25 分无上界，clamp 到 S5Hit 的 0~1。"""
    out = []
    for h in hits:
        src = "vector" if h.source == "vector" else ("bm25" if h.source == "bm25" else "fusion")
        out.append(S5Hit(
            chunk_id=h.id, doc_id=h.doc_id, text=h.text,
            score=min(max(float(h.score), 0.0), 1.0),
            rank_source=src,
            # P6：元数据随 SearchHit 由 Qdrant payload 携带，不再查检索器进程内数组
            quality_score=float(h.quality),
            chunk_index=int(h.chunk_index),
            chunk_count=int(h.chunk_count),
            doc_title=ret.doc_titles.get(h.doc_id)))
    return out


def build_done(ans) -> S9Done:
    """AgentAnswer → S9 done 事件（§14）。fallback 非空时其文本即最终回复。"""
    by_step = {}
    for s in ans.path:
        by_step[s["step"]] = by_step.get(s["step"], 0) + s.get("cost", 0)
    verification = S7Verification.model_validate(ans.verification) if ans.verification else None
    return S9Done(
        answer=ans.answer or ans.fallback,
        citations=ans.citations,
        verification=verification,
        timings_ms=ans.timings or {},
        cost_tokens=S7Cost(total=ans.cost, by_step=by_step))


def run_chat_blocking(agent, question: str, history: List[dict], session_id: str,
                      hub: SSEHub, chat_history, metrics, user_id: int = 0,
                      username: str = "") -> None:
    """worker 线程入口（def 路由内启动）：跑完整 agent.run，事件进 hub，
    结束写会话历史（客户端断开也照常写）。异常只走 error 事件（业务降级已在 agent 内消化）。
    user_id 用于 MySQL 会话归属（懒落库 conversations，§14.5）；
    username 传给 Langfuse trace 的 User 维度（§13.5）。"""
    try:
        ans = agent.run(question, history=history, session_id=session_id, verify=True,
                        stream=True,
                        on_delta=lambda t: hub.emit("delta", S9Delta(delta=t)),
                        on_reasoning=lambda t: hub.emit("reasoning", S9Delta(delta=t)),
                        query_id=hub.query_id, user=username)
        if chat_history is not None and session_id:
            chat_history.append(session_id, question, ans.answer or ans.fallback,
                                user_id=user_id)
        hooks = getattr(agent, "hooks", None)
        if hooks and hasattr(hooks, "ensure_meta"):
            hooks.ensure_meta()      # S9 时序契约：done 前必须已发 meta（任何路径）
        hub.emit("done", build_done(ans))
    except Exception as e:  # noqa: BLE001 — 真异常（LLM 不可用等业务降级不在此）
        metrics.inc("chat_error")
        hooks = getattr(agent, "hooks", None)
        if hooks and hasattr(hooks, "ensure_meta"):
            hooks.ensure_meta()
        hub.emit("error", S9Error(code="internal_error",
                                  message=f"{type(e).__name__}: {e}"[:500]))
    finally:
        metrics.inc("sse_drop", hub.drop_count)
        hub.finished.set()
