# -*- coding: utf-8 -*-
"""P5 Web 服务入口（大纲 §14）：FastAPI + SSE 流式 + 多轮对话 + 监控告警 + 降级收口。

启动（systemd，workers=1——v1 保留：检索器含 bge-m3/BM25 常驻内存与进程内缓存，
向量路已外置 Qdrant（P6），内存大头卸载）:
    .venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8001 --workers 1

路由：
    POST /chat          S9 问答（SSE：meta → sources → reasoning/delta* → done）
    GET  /sources/{q}   检索调试（S9SourcesResponse，§14）
    GET  /health        健康状态（维护态/降级态/Redis 可用性）
    GET  /metrics       Prometheus 文本（§13.5）

关键约束：
- 同步 LLM 调用严禁进 async 路由：/chat 为 def 路由，worker 线程跑 agent.run，
  生成器 drain 由 starlette 放线程池
- 并发靠 CHAT_CONCURRENCY 信号量（LLM 调用 IO 密集）；超时 429
- Redis 仅作增强：不可用自动降级 Memory，服务不挂（§13.3）
"""
from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import config
from corpus.agent import SearchAgent, new_query_id
from llm_client import set_retry_callback
from prompts.refusals import MAINTENANCE
from schemas.models import S9ChatRequest, S9SourcesResponse
from server import auth
from server.auth_guard import ip_throttle
from server.captcha import CaptchaStore
from server.chat_history import (ChatHistoryStore, MemoryChatHistory, MySQLChatHistory,
                                 merge_history, trim_history)
from server.db import db
from server.hooks import ServiceHooks
from server.logging_s8 import S8Logger
from server.metrics import Metrics
from server.session_redis import RedisHealth, RedisSessionStore
from server.sse import SSEHub, run_chat_blocking, to_s5hits
from server.state import HealthState

metrics = Metrics()
s8: S8Logger = S8Logger(stdout=False)      # lifespan 中按配置重建
retriever = None
state: HealthState = None
redis_health: RedisHealth = None
session_store = None
chat_history: ChatHistoryStore = MemoryChatHistory()
_sem = threading.BoundedSemaphore(config.CHAT_CONCURRENCY)


@asynccontextmanager
async def lifespan(_app):
    global retriever, state, redis_health, session_store, chat_history, s8
    s8 = S8Logger(path=config.S8_LOG_PATH, stdout=config.S8_LOG_STDOUT)
    set_retry_callback(metrics.on_retry)

    # 检索器加载一次（~1min）；失败以维护态启动，探针周期重试（§13.3）
    try:
        from corpus.retrieval import HybridRetriever
        retriever = HybridRetriever()
        s8.emit(query_id="boot", step="startup", level="info",
                payload={"msg": f"检索器就绪 {retriever.n_chunks:,} chunks（向量路=Qdrant）"})
    except Exception as e:  # noqa: BLE001
        s8.emit(query_id="boot", step="startup", level="error", payload={"err": str(e)})

    redis_health = RedisHealth()
    session_store = RedisSessionStore(redis_health)

    # 用户系统（2026-10-19 加）：MySQL 建表 + 预置管理员；失败静默（登录暂不可用，问答照常）
    db.init_schema()
    auth.ensure_admin()
    auth.captcha_store = CaptchaStore(redis_health)
    # 登录安全加固（2026-10-22）：auth 与 IP 限流器挂上指标
    auth.metrics = metrics
    ip_throttle.set_health(redis_health)  # IP 限流迁 Redis 滑动窗口（2026-10-25，多 worker 共享）
    ip_throttle.metrics = metrics
    # 会话历史主存储 MySQL（Memory 兜底，§14.5）；Redis fin:conv 退役
    chat_history = MySQLChatHistory(db)

    state = HealthState(retriever, metrics)
    if retriever is None:
        state.enter_maintenance("启动加载失败")
    yield
    db.close()
    # shutdown：无重资源（探针/worker 均为 daemon 线程）


app = FastAPI(title="金融 RAG 问答服务", version="1.0.0", lifespan=lifespan)


@app.post("/chat")
def chat(req: S9ChatRequest, user: dict = Depends(auth.require_user)):
    """S9 问答（§14）：SSE 流式，需登录（Bearer token，§14.5）。事件时序见 server/sse.py。

    维护态 503；并发打满 429；业务降级（拒答/LLM 不可用/验证 fail/空检索）
    已被 agent 消化为 done.answer 中的话术，error 事件只代表真异常。
    """
    if state.status == state.MAINTENANCE:
        raise HTTPException(503, detail=MAINTENANCE)
    if not req.session_id:
        raise HTTPException(400, detail="缺少 session_id（请先新建会话）")
    if not _sem.acquire(timeout=config.CHAT_SEMAPHORE_TIMEOUT):
        raise HTTPException(429, detail="系统繁忙，请稍后再试。")
    metrics.inc("chat_total")

    query_id = new_query_id()
    session_id = req.session_id
    server_history = chat_history.get(session_id)
    history = trim_history(merge_history([m.model_dump() for m in req.history],
                                         server_history))
    hub = SSEHub(query_id)
    agent = SearchAgent(retriever,
                        hooks=ServiceHooks(hub, s8, metrics, state, retriever, query_id),
                        store=session_store)
    threading.Thread(target=run_chat_blocking,
                     args=(agent, req.question, history, session_id, hub,
                           chat_history, metrics, user["id"], user["username"]),
                     daemon=True, name=f"chat-{query_id}").start()

    def gen():
        try:
            yield from hub.drain(config.SSE_HEARTBEAT_SEC)
        except GeneratorExit:   # 客户端断开：后续 emit 静默，worker 照常完成写历史
            hub.closed.set()
        finally:
            _sem.release()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/sources/{query}")
def sources(query: str):
    """检索调试接口（§14）：生产配置检索一次，返回 S5Hit 列表。"""
    if state.status == state.MAINTENANCE:
        raise HTTPException(503, detail=MAINTENANCE)
    hits = retriever.retrieve(query, rerank=True, n=20)
    return S9SourcesResponse(query=query,
                             results=to_s5hits(hits, retriever)).model_dump()


@app.get("/health")
def health():
    """健康状态（§13.5）：维护态/降级态/Redis 可用性。"""
    streak = -1
    if retriever is not None:
        with retriever._cache_lock:
            streak = retriever.vector_fail_streak
    return {
        "status": state.status,
        "reason": state.reason,
        "vector_fail_streak": streak,
        "redis": redis_health.available if redis_health else False,
        "mysql": db.available,
        "uptime_s": int(time.time() - state.started_at),
    }


@app.get("/metrics")
def metrics_endpoint():
    """Prometheus 文本（§13.5）：计数器 + 直方图 p50/p95/avg。"""
    return PlainTextResponse(metrics.render_prometheus())


# 用户系统路由（§14.5）：注册在 mount 之前，/api/* 优先于静态托管
app.include_router(auth.router)

# 前端静态托管（§14）：API 路由先于 mount 注册、匹配优先；目录缺失优雅降级（纯 API 模式）。
# html=True 仅对目录路径回退 index.html，未知路径 404（单视图应用无深链，正确行为）。
if config.FRONTEND_DIST.is_dir() and (config.FRONTEND_DIST / "index.html").is_file():
    app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIST), html=True), name="frontend")
