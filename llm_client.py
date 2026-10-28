# -*- coding: utf-8 -*-
"""双平台 LLM 客户端：百炼（DashScope）+ DeepSeek（大纲 §3）。

统一封装：鉴权、重试（指数退避+抖动，仅限流/5xx/网络错误）、超时、usage 回传（§13.1）。
模型路由见 config.MODELS，切换模型只改配置不改代码。

用法:
    client = LLMClient()

    r = client.chat([{"role": "user", "content": "..."}], route="generate_fact")
    print(r.content, r.usage)                      # Usage: prompt/completion/reasoning/total

    for ev in client.chat_stream(messages, route="generate_fact"):
        ...                                        # {"type":"reasoning"|"content","text"} / {"type":"usage","usage"}

    vecs = client.embed(["文本1", "文本2"])          # -> 按输入顺序的 1024 维向量列表
    hits = client.rerank("query", ["doc1", "doc2"])  # -> [{"index":0,"score":0.93}, ...]
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator, List, Optional

import requests
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential, wait_random

import config


class APIError(Exception):
    """带状态码与响应体的 API 错误（重试耗尽后抛出）。"""

    def __init__(self, provider: str, status: int, body: dict):
        self.provider = provider
        self.status = status
        self.body = body
        super().__init__(f"{provider} HTTP {status}: {json.dumps(body, ensure_ascii=False)[:200]}")


@dataclass
class Usage:
    """一次调用的 token 用量（大纲 §13.1，含推理 token）。"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_response(cls, usage: Optional[dict]) -> "Usage":
        if not usage:
            return cls()
        details = usage.get("completion_tokens_details") or {}
        return cls(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            reasoning_tokens=details.get("reasoning_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class ChatResult:
    content: str
    usage: Usage
    model: str = ""
    finish_reason: str = ""


def _is_retryable(exc: BaseException) -> bool:
    """仅网络错误/超时/限流/5xx 触发重试（大纲 §13.2）。"""
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    if isinstance(exc, APIError):
        return exc.status == 429 or exc.status >= 500
    return False


# P5 指标观测：重试前回调（重试率指标唯一来源）。服务启动时注册，None 无操作。
_retry_callback: Optional[Callable] = None


def set_retry_callback(cb: Optional[Callable]) -> None:
    """注册重试回调 cb(retry_state, label)，label 为路由/工序名（P5 server/metrics）。"""
    global _retry_callback
    _retry_callback = cb


def _before_sleep(retry_state, label: str) -> None:
    if _retry_callback:
        try:
            _retry_callback(retry_state, label)
        except Exception:  # noqa: BLE001 — 观测回调不得影响重试主流程
            pass


# ---- Langfuse 追踪层（§13.5，2026-10-20 加）--------------------------------
# 自托管实例（config.LANGFUSE_HOST，ECS 同机，数据不出境）。设计约束：
# ① 未启用/未装 SDK/初始化失败 → 全部 no-op，问答主流程零影响；
# ② 一次问答 = 一条 trace，trace_id 由 query_id 确定性派生（可在 UI 反查 S8 日志）；
# ③ 每个 LLM 工序一个子 span，经 OTel contextvars 自动挂到当前 trace
#    （workers=1，单请求单线程执行，传播安全）。
_lf_client = None          # 惰性单例；None = 追踪关闭
_lf_disabled = False       # 初始化失败后本次进程内不再重试


def _lf():
    """惰性创建 Langfuse 客户端；未启用或 SDK 缺失时返回 None（追踪 no-op）。"""
    global _lf_client, _lf_disabled
    if _lf_client is not None or _lf_disabled:
        return _lf_client
    if not (config.LANGFUSE_ENABLED and config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        return None
    try:
        from langfuse import Langfuse  # noqa: PLC0415 — 可选依赖，惰性导入
        _lf_client = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
        )
    except Exception:  # noqa: BLE001 — SDK 缺失/初始化失败 → 静默降级
        _lf_disabled = True
    return _lf_client


def _safe_update(gen, **kw) -> None:
    """span.update 包装：追踪链路任何异常不得影响主流程。"""
    try:
        gen.update(**kw)
    except Exception:  # noqa: BLE001
        pass


class TraceHandle:
    """llm_trace 产物：追踪关闭时为 no-op 句柄（update 静默忽略），调用方无需判空。"""

    def __init__(self, raw=None):
        self.raw = raw

    def update(self, **kw) -> None:
        if self.raw is not None:
            _safe_update(self.raw, **kw)


@contextmanager
def llm_trace(query_id: str, question: str, session_id: str = "",
              user_id: str = "") -> Iterator[TraceHandle]:
    """一次问答的 Langfuse trace 上下文（agent.run 入口调用）。

    with 块内所有 LLM 调用自动成为该 trace 的子 span；关闭/失败时 no-op。
    session_id/user_id 走 trace_context 一等字段（UI Session/User 列，§13.5）。
    """
    lf = _lf()
    if lf is None:
        yield TraceHandle()
        return
    try:
        trace_id = lf.create_trace_id(seed=query_id)   # query_id → 32 位 hex（确定性，可反查）
        import langfuse as _lf_pkg
        # trace 级属性（UI Session/User 列）：官方推荐 propagate_attributes，早于首 span 开启，
        # 经 OTel contextvars 挂到 root span 再被服务端提升为 trace 维度
        pa = _lf_pkg.propagate_attributes(session_id=session_id or None,
                                          user_id=user_id or None)
        cm = lf.start_as_current_observation(
            name=question[:60] or "金融问答",
            as_type="span",
            trace_context={"trace_id": trace_id},
            input=question,
            metadata={"query_id": query_id, "session_id": session_id,
                      "user_id": user_id, "service": "financial-rag"},
        )
    except Exception:  # noqa: BLE001 — 创建失败 → 本次追踪放弃
        yield TraceHandle()
        return
    with pa, cm as trace:
        try:
            yield TraceHandle(trace)
        except BaseException as e:  # noqa: BLE001 — 标注失败 trace 后原样上抛
            _safe_update(trace, level="ERROR",
                         status_message=f"{type(e).__name__}: {e}"[:500])
            raise


def _lf_span(name: str, as_type: str, *, input=None, model: str = "",
             metadata: Optional[dict] = None, prompt=None):
    """开启一个 Langfuse 子 span（自动挂到当前 trace；无 trace 时 SDK 自建新 trace）。

    prompt 为 Langfuse 提示词客户端（prompts/remote.py get_prompt 返回值）——
    UI 该 span 显示所用提示词名与版本（§13.5 版本联动）。追踪关闭或创建失败
    → (nullcontext(), False)，主流程照常。
    """
    lf = _lf()
    if lf is None:
        return nullcontext(), False
    try:
        cm = lf.start_as_current_observation(
            name=name, as_type=as_type, input=input, model=model or None,
            metadata=metadata or None, prompt=prompt or None)
        return cm, True
    except Exception:  # noqa: BLE001
        return nullcontext(), False


class LLMClient:
    def __init__(self):
        self._keys = {
            "dashscope": config.DASHSCOPE_API_KEY,
            "deepseek": config.DEEPSEEK_API_KEY,
        }
        self._chat_bases = {
            # 百炼走 OpenAI 兼容模式；DeepSeek 原生 OpenAI 兼容
            "dashscope": f"{config.DASHSCOPE_BASE}/compatible-mode/v1",
            "deepseek": config.DEEPSEEK_BASE,
        }

    # ---------- 内部 ----------

    def _post(self, provider: str, url: str, payload: dict, timeout: float,
              label: Optional[str] = None) -> dict:
        """带重试的 POST；重试耗尽抛 APIError。label 供重试回调按工序计数（P5）。"""
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10) + wait_random(0, 1),
            retry=retry_if_exception(_is_retryable),
            before_sleep=lambda rs: _before_sleep(rs, label or provider),
            reraise=True,
        )
        def _do() -> dict:
            r = requests.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {self._keys[provider]}"},
                timeout=timeout,
            )
            body = r.json() if r.content else {}
            if r.status_code != 200:
                raise APIError(provider, r.status_code, body)
            return body

        return _do()

    def _route(self, route: str):
        cfg = config.MODELS[route]
        return cfg["provider"], cfg["model"]

    @staticmethod
    def parse_json_content(content: str) -> dict:
        """从 LLM 回复提取 JSON（容忍 ```json 代码围栏，§9 校验前用）。"""
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)

    # ---------- 对话 ----------

    def chat(
        self,
        messages,
        *,
        route: str = "generate_fact",
        json_mode: bool = False,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        timeout: float = 180,
        prompt_client=None,
    ) -> ChatResult:
        """非流式对话。max_tokens 需覆盖推理+正文（大纲 §3 实测坑）。
        prompt_client 为 Langfuse 提示词客户端（span 版本联动，可选）。"""
        provider, model = self._route(route)
        if json_mode and provider != "dashscope":
            raise ValueError("json_mode 仅支持百炼（DeepSeek 推理模型不支持 response_format）")
        t0 = time.time()
        cm, live = _lf_span(route, "generation", input=messages, model=model,
                            metadata={"provider": provider, "stream": False},
                            prompt=prompt_client)
        with cm as gen:
            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}
                body = self._post(provider, f"{self._chat_bases[provider]}/chat/completions",
                                  payload, timeout, label=route)
                choice = body["choices"][0]
                msg = choice["message"]
                result = ChatResult(
                    content=msg.get("content") or "",
                    usage=Usage.from_response(body.get("usage")),
                    model=body.get("model", model),
                    finish_reason=choice.get("finish_reason", ""),
                )
            except BaseException as e:  # noqa: BLE001 — 失败标注后原样上抛（重试语义不变）
                if live:
                    _safe_update(gen, level="ERROR",
                                 status_message=f"{type(e).__name__}: {e}"[:500])
                raise
            if live:
                _safe_update(
                    gen,
                    output=result.content,
                    usage_details={"input": result.usage.prompt_tokens,
                                   "output": result.usage.completion_tokens,
                                   "total": result.usage.total_tokens},
                    metadata={"provider": provider, "stream": False,
                              "model_actual": result.model,
                              "finish_reason": result.finish_reason,
                              "reasoning_tokens": result.usage.reasoning_tokens,
                              "latency_ms": int((time.time() - t0) * 1000)},
                )
            return result

    def chat_stream(
        self,
        messages,
        *,
        route: str = "generate_fact",
        max_tokens: int = 4096,
        temperature: float = 0.3,
        timeout: float = 180,
        prompt_client=None,
    ) -> Iterator[dict]:
        """流式对话。事件: {"type":"reasoning"|"content","text"}，末条 {"type":"usage","usage"}。

        注意：流式请求不重试（§13.2 由调用方兜底）。
        生成器在迭代期间持有 span 上下文（§13.5），调用方在单线程内消费即正确归属。
        prompt_client 为 Langfuse 提示词客户端（span 版本联动，可选）。
        """
        provider, model = self._route(route)
        cm, live = _lf_span(route, "generation", input=messages, model=model,
                            metadata={"provider": provider, "stream": True},
                            prompt=prompt_client)
        t0 = time.time()
        with cm as gen:
            acc: List[str] = []
            usage = None
            try:
                for ev in self._chat_stream_impl(messages, route=route, max_tokens=max_tokens,
                                                 temperature=temperature, timeout=timeout):
                    if ev["type"] == "content":
                        acc.append(ev["text"])
                    elif ev["type"] == "usage":
                        usage = ev["usage"]
                    yield ev
            except BaseException as e:  # noqa: BLE001 — 失败标注后原样上抛
                if live:
                    _safe_update(gen, level="ERROR",
                                 status_message=f"{type(e).__name__}: {e}"[:500])
                raise
            if live:
                _safe_update(
                    gen,
                    output="".join(acc),
                    usage_details={"input": usage.prompt_tokens,
                                   "output": usage.completion_tokens,
                                   "total": usage.total_tokens} if usage else None,
                    metadata={"provider": provider, "stream": True,
                              "reasoning_tokens": usage.reasoning_tokens if usage else 0,
                              "latency_ms": int((time.time() - t0) * 1000)},
                )

    def _chat_stream_impl(
        self,
        messages,
        *,
        route: str = "generate_fact",
        max_tokens: int = 4096,
        temperature: float = 0.3,
        timeout: float = 180,
    ) -> Iterator[dict]:
        """流式请求实现体（无追踪，由 chat_stream 包装）。"""
        provider, model = self._route(route)
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        r = requests.post(
            f"{self._chat_bases[provider]}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self._keys[provider]}"},
            timeout=timeout,
            stream=True,
        )
        if r.status_code != 200:
            raise APIError(provider, r.status_code, r.json() if r.content else {})
        usage = None
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("reasoning_content"):
                yield {"type": "reasoning", "text": delta["reasoning_content"]}
            if delta.get("content"):
                yield {"type": "content", "text": delta["content"]}
            if chunk.get("usage"):
                usage = Usage.from_response(chunk["usage"])
        if usage:
            yield {"type": "usage", "usage": usage}

    # ---------- Embedding（百炼） ----------

    def embed(self, texts: List[str], *, model: Optional[str] = None, timeout: float = 60) -> List[List[float]]:
        """批量向量化，返回与输入顺序对齐的向量列表。"""
        model = model or config.MODELS["embedding"]["model"]
        t0 = time.time()
        cm, live = _lf_span("embedding", "embedding", model=model,
                            metadata={"n": len(texts)})
        with cm as gen:
            try:
                payload = {"model": model, "input": {"texts": texts}}
                body = self._post(
                    "dashscope",
                    f"{config.DASHSCOPE_BASE}/api/v1/services/embeddings/text-embedding/text-embedding",
                    payload,
                    timeout,
                    label="embedding",
                )
                embs = {e["text_index"]: e["embedding"] for e in body["output"]["embeddings"]}
                result = [embs[i] for i in range(len(texts))]
            except BaseException as e:  # noqa: BLE001
                if live:
                    _safe_update(gen, level="ERROR",
                                 status_message=f"{type(e).__name__}: {e}"[:500])
                raise
            if live:
                _safe_update(gen, metadata={"n": len(texts),
                                            "dim": len(result[0]) if result else 0,
                                            "latency_ms": int((time.time() - t0) * 1000)})
            return result

    # ---------- Rerank（百炼） ----------

    def rerank(self, query: str, documents: List[str], *, model: Optional[str] = None, timeout: float = 60) -> List[dict]:
        """精排打分，返回 [{"index", "score"}]，按相关度降序。"""
        model = model or config.MODELS["rerank"]["model"]
        t0 = time.time()
        cm, live = _lf_span("rerank", "retriever", input=query, model=model,
                            metadata={"n_docs": len(documents)})
        with cm as gen:
            try:
                payload = {"model": model, "input": {"query": query, "documents": documents}}
                body = self._post(
                    "dashscope",
                    f"{config.DASHSCOPE_BASE}/api/v1/services/rerank/text-rerank/text-rerank",
                    payload,
                    timeout,
                    label="rerank",
                )
                results = sorted(
                    ({"index": r["index"], "score": r["relevance_score"]} for r in body["output"]["results"]),
                    key=lambda x: -x["score"],
                )
            except BaseException as e:  # noqa: BLE001
                if live:
                    _safe_update(gen, level="ERROR",
                                 status_message=f"{type(e).__name__}: {e}"[:500])
                raise
            if live:
                _safe_update(gen, output=[{"index": r["index"], "score": r["score"]}
                                          for r in results[:10]],
                             metadata={"n_docs": len(documents),
                                       "latency_ms": int((time.time() - t0) * 1000)})
            return results


# 模块级单例
client = LLMClient()
