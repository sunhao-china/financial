# -*- coding: utf-8 -*-
"""ServiceHooks（§13.5/§14）：BuiltinHooks 强制规则之上，把 agent 步骤事件分派到
SSE / S8 日志 / 指标。每请求一个实例（绑定该请求的 hub 与 query_id）。

分派规则：
- post_step(expand)  → SSE meta（S2 出站子集）；guard_refuse → meta（guard 正则意图）
- post_step(assemble)→ SSE sources（ctx.chunks 终稿 → S5Hit 列表）
- 全部 step         → S8 事件（step 名与 ctx.steps 对齐，query_id 贯穿）+ 指标采集
- pre_reply 拦截    → no_result 指标
"""
from __future__ import annotations

import time

from corpus.agent import BuiltinHooks
from corpus.retrieval import S2
from schemas.models import S7Cost, S9Meta, S9Sources

from server.sse import to_s5hits


class ServiceHooks(BuiltinHooks):
    def __init__(self, hub, s8, metrics, state, ret, query_id: str, token_budget: int = 50_000):
        super().__init__(token_budget)
        self.hub = hub
        self.s8 = s8
        self.metrics = metrics
        self.state = state
        self.ret = ret
        self.query_id = query_id
        self._t0 = {}          # step → 开始时间（pre_step 记，post_step 算 latency）
        self._meta_sent = False

    # ---------- 钩子 ----------

    def pre_step(self, ctx, step: str, payload: dict):
        self._t0[step] = time.monotonic()
        return super().pre_step(ctx, step, payload)

    def post_step(self, ctx, step: str, result: dict):
        decision = super().post_step(ctx, step, result)   # 先落 ctx.steps（S7 path）
        latency = int((time.monotonic() - self._t0.get(step, time.monotonic())) * 1000)

        # --- SSE 分派 ---
        if step == "expand":
            self._emit_meta(ctx.expanded)
        elif step == "guard_refuse":
            self._emit_meta(S2(intent=result.get("intent") or "fact_qa"))
        elif step == "assemble":
            self.hub.emit("sources", S9Sources(
                references=to_s5hits(ctx.chunks[:20], self.ret)))

        # --- 指标 ---
        if step in ("refuse", "guard_refuse"):
            self.metrics.inc("chat_refuse")
        if step == "verify" and "verdict" in result:
            if result["verdict"] == "fail":
                self.metrics.inc("verify_fail")
            else:
                self.metrics.inc("verify_pass")
        if step in ("retrieve", "retrieve_followup"):
            with self.ret._cache_lock:
                degraded = self.ret.vector_fail_streak > 0
            if degraded:
                self.metrics.inc("vector_degraded")
        cost = result.get("cost", 0)
        if cost:
            self.metrics.inc(f"token_total_{step}", cost)
        self.metrics.observe(f"step_latency_ms_{step}", latency)

        # --- S8 事件（§13.5，query_id 贯穿） ---
        if not result.get("ok"):
            level = "error" if step == "generate" else "warning"
        else:
            level = "info"
        payload = {k: result[k] for k in ("ok", "n", "cost", "err", "verdict", "sufficient")
                   if k in result}
        self.s8.emit(query_id=self.query_id, step=step, level=level, payload=payload,
                     latency_ms=latency,
                     cost_tokens=S7Cost(total=cost) if cost else None)
        return decision

    def pre_reply(self, ctx, answer: str):
        d = super().pre_reply(ctx, answer)
        if not d.allow:
            self.metrics.inc("no_result")
        return d

    # ---------- meta 事件（S9 时序契约：done 前必已发） ----------

    def _emit_meta(self, s2: S2) -> None:
        self.hub.emit("meta", S9Meta(intent=s2.intent or "fact_qa",
                                     entities=list(s2.entities or []),
                                     queries=list(s2.queries or []),
                                     keywords=list(s2.keywords or [])))
        self._meta_sent = True

    def ensure_meta(self) -> None:
        """兜底：expand 被钩子跳过等罕见路径，done/error 前补发默认 meta。"""
        if not self._meta_sent:
            self._emit_meta(S2())
