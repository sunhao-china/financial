# -*- coding: utf-8 -*-
"""轻量 Agent 循环 + 钩子机制（大纲 §6.1）+ P4 生成与验证关（§6.3/§11）。

- 拒答意图先行拦截（§4 guard 正则）：realtime/investment_advice → 固定话术，零 LLM 成本
- 单轮快路径：检索 → 自检充分 → 组装上下文 → 生成（DeepSeek，按意图路由）
- 慢路径：自检不足 → 生成补充查询 → 再检索 → 合并去重 → 再自检（补检轮数 ≤ config.MAX_REFILL_ROUNDS）
- 钩子：pre_step / post_step / pre_reply，返回 {allow, reason, action}，
  action ∈ {continue, skip, terminate, fallback}（skip 对应 §6.1「丢弃该步」）
- 验证关（P4，§6.3）：生成后强制闸门——pass 直接回复；downgrade 附可靠性说明；
  fail → 重生成 1 次 → 仍失败 → VERIFY_FAIL 降级话术 + 相关片段
- S7 会话记忆（P4，§11）：问答结束组装调用路径快照写入 SessionStore（7 天排障窗口）
- 响应有界：最坏情况 = 终止循环 → 用已有结果生成 → 验证失败走降级话术
"""
from __future__ import annotations

import re
import secrets
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from corpus import generation, verify_gate
from corpus.retrieval import HybridRetriever, S2, SearchHit
from corpus.session_store import SessionStore
from prompts.guard import quick_guard
from prompts.refusals import (CHITCHAT, INVESTMENT_ADVICE, LLM_UNAVAILABLE,
                              NO_RESULT, REALTIME, VERIFY_FAIL)
from schemas.models import (Message, S6Context, S6Reference, S7Cost, S7RetrievalRound,
                            S7RoundChunk, S7SelfCheck, S7Snapshot, S7Verification)

_NORM = re.compile(r"[^一-鿿\w]")


def new_query_id() -> str:
    """P5：query_id 在 run() 开头生成，贯穿 S7 快照 / S8 日志 / S9 SSE 信封。"""
    return f"q_{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"

# LLM 意图判定为拒答类 → 固定话术（§6.2/§13.4，不检索不生成）
REFUSAL_INTENTS = {
    "realtime": REALTIME,
    "investment_advice": INVESTMENT_ADVICE,
    "chitchat": CHITCHAT,
}

# 对话记忆类问题（§14.5 记忆层次修复）：「我上一个问题问了什么」被意图模型判为 chitchat，
# 但应从会话历史回答而非闲聊话术。纯规则识别 + 回显，零 LLM 成本。
_MEM_HINTS = ("上个", "上一个", "上一条", "刚才", "刚刚", "之前", "此前", "前面", "上次", "上一轮")
_MEM_ASKS = ("问了什么", "问的什么", "问过什么", "问了啥", "问过啥",
             "问题是什么", "什么问题", "聊了什么", "聊过什么")


def _ask_about_history(q: str) -> bool:
    """问题是否在询问会话历史本身。仅在 chitchat 分支内调用（金融追问走正常路径）。"""
    return any(h in q for h in _MEM_HINTS) and any(a in q for a in _MEM_ASKS)


def _history_answer(ctx: "AgentContext") -> str:
    """从会话历史回显最近一个用户问题；无历史时提示这是首问。"""
    user_qs = []
    for m in ctx.history or []:
        if isinstance(m, dict):
            if m.get("role") == "user" and m.get("content"):
                user_qs.append(m["content"].strip())
        elif getattr(m, "role", None) == "user" and getattr(m, "content", None):
            user_qs.append(m.content.strip())
    if not user_qs:
        return "这是我们对话中的第一个问题，还没有「上一个问题」可以回答。"
    return f"您上一个问题是：「{user_qs[-1]}」"


def normalize(q: str) -> str:
    """查询归一化（§6.1 重复判定）：只保留中英文数字，去空白。"""
    return _NORM.sub("", q).lower()


@dataclass
class HookDecision:
    allow: bool = True
    reason: str = ""
    action: str = "continue"   # continue / skip / terminate / fallback


@dataclass
class AgentContext:
    """会话级上下文（§6.1 post_step 写入 S7 路径快照）。"""
    question: str
    history: Optional[List[dict]] = None
    rounds: int = 0                      # 补检轮数
    cost: int = 0                        # 累计 token 成本（prompt+completion，§13.1）
    queries_tried: List[str] = field(default_factory=list)
    expanded: Optional[S2] = None
    chunks: List[SearchHit] = field(default_factory=list)   # 合并去重后的检索片段
    retrieval_rounds: List[dict] = field(default_factory=list)  # 每轮 hits（S7 retrieval）
    self_check: Optional[dict] = None    # 最近一次 S3
    generated: Optional[generation.Generated] = None   # 最近一次生成（S7 citations）
    verify: Optional[verify_gate.VerifyResult] = None  # 最近一次验证关结果
    steps: List[dict] = field(default_factory=list)         # S7 路径快照
    answer: str = ""
    fallback: str = ""


class BuiltinHooks:
    """§6.1 强制规则（代码级限制，不依赖提示词）。"""

    def __init__(self, token_budget: int = 50_000):
        self.token_budget = token_budget

    def pre_step(self, ctx: AgentContext, step: str, payload: dict) -> HookDecision:
        if ctx.rounds >= config.MAX_REFILL_ROUNDS and step == "retrieve_followup":
            return HookDecision(False, f"补检轮数 {ctx.rounds} 已达上限 {config.MAX_REFILL_ROUNDS}",
                                "terminate")
        if ctx.cost > self.token_budget:
            return HookDecision(False, f"token 成本 {ctx.cost} 超预算 {self.token_budget}",
                                "terminate")
        if step == "retrieve_followup":
            for q in payload.get("queries", []):
                if normalize(q) in {normalize(t) for t in ctx.queries_tried}:
                    return HookDecision(False, f"查询重复：{q}", "skip")
        return HookDecision(True)

    def post_step(self, ctx: AgentContext, step: str, result: dict) -> HookDecision:
        ctx.steps.append({"step": step, "ts": round(time.time(), 2), "ok": result.get("ok"),
                          "cost": result.get("cost", 0), "n": result.get("n", 0)})
        return HookDecision(True)

    def pre_reply(self, ctx: AgentContext, answer: str) -> HookDecision:
        if not ctx.chunks:
            return HookDecision(False, "无检索片段，走空结果话术", "fallback")
        return HookDecision(True)


@dataclass
class AgentAnswer:
    question: str
    answer: str = ""
    intent: str = ""
    chunks: List[SearchHit] = field(default_factory=list)
    context: dict = field(default_factory=dict)   # S6
    rounds: int = 0
    cost: int = 0
    path: List[dict] = field(default_factory=list)
    fallback: str = ""                       # 非空 = 走了拒答/降级路径（展示文本）
    query_id: str = ""
    generated_by_stub: bool = False
    timings: dict = field(default_factory=dict)        # P5：S9 done 事件携带
    verification: Optional[dict] = None                # P5：S9 done 事件携带
    citations: List[str] = field(default_factory=list) # P5：S9 done 事件携带


class SearchAgent:
    """受限两段式检索 Agent。generate_fn 注入覆盖内置生成（评测/降级用）。"""

    def __init__(self, retriever: HybridRetriever, hooks: Optional[BuiltinHooks] = None,
                 generate_fn: Optional[Callable] = None, store: Optional[SessionStore] = None):
        self.ret = retriever
        self.hooks = hooks or BuiltinHooks()
        self.generate_fn = generate_fn
        self.store = store

    # ---------- 内部步骤 ----------

    def _expand(self, ctx: AgentContext):
        try:
            from llm_client import client
            from prompts import remote
            from prompts.intent_expand import build_intent_messages
            r = client.chat(build_intent_messages(ctx.question, ctx.history),
                            route="intent", json_mode=True, max_tokens=1000,
                            prompt_client=remote.get_prompt("intent_expand"))
            d = client.parse_json_content(r.content)
            ctx.expanded = S2(intent=d.get("intent", "fact_qa"), confidence=d.get("confidence", 0),
                              entities=d.get("entities", []), queries=d.get("queries", []),
                              keywords=d.get("keywords", []))
            ctx.cost += r.usage.total_tokens
            ctx.queries_tried.append(ctx.question)
            return {"ok": True, "cost": r.usage.total_tokens, "n": len(ctx.expanded.queries)}
        except Exception as e:  # noqa: BLE001
            ctx.expanded = S2()
            return {"ok": False, "cost": 0, "n": 0, "err": str(e)}

    def _retrieve(self, ctx: AgentContext, s2: Optional[S2]):
        hits = self.ret.retrieve(ctx.question, s2=s2, rerank=True, n=20)
        ctx.chunks = self._merge(ctx.chunks, hits)
        ctx.retrieval_rounds.append({"chunks": hits,
                                     "source_count": len({h.doc_id for h in hits})})
        return {"ok": len(hits) > 0, "n": len(hits)}

    def _self_check(self, ctx: AgentContext):
        try:
            from llm_client import client
            from prompts import remote
            from prompts.self_check import build_self_check_messages
            r = client.chat(build_self_check_messages(
                ctx.question, ctx.queries_tried, ctx.chunks[:5]),
                route="self_check", json_mode=True, max_tokens=800,
                prompt_client=remote.get_prompt("self_check"))
            d = client.parse_json_content(r.content)
            ctx.self_check = d
            ctx.cost += r.usage.total_tokens
            return {"ok": True, "cost": r.usage.total_tokens, "sufficient": d.get("sufficient")}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "err": str(e)}

    def _followup(self, ctx: AgentContext):
        followups = [q for q in (ctx.self_check or {}).get("followup_queries", [])
                     if normalize(q) not in {normalize(t) for t in ctx.queries_tried}]
        if not followups:
            return {"ok": True, "n": 0}
        hits = []
        for q in followups[:2]:   # §10.3：followup 最多 2 个
            hits.extend(self.ret.retrieve(q, s2=S2(queries=[q]), rerank=True, n=20))
        ctx.chunks = self._merge(ctx.chunks, hits)
        ctx.retrieval_rounds.append({"chunks": hits,
                                     "source_count": len({h.doc_id for h in hits})})
        ctx.queries_tried.extend(followups)
        return {"ok": True, "n": len(followups)}

    @staticmethod
    def _merge(old: List[SearchHit], new: List[SearchHit]) -> List[SearchHit]:
        """合并去重（按 chunk id，保留最高分），§6.1 慢路径。"""
        by_id = {h.id: h for h in old}
        for h in new:
            if h.id not in by_id or h.score > by_id[h.id].score:
                by_id[h.id] = h
        return sorted(by_id.values(), key=lambda x: -x.score)

    def _assemble(self, ctx: AgentContext) -> S6Context:
        """S6 上下文（§9.4）：生成 LLM 的输入，ref_no 从 1 连续编号。
        history 取末 6 条（3 轮，P5 多轮指代消解；§14 裁剪已在上游完成）。"""
        refs = []
        for i, h in enumerate(ctx.chunks[:20]):
            refs.append(S6Reference(
                ref_no=i + 1, chunk_id=h.id, doc_id=h.doc_id,
                quality_score=float(h.quality),   # P6：随 SearchHit 由 Qdrant payload 携带
                text=h.text))
        history = [Message(role=m["role"], content=m["content"])
                   for m in (ctx.history or [])[-6:]
                   if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
        return S6Context(schema_version=config.SCHEMA_VERSION, question=ctx.question,
                         references=refs, history=history)

    # ---------- 生成 + 验证关（P4，§6.3/§11） ----------

    def _generate(self, ctx: AgentContext, s6: S6Context, stream: bool,
                  on_delta: Optional[Callable], on_reasoning: Optional[Callable] = None,
                  route: Optional[str] = None) -> tuple:
        """一次生成。返回 (Generated, 毫秒耗时)；API 失败抛异常（§13.3 由 run 兜底）。
        route 显式传入可覆盖默认路由（验证关降级链升级用）。
        on_reasoning 流式推理增量回调（P5 SSE reasoning 事件源）。"""
        intent = ctx.expanded.intent if ctx.expanded else "fact_qa"
        t1 = time.time()
        gen = generation.generate(s6, intent=intent, stream=stream, on_delta=on_delta,
                                  on_reasoning=on_reasoning, route=route)
        dt = int((time.time() - t1) * 1000)
        ctx.cost += gen.cost_tokens.get("total_tokens", 0)
        ctx.generated = gen
        return gen, dt

    def _verify(self, ctx: AgentContext, question: str, answer: str,
                refs: List[S6Reference]) -> tuple:
        """一次验证。返回 (VerifyResult, 毫秒耗时)。"""
        t1 = time.time()
        vr = verify_gate.verify_answer(question, answer, refs)
        dt = int((time.time() - t1) * 1000)
        ctx.cost += vr.cost_tokens.get("total_tokens", 0)
        ctx.verify = vr
        return vr, dt

    # ---------- S7 会话记忆（§11） ----------

    def _save_snapshot(self, ctx: AgentContext, ans: AgentAnswer, session_id: str,
                       timings: dict, verify_info: Optional[dict]):
        # P5：S9 done 事件组装字段，无论是否落快照都回填
        ans.timings = dict(timings)
        ans.verification = dict(verify_info) if verify_info else None
        ans.citations = list(ctx.generated.citations) if ctx.generated else []
        if self.store is None:
            return
        query_id = ans.query_id
        rounds = [S7RetrievalRound(
            round=i + 1,
            chunks=[S7RoundChunk(id=h.id, doc_id=h.doc_id, score=h.score)
                    for h in rnd["chunks"]],
            source_count=rnd["source_count"])
            for i, rnd in enumerate(ctx.retrieval_rounds)]
        by_step = {}
        for s in ctx.steps:
            by_step[s["step"]] = by_step.get(s["step"], 0) + s.get("cost", 0)
        status = "degraded" if ans.fallback else (
            "verified" if verify_info and verify_info.get("verdict") == "pass" else "answering")
        snap = S7Snapshot(
            query_id=query_id, session_id=session_id or "",
            status=status, question=ctx.question,
            intent=ctx.expanded.intent if ctx.expanded else None,
            entities=ctx.expanded.entities if ctx.expanded else [],
            queries=ctx.queries_tried,
            keywords=ctx.expanded.keywords if ctx.expanded else [],
            retrieval=rounds,
            self_check=S7SelfCheck(sufficient=bool((ctx.self_check or {}).get("sufficient")),
                                   rounds=ctx.rounds + 1) if ctx.self_check else None,
            verification=S7Verification(
                verdict=verify_info.get("verdict"),
                single_source_items=verify_info.get("single_source_items", [])
            ) if verify_info else None,
            answer=ans.answer or ans.fallback,
            citations=list(ctx.generated.citations) if ctx.generated else [],
            timings_ms=timings,
            cost_tokens=S7Cost(total=ctx.cost, by_step=by_step))
        self.store.save(snap)

    # ---------- 主循环 ----------

    def run(self, question: str, history: Optional[List[dict]] = None,
            session_id: str = "", verify: bool = True,
            stream: bool = False, on_delta: Optional[Callable] = None,
            on_reasoning: Optional[Callable] = None,
            query_id: str = "", user: str = "") -> AgentAnswer:
        """执行一次问答。verify=False 跳过验证关（消融对比用）。
        on_delta/on_reasoning 流式回调（P5 SSE）；query_id 缺省自动生成，
        服务端预生成以保证 S7/S8/S9 全链路同一 id（P5）。
        user 为登录用户名（§13.5 Langfuse User 列，空=未登录/CLI）。
        §13.5：整次问答包在 Langfuse trace 中（trace_id 由 query_id 派生）；
        追踪未启用时 llm_trace 为 no-op。"""
        qid = query_id or new_query_id()
        from llm_client import llm_trace  # 惰性导入（保持模块轻依赖，§13.5）
        with llm_trace(qid, question, session_id or "", user or "") as trace:
            ans = self._run_impl(question, history, session_id, verify, stream,
                                 on_delta, on_reasoning, qid)
            trace.update(output=ans.answer or ans.fallback,
                         metadata={"intent": ans.intent,
                                   "steps": ",".join(s.get("step", "") for s in ans.path),
                                   "cost_tokens": ans.cost})
            return ans

    def _run_impl(self, question: str, history: Optional[List[dict]] = None,
                  session_id: str = "", verify: bool = True,
                  stream: bool = False, on_delta: Optional[Callable] = None,
                  on_reasoning: Optional[Callable] = None,
                  query_id: str = "") -> AgentAnswer:
        """run() 实现体（query_id 已由 run() 生成；trace 包裹在 run() 层）。"""
        ctx = AgentContext(question=question, history=history)
        ans = AgentAnswer(question=question, query_id=query_id or new_query_id())
        timings = {}

        # 0. 拒答意图先行拦截（§4 guard 正则高置信拦截，零 LLM 成本）
        g = quick_guard(question)
        if g:
            ans.intent = g
            ans.fallback = INVESTMENT_ADVICE if g == "investment_advice" else REALTIME
            self.hooks.post_step(ctx, "guard_refuse", {"ok": True, "n": 0, "intent": g})
            ans.path = ctx.steps
            self._save_snapshot(ctx, ans, session_id, timings, None)
            return ans

        # 1. 查询扩展
        d = self.hooks.pre_step(ctx, "expand", {})
        if d.action == "fallback":
            ans.fallback = "抱歉，我暂时无法处理这个问题。"
            ans.path = ctx.steps
            self._save_snapshot(ctx, ans, session_id, timings, None)
            return ans
        if d.allow:
            t1 = time.time()
            res = self._expand(ctx)
            timings["intent"] = int((time.time() - t1) * 1000)
            self.hooks.post_step(ctx, "expand", res)
        else:
            ctx.expanded = S2()
            self.hooks.post_step(ctx, "expand", {"ok": False, "err": d.reason})

        # 1.5 LLM 意图判定为拒答类（§6.2/§13.4）：固定话术，不检索不生成。
        # 例外（§14.5）：问会话历史本身（「我上一个问题问了什么」）从 history 直接回显
        if ctx.expanded and ctx.expanded.intent in REFUSAL_INTENTS:
            ans.intent = ctx.expanded.intent
            if ctx.expanded.intent == "chitchat" and _ask_about_history(question):
                ans.answer = _history_answer(ctx)
                self.hooks.post_step(ctx, "refuse", {"ok": True, "n": 0, "history_echo": True})
            else:
                ans.fallback = REFUSAL_INTENTS[ctx.expanded.intent]
                self.hooks.post_step(ctx, "refuse", {"ok": True, "n": 0})
            ans.path = ctx.steps
            ans.cost = ctx.cost
            self._save_snapshot(ctx, ans, session_id, timings, None)
            return ans

        # 2. 首轮检索（快路径）
        d = self.hooks.pre_step(ctx, "retrieve", {})
        if not d.allow and d.action == "fallback":
            ans.fallback = "抱歉，我暂时无法处理这个问题。"
            ans.path = ctx.steps
            self._save_snapshot(ctx, ans, session_id, timings, None)
            return ans
        if d.allow:
            t1 = time.time()
            self._retrieve(ctx, ctx.expanded)
            timings["retrieval"] = int((time.time() - t1) * 1000)
            self.hooks.post_step(ctx, "retrieve", {"ok": True, "n": len(ctx.chunks)})

        # 3. 自检
        d = self.hooks.pre_step(ctx, "self_check", {})
        if d.allow and ctx.chunks:
            res = self._self_check(ctx)
            self.hooks.post_step(ctx, "self_check", res)

        # 4. 慢路径：自检不足 → 补检（轮数上限由 pre_step 强制）
        while ctx.chunks and (ctx.self_check or {}).get("sufficient") is False:
            d = self.hooks.pre_step(ctx, "retrieve_followup",
                                    {"queries": ctx.self_check.get("followup_queries", [])})
            if not d.allow:
                self.hooks.post_step(ctx, "retrieve_followup", {"ok": False, "err": d.reason})
                break
            ctx.rounds += 1
            t1 = time.time()
            res = self._followup(ctx)
            timings["retrieval"] = timings.get("retrieval", 0) + int((time.time() - t1) * 1000)
            self.hooks.post_step(ctx, "retrieve_followup", res)
            res = self._self_check(ctx)
            self.hooks.post_step(ctx, "self_check", res)

        # 5. 组装 S6 → 生成（P4 真实链路，§11；空检索跳过组装，pre_reply 走 NO_RESULT）
        s6 = self._assemble(ctx) if ctx.chunks else None
        if s6:
            self.hooks.post_step(ctx, "assemble", {"ok": True, "n": len(s6.references)})
        answer, verify_info = "", None
        if s6 and s6.references:
            if self.generate_fn:      # 注入的生成器（评测/降级用），跳过验证关
                answer = self.generate_fn(ctx.question,
                                         {"chunks": [{"chunk_id": r.chunk_id, "doc_id": r.doc_id,
                                                      "text": r.text, "score": 0} for r in s6.references]})
                ans.generated_by_stub = True
            else:
                try:
                    gen, dt = self._generate(ctx, s6, stream, on_delta, on_reasoning)
                    timings["generate"] = dt
                    answer = gen.answer
                    self.hooks.post_step(ctx, "generate",
                                         {"ok": True, "cost": gen.cost_tokens.get("total_tokens", 0)})
                except Exception as e:  # noqa: BLE001 — §13.3 LLM 不可用：片段原文兜底
                    self.hooks.post_step(ctx, "generate", {"ok": False, "err": str(e)})
                    ans.fallback = LLM_UNAVAILABLE + "\n\n" + "\n\n".join(
                        f"[{r.ref_no}] {r.text[:120]}…" for r in s6.references)
                    self._save_snapshot(ctx, ans, session_id, timings, None)
                    ans.intent = ctx.expanded.intent if ctx.expanded else ""
                    ans.chunks = ctx.chunks
                    ans.rounds = ctx.rounds
                    ans.cost = ctx.cost
                    ans.path = ctx.steps
                    return ans

                # 5.5 空答案兜底（重试后仍空）：不进验证关，直接资料不足话术（§13.2）
                if not answer.strip():
                    self.hooks.post_step(ctx, "generate", {"ok": False, "err": "content 空"})
                    ans.fallback = VERIFY_FAIL
                    ans.intent = ctx.expanded.intent if ctx.expanded else ""
                    ans.chunks = ctx.chunks
                    ans.context = {"schema_version": s6.schema_version, "question": s6.question,
                                   "references": [r.model_dump() for r in s6.references]}
                    ans.rounds = ctx.rounds
                    ans.cost = ctx.cost
                    ans.path = ctx.steps
                    self._save_snapshot(ctx, ans, session_id, timings, None)
                    return ans

                # 6. 验证关（§6.3 强制闸门；verify=False 跳过，供消融）
                if verify:
                    vr, dt = self._verify(ctx, ctx.question, answer, s6.references)
                    timings["verify"] = dt
                    self.hooks.post_step(ctx, "verify",
                                         {"ok": vr.parse_ok,
                                          "cost": vr.cost_tokens.get("total_tokens", 0),
                                          "verdict": vr.verdict})
                    verify_info = {"verdict": vr.verdict,
                                   "single_source_items": vr.single_source_items}
                    if vr.verdict == "fail":
                        # 重生成 1 次（同上下文，同 flash 路由）→ 复验；仍失败走降级话术（§6.3）。
                        # P4 对照实验结论（2026-10-18）：flash judge 一致性与 v4-pro 持平且快
                        # 4.7 倍，全链路默认只用 flash，不再升级 v4-pro。
                        try:
                            gen2, dt2 = self._generate(ctx, s6, False, None)
                            timings["generate"] = timings.get("generate", 0) + dt2
                            self.hooks.post_step(ctx, "generate",
                                                 {"ok": True, "cost": gen2.cost_tokens.get("total_tokens", 0)})
                            vr2, dt2 = self._verify(ctx, ctx.question, gen2.answer, s6.references)
                            timings["verify"] += dt2
                            self.hooks.post_step(ctx, "verify",
                                                 {"ok": vr2.parse_ok,
                                                  "cost": vr2.cost_tokens.get("total_tokens", 0),
                                                  "verdict": vr2.verdict})
                            verify_info = {"verdict": vr2.verdict,
                                           "single_source_items": vr2.single_source_items}
                            if vr2.verdict in ("pass", "downgrade"):
                                answer = gen2.answer
                            else:
                                ans.fallback = VERIFY_FAIL
                        except Exception as e:  # noqa: BLE001
                            self.hooks.post_step(ctx, "generate", {"ok": False, "err": str(e)})
                            ans.fallback = VERIFY_FAIL
                    if verify_info["verdict"] == "downgrade":
                        answer += verify_gate.DOWNGRADE_NOTE

        # 7. pre_reply 强制关（§6.1：回复前必须通过；空结果 → NO_RESULT）
        d = self.hooks.pre_reply(ctx, answer)
        if not d.allow:
            ans.fallback = NO_RESULT
        elif ans.fallback:      # 验证关降级（VERIFY_FAIL）：话术即最终回复
            ans.answer = ans.fallback
        else:
            ans.answer = answer

        ans.intent = ctx.expanded.intent if ctx.expanded else ""
        ans.chunks = ctx.chunks
        if s6:
            ans.context = {"schema_version": s6.schema_version, "question": s6.question,
                           "references": [r.model_dump() for r in s6.references]}
        ans.rounds = ctx.rounds
        ans.cost = ctx.cost
        ans.path = ctx.steps
        # 8. S7 快照写入会话记忆（§11，7 天排障窗口）
        self._save_snapshot(ctx, ans, session_id, timings, verify_info)
        return ans


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ret = HybridRetriever()
    agent = SearchAgent(ret)
    for q in sys.argv[1:] or ["2021年全国综合保税区进出口值是多少"]:
        a = agent.run(q)
        print(f"\nQ: {a.question}\nintent={a.intent} rounds={a.rounds} cost={a.cost} "
              f"fallback={bool(a.fallback)}")
        print(a.answer[:300] or a.fallback[:300])
        for s in a.path:
            print(" ", s)
