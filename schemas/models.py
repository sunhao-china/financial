# -*- coding: utf-8 -*-
"""全链路 JSON 契约的 pydantic 模型（大纲 §9）。

- S1 向量库入库记录（§8）：P0 定稿
- S2 意图识别+查询扩展输出（§4）
- S3 信息自检输出（§6.1）
- S4 答案校验输出（§6.3）
- S5 检索结果条目（§9.3）
- S6 喂给生成 LLM 的上下文（§9.4）
- S7 会话记忆·调用路径快照（§11）
- S8 日志/链路事件（§13.5）
- S9 API 请求/响应与 SSE 事件（§14）

原则（§9.1）：每个结构顶层必带 schema_version；解析/写入前强制校验，不合规拒绝流转。
LLM 输出类（S2/S3/S4）与 API 类（S9）容忍多余字段（extra=ignore，防 LLM/客户端加料误伤），
其中 S2/S3/S4 的 schema_version 为必填（提示词强制，缺失重试 1 次后走兜底）；
代码组装类（S5~S8）严格模式（extra=forbid，装配错误当场暴露）。
"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

import config

SV = config.SCHEMA_VERSION

Intent = Literal["fact_qa", "analysis", "comparison", "realtime", "investment_advice", "chitchat"]
Verdict = Literal["pass", "downgrade", "fail"]
RankSource = Literal["vector", "bm25", "fusion"]
Role = Literal["user", "assistant", "system"]


class Message(BaseModel):
    """通用对话消息（S6 history / S9 请求）。"""
    role: Role
    content: str


# ---------- S1 向量库入库记录（§8） ----------

class S1Metadata(BaseModel):
    """S1 metadata：溯源与版本字段（§8）。"""
    doc_id: str
    source_file: str
    industry_type: str
    quality_score: float
    perplexity: float
    num_words: int
    chunk_index: int
    chunk_count: int
    char_start: int
    char_end: int
    chunking_method: str
    chunking_version: str = "v1.0"
    embedding_model: str = "text-embedding-v3"
    embedding_dim: int = 1024
    data_version: str
    ingested_at: str


class S1Record(BaseModel):
    """S1：向量库入库记录，一 chunk 一条（§8）。"""
    id: str = Field(..., description="唯一 chunk id：{source_file}_{doc_id}_c{chunk_index}")
    schema_version: str = SV
    text: str = Field(..., description="chunk 原文片段（即转向量前的文本）")
    embedding: List[float] = Field(..., description="向量，写入向量库前生成")
    metadata: S1Metadata


# ---------- S2 意图识别+查询扩展输出（§4） ----------

class S2IntentExpand(BaseModel):
    """S2：意图识别+查询扩展输出，qwen-plus 一次调用（§4/§9.2）。"""
    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(..., description="必填，固定 1.0.0")
    intent: Intent
    confidence: float = Field(..., ge=0, le=1, description="意图置信度 0~1")
    entities: List[str] = Field(default_factory=list, max_length=6, description="关键实体：公司/政策/指标/时间")
    queries: List[str] = Field(default_factory=list, max_length=5,
                               description="检索变体 3~5 个；realtime/investment_advice/chitchat 为空")
    keywords: List[str] = Field(default_factory=list, max_length=8, description="金融实体词（BM25 检索用）")


# ---------- S3 信息自检输出（§6.1） ----------

class S3SelfCheck(BaseModel):
    """S3：信息自检输出（§6.1/§10.3），Agent 循环慢路径判据。"""
    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(..., description="必填，固定 1.0.0")
    sufficient: bool
    missing: List[str] = Field(default_factory=list, description="不足时逐条列出缺失信息点")
    followup_queries: List[str] = Field(default_factory=list, max_length=2,
                                        description="补充查询 ≤2 个，不得与已尝试查询重复")


# ---------- S4 答案校验输出（§6.3） ----------

class S4Verify(BaseModel):
    """S4：答案校验输出（§6.3/§10.4），pre_reply 钩子强制关口。"""
    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(..., description="必填，固定 1.0.0")
    verdict: Verdict = Field(..., description="pass=正常回复；downgrade=附可靠性说明回复；"
                                             "fail=重生成一次，仍失败走降级话术")
    unverified_claims: List[str] = Field(default_factory=list, description="无法在资料原文定位的数字/日期结论")
    single_source_items: List[str] = Field(default_factory=list, description="仅单一 doc_id 支撑的结论")
    notes: str = ""


# ---------- S5 检索结果条目（§9.3） ----------

class S5Hit(BaseModel):
    """S5：检索结果条目（§9.3），检索服务所有路径的统一出口。"""
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SV
    chunk_id: str
    doc_id: str
    text: str
    score: float = Field(..., ge=0, le=1, description="融合后归一化分数")
    rank_source: RankSource
    quality_score: float
    chunk_index: int
    chunk_count: int
    # §14.5：来源文章文献名（正文首句规则提炼）；缺 title_map 时为 None，前端回退「文章 #id」
    doc_title: Optional[str] = None


# ---------- S6 生成上下文（§9.4） ----------

class S6Reference(BaseModel):
    """S6 引用块：ref_no 与生成提示词中的【资料n】严格对应。"""
    model_config = ConfigDict(extra="forbid")

    ref_no: int = Field(..., ge=1)
    chunk_id: str
    doc_id: str
    quality_score: float
    text: str


class S6Context(BaseModel):
    """S6：喂给生成 LLM 的上下文（§9.4），组装 prompt 前校验。"""
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SV
    question: str
    references: List[S6Reference] = Field(..., min_length=1)
    history: List[Message] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ref_no_continuous(self):
        nos = [r.ref_no for r in self.references]
        if sorted(nos) != list(range(1, len(nos) + 1)):
            raise ValueError(f"ref_no 必须从 1 连续编号，实际 {nos}")
        return self


# ---------- S7 会话记忆·调用路径快照（§11） ----------

class S7RoundChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    doc_id: str
    score: float


class S7RetrievalRound(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round: int = Field(..., ge=1)
    chunks: List[S7RoundChunk] = Field(default_factory=list)
    source_count: int = 0          # 去重后不同 doc_id 数


class S7SelfCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sufficient: bool
    rounds: int = 1


class S7Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: Optional[Verdict] = None
    single_source_items: List[str] = Field(default_factory=list)


class S7Cost(BaseModel):
    """token 成本（§13.1）：total + 按环节拆分。"""
    model_config = ConfigDict(extra="forbid")
    total: int = 0
    by_step: Dict[str, int] = Field(default_factory=dict)


class S7Snapshot(BaseModel):
    """S7：会话记忆·调用路径快照（§11），post_step 钩子写入，7 天排障窗口。"""
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SV
    query_id: str
    session_id: str
    status: Literal["answering", "verified", "degraded"] = "answering"
    question: str
    intent: Optional[Intent] = None
    entities: List[str] = Field(default_factory=list)
    queries: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    retrieval: List[S7RetrievalRound] = Field(default_factory=list)
    self_check: Optional[S7SelfCheck] = None
    verification: Optional[S7Verification] = None
    answer: str = ""
    citations: List[str] = Field(default_factory=list)
    timings_ms: Dict[str, int] = Field(default_factory=dict)
    cost_tokens: S7Cost = Field(default_factory=S7Cost)


# ---------- S8 日志/链路事件（§13.5） ----------

class S8Event(BaseModel):
    """S8：日志/链路事件（§13.5），落盘前校验；query_id 贯穿全链路。"""
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SV
    ts: str = Field(..., description="ISO 8601 时间戳")
    query_id: str
    step: str = Field(..., description="intent/retrieval/self_check/generate/verify/error …")
    level: Literal["debug", "info", "warning", "error"] = "info"
    payload: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[int] = None
    cost_tokens: Optional[S7Cost] = None


# ---------- S9 API 请求/响应与 SSE 事件（§14） ----------

class S9ChatRequest(BaseModel):
    """S9：POST /chat 请求体（§14）。"""
    model_config = ConfigDict(extra="ignore")

    schema_version: str = SV
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    history: List[Message] = Field(default_factory=list)


class S9Meta(BaseModel):
    """S9 meta 事件：意图识别完成（S2 出站子集）。"""
    intent: Intent
    entities: List[str] = Field(default_factory=list)
    queries: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


class S9Delta(BaseModel):
    """S9 reasoning/delta 事件：流式增量。"""
    delta: str


class S9Sources(BaseModel):
    """S9 sources 事件：最终引用来源列表。"""
    references: List[S5Hit]


class S9Done(BaseModel):
    """S9 done 事件：问答结束汇总。"""
    answer: str
    citations: List[str] = Field(default_factory=list)
    verification: Optional[S7Verification] = None
    timings_ms: Dict[str, int] = Field(default_factory=dict)
    cost_tokens: S7Cost = Field(default_factory=S7Cost)


class S9Error(BaseModel):
    """S9 error 事件。"""
    code: str
    message: str


_S9_PAYLOADS = {
    "meta": S9Meta, "reasoning": S9Delta, "delta": S9Delta,
    "sources": S9Sources, "done": S9Done, "error": S9Error,
}


class S9Event(BaseModel):
    """S9：SSE 事件信封（§14），data 按 event 类型双向校验。"""
    model_config = ConfigDict(extra="ignore")

    schema_version: str = SV
    query_id: str
    event: Literal["meta", "reasoning", "delta", "sources", "done", "error"]
    data: Dict[str, Any]
    ts: str = ""

    @model_validator(mode="after")
    def _check_payload(self):
        _S9_PAYLOADS[self.event].model_validate(self.data)
        return self


def make_sse(event: str, query_id: str, payload: BaseModel, ts: str = "") -> S9Event:
    """按事件类型校验 payload 后组装 SSE 事件（§9.2：写入与读取双向校验）。"""
    return S9Event(query_id=query_id, event=event, data=payload.model_dump(), ts=ts)


class S9SourcesResponse(BaseModel):
    """S9：GET /sources/{query} 响应（§14 检索调试接口）。"""
    model_config = ConfigDict(extra="ignore")

    schema_version: str = SV
    query: str
    results: List[S5Hit] = Field(default_factory=list)
