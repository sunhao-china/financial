# -*- coding: utf-8 -*-
"""S1~S9 契约离线校验（不调 API，大纲 §9.1 强制校验的单元测试）。

覆盖：合法样例必须通过、非法样例必须拒绝、S6 ref_no 连续性、S9 SSE payload
双向校验、LLM 输出 JSON 提取、意图兜底正则。
用法: python tests/test_schemas.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import ValidationError

from prompts.guard import quick_guard
from schemas.models import (
    S1Metadata, S1Record, S2IntentExpand, S3SelfCheck, S4Verify, S5Hit,
    S6Context, S6Reference, S7Snapshot, S8Event, S9ChatRequest, S9Delta,
    S9Event, S9SourcesResponse, make_sse,
)
from schemas.parsing import extract_json

PASS, FAIL = [], []


def ok(name, cond):
    (PASS if cond else FAIL).append(name)
    if not cond:
        print(f"  ✗ {name}")


def raises(name, fn, exc=ValidationError):
    try:
        fn()
    except exc:
        PASS.append(name)
    else:
        FAIL.append(name)
        print(f"  ✗ {name}（应拒绝但通过了）")


def test_s1():
    meta = S1Metadata(doc_id="d1", source_file="rank_00045", industry_type="金融_经济",
                      quality_score=4.1, perplexity=100, num_words=100, chunk_index=0,
                      chunk_count=1, char_start=0, char_end=100, chunking_method="m",
                      data_version="2026-10-16", ingested_at="2026-10-16T12:00:00+08:00")
    r = S1Record(id="rank_00045_d1_c0", text="t", embedding=[0.1, 0.2], metadata=meta)
    ok("S1 合法样例", r.schema_version == "1.0.0")
    raises("S1 缺 metadata 拒绝", lambda: S1Record(id="x", text="t", embedding=[]))


def test_s2():
    d = {"schema_version": "1.0.0", "intent": "fact_qa", "confidence": 0.95,
         "entities": ["综合保税区"], "queries": ["q1", "q2", "q3"], "keywords": ["k"]}
    m = S2IntentExpand.model_validate(d)
    ok("S2 合法样例", m.intent == "fact_qa")
    ok("S2 容忍 LLM 多余字段", S2IntentExpand.model_validate({**d, "extra_field": 1}).intent == "fact_qa")
    raises("S2 非法意图拒绝", lambda: S2IntentExpand.model_validate({**d, "intent": "whatever"}))
    raises("S2 置信度越界拒绝", lambda: S2IntentExpand.model_validate({**d, "confidence": 1.5}))
    raises("S2 queries 超 5 个拒绝", lambda: S2IntentExpand.model_validate({**d, "queries": ["q"] * 6}))
    raises("S2 缺 schema_version 拒绝", lambda: S2IntentExpand.model_validate(
        {k: v for k, v in d.items() if k != "schema_version"}))


def test_s3():
    d = {"schema_version": "1.0.0", "sufficient": False, "missing": ["缺 2021 年数据"],
         "followup_queries": ["2021年综合保税区进出口数据"]}
    ok("S3 合法样例", S3SelfCheck.model_validate(d).followup_queries[0].startswith("2021"))
    raises("S3 followup 超 2 个拒绝", lambda: S3SelfCheck.model_validate(
        {**d, "followup_queries": ["a", "b", "c"]}))
    raises("S3 缺 sufficient 拒绝", lambda: S3SelfCheck.model_validate(
        {k: v for k, v in d.items() if k != "sufficient"}))


def test_s4():
    d = {"schema_version": "1.0.0", "verdict": "downgrade",
         "unverified_claims": [], "single_source_items": ["结论：5.9万亿元仅资料2提及"]}
    ok("S4 合法样例", S4Verify.model_validate(d).verdict == "downgrade")
    raises("S4 非法 verdict 拒绝", lambda: S4Verify.model_validate({**d, "verdict": "maybe"}))


def test_s5():
    d = {"schema_version": "1.0.0", "chunk_id": "a_b_c0", "doc_id": "b", "text": "t",
         "score": 0.87, "rank_source": "fusion", "quality_score": 4.1,
         "chunk_index": 0, "chunk_count": 1}
    ok("S5 合法样例", S5Hit.model_validate(d).rank_source == "fusion")
    raises("S5 非法 rank_source 拒绝", lambda: S5Hit.model_validate({**d, "rank_source": "rrf"}))
    raises("S5 多余字段拒绝", lambda: S5Hit.model_validate({**d, "extra": 1}))


def test_s6():
    refs = [S6Reference(ref_no=i, chunk_id=f"c{i}", doc_id=f"d{i}", quality_score=4.0, text="t")
            for i in (1, 2, 3)]
    ok("S6 连续编号合法", S6Context(question="q", references=refs).references[2].ref_no == 3)
    raises("S6 编号不连续拒绝", lambda: S6Context(question="q", references=[refs[0], refs[2]]))
    raises("S6 编号未从 1 开始拒绝",
           lambda: S6Context(question="q", references=[S6Reference(
               ref_no=2, chunk_id="c", doc_id="d", quality_score=1, text="t")]))
    raises("S6 空引用拒绝", lambda: S6Context(question="q", references=[]))


def test_s7():
    snap = S7Snapshot(query_id="q_1", session_id="s_1", question="q", status="answering")
    ok("S7 进行中最小快照合法", snap.cost_tokens.total == 0 and snap.answer == "")
    raises("S7 非法 status 拒绝", lambda: S7Snapshot(query_id="q", session_id="s", question="q", status="done"))


def test_s8():
    ev = S8Event(ts="2026-10-16T10:00:00+08:00", query_id="q_1", step="intent",
                 payload={"intent": "fact_qa"})
    ok("S8 合法样例", ev.level == "info")
    raises("S8 缺 query_id 拒绝", lambda: S8Event(ts="2026-10-16T10:00:00+08:00", step="intent"))


def test_s9():
    req = S9ChatRequest(question="你好")
    ok("S9 请求最小合法", req.history == [])
    raises("S9 空问题拒绝", lambda: S9ChatRequest(question=""))
    ev = make_sse("delta", "q_1", S9Delta(delta="你好"))
    ok("S9 SSE delta 事件合法", ev.event == "delta" and ev.data == {"delta": "你好"})
    raises("S9 SSE delta 空 payload 拒绝", lambda: S9Event(query_id="q_1", event="delta", data={}))
    raises("S9 SSE 未知事件类型拒绝", lambda: S9Event(query_id="q_1", event="thinking", data={"delta": "x"}))
    ok("S9 sources 响应合法", S9SourcesResponse(query="q").results == [])


def test_s9_full():
    """P5（§14）：6 类 SSE 事件 round-trip 双向校验（make_sse 写入 ↔ S9Event 读取）。"""
    from schemas.models import (S9Done, S9Error, S9Meta, S9Sources, S7Verification)
    qid = "q_test"
    payloads = [
        ("meta", S9Meta(intent="fact_qa", entities=["e"], queries=["q"], keywords=["k"])),
        ("reasoning", S9Delta(delta="推理")),
        ("delta", S9Delta(delta="正文")),
        ("sources", S9Sources(references=[dict(
            schema_version="1.0.0", chunk_id="a_b_c0", doc_id="b", text="t",
            score=0.87, rank_source="fusion", quality_score=4.1,
            chunk_index=0, chunk_count=1)])),
        ("done", S9Done(answer="答案", citations=["a_b_c0"],
                        verification=S7Verification(verdict="pass"))),
        ("error", S9Error(code="internal_error", message="m")),
    ]
    for name, payload in payloads:
        ev = make_sse(name, qid, payload)
        back = S9Event.model_validate_json(ev.model_dump_json())
        ok(f"S9 {name} round-trip", back.model_dump() == ev.model_dump())
    raises("S9 done 塞 delta payload 拒绝", lambda: make_sse("done", qid, S9Delta(delta="x")))
    raises("S9 meta 非法 intent 拒绝", lambda: make_sse("meta", qid, S9Meta(intent="nope")))


def test_history():
    """P5（§14）：merge/trim 边界。"""
    from server.chat_history import merge_history, trim_history
    client_h = [{"role": "user", "content": "c1"}]
    server_h = [{"role": "user", "content": "s1"}]
    ok("history client 显式优先", merge_history(client_h, server_h) == client_h)
    ok("history 无 client 用 server", merge_history([], server_h) == server_h)
    msgs = [{"role": "user", "content": "x" * 10}] * 20      # 10 轮 × 2 条
    ok("history 轮数上限 20 条", len(trim_history(msgs, max_turns=10, max_chars=100000)) == 20)
    ok("history 超轮数裁剪", len(trim_history(msgs, max_turns=3, max_chars=100000)) == 6)
    big = [{"role": "user", "content": "字" * 5000}] * 10
    t = trim_history(big, max_turns=10, max_chars=12000)
    ok("history 字符上限", sum(len(m["content"]) for m in t) <= 12000 and len(t) < 10)
    ok("history 过滤 system 与非 dict",
       trim_history([{"role": "system", "content": "s"}, "junk"] + msgs,
                    max_turns=10, max_chars=100000) == msgs)


def test_maintenance():
    from prompts.refusals import MAINTENANCE
    ok("MAINTENANCE 话术存在", "维护" in MAINTENANCE)


def test_extract():
    ok("提取裸 JSON", extract_json('{"a": 1}') == {"a": 1})
    ok("提取代码围栏", extract_json('```json\n{"a": 1}\n```') == {"a": 1})
    ok("提取前后杂文", extract_json('好的，结果如下：\n{"a": 1}\n以上。') == {"a": 1})
    raises("无 JSON 拒绝", lambda: extract_json("没有 json"), ValueError)


def test_guard():
    ok("realtime 拦截", quick_guard("今天A股大盘涨了多少？") == "realtime")
    ok("advice 拦截", quick_guard("现在买什么股票比较好？") == "investment_advice")
    ok("advice 优先于时间词", quick_guard("现在买入茅台合适吗？") == "investment_advice")
    ok("fact_qa 不拦截", quick_guard("2021年综合保税区进出口值是多少？") is None)


def main():
    test_s1(), test_s2(), test_s3(), test_s4(), test_s5(), test_s6()
    test_s7(), test_s8(), test_s9(), test_s9_full(), test_history()
    test_maintenance(), test_extract(), test_guard()
    print(f"\n通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
