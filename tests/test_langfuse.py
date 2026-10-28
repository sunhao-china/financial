# -*- coding: utf-8 -*-
"""Langfuse 追踪层离线测试（不调 API、不依赖真实 SDK，大纲 §13.5）。

覆盖：未启用 no-op 路径（主流程零影响）、trace 由 query_id 确定性派生、
span 归属与 usage/output/metadata 记录、失败标注 level=ERROR 后原样上抛、
chat_stream 生成器在迭代期持有 span 上下文、SDK 缺失静默降级。
用法: python tests/test_langfuse.py
"""
import hashlib
import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402

PASS, FAIL = [], []


def ok(name, cond):
    (PASS if cond else FAIL).append(name)
    if not cond:
        print(f"  ✗ {name}")


# ---------- 假 SDK（惰性导入 llm_client 时挂到 sys.modules["langfuse"]） ----------

class FakeSpan:
    """模拟 start_as_current_observation 返回的 span 对象。"""

    created = []

    def __init__(self, name, **kw):
        self.name = name
        self.kw = kw
        self.updates = []

    def update(self, **kw):
        self.updates.append(kw)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeLangfuse:
    def __init__(self, **kw):
        self.kw = kw

    @staticmethod
    def create_trace_id(*, seed):
        return hashlib.md5(seed.encode()).hexdigest()

    def start_as_current_observation(self, **kw):
        s = FakeSpan(kw.pop("name"), **kw)
        FakeSpan.created.append(s)
        return s


class FakePropagate:
    """模拟 propagate_attributes（trace 级属性上下文，§13.5）。"""

    calls = []

    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        FakePropagate.calls.append(self.kw)
        return self

    def __exit__(self, *a):
        return False


def install_fake_sdk():
    m = types.ModuleType("langfuse")
    m.Langfuse = FakeLangfuse
    m.propagate_attributes = lambda **kw: FakePropagate(**kw)
    sys.modules["langfuse"] = m


def uninstall_fake_sdk():
    # 置 None 使 `from langfuse import ...` 抛 ImportError（模拟未安装）；
    # 注意本机可能真装了 langfuse，仅 pop 会导入到真 SDK。
    sys.modules["langfuse"] = None


def reload_client():
    import llm_client
    return importlib.reload(llm_client)


def reset_config():
    config.LANGFUSE_ENABLED = False
    config.LANGFUSE_PUBLIC_KEY = ""
    config.LANGFUSE_SECRET_KEY = ""
    config.LANGFUSE_HOST = "http://127.0.0.1:3000"


# ---------- 场景 1：未启用 → 全链路 no-op ----------

def test_noop_disabled():
    reset_config()
    lc = reload_client()
    ok("未启用时 _lf() 返回 None", lc._lf() is None)
    ok("未启用时模块级单例为 None", lc._lf_client is None)
    with lc.llm_trace("q1", "问题") as t:
        ok("llm_trace 产出 no-op 句柄", t.raw is None)
        t.update(output="x")          # 必须不抛
    with lc.llm_trace("q2", "问题2") as t:
        # no-op 下调用 chat（_post 打桩）不得创建任何 span
        lc.LLMClient._post = lambda self, p, u, pl, to, label=None: {
            "choices": [{"message": {"content": "答"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            "model": "m"}
        r = lc.LLMClient().chat([{"role": "user", "content": "hi"}], route="generate_fact")
        ok("no-op 下 chat 结果正常", r.content == "答" and r.usage.total_tokens == 3)
    ok("no-op 下零 span 创建", len(FakeSpan.created) == 0)


# ---------- 场景 2：启用 + 假 SDK → trace 派生 / span 记录 ----------

def test_trace_and_spans():
    reset_config()
    install_fake_sdk()
    config.LANGFUSE_ENABLED = True
    config.LANGFUSE_PUBLIC_KEY = "pk"
    config.LANGFUSE_SECRET_KEY = "sk"
    lc = reload_client()
    FakeSpan.created.clear()
    lc.LLMClient._post = lambda self, p, u, pl, to, label=None: {
        "choices": [{"message": {"content": "答"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                  "completion_tokens_details": {"reasoning_tokens": 10},
                  "total_tokens": 150},
        "model": "qwen-plus"}
    cli = lc.LLMClient()
    with lc.llm_trace("q_abc", "2021年综合保税区进出口值", "s1", "alice") as t:
        r = cli.chat([{"role": "user", "content": "hi"}], route="self_check")
        t.update(output="最终答案", metadata={"intent": "fact_qa"})
    ok("trace span 名称 = 问题截断", FakeSpan.created[0].name == "2021年综合保税区进出口值")
    tc = FakeSpan.created[0].kw["trace_context"]["trace_id"]
    ok("trace_id 由 query_id 确定性派生", tc == hashlib.md5(b"q_abc").hexdigest())
    ok("propagate_attributes 绑定 session/user 维度（UI 列）",
       FakePropagate.calls and FakePropagate.calls[-1]["session_id"] == "s1"
       and FakePropagate.calls[-1]["user_id"] == "alice")
    ok("trace metadata 含 query_id/session_id/user_id",
       FakeSpan.created[0].kw["metadata"]["query_id"] == "q_abc"
       and FakeSpan.created[0].kw["metadata"]["session_id"] == "s1"
       and FakeSpan.created[0].kw["metadata"]["user_id"] == "alice")
    ok("chat 生成子 span", len(FakeSpan.created) == 2 and FakeSpan.created[1].name == "self_check")
    gen = FakeSpan.created[1]
    ok("子 span 类型 generation", gen.kw["as_type"] == "generation")
    ok("子 span 记录模型", gen.kw["model"] == "qwen-plus")
    upd = gen.updates[-1]
    ok("子 span 记录 output", upd["output"] == "答")
    ok("子 span 记录 usage(input/output/total)",
       upd["usage_details"] == {"input": 100, "output": 50, "total": 150})
    ok("子 span metadata 含 reasoning/延迟",
       upd["metadata"]["reasoning_tokens"] == 10 and "latency_ms" in upd["metadata"])
    ok("trace 句柄 update 生效", FakeSpan.created[0].updates[-1]["output"] == "最终答案")


def test_error_marked_and_reraised():
    reset_config()
    install_fake_sdk()
    config.LANGFUSE_ENABLED = True
    config.LANGFUSE_PUBLIC_KEY = "pk"
    config.LANGFUSE_SECRET_KEY = "sk"
    lc = reload_client()
    FakeSpan.created.clear()

    def boom(self, p, u, pl, to, label=None):
        raise lc.APIError("dashscope", 500, {"err": "x"})

    lc.LLMClient._post = boom
    cli = lc.LLMClient()
    raised = False
    try:
        cli.chat([{"role": "user", "content": "hi"}], route="self_check")
    except lc.APIError:
        raised = True
    ok("API 失败原样上抛", raised)
    err_upd = [u for u in FakeSpan.created[0].updates if u.get("level") == "ERROR"]
    ok("失败 span 标注 level=ERROR", len(err_upd) == 1 and "500" in err_upd[0]["status_message"])


def test_chat_stream_span():
    reset_config()
    install_fake_sdk()
    config.LANGFUSE_ENABLED = True
    config.LANGFUSE_PUBLIC_KEY = "pk"
    config.LANGFUSE_SECRET_KEY = "sk"
    lc = reload_client()
    FakeSpan.created.clear()
    cli = lc.LLMClient()

    def fake_impl(self, messages, **kw):
        yield {"type": "content", "text": "a"}
        yield {"type": "content", "text": "bc"}
        yield {"type": "usage", "usage": lc.Usage(prompt_tokens=7, completion_tokens=3,
                                                   total_tokens=10)}

    lc.LLMClient._chat_stream_impl = fake_impl
    evs = list(cli.chat_stream([{"role": "user", "content": "hi"}], route="generate_fact"))
    ok("流式事件透传", [e["text"] for e in evs if e["type"] == "content"] == ["a", "bc"])
    gen = FakeSpan.created[0]
    upd = gen.updates[-1]
    ok("流式 span output = 全文拼接", upd["output"] == "abc")
    ok("流式 span 记录 usage", upd["usage_details"] == {"input": 7, "output": 3, "total": 10})
    ok("流式 metadata 标记 stream", upd["metadata"]["stream"] is True)


def test_sdk_missing_degrades():
    reset_config()
    uninstall_fake_sdk()
    config.LANGFUSE_ENABLED = True
    config.LANGFUSE_PUBLIC_KEY = "pk"
    config.LANGFUSE_SECRET_KEY = "sk"
    lc = reload_client()
    ok("SDK 缺失时 _lf() 降级 None", lc._lf() is None)
    ok("SDK 缺失标记 disabled", lc._lf_disabled)
    with lc.llm_trace("q1", "问题") as t:
        ok("SDK 缺失 llm_trace 仍 no-op", t.raw is None)


# ---------- 汇总 ----------

test_noop_disabled()
test_trace_and_spans()
test_error_marked_and_reraised()
test_chat_stream_span()
test_sdk_missing_degrades()

print(f"\nLangfuse 追踪层测试：{len(PASS)}/{len(PASS) + len(FAIL)} 通过")
sys.exit(1 if FAIL else 0)
