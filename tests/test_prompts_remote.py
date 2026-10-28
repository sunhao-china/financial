# -*- coding: utf-8 -*-
"""提示词远端化（prompts/remote.py）离线测试（不调 API、假 SDK，2026-10-22）。

覆盖：关闭/开关关闭 → get_prompt None 且 prompt_text 返回本地 SYSTEM 默认；
远端正常 → 返回远端 production 文本；远端缺失/空文本/异常 → 本地兜底；
5 套模板文本兜底一致性；llm_client.chat/chat_stream 把 prompt_client 透传给 span。
用法: python tests/test_prompts_remote.py
"""
import importlib
import os
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


# ---------- 假 SDK（get_prompt 可注入文本/异常） ----------

class FakePrompt:
    def __init__(self, text):
        self.prompt = text


class FakeSpan:
    created = []

    def __init__(self, name, **kw):
        self.kw = kw
        self.updates = []

    def update(self, **kw):
        self.updates.append(kw)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeLangfuse:
    """get_prompt 行为由类变量控制：文本 / None / 抛异常。"""

    result = "远端版提示词"
    boom = False

    def __init__(self, **kw):
        self.kw = kw

    def get_prompt(self, name, **kw):
        FakeLangfuse.calls.append(name)
        if FakeLangfuse.boom:
            raise RuntimeError("网络炸了")
        return FakePrompt(FakeLangfuse.result) if FakeLangfuse.result is not None else None

    calls = []

    @staticmethod
    def create_trace_id(*, seed):
        return "t" * 32

    def start_as_current_observation(self, **kw):
        s = FakeSpan(kw.pop("name"), **kw)
        FakeSpan.created.append(s)
        return s


def install_fake_sdk():
    m = types.ModuleType("langfuse")
    m.Langfuse = FakeLangfuse
    sys.modules["langfuse"] = m


def reload_llm_client():
    import llm_client
    return importlib.reload(llm_client)


def disable_langfuse():
    config.LANGFUSE_ENABLED = False
    config.LANGFUSE_PUBLIC_KEY = ""
    config.LANGFUSE_SECRET_KEY = ""
    config.LANGFUSE_HOST = "http://127.0.0.1:3000"


def enable_langfuse():
    config.LANGFUSE_ENABLED = True
    config.LANGFUSE_PUBLIC_KEY = "pk"
    config.LANGFUSE_SECRET_KEY = "sk"
    config.LANGFUSE_HOST = "http://127.0.0.1:3000"


def set_env_flag(val):
    if val is None:
        os.environ.pop("FIN_LANGFUSE_PROMPTS", None)
    else:
        os.environ["FIN_LANGFUSE_PROMPTS"] = val


# ---------- 场景 1：追踪未启用 → 全量本地兜底 ----------

def test_disabled_fallback():
    from prompts import remote
    disable_langfuse()
    set_env_flag(None)
    reload_llm_client()
    for name in remote.NAMES:
        ok(f"未启用时 get_prompt({name}) = None", remote.get_prompt(name) is None)
    for name in remote.NAMES:
        expect = remote._default_text(name)
        ok(f"未启用时 prompt_text({name}) = 本地 SYSTEM",
           remote.prompt_text(name) == expect and len(expect) > 100)


# ---------- 场景 2：FIN_LANGFUSE_PROMPTS=0 整体关闭 ----------

def test_env_flag_off():
    from prompts import remote
    enable_langfuse()
    set_env_flag("0")
    reload_llm_client()
    ok("开关=0 时 get_prompt 不触远端", remote.get_prompt("judge") is None)
    ok("开关=0 时 prompt_text 回落本地",
       remote.prompt_text("judge") == remote._default_text("judge"))
    set_env_flag(None)


# ---------- 场景 3：远端正常 → 用远端文本 ----------

def test_remote_text_used():
    from prompts import remote
    enable_langfuse()
    install_fake_sdk()
    reload_llm_client()
    FakeLangfuse.boom = False
    FakeLangfuse.result = "远端生产版模板文本"
    FakeLangfuse.calls.clear()
    ok("远端可用时 prompt_text 返回远端文本",
       remote.prompt_text("generate") == "远端生产版模板文本")
    ok("get_prompt 请求了 production 标签名",
       FakeLangfuse.calls and FakeLangfuse.calls[-1] == remote.PREFIX + "generate")


# ---------- 场景 4：远端缺失/空文本/异常 → 本地兜底 ----------

def test_remote_failures_fallback():
    from prompts import remote
    enable_langfuse()
    install_fake_sdk()
    reload_llm_client()
    local = remote._default_text("intent_expand")
    FakeLangfuse.result = None          # 远端无此提示词
    ok("远端缺失 → 本地兜底", remote.prompt_text("intent_expand") == local)
    FakeLangfuse.result = "   "         # 远端空文本
    ok("远端空文本 → 本地兜底", remote.prompt_text("intent_expand") == local)
    FakeLangfuse.result = "x"
    FakeLangfuse.boom = True            # SDK 抛异常
    ok("远端异常 → 本地兜底", remote.prompt_text("intent_expand") == local)
    FakeLangfuse.boom = False


# ---------- 场景 5：5 套模板兜底一致性（迁移前后文本不变） ----------

def test_all_five_local_defaults():
    from prompts import remote
    disable_langfuse()
    reload_llm_client()
    for name in remote.NAMES:
        ok(f"{name} 兜底文本 = prompts/{name}.py SYSTEM（{len(remote._default_text(name))} 字符）",
           remote.prompt_text(name) == remote._default_text(name))
    ok("五套模板名与 prompts/*.py 一一对应",
       remote.NAMES == ("intent_expand", "generate", "self_check", "verify", "judge"))


# ---------- 场景 6：prompt_client 透传到 span（版本联动） ----------

def test_span_prompt_linkage():
    enable_langfuse()
    install_fake_sdk()
    lc = reload_llm_client()
    FakeSpan.created.clear()
    lc.LLMClient._post = lambda self, p, u, pl, to, label=None: {
        "choices": [{"message": {"content": "答"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        "model": "m"}
    pc = object()   # 模拟 prompts/remote.py get_prompt 返回的提示词客户端
    r = lc.LLMClient().chat([{"role": "user", "content": "hi"}],
                            route="self_check", prompt_client=pc)
    ok("chat 结果正常", r.content == "答")
    ok("span 收到 prompt 客户端（UI 显示提示词版本）",
       FakeSpan.created[0].kw.get("prompt") is pc)

    lc.LLMClient._chat_stream_impl = lambda self, messages, **kw: iter([
        {"type": "usage", "usage": lc.Usage(prompt_tokens=1, completion_tokens=1,
                                            total_tokens=2)}])
    FakeSpan.created.clear()
    list(lc.LLMClient().chat_stream([{"role": "user", "content": "hi"}],
                                    route="generate_fact", prompt_client=pc))
    ok("chat_stream span 收到 prompt 客户端",
       FakeSpan.created[0].kw.get("prompt") is pc)
    # 不传 prompt_client 时 span 无 prompt 字段（老调用点行为不变）
    FakeSpan.created.clear()
    list(lc.LLMClient().chat_stream([{"role": "user", "content": "hi"}],
                                    route="generate_fact"))
    ok("不传时 span 无 prompt 字段（兼容）",
       FakeSpan.created[0].kw.get("prompt") is None)


# ---------- 汇总 ----------

test_disabled_fallback()
test_env_flag_off()
test_remote_text_used()
test_remote_failures_fallback()
test_all_five_local_defaults()
test_span_prompt_linkage()

print(f"\n提示词远端化测试：{len(PASS)}/{len(PASS) + len(FAIL)} 通过")
sys.exit(1 if FAIL else 0)
