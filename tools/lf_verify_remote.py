# -*- coding: utf-8 -*-
"""Langfuse 部署验证（ECS 上运行）：登录 → 问一题 → 输出 query_id。

用法（ECS）: .venv/bin/python tools/lf_verify_remote.py
退出码 0 = 拿到 done 与 query_id；1 = 失败。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools import smoke_api as s  # noqa: E402

base = "http://127.0.0.1:8001"
print(f"=== Langfuse 验证 @ {base} ===", flush=True)
token = s.login(base)
evs = s.post_chat(base, {"question": "2021年全国综合保税区进出口值是多少？",
                         "session_id": f"lf_verify_{int(time.time())}"}, token)
types = [e.event for e in evs]
qid = evs[0].query_id
done = next((e for e in evs if e.event == "done"), None)
ans = done.data["answer"][:60].replace("\n", " ") if done else "NONE"
print(f"EVENTS: {' → '.join(t for t in types if t not in ('reasoning', 'delta'))}", flush=True)
print(f"QUERY_ID: {qid}", flush=True)
print(f"ANSWER: {ans}", flush=True)
sys.exit(0 if (done and qid) else 1)
