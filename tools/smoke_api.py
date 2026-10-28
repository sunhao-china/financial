# -*- coding: utf-8 -*-
"""P5 上线冒烟（ECS 上运行）：级联验证 登录 → /health → SSE 时序 → /sources → 多轮
（MySQL messages）→ 会话列表/标题 → 拒答路径 → Redis 降级演练 → /metrics + S8 链路
→ 4 并发不串线。

用法（ECS）: .venv/bin/python tools/smoke_api.py [--host 127.0.0.1] [--port 8001]
退出码 0 = 全部通过；1 = 有失败项（每项单独 PASS/FAIL，失败项不中断后续检查）。
登录：GET /api/captcha 拿 captcha_id，从 Redis 读 fin:captcha:{id} 得 code（ECS 本机可连
Redis；验证码一次性，登录后即可）。账号 admin（config.ADMIN_PASSWORD）。
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402
import requests  # noqa: E402
from schemas.models import S9Event, S9SourcesResponse  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  {'✓' if cond else '✗'} {name}" + (f" — {detail}" if detail else ""), flush=True)
    return bool(cond)


def login(base, host="127.0.0.1"):
    """验证码 → Redis 读 code → 登录 → token。失败抛异常（后续全部检查依赖登录态）。"""
    r = requests.get(f"{base}/api/captcha", timeout=10)
    assert r.status_code == 200, f"captcha HTTP {r.status_code}: {r.text[:200]}"
    cid = r.json()["captcha_id"]
    try:
        import redis as _r
        code = _r.Redis(host=host, port=6379, socket_timeout=2).get(f"fin:captcha:{cid}")
        assert code, "Redis 中无该验证码（Redis 降级为进程内存时 smoke 无法读码）"
        code = code.decode()
    except ImportError:
        raise AssertionError("无 redis 包，无法读验证码（需在 ECS venv 运行）")
    r = requests.post(f"{base}/api/login", timeout=10, json={
        "username": config.ADMIN_USERNAME, "password": config.ADMIN_PASSWORD,
        "captcha_id": cid, "captcha_code": code})
    assert r.status_code == 200, f"login HTTP {r.status_code}: {r.text[:200]}"
    token = r.json()["token"]
    print(f"  - 已登录 {config.ADMIN_USERNAME}（token {token[:12]}…）", flush=True)
    return token


def post_chat(base, payload, token, timeout=180):
    """POST /chat（带 Bearer token），消费 SSE，返回事件列表（每项为已校验的 S9Event）。"""
    events = []
    headers = {"Authorization": f"Bearer {token}"}
    with requests.post(f"{base}/chat", json=payload, timeout=timeout, stream=True,
                       headers=headers) as r:
        assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:200]}"
        assert "text/event-stream" in r.headers.get("content-type", ""), r.headers
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            ev = S9Event.model_validate_json(line[5:].strip())
            assert ev.query_id, "query_id 为空"
            events.append(ev)
    return events


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=config.API_PORT)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"
    print(f"=== P5 冒烟 @ {base} ===", flush=True)

    # 0. 登录（验证码 + admin 账号）；未登录 /chat 必须 401
    token = ""
    try:
        token = login(base, host=args.host)
        check("登录成功（验证码→Redis 读码→token）", bool(token))
    except Exception as e:  # noqa: BLE001
        check("登录成功（验证码→Redis 读码→token）", False, str(e))
        print("  ✗ 登录失败，后续 /chat 检查全部跳过", flush=True)
    if token:
        r = requests.post(f"{base}/chat", json={"question": "测试", "session_id": "x"},
                          timeout=10)
        check("未登录 /chat → 401", r.status_code == 401, f"HTTP {r.status_code}")

    # 1. /health
    try:
        h = requests.get(f"{base}/health", timeout=10).json()
        check("health 200 且状态 ok（含 mysql 字段）",
              h.get("status") == "ok" and "mysql" in h, str(h))
    except Exception as e:  # noqa: BLE001
        check("health 200 且状态 ok", False, str(e))

    # 2. /chat SSE 时序（fact_qa 快路径）
    evs2_query_id = ""
    try:
        evs = post_chat(base, {"question": "2021年全国综合保税区进出口值是多少？",
                               "session_id": f"smoke_seq_{int(time.time())}"}, token)
        evs2_query_id = evs[0].query_id
        types = [e.event for e in evs]
        check("SSE 事件时序 meta→sources→done",
              types and types[0] == "meta" and "sources" in types and types[-1] == "done",
              " → ".join(types))
        done = next(e for e in evs if e.event == "done")
        check("done 携带非空答案", bool(done.data["answer"].strip()),
              done.data["answer"][:60].replace("\n", " "))
        check("sources 含合法 S5Hit",
              any(e.event == "sources" and e.data.get("references") for e in evs))
        # §14.5 文献名：title_map 覆盖 98.7%，断言绝大多数命中（全命中概率仅 ~77%，易假失败）
        srefs = next(e.data["references"] for e in evs if e.event == "sources")
        titled = sum(1 for r in srefs if r.get("doc_title"))
        check("S5Hit 携带文献名 doc_title", titled >= max(1, len(srefs) - 2),
              f"{titled}/{len(srefs)} 条有文献名")
        check("delta/reasoning 增量存在", any(e.event in ("delta", "reasoning") for e in evs))
    except Exception as e:  # noqa: BLE001
        check("chat SSE 时序", False, str(e))

    # 3. /sources 调试接口
    try:
        r = requests.get(f"{base}/sources/2021年综合保税区进出口值", timeout=120)
        resp = S9SourcesResponse.model_validate(r.json())
        check("sources 调试接口返回结果", len(resp.results) > 0, f"{len(resp.results)} 条")
    except Exception as e:  # noqa: BLE001
        check("sources 调试接口", False, str(e))

    # 4. 多轮：同一 session_id 两问，第二问指代消解；MySQL messages 落库 + 会话标题懒落库
    sid = f"smoke_{int(time.time())}"
    try:
        evs1 = post_chat(base, {"question": "2021年全国综合保税区进出口值是多少？",
                                "session_id": sid}, token)
        evs2 = post_chat(base, {"question": "这个数据比上一年增长了多少？",
                                "session_id": sid}, token)
        check("多轮两问均完成",
              evs1[-1].event == "done" and evs2[-1].event == "done"
              and bool(evs2[-1].data["answer"].strip()),
              evs2[-1].data["answer"][:50].replace("\n", " "))
    except Exception as e:  # noqa: BLE001
        check("多轮对话", False, str(e))
    try:
        h = {"Authorization": f"Bearer {token}"}
        msgs = requests.get(f"{base}/api/conversations/{sid}/messages",
                            headers=h, timeout=10).json()["messages"]
        check("MySQL 会话历史落库（messages=4 且角色交错）",
              len(msgs) == 4
              and [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"],
              f"n={len(msgs)}")
        convs = requests.get(f"{base}/api/conversations", headers=h, timeout=10).json()
        mine = [c for c in convs["conversations"] if c["id"] == sid]
        check("会话列表含该会话且标题=首问截断",
              bool(mine) and mine[0]["title"] == "2021年全国综合保税区进出口值是多少？",
              mine[0]["title"] if mine else "缺失")
    except Exception as e:  # noqa: BLE001
        check("MySQL 会话落库与列表", False, str(e))

    # 5. 拒答路径（realtime guard）：meta→done，零生成
    try:
        evs = post_chat(base, {"question": "今天A股大盘涨了多少？",
                               "session_id": f"smoke_refuse_{int(time.time())}"}, token)
        types = [e.event for e in evs]
        check("拒答路径 meta→done 零生成",
              types[0] == "meta" and types[-1] == "done"
              and not any(t in ("delta", "reasoning", "sources") for t in types),
              " → ".join(types))
        check("拒答话术非空", bool(evs[-1].data["answer"].strip()))
    except Exception as e:  # noqa: BLE001
        check("拒答路径", False, str(e))

    # 6. Redis 降级演练（docker 可用时）
    if shutil.which("docker"):
        try:
            subprocess.run(["docker", "stop", "fin-redis"], check=True,
                           capture_output=True, timeout=60)
            time.sleep(2)
            evs = post_chat(base, {"question": "什么是存款准备金率？",
                                   "session_id": f"smoke_redis_{int(time.time())}"}, token)
            check("Redis 停止后问答仍可用（Memory 降级）",
                  evs[-1].event == "done" and bool(evs[-1].data["answer"].strip()))
            subprocess.run(["docker", "start", "fin-redis"], check=True,
                           capture_output=True, timeout=60)
            print("  - fin-redis 已恢复", flush=True)
        except subprocess.CalledProcessError as e:  # noqa: BLE001 — 容器本就不在：跳过
            print(f"  - 降级演练跳过（fin-redis 未运行: {e.stderr.decode()[:80]}）", flush=True)
        except Exception as e:  # noqa: BLE001
            check("Redis 降级演练", False, str(e))
    else:
        print("  - 降级演练跳过（无 docker）", flush=True)

    # 7. /metrics 非零 + S8 链路（复用步骤 2 的 query_id）
    try:
        m = requests.get(f"{base}/metrics", timeout=10).text
        chat_total = next((l for l in m.splitlines() if l.startswith("fin_chat_total ")), "")
        check("/metrics 计数器非零", bool(chat_total) and int(chat_total.split()[1]) >= 3,
              chat_total)
        if config.S8_LOG_PATH.exists():
            s8_lines = [l for l in open(config.S8_LOG_PATH, encoding="utf-8")
                        if f'"{evs2_query_id}"' in l]
            check("S8 日志同 query_id 链路", len(s8_lines) >= 3, f"{len(s8_lines)} 条")
    except Exception as e:  # noqa: BLE001
        check("/metrics + S8 链路", False, str(e))

    # 8. 4 并发不串线
    try:
        qs = ["2021年全国综合保税区进出口值是多少？", "什么是存款准备金率？",
              "2019年央企海外并购有什么特点？", "2020年新能源汽车补贴政策是什么？"]
        with ThreadPoolExecutor(max_workers=4) as ex:
            fut = [ex.submit(post_chat, base,
                             {"question": q, "session_id": f"smoke_c{i}_{int(time.time())}"},
                             token) for i, q in enumerate(qs)]
        qids, answers = [], []
        for f in fut:
            evs = f.result()
            qids.append(evs[0].query_id)
            answers.append(evs[-1].data["answer"].strip())
        check("4 并发全部完成且 query_id 唯一", len(set(qids)) == 4 and all(answers),
              f"answers={[len(a) for a in answers]}")
    except Exception as e:  # noqa: BLE001
        check("4 并发", False, str(e))

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n=== 冒烟完成：{len(RESULTS) - n_fail}/{len(RESULTS)} 通过 ===", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
