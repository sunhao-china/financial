# -*- coding: utf-8 -*-
"""读取最近一次问答内容（页面同款数据路径，只读）：登录 → 会话列表 → 最新会话消息。
排查「页面上看到的推理过程/回答」用。需在 ECS 本机跑（验证码读本机 Redis）。
用法: cd /data/financial && .venv/bin/python tools/last_messages.py
"""
import json

import redis
import requests

import config

BASE = "http://127.0.0.1:8001"


def main():
    # 登录（验证码 + admin，同 smoke_api 方式）
    r = requests.get(f"{BASE}/api/captcha", timeout=10)
    assert r.status_code == 200, f"captcha {r.status_code}: {r.text[:200]}"
    cid = r.json()["captcha_id"]
    code = redis.Redis(host="127.0.0.1", port=6379, socket_timeout=2) \
        .get(f"fin:captcha:{cid}").decode()
    r = requests.post(f"{BASE}/api/login", timeout=10, json={
        "username": "admin", "password": config.ADMIN_PASSWORD,
        "captcha_id": cid, "captcha_code": code})
    assert r.status_code == 200, f"login {r.status_code}: {r.text[:200]}"
    token = r.json()["token"]
    h = {"Authorization": f"Bearer {token}"}

    sessions = requests.get(f"{BASE}/api/sessions", headers=h, timeout=10).json()
    print(f"共 {len(sessions)} 个会话，最新 3 个：")
    for s in sessions[:3]:
        print(json.dumps(s, ensure_ascii=False)[:300])

    if not sessions:
        return
    conv = sessions[0]
    r = requests.get(f"{BASE}/api/sessions/{conv['id']}", headers=h, timeout=10)
    assert r.status_code == 200, f"messages {r.status_code}: {r.text[:200]}"
    msgs = r.json()
    print(f"\n最新会话（{conv['id']}）最后 {min(len(msgs), 4)} 条消息：")
    for m in msgs[-4:]:
        print(f"--- {m.get('role')} ---")
        print((m.get('content') or '')[:800])


if __name__ == "__main__":
    main()
