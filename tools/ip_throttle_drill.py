# -*- coding: utf-8 -*-
"""IP 限流（Redis 滑动窗口）线上演练（2026-10-25，需在 ECS 本机跑）：
伪造验证码连打登录接口——验证码失败也计 IP 失败（正是「一次一账号」攻击者的
行为模型），验证 30 次后 403 封禁 + 计数/封禁 key 落 Redis，最后清理演练痕迹。
用法: cd /data/financial && .venv/bin/python tools/ip_throttle_drill.py
"""
import json
import subprocess
import sys
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8001"
LOGIN_KEYS = ("fin:ipblock:127.0.0.1", "fin:iplimit:127.0.0.1")
CAPTCHA_KEY = "fin:caplimit:127.0.0.1"

PASS, FAIL = [], []


def ok(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  ok  " if cond else "  ✗   ") + name)


def post(path, body):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def get(path):
    req = urllib.request.Request(BASE + path, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def redis_cli(*args):
    out = subprocess.run(["docker", "exec", "fin-redis", "redis-cli"] + list(args),
                         capture_output=True, text=True)
    return out.stdout.strip()


def bad_login():
    """伪造验证码打登录：验证码错计 IP 失败，但不触账号锁定（admin 安全）。"""
    return post("/api/login", {"username": "admin", "password": "x",
                               "captcha_id": "drill", "captcha_code": "XXXX"})


# ---------- 1) 登录失败 30 次 → 封禁 ----------

for k in list(LOGIN_KEYS) + [CAPTCHA_KEY]:
    redis_cli("DEL", k)   # 清场（防上次残留）

codes = [bad_login() for _ in range(30)]
ok("前 30 次均为 401（统一话术防枚举）", codes == [401] * 30)
ok("第 31 次被 IP 封禁 403", bad_login() == 403)
ok("封禁标记落 Redis", redis_cli("GET", LOGIN_KEYS[0]) == "1")
ok("触发封禁后失败计数清零（期满给全新窗口）", redis_cli("EXISTS", LOGIN_KEYS[1]) == "0")

# ---------- 2) 验证码生成限频 20 次/分钟 → 429 ----------

codes = [get("/api/captcha") for _ in range(20)]
ok("验证码生成 20 次后 429", codes[-1] == 429 and codes[:-1] == [200] * 19)
ok("验证码计数落 Redis", redis_cli("EXISTS", CAPTCHA_KEY) == "1")

# ---------- 3) 清理演练痕迹 ----------

for k in list(LOGIN_KEYS) + [CAPTCHA_KEY]:
    redis_cli("DEL", k)
print(f"\n演练 {len(PASS)}/{len(PASS) + len(FAIL)} 通过")
sys.exit(1 if FAIL else 0)
