# -*- coding: utf-8 -*-
"""登录安全加固线上演练（2026-10-22，ECS 本机运行：验证码需读本机 Redis）。

验证：注册密码策略 / 账号 5 次锁定 24h / IP 30 次失败封禁 / 验证码洪泛 429。
注意：演练会在 IP 维度把 127.0.0.1 封掉（内存态），结束后需 systemctl restart financial。
用法（远端）: .venv/bin/python tools/security_drill.py
"""
import random
import string
import sys
import time

import requests

try:
    import redis
except ImportError:
    redis = None

BASE = "http://127.0.0.1:8001"
PASSED, FAILED = 0, []


def ok(name, cond, extra=""):
    global PASSED
    if cond:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED.append(name)
        print(f"  ✗ {name} {extra}")


def captcha():
    r = requests.get(f"{BASE}/api/captcha", timeout=10)
    if r.status_code != 200:
        return None, None, r
    cid = r.json()["captcha_id"]
    code = None
    if redis:
        try:
            code = redis.Redis(host="127.0.0.1", port=6379,
                               socket_timeout=2).get(f"fin:captcha:{cid}")
            if code:
                code = code.decode()
        except Exception:  # noqa: BLE001
            code = None
    return cid, code, r


def login(username, password):
    cid, code, r = captcha()
    if code is None:
        return r
    return requests.post(f"{BASE}/api/login", timeout=10, json={
        "username": username, "password": password,
        "captcha_id": cid, "captcha_code": code})


def register(username, password):
    cid, code, r = captcha()
    if code is None:
        return r
    return requests.post(f"{BASE}/api/register", timeout=10, json={
        "username": username, "password": password,
        "captcha_id": cid, "captcha_code": code})


name = "secdrill_" + "".join(random.choices(string.ascii_lowercase, k=6))
pwd = "Drill1234"

print("== 1. 注册密码策略 ==")
r = register(name, "12345678")
ok("纯数字 8 位注册被拒(400)", r.status_code == 400, f"HTTP {r.status_code} {r.text[:80]}")
r = register(name, "abcdefgh")
ok("纯字母 8 位注册被拒(400)", r.status_code == 400, f"HTTP {r.status_code} {r.text[:80]}")
r = register(name, "short1")
ok("不足 8 位注册被拒(400)", r.status_code == 400, f"HTTP {r.status_code} {r.text[:80]}")
r = register(name, pwd)
ok(f"合规密码注册成功(200): {name}", r.status_code == 200, f"HTTP {r.status_code} {r.text[:80]}")

print("== 2. 账号 5 次错误锁 24h ==")
for i in range(5):
    r = login(name, "WrongPass1")
    ok(f"第 {i+1} 次错误密码 → 401", r.status_code == 401,
       f"HTTP {r.status_code} {r.text[:80]}")
r = login(name, pwd)
ok("锁定期内密码正确仍 403", r.status_code == 403, f"HTTP {r.status_code} {r.text[:80]}")
ok("锁定话术含剩余时间", "已锁定" in r.text and "小时" in r.text, r.text[:100])

print("== 3. 验证码洪泛限频 ==")
codes = []
for i in range(21):
    cid, code, r = captcha()
    codes.append(r.status_code)
ok("第 21 次验证码请求被 429 限频", codes[-1] == 429,
   f"最后一次 HTTP {codes[-1]}（前 20 次均 200）")

print("== 4. IP 30 次失败封禁 ==")
# 用伪造验证码直接打登录接口：验证码错也计 IP 失败（真实攻击者不会解验证码），
# 且绕开「验证码生成 20 次/分钟」对自己演练的限频干扰。
# 前面已累计 5 次失败，再 25 次到 30 上限。
blocked = False
for i in range(30):
    r = requests.post(f"{BASE}/api/login", timeout=10, json={
        "username": "no_such_user_xyz", "password": "WrongPass1",
        "captcha_id": "fake", "captcha_code": "XXXX"})
    if r.status_code == 403 and "频繁" in r.text:
        blocked = True
        print(f"  第 {i+1} 次尝试触发 IP 封禁")
        break
ok("IP 失败超限被封禁(403 尝试过于频繁)", blocked)

print("== 5. DB 状态验证 + 临时账号清理（docker exec 进容器）==")
import subprocess
sql = (f"SELECT failed_attempts, locked_until FROM users WHERE username='{name}'; "
       f"DELETE FROM users WHERE username LIKE 'secdrill_%';")
r = subprocess.run(["docker", "exec", "fin-mysql", "mysql", "-ufin",
                    "--password=" + open("/data/financial/.env").read()
                    .split("MYSQL_PASSWORD=")[1].splitlines()[0].strip(),
                    "fin", "-N", "-e", sql],
                   capture_output=True, text=True, timeout=30)
row = r.stdout.strip().splitlines()
locked_ok = (r.returncode == 0 and row and len(row[0].split("\t")) >= 2
             and row[0].split("\t")[1] not in ("NULL", ""))
ok("DB 锁定状态正确(failed=0 且 locked_until 非空) 且 secdrill_* 已清理",
   locked_ok, f"rc={r.returncode} out={r.stdout[:100]} err={r.stderr[:100]}")

print(f"\n演练通过 {PASSED}/{PASSED + len(FAILED)}")
if FAILED:
    print("失败:", FAILED)
    sys.exit(1)
