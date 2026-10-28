# -*- coding: utf-8 -*-
"""登录安全加固单元测试（2026-10-22，不起服务、不连库）。

覆盖：密码哈希新旧格式往返/机会式升级、账号锁定状态机（fake db）、
IP 限流（内存模式 + Redis 滑动窗口模式 + Redis 挂降级）、token 签发/篡改/过期、
注册密码策略、验证码一次性。
用法: python tests/test_auth.py
"""
import importlib
import sys
import time
import hashlib
import hmac
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import HTTPException

import config
from server import auth
from server.auth_guard import IpThrottle, lock_remaining, record_fail
from server.captcha import CaptchaStore
from server.db import CST

PASS, FAIL = [], []


def ok(name, cond):
    (PASS if cond else FAIL).append(name)
    if not cond:
        print(f"  ✗ {name}")


def raises(name, fn, exc=HTTPException):
    try:
        fn()
    except exc:
        PASS.append(name)
    else:
        FAIL.append(name)
        print(f"  ✗ {name}（应拒绝但通过了）")


# ---------- 1. 密码哈希：新格式往返 + 旧格式兼容 ----------

salt, h = auth.hash_password("Abc12345")
ok("新哈希盐带轮数前缀", salt.startswith(f"{config.PBKDF2_ROUNDS}$"))
ok("新哈希验证通过", auth.verify_password("Abc12345", salt, h))
ok("新哈希错误密码拒绝", not auth.verify_password("Abc12346", salt, h))

legacy_salt = "a" * 32  # 旧格式纯 hex
legacy_h = hashlib.pbkdf2_hmac("sha256", b"OldPass1", legacy_salt.encode(), 100_000).hex()
ok("旧格式(10万轮)哈希仍可验证", auth.verify_password("OldPass1", legacy_salt, legacy_h))
ok("旧格式错误密码拒绝", not auth.verify_password("OldPass2", legacy_salt, legacy_h))

# ---------- 2. 机会式哈希升级（fake db） ----------

upgraded = {}
_orig_upd = auth.db.update_password
auth.db.update_password = lambda uid, hh, ss: upgraded.update(uid=uid, h=hh, s=ss)
auth._upgrade_password({"id": 7, "salt": legacy_salt}, "OldPass1")
ok("旧轮数登录成功触发重哈希落库", upgraded.get("uid") == 7
   and upgraded["s"].startswith(f"{config.PBKDF2_ROUNDS}$")
   and auth.verify_password("OldPass1", upgraded["s"], upgraded["h"]))
upgraded.clear()
auth._upgrade_password({"id": 7, "salt": salt}, "Abc12345")
ok("新轮数不重复升级", not upgraded)
auth.db.update_password = _orig_upd

# ---------- 3. 账号锁定状态机（fake db） ----------

# 锁定时间用 CST 显式构造（不用 datetime.now() 本地时区——本机 UTC+7 与 ECS CST
# 会算出不同剩余小时数，时区脆弱）；+30s 余量保证向下取整后仍是「23 小时」
locked_row = {"id": 1, "locked_until": datetime.now(CST) + timedelta(hours=23, seconds=30)}
msg = lock_remaining(locked_row)
ok("锁定期内返回剩余时间提示", msg and "已锁定" in msg and "23 小时" in msg)
ok("锁定过期返回 None", lock_remaining(
    {"id": 1, "locked_until": datetime.now(CST) - timedelta(seconds=1)}) is None)
ok("无锁定字段返回 None", lock_remaining({"id": 1}) is None)

lock_writes = []
_orig_rlf = auth.db.record_login_fail
auth.db.record_login_fail = lambda uid, locked_until=None: lock_writes.append(
    (uid, locked_until))
ok("第5次错误触发锁定(写锁定时间并清零)", record_fail(
    {"id": 1, "failed_attempts": 4}) and lock_writes[-1][1] is not None)
ok("未达上限只计数+1", not record_fail({"id": 1, "failed_attempts": 1})
   and lock_writes[-1][1] is None)
auth.db.record_login_fail = _orig_rlf

# ---------- 4. IP 限流 ----------

th = IpThrottle()
for _ in range(config.IP_FAIL_LIMIT):
    th.note_login_fail("9.9.9.9")
raises("IP 失败超限后被封", lambda: th.throttle_login("9.9.9.9"))
th.throttle_login("8.8.8.8")  # 其他 IP 不受影响
th.note_login_ok("9.9.9.9")
th.throttle_login("9.9.9.9")  # 登录成功清零后可再试
for _ in range(config.CAPTCHA_IP_LIMIT):
    th.throttle_captcha("1.2.3.4")
raises("验证码生成超频 429", lambda: th.throttle_captcha("1.2.3.4"), exc=HTTPException)

# ---------- 5. token ----------

tok = auth.issue_token(42)
ok("token 可解析回 user_id", auth.parse_token(tok) == 42)
tampered = tok[:-1] + ("0" if tok[-1] != "0" else "1")
ok("篡改签名拒绝", auth.parse_token(tampered) is None)
expired_body = f"v1.42.{int(time.time()) - 10}"
expired = (expired_body + "." +
           hmac.new(config.TOKEN_SECRET.encode(), expired_body.encode(),
                    hashlib.sha256).hexdigest())
ok("过期 token 拒绝", auth.parse_token(expired) is None)
ok("乱码 token 拒绝", auth.parse_token("garbage") is None)

# ---------- 6. 注册密码策略 ----------

ok("8位字母数字混合通过", auth._password_ok("abc12345"))
ok("纯字母拒绝", not auth._password_ok("abcdefgh"))
ok("纯数字拒绝", not auth._password_ok("12345678"))
ok("不足8位拒绝", not auth._password_ok("ab1"))
ok("中文+数字通过(含字母数字)", auth._password_ok("密码abc123"))

# ---------- 7. 验证码一次性（内存模式） ----------

store = CaptchaStore(health=None)
cid, _img = store.new()
code = store._mem[cid][0]
ok("验证码正确校验通过", store.verify(cid, code))
ok("验证码一次性：二次校验失败", not store.verify(cid, code))
cid2, _ = store.new()
ok("验证码错误码拒绝", not store.verify(cid2, "XXXX"))
ok("验证码不存在的 id 拒绝", not store.verify("no-such-id", "ABCD"))
cid3, _ = store.new()
code3 = store._mem[cid3][0]
ok("验证码大小写不敏感", store.verify(cid3, code3.lower()))

# ---------- 8. IP 限流 Redis 路径（假 redis + 假健康探针） ----------

class FakeRedisClient:
    """dict 版 Redis：zset 存 {member: score}，TTL 只记账不清理（测试窗口不跨 TTL）。"""

    def __init__(self):
        self.d = {}
        self.expires = {}
        self.cmds = []

    def get(self, k):
        self.cmds.append(("get", k))
        return self.d.get(k)

    def set(self, k, v, ex=None):
        self.cmds.append(("set", k, ex))
        self.d[k] = v
        if ex is not None:
            self.expires[k] = time.time() + ex

    def delete(self, *ks):
        self.cmds.append(("delete", ks))
        for k in ks:
            self.d.pop(k, None)

    def zadd(self, k, mapping):
        self.cmds.append(("zadd", k))
        self.d.setdefault(k, {}).update(mapping)
        return len(mapping)

    def zremrangebyscore(self, k, mn, mx):
        self.cmds.append(("zremrangebyscore", k))
        z = self.d.get(k, {})
        gone = [m for m, s in z.items() if mn <= s <= mx]
        for m in gone:
            del z[m]
        return len(gone)

    def zcard(self, k):
        self.cmds.append(("zcard", k))
        return len(self.d.get(k, {}))

    def expire(self, k, s):
        self.cmds.append(("expire", k, s))
        self.expires[k] = time.time() + s
        return True

    def pipeline(self, transaction=False):
        class _Pipe:
            def __init__(self, cli):
                self._cli = cli
                self._q = []

            def __getattr__(self, name):
                def enqueue(*a, **kw):
                    self._q.append((name, a, kw))
                    return self
                return enqueue

            def execute(self):
                return [getattr(self._cli, n)(*a, **kw) for n, a, kw in self._q]
        return _Pipe(self)


class FakeRedisModule:
    class Redis:
        clients = []

        @classmethod
        def from_url(cls, url, **kw):
            c = FakeRedisClient()
            cls.clients.append(c)
            return c


class FakeHealth:
    def __init__(self):
        self.available = True

    def mark_down(self):
        self.available = False


sys.modules["redis"] = FakeRedisModule()
import server.auth_guard as ag
importlib.reload(ag)   # 让 auth_guard 重新绑定假 redis 模块

health = FakeHealth()
th = ag.IpThrottle(health=health)
for _ in range(config.IP_FAIL_LIMIT - 1):
    th.note_login_fail("9.9.9.9")
th.throttle_login("9.9.9.9")   # 29 次未超限
th.note_login_fail("9.9.9.9")
raises("Redis 路径：第30次失败触发封禁", lambda: th.throttle_login("9.9.9.9"))
th.throttle_login("8.8.8.8")   # 其他 IP 不受影响
th.note_login_ok("9.9.9.9")
th.throttle_login("9.9.9.9")   # 登录成功清零后解封

for _ in range(config.CAPTCHA_IP_LIMIT - 1):
    th.throttle_captcha("1.2.3.4")
raises("Redis 路径：验证码超频 429", lambda: th.throttle_captcha("1.2.3.4"),
       exc=HTTPException)

cli = FakeRedisModule.Redis.clients[-1]
ok("计数/封禁落 Redis 且成功清零",
   "fin:iplimit:9.9.9.9" not in cli.d and "fin:ipblock:9.9.9.9" not in cli.d)
ok("验证码计数落 Redis", "fin:caplimit:1.2.3.4" in cli.d)

# ---------- 9. Redis 挂 → 降级内存 ----------

class BoomRedisModule:
    class Redis:
        @classmethod
        def from_url(cls, url, **kw):
            raise OSError("连接不上")


sys.modules["redis"] = BoomRedisModule()
importlib.reload(ag)
h2 = FakeHealth()   # 探测线程滞后：available 仍为 True，靠调用异常降级
th2 = ag.IpThrottle(health=h2)
for _ in range(config.IP_FAIL_LIMIT):
    th2.note_login_fail("9.9.9.9")
ok("Redis 异常时降级内存并广播 mark_down", not h2.available)
raises("降级后内存封禁仍生效", lambda: th2.throttle_login("9.9.9.9"))
th2.note_login_ok("9.9.9.9")
th2.throttle_login("9.9.9.9")

# ---------- 汇总 ----------

print(f"通过 {len(PASS)}/{len(PASS) + len(FAIL)}")
if FAIL:
    print("失败用例:", FAIL)
    sys.exit(1)
