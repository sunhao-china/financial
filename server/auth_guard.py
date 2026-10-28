# -*- coding: utf-8 -*-
"""登录安全守卫（2026-10-22，§14.5 安全加固）：防暴力破解三层中的前两层。

- 账号锁定（MySQL 持久化，防重启失效）：连续密码错误 LOCKOUT_MAX_FAIL 次
  → 锁 LOCKOUT_SEC 秒。**验证码错误绝不计数**（否则攻击者故意输错验证码
  就能把任意账号锁死——锁定必须是「密码错误」的专有后果）；登录成功清零。
  锁定期满自动给新一轮额度。
- IP 限流（2026-10-25 迁 Redis 滑动窗口，多 worker 共享）：单 IP 窗口内失败次数
  （含验证码错）超限 → 封该 IP 一段时间，挡「一次一账号」的分布式扫号器；
  验证码生成单独限频（防洪泛喂 OCR 农场）。计数/封禁落在 fin:iplimit:{ip}
  （ZSET 滑动窗口，语义与原内存时间戳列表一致）与 fin:ipblock:{ip}
  （TTL 封禁标记）；Redis 挂 → 自动降级进程内存（单 worker 内仍有效，重启
  清空可接受——账号锁定在 MySQL 是持久的）。health=None（默认）纯内存模式，
  main.py lifespan 注入 RedisHealth。
- dummy pbkdf2（auth.py 中）：用户不存在也跑一次等价的哈希计算，
  均衡响应时间，防用户名枚举。
"""
from __future__ import annotations

import secrets
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

from fastapi import HTTPException

import config
from server.db import CST, db

try:
    import redis
except ImportError:  # 本地离线测试无 redis 包：Redis 路径不走，内存兜底
    redis = None

# 锁定时间戳统一上海时间（db.py 的 CST 约定），比较在 Python 侧完成，
# 不用 SQL NOW()/UNIX_TIMESTAMP()——避免 MySQL 会话时区引入的 8 小时坑。


def _now() -> datetime:
    return datetime.now(CST)


def lock_remaining(user_row: dict) -> Optional[str]:
    """锁定期内 → 返回剩余时间提示；未锁定/已过期 → None。"""
    lu = user_row.get("locked_until")
    if not lu:
        return None
    if isinstance(lu, str):
        lu = datetime.strptime(lu[:19], "%Y-%m-%d %H:%M:%S")
    if lu.tzinfo is None:  # pymysql 返回 naive，按 db 约定即上海时间
        lu = lu.replace(tzinfo=CST)
    if lu > _now():
        remain = lu - _now()
        return (f"账号已锁定，请约 {remain.seconds // 3600} 小时 "
                f"{(remain.seconds % 3600) // 60} 分钟后重试")
    return None


def record_fail(user_row: dict) -> bool:
    """密码错误：计数 +1；达上限 → 写锁定时间并清零计数（锁定期满给新一轮额度）。
    返回是否刚触发锁定。"""
    failed = int(user_row.get("failed_attempts") or 0) + 1
    if failed >= config.LOCKOUT_MAX_FAIL:
        locked = (_now() + timedelta(seconds=config.LOCKOUT_SEC)
                  ).strftime("%Y-%m-%d %H:%M:%S")
        db.record_login_fail(user_row["id"], locked_until=locked)
        return True
    db.record_login_fail(user_row["id"])
    return False


class IpThrottle:
    """IP 维度限流：Redis 滑动窗口（多 worker 共享），Redis 挂自动降级进程内存。

    health=None 即纯内存模式（本地/离线）；main.py lifespan 注入 RedisHealth
    （探测线程 + 各 Redis 实现共用的降级开关，同 captcha/s7 快照）。
    """

    _FKEY = "fin:iplimit:{ip}"    # 登录/注册失败时间戳 ZSET（滑动窗口）
    _BKEY = "fin:ipblock:{ip}"    # 封禁标记（TTL=IP_BLOCK_SEC）
    _CKEY = "fin:caplimit:{ip}"   # 验证码生成时间戳 ZSET（60s 窗口）

    def __init__(self, health=None):
        self.metrics = None
        self._health = health          # RedisHealth（None = 纯内存模式）
        self._r = None
        self._lock = threading.Lock()
        self._fails: dict = {}         # 内存兜底：ip -> [失败时间戳]（窗口内）
        self._blocks: dict = {}        # 内存兜底：ip -> 封禁截止时间戳
        self._captcha: dict = {}       # 内存兜底：ip -> [验证码生成时间戳]

    def set_health(self, health) -> None:
        """main.py lifespan 在 RedisHealth 探测线程就绪后注入。"""
        self._health = health

    def _client(self):
        if self._r is None:
            self._r = redis.Redis.from_url(config.REDIS_URL,
                                           socket_connect_timeout=2, socket_timeout=2)
        return self._r

    def _down(self) -> None:
        """Redis 调用失败：丢连接缓存并广播降级（探测线程恢复后自动切回）。"""
        self._r = None
        if self._health is not None:
            self._health.mark_down()

    def _inc(self, name: str) -> None:
        if self.metrics:
            self.metrics.inc(name)

    @staticmethod
    def _zadd(pipe, key: str, now: float) -> None:
        """记一次事件时间戳。member 加随机后缀：防同纳秒两个事件在 ZSET 里
        被去重（去重会把计数吞掉 1，限流就漏了一格）。"""
        pipe.zadd(key, {f"{time.time_ns()}:{secrets.token_hex(4)}": now})

    # ---------- 登录/注册 ----------

    def throttle_login(self, ip: str) -> None:
        """封禁中 → 403。Redis 优先，失败降级内存。"""
        if self._health is not None and self._health.available:
            try:
                if self._client().get(self._BKEY.format(ip=ip)):
                    self._inc("ip_blocked")
                    raise HTTPException(403, "尝试过于频繁，请稍后再试")
                return
            except HTTPException:
                raise
            except Exception:  # noqa: BLE001 — Redis 挂：内存兜底
                self._down()
        now = time.time()
        with self._lock:
            until = self._blocks.get(ip)
            if until and until > now:
                self._inc("ip_blocked")
                raise HTTPException(403, "尝试过于频繁，请稍后再试")
            if until:
                del self._blocks[ip]

    def note_login_fail(self, ip: str) -> None:
        """窗口内失败 +1；达上限 → 封 IP_BLOCK_SEC 并清计数（期满给全新窗口）。"""
        if self._health is not None and self._health.available:
            try:
                self._redis_fail(ip)
                return
            except Exception:  # noqa: BLE001
                self._down()
        now = time.time()
        with self._lock:
            ts = [t for t in self._fails.get(ip, []) if now - t < config.IP_WINDOW_SEC]
            ts.append(now)
            self._fails[ip] = ts
            if len(ts) >= config.IP_FAIL_LIMIT:
                self._blocks[ip] = now + config.IP_BLOCK_SEC
                del self._fails[ip]
                self._inc("ip_blocked")

    def _redis_fail(self, ip: str) -> None:
        """ZSET 滑动窗口：修剪窗口外旧时间戳 → 记本次 → 计数（一次 pipeline 往返）。
        超限 → TTL 封禁标记 + 清计数（语义与原内存实现逐条一致）。"""
        now = time.time()
        fkey, bkey = self._FKEY.format(ip=ip), self._BKEY.format(ip=ip)
        pipe = self._client().pipeline(transaction=False)
        pipe.zremrangebyscore(fkey, 0, now - config.IP_WINDOW_SEC)
        self._zadd(pipe, fkey, now)
        pipe.zcard(fkey)
        pipe.expire(fkey, config.IP_WINDOW_SEC)
        if pipe.execute()[2] >= config.IP_FAIL_LIMIT:
            self._client().set(bkey, "1", ex=config.IP_BLOCK_SEC)
            self._client().delete(fkey)
            self._inc("ip_blocked")

    def note_login_ok(self, ip: str) -> None:
        """登录/注册成功：清零该 IP 的失败计数与封禁（能过密码+验证码说明不是盲扫）。"""
        if self._health is not None and self._health.available:
            try:
                self._client().delete(self._FKEY.format(ip=ip),
                                      self._BKEY.format(ip=ip))
                return
            except Exception:  # noqa: BLE001
                self._down()
        with self._lock:
            self._fails.pop(ip, None)
            self._blocks.pop(ip, None)

    # ---------- 验证码生成 ----------

    def throttle_captcha(self, ip: str) -> None:
        """每 IP 60s 内生成超 CAPTCHA_IP_LIMIT 张 → 429（防洪泛喂 OCR 农场）。"""
        if self._health is not None and self._health.available:
            try:
                self._redis_captcha(ip)
                return
            except HTTPException:
                raise
            except Exception:  # noqa: BLE001 — Redis 挂：内存兜底
                self._down()
        now = time.time()
        with self._lock:
            ts = [t for t in self._captcha.get(ip, []) if now - t < 60]
            if len(ts) >= config.CAPTCHA_IP_LIMIT:
                self._inc("captcha_blocked")
                raise HTTPException(429, "验证码获取过于频繁，请稍后再试")
            ts.append(now)
            self._captcha[ip] = ts

    def _redis_captcha(self, ip: str) -> None:
        now = time.time()
        ckey = self._CKEY.format(ip=ip)
        pipe = self._client().pipeline(transaction=False)
        pipe.zremrangebyscore(ckey, 0, now - 60)
        self._zadd(pipe, ckey, now)
        pipe.zcard(ckey)
        pipe.expire(ckey, 60)
        if pipe.execute()[2] >= config.CAPTCHA_IP_LIMIT:
            self._inc("captcha_blocked")
            raise HTTPException(429, "验证码获取过于频繁，请稍后再试")


ip_throttle = IpThrottle()
