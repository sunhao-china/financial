# -*- coding: utf-8 -*-
"""S7 快照 Redis 实现（session_store.py 头部预留，§11）+ Redis 可用性探测器。

- RedisHealth：后台线程每 PROBE_INTERVAL_SEC ping 一次；宕机 → 各 Redis 实现自动
  降级进程内 Memory，恢复后自动切回（§13.3 降级矩阵）
- RedisSessionStore：key = fin:s7:{query_id}，SETEX 7 天；S7 快照仅诊断用，
  Redis 不可用不阻塞问答（快照落内存，进程重启即丢，可接受）
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import config
from corpus.session_store import MemorySessionStore, SessionStore, TTL_SECONDS
from schemas.models import S7Snapshot

try:
    import redis
except ImportError:  # 本地离线测试无 redis 包：Redis 实现全部走 Memory 降级
    redis = None


class RedisHealth:
    """Redis 可用性状态（供所有 Redis 实现共享的降级开关）。"""

    def __init__(self, interval: int = None):
        self._lock = threading.Lock()
        self.available = False
        self._t = threading.Thread(target=self._loop, args=(interval or config.PROBE_INTERVAL_SEC,),
                                   daemon=True, name="redis-probe")
        self._t.start()

    def mark_down(self) -> None:
        with self._lock:
            self.available = False

    def _loop(self, interval: int) -> None:
        while True:
            try:
                if redis is None:
                    raise RuntimeError("redis 包未安装")
                r = redis.Redis.from_url(config.REDIS_URL,
                                         socket_connect_timeout=2, socket_timeout=2)
                r.ping()
                with self._lock:
                    self.available = True
            except Exception:  # noqa: BLE001
                with self._lock:
                    self.available = False
            time.sleep(interval)


class RedisSessionStore(SessionStore):
    """S7 快照 Redis 实现：key = fin:s7:{query_id}，SETEX 7 天（§11）。
    Redis 不可用自动降级 MemorySessionStore。"""

    def __init__(self, health: RedisHealth):
        self._health = health
        self._mem = MemorySessionStore()
        self._r = None

    def _client(self):
        if self._r is None:
            self._r = redis.Redis.from_url(config.REDIS_URL,
                                           socket_connect_timeout=2, socket_timeout=2)
        return self._r

    def save(self, snap: S7Snapshot) -> None:
        S7Snapshot.model_validate(snap.model_dump())   # 写前强制校验（§9.2）
        if not self._health.available:
            return self._mem.save(snap)
        try:
            self._client().setex(f"fin:s7:{snap.query_id}", TTL_SECONDS, snap.model_dump_json())
        except Exception:  # noqa: BLE001
            self._r = None
            self._health.mark_down()
            self._mem.save(snap)

    def get(self, query_id: str) -> Optional[S7Snapshot]:
        if not self._health.available:
            return self._mem.get(query_id)
        try:
            raw = self._client().get(f"fin:s7:{query_id}")
            return S7Snapshot.model_validate_json(raw) if raw else None
        except Exception:  # noqa: BLE001
            self._r = None
            self._health.mark_down()
            return self._mem.get(query_id)

    def delete(self, query_id: str) -> None:
        if not self._health.available:
            return self._mem.delete(query_id)
        try:
            self._client().delete(f"fin:s7:{query_id}")
        except Exception:  # noqa: BLE001
            self._r = None
            self._health.mark_down()
            self._mem.delete(query_id)
