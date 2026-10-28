# -*- coding: utf-8 -*-
"""多轮对话历史（§14）：Redis + Memory 双实现 + merge/trim 规则。

- Redis：fin:conv:{session_id} LIST（RPUSH user/assistant 两条 → LTRIM -20 -1 →
  EXPIRE 7 天滑动）；不可用自动降级 Memory（重启即丢，服务不挂）
- 合并规则：client 显式 history 非空则以其为准；否则读 server 端会话历史；
  一律过 trim_history；请求结束 append（拒答话术也入史，供指代消解）
- V1 不做 LLM 摘要（HISTORY_SUMMARIZE=False 预留）：下游只消费末 3~6 轮，
  裁剪 + HISTORY_MAX_CHARS 上限足够
"""
from __future__ import annotations

import json
import threading
import time
from typing import Dict, List, Optional

import config
from server.db import make_title

try:
    import redis
except ImportError:  # 本地离线测试无 redis 包：Redis 实现全部走 Memory 降级
    redis = None


class ChatHistoryStore:
    """多轮会话历史接口。history 条目格式：{"role": "user"|"assistant", "content": str}。
    user_id 仅 MySQL 实现用于会话归属（懒落库 conversations），其余实现忽略。"""

    def get(self, session_id: str) -> List[dict]:
        raise NotImplementedError

    def append(self, session_id: str, question: str, answer: str, user_id: int = 0) -> None:
        raise NotImplementedError

    def clear(self, session_id: str) -> None:
        raise NotImplementedError


class MemoryChatHistory(ChatHistoryStore):
    """进程内实现：dict + 写入时间戳，TTL 惰性清理（与 MemorySessionStore 同风格）。"""

    def __init__(self, ttl: int = 7 * 24 * 3600):
        self._ttl = ttl
        self._data: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> List[dict]:
        with self._lock:
            ent = self._data.get(session_id)
        if not ent:
            return []
        msgs, ts = ent
        if time.time() - ts > self._ttl:
            with self._lock:
                self._data.pop(session_id, None)
            return []
        return list(msgs)

    def append(self, session_id: str, question: str, answer: str, user_id: int = 0) -> None:
        with self._lock:
            ent = self._data.get(session_id)
            msgs = list(ent[0]) if ent else []
            msgs.append({"role": "user", "content": question})
            msgs.append({"role": "assistant", "content": answer})
            msgs = msgs[-(config.HISTORY_MAX_TURNS * 2):]
            self._data[session_id] = (msgs, time.time())

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)


class RedisChatHistory(ChatHistoryStore):
    """Redis 实现：fin:conv:{session_id} LIST，RPUSH 保序（旧→新），LTRIM 保留末 20 条。
    Redis 不可用降级 Memory（写入两边都记，恢复后 Redis 侧从 Memory 补齐）。"""

    def __init__(self, health):
        self._health = health
        self._mem = MemoryChatHistory()
        self._r = None

    def _client(self):
        if self._r is None:
            self._r = redis.Redis.from_url(config.REDIS_URL,
                                           socket_connect_timeout=2, socket_timeout=2)
        return self._r

    @staticmethod
    def _key(session_id: str) -> str:
        return f"fin:conv:{session_id}"

    def get(self, session_id: str) -> List[dict]:
        if not self._health.available:
            return self._mem.get(session_id)
        try:
            raw = self._client().lrange(self._key(session_id), 0, -1)
            return [json.loads(x) for x in raw]
        except Exception:  # noqa: BLE001
            self._r = None
            self._health.mark_down()
            return self._mem.get(session_id)

    def append(self, session_id: str, question: str, answer: str, user_id: int = 0) -> None:
        self._mem.append(session_id, question, answer)   # Memory 始终记（降级兜底）
        if not self._health.available:
            return
        try:
            pipe = self._client().pipeline()
            pipe.rpush(self._key(session_id),
                       json.dumps({"role": "user", "content": question}, ensure_ascii=False),
                       json.dumps({"role": "assistant", "content": answer}, ensure_ascii=False))
            pipe.ltrim(self._key(session_id), -config.HISTORY_MAX_TURNS * 2, -1)
            pipe.expire(self._key(session_id), 7 * 24 * 3600)
            pipe.execute()
        except Exception:  # noqa: BLE001
            self._r = None
            self._health.mark_down()

    def clear(self, session_id: str) -> None:
        self._mem.clear(session_id)
        if not self._health.available:
            return
        try:
            self._client().delete(self._key(session_id))
        except Exception:  # noqa: BLE001
            self._r = None
            self._health.mark_down()


class MySQLChatHistory(ChatHistoryStore):
    """MySQL 主存储（2026-10-19 加）：messages 表即历史，conversations 懒落库。
    Memory 始终兜底（MySQL 挂 → 本进程内历史仍可续答；重启后丢，服务不挂）。"""

    def __init__(self, db):
        self._db = db
        self._mem = MemoryChatHistory()

    def get(self, session_id: str) -> List[dict]:
        try:
            rows = self._db.get_messages(session_id, limit=config.HISTORY_MAX_TURNS * 2)
            return [{"role": r["role"], "content": r["content"]} for r in rows]
        except Exception:  # noqa: BLE001 — MySQL 挂：Memory 兜底
            return self._mem.get(session_id)

    def append(self, session_id: str, question: str, answer: str, user_id: int = 0) -> None:
        self._mem.append(session_id, question, answer)
        try:
            # 懒落库：首条消息建 conversations 行（title=首问截断；已存在只刷 updated_at）
            if user_id:
                self._db.upsert_conversation(session_id, user_id,
                                             make_title(question))
            self._db.append_messages(session_id, question, answer)
        except Exception:  # noqa: BLE001
            pass

    def clear(self, session_id: str) -> None:
        self._mem.clear(session_id)


def merge_history(client_history: List[dict], server_history: List[dict]) -> List[dict]:
    """合并规则（§14）：client 显式 history 优先；否则 server 端会话历史。返回新 list。"""
    return list(client_history) if client_history else list(server_history)


def trim_history(msgs: List[dict], max_turns: Optional[int] = None,
                 max_chars: Optional[int] = None) -> List[dict]:
    """裁剪（§14）：过滤非对话角色 → 保留末 max_turns 轮 → 总字符超 max_chars 从最旧裁剪。"""
    max_turns = max_turns or config.HISTORY_MAX_TURNS
    max_chars = max_chars or config.HISTORY_MAX_CHARS
    msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)]
    out = msgs[-(max_turns * 2):]
    total = sum(len(m["content"]) for m in out)
    while len(out) > 1 and total > max_chars:
        total -= len(out[0]["content"])
        out = out[1:]
    return out
