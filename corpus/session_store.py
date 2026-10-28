# -*- coding: utf-8 -*-
"""S7 会话记忆存储（大纲 §11）：调用路径快照，7 天排障窗口。

V1 用进程内 dict 实现（开发期单机/评测够用）；P5 服务化时新增 Redis 实现
（redis-py，key = query_id，EXPIRE 7 天），上层只依赖 SessionStore 接口，
save/get/delete 语义不变。写入前按 S7 结构强制校验（§9.2 双向校验）。
"""
import json
import threading
import time
from typing import Dict, Optional

from schemas.models import S7Snapshot

TTL_SECONDS = 7 * 24 * 3600   # §11：7 天排障窗口


class SessionStore:
    """S7 存储接口。"""

    def save(self, snap: S7Snapshot) -> None:
        raise NotImplementedError

    def get(self, query_id: str) -> Optional[S7Snapshot]:
        raise NotImplementedError

    def delete(self, query_id: str) -> None:
        raise NotImplementedError


class MemorySessionStore(SessionStore):
    """进程内实现：dict + 写入时间戳；读取时超过 TTL 视为过期并惰性清理。

    评测/冒烟脚本可复用同一实例并在结束时 dump 全部快照排障。
    """

    def __init__(self, ttl: int = TTL_SECONDS):
        self._ttl = ttl
        self._data: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    def save(self, snap: S7Snapshot) -> None:
        S7Snapshot.model_validate(snap.model_dump())   # 写前强制校验（§9.2）
        with self._lock:
            self._data[snap.query_id] = (snap, time.time())

    def get(self, query_id: str) -> Optional[S7Snapshot]:
        with self._lock:
            ent = self._data.get(query_id)
        if not ent:
            return None
        snap, ts = ent
        if time.time() - ts > self._ttl:
            with self._lock:
                self._data.pop(query_id, None)
            return None
        return snap

    def delete(self, query_id: str) -> None:
        with self._lock:
            self._data.pop(query_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def dump(self, path) -> None:
        """评测结束时把全部快照落盘（JSON Lines，S7 结构）。"""
        with self._lock:
            snaps = [s.model_dump() for s, _ in self._data.values()]
        with open(path, "w", encoding="utf-8") as f:
            for s in snaps:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"已落盘 {len(snaps)} 条 S7 快照 → {path}")
