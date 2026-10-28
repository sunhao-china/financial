# -*- coding: utf-8 -*-
"""MySQL 持久层（用户系统 + 会话历史，2026-10-19 加）。

设计（§14.5）：
- workers=1 → 单连接 + ping 重连足够，不引连接池
- 启动时自动建表（CREATE TABLE IF NOT EXISTS，utf8mb4）
- 全部参数化查询（pymysql %s），无拼接
- 标题提炼为纯规则（首问去空白截 20 字），不额外消耗 LLM token
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import config

try:
    import pymysql
except ImportError:  # 本地无 MySQL 包时仍可 import（问答走 Memory 降级，登录不可用）
    pymysql = None

# ---------- 表结构（启动自动建） ----------

DDL = [
    """CREATE TABLE IF NOT EXISTS users (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        username VARCHAR(64) NOT NULL UNIQUE,
        pass_hash VARCHAR(256) NOT NULL,
        salt VARCHAR(64) NOT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        failed_attempts INT NOT NULL DEFAULT 0,
        locked_until DATETIME NULL
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS conversations (
        id VARCHAR(36) PRIMARY KEY,
        user_id BIGINT NOT NULL,
        title VARCHAR(64) NOT NULL DEFAULT '',
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        KEY idx_user_updated (user_id, updated_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS messages (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        conv_id VARCHAR(36) NOT NULL,
        role VARCHAR(10) NOT NULL,
        content TEXT NOT NULL,
        ts DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
        KEY idx_conv_id (conv_id, id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]


def make_title(question: str, limit: int = 20) -> str:
    """会话标题：首问去空白截断（纯规则，不消耗 LLM token）。"""
    q = " ".join(question.split())
    return q[:limit] + ("…" if len(q) > limit else "") or "新会话"


# 历史时间统一上海时间（§14.5 时区修复）：Python 显式写 CST，不依赖系统/MySQL 容器时区。
# 前端按 +08:00 解析（server 契约 = 上海时间字符串，无时区标记）。
CST = timezone(timedelta(hours=8))


def _now_cst() -> str:
    """上海时间 'YYYY-MM-DD HH:MM:SS.mmm'（DATETIME(3) 列格式）。"""
    return datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class DB:
    """pymysql 薄封装：懒连接、失败重连一次、查询/写入辅助。"""

    def __init__(self):
        self._conn = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """供 health 上报与降级判断（不主动建连）。"""
        return pymysql is not None and self._conn is not None

    def _connect(self):
        return pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            database=config.MYSQL_DB, charset="utf8mb4",
            connect_timeout=3, read_timeout=5, write_timeout=5,
            autocommit=True)

    def _ensure(self):
        if self._conn is None:
            self._conn = self._connect()
        try:
            self._conn.ping(reconnect=True)
        except Exception:  # noqa: BLE001 — 断线重连一次
            try:
                self._conn = self._connect()
            except Exception:  # noqa: BLE001 — 仍失败则置空，走 Memory 降级
                self._conn = None
                raise
        return self._conn

    def init_schema(self) -> bool:
        """建表 + 预置管理员。成功返回 True；失败静默（服务照常起，登录暂不可用）。"""
        try:
            with self._lock:
                cur = self._ensure().cursor()
                for ddl in DDL:
                    cur.execute(ddl)
                # 幂等补列（2026-10-22 安全加固）：存量表缺锁定列时 ALTER。
                # MySQL 8 无 ADD COLUMN IF NOT EXISTS，只能先查 information_schema。
                self._ensure_column(cur, "users", "failed_attempts",
                                    "INT NOT NULL DEFAULT 0")
                self._ensure_column(cur, "users", "locked_until", "DATETIME NULL")
                cur.close()
            return True
        except Exception:  # noqa: BLE001
            self._conn = None
            return False

    @staticmethod
    def _ensure_column(cur, table: str, col: str, ddl: str) -> None:
        cur.execute(
            "SELECT COUNT(*) AS n FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME=%s AND COLUMN_NAME=%s",
            (table, col))
        if cur.fetchone()[0] == 0:
            cur.execute(f"ALTER TABLE `{table}` ADD COLUMN `{col}` {ddl}")

    # ---------- 查询/写入辅助 ----------

    def query(self, sql: str, args: tuple) -> List[dict]:
        with self._lock:
            cur = self._ensure().cursor()
            cur.execute(sql, args)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            cur.close()
        return [dict(zip(cols, r)) for r in rows]

    def execute(self, sql: str, args: tuple) -> int:
        with self._lock:
            cur = self._ensure().cursor()
            cur.execute(sql, args)
            cur.close()
        return cur.rowcount

    # ---------- users ----------

    def get_user(self, username: str) -> Optional[dict]:
        rows = self.query("SELECT id, username, pass_hash, salt, failed_attempts, locked_until "
                          "FROM users WHERE username=%s", (username,))
        return rows[0] if rows else None

    def create_user(self, username: str, pass_hash: str, salt: str) -> int:
        self.execute("INSERT INTO users (username, pass_hash, salt) VALUES (%s, %s, %s)",
                     (username, pass_hash, salt))
        return int(self.query("SELECT id FROM users WHERE username=%s", (username,))[0]["id"])

    # ---------- 登录安全（2026-10-22）：锁定计数与机会式哈希升级 ----------

    def update_password(self, user_id: int, pass_hash: str, salt: str) -> None:
        self.execute("UPDATE users SET pass_hash=%s, salt=%s WHERE id=%s",
                     (pass_hash, salt, user_id))

    def record_login_fail(self, user_id: int, locked_until: str = None) -> None:
        """密码错误 +1；命中上限时写锁定时间并清零计数（锁定期满给新一轮额度）。"""
        if locked_until:
            self.execute("UPDATE users SET failed_attempts=0, locked_until=%s WHERE id=%s",
                         (locked_until, user_id))
        else:
            self.execute("UPDATE users SET failed_attempts=failed_attempts+1 WHERE id=%s",
                         (user_id,))

    def reset_login_fail(self, user_id: int) -> None:
        self.execute("UPDATE users SET failed_attempts=0, locked_until=NULL WHERE id=%s",
                     (user_id,))

    # ---------- conversations / messages ----------

    def upsert_conversation(self, conv_id: str, user_id: int, title: str) -> None:
        """首条消息懒落库：已存在只刷 updated_at（title 保留首次设置）。
        created/updated 显式写上海时间（§14.5），不依赖 MySQL DEFAULT CURRENT_TIMESTAMP。"""
        now = _now_cst()
        self.execute(
            "INSERT INTO conversations (id, user_id, title, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE updated_at=%s",
            (conv_id, user_id, title, now, now, now))

    def list_conversations(self, user_id: int) -> List[dict]:
        return self.query(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE user_id=%s ORDER BY updated_at DESC LIMIT 100",
            (user_id,))

    def get_messages(self, conv_id: str, limit: int = 40) -> List[dict]:
        rows = self.query(
            "SELECT role, content, ts FROM messages WHERE conv_id=%s "
            "ORDER BY id DESC LIMIT %s", (conv_id, limit))
        return [{"role": r["role"], "content": r["content"],
                 "ts": r["ts"].strftime("%Y-%m-%d %H:%M:%S")} for r in reversed(rows)]

    def append_messages(self, conv_id: str, question: str, answer: str) -> None:
        """两条消息同 ts，显式写上海时间（§14.5：原 datetime.now() 依赖系统时区，
        conversations 由 MySQL DEFAULT 生成却是容器 UTC，两者差 8 小时）。"""
        now = _now_cst()
        self.execute(
            "INSERT INTO messages (conv_id, role, content, ts) VALUES "
            "(%s, 'user', %s, %s), (%s, 'assistant', %s, %s)",
            (conv_id, question, now, conv_id, answer, now))

    def message_count(self, conv_id: str) -> int:
        rows = self.query("SELECT COUNT(*) AS n FROM messages WHERE conv_id=%s", (conv_id,))
        return int(rows[0]["n"]) if rows else 0

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None


db = DB()
