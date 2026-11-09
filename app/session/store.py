"""Session 持久化层（SQLite，标准库 sqlite3，不引 ORM）。

三张表：
- sessions：完整消息历史（JSON）+ memory（摘要缓存）——原始数据不丢，
  context 是每次渲染出来的视图，压缩只发生在构建 prompt 时。
- todos：结构化记忆，挂 session_id，多窗口天然隔离。
- trace：工具调用 / LLM 请求响应的结构化事件日志。

并发：单机 demo 用全局锁保护写入；README 注明生产应换 aiosqlite/连接池。
"""
import json
import os
import sqlite3
import threading
import time
import uuid
from typing import List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    messages   TEXT NOT NULL DEFAULT '[]',
    memory     TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS todos (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    content    TEXT NOT NULL,
    priority   TEXT NOT NULL DEFAULT 'low',
    status     TEXT NOT NULL DEFAULT 'open',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS trace (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ts         REAL NOT NULL,
    event      TEXT NOT NULL,
    payload    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_todos_session ON todos(session_id);
CREATE INDEX IF NOT EXISTS idx_trace_session ON trace(session_id);
"""


class SessionStore:
    def __init__(self, db_path: str):
        directory = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ---- sessions ----

    def create_session(self, session_id: Optional[str] = None) -> str:
        sid = session_id or f"s-{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
                (sid, now, now),
            )
            self._conn.commit()
        return sid

    def load_session(self, session_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT session_id, messages, memory FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row[0],
            "messages": json.loads(row[1]),
            "memory": json.loads(row[2]),
        }

    def save_session(self, session_id: str, messages: List[dict], memory: Optional[dict] = None) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """INSERT INTO sessions (session_id, created_at, updated_at, messages, memory)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     updated_at = excluded.updated_at,
                     messages = excluded.messages,
                     memory = excluded.memory""",
                (session_id, now, now, json.dumps(messages, ensure_ascii=False),
                 json.dumps(memory or {}, ensure_ascii=False)),
            )
            self._conn.commit()

    def list_sessions(self) -> List[dict]:
        rows = self._conn.execute(
            "SELECT session_id, updated_at, messages FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        result = []
        for sid, updated_at, messages_json in rows:
            messages = json.loads(messages_json)
            preview = next(
                (m.get("content", "") for m in messages if m.get("role") == "user"), ""
            )
            result.append({
                "session_id": sid,
                "updated_at": updated_at,
                "preview": (preview or "")[:50],
            })
        return result

    # ---- todos ----

    def add_todo(self, session_id: str, content: str, priority: str = "low") -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO todos (session_id, content, priority, status, created_at) VALUES (?, ?, ?, 'open', ?)",
                (session_id, content, priority, now),
            )
            self._conn.commit()
            return cur.lastrowid

    def list_todos(self, session_id: str, include_done: bool = False) -> List[dict]:
        sql = "SELECT id, content, priority, status FROM todos WHERE session_id = ?"
        if not include_done:
            sql += " AND status = 'open'"
        sql += " ORDER BY id"
        rows = self._conn.execute(sql, (session_id,)).fetchall()
        return [
            {"id": r[0], "content": r[1], "priority": r[2], "status": r[3]}
            for r in rows
        ]

    def complete_todo(self, session_id: str, todo_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE todos SET status = 'done' WHERE session_id = ? AND id = ? AND status = 'open'",
                (session_id, todo_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    # ---- trace ----

    def log_trace(self, session_id: str, event: str, payload: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO trace (session_id, ts, event, payload) VALUES (?, ?, ?, ?)",
                (session_id, time.time(), event, json.dumps(payload, ensure_ascii=False, default=str)),
            )
            self._conn.commit()

    def get_trace(self, session_id: str, limit: int = 200) -> List[dict]:
        rows = self._conn.execute(
            "SELECT ts, event, payload FROM trace WHERE session_id = ? ORDER BY id LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [{"ts": r[0], "event": r[1], "payload": json.loads(r[2])} for r in rows]

    def close(self) -> None:
        self._conn.close()
