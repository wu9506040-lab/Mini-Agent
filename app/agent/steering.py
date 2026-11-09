"""简版 Steering（借鉴 pi 的转向机制）：

Agent 忙碌（执行工具循环）期间，用户插入的新消息先排队；
runtime 在每轮工具执行完毕后检查队列，把插入消息注入历史再继续 loop。
对应架构题模块四第 2 题（busy session 收到新消息怎么处理）。
"""
import threading
from typing import Dict, List


class SteeringQueue:
    def __init__(self):
        self._queues: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    def push(self, session_id: str, message: str) -> None:
        with self._lock:
            self._queues.setdefault(session_id, []).append(message)

    def pop_all(self, session_id: str) -> List[str]:
        with self._lock:
            messages = self._queues.get(session_id)
            if not messages:
                return []
            self._queues[session_id] = []
            return messages
