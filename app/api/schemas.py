"""Pydantic 请求/响应模型。"""
from typing import List, Optional

from pydantic import BaseModel, Field


class SessionCreated(BaseModel):
    session_id: str


class SessionInfo(BaseModel):
    session_id: str
    updated_at: float
    preview: str


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="目标会话 ID（多窗口各持一个）")
    message: str = Field(..., min_length=1, description="用户输入")


class ToolCallRecordOut(BaseModel):
    name: str
    args: dict
    result: str
    duration_ms: int
    round: int
    status: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_calls: List[ToolCallRecordOut]
    rounds_used: int
    truncated: bool


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[dict]


class TraceResponse(BaseModel):
    session_id: str
    events: List[dict]


class SteerRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Agent 忙碌期间插入的转向消息")


class SteerResponse(BaseModel):
    session_id: str
    queued: bool
