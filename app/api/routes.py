"""HTTP 端点。"""
import logging

from fastapi import APIRouter, HTTPException

from .schemas import (ChatRequest, ChatResponse, HistoryResponse,
                      SessionCreated, SessionInfo, SteerRequest,
                      SteerResponse, ToolCallRecordOut, TraceResponse)

logger = logging.getLogger("mini-agent.api")


def create_router(store, runtime, steering, settings) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.post("/sessions", response_model=SessionCreated)
    def create_session():
        sid = store.create_session()
        return SessionCreated(session_id=sid)

    @router.get("/sessions", response_model=list[SessionInfo])
    def list_sessions():
        return [SessionInfo(**s) for s in store.list_sessions()]

    @router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
    def get_history(session_id: str):
        session = store.load_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return HistoryResponse(session_id=session_id, messages=session["messages"])

    @router.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        try:
            reply = await runtime.run(req.session_id, req.message)
        except Exception as e:  # LLM 连接失败等致命错误 → 502
            logger.exception("chat 失败 session=%s", req.session_id)
            raise HTTPException(status_code=502, detail=f"Agent 执行失败: {e}") from e
        return ChatResponse(
            session_id=req.session_id,
            reply=reply.content,
            tool_calls=[
                ToolCallRecordOut(
                    name=r.name, args=r.args, result=r.result,
                    duration_ms=r.duration_ms, round=r.round, status=r.status,
                )
                for r in reply.tool_calls
            ],
            rounds_used=reply.rounds_used,
            truncated=reply.truncated,
        )

    @router.post("/sessions/{session_id}/steer", response_model=SteerResponse)
    def steer(session_id: str, req: SteerRequest):
        """Agent 忙碌期间的插入消息：进入队列，工具轮间注入。"""
        if store.load_session(session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        steering.push(session_id, req.message)
        return SteerResponse(session_id=session_id, queued=True)

    @router.get("/sessions/{session_id}/trace", response_model=TraceResponse)
    def get_trace(session_id: str):
        return TraceResponse(session_id=session_id, events=store.get_trace(session_id))

    @router.get("/tools")
    def list_tools():
        return runtime.registry.schemas()

    @router.get("/healthz")
    def healthz():
        return {"status": "ok", "model": settings.model, "base_url": settings.base_url}

    return router
