"""FastAPI 应用入口。

app 工厂模式：create_app(llm_client=None, db_path=None)。
- 生产：uvicorn app.main:app，启动时按环境变量创建真实 LLMClient。
- 测试：注入 FakeLLMClient + 临时 db，全链路离线可测。
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .agent.runtime import AgentRuntime
from .agent.steering import SteeringQueue
from .api.routes import create_router
from .config import Settings
from .session.store import SessionStore
from .tools import init_todo_store, registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


def create_app(llm_client=None, db_path: str = None) -> FastAPI:
    settings = Settings.from_env()
    store = SessionStore(db_path or settings.db_path)
    init_todo_store(store)
    steering = SteeringQueue()

    runtime = AgentRuntime(
        llm=llm_client,               # None → startup 时按环境变量创建真实客户端
        registry=registry,
        store=store,
        max_rounds=settings.max_rounds,
        steering_queue=steering,
    )
    # 接线：让 .env 里的两个压缩参数真正生效（否则只是死配置）
    runtime.context.max_history = settings.max_history_messages
    runtime.context.recent_keep = settings.recent_keep

    @asynccontextmanager
    async def lifespan(a: FastAPI):
        if runtime.llm is None:
            from .llm.client import LLMClient

            runtime.llm = LLMClient(settings)  # 缺 key 时在这里给出明确报错
        yield

    app = FastAPI(title="Mini Agent",
                  description="从零实现的最小可用 Agent Runtime",
                  lifespan=lifespan)
    app.state.store = store
    app.state.runtime = runtime
    app.state.settings = settings

    app.include_router(create_router(store, runtime, steering, settings))
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


app = create_app()
