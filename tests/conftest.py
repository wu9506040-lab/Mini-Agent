"""pytest 公共设施：FakeLLMClient（脚本回放）+ fixture。

核心策略：mock llm/client.py 这一层，Runtime 及以上全部逻辑离线可测。
脚本项格式：
  {"content": "文本"}                       —— 纯文本最终回答
  {"tool": (name, args_dict)}               —— 发起一次工具调用
  {"raw_tool": (name, "非法JSON字符串")}     —— 参数 JSON 非法的工具调用
  {"thought": "...", "tool": (...)}         —— 带思考过程的工具调用
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.runtime import AgentRuntime  # noqa: E402
from app.llm.client import LLMResponse, ToolCall  # noqa: E402
from app.session.store import SessionStore  # noqa: E402
from app.tools import init_todo_store, registry  # noqa: E402


def make_llm_response(content="", tool_calls=None):
    tool_calls = tool_calls or []
    raw = {"role": "assistant", "content": content}
    if tool_calls:
        raw["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in tool_calls
        ]
    return LLMResponse(content=content, tool_calls=tool_calls, raw_message=raw)


class FakeLLMClient:
    """按脚本依次返回预设响应，记录每次收到的 messages/tools 供断言。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def chat(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        if not self.script:
            raise AssertionError("FakeLLM 脚本耗尽，收到的最后一条消息: "
                                 + json.dumps(messages[-1], ensure_ascii=False)[:300])
        item = self.script.pop(0)

        if "tool" in item or "raw_tool" in item:
            if "raw_tool" in item:
                name, raw_args = item["raw_tool"]
                tc = ToolCall(id=item.get("id", "call_1"), name=name, arguments=raw_args)
            else:
                name, args = item["tool"]
                tc = ToolCall(id=item.get("id", "call_1"),
                              name=name,
                              arguments=json.dumps(args, ensure_ascii=False))
            content = item.get("thought", "")
            return make_llm_response(content=content, tool_calls=[tc])

        return make_llm_response(content=item.get("content", ""))


@pytest.fixture()
def store(tmp_path):
    s = SessionStore(str(tmp_path / "test.db"))
    init_todo_store(s)
    yield s
    s.close()


@pytest.fixture()
def make_runtime(store):
    """工厂：每个测试自带脚本与参数，互不干扰。"""

    def _make(script, **kwargs):
        llm = FakeLLMClient(script)
        runtime = AgentRuntime(
            llm=llm, registry=registry, store=store,
            max_rounds=kwargs.pop("max_rounds", 8), **kwargs,
        )
        return runtime, llm

    return _make


@pytest.fixture()
def make_client(tmp_path):
    """API 全链路测试：create_app 注入 FakeLLM + 临时 db。"""

    def _make(script):
        from fastapi.testclient import TestClient

        from app.main import create_app

        llm = FakeLLMClient(script)
        db = str(tmp_path / "api.db")
        app = create_app(llm_client=llm, db_path=db)
        return TestClient(app), llm, app.state.store

    return _make
