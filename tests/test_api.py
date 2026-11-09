"""API 端到端测试（FakeLLM 注入，全链路离线）。"""
from app.config import Settings
from app.llm.client import LLMClient


def test_full_chat_flow_with_tool(make_client):
    client, llm, store = make_client([
        {"tool": ("calculator", {"expression": "2+3"})},
        {"content": "2 加 3 等于 5"},
        {"content": "在呢"},
    ])
    sid = client.post("/api/sessions").json()["session_id"]

    resp = client.post("/api/chat", json={"session_id": sid, "message": "2+3等于多少"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "2 加 3 等于 5"
    assert body["rounds_used"] == 2
    assert body["truncated"] is False
    assert body["tool_calls"][0]["name"] == "calculator"

    # 同 session 第二轮：历史生效
    resp2 = client.post("/api/chat", json={"session_id": sid, "message": "谢谢"})
    assert resp2.json()["reply"] == "在呢"
    history = client.get(f"/api/sessions/{sid}/history").json()
    assert len(history["messages"]) >= 4

    # trace 一致性：每次工具调用都有记录
    trace = client.get(f"/api/sessions/{sid}/trace").json()["events"]
    tool_events = [e for e in trace if e["event"] == "tool_call"]
    assert len(tool_events) == 1 and tool_events[0]["payload"]["name"] == "calculator"


def test_multi_session_isolation_via_api(make_client, tmp_path):
    client, llm, store = make_client([
        {"tool": ("todo_add", {"content": "A窗口的待办"})},
        {"content": "已记录"},
        {"content": "B窗口回复"},
    ])
    s1 = client.post("/api/sessions").json()["session_id"]
    s2 = client.post("/api/sessions").json()["session_id"]

    client.post("/api/chat", json={"session_id": s1, "message": "记个待办"})
    client.post("/api/chat", json={"session_id": s2, "message": "你好"})

    assert len(store.list_todos(s1)) == 1
    assert len(store.list_todos(s2)) == 0  # 窗口隔离：B 窗口看不到 A 的待办
    h1 = client.get(f"/api/sessions/{s1}/history").json()["messages"]
    h2 = client.get(f"/api/sessions/{s2}/history").json()["messages"]
    assert h1 != h2


def test_tools_endpoint(make_client):
    client, _, _ = make_client([])
    tools = client.get("/api/tools").json()
    names = {t["function"]["name"] for t in tools}
    assert {"calculator", "todo_add", "search", "weather"} <= names


def test_healthz_no_key_leak(make_client):
    client, _, _ = make_client([])
    body = client.get("/api/healthz").json()
    assert body["status"] == "ok"
    assert "key" not in str(body).lower() or "sk-" not in str(body)


def test_steer_endpoint_queues_and_injects(make_client):
    client, llm, store = make_client([
        {"tool": ("calculator", {"expression": "1+1"})},
        {"content": "算完了"},
    ])
    sid = client.post("/api/sessions").json()["session_id"]

    # Agent 执行前插入 steering 消息（真实场景是执行期间插入）
    resp = client.post(f"/api/sessions/{sid}/steer", json={"message": "顺便查天气"})
    assert resp.json()["queued"] is True

    result = client.post("/api/chat", json={"session_id": sid, "message": "算个加法"})
    assert result.json()["reply"] == "算完了"

    trace = client.get(f"/api/sessions/{sid}/trace").json()["events"]
    assert any(e["event"] == "steering_injected" for e in trace)


def test_chat_unknown_session_auto_creates(make_client):
    client, _, store = make_client([{"content": "你好"}])
    resp = client.post("/api/chat", json={"session_id": "brand-new", "message": "hi"})
    assert resp.status_code == 200
    assert store.load_session("brand-new") is not None


def test_llm_failure_returns_502(tmp_path):
    """基本异常处理：LLM 连接失败这类致命错误上抛为 502，不泄漏堆栈给用户。"""
    from fastapi.testclient import TestClient

    from app.main import create_app

    class FailingLLM:
        async def chat(self, messages, tools=None):
            raise ConnectionError("模拟 LLM 连接失败")

    app = create_app(llm_client=FailingLLM(), db_path=str(tmp_path / "fail.db"))
    with TestClient(app) as client:
        resp = client.post("/api/chat", json={"session_id": "s1", "message": "你好"})
    assert resp.status_code == 502
    assert "Agent 执行失败" in resp.json()["detail"]


def test_missing_api_key_raises_clear_error():
    settings = Settings(base_url="https://x", api_key="", model="m",
                        max_rounds=8, max_history_messages=40,
                        recent_keep=20, db_path="data/agent.db")
    try:
        LLMClient(settings)
        raise AssertionError("应当在缺 key 时报错")
    except RuntimeError as e:
        assert "LLM_API_KEY" in str(e)
