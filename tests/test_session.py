"""Session 存储测试：多 session 隔离 / 持久化续聊 / trace。"""


def test_resume_conversation_after_store_restart(tmp_path):
    """题目：用户 A 可以随时接着窗口继续聊——重启进程后历史仍可续接。"""
    import asyncio

    from app.agent.runtime import AgentRuntime
    from app.session.store import SessionStore
    from app.tools import init_todo_store, registry
    from conftest import FakeLLMClient

    db = str(tmp_path / "resume.db")
    store1 = SessionStore(db)
    init_todo_store(store1)
    store1.create_session("s1")
    rt1 = AgentRuntime(llm=FakeLLMClient([{"content": "北京今天晴 25°C"}]),
                       registry=registry, store=store1)
    asyncio.run(rt1.run("s1", "北京天气怎么样？"))
    store1.close()

    # 模拟重启：全新 store 实例、同一个库、全新 LLM 客户端
    store2 = SessionStore(db)
    init_todo_store(store2)
    llm2 = FakeLLMClient([{"content": "你说的还是北京"}])
    rt2 = AgentRuntime(llm=llm2, registry=registry, store=store2)
    reply = asyncio.run(rt2.run("s1", "刚才那个城市呢？"))
    assert reply.content == "你说的还是北京"
    second = llm2.calls[0]["messages"]
    assert any(m.get("role") == "assistant" and "北京今天晴" in (m.get("content") or "")
               for m in second)
    assert len(store2.load_session("s1")["messages"]) == 4
    store2.close()


def test_two_sessions_todo_isolation(store):
    store.create_session("s1")
    store.create_session("s2")
    store.add_todo("s1", "写周报")
    store.add_todo("s2", "查天气记录")
    s1_contents = [t["content"] for t in store.list_todos("s1")]
    s2_contents = [t["content"] for t in store.list_todos("s2")]
    assert "写周报" in s1_contents and "查天气记录" not in s1_contents
    assert "查天气记录" in s2_contents and "写周报" not in s2_contents


def test_persistence_roundtrip(tmp_path):
    from app.session.store import SessionStore

    db = str(tmp_path / "persist.db")
    s1 = SessionStore(db)
    s1.create_session("s1")
    s1.save_session("s1", [{"role": "user", "content": "你好"}], {"summary": "x"})
    s1.close()

    s2 = SessionStore(db)  # 模拟重启：新实例、同一个库
    session = s2.load_session("s1")
    assert session["messages"][0]["content"] == "你好"
    assert session["memory"] == {"summary": "x"}
    s2.close()


def test_save_message_history_roundtrip(make_runtime, store):
    import asyncio

    rt, _ = make_runtime([{"content": "在"}])
    asyncio.run(rt.run("s1", "在吗？"))
    loaded = store.load_session("s1")
    assert [m["role"] for m in loaded["messages"]] == ["user", "assistant"]
    assert loaded["messages"][1]["content"] == "在"


def test_trace_log_and_query(store):
    store.create_session("s1")
    store.log_trace("s1", "tool_call", {"name": "calculator", "result": "2"})
    store.log_trace("s1", "final_reply", {"content": "答案是 2"})
    events = store.get_trace("s1")
    assert [e["event"] for e in events] == ["tool_call", "final_reply"]
    assert events[0]["payload"]["name"] == "calculator"


def test_session_list_preview(store):
    store.create_session("s1")
    store.save_session("s1", [{"role": "user", "content": "帮我写一份周报，关于项目进度"}])
    sessions = store.list_sessions()
    assert sessions[0]["session_id"] == "s1"
    assert "周报" in sessions[0]["preview"]
