"""Context 管理测试：追问历史 / 摘要级压缩 / 安全切割（不拆散工具消息配对）。"""
import asyncio

from app.agent.context import ContextBuilder, _rule_summary


def make_session(n_messages):
    """交替 user/assistant 的普通历史。"""
    return {
        "session_id": "s1",
        "memory": {},
        "messages": [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg{i}"}
            for i in range(n_messages)
        ],
    }


def test_no_compression_when_short():
    builder = ContextBuilder(max_history=40, recent_keep=20)
    session = make_session(10)
    messages, meta = asyncio.run(builder.build(session, "system-prompt"))
    assert meta["compressed"] is False
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "system-prompt"
    assert len(messages) == 11


def test_compression_with_llm_summarizer():
    calls = {"n": 0}

    async def summarizer(dropped):
        calls["n"] += 1
        return "这是对话摘要"

    builder = ContextBuilder(summarizer=summarizer, max_history=40, recent_keep=20)
    session = make_session(45)
    messages, meta = asyncio.run(builder.build(session, "system-prompt"))
    assert meta["compressed"] is True and meta["summary_used"] is True
    assert "[此前对话摘要]" in messages[0]["content"] and "这是对话摘要" in messages[0]["content"]
    assert len(messages) <= 1 + 20 + 2  # system + 窗口（切点取整到 user 边界）
    assert calls["n"] == 1
    # 摘要已缓存进 session.memory，再次构建不再调用 summarizer
    messages2, meta2 = asyncio.run(builder.build(session, "system-prompt"))
    assert meta2["summary_used"] is True
    assert calls["n"] == 1


def test_safe_cut_never_splits_tool_pairing():
    """窗口头部必须是 user 消息，assistant(tool_calls)↔tool 配对完整。"""
    builder = ContextBuilder(max_history=40, recent_keep=20)
    messages = []
    for i in range(23):  # 46 条 > 40，触发压缩
        messages.append({"role": "user", "content": f"u{i}"})
        messages.append({"role": "assistant", "content": f"a{i}"})
    # 埋一组工具消息对（将被压缩丢弃的区间内）
    messages[10] = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "calculator", "arguments": "{}"}}]}
    messages[11] = {"role": "tool", "tool_call_id": "c1", "content": "2"}
    session = {"session_id": "s1", "memory": {}, "messages": messages}

    result, meta = asyncio.run(builder.build(session, "sys"))
    assert meta["compressed"] is True
    window = result[1:]  # 去掉 system
    assert window[0]["role"] == "user"  # 窗口头部是 user，配对不会被切在中间
    tool_pair_heads = sum(1 for m in window if m.get("tool_calls"))
    tool_msgs = sum(1 for m in window if m["role"] == "tool")
    assert tool_pair_heads == tool_msgs  # 配对完整：有 tool_calls 必有对应 tool 消息


def test_rule_summary_fallback_without_llm():
    builder = ContextBuilder(summarizer=None, max_history=40, recent_keep=20)
    session = make_session(45)
    messages, meta = asyncio.run(builder.build(session, "sys"))
    assert meta["compressed"] is True and meta["summary_used"] is True


def test_rule_summary_extracts_assistant_answers():
    dropped = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "北京今天晴，25 度"},
    ]
    assert "北京今天晴" in _rule_summary(dropped)


def test_plain_follow_up_uses_history(make_runtime):
    """纯对话追问：第二轮不再调工具，但 context 仍带上第一轮 assistant 回答（指代消解）。"""
    rt, llm = make_runtime([
        {"content": "北京今天晴 25°C"},
        {"content": "你说的还是北京"},
    ])
    asyncio.run(rt.run("s1", "北京天气怎么样？"))
    asyncio.run(rt.run("s1", "刚才那个城市呢？"))
    assert len(llm.calls) == 2  # 两轮各一次 LLM 调用，纯对话追问不进工具循环
    second = llm.calls[1]["messages"]
    assert any(m.get("role") == "assistant" and "北京今天晴" in (m.get("content") or "")
               for m in second)
    assert any(m.get("role") == "user" and "刚才那个城市呢" in str(m.get("content", ""))
               for m in second)


def test_history_enables_follow_up_questions(make_runtime, store):
    """带工具的追问：历史里既有上一轮 assistant 回答，也有完整 tool 消息。"""
    rt, llm = make_runtime([
        {"tool": ("weather", {"city": "北京"})},
        {"content": "北京今天雷阵雨 31°C"},
        {"tool": ("todo_add", {"content": "带伞"})},           # 带工具的追问
        {"content": "已记下带伞待办"},
    ])
    asyncio.run(rt.run("s1", "北京天气怎么样？"))
    asyncio.run(rt.run("s1", "那把带伞加进待办"))
    # 第二轮的 context 必须包含第一轮的 assistant 回答与 tool 消息
    second = llm.calls[-1]["messages"]
    assert any(m["role"] == "assistant" and "雷阵雨" in (m.get("content") or "")
               for m in second)
    assert any(m["role"] == "tool" for m in second)
