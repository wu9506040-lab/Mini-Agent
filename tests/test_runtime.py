"""Runtime 循环测试：覆盖直接回复 / 单工具 / 多轮工具 / 熔断 / 异常喂回 / 未知工具 / steering。"""
import asyncio

from app.agent.steering import SteeringQueue


def run(coro):
    return asyncio.run(coro)


def test_final_reply_only(make_runtime):
    rt, _ = make_runtime([{"content": "你好！"}])
    reply = run(rt.run("s1", "你好"))
    assert reply.content == "你好！"
    assert reply.rounds_used == 1
    assert reply.truncated is False
    assert reply.tool_calls == []


def test_single_tool_call_loop(make_runtime, store):
    rt, _ = make_runtime([
        {"tool": ("calculator", {"expression": "3.5*12"})},
        {"content": "3.5 乘以 12 等于 42"},
    ])
    reply = run(rt.run("s1", "3.5*12 等于多少？"))
    assert reply.rounds_used == 2
    assert reply.tool_calls[0].name == "calculator"
    assert "42" in reply.tool_calls[0].result
    # 协议强约束：assistant(tool_calls) 必须紧跟配对的 tool 消息
    messages = store.load_session("s1")["messages"]
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            nxt = messages[i + 1]
            assert nxt["role"] == "tool"
            assert nxt["tool_call_id"] == m["tool_calls"][0]["id"]


def test_multi_round_tool_loop(make_runtime, store):
    rt, _ = make_runtime([
        {"tool": ("weather", {"city": "北京"})},
        {"tool": ("todo_add", {"content": "带伞", "priority": "high"})},
        {"content": "北京今天雷阵雨 31°C，已帮你记下带伞待办"},
    ])
    reply = run(rt.run("s1", "查北京天气，记得提醒我带伞"))
    assert reply.rounds_used == 3
    assert [r.name for r in reply.tool_calls] == ["weather", "todo_add"]
    todos = store.list_todos("s1")
    assert len(todos) == 1 and todos[0]["content"] == "带伞"


def test_max_rounds_circuit_break(make_runtime):
    rt, _ = make_runtime(
        [{"tool": ("calculator", {"expression": "1+1"})},
         {"tool": ("calculator", {"expression": "2+2"})}],
        max_rounds=2,
    )
    reply = run(rt.run("s1", "算数"))
    assert reply.truncated is True
    assert reply.rounds_used == 2
    assert "最大工具调用轮次" in reply.content


def test_tool_exception_fed_back_not_raised(make_runtime):
    from app.tools.base import BaseTool, EmptyArgs, register_tool, remove_tool

    @register_tool
    class BrokenTool(BaseTool):
        name = "broken"
        description = "总是抛异常的工具"
        args_model = EmptyArgs

        async def run(self):
            raise RuntimeError("数据库爆炸")

    try:
        rt, _ = make_runtime([
            {"tool": ("broken", {})},
            {"content": "工具坏了，我如实告诉你：暂时查不了"},
        ])
        reply = run(rt.run("s1", "试试坏工具"))
        assert reply.tool_calls[0].status == "exception"
        assert reply.tool_calls[0].result.startswith("Error:")
        assert reply.truncated is False  # 异常没有击穿循环，LLM 拿到错误后正常收尾
        assert "如实" in reply.content
    finally:
        remove_tool("broken")


def test_unknown_tool_error_message(make_runtime):
    rt, _ = make_runtime([
        {"tool": ("nope", {})},
        {"content": "抱歉，没有这个工具"},
    ])
    reply = run(rt.run("s1", "调用不存在的工具"))
    assert reply.tool_calls[0].status == "unknown_tool"
    assert "Error" in reply.tool_calls[0].result


def test_invalid_args_json(make_runtime):
    rt, _ = make_runtime([
        {"raw_tool": ("calculator", "表达式不是JSON")},
        {"content": "参数有问题，我重新组织"},
    ])
    reply = run(rt.run("s1", "算一下"))
    assert reply.tool_calls[0].status == "invalid_args"


def test_llm_receives_tool_schemas_for_autonomous_decision(make_runtime):
    """题目：LLM 基于 Schema 自主决策调用——注册表 Schema 必须真的传进 llm.chat。"""
    rt, llm = make_runtime([{"content": "好的"}])
    run(rt.run("s1", "打个招呼"))
    tools = llm.calls[0]["tools"]
    assert tools and all(t["type"] == "function" for t in tools)
    names = {t["function"]["name"] for t in tools}
    assert {"calculator", "todo_add", "search", "weather"} <= names
    for t in tools:
        assert t["function"]["description"]
        assert "properties" in t["function"]["parameters"]


def test_tool_result_feeds_next_round_context(make_runtime):
    """Step four：工具结果决定继续 loop，且结果必须出现在下一轮 context。"""
    rt, llm = make_runtime([
        {"tool": ("calculator", {"expression": "2+2"})},
        {"content": "结果是 4"},
    ])
    run(rt.run("s1", "算一下 2+2"))
    assert len(llm.calls) == 2
    second = llm.calls[1]["messages"]
    assert any(m.get("role") == "tool" and "4" in str(m.get("content", ""))
               for m in second)
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second)


def test_steering_message_injected_between_rounds(make_runtime, store):
    queue = SteeringQueue()
    rt, _ = make_runtime([
        {"tool": ("calculator", {"expression": "1+1"})},
        {"content": "算完了"},
    ], steering_queue=queue)
    queue.push("s1", "顺便查下天气")
    reply = run(rt.run("s1", "算个加法"))
    assert reply.content == "算完了"
    messages = store.load_session("s1")["messages"]
    assert any(m.get("role") == "user" and "[用户插话]" in m.get("content", "")
               for m in messages)
    events = store.get_trace("s1")
    assert any(e["event"] == "steering_injected" for e in events)
