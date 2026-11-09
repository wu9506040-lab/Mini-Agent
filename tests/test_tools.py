"""工具层测试：安全求值 / todo 闭环 / Schema 推导 / mock 确定性。"""
import asyncio

from app.tools import registry


def run(coro):
    return asyncio.run(coro)


# ---- calculator：ast 白名单，禁 eval ----

def test_calculator_basic():
    tool = registry.get("calculator")
    assert "42" in run(tool.run(expression="3.5*12"))
    assert "25" in run(tool.run(expression="(2+3)**2"))


def test_calculator_error_paths():
    tool = registry.get("calculator")
    assert "Error" in run(tool.run(expression="1/0"))
    assert "Error" in run(tool.run(expression="1+"))
    assert "Error" in run(tool.run(expression="__import__('os')"))  # 非法元素被拒


# ---- todo：增/查/完成闭环 ----

def test_todo_cycle(store):
    add = registry.get("todo_add")
    listing = registry.get("todo_list")
    done = registry.get("todo_done")

    from app.tools.todo import set_current_session

    set_current_session("s1")
    assert "#" in run(add.run(content="写周报", priority="high"))
    out = run(listing.run())
    assert "写周报" in out and "✓" not in out
    todo_id = store.list_todos("s1")[0]["id"]
    assert "已标记完成" in run(done.run(todo_id=todo_id))
    assert "没有待办" in run(listing.run())


def test_todo_done_wrong_id_returns_error(store):
    from app.tools.todo import set_current_session

    set_current_session("s1")
    assert "Error" in run(registry.get("todo_done").run(todo_id=999))


# ---- 注册机制：Schema 推导符合 OpenAI tools 格式 ----

def test_registry_schemas_format():
    schemas = registry.schemas()
    names = {s["function"]["name"] for s in schemas}
    assert {"calculator", "todo_add", "todo_list", "todo_done", "search", "weather"} <= names
    for s in schemas:
        assert s["type"] == "function"
        assert s["function"]["description"]  # 每个工具必须有描述
        assert "properties" in s["function"]["parameters"]


def test_todo_add_schema_descriptions():
    schema = registry.get("todo_add").to_schema()
    props = schema["function"]["parameters"]["properties"]
    assert props["content"]["description"]
    assert set(props["priority"].get("enum", [])) == {"high", "low"}


# ---- mock 工具：确定性输出 ----

def test_search_mock_deterministic():
    tool = registry.get("search")
    r1 = run(tool.run(query="python 是什么"))
    r2 = run(tool.run(query="python 是什么"))
    assert r1 == r2 and "Python" in r1


def test_weather_mock():
    tool = registry.get("weather")
    assert "25" in run(tool.run(city="北京"))
    assert "mock" in run(tool.run(city="不出名的城市"))
