"""Mini Agent 演示入口（三模式）。

  python demo.py --serve     零密钥浏览器演示：注入剧本 LLM，起本地服务，浏览器点完整链路
  python demo.py --replay    CLI 逐轮打印链路：messages / trace / 工具结果 / 熔断 / 压缩
  python demo.py --check     填完 .env 后自检：用真实 LLM（MiniMax 订阅）发一次带 tools 的请求

设计要点：
- 注入 llm_client 后 runtime.llm 非空 → app/main.py 的 lifespan 跳过 LLMClient 创建
  → validate_llm 不触发 → 零密钥、不联网、不消耗 token、结果完全可复现。
- uvicorn.run 必须传 app 对象；传 "app.main:app" 会加载未注入的模块级实例而启动失败。
- 演示数据写在 data/demo.db，不污染笔试产物 data/agent.db。
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# 必须在 import app.main 之前设置：模块级 app = create_app() 也会读这些环境变量
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ["DB_PATH"] = "data/demo.db"          # 强制隔离（覆盖 .env 可能存在的 DB_PATH）
os.environ["MAX_ROUNDS"] = "3"                   # 演示熔断用，正常链路最多 3 轮也够
DB_PATH = "data/demo.db"

from app.agent.context import ContextBuilder  # noqa: E402
from app.llm.client import LLMResponse, ToolCall  # noqa: E402
from app.main import create_app  # noqa: E402


# ---------------------------------------------------------------------------
# 剧本 LLM：鸭子类型替换 LLMClient，按"最后一条 user 消息 + 已有工具结果"路由
# ---------------------------------------------------------------------------

def make_response(content="", tool_calls=None):
    tool_calls = tool_calls or []
    raw = {"role": "assistant", "content": content}
    if tool_calls:
        raw["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in tool_calls
        ]
    return LLMResponse(content=content, tool_calls=tool_calls, raw_message=raw)


class ScriptedLLM:
    """零密钥剧本 LLM。用消息内容判断链路位置，支持多 session 并行点按。"""

    def __init__(self):
        self.calls = []

    @staticmethod
    def _tail(messages):
        """取最后一条 user 消息之后的对话（本次请求内产生的消息）。"""
        idx = -1
        for i, m in enumerate(messages):
            if m.get("role") == "user":
                idx = i
        return messages[idx + 1:] if idx >= 0 else []

    @staticmethod
    def _tool(name, args):
        import uuid
        return ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name,
                        arguments=json.dumps(args, ensure_ascii=False))

    async def chat(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})

        # 1) 压缩调用隔离：context.py 的 summarize_with_llm 会再发一次纯文本请求
        if messages and "对话压缩器" in str(messages[0].get("content", "")):
            return make_response(content="此前用户查过天气、记过待办，并做了若干计算与追问。")

        text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                text = str(m.get("content", ""))
                break
        tail = self._tail(messages)
        tool_count = sum(1 for m in tail if m.get("role") == "tool")

        # 2) 熔断剧本：每次都返回工具调用，直到 max_rounds 熔断
        if "熔断" in text or "死循环" in text:
            return make_response(tool_calls=[self._tool("calculator", {"expression": "1+1"})])

        # 3) 天气 + 待办：两轮工具链（weather → todo_add → 最终回答）
        if "天气" in text and any(k in text for k in ("待办", "带伞", "记")):
            if tool_count == 0:
                return make_response(content="先查天气", tool_calls=[self._tool("weather", {"city": "北京"})])
            if tool_count == 1:
                return make_response(content="再记待办", tool_calls=[self._tool("todo_add", {"content": "带伞", "priority": "high"})])
            return make_response(content="北京今天晴 25°C，已帮你把「带伞」记进待办（剧本演示，与 mock 工具结果一致）。")

        # 4) 计算：单工具 loop
        if "算" in text or "*" in text or "等于多少" in text:
            if tool_count == 0:
                return make_response(content="需要精确计算",
                                     tool_calls=[self._tool("calculator", {"expression": "3.5*12"})])
            return make_response(content="3.5 × 12 = 42。")

        # 5) 列待办：单工具 loop
        if "待办" in text and tool_count == 0:
            return make_response(content="查一下待办清单",
                                 tool_calls=[self._tool("todo_list", {"include_done": False})])
        if "待办" in text:
            return make_response(content="以上是当前待办清单。")

        # 6) 异常剧本：未知工具 → 错误喂回 LLM → 正常收尾
        if "不存在" in text:
            if tool_count == 0:
                return make_response(tool_calls=[self._tool("not_exist_tool", {})])
            return make_response(content="刚才那个工具不存在，我已改用现有能力回答你。")

        # 7) 追问 / 闲聊：纯文本直接回复（历史由 context 整段带入，演示指代消解）
        if "城市" in text or "呢" in text:
            return make_response(content="你说的还是北京，今天晴 25°C。")
        if "谢谢" in text:
            return make_response(content="不客气，随时可以继续问。")
        return make_response(content=f"收到：{text[:60]}（这是剧本 LLM 的直接回复分支）")


def build_demo_app():
    llm = ScriptedLLM()
    app = create_app(llm_client=llm, db_path=DB_PATH)
    runtime = app.state.runtime
    # 压缩演示：把阈值调低，聊到第 9 条消息即可触发（生产代码零改动）
    runtime.context = ContextBuilder(summarizer=runtime._summarize,
                                     max_history=8, recent_keep=4)
    return app, llm


# ---------------------------------------------------------------------------
# --replay：CLI 逐轮打印链路
# ---------------------------------------------------------------------------

def replay():
    from fastapi.testclient import TestClient

    app, _ = build_demo_app()
    with TestClient(app) as client:
        def chat(sid, text, label):
            r = client.post("/api/chat", json={"session_id": sid, "message": text}).json()
            events = client.get(f"/api/sessions/{sid}/trace").json()["events"]
            print(f"\n=== {label} ===")
            print(f"用户输入 : {text}")
            print(f"最终回复 : {r['reply']}")
            print(f"轮次     : {r['rounds_used']}  截断: {r['truncated']}")
            for rec in r["tool_calls"]:
                print(f"  [工具] round{rec['round']} {rec['name']}({rec['args']}) "
                      f"-> {rec['result']}  [{rec['status']}/{rec['duration_ms']}ms]")
            print("  [trace] " + " → ".join(e["event"] for e in events[-12:]))
            return r

        sid = client.post("/api/sessions").json()["session_id"]
        print(f"session_id = {sid}   (DB_PATH={DB_PATH}, MAX_ROUNDS={os.getenv('MAX_ROUNDS')})")

        chat(sid, "查北京天气并记个带伞待办", "① 多轮工具链 weather → todo_add → 回答")
        chat(sid, "那个城市呢？", "② 追问（历史整段带入 context）")
        chat(sid, "算 3.5*12 等于多少", "③ 单工具 loop")
        chat(sid, "触发熔断测试", "④ max_rounds 熔断兜底")
        chat(sid, "调用不存在的工具", "⑤ 未知工具异常喂回")

        # ⑥ steering：插话消息在工具轮间注入
        client.post(f"/api/sessions/{sid}/steer", json={"message": "顺便算个账"})
        chat(sid, "列一下我的待办", "⑥ steering 插话注入")

        # ⑦ 压缩：连发纯文本消息，触发摘要压缩
        for i in range(6):
            client.post("/api/chat", json={"session_id": sid, "message": f"第{i+1}条闲聊"})
        events = client.get(f"/api/sessions/{sid}/trace").json()["events"]
        compressed = [e for e in events if e["event"] == "llm_request" and e["payload"].get("compressed")]
        print(f"\n=== ⑦ 摘要压缩 ===")
        print(f"触发压缩的轮次数: {len(compressed)}")
        if compressed:
            print(f"样例: round={compressed[0]['payload']['round']} "
                  f"n_messages={compressed[0]['payload']['n_messages']}")
        session = app.state.store.load_session(sid)
        print(f"缓存摘要: {session['memory'].get('summary', '(无)')}")
        print(f"历史总条数: {len(session['messages'])}（全量保存，压缩只是渲染视图）")

        # ⑧ 多 session 隔离
        s2 = client.post("/api/sessions").json()["session_id"]
        chat(s2, "我的待办有哪些", "⑧ 另一个窗口：查不到 A 窗口的待办")
        print(f"\n窗口 A 待办数 = {len(app.state.store.list_todos(sid))}，"
              f"窗口 B 待办数 = {len(app.state.store.list_todos(s2))}  → 隔离生效")

    print("\n✅ replay 完成：以上全部离线、零密钥、可复现。")


# ---------------------------------------------------------------------------
# --check：填完 .env 后自检真实 LLM（MiniMax 订阅）的 function calling
# ---------------------------------------------------------------------------

def check():
    from app.config import Settings
    from app.llm.client import LLMClient
    from app.tools import registry

    settings = Settings.from_env()
    if not settings.api_key or any(k in settings.api_key for k in ("粘贴", "填入", "xxxx", "your")):
        print("❌ 请先把 MiniMax 订阅 Key 粘贴到 .env 的 LLM_API_KEY= 后面，再运行 --check")
        sys.exit(1)
    print(f"端点   : {settings.base_url}")
    print(f"模型   : {settings.model}")
    print(f"工具数 : {len(registry.schemas())}（calculator/todo_*/search/weather）")
    print("发送一条带 tools 的请求，检查订阅端点是否支持 function calling ...\n")

    async def main():
        client = LLMClient(settings)
        resp = await client.chat(
            [
                {"role": "system", "content": "你是助手，需要计算时必须调用 calculator 工具。"},
                {"role": "user", "content": "请计算 3*4 等于多少"},
            ],
            tools=registry.schemas(),
        )
        if resp.tool_calls:
            print("✅ function calling 可用，模型决定调用：")
            for tc in resp.tool_calls:
                print(f"   {tc.name}({tc.arguments})")
        else:
            print("⚠️ 模型直接回答了，没有调用工具：")
            print(f"   {resp.content}")
            print("   可换模型重试（.env 的 LLM_MODEL），或确认订阅套餐是否开放 tools 参数。")

    try:
        asyncio.run(main())
    except Exception as e:
        print(f"❌ 调用失败: {type(e).__name__}: {e}")
        print("   排查顺序：① Key 是否为 Coding Plan 订阅专用 Key ② 端点是否为 "
              "https://api.minimaxi.com/v1 ③ 模型名是否在订阅支持列表 ④ 网络/额度")
        sys.exit(1)


# ---------------------------------------------------------------------------
# --serve：浏览器演示
# ---------------------------------------------------------------------------

def serve():
    import uvicorn

    app, _ = build_demo_app()
    print("=" * 68)
    print("Mini Agent 零密钥演示（剧本 LLM，不联网 / 不消耗 token / 结果可复现）")
    print("打开 http://127.0.0.1:8000")
    print()
    print("建议依次点：")
    print("  1. ＋新建会话建 A/B 两个窗口 → A 发『记个待办：带伞』，B 发『我的待办』→ 隔离")
    print("  2. 『查北京天气并记个带伞待办』→ 展开工具调用面板（weather → todo_add）")
    print("  3. 『那个城市呢？』→ 追问；勾选『插话』→ steering 注入")
    print("  4. 『触发熔断测试』→ 已达最大轮次截断提示")
    print("  5. 连发 6 条以上闲聊 → /api/sessions/{id}/trace 里 compressed=true")
    print()
    print("调试接口：GET /api/sessions/{id}/trace 、GET /api/tools 、GET /api/healthz")
    print("停止服务：Ctrl+C")
    print("=" * 68)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini Agent 演示入口")
    parser.add_argument("--serve", action="store_true", help="起本地服务，浏览器零密钥演示")
    parser.add_argument("--replay", action="store_true", help="CLI 逐轮打印链路")
    parser.add_argument("--check", action="store_true", help="自检真实 LLM 的 function calling")
    args = parser.parse_args()
    if args.replay:
        replay()
    elif args.check:
        check()
    else:
        serve()  # 默认即 --serve
