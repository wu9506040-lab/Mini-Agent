# Mini Agent — 从零实现的最小可用 Agent Runtime

一个不依赖任何 Agent 框架（langgraph / openhands / openclaw）的最小可用 Agent：
**自研 Agent Runtime 内核 + FastAPI 壳 + 单文件前端**。设计理念借鉴
[pi](https://github.com/badlogic/pi-mono)（极简 harness：模型够聪明，框架要够薄）
与 Claude Code 的工程实践。

> 零密钥演示：`python demo.py --serve`（内置剧本 LLM，无需 API Key，浏览器可点完整链路）。

## 快速开始

```bash
# 1. 配置：复制模板为 .env（.env 不入库，.gitignore 已忽略）
cp .env.example .env
#    把 MiniMax 订阅专用 Key 粘贴到 .env 的 LLM_API_KEY= 后面，代码会自动加载
#    ⚠️ Coding Plan 订阅 Key 与按量计费 Key 不通用；国际版把端点换为 https://api.minimax.io/v1
#    可选模型：MiniMax-M3 / MiniMax-M2.5 / M2.5-highspeed / M2.1 / M2（当前用 M3，已实测 FC 可用）
#    （若不用 .env，也可手动 export；PowerShell 为 $env:LLM_API_KEY=...）

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动（.env 自动加载，无需 --env-file）
uvicorn app.main:app --reload
# 打开 http://127.0.0.1:8000

# 填完 key 后自检订阅端点是否支持 function calling：
python demo.py --check
```

零密钥演示（不联网、不消耗 token，用剧本 LLM 回放完整链路）：

```bash
python demo.py --serve     # 浏览器点：多会话隔离 / 工具面板 / 追问 / 插话 / 熔断 / 压缩
python demo.py --replay    # CLI 逐轮打印 messages / trace / 工具结果
```

运行测试（离线全绿，不消耗 token）：

```bash
python -m pytest tests/ -v
```

## 1. 系统设计

```
┌─────────────┐   HTTP    ┌──────────────────────────────────────────┐
│ static/     │ ────────► │ FastAPI (api/routes.py)                  │
│ index.html  │           │   POST /api/chat  POST /api/sessions     │
│ (harness层) │           │   /api/sessions/{id}/history|trace|steer │
└─────────────┘           ├──────────────────────────────────────────┤
                          │ AgentRuntime (agent/runtime.py)          │
                          │   loop: context → LLM → parse            │
                          │        → tool → 回灌 → 继续/返回          │
                          │   max_rounds 熔断 / steering / trace     │
                          ├──────────────┬───────────────┬───────────┤
                          │ llm/client.py│ tools/ 注册表  │ session/  │
                          │ (协议适配层， │ (Schema 推导+  │ store.py  │
                          │  唯一 mock 边界)│  校验)       │ (SQLite)  │
                          └──────────────┴───────────────┴───────────┘
```

分层对标 pi：`llm/client.py`（≈ pi-ai，纯协议适配）→ `agent/runtime.py`（≈
pi-agent-core，循环与编排）→ `api/ + static/`（harness 交互层）。下层对上层零依赖。

**为什么 `openai` SDK 不属于"现有 agent 框架"**：它只做 HTTP 协议适配，
不实现 agent loop / 工具调度 / context 管理——这些全部自研于
`runtime.py / parser.py / context.py / store.py`。判据是"库是否替你写了主流程"。

### 1.1 Agent 循环

```
用户输入 → [system(规则+todos) + 历史(压缩视图) + 新输入]
        → LLM(tools=Schema 列表)
        ├─ 返回 tool_calls → assistant 消息原样落历史（协议强约束）
        │    → 逐个执行工具（异常转字符串喂回 LLM）
        │    → tool 结果以 role=tool 回灌 → 轮次+1 → 继续 loop
        │    → 轮间检查 steering 队列，注入用户插话
        └─ 返回纯文本 → 最终答案，保存 session，返回用户
退出条件：① 纯文本回答 ② max_rounds=8 熔断兜底 ③ 致命错误上抛(502)
```

### 1.2 工具注册机制

装饰器 `@register_tool` + Pydantic 参数模型。`Field(description=...)`
一处声明三处复用：LLM 可见（`to_schema()` 推导 OpenAI tools 格式）、
入参校验（`model_validate`，失败转可读错误喂回 LLM）、文档。
新增工具三步走：继承 `BaseTool` → 声明 `name/description/args_model` →
在 `tools/__init__.py` 导入即注册。

### 1.3 LLM 输出解析层（可替换接口）

- **主方案 `FunctionCallingParser`**：结构化约束下沉到 API 协议层，
  解析退化为"取字段"，零格式漂移风险。
- **备选 `TextParser`（pi 式 XML 文本协议）**：LLM 在文本中用
  `<tool_call>{...}</tool_call>` 发起调用——模型无关、推理过程可观察，
  但要对抗格式漂移（JSON 转义、漏配、不守格式），需要额外修复逻辑。
- 二者实现同一个 `BaseParser` 接口，换协议只动一个文件（见
  `tests/test_parser.py::test_parser_interface_replaceable`）。

### 1.4 Context 管理与压缩

- **最大轮次**：`MAX_ROUNDS=8`（环境变量可调），指单次请求内 LLM 调用
  次数上限。熔断时把已有工具结果兜底返回而非硬报错。
  > 刻意偏离 pi 的"无步数限制"设计：考题明确要求资源护栏，且无上限的
  > 自我修正循环在无人工监督的生产环境是成本黑洞。
- **追问**：纯对话追问与带工具追问走同一条路——messages 里保留了完整的
  tool 调用痕迹，LLM 既能做指代消解（"那个城市呢？"），又能直接接着调工具。
- **压缩**（pi 式 compaction）：历史超过 `MAX_HISTORY_MESSAGES=40` 条时，
  保头尾丢中间——system 与最近 20 条保留；被丢弃区间**调一次 LLM 生成
  摘要**追加到 system 末尾（摘要缓存于 session.memory，不重复生成）。
  切割点只选 user 消息处，保证 `assistant(tool_calls)↔tool` 配对不被拆散
  （OpenAI 协议强约束，也是最容易踩的坑）。

### 1.5 Context 里放什么（题目"哪些信息要塞入 context 更合适，自行判断"）

`context = [system] + [压缩后的消息序列]`。逐类决策如下：

| 信息 | 是否进 context | 放置位置与理由 |
|---|---|---|
| 用户输入 | 进 | 最新一条必带；历史 user 消息随窗口保留，是任务起点的锚 |
| assistant 最终回答 | 进 | 保留在消息序列里，供指代消解（"那个城市呢"）与结论传递 |
| `assistant(tool_calls)` | 进 | 协议强约束必须落历史；也让 LLM 看见自己调过什么，避免重复调用 |
| 工具执行结果 | 进，但优先被压缩 | 体积最大、复用价值最低：最近 N 条保留，更早的进摘要 |
| 结构化待办 | 进 system，不进消息流 | 每轮全量注入（见第 2 节），位置稳定、不与历史重复 |
| 压缩摘要 | 进 system 末尾 | 一次生成、缓存复用，不随轮次重复计费 |
| **Agent 思考过程** | **不主动注入** | trace 里单独记 `llm_response.thought` 供排查；FC 下若模型在 `assistant(tool_calls).content` 附带思考文本，因协议要求消息原样落历史会随历史保留，但**不额外把思考重新拼进 context、不塞 reasoning 元数据** |
| trace / 日志 / API 响应元数据 | 不进 | 调试数据不是模型证据 |

判断标准三条：① 协议要求必须带的（`tool_calls` 配对，含其 content）；② 对下一步决策有信息增益的（工具结果、结论）；③ 位置稳定且成本可控的（system 里的 todos / 摘要）。凡"只是过程记录"的不作为独立信息源注入，只落 trace；协议要求随消息保留的部分按协议处理。

### 1.6 Step two / Step four 怎么判定

- **Step two（直接回复还是调工具）**：不看关键词，由 `parser.parse()` 的
  `is_final` 决定——LLM 返回 `tool_calls` 就是"调工具"，只返回纯文本就是
  "直接回复"。判定权交给 LLM 基于 Schema 自主决策，Runtime 不硬编码规则。
- **Step four（继续 loop 还是返回）**：有工具调用 → 结果以 `role=tool`
  回灌后 `continue`；无工具调用 → 落盘并返回。另加 `max_rounds` 熔断
  与异常喂回两个护栏，见 1.1 / 6 节。

## 2. Memory：召回时机与放置方式（题目点名章节）

**memory 的本质问题是：什么状态、何时、以什么形态进入 context。**
本项目把状态拆成三层，每层召回时机与放置位置不同：

| 层 | 存放 | 内容 | 召回时机 | 放置方式 |
|---|---|---|---|---|
| 短期记忆 | `sessions.messages`（SQLite 全量持久化） | 完整对话历史 + 工具调用痕迹 | 每次构建 context 时，经滑动窗口裁剪整段带入 | 消息序列（供指代消解与多轮工具链） |
| 结构化记忆 | `todos` 表（挂 session_id） | 用户事实型状态（待办） | **每次 LLM 调用前全量注入** | system prompt 尾部，fenced 结构化块 |
| 压缩记忆 | `sessions.memory`（摘要缓存） | 被裁剪历史的摘要 | 超长触发压缩时生成，其后每轮随 system 注入 | system 末尾 `[此前对话摘要]` 行 |

**为什么 todos 用"每轮全量注入"而不是检索式召回**：本规模下待办只有
几条，全量注入的成本趋近于零且零漏召回；检索式（向量库/RAG）在这个
任务规模会引入漏召回风险与一整套基础设施依赖，是负优化。**工程取舍的
依据是规模，不是潮流。**

**为什么放 system 而不是 user/assistant 消息**：system 位置稳定、不参与
用户可读对话流、指令优先级最高；用 fenced block 结构化嵌入，LLM 定位
成本最低。

**诚实边界**：本项目未实现跨 session 长期记忆（题目未要求）。若扩展：
`memory.py` 增加"写入钩子"（assistant 最终回答后抽取用户事实）+
"检索召回"（新 session 首轮按相似度注入相关记忆），接口已预留。

## 3. Trace / 执行日志

每次工具调用、LLM 请求/响应、steering 注入、熔断都写入 SQLite trace 表
（结构化事件：args / result / duration_ms / round / status，**绝不记录密钥**）。
两个消费出口：`GET /api/sessions/{id}/trace` 与前端"工具调用过程"折叠面板
（演示时逐条展开讲 LLM 为什么调这个工具、传了什么参数、结果如何回灌）。

## 4. API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/sessions` | 新建会话（新开窗口） |
| GET | `/api/sessions` | 会话列表（侧栏） |
| GET | `/api/sessions/{id}/history` | 刷新后续聊：恢复对话 |
| POST | `/api/chat` | 主端点：`{session_id, message}` → 回答 + 工具调用明细 |
| POST | `/api/sessions/{id}/steer` | Agent 忙碌期间插入转向消息（pi 式 steering） |
| GET | `/api/sessions/{id}/trace` | 工具调用链查询 |
| GET | `/api/tools` | 当前注册的工具 Schema |
| GET | `/api/healthz` | 自检（不回显密钥） |

## 5. 测试

`python -m pytest tests/ -v`——46 个用例，mock `llm/client.py` 层离线回放：

| 文件 | 覆盖 |
|---|---|
| test_runtime.py | 直接回复 / 单工具 / 多轮工具 / 工具结果回灌 / Schema 传给 LLM / max_rounds 熔断 / 异常喂回 / 未知工具 / steering 注入 |
| test_parser.py | FC 提取 / XML 文本协议 / 接口可替换 |
| test_tools.py | calculator 安全求值 / todo 闭环 / Schema 格式 / mock 确定性 |
| test_context.py | 纯对话追问 / 带工具追问 / 摘要压缩与缓存 / 安全切割不拆配对 |
| test_session.py | 多 session 隔离 / 持久化续聊（含重启） / trace |
| test_api.py | 端到端 / trace 一致性 / 缺 key 明确报错 / LLM 异常 502 |
| test_security.py | 源码 grep 无 API Key 字面量（题目红线防呆） |

## 6. 设计取舍记录

1. **FC 为主 + Parser 接口化**：文本协议的问题不是不能做，而是要自己对抗
   格式漂移；FC 把结构化约束下沉到协议层，可靠性高一个量级，但不想绑死
   某家行为，所以 TextParser 留了可热切换的实现。
2. **工具异常喂回 LLM + max_rounds 熔断**：Agent 的错误处理对象是"循环"
   而不是"用户"——工具失败是 LLM 可读取并自我修正的 Observation；但自
   修正不能无限递归，用最大轮次做资源护栏，熔断时兜底返回已有信息。
3. **memory 三层 + 全量注入而非检索式**：按规模选复杂度（见第 2 节）。

## 7. 已知限制与路线

- 非流式响应（SSE 流式为后续加分项；FC 下真流式需先非流式判定是否调工具，复杂度翻倍）
- 摘要压缩用规则兜底时较粗糙；search/weather 为 mock（接入点已预留）
- 未实现跨 session 长期记忆（扩展路径见第 2 节）
- SQLite 全局锁写保护——生产应换 aiosqlite/连接池

## 8. AI Prompt 与问题解决记录

见 [docs/ai-usage.md](docs/ai-usage.md)：开发过程中使用 AI 辅助的记录、
关键 prompt 清单、自己拍板的取舍、踩坑与解决过程、验证记录。

## 9. 题目要求覆盖对照表（逐条销账）

| 题目要求 | 实现位置 | 验证用例 |
|---|---|---|
| 从零完成，不依赖 agent 框架 | `app/agent/runtime.py` 主循环全自研；`openai` SDK 仅协议适配 | 1.1 节论证 + `test_security.py` |
| Loop 四步（接收/判断/调工具/决定继续或返回） | `runtime.py:run()` + `parser.is_final` | `test_final_reply_only` / `test_single_tool_call_loop` / `test_tool_result_feeds_next_round_context` |
| ≥3 工具，todo 必须实现 | `app/tools/` 六个工具 | `test_tools.py` 8 例 |
| 注册机制：名称/描述/参数 Schema，LLM 基于 Schema 自主决策 | `tools/base.py` + `runtime.py` 把 `registry.schemas()` 传入 `llm.chat` | `test_registry_schemas_format` / `test_llm_receives_tool_schemas_for_autonomous_decision` |
| 输出解析：思考过程 / 工具调用 / 最终答案 | `agent/parser.py`（FC 主 + XML 备选），思考落 trace | `test_parser.py` 5 例 |
| 多窗口独立 session、随时续聊 | `session/store.py` 三表挂 `session_id` | `test_two_sessions_todo_isolation` / `test_resume_conversation_after_store_restart` |
| 最大轮次限制 | `runtime.py` `for round_no` + `_circuit_break` | `test_max_rounds_circuit_break` |
| 记住之前状态 + 纯对话/带工具两类追问 | `context.build()` 整段带入 | `test_plain_follow_up_uses_history` / `test_history_enables_follow_up_questions` |
| 哪些信息塞入 context 更合适 | 1.5 节 | —（文档回答题目提问） |
| context 过长基础压缩 | `context.py` 滑动窗口 + 摘要缓存 + `_safe_cut` | `test_context.py` 6 例 |
| 基本异常处理 | `_exec_tool` 异常喂回 + API 502 | `test_tool_exception_fed_back_not_raised` / `test_llm_failure_returns_502` |
| 工具调用 trace / 执行日志 | `store.log_trace` 7 个埋点 + trace 端点 + 前端面板 | `test_trace_log_and_query` / `test_full_chat_flow_with_tool` |
| 测试用例构建 | `tests/` 7 文件 46 例，FakeLLM 离线回放 | `python -m pytest tests/ -q` |
| 代码零 LLM API Key | `config.py` 环境变量注入 + `.env` 忽略 | `test_security.py` 2 例 |
| README：运行方式 / 系统设计 / memory 召回时机与放置方式 | 第 0 / 1 / **2** 节 | — |
| AI Prompt 与问题解决记录 | 第 8 节 + `docs/ai-usage.md` | — |
| 代码链接（GitHub） | 由提交者推送后提供，见第 10 节自查 | — |

超额项（题目未要求）：并行多工具调用、steering 插话、摘要级压缩、
零密钥 `demo.py` 演示、真实 MiniMax-M3 全链路验证。

## 10. 提交前自查（题目要求"发送 GitHub 链接即可"）

- [ ] `git init` 后先跑 `git status`，**确认 `.env` 没有出现在待提交列表**
      （`.gitignore` 已忽略；本地有真实订阅 Key，绝不能入库）
- [ ] `grep -rE "sk-[A-Za-z0-9]{20,}" app/ tests/ demo.py` 无输出
- [ ] `python -m pytest tests/ -q` 全绿
- [ ] README 三节齐全：运行方式 / 系统设计 / memory 召回时机与放置方式
- [ ] `docs/ai-usage.md` 已提交
- [ ] 推送后**只把仓库链接**发给对接人，不要附 Key

