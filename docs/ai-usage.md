# AI Prompt 与问题解决记录

记录 vibe coding 题中 AI 辅助的真实用法。对应题目注意事项第 3 条：
**用大模型帮助思考，而不是用大模型替自己完成任务**——AI 负责列方案、
找差异、生成骨架和跑回归；技术选型由候选人拍板，结论由实测验证。

## 1. 总体使用方式

- **先要方案对比，再要代码**：工具注册的 A/B/C 三方案、SQLite vs JSON、
  CLI vs FastAPI、FC vs 文本协议——都先让 AI 给优缺点与成本，自己选。
- **先要差异清单，再决定是否动手**：把"题目要求逐条对账"作为独立步骤，
  先输出缺口（架构题 0/5、纯对话追问用例缺失、README 写错 `.env`），
  再决定改哪些，避免 AI 自作主张改代码。
- **AI 产出必须过验证**：每次改动后跑 `pytest`、真实 LLM 链路、健康检查和
  密钥 grep，AI 说"完成"不算完成。

## 2. 向 AI 追问的问题（用 AI 帮思考的证据）

| 追问 | 得到的结论 |
|---|---|
| "不能依赖现有 agent 框架，但代码里用了 openai SDK，算吗？" | 判据是**库是否替你写了主流程**：langgraph/openhands 实现 agent loop 所以禁；SDK 只做 HTTP 协议适配所以不算。这条边界写进了 README 1 节 |
| "用 pi agent 的设计理念怎么样？" | 分层、极简 harness、steering、compaction 可移植；但 pi 的"无步数限制"与题目"最大轮次限制"冲突，因此保留 `max_rounds`，并在 README 里主动说明**刻意偏离 pi 的理由** |
| "CLI 和 FastAPI 有什么区别？" | 先要差异表再选形态，避免直接开工后发现工作量翻倍 |
| "对比题目现在符合要求吗？" | 拿到逐条对账表，发现 README 的 `.env` 表述、纯对话追问用例、架构题 0/5 三处缺口 |

## 3. 由候选人确认的技术决策

| 决策点 | 候选方案 | 最终选择 | 理由（候选人确认） |
|---|---|---|---|
| 实现形态 | CLI / CLI+HTTP / FastAPI+前端 | **FastAPI + 单文件前端** | 要能直观演示多窗口独立 session |
| LLM 输出解析 | FC / 文本协议 / 两者都提 | **Function Calling 为主** | 解析可靠；Parser 接口预留文本协议备份 |
| 密钥注入 | 硬编码 / 环境变量 | **环境变量 + `.env` 自动加载** | 题目红线，绝不把 Key 写进代码 |
| 设计理念 | 常规实现 / pi 理念移植 | **理念移植版** | 分层对标 pi-ai / pi-agent-core / harness，保留题目要求的轮次护栏 |
| 两部分优先级 | 架构题优先 / Coding 题优先 | **Coding 题优先** | 先把 70 分线拿稳 |
| 真实模型 | DeepSeek / GLM / MiniMax 订阅 | **MiniMax Coding Plan + MiniMax-M3** | 候选人自己提供订阅 Key 并指定模型，实测 FC 可用 |
| 缺口是否修 | 只修 README / 两个都修 / 都不修 | **两个都修** | `.env` 加载 + 压缩参数死配置 |

## 4. 关键问题与解决

### 4.1 `assistant(tool_calls)` 消息必须原样落历史
**现象**：工具结果直接拼进历史后，下一轮请求 400。
**根因**：OpenAI 协议强约束——`role=tool` 必须紧跟带对应 `tool_call_id` 的 assistant 消息。
**解决**：`LLMResponse` 携带 `raw_message`，runtime 原样 append；`tests/test_runtime.py::test_single_tool_call_loop` 断言配对。

### 4.2 压缩切割点不能拆散工具配对
**现象**：按条数硬切窗口，可能把 `assistant(tool_calls)` 切在窗口头、tool 消息被丢进压缩区。
**解决**：`_safe_cut` 只回退到最近的 user 消息；一个 user 开启的完整交换（含全部工具配对）要么整体保留要么整体丢弃。`test_safe_cut_never_splits_tool_pairing` 证伪。

### 4.3 Runtime 构造时 LLM 客户端尚未创建
**现象**：summarizer 在 `__init__` 绑定了当时的 `llm`（None），启动创建真客户端后摘要器仍指向 None。
**解决**：改为延迟绑定（`self._summarize` 在调用时读 `self.llm`）。

### 4.4 `.env` 写了却不生效
**现象**：README 让用户 `cp .env.example .env` 后直接 `uvicorn`，但全仓只有 `os.getenv()`，`.env` 根本没被加载。
**解决**：`config.py` 加 `load_dotenv(override=False)`（可选依赖，环境变量优先），`requirements.txt` 加 `python-dotenv`；README 同步改成 `cp .env.example .env` 并明确"clone 后需自行创建"。

### 4.5 两个"死配置"
**现象**：`MAX_HISTORY_MESSAGES / RECENT_KEEP` 在 `config.py` 定义却无消费方，改了不生效。
**解决**：`main.py` 在 Runtime 构造后接线 `runtime.context.max_history / recent_keep`，并用环境变量注入验证生效。

### 4.6 零密钥演示无法启动
**现象**：无 `LLM_API_KEY` 时 lifespan 的 `validate_llm()` 抛错，`Application startup failed`。
**解决**：不改成"无 key 静默启动"（那样会掩盖配置错误），而是新增 `demo.py`：注入 `ScriptedLLM` 后 `runtime.llm` 非空，lifespan 跳过真客户端创建——**零密钥、零联网、结果可复现**，生产路径的明确报错保持不变。

### 4.7 剧本 LLM 会误消费压缩请求
**现象**：`summarize_with_llm` 会再发一次带"对话压缩器" system 的 `llm.chat`，若剧本按顺序消费会打乱主链路。
**解决**：`ScriptedLLM.chat` 优先识别压缩请求特征并返回固定摘要，再按"最后一条 user 消息 + 已有工具结果数"路由主剧本。

### 4.8 真实 MiniMax 订阅接入验证
**过程**：填 Key → `python demo.py --check` → 端点返回 200 且模型主动调 `calculator` → 再跑 4 轮真实全链路 → 观察并行双工具调用、追问时自主追加工具、待办落库 → 最后切 `uvicorn app.main:app` 并用 curl 验证 `/api/chat`。
**结论**：FC 解析、多工具回灌、历史注入、session 持久化在真实模型下成立，不是 mock 自嗨。

### 4.9 注释声称覆盖但测试没覆盖
**现象**：`test_context.py` docstring 写"纯对话 + 带工具两类追问"，实际只有带工具追问。
**解决**：拆成两个用例，补 `test_plain_follow_up_uses_history`，并把注释改成与实际断言一致。

## 5. AI Prompt 清单（按用途）

**方案对比类**
- "对比工具注册的手写 dict / 装饰器+Pydantic / docstring 推导三方案，从 Schema 与实现一致性、新增工具成本两个维度打分"
- "对比 SQLite 与 JSON 文件做 session 存储：并发写、隔离、关联查询、面试观感"
- "对比 CLI 与 FastAPI+前端：工作量、session 隔离的体现方式、评分点契合度"
- "分析 pi agent 的设计理念能移植哪些、哪些与本题冲突"

**实现约束类**
- "把 OpenAI 协议里 tool 消息的配对约束写成 runtime 的不变式，并设计一个能证伪它的测试用例"
- "要求：至少 3 工具含 todo、注册机制带参数 Schema、多窗口独立 session、最大轮次、两类追问、基础压缩、trace、41+ 离线用例；代码零密钥"
- "设计一个零密钥、可在浏览器点、能演示完整链路的 demo 入口，注意不要污染笔试产物数据库"

**对账与排查类**
- "逐条对照题目原文列出当前仓库的符合/不符合项，先给差异清单不要动手改"
- "检查没有 API key 时 uvicorn 能否启动、.env 是否会被自动加载、两个压缩环境变量有没有消费方"
- "真实订阅端点报错时给出排查顺序：Key 类型 → 端点 → 模型名 → 网络/额度"

**文档类**
- "README 的 memory 章节要写清三层状态、召回时机、放置位置理由，并主动声明边界"
- "回答题目提问：哪些信息塞进 context 更合适，哪些只落 trace，给判断标准"

## 6. 验证记录

| 验证 | 结果 |
|---|---|
| `python -m pytest tests/ -q` | **46 passed**（离线，FakeLLM 脚本回放） |
| `tests/test_security.py` + `grep -rE "sk-..."` | 源码无 Key 字面量，`.env` 已被 `.gitignore` 忽略 |
| `python demo.py --replay` | 8 个场景：多轮工具链 / 追问 / 单工具 / 熔断 / 异常喂回 / steering / 压缩 / 双窗口隔离 |
| `python demo.py --check` | MiniMax 订阅端点 200，模型主动调 `calculator`，FC 可用 |
| 真实模型 4 轮全链路 | 并行双工具、自主追问、待办落库，全部 2 轮收敛无截断 |
| `uvicorn app.main:app` + curl | healthz 200，`/api/chat` 返回 `88 ÷ 4 = 22` |
| `MAX_HISTORY_MESSAGES=8` 注入 | `runtime.context.max_history == 8`，死配置已接线 |

## 7. 诚实边界

- 代码骨架、测试与文档由 AI 辅助生成；**选型、取舍论证、真实模型验证由候选人完成并确认**（第 3 节）。
- 本仓库只含 vibe coding 题的提交物；架构设计题 5 模块各 1 题另行完成，不在本次提交范围。
- 全部测试与演示默认离线可复现；接真实模型时会消耗订阅额度。
