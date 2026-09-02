# 30 分钟完整项目代码演示稿

> 用法：按时间顺序打开文件，照着“讲解台词”说；每句话下面紧跟对应代码。正常语速约 28 分钟，预留 2 分钟回答问题。

## 演示前准备（不计时）

按下面顺序预先打开编辑器标签页，演示时只需顺序切换：

1. `unified_market_agent.py`
2. `watchers/common.py`
3. `market_agent/backend/api.py`
4. `market_agent/backend/container.py`
5. `market_agent/workflow_harness.py`
6. `market_agent/workflow_production_application.py`
7. `market_agent/workflow_graph.py`
8. `market_agent/workflow_coordinator_agent.py`
9. `market_agent/workflow_agents/common.py`
10. `market_agent/workflow_agent_driver.py`
11. `market_agent/workflow_reflection_agent.py`
12. `market_agent/workflow_historical_answer_cache.py`
13. `market_agent/workflow_memory_retrieval.py`
14. `market_agent/workflow_audit.py`

演示主线只记这一句：

```text
事件/请求 → API → 队列 → Harness 状态机 → 生产工作流 → Coordinator
→ 专项 Agent → 模型/工具 → 客观反思 → 风险门禁 → 缓存与长期记忆 → 审计输出
```

---

## 0:00–1:30　项目定位与两条运行链路

### 讲解台词 1

“这个项目不是单纯把一个提示词发给一个大模型。原始链路负责采集市场事件、生成交易计划并提供交易操作；新生产链路在外面增加了 API、Harness、协调者、多 Agent、缓存、记忆、审计和安全门禁。”

代码：[README.md 第 1–25 行](../README.md#L1-L25)

屏幕上指出：`unified_market_agent.py`、`watchers/`、`web_trade/` 是原始业务入口，`market_agent/backend/` 与 `workflow_*` 是生产治理层。

### 讲解台词 2

“原始入口 `main()` 创建 `UnifiedMarketAgent` 后调用 `run_forever()`；它仍然保留，便于单机事件处理，但不是新的受治理 API 主链路。”

代码：[unified_market_agent.py 第 554–586 行](../unified_market_agent.py#L554-L586)

屏幕上指出：`main → UnifiedMarketAgent → run_forever`。

### 讲解台词 3

“生产 API 默认不允许调用旧的 `generate_playbook` 任务，只有显式打开 legacy 开关才允许，防止绕过 Harness。”

代码：[market_agent/backend/api.py 第 319–354 行](../market_agent/backend/api.py#L319-L354)

屏幕上指出：`execute_harness_workflow` 被禁止从通用任务端点调用，以及 `legacy_agent_task_enabled` 判断。

过渡句：“下面先用一分钟看业务数据从哪里来，再进入新的生产控制链。”

---

## 1:30–3:00　事件采集与原始业务闭环

### 讲解台词 1

“所有 watcher 输出统一的 `Event` 合约，包含来源、事件类型、时间、标题、正文和原始载荷；状态与 JSONL 写入也封装在公共层。”

代码：[watchers/common.py 第 51–160 行](../watchers/common.py#L51-L160)

屏幕上指出：`Event`、`StateStore`、`JsonlWriter` 三个类型。

### 讲解台词 2

“采集器都继承 `BaseWatcher`，HTML 和 RSS 只是不同适配器；网络请求自带次数、时间预算和退避约束。”

代码：[watchers/common.py 第 172–193 行](../watchers/common.py#L172-L193)；[HTML/RSS watcher 第 291–450 行](../watchers/common.py#L291-L450)；[请求重试第 451–507 行](../watchers/common.py#L451-L507)

屏幕上指出：公共基类、`HtmlPageWatcher`、`RssWatcher` 和请求预算。

### 讲解台词 3

“统一 Agent 将配置、LLM 引擎、行情、交易执行和事件文件 watcher 组合起来；事件循环增量读取 JSONL，然后处理新事件。”

代码：[unified_market_agent.py 第 222–552 行](../unified_market_agent.py#L222-L552)；[market_agent/agent_execution_loop.py 第 2005–2100 行](../market_agent/agent_execution_loop.py#L2005-L2100)

屏幕上指出：`EventFileWatcher`、`UnifiedMarketAgent` 的依赖组合，以及 `run_forever` 中的事件轮询。

### 讲解台词 4

“交易 Web 后端把杠杆、保证金、下单、止盈止损等输入定义成独立请求模型，并在路由入口统一做令牌校验。”

代码：[web_trade/backend/web_trade/api.py 第 17–67 行](../web_trade/backend/web_trade/api.py#L17-L67)

屏幕上指出：`LeverageRequest`、`OrderRequest`、`TpslRequest`、`_require_token`。

过渡句：“原始业务能力不改，新的部分是把调用过程收进一个可控制、可恢复、可审计的生产入口。”

---

## 3:00–5:30　API 合约、身份与 Trace

### 讲解台词 1

“API 合约默认 `extra='forbid'`，客户端多传字段会直接失败；工作流提交和接受响应都有稳定版本号、任务 ID、Harness ID 和 Trace ID。”

代码：[market_agent/backend/api_contracts.py 第 13–47 行](../market_agent/backend/api_contracts.py#L13-L47)

屏幕上指出：`ApiModel`、`WorkflowSubmission`、`WorkflowAccepted`。

### 讲解台词 2

“每个请求先解析或创建 W3C Trace Context，整条链路使用同一个 trace ID；响应头返回 `traceparent`，请求耗时和状态同时进入指标。”

代码：[market_agent/backend/api.py 第 145–194 行](../market_agent/backend/api.py#L145-L194)；[market_agent/workflow_tracing.py 第 26–104 行](../market_agent/workflow_tracing.py#L26-L104)

屏幕上指出：中间件中的 `extract_or_create`、响应头写入和 `same_trace`。

### 讲解台词 3

“身份验证使用 Bearer token；创建工作流时还校验 tenant、幂等键和 Harness 可用性，不能只凭请求正文越权。”

代码：[market_agent/backend/api.py 第 196–216 行](../market_agent/backend/api.py#L196-L216)；[工作流提交第 356–421 行](../market_agent/backend/api.py#L356-L421)

屏幕上指出：`_require_bearer_token`、tenant 校验、`Idempotency-Key` 和 `harness.create_run`。

### 讲解台词 4

“提交接口立即返回 `202 Accepted`，真实执行进入后台队列；查询、取消和事件读取是分开的资源接口。”

代码：[market_agent/backend/api.py 第 356–448 行](../market_agent/backend/api.py#L356-L448)

屏幕上指出：`POST /v1/workflows`、status URL、events URL、cancel 路由。

过渡句：“API 只负责接收和治理请求，下面看任务如何持久化并可靠执行。”

---

## 5:30–7:30　数据库、缓存、异步队列与依赖装配

### 讲解台词 1

“任务状态不是只放在内存中。SQLite 使用 WAL，保存 job、幂等记录和状态事件；状态迁移和事件写入在同一个存储抽象里。”

代码：[market_agent/backend/database.py 第 97–174 行](../market_agent/backend/database.py#L97-L174)；[状态迁移第 317–387 行](../market_agent/backend/database.py#L317-L387)

屏幕上指出：WAL、job 表、event 表和 transition 方法。

### 讲解台词 2

“后台队列是有界的，并支持任务注册、启动恢复、并发槽位、超时和重试；进程重启后会重新接管未完成任务。”

代码：[market_agent/backend/task_queue.py 第 33–95 行](../market_agent/backend/task_queue.py#L33-L95)；[恢复逻辑第 98–181 行](../market_agent/backend/task_queue.py#L98-L181)；[执行与重试第 353–400 行](../market_agent/backend/task_queue.py#L353-L400)

屏幕上指出：容量、handler registry、recover 和 retry 分支。

### 讲解台词 3

“多实例部署时，Redis 同时承担租户缓存和 Streams 消息总线；消费者包含 reclaim、ack、dead-letter 与异常退避。”

代码：[market_agent/backend/redis_adapters.py 第 116–174 行](../market_agent/backend/redis_adapters.py#L116-L174)；[Streams 第 216–323 行](../market_agent/backend/redis_adapters.py#L216-L323)；[消费者循环第 324–443 行](../market_agent/backend/redis_adapters.py#L324-L443)

屏幕上指出：key 带 tenant、consumer group、pending recovery 和 dead letter。

### 讲解台词 4

“容器是唯一装配点：开发环境可用本地实现，生产环境按配置接 Redis、PostgreSQL/pgvector、提示词版本管理、Harness 和生产工作流。”

代码：[market_agent/backend/container.py 第 52–95 行](../market_agent/backend/container.py#L52-L95)；[存储与提示词装配第 108–159 行](../market_agent/backend/container.py#L108-L159)；[生产工作流装配第 161–195 行](../market_agent/backend/container.py#L161-L195)

屏幕上指出：Composition Root 中的环境分支，不让业务模块自己读取基础设施。

过渡句：“任务进入队列后，第一道执行边界不是 LLM，而是确定性的 Harness。”

---

## 7:30–10:30　Harness：全局任务状态机与防死循环

### 讲解台词 1

“Harness 的计划模板是冻结的。只允许根据触发原因、模式和风险等级选择计划，用户自然语言不能临时改变执行拓扑。”

代码：[market_agent/workflow_plan_registry.py 第 54–132 行](../market_agent/workflow_plan_registry.py#L54-L132)；[计划选择边界第 203–214 行](../market_agent/workflow_plan_registry.py#L203-L214)

屏幕上指出：template、registry key 和禁止从用户 prose 选计划的判断。

### 讲解台词 2

“运行、工作项、尝试和结果都有枚举状态；计划创建时验证图结构，并明确规定被动分析不能包含执行副作用。”

代码：[market_agent/workflow_harness_contracts.py 第 25–97 行](../market_agent/workflow_harness_contracts.py#L25-L97)；[计划校验第 285–333 行](../market_agent/workflow_harness_contracts.py#L285-L333)

屏幕上指出：`RunState`、`WorkItemState`、`AttemptState` 和 passive side-effect validator。

### 讲解台词 3

“创建运行时先编译固定计划，再进行无副作用注册和带签名的二阶段注册；恢复时从持久化快照继续，而不是重新猜测执行到哪里。”

代码：[market_agent/workflow_harness.py 第 403–511 行](../market_agent/workflow_harness.py#L403-L511)

屏幕上指出：`create_run`、registration receipt、`resume_run` 和 snapshot。

### 讲解台词 4

“每次 advance 都由状态机决定下一个动作。结束、取消、失败和恢复是显式状态，不靠 Agent 自己说‘我完成了’。”

代码：[market_agent/workflow_harness.py 第 514–699 行](../market_agent/workflow_harness.py#L514-L699)

屏幕上指出：advance 的状态分支和 cancel 的终态处理。

### 讲解台词 5

“防死循环不只有最大步数：系统同时计算动作指纹、结果指纹和状态指纹，检测重复动作、循环周期、无进展和恢复次数。”

代码：[market_agent/workflow_loop_guard.py 第 395–457 行](../market_agent/workflow_loop_guard.py#L395-L457)；[停止判断第 458–555 行](../market_agent/workflow_loop_guard.py#L458-L555)

屏幕上指出：三个 fingerprint、cycle detection、progress comparison 和 recovery limit。

### 讲解台词 6

“Harness 的完成结果需要签名收据，并绑定计划、运行快照和视图；验签失败时生产环境 fail closed。”

代码：[market_agent/workflow_execution_backend.py 第 133–190 行](../market_agent/workflow_execution_backend.py#L133-L190)；[验签与绑定第 652–807 行](../market_agent/workflow_execution_backend.py#L652-L807)

屏幕上指出：signed receipt、pinned public key 和 exact binding。

过渡句：“Harness 决定能不能执行，真正的分析则进入生产工作流应用层。”

---

## 10:30–13:00　生产工作流入口：缓存、记忆、预算、执行

### 讲解台词 1

“生产应用先固定提示词版本和截止时间，然后在进入 Agent 之前查询历史答案缓存；命中就可以直接返回，不产生新的模型成本。”

代码：[market_agent/workflow_production_application.py 第 161–189 行](../market_agent/workflow_production_application.py#L161-L189)

屏幕上指出：prompt pin、active/passive deadline 和 `_lookup_historical_answer`。

### 讲解台词 2

“未命中时才记录请求、检索长期记忆并创建 Coordinator。给 Agent 的不是无限历史，而是经过摘要的核心经验和有限恢复上下文。”

代码：[market_agent/workflow_production_application.py 第 190–240 行](../market_agent/workflow_production_application.py#L190-L240)

屏幕上指出：request record、memory retrieval、bounded context 和 capability grant。

### 讲解台词 3

“主动模式和被动模式预算不同，且预算同时限制总成本、尝试次数和截止时间。工作流执行完还要验证 trace 一致性和审计健康。”

代码：[market_agent/workflow_production_application.py 第 242–288 行](../market_agent/workflow_production_application.py#L242-L288)

屏幕上指出：`0.75/0.30` 成本上限、attempt budget、graph invoke、trace check 和 audit health。

### 讲解台词 4

“依赖创建同样是惰性的：OpenAI、Embedding、断路器、降级链和缓存只有在生产请求需要时才创建，模型费用表也在装配时注入。”

代码：[market_agent/workflow_production_application.py 第 327–409 行](../market_agent/workflow_production_application.py#L327-L409)

屏幕上指出：OpenAI key、driver factory、retry/circuit/fallback/model cost wiring。

过渡句：“接下来进入核心：主 Agent 怎么拆任务，以及子 Agent 怎么被路由。”

---

## 13:00–16:00　Coordinator：拆分、路由、冲突与重排

### 讲解台词 1

“Coordinator 不直接包办所有分析，而是生成 1 到 5 个有依赖关系的子任务；每个任务带任务类型、模型等级、成本上限和 3 到 5 步分析预算。”

代码：[market_agent/workflow_coordinator_agent.py 第 39–93 行](../market_agent/workflow_coordinator_agent.py#L39-L93)

屏幕上指出：task creation、model tier、cost budget、analysis steps，以及 active/passive 的不同任务集合。

### 讲解台词 2

“任务先绑定摘要上下文和权限，再交给对应 Specialist；子 Agent 失败时不能伪造成功，无法确定的结论必须返回‘不知道’。”

代码：[market_agent/workflow_coordinator_agent.py 第 96–143 行](../market_agent/workflow_coordinator_agent.py#L96-L143)

屏幕上指出：context handoff、capability grant、dispatch 和 unknown fallback。

### 讲解台词 3

“如果子 Agent 冲突或报错，Coordinator 保留已经成功的报告，只替换失败或冲突任务；重排受修订次数和剩余预算约束。”

代码：[market_agent/workflow_coordinator_agent.py 第 146–222 行](../market_agent/workflow_coordinator_agent.py#L146-L222)；[market_agent/workflow_service_factory.py 第 86–138 行](../market_agent/workflow_service_factory.py#L86-L138)

屏幕上指出：conflict reconciliation、fatal fail-closed、revision limit、preserved reports。

### 讲解台词 4

“最终总结仍由 Coordinator 完成，但它只能使用已验证的子报告；核心决策任务固定使用 Terra，反思使用成本最低的 Luna。”

代码：[market_agent/workflow_coordinator_services.py 第 44–181 行](../market_agent/workflow_coordinator_services.py#L44-L181)

屏幕上指出：decision task 的 `TERRA`、4 steps、‘不知道’，以及 reflection 的 `LUNA` 与成本上限。

过渡句：“Coordinator 负责安排工作；权限、上下文和输出格式则由三个独立边界控制。”

---

## 16:00–18:30　上下文摘要、Agent 权限与结构化输出

### 讲解台词 1

“Agent 之间不直接传整段原始上下文。`ContextSelection` 先选取允许的信息，`ContextHandoff` 再验证来源、摘要、证据和长度。”

代码：[market_agent/workflow_context_summary.py 第 181–245 行](../market_agent/workflow_context_summary.py#L181-L245)；[交接校验第 261–362 行](../market_agent/workflow_context_summary.py#L261-L362)

屏幕上指出：selection、handoff、evidence references 和 size limits。

### 讲解台词 2

“每个 Agent 使用 capability grant：可以精确限制读哪些上下文、调用哪些工具、读写哪些状态以及访问哪些服务；未授权动作默认拒绝。”

代码：[market_agent/workflow_capabilities.py 第 62–108 行](../market_agent/workflow_capabilities.py#L62-L108)；[授权检查第 182–253 行](../market_agent/workflow_capabilities.py#L182-L253)

屏幕上指出：scope、grant、authorize read/tool/state/service 和 deny 分支。

### 讲解台词 3

“每个 Specialist 的职责、模型和 3 到 4 步推理预算都是静态配置：简单事件过滤走 Luna，常规专业分析走 Terra，升级与冲突处理才用 Sol。”

代码：[market_agent/workflow_agents/common.py 第 49–102 行](../market_agent/workflow_agents/common.py#L49-L102)

屏幕上指出：`SPECIALIST_PROFILES` 中六类角色、模型与 step budget。

### 讲解台词 4

“模型输出不是自由文本。调用合约和结果使用严格 Pydantic 模型，JSON Schema 校验失败就不会进入下游；证据和冲突也是结构化字段。”

代码：[market_agent/workflow_agent_contracts.py 第 19–126 行](../market_agent/workflow_agent_contracts.py#L19-L126)；[market_agent/workflow_agents/common.py 第 119–204 行](../market_agent/workflow_agents/common.py#L119-L204)

屏幕上指出：`extra='forbid'`、invocation constraints、result variants、schema validation、evidence/conflict binding。

过渡句：“完成这些约束后，AgentDriver 才被允许真正调用模型。”

---

## 18:30–21:30　AgentDriver：Prompt Cache、重试、断路器与分层降级

### 讲解台词 1

“AgentDriver 是所有模型调用的统一入口。它先固定提示词版本、检查上下文和记忆摘要，再按缓存、模型、降级的顺序执行，并把每一步写入审计。”

代码：[market_agent/workflow_agent_driver.py 第 166–305 行](../market_agent/workflow_agent_driver.py#L166-L305)

屏幕上指出：prompt release pin、cache lookup、model call、fallback、audit event。

### 讲解台词 2

“Prompt Cache 的关键是稳定前缀。系统提示词单独计算哈希作为 cache key，动态任务内容放在 user message；请求使用 Responses API 的严格 JSON Schema。”

代码：[market_agent/workflow_openai_client.py 第 15–61 行](../market_agent/workflow_openai_client.py#L15-L61)；[market_agent/workflow_prompt_release.py 第 22–92 行](../market_agent/workflow_prompt_release.py#L22-L92)

屏幕上指出：`stable_system_prompt`、hash、`prompt_cache_key`、dynamic canonical JSON 和 strict schema。

### 讲解台词 3

“稳定系统提示词不允许出现动态占位符；版本只能从 Git 跟踪的 release 加载，因此提示词和温度可以像代码一样审查、激活和回滚。”

代码：[market_agent/workflow_prompt_config.py 第 30–43 行](../market_agent/workflow_prompt_config.py#L30-L43)；[版本加载与回滚第 78–235 行](../market_agent/workflow_prompt_config.py#L78-L235)

屏幕上指出：dynamic-placeholder rejection、Git tracked check、activate、rollback 和 pending audit。

### 讲解台词 4

“单次模型调用同时受 deadline、最大尝试次数和美元成本约束。可重试错误采用 capped exponential backoff 加 full jitter，并尊重服务端 Retry-After。”

代码：[market_agent/workflow_agent_driver.py 第 434–539 行](../market_agent/workflow_agent_driver.py#L434-L539)；[market_agent/workflow_retry_policy.py 第 38–100 行](../market_agent/workflow_retry_policy.py#L38-L100)

屏幕上指出：attempt/cost/deadline checks、`uniform(0, cap)`、408/409/429/5xx 分类。

### 讲解台词 5

“断路器按 `(model, task_kind)` 隔离故障；连续失败后进入 OPEN，到恢复窗口只放一个 HALF_OPEN 探针，避免故障模型拖垮全系统。”

代码：[market_agent/workflow_circuit_breaker.py 第 21–62 行](../market_agent/workflow_circuit_breaker.py#L21-L62)

屏幕上指出：key、state transition 和 half-open probe。

### 讲解台词 6

“降级顺序固定为高一级模型到低一级模型，再到本地知识库，最后返回‘不知道’；任何一层都不能绕过结构化校验和安全规则。”

代码：[market_agent/workflow_fallback.py 第 31–65 行](../market_agent/workflow_fallback.py#L31-L65)；[market_agent/workflow_agent_driver.py 第 361–393 行](../market_agent/workflow_agent_driver.py#L361-L393)

屏幕上指出：Sol → Terra → Luna → local knowledge → abstain。

过渡句：“模型有结果并不代表结果可信，核心步骤还要经过客观反思和风险门禁。”

---

## 21:30–23:30　LangGraph、客观反思与纠错停止

### 讲解台词 1

“固定工作流由 LangGraph 表达：plan、dispatch、recover、decide、reflect、risk、assemble、finalize 都是显式节点和条件边。”

代码：[market_agent/workflow_graph.py 第 54–186 行](../market_agent/workflow_graph.py#L54-L186)；[图装配第 189–227 行](../market_agent/workflow_graph.py#L189-L227)

屏幕上指出：node functions、conditional edges 和 safe failure path。

### 讲解台词 2

“反思只用于核心决策，不评价文风，只检查格式、关键字段、证据引用、数字一致性、风险约束和结论是否自洽。”

代码：[market_agent/workflow_reflection_agent.py 第 71–205 行](../market_agent/workflow_reflection_agent.py#L71-L205)；[market_agent/workflow_decision_verifier.py 第 36–100 行](../market_agent/workflow_decision_verifier.py#L36-L100)

屏幕上指出：Luna、temperature 0、objective checks、core decision guard。

### 讲解台词 3

“纠错时把验证错误写进 `CorrectionContext` 再重试；先做允许字段内的补丁，补丁无效才完整重写，而且禁止修改身份、Schema 和证据边界。”

代码：[market_agent/workflow_reflection_agent.py 第 208–267 行](../market_agent/workflow_reflection_agent.py#L208-L267)

屏幕上指出：error context、patch allowlist 和 immutable fields。

### 讲解台词 4

“纠错最多两轮，并比较错误是否减少；结果哈希重复或错误变多就立即停止，最终走安全的 no-trade/不知道兜底。”

代码：[market_agent/workflow_reflection_agent.py 第 278–335 行](../market_agent/workflow_reflection_agent.py#L278-L335)

屏幕上指出：max corrections、improvement guard、repeated hash、rewrite fallback。

### 讲解台词 5

“反思通过后还必须过确定性风险门禁；输入不足、约束冲突或超出政策时直接拒绝，不由 LLM 自行放行。”

代码：[market_agent/workflow_risk_gate.py 第 15–70 行](../market_agent/workflow_risk_gate.py#L15-L70)；[market_agent/workflow_playbook_assembler.py 第 13–50 行](../market_agent/workflow_playbook_assembler.py#L13-L50)

屏幕上指出：`RiskPolicy`、reject reason 和 `unknown_playbook`。

过渡句：“高质量结果会被缓存；但缓存命中也必须重新验证版本、租户和安全元数据。”

---

## 23:30–25:30　高频答案与 95% 向量缓存

### 讲解台词 1

“历史请求保存请求向量、响应、时间戳、模型版本、Embedding 版本、提示词版本、Schema 和安全策略版本，避免旧结果跨版本误用。”

代码：[market_agent/workflow_historical_answer_cache.py 第 15–84 行](../market_agent/workflow_historical_answer_cache.py#L15-L84)

屏幕上指出：`HistoricalAnswerMetadata`、request/response timestamp 和 vector record。

### 讲解台词 2

“相似度条件是严格大于 0.95，不是大于等于；命中前还要检查 tenant、版本兼容和过期时间，相同分数使用确定性排序。”

代码：[market_agent/workflow_historical_answer_cache.py 第 93–130 行](../market_agent/workflow_historical_answer_cache.py#L93-L130)

屏幕上指出：`score > threshold`、metadata match、expires_at 和 deterministic tie-break。

### 讲解台词 3

“生产实现使用 PostgreSQL 加 pgvector，在 SQL 层完成向量距离筛选和 TTL 清理；仓库还预置了五个高频固定答案种子。”

代码：[market_agent/workflow_historical_answer_cache.py 第 133–233 行](../market_agent/workflow_historical_answer_cache.py#L133-L233)

屏幕上指出：DDL、vector operator、`> 0.95`、cleanup 和 fixed seeds。

### 讲解台词 4

“缓存真正接在生产入口最前面，只缓存静态信息类请求；写回时附带完整元数据，交易时效性问题不会被当成固定答案长期复用。”

代码：[market_agent/workflow_production_application.py 第 723–834 行](../market_agent/workflow_production_application.py#L723-L834)

屏幕上指出：static informational detection、lookup、metadata build 和 store。

过渡句：“短期复用由缓存解决，跨请求学习则进入三层长期记忆。”

---

## 25:30–27:30　三层长期记忆、RAG 闭环与遗忘

### 讲解台词 1

“长期记忆分三层：事件层保存原材料，知识层保存经过验证的经验，决策层保存决策、结果和教训；所有写入都带 tenant、版本和来源。”

代码：[market_agent/workflow_long_term_memory.py 第 85–218 行](../market_agent/workflow_long_term_memory.py#L85-L218)

屏幕上指出：Event、Knowledge、Decision、Outcome、Lesson contracts。

### 讲解台词 2

“存储采用 PostgreSQL：原始记录、当前 head、幂等记录、审计记录和 pgvector 索引分开；Agent 没有直写权限，只有受信 host writer 能变更记忆。”

代码：[market_agent/workflow_memory_postgres.py 第 42–141 行](../market_agent/workflow_memory_postgres.py#L42-L141)

屏幕上指出：tables、tenant boundary、writer authority 和 knowledge activation。

### 讲解台词 3

“新任务先构造带 tenant、scope、Top K、版本和新鲜度约束的查询，再做向量与关键词混合检索；证据冲突会保留，最后压缩成核心经验摘要注入上下文。”

代码：[market_agent/workflow_memory_retrieval.py 第 25–121 行](../market_agent/workflow_memory_retrieval.py#L25-L121)；[检索与摘要第 122–377 行](../market_agent/workflow_memory_retrieval.py#L122-L377)

屏幕上指出：`MemoryQuery`、hybrid similarity、Top K、evidence conflict、`CoreExperienceSummary`。

### 讲解台词 4

“执行结束后由 host 记录事件、决策和结果，符合验证条件的经验再晋升到知识层，形成‘检索—注入—执行—更新’闭环。”

代码：[market_agent/workflow_memory_result_writer.py 第 16–69 行](../market_agent/workflow_memory_result_writer.py#L16-L69)；[market_agent/workflow_memory_promotion.py 第 15–37 行](../market_agent/workflow_memory_promotion.py#L15-L37)

屏幕上指出：result writer 和 verified promotion。

### 讲解台词 5

“遗忘不是简单删除：先做置信度衰减，再按保留期、容量、保护标记和 legal hold 生成 archive、tombstone 或 purge 计划；后台维护任务周期执行。”

代码：[market_agent/workflow_memory_lifecycle.py 第 17–182 行](../market_agent/workflow_memory_lifecycle.py#L17-L182)；[LifecycleWorker 第 185–238 行](../market_agent/workflow_memory_lifecycle.py#L185-L238)；[market_agent/backend/memory_maintenance.py 第 14–67 行](../market_agent/backend/memory_maintenance.py#L14-L67)

屏幕上指出：confidence decay、protected/legal hold、three actions 和 scheduler。

过渡句：“最后看一条请求怎样被完整追踪，以及提示词怎样安全发布和回退。”

---

## 27:30–29:15　全链路审计、日志、指标和提示词回滚

### 讲解台词 1

“审计事件强制绑定 trace、tenant、run、work item 和 attempt，并记录模型、提示词、Schema、token、成本、结果摘要；敏感字段在写入前拒绝或脱敏。”

代码：[market_agent/workflow_audit.py 第 299–425 行](../market_agent/workflow_audit.py#L299-L425)

屏幕上指出：audit payload、trace binding、usage/cost、secret rejection 和 Agent 只读 observer。

### 讲解台词 2

“审计库采用 WAL、索引和 append-only 触发器，业务代码不能更新或删除历史审计记录。”

代码：[market_agent/workflow_audit.py 第 525–552 行](../market_agent/workflow_audit.py#L525-L552)；[不可变约束与写入第 641–719 行](../market_agent/workflow_audit.py#L641-L719)

屏幕上指出：append-only triggers、indexes 和 append/list API。

### 讲解台词 3

“结构化日志只保留安全摘要；指标限制标签基数并统计成功率、延迟、token 和成本；工具调用单独记录输入摘要、输出摘要和耗时。”

代码：[market_agent/workflow_structured_logging.py 第 45–130 行](../market_agent/workflow_structured_logging.py#L45-L130)；[market_agent/workflow_metrics.py 第 55–122 行](../market_agent/workflow_metrics.py#L55-L122)；[market_agent/workflow_tool_observability.py 第 26–99 行](../market_agent/workflow_tool_observability.py#L26-L99)

屏幕上指出：redaction、bounded labels、histogram、tool span。

### 讲解台词 4

“运维可以通过 API 查看 metrics 和 trace，也可以激活或回滚提示词版本。发布事故时不改代码即可切回上一 release，但操作本身仍有审计。”

代码：[market_agent/backend/api.py 第 288–317 行](../market_agent/backend/api.py#L288-L317)；[观测接口第 475–490 行](../market_agent/backend/api.py#L475-L490)

屏幕上指出：prompt activate/rollback、`/metrics` 和 trace endpoint。

过渡句：“最后用发布门禁和生产信任边界收尾。”

---

## 29:15–30:00　评测门禁、生产密钥边界与收尾

### 讲解台词 1

“评测不是只算一个平均分。它记录每个 case 的正确性、安全性、成本和延迟，再与 baseline 做成对比较；安全回归是硬阻断条件。”

代码：[market_agent/workflow_evaluation.py 第 14–119 行](../market_agent/workflow_evaluation.py#L14-L119)；[market_agent/workflow_eval_metrics.py 第 12–130 行](../market_agent/workflow_eval_metrics.py#L12-L130)

屏幕上指出：case score、aggregate、paired comparison、release thresholds 和 hard safety gate。

### 讲解台词 2

“生产环境还要求 Redis、PostgreSQL、模型版本固定和受信 Host Factory。HSM/KMS 的仓库职责是定义协议和 fail-closed 信任边界，具体云厂商适配器由部署方注入。”

代码：[market_agent/backend/settings.py 第 92–159 行](../market_agent/backend/settings.py#L92-L159)；[market_agent/backend/governed_bootstrap.py 第 17–45 行](../market_agent/backend/governed_bootstrap.py#L17-L45)；[docs/harness-host-factory.md 第 1–15 行](harness-host-factory.md#L1-L15)

屏幕上指出：生产配置校验、`MARKET_AGENT_HARNESS_HOST_FACTORY` 和 trusted signer/verifier boundary。

### 收尾台词

“所以这个项目的核心不是让更多 Agent 自由发挥，而是用 Harness 固定流程，用 Coordinator 拆分任务，用最合适的模型执行，用结构化输出、客观反思和风险门禁约束结果，再用缓存、长期记忆和全链路审计把它变成可运营的生产系统。”

代码：[market_agent/workflow_graph.py 第 189–227 行](../market_agent/workflow_graph.py#L189-L227)；[market_agent/workflow_harness_application.py 第 40–137 行](../market_agent/workflow_harness_application.py#L40-L137)

屏幕上指出：LangGraph 内部流程与 Harness 外部状态机是两层控制，不是让 LLM 自己决定全局状态。

---

## 现场演示操作清单

如果需要边讲边发一个请求，按以下顺序操作：

1. 提交 `POST /v1/workflows`，请求头带 `Authorization`、`Idempotency-Key` 和可选 `traceparent`。
   代码：[market_agent/backend/api.py 第 356–421 行](../market_agent/backend/api.py#L356-L421)
2. 从 `202` 响应读取 `workflow_id`、`status_url`、`events_url` 和 `trace_id`。
   代码：[market_agent/backend/api_contracts.py 第 32–47 行](../market_agent/backend/api_contracts.py#L32-L47)
3. 查询 status，展示任务从 accepted/running 到终态。
   代码：[market_agent/backend/api.py 第 423–448 行](../market_agent/backend/api.py#L423-L448)
4. 查询 events，展示状态迁移而不是只展示最后答案。
   代码：[market_agent/backend/database.py 第 317–387 行](../market_agent/backend/database.py#L317-L387)
5. 用返回的 trace ID 查询链路，并打开 `/metrics` 展示延迟、成功率、token 和成本。
   代码：[market_agent/backend/api.py 第 475–490 行](../market_agent/backend/api.py#L475-L490)

---

## 高频问题的 20 秒回答

### Q1：为什么同时有 Harness 和 LangGraph？

回答：“Harness 管外层的可信状态、恢复、预算、循环检测和完成签名；LangGraph 管一次分析内部的固定节点编排。前者不信任 LLM，后者提高 Agent 协作效率。”

代码：[market_agent/workflow_harness.py 第 403–511 行](../market_agent/workflow_harness.py#L403-L511)；[market_agent/workflow_graph.py 第 189–227 行](../market_agent/workflow_graph.py#L189-L227)

### Q2：为什么不让一个最强模型完成全部任务？

回答：“任务类型、模型等级、3–5 步预算和权限都被拆开。简单过滤走 Luna，专业分析走 Terra，只有升级和冲突才走 Sol，这样成本、风险和故障域都更可控。”

代码：[market_agent/workflow_coordinator_agent.py 第 39–93 行](../market_agent/workflow_coordinator_agent.py#L39-L93)；[market_agent/workflow_agents/common.py 第 79–102 行](../market_agent/workflow_agents/common.py#L79-L102)

### Q3：Prompt Cache 是否真的启用？

回答：“是。系统提示词作为稳定前缀，动态数据放 user message，并把稳定前缀哈希传给 `prompt_cache_key`；系统提示词中还禁止动态占位符。”

代码：[market_agent/workflow_openai_client.py 第 35–61 行](../market_agent/workflow_openai_client.py#L35-L61)；[market_agent/workflow_prompt_config.py 第 30–43 行](../market_agent/workflow_prompt_config.py#L30-L43)

### Q4：缓存会不会把过期或别人的答案返回？

回答：“不会只看向量分数。系统还检查 tenant、模型、Embedding、提示词、Schema、安全策略、上下文和知识版本以及过期时间；相似度必须严格大于 95%。”

代码：[market_agent/workflow_historical_answer_cache.py 第 15–130 行](../market_agent/workflow_historical_answer_cache.py#L15-L130)

### Q5：Agent 能不能直接修改长期记忆？

回答：“不能。Agent 只获得受限上下文和 capability grant，持久记忆写入必须经过受信 host writer；知识晋升与遗忘也由独立生命周期服务执行。”

代码：[market_agent/workflow_capabilities.py 第 182–253 行](../market_agent/workflow_capabilities.py#L182-L253)；[market_agent/workflow_memory_postgres.py 第 62–141 行](../market_agent/workflow_memory_postgres.py#L62-L141)

### Q6：反思会不会把正确答案越改越错？

回答：“验证器只做客观检查；纠错受字段白名单、最多两轮、错误数量改善和重复哈希检测约束。错误增加就停止，完整重写只作为最后兜底。”

代码：[market_agent/workflow_reflection_agent.py 第 208–335 行](../market_agent/workflow_reflection_agent.py#L208-L335)

### Q7：HSM/KMS 现在是否已经能直接部署？

回答：“仓库已经有供应商无关协议、Host Factory 和签名验签边界，但没有绑定某一家云厂商。部署时需要注入 AWS KMS、GCP KMS、Azure Key Vault 或实际 HSM 的适配器、身份权限和密钥资源。”

代码：[market_agent/backend/governed_bootstrap.py 第 17–45 行](../market_agent/backend/governed_bootstrap.py#L17-L45)；[market_agent/workflow_execution_backend.py 第 652–807 行](../market_agent/workflow_execution_backend.py#L652-L807)

---

## 演示时不能说错的边界

1. 不要说“全部数据都在 PostgreSQL”。任务、幂等和本地审计目前有 SQLite 实现；长期记忆和向量缓存有 PostgreSQL/pgvector 生产实现。
   代码：[market_agent/backend/database.py 第 97–174 行](../market_agent/backend/database.py#L97-L174)；[market_agent/workflow_memory_postgres.py 第 42–53 行](../market_agent/workflow_memory_postgres.py#L42-L53)
2. 不要说“HSM/KMS 已经绑定某家云厂商”。当前是协议、验签器和部署注入点；缺少受信 Host Factory 时生产启动失败。
   代码：[market_agent/backend/settings.py 第 92–159 行](../market_agent/backend/settings.py#L92-L159)；[market_agent/backend/governed_bootstrap.py 第 22–45 行](../market_agent/backend/governed_bootstrap.py#L22-L45)
3. 不要说“所有请求都会走历史答案缓存”。只有被判断为静态信息类的请求才允许直接复用。
   代码：[market_agent/workflow_production_application.py 第 723–834 行](../market_agent/workflow_production_application.py#L723-L834)
4. 不要说“Agent 自己可以宣布完成”。完成状态由 Harness、持久化快照和签名收据共同决定。
   代码：[market_agent/workflow_harness_application.py 第 69–137 行](../market_agent/workflow_harness_application.py#L69-L137)；[market_agent/workflow_execution_backend.py 第 729–807 行](../market_agent/workflow_execution_backend.py#L729-L807)
5. 不要在没有实际执行验收命令时说“所有测试和安全扫描已经通过”。代码内有测试集和 release gate，但运行结果必须以当次 CI/命令输出为准。
   代码：[market_agent/workflow_eval_metrics.py 第 84–130 行](../market_agent/workflow_eval_metrics.py#L84-L130)；[evaluation/workflow_acceptance.jsonl](../evaluation/workflow_acceptance.jsonl)
