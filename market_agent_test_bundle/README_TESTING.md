# unified_market_agent.py 全面测试包

这个测试包覆盖四层：

1. **纯函数 / 单元测试**
2. **状态机 / 触发器测试**
3. **端到端回放 pytest**
4. **独立 deterministic replay harness**

我已经用目标文件 `/mnt/data/unified_market_agent.py` 实跑过一遍，结果是：

- unit tests: **30 passed**
- state-machine tests: **20 passed**
- replay pytest: **1 passed**
- deterministic replay harness: **passed**

合计：**51 个 pytest 测试全部通过**。

---

## 一、这个测试包覆盖了什么

### 1) 基础逻辑
文件：`tests/test_unified_market_agent_unit.py`

覆盖内容：
- `validate_decision`
- `validate_condition`
- `validate_playbook`
- `evaluate_condition` 的 8 种条件类型
- `HyperliquidExecutor._apply_pct_exit`
- `HyperliquidExecutor.resolve_exit_levels`
- `build_symbol_position_signature`
- `build_all_positions_signature`
- `EventFileWatcher` 对坏 JSONL 的容错
- LLM 返回非法 JSON 时 `_call_model` 的失败路径
- live 模式下 `mid_price=None` 时跳过开仓

### 2) 状态机 / 触发器
文件：`tests/test_unified_market_agent_state_machine.py`

覆盖内容：
- active / passive search mode 分流
- `query_new_playbook`
- `execute_decision`
- `step_scenario_session`
  - arm
  - execute
  - cancel
  - timeout
- `step_risk_session`
  - take profit
  - stop loss
  - holding timeout
  - 手动改仓 / 改 size 触发 replan
- `active_query_allowed_now`
- `all_positions_changed`
- `run_forever` 关键 query reason：
  - `startup`
  - `passive_event_trigger`
  - `manual_or_external_position_change`
  - `active_periodic_refresh`

### 3) 端到端回放 pytest
文件：`tests/test_unified_market_agent_replay.py`

覆盖链路：
- scenario arm
- scenario execute
- 生成 risk session
- 命中 take profit
- 自动平仓
- 最终仓位回到 flat

### 4) 独立 replay 脚本
文件：`replay_agent_harness.py`

作用：
- 不依赖真实 OpenAI / Hyperliquid
- 用确定性价格带回放整个状态机
- 直接打印 action 流程
- 非常适合你后续每次改代码后的 smoke test

---

## 二、目录结构

```text
market_agent_test_bundle/
├── README_TESTING.md
├── replay_agent_harness.py
├── run_all_tests.sh
└── tests/
    ├── conftest.py
    ├── test_unified_market_agent_unit.py
    ├── test_unified_market_agent_state_machine.py
    └── test_unified_market_agent_replay.py
```

---

## 三、你应该怎么放置这些文件

你有两种方式。

### 方式 A：最省事
把这个测试包解压到 **和 `unified_market_agent.py` 同一个目录**。

例如：

```text
/workspace/
├── unified_market_agent.py
└── market_agent_test_bundle/
```

然后运行时用绝对路径指定目标脚本。

### 方式 B：放在任何目录
也可以把测试包放到别的目录，但运行时必须显式指定：

```bash
TARGET_MODULE_PATH=/absolute/path/to/unified_market_agent.py
```

---

## 四、一步一步运行测试

下面按推荐顺序来。

### 第 0 步：进入测试包目录

```bash
cd /path/to/market_agent_test_bundle
```

### 第 1 步：确认 pytest 可用

```bash
python -m pytest --version
```

如果这里报错，就先安装：

```bash
pip install pytest
```

### 第 2 步：只跑基础逻辑测试

```bash
TARGET_MODULE_PATH=/absolute/path/to/unified_market_agent.py \
python -m pytest -q tests/test_unified_market_agent_unit.py
```

你应该看到类似：

```text
30 passed
```

这一步先确认底层函数没坏。

### 第 3 步：跑状态机 / 触发器测试

```bash
TARGET_MODULE_PATH=/absolute/path/to/unified_market_agent.py \
python -m pytest -q tests/test_unified_market_agent_state_machine.py
```

你应该看到类似：

```text
20 passed
```

这一步确认：
- query 触发条件
- scenario arm/execute/cancel/timeout
- risk monitor TP/SL/timeout
- manual position change 触发 replan

### 第 4 步：跑端到端 replay pytest

```bash
TARGET_MODULE_PATH=/absolute/path/to/unified_market_agent.py \
python -m pytest -q tests/test_unified_market_agent_replay.py
```

你应该看到类似：

```text
1 passed
```

### 第 5 步：跑独立 replay harness

```bash
python replay_agent_harness.py --target /absolute/path/to/unified_market_agent.py
```

这一步会打印完整的回放过程，包括：
- tick
- scenario_armed
- scenario_execute
- risk_monitor_plan
- take_profit_hit
- final_position

最后应看到：

```text
[replay_ok] scenario arm -> entry -> take profit close works
```

### 第 6 步：一键跑完全部测试

```bash
./run_all_tests.sh /absolute/path/to/unified_market_agent.py
```

或者：

```bash
TARGET_MODULE_PATH=/absolute/path/to/unified_market_agent.py ./run_all_tests.sh
```

这个脚本会按顺序执行：
1. unit tests
2. state-machine tests
3. replay pytest
4. replay harness

---

## 五、你每次改代码后，推荐怎么测

### 日常小改
只跑：

```bash
TARGET_MODULE_PATH=/absolute/path/to/unified_market_agent.py \
python -m pytest -q tests/test_unified_market_agent_unit.py tests/test_unified_market_agent_state_machine.py
```

### 改了 scenario / risk / query 逻辑
跑：

```bash
./run_all_tests.sh /absolute/path/to/unified_market_agent.py
```

### 改了 run_forever / 执行链 / 价格触发逻辑
一定要再加上：

```bash
python replay_agent_harness.py --target /absolute/path/to/unified_market_agent.py
```

---

## 六、如果你想替换 replay 的价格带

你可以自己准备一个 JSON 文件，例如 `custom_prices.json`：

```json
[
  [100.0, 99.0],
  [101.0, 100.0],
  [102.0, 101.0],
  [103.0, 101.3],
  [104.0, 104.0],
  [105.0, 105.2]
]
```

然后运行：

```bash
python replay_agent_harness.py \
  --target /absolute/path/to/unified_market_agent.py \
  --prices /absolute/path/to/custom_prices.json
```

---

## 七、这套测试的边界

这套测试已经把你上次要求的重点都覆盖了，但它仍然是 **mock / replay 为主**，不是实盘联调。

它不直接覆盖：
- 真实 OpenAI API 网络调用
- 真实 Hyperliquid REST/下单
- 真实 `events.jsonl` 长时间运行竞争条件
- 真实多小时 / 多天级别运行的内存与状态积累问题

所以在这套测试全部通过之后，下一步仍然建议：

1. `ENABLE_LIVE_TRADING=false` 跑 dry-run
2. 连真实 Hyperliquid 读仓位和价格
3. 连真实 OpenAI 出 playbook
4. 连续观察一段时间日志
5. 最后才上小资金 live

---

## 八、最关键的判断标准

只要下面四点都成立，这版代码的“核心行为”就算通过：

1. **该 query 的时候一定 query，不该 query 的时候不会乱 query**
2. **scenario 只会在到关键位后 arm，不会提前执行**
3. **入场后 TP/SL/超时 能可靠关闭仓位**
4. **手动改仓 / 外部改仓后会触发 replan，不会继续沿用旧计划**

这套测试正是围绕这四点写的。
