# Hyperliquid Funding Hedge

这个独立项目实现的是：**主账户原仓位始终不动，在专用子账户/钱包对同一份永续合约做临时反向仓，确认目标小时的 `userFunding` 记录后，再 reduce-only 平掉对冲仓。**

它不再采用“主仓结算前平仓、结算后买回”的旧方案，因此不会造成亏损主仓平掉后因保证金不足而买不回原数量。

## 当前边界

- 主账户只有 `/info` 读取路径；代码没有任何向主账户发单的接口。
- 子账户订单同时绑定 `account_address=hedge_address` 与 `vault_address=hedge_address`。
- 默认只报告，不下单。实盘必须同时提供 CLI `--execute` 和 `FUNDING_HEDGE_EXECUTE=true`。
- `--execute` 强制要求 `--watch`，不允许开仓后立即退出的一次性实盘进程。
- 不创建子账户、不自动转账、不触碰人工挂单，也不自动取消未知 CLOID。
- 当前 REST 保证金检查只允许 Standard account mode 开新仓；Unified/Portfolio Margin 因 dex-scoped `withdrawable` 无法可靠代表可交易余额而 fail closed。已有 tracked hedge 仍允许安全退出。
- live executor 目前使用 SDK `order` 构造 aggressive IOC；退出始终 `reduce_only=True`。maker/ALO 尚未开放，避免整点后遗留挂单成交。

## 非常重要：真实合约名

截至 2026-07-10，界面显示 `SKHYNIX` 的真实 L1 coin 是 **`xyz:SKHX`**，不是 `xyz:SKHYNIX`；同时还存在另一个不同合约 `xyz:SKHY`。

最安全的做法是读取主账户 `clearinghouseState(dex="xyz")` 返回的 `position.coin`，把这个 exact coin 写入 `FUNDING_HEDGE_SYMBOLS`。程序不会根据显示名猜合约，也不会剥掉 `xyz:` 前缀。

## 决策与成本

正 funding 时 long 付款、short 收款；负 funding 时相反。每小时结算，美元金额使用 oracle price：

```text
primary_debit = abs(primary_size) * oracle_price * funding_rate * sign(primary_size)
hedge_credit  = max(0, -abs(hedge_size) * oracle_price
                        * funding_rate * sign(hedge_size))

round_trip_cost = hedge_notional
                  * [2 * (taker_fee + extra_fee + slippage) + risk_buffer]
                  * cost_safety_multiplier
```

只有当前小时预计 hedge credit 覆盖开/平手续费、滑点、缓冲并达到最低净节省时才允许开仓。`hedge_ratio=0.4` 会按精度向下取整，只对冲约 40%，保留约 60% 净方向敞口。

HIP-3 taker fee 不硬编码。适配器用 hedge address 的 `userFees`、DEX 的 `perpDexs.deployerFeeScale` 与资产的 `growthMode` 按[官方公式](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees#fee-formula-for-developers)动态计算。以当前 tier-0、无 referral 的 growth-mode `xyz:SKHX` 为例，约为 0.009%/腿，而不是主 DEX 基础值 0.045%/腿。动态字段缺失时禁止开新 hedge，除非明确启用保守的手工 fallback。

## 状态机与恢复

```text
ARMED -> OPEN_SUBMITTED -> OPEN_PARTIAL/HEDGED
      -> AWAITING_FUNDING -> CLOSE_SUBMITTED/CLOSE_PARTIAL -> COMPLETED
      -> ABORTED / RECOVERY_REQUIRED
```

关键约束：

- intent 在发单前原子写入状态文件；一个进程锁保护每个 cycle。
- 每个结算小时和阶段使用稳定 CLOID；模糊响应先查真实 `szi`/order status，绝不盲目重复开仓。
- 每轮最多新开一个 symbol，下一轮刷新保证金后才考虑第二个。
- 每次开仓前再次刷新主仓、hedge 仓、funding、费用、保证金与订单，再做最终决策。
- 部分 IOC 成交按实际 hedge `szi` 计覆盖与 funding，不把 requested size 当作 filled size，也不盲目补单。
- 主仓缩小会 reduce-only 缩小超额 hedge；主仓归零/反向、funding 方向反转、行情读取失败或出现未知订单时，tracked hedge 优先退出。
- `userFunding` 必须 exact coin 且属于目标 UTC 小时；账本延迟超过上限仍强制 reduce-only 平仓。
- close 响应不明确时，每次先核对真实剩余仓位，等待配置的恢复窗口后才以新 CLOID 重试。
- 下单瞬间重新读取本地 UTC 时钟，HTTP timeout 为 5 秒，action 默认 15 秒过期；open expiry 还会被结算前 cutoff 截短，避免慢请求跨过整点才开仓。
- executor 在开仓前显式设置配置的 leverage 与 cross/isolated 模式。

## 配置与运行

把 [.env.example](.env.example) 中需要的变量复制到仓库根目录 `.env`（或在 shell 中 `export`）。建议为这个进程和 hedge subaccount 单独批准一个 API wallet，避免与其他进程共享 signer nonce。

离线演示：

```bash
.venv/bin/python -m funding_fee_avoidance \
  --snapshot funding_fee_avoidance/examples/positions.json \
  --now 2026-07-10T12:58:30Z
```

真实账户只读报告：

```bash
.venv/bin/python -m funding_fee_avoidance --watch
```

完成 dry-run、地址与 exact coin 核对后，实盘需要双重确认：

```bash
export FUNDING_HEDGE_EXECUTE=true
.venv/bin/python -m funding_fee_avoidance --watch --execute
```

第一次实盘前至少确认输出中的：

- `primary_account_is_read_only: true`
- `order_route: hedge_account_only`
- primary/hedge 地址不同，ownership verified
- exact symbol 为主仓 API 返回值
- fee source 为 `official_hip3_fee_formula`
- margin、hedge ratio、最大 notional、滑点与净节省合理

## 验证

```bash
.venv/bin/python -m pytest -q funding_fee_avoidance/tests
.venv/bin/python -m py_compile funding_fee_avoidance/*.py
```

测试覆盖方向与费率符号、部分对冲、oracle funding notional、HIP-3 动态费用、主/对冲地址隔离、`vault_address` 路由、稳定 CLOID、部分成交、funding 入账确认、超时平仓、主仓变化、模糊 close 重试和 late-open 防护。

## 开源项目取舍

截至 2026-07-10，优先使用 stars 多、维护活跃、许可证清晰的底座：

| 项目 | Stars（核验时） | 用法 |
|---|---:|---|
| [Freqtrade](https://github.com/freqtrade/freqtrade) | 52,225 | 借鉴专用账户、运维和 fail-closed 原则；GPL 代码未复制 |
| [官方 Hyperliquid Python SDK](https://github.com/hyperliquid-dex/hyperliquid-python-sdk) | 1,743 | MIT；实际执行、CLOID、HIP-3 asset mapping、reduce-only |
| [funding-arb-bot](https://github.com/Gajesh2007/funding-arb-bot) | 21 | 只参考双腿失败恢复思路；无许可证且不支持本场景 |
| [keitaj/hyperliquid-bot](https://github.com/keitaj/hyperliquid-bot) | 3 | 只参考 HIP-3 错误分类和风险状态机 |

没有一个成熟仓库能直接完成“读取另一个主账户仓位 -> 同合约子账户临时 hedge -> 等 exact `userFunding` -> 平仓”。因此本项目直接依赖官方 SDK，并自己实现小型、可恢复、fail-closed 的状态机，而不是运行低-star 仓库的未审计策略代码。

## 风险

这不是无风险套利。对冲期间净方向敞口会下降，可能错过有利价格变化；还存在 funding 临近整点变化、滑点、IOC 部分成交、OI cap、oracle 偏离、API/网络、保证金、清算、API wallet nonce 与智能合约/协议风险。它只能降低特定 funding 支出，不能同时保留完整多头敞口、零额外保证金、零交易成本。

官方参考：[Funding](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding)、[Subaccounts](https://hyperliquid.gitbook.io/hyperliquid-docs/trading/sub-accounts)、[Perpetual API](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals)、[Nonces/API wallets](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/nonces-and-api-wallets)、[Exchange endpoint](https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint)。
