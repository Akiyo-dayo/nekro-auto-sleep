# NekroAgent 自动睡眠插件

为每个会话提供独立的拟人化睡眠周期，无需修改 NekroAgent 核心代码。
同时兼容 Akiyo 版 (`Akiyo-dayo/NekroAgent_ByAkiyo`) 与原版上游 (`KroMiose/nekro-agent`)。

## 功能

- **定时入睡与随机起床**：默认每晚 23:00 静默入睡，次日 06:45–08:30 之间随机自然醒
- **两次叫醒协议**：睡眠中用户有效呼叫→固定提示（不经 LLM）→同一用户再次呼叫→正常唤醒进 LLM
- **唤醒后上下文连贯**：整夜消息照常入库，叫醒提示进历史，被叫醒后每一轮都能拿到
  「几点就寝、被谁叫醒、醒了多久、夜里的消息是睡着时收到的」
- **重启自愈**：启动时对账所有频道，补结算错过的自然醒、补入睡错过的夜晚
- **主动睡下**：Bot 可调用 `resume_sleep` 工具重新入睡
- **空闲自动睡回**：被叫醒后无新互动自动静默睡回（默认 10 分钟）
- **睡眠质量统计**：基于连续积分模型计算质量百分比
- **自然醒播报**：默认每次自然醒都播报，可切换为「仅当夜里被打扰过」或「从不」

## 安装

把 **`nekro_auto_sleep/`** 这个目录（不是仓库根目录）复制到 NekroAgent 的 `plugins/external/` 下：

```
plugins/external/nekro_auto_sleep/
├── __init__.py
├── models.py
├── schedule.py
├── quality.py
├── engine.py
├── persistence.py
└── runtime.py
```

重启 NekroAgent 后在插件管理页面启用「自动睡眠」插件。

## 配置

所有配置项均支持 WebUI 中文界面。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| 启用自动睡眠 | `true` | 总开关 |
| 时区 | `Asia/Shanghai` | IANA 时区名称 |
| 入睡时间 | `23:00` | 每日自动入睡时间（HH:MM） |
| 起床时间范围（起始 / 结束） | `06:45` / `08:30` | 随机起床区间 |
| 起床随机粒度 | `1` 分钟 | 候选起床点间隔 |
| 接近起床判定比例 | `0.15` | 末段比例，提示语切换为「还没起床」 |
| 叫醒确认窗口 | `180` 秒 | 二次叫醒的时间窗口 |
| 历史记录模式 | `preserve` | `preserve`=夜间消息与叫醒提示都入库 / `strict`=完全拦截不留痕 |
| 呼叫关键词 | `醒醒,起床,在吗` | 触发叫醒的关键词（逗号分隔） |
| 默认人格名 | `Bot` | 取不到预设名时的回退名 |
| 提前叫醒空闲超时 | `10` 分钟 | 被叫醒后无互动的睡回时间 |
| **起床播报策略** | `always` | `always` / `if_disturbed` / `never` |
| 离线补发宽限期 | `120` 分钟 | 重启后补发自然醒播报的最大延迟，超时静默结算 |
| **启动装载活跃天数** | `14` | 启动时装载最近多少天有消息的频道 |
| 维护循环间隔 | `15` 秒 | 后台检查间隔 |
| 定时任务等待超时 | `900` 秒 | 预留给 `NIGHT_TIMER_POLICY`，当前未使用 |
| 睡眠质量下限 / 上限 | `60` / `120` | 质量百分比夹取区间 |
| 质量稳定扰动幅度 | `4.0` | 每次质量的随机扰动范围 |

## 工作流程

### 正常睡眠

```
23:00 → 静默入睡（不发消息）
       ↓ 夜间普通消息：入库但不触发 LLM（strict 模式下完全不入库）
06:45-08:30 → 随机时间自然醒 → 【{persona}已起床：昨日睡眠质量 N%，睡眠时长 X】
```

### 叫醒流程

```
用户有效呼叫 → 【{persona}已经睡了 要叫醒{persona}吗？】(不经 LLM，preserve 模式下进历史)
       ↓ 同用户 3 分钟内再次呼叫
       → AWAKE_EARLY，本轮 FORCE_TRIGGER 进 LLM
       → 每一轮都注入：几点就寝、被谁叫醒、醒了多久、
         「就寝之后的消息你是在睡着时收到的，刚醒来才看见」
       ↓ 10 分钟无新互动 → 静默睡回
       ↓ 或 Bot 调用 resume_sleep → 【{persona}已睡下】
```

接近起床时提示语改为 **【{persona}还没起床 要叫醒{persona}吗？】**。

### 重启

启动时枚举「已有睡眠状态的频道」∪「近 N 天有消息的频道」，逐个对账：

- 睡着且已过计划起床点 → 立刻结算；在宽限期内补发播报，超时则静默结算
- 醒着但当前正处于夜间窗口 → 入睡，且**睡眠段回填到真实就寝点**（不是启动时刻）
- 之后维护循环覆盖所有装载过的频道，不再依赖「有人先说话」

## 状态机

```
AWAKE ──(到达入睡点)──→ ASLEEP
                          │
              ┌───────────┤
              │           │
         (二次叫醒)   (计划起床)
              │           │
              ▼           ▼
        AWAKE_EARLY ──→ AWAKE
              │
     (空闲/resume_sleep)
              │
              ▼
           ASLEEP
```

## 开发

```bash
python -m pytest tests -q
```

`tests/hoststub.py` 伪造 `nekro_agent.*`，让接线层（`__init__.py`）可以脱离宿主被测试。
`tests/test_host_contract.py` 用 `ast` 解析真实 NekroAgent 源码校验这些假设没有漂移
（`db_chat_channel` 仍是普通 property、`MsgSignal` 取值不变、`BLOCK_ALL` 仍早于消息落库等）。
它默认找同级目录 `NekroAgent_ByAkiyo` / `nekro-agent`，也可以显式指定；两个版本都应通过：

```bash
NEKRO_AGENT_SRC=/path/to/nekro-agent python -m pytest tests/test_host_contract.py -q
```

## 已知未完项

- **定时任务夜间策略**：当前夜间定时提醒会入库但不触发 Agent。
  计划中的 `NIGHT_TIMER_POLICY`（`run` / `defer` / `block`）尚未实现，
  `TIMER_AGENT_WAIT_TIMEOUT_SECONDS` 与 `runtime.py` 里的租约系统在它落地前是预留代码。
- **睡眠质量模型重标定**：分数目前集中在 88–103，`QUALITY_MAX` 实际够不到，
  `>100%` 还来自随机扰动而非「睡得比目标更久」。
- **唤醒协议**：确认唤醒仍要求第二次「有效呼叫」，用户回「要」暂时不算数；
  否定意图、提示冷却与每夜提示上限尚未实现。

## 技术要点

- 不修改任何 NekroAgent 核心文件
- 使用 `PluginStore` 持久化，不建表；启动枚举直接查 `DBPluginData` / `DBChatMessage`
- 每个 `chat_key` 独立状态、独立异步锁
- 领域层（`models` / `schedule` / `engine` / `quality`）不导入宿主单例，可独立测试
- 通过运行时 `getattr` 能力探测兼容两个版本，不按版本号分支

## 许可

与 NekroAgent 保持一致。
