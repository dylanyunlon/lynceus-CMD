# Lynceus-CMD 代码审计报告 — 生产级缺陷批判

> 审计者立场：以《计算机程序设计艺术》作者的标准，对 M015–M020（Claude #8–#10）
> 的全部修改做端到端运行验证 + 逐行批判。审计在真实环境跑通了核心调用链
> （拓扑→成本模型→路由→流水线→缓存→FP8→benchmark），未新增任何测试代码。

审计结论：**6 个缺陷，其中 2 个致命（数据正确性），1 个概念性错误，3 个系统/健壮性问题。**
全部源自 M015–M020，且每一个都会在生产环境产生错误数据或错误路由决策。

---

## 缺陷清单（按严重度排序）

### 【致命-1】pipeline_scheduler：transfer 成本从 critical path 凭空消失（246× 假加速）

**位置**：`lynceus/pipeline_scheduler.py::_pipelined_cost`，第 324 行
`compute = a.cost.total_us - a.cost.transfer_cost_us`

**复现**：JOIN 查询，`serial=15062µs`，`pipelined=61µs`，`speedup=246×`，但 `devices_used=['gpu0']`。
单设备不可能有流水线加速。

**根因**：SCAN stage 的 `total=15011µs`，其中 `transfer=15001µs`（cpu0→gpu0 的真实 PCIe
搬运）。代码把 transfer 从 compute 里减掉，留下 10.45µs，然后将 15001µs 当作"可与计算
完全重叠、可隐藏"。但 `transfer_us` 累加（第 313 行）只统计**相邻 stage 间设备切换**的
transfer；SCAN 是第一个 stage，其 transfer 来自 `data_location → 首设备`，**没有前驱**，
既不进 `transfer_us`，又被从 `compute` 减掉——**15001µs 净蒸发**。

**用户视角批判**：用户会看到"15ms 查询被流水线优化到 61µs"，据此把查询路由到 GPU。
实际 GPU 要先付 15ms PCIe，性能预测偏差 246×，容量规划全错。
**系统视角批判**：`serial_cost_us` 含 transfer，`pipelined_cost_us` 去掉 transfer，
两者口径不一致，其比值 `speedup` 在数学上无意义。这是把两套不可比的度量相除。

---

### 【概念错误-2】pipeline_scheduler：对单查询套 Megatron fill-drain，本质错误

**位置**：`_pipelined_cost` 整体，及 `PipelineSchedule.speedup`

**根因**：Megatron 流水线的加速来自 **m 个 microbatch 流过 p 个 stage**，bubble 比例
= (p−1)/(m+p−1)（Narayanan 2021，已查证）。单个查询的 stage 之间是**严格数据依赖**
（SCAN→FILTER→JOIN→AGG→SORT，后者必须等前者完成），等价于 m=1，bubble=(p−1)/p，
**几乎无加速**。我写的 `hidden = min(non_bottleneck, bottleneck) × (warmup/runs)` 是
无理论依据的拍脑袋公式，不对应任何真实排队论或大厂模型。

**正确做法**：流水线并行性只在**多个独立查询**同时在流水线中流动时才出现。应改为
批量调度模型：m 个查询、p 个 stage（设备），用 Megatron 真实 bubble 公式
T_pipe = T_serial × (m + p − 1) / (m × p) 计算吞吐。单查询的 latency 必须等于
critical path（含 transfer），speedup 恒为 1。

---

### 【致命-3】cache_manager：rstrip 误用导致逻辑表身份塌缩（命中率造假）

**位置**：`lynceus/cache_manager.py::required_blocks`，第 202 行
`table = query.query_id.split("::", 1)[0].rstrip("0123456789_") or "t"`

**根因**：`rstrip(charset)` 是"按字符集剥尾部字符"，不是"剥后缀"。
- benchmark 真实生成的 `q_00000..q_01999`，全部 rstrip 成 `"q"` → **2000 个不同查询
  塌缩成同一张逻辑表**，缓存"命中率"被人为推到接近 100%。
- `"table2name"`→`"table2name"`（中间 2 不剥），`"name2"`→`"name"`：同表不同查询
  可能落到不同 table key，块共享反而失效。

**用户视角批判**：benchmark 产出的 cache hit rate 是假数据，论文图表不可信。
**系统视角批判**：用"字符串形状"猜测"逻辑表身份"是根本性抽象错误。表身份必须是
`QueryDescriptor` 的显式字段，不能从 query_id 反推。

---

### 【系统-4】cost_model 拓扑：cpu1→GPU 不可达，砍掉一半 NUMA 内存的加速

**位置**：`lynceus/schema.py::HardwareTopology.get_transfer_cost`（仅查直接边）
+ `create_default_topology`（cpu1 只连 cpu0，不连任何 GPU）

**复现**：`get_transfer_cost('cpu1','gpu0', 1e6)` → `inf`。

**根因**：`get_transfer_cost` 第 318–322 行只查**单跳直接边**，无多跳最短路。真实 NCCL
`ncclTopoComputePaths` 做的是全图最短路。当前实现下，数据若驻留在 cpu1（双路服务器
第二 NUMA 节点，完全正常），到 GPU 的成本是 inf → GPU 成本 inf → 路由器永不把 cpu1
数据送 GPU。

**用户视角批判**：双路机器一半内存上的数据被判 GPU 不可达，砍掉 50% 加速机会。
**系统视角批判**：拓扑成本模型必须支持多跳路径（cpu1→cpu0→gpu0），否则只能描述
星形拓扑，违背"异构拓扑感知"这一项目立身之本。

---

### 【健壮性-5】fp8_stats：max_rel 漏掉 orig==0 的污染 + NaN 静默吞没

**位置**：`lynceus/fp8_stats.py::measure_error`，第 156 行 `if abs(o) > 0.0`
（C++ 版 `core/fp8_stats_quant.cuh::measure_error` 同样问题）

**根因**：
- (a) `max_rel` 仅在 `abs(o)>0` 时更新。原值为 0 但量化后变非零（0 被污染成 ±subnormal×scale）
  的误差完全不计。统计列（直方图大量 0 桶）是真实场景，污染会被判为"零误差完美"。
- (b) 若 `orig` 含 NaN，sig/noise/mrel 全变 NaN，`acceptable` 因 NaN 比较返回 False，
  **静默拒绝**，用户不知道输入有 NaN。

**用户视角批判**：稀疏统计列被 FP8 破坏（0→非零）却显示完美，错误采用 FP8。
**系统视角批判**：误差度量必须对 orig==0 用绝对误差兜底，且对 NaN/Inf 输入显式报错或
标记，而非静默吞没。

---

### 【健壮性-6】pipeline_scheduler：assign_stages 把中间结果 device 当 data_location，
但下一 stage 的 transfer 已被【致命-1】抹掉，二者叠加掩盖了真实搬运

**位置**：`assign_stages` 第 280–289 行 + `_pipelined_cost`

**说明**：`assign_stages` 正确地把上一 stage 落点作为下一 stage 的 `data_location`
（中间结果就地驻留），这部分逻辑是对的。但 `_pipelined_cost` 把每个 stage 的 transfer
从 compute 减掉后只在"设备切换"时部分加回，导致**首 stage 的输入搬运**和**stage 间
中间结果搬运**口径不一。修复【致命-1】时必须统一：所有 transfer 都留在 critical path。

---

## 修复原则（生产级，融合大厂实践）

1. **transfer 永不消失**：单查询 latency = Σ(stage compute) + Σ(stage transfer)，
   即完整 critical path。这是数据依赖链的物理下界。
2. **流水线加速只对批量**：新增 `schedule_pipeline(queries)`，用 Megatron 真实公式
   `(m+p−1)/(m·p)` 计算 m 个查询过 p 个设备的吞吐；单查询 speedup 恒为 1。
3. **逻辑表身份显式化**：`QueryDescriptor` 增加 `table_name` 字段；cache_manager 用它，
   不再 rstrip 猜测。向后兼容（默认值）。
4. **拓扑多跳最短路**：`get_transfer_cost` 改 Dijkstra/Floyd 最短路（NCCL 风格），
   cpu1→cpu0→gpu0 可达。
5. **误差度量兜底**：orig==0 用绝对误差项；输入含 NaN/Inf 显式置位标记。

每条修复都将逐一验证不引入新 bug（用户视角 + 系统视角双重检查）。
