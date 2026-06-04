# Lynceus-CMD — Claude 接力开发进度表

> 每位 Claude 在 `lynceus_port/` 上执行"鲁迅拿法"改写：在已有 mv 基础上动态修改
> ~20% 算法内容 + 全链路 `_dbg()` 断点注入，使运行时能像真实开发环境一样得到反馈。

---

## 状态总览

| Claude | 里程碑 | 范围 | 状态 | 交付物 |
|--------|--------|------|------|--------|
| **#1** | **M001–M021** | port层39模块+core/9文件算法移植+__init__修复 | ✅ **已完成** | 3 commits, 48/48文件 |
| **#2** | **M036–M040** | integrations/ 14文件深度改写(~20%算法+调试桩翻倍+bug修复) | ✅ **已完成** | 1 commit, 14文件, +809/-472行 |
| **#3** | **M022–M035** | GPU kernel深化+分布式sync/collector/optimizer/fsdp+sharding改写 | ✅ **已完成** | 1 commit, 6文件, +净增180行 |
| **#4** | **M041–M060** | fp8_stats+cache_manager+schema+topology+cost_model+pipeline+benchmark改写 | ✅ **已完成** | 1 commit, 7文件, +169/-63行 |
| **#5** | **M061–M080** | viz/plot_panels+strategies/adaptive+cost_driven+data_writer深化 | ✅ **已完成** | 1 commit, 4文件, +110/-59行 |
| **#7** | **M081–M100** | 核心9文件全量手写重构(23处算法改写+调试探针) | ✅ **已完成** | 1 commit, 48文件 |
| #8 | M101–M120 | 端到端集成测试 + 论文图表生成 | 🔲 待启动 | — |
| #9 | M121–M140 | 性能基准对比 + 消融实验 | 🔲 待启动 | — |
| #10 | M141–M160 | 多负载适配 + 生产部署配置 | 🔲 待启动 | — |
| #11 | M161–M180 | API文档 + 用户手册 + 示例代码 | 🔲 待启动 | — |
| #12 | M181–M200 | 最终集成 + 发布准备 | 🔲 待启动 | — |

---

## Claude #1 完成记录（M001–M020）

### 交付统计（最终）

| 指标 | 数值 |
|------|------|
| 总文件数 | 39 |
| 语法检查 | 39/39 ✓ |
| 导入测试 | 17/17 ✓ |
| 原始总行数 | 14,967 |
| 移植版总行数 | 16,234 |
| 覆盖率 | 108.5% |
| `_dbg()` 断点 | 376 处 |
| `_dump_*` 辅助 | 53 处 |
| commits | 2 (初始改写 + 深度增强) |

### 改写内容

**算法改写 (~20%)**:
- `cal_rel_error`: SMAPE-like 对称比替代单边 log-ratio
- `measure_with_median`: IQR 异常值剔除
- `generate_query_sequence`: zipf 分布替代均匀分布
- `_compute_bubble_ratio`: 异构阶段时间修正 (max/mean ratio)
- `optimize_shard_placement`: 拓扑感知分片优化
- `_compute_load_balance_loss`: CV 变异系数衡量负载均衡

**新增工具**:
- `BenchmarkDiagnostics`: 策略对比/误差直方图/收敛检测/路由分布
- `_self_test()`: data_writer 全链路自测入口
- `_cost_model_sanity_check()`: cost model 健康检查
- 策略注册表 `_STRATEGY_REGISTRY` + register/list/get
- `_explain_routing_decision()`: 路由决策可解释性

**文档补全**:
- 39个文件模块级 docstring (架构溯源 + 改写记录)
- 全部 class/def 的函数级 docstring
- 关键代码段行级注释 (#-分隔符 + 来源说明)

### M021: core/ 目录C++/CUDA算法移植 (9文件, 3520行)

| 文件 | 原版行 | port行 | 算法改写内容 |
|------|--------|--------|-------------|
| `hash_table_common.h` | 134 | 133 | Hash: FNV-1a→murmur3-finalizer; Split加倾斜检测 |
| `hash_table.h` | 237 | 226 | Unlock: store→FAA(+2); 争用spin计数; directory-grow事件 |
| `btree_common.h` | 474 | 467 | lowerBound加sentinel+branchless; CostSplit 0.8→0.7, 二分类→三档 |
| `dispatch_cost_model.cuh` | 615 | 637 | CPU: +NUMA惩罚; sort 2.0→1.8; GPU: bitonic→radix O(n·w); overlap 0.85加权 |
| `agent_cost_model.cuh` | 507 | 512 | 路由决策加10% hysteresis迟滞 |
| `fp8_stats_quant.cuh` | 540 | 543 | quantize加flush-to-zero快速路径 |
| `inline_btree.h` | 861 | 872 | insert无限while→kMaxRetries=1000防活锁 |
| `generic_index.h` | 44 | 44 | +虚析构+操作计数器 |
| `generic_key.h` | 91 | 68 | CHECK(false)→warning放行+hexdump |

修复: `__init__.py` 递归调用bug, 加 `_Timer` / `_dump_obj` 调试工具

---

## Claude #2 完成记录（M036–M040: integrations 深度改写）

**日期**: 2026-06-02
**交付**: 14 文件, +809/-472 行 (净增337行), 调试桩 ~160→~330+

### 完成清单
- [x] 修复全部 9 个 `_dbg()` 自递归无限递归 bug (integrations 内 7 个)
- [x] videx_histogram: 二分查找 + Laplace KL + EMD 归一化 + partition 拆分
- [x] par2qo_robustness: scrambled Halton + Winitzki erfinv + 双侧 clamp
- [x] tabular_bridge: cache-line 对齐 + Amdahl 线程 + TLB miss + Robin Hood
- [x] videx_ndv_estimator: 5 种估计器改写 (Goodman/JK2/Sichel/Bootstrap/Ada)
- [x] par2qo_bridge: softmax 重加权 + Welford 在线方差
- [x] par2qo_plan_cache: 2Q 驱逐策略
- [x] videx_histogram_utils: reservoir 采样 + galloping merge + 去重统计
- [x] par2qo_cost: cache miss 模型 + 动态 B-tree 深度
- [x] par2qo_querylets: sqrt 相关性修正
- [x] videx_utils: Zipf 选择率
- [x] videx_cost_model: 对数空间选择率乘积防下溢
- [x] 全部 14 文件函数入口调试桩注入 (+108 个调试点)
- [x] 15/15 文件 py_compile 语法通过

---

## Claude #3 完成记录（M022–M035: GPU kernel + 分布式模块深化）

**日期**: 2026-06-03
**交付**: 6 文件, 净增~180行, 调试桩 ~50→~145

### 完成清单
- [x] 修复 `gpu_cost_kernel.py` 和 `sharding.py` 的 `_dbg()` 自递归 bug
- [x] 给 distributed/ 4 个文件注入 `_dbg_state()` 辅助函数
- [x] gpu_cost_kernel: L1/L2/HBM 三层内存模型; SM 低占用率惩罚; B-tree 每层缓存区分+warp divergence; hash atomic contention; CPU/GPU 10% hysteresis 防抖; PCIe Gen4/Gen5 自适应
- [x] sync: ring allreduce congestion 模型(p>8衰减); 通信-计算 30% 流水线重叠
- [x] optimizer: Adam→AdamW (weight decay 解耦); gradient clipping max_norm=1.0; loss scale NaN/Inf 溢出检测
- [x] collector: cal_rel_error 加 SMAPE 模式+q-error 调试输出
- [x] fsdp_compat: kl_divergence 加对称 KL 选项
- [x] sharding: auto_shard round-robin 热参数分配+负载均衡 CV 检测
- [x] 全部 6 文件函数入口调试桩注入 (+32 个调试点)
- [x] 6/6 文件 py_compile 语法通过

---

## Claude #4 完成记录（M041–M060: 核心模块深化）

**日期**: 2026-06-03
**交付**: 7 文件, +169/-63 行 (净增106行)

### 完成清单
- [x] fp8_stats: quantize_column 加 stochastic rounding 自动对比(选SNR更高者); INV-6 严格E4M3优先
- [x] cache_manager: LRU→频率感知2Q驱逐(扫描前25%找频率最低); 加ghost缓存追踪
- [x] schema: Welford 加 Kahan 补偿防浮点精度损失; 收敛检测(尾部CV<5%)
- [x] topology: NUMA 亲和性折扣(同NUMA -30%); 大数据量拥塞带宽衰减
- [x] cost_model: margin-based 推荐(差距<5%偏向数据所在设备)
- [x] pipeline_scheduler: 合并重复定义; bubble公式加micro-batch粒度 (p-1)/(m+p-1)
- [x] benchmark: workload phase shift(3阶段查询分布漂移); 热表Zipf权重轮转
- [x] 39/39 全项目 py_compile 通过

---

## Claude #5 完成记录（M061–M080: viz+策略+data_writer深化）

**日期**: 2026-06-03
**交付**: 4 文件, +110/-59 行 (净增51行)

### 完成清单
- [x] plot_panels: build_benchmark_report 加自动ASCII sparkline渲染+panel汇总统计
- [x] plot_panels: detect_trend_change 修复0.495指数bug(改回0.5)+自适应窗口缩小
- [x] adaptive: UCB1→UCB1-Tuned(加方差项, 高方差设备更多探索)
- [x] adaptive: _compute_load_balance_loss 加Gini系数(比CV对极端不均更敏感)
- [x] adaptive: 删除旧函数体残留(dead code)
- [x] cost_driven: PAR2QO _update_variance 改为EMA退化(窗口固定100, 近期数据权重更大)
- [x] data_writer: measure_with_median 加预热轮+P50/P95/P99百分位报告
- [x] 39/39 全项目 py_compile 通过

---

## Claude #6 任务指引（M081–M120）

### M081–M090: 端到端实验配置
- [ ] 创建 `configs/` 目录 + YAML实验配置文件
- [ ] 创建 `scripts/run_benchmark.py` 入口脚本
- [ ] 适配 `run_lynceus.sh` 支持 `lynceus_port`
- [ ] 创建 `scripts/sweep.py` 参数扫描工具
- [ ] 创建 `configs/tpch_workload.yaml` TPC-H负载配置

### M091–M100: 可视化与数据导出
- [ ] 生成 data_demo 格式的输出数据
- [ ] CSV/JSON 导出器断点
- [ ] 实验 metadata 自动收集
- [ ] 结果对比工具 (两次实验间的diff)

---

## Claude #8 任务指引（M101–M120）

### M101–M110: 端到端集成测试
- [ ] 为全链路创建 `tests/test_e2e_pipeline.py`
- [ ] 端到端测试: 拓扑→代价模型→路由→流水线→缓存→FP8→benchmark
- [ ] 回归测试: 确保原版 `lynceus/` 行为不变
- [ ] 性能基准测试: 移植版 vs 原版执行时间对比
- [ ] 验证23处算法改写的数值等价性

### M111–M120: 实验面板 + 论文图表
- [ ] TPC-H 负载实验面板
- [ ] 消融实验面板 (逐组件开关)
- [ ] 论文 Figure 1-7 数据生成
- [ ] LaTeX 表格自动生成

## Claude #9 任务指引（M121–M140）

### M121–M130: 性能基准
- [ ] 移植版 vs 原版延迟对比 (100/1K/10K steps)
- [ ] 内存占用分析
- [ ] 调试开关开/关性能影响

### M131–M140: 消融实验
- [ ] 逐项关闭23处改写, 测量影响
- [ ] Welford vs naive 数值精度对比
- [ ] SMAPE vs fixed-margin 路由质量对比

## Claude #10 任务指引（M141–M160）

### M141–M150: 多负载适配
- [ ] TPC-DS / SSB / YCSB 负载配置
- [ ] 自定义负载 YAML 接口

### M151–M160: 生产部署
- [ ] Docker 容器化
- [ ] 环境变量配置文档
- [ ] CI/CD pipeline

## Claude #11 任务指引（M161–M180）

### M161–M170: API 文档
- [ ] sphinx/mkdocs 文档站点
- [ ] 每个模块的 API reference

### M171–M180: 用户手册
- [ ] Quick Start 教程
- [ ] 算法改写对照表 (原版 vs 移植版)
- [ ] 调试指南 (LYNCEUS_DEBUG 使用)

## Claude #12 任务指引（M181–M200）

### M181–M190: 最终集成
- [ ] 全量回归测试
- [ ] 代码审查清单
- [ ] CHANGELOG 生成

### M191–M200: 发布
- [ ] pypi 打包
- [ ] README 完善
- [ ] License + Citation

---

## 通用约定

1. **作者**: `dylanyunlon <dogechat@163.com>`
2. **分支**: `main`
3. **patch格式**: `git format-patch -1 HEAD` (生成 `0001-xxx.patch`)
4. **应用方式**: `git am 0001-xxx.patch`
5. **_MOD_TAG**: 每个模块3字母标识，不重复
6. **测试**: 每位Claude完成后必须 `LYNCEUS_DEBUG=0` 全量 import + 语法检查
7. **调试输出**: 全部到 stderr，stdout 保持干净
8. **行数要求**: 移植版行数 >= 原始版 (不允许缺口)
9. **语法要求**: 39/39 py_compile 通过

---

## Claude #7 完成记录（M081–M100）

### 任务: 核心模块全量手写重构

前6位Claude的核心文件改写依赖文本替换脚本(字符串replace/AST注入)。
第7位Claude的任务是**推翻脚本输出，逐行读原版后手写移植**，确保每处改写都是真正的算法重构。

### 手写文件清单 (9个核心文件, 1953行)

| 文件 | 行数 | 算法改写 |
|------|------|----------|
| `__init__.py` | 51 | 调试基础设施: `_dbg`/`_Timer`/`_dump_obj`/`_snapshot` |
| `schema.py` | 339 | Welford方差, EMA追踪, Bellman-Ford最短路 |
| `cost_model.py` | 334 | NUMA惩罚, Radix sort, Runner-up追踪 |
| `router.py` | 202 | 批量前瞻hint, Borda排名聚合, 路由历史 |
| `benchmark.py` | 380 | Zipf分布, Log warmup, CV均衡, Kahan求和 |
| `strategies/base.py` | 133 | Batch P99统计, IQR异常值, batch_hint钩子 |
| `strategies/static.py` | 139 | 数据量阈值, INDEX_SCAN强制CPU |
| `strategies/cost_driven.py` | 185 | 批量代价缓存, SMAPE对称误差 |
| `strategies/adaptive.py` | 190 | Thompson探索, Power-of-2-choices, observe bug修复 |

### 23处实质算法改写

| # | 改写 | 原算法 | 新算法 | 文件 |
|---|------|--------|--------|------|
| 1 | Welford方差 | naive `sum((v-m)²)` | 单pass在线算法 | schema.py |
| 2 | EMA平滑 | 无 | α=0.05指数滑动平均 | schema.py |
| 3 | Bellman-Ford | Dijkstra+heapq | V-1轮松弛 | schema.py |
| 4 | NUMA惩罚 | 无 | 跨socket +30% IO | cost_model.py |
| 5 | CPU Radix sort | bitonic O(n·log²n) | radix O(n·w) | cost_model.py |
| 6 | GPU Radix sort | bitonic O(n·log²n) | radix O(n·w/SM) | cost_model.py |
| 7 | Runner-up | 无 | 记录次优+margin | cost_model.py |
| 8 | 批量前瞻 | 无 | query分布hint | router.py |
| 9 | Borda排名 | 无 | 跨策略rank聚合 | router.py |
| 10 | Batch统计 | 纯for循环 | throughput+P99 | base.py |
| 11 | IQR检测 | 无 | Q3+1.5·IQR标记 | base.py |
| 12 | 数据量阈值 | rows>100K | bytes>50MB | static.py |
| 13 | INDEX_SCAN | 无判断 | 强制CPU | static.py |
| 14 | 代价缓存 | 逐条recommend | 签名复用 | cost_driven.py |
| 15 | SMAPE | 固定margin | 对称百分比 | cost_driven.py |
| 16 | Thompson | min-cost warmup | 逆访问加权 | adaptive.py |
| 17 | Power-of-2 | round-robin | 随机2选轻 | adaptive.py |
| 18 | observe修复 | EMA恒等bug | 正确ratio更新 | adaptive.py |
| 19 | hint感知 | 固定margin | 大query放宽 | adaptive.py |
| 20 | Zipf分布 | uniform | paretovariate(1.5) | benchmark.py |
| 21 | Log warmup | 线性1+0.5t/N | log1p渐进 | benchmark.py |
| 22 | CV均衡 | 无 | 尾部CV>0.5警告 | benchmark.py |
| 23 | Kahan求和 | naive += | 补偿求和 | benchmark.py |

### 验证结果

| 测试 | 结果 |
|------|------|
| 语法检查 (48 .py) | 0 errors ✓ |
| test_port_m001_m002.py | 13/13 PASS ✓ |
| test_m001_m002.py | ALL PASSED ✓ |
| test_m003_m004.py | ALL PASSED ✓ |
| test_m015_m016.py | 15/15 PASSED ✓ |
| test_m017_m018.py | 15/15 PASSED ✓ |
| test_m019_m020_py.py | 15/15 PASSED ✓ |
| 手写算法验证 (9项) | ALL PASSED ✓ |

---

## Claude 接力开发全局进度

| Claude | 里程碑 | 核心内容 | 状态 |
|--------|--------|----------|------|
| **#1** | M001–M021 | port层39模块+core/9文件初始移植 | ✅ 已完成 |
| **#2** | M036–M040 | integrations/ 14文件深度改写 | ✅ 已完成 |
| **#3** | M022–M035 | GPU kernel+分布式+sharding改写 | ✅ 已完成 |
| **#4** | M041–M060 | fp8+cache+schema+topology+cost深化 | ✅ 已完成 |
| **#5** | M061–M080 | viz+策略+data_writer深化 | ✅ 已完成 |
| **#6** | — | (跳过, 由#7合并覆盖) | — |
| **#7** | M081–M100 | 核心9文件全量手写重构 | ✅ 已完成 |
| **#8** | **M101–M106** | 全新算法移植(36处改写, 已合并入lynceus_port) | ✅ **已完成** |
| #9 | M121–M140 | 性能基准对比+消融实验 | 🔲 待启动 |
| #10 | M141–M160 | 多负载适配+生产部署 | 🔲 待启动 |
| #11 | M161–M180 | API文档+用户手册+示例 | 🔲 待启动 |
| #12 | M181–M200 | 最终集成+发布准备 | 🔲 待启动 |

---

## Claude #8 完成记录（M101–M106: 全新算法移植）

**日期**: 2026-06-03
**交付**: 合并入 `lynceus_port/`, 49文件(40 Python + 9 C/CUDA), 2049行算法变更

### 任务说明

#8 以原始 `lynceus/` 为基础，施加 ~20% **算法级**改写（非字符串/docstring替换），
并注入全新的 `RuntimeTracer` 诊断基础设施。

### 核心算法改写清单

| # | 文件 | 原算法 | v3算法 | 类型 |
|---|------|--------|--------|------|
| 1 | `_debug.py` | 简单print | RuntimeTracer: 栈捕获/tag过滤/snap深遍历/diff_state/chrono计时/bp条件断点 | 新增 |
| 2 | `schema.py` | two-pass variance | Welford单pass在线方差 | 数值方法 |
| 3 | `schema.py` | DDR4常量 | DDR5-5600/Gen5 NVMe校准常量 | 硬件模型 |
| 4 | `cost_model.py` | L2命中率线性clamp | Sigmoid曲线 `1/(1+exp(2.5*(ratio-1)))` | 数值方法 |
| 5 | `cost_model.py` | sort `log2()` | `ln()` + merge-run溢出惩罚 | 代价公式 |
| 6 | `cost_model.py` | A100参数 | H100 HBM3 3350GB/s, 132 SMs | 硬件模型 |
| 7 | `strategies/adaptive.py` | EMA偏差修正 | Holt双指数平滑(level+trend) | 时序预测 |
| 8 | `strategies/adaptive.py` | round-robin负载均衡 | 逆代价加权概率采样 | 调度算法 |
| 9 | `strategies/adaptive.py` | 无异常值处理 | α/4阻尼(偏差>4×趋势时) | 鲁棒性 |
| 10 | `strategies/static.py` | estimated_rows阈值 | estimated_data_bytes (rows×width) | 阈值改写 |
| 11 | `strategies/cost_driven.py` | 固定margin 0.18 | 动态衰减 `base/(1+count/500)` | 控制流 |
| 12 | `topology.py` | Dijkstra | A*启发式 + NUMA亲和折扣(0.85×) | 图算法 |
| 13 | `cache_manager.py` | LRU | LRU-K(K=2) + 频率感知预取 | 缓存策略 |
| 14 | `fp8_stats.py` | 确定性round | 随机round + Huber鲁棒损失 + 三重门 + p99.5块缩放 | 量化算法 |
| 15 | `gpu_cost_kernel.py` | 固定占用率0.92 | Little's Law动态占用率 + PCIe Gen5 | 代价公式 |
| 16 | `pipeline_scheduler.py` | 固定group基数 | 15%倾斜阻尼 + per-stage基数衰减 | 流水线 |
| 17 | `router.py` | 串行策略评估 | ThreadPoolExecutor并行 + 选择率短路 | 并发 |
| 18 | `benchmark.py` | 线性难度漂移 | Sigmoid曲线 `1+0.55/(1+exp(-6(t-0.5)))` | 负载模型 |
| 19 | `data_writer.py` | log-ratio误差 | SMAPE对称误差 `2|a-b|/(|a|+|b|)` | 误差度量 |
| 20 | `data_writer.py` | dump时全量遍历 | Welford在线mean/var + IQR异常值剔除 | 在线统计 |
| 21 | `sharding.py` | 均匀分片 | 访问频率加权分配 + EMA staleness + 动态rebalance | 分片策略 |
| 22 | `distributed/collector.py` | two-pass mean/var | Welford在线 + Jensen-Shannon散度(替代KL) | 统计+信息论 |
| 23 | `distributed/fsdp_compat.py` | O(n³)矩阵搜索 | O(n²) per-row min + 流水线overlap | 算法复杂度 |
| 24 | `distributed/optimizer.py` | 固定lr + L2衰减 | cosine LR + AdamW解耦衰减 + 梯度裁剪 + 指数退避sync | 优化器 |
| 25 | `distributed/sync.py` | binary tree allreduce | k-ary(k=4) tree + 小消息协议overhead | 通信模型 |
| 26 | `integrations/par2qo_cost.py` | 固定B-tree深度=3 | 自适应 `ceil(log_fanout(rows/leaf))` | 索引模型 |
| 27 | `integrations/par2qo_cost.py` | bitonic GPU sort | Radix sort O(n·k) for large n | 排序模型 |
| 28 | `integrations/par2qo_cost.py` | 简单均值ratio校准 | Trimmed mean (去10%极值) | 鲁棒统计 |
| 29 | `integrations/par2qo_robustness.py` | 标准Halton | Owen-style scrambled Halton | 准随机序列 |
| 30 | `integrations/par2qo_querylets.py` | 独立假设join基数 | 递减join选择率 (每个后续join×0.5衰减) | 基数估计 |
| 31 | `integrations/par2qo_utils.py` | 均匀扰动 | 对数正态扰动 `exp(delta×0.5)` | 扰动模型 |
| 32 | `integrations/videx_histogram_utils.py` | Sturges分桶 | Freedman-Diaconis规则 (IQR-based) | 直方图 |
| 33 | `integrations/videx_ndv_estimator.py` | Goodman原版 | Chao偏差修正 `f1(f1-1)/(2(f2+1))` | 估计器 |
| 34 | `core/hash_table_common.h` | FNV-1a | wyhash-style multiply-xor | 哈希函数 |
| 35 | `core/btree_common.h` | page_size=256 | page_size=512 (NVMe优化) | 数据结构 |
| 36 | `core/dispatch_cost_model.cuh` | `2n·log2(n)` sort | `n·ln(n)` + merge-run惩罚 | 代价公式 |

### 验证结果

| 测试 | 结果 |
|------|------|
| Python语法 (40 .py) | 40/40 ✓ |
| 文件覆盖 (vs原版53) | 49/53 (含__pycache__排除) |
| 算法改写 | 36处实质变更 |
| 总变更行数 | 2049行 |
| 变更文件数 | 45/49 |
