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
| #4 | M041–M060 | Mixed-precision optimizer + Auto-sharding算法改写 | 🔲 待启动 | — |
| #5 | M061–M080 | FSDP compat + viz重构 + cache/schema二次改写 | 🔲 待启动 | — |
| #6 | M081–M120 | 端到端实验配置 + 集成测试 + 论文图表生成 | 🔲 待启动 | — |

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

## Claude #4 任务指引（M041–M060）

**注意**: 所有 `_dbg()` 自递归 bug 已在 #2 和 #3 中全部修复

### M041–M045: Mixed-precision optimizer 深化
- [ ] FP8 stats 量化链路断点深化 (E4M3/E5M2每步)
- [ ] loss scaling自动调整断点
- [ ] 算法改写: 量化舍入模式 (nearest→stochastic)

### M046–M050: Auto-sharding 深化
- [ ] shard分配决策的完整推理链断点
- [ ] 负载均衡指标追踪 (per-epoch)
- [ ] NUMA感知通信开销估算改写
- [ ] 分片迁移策略改写

### M051–M060: cache_manager + schema 二次改写
- [ ] cache eviction策略改写 (LRU→ARC或LIRS)
- [ ] schema统计信息更新策略改写
- [ ] Welford在线均值/方差的数值稳定性改进

---

## Claude #4 任务指引（M061–M080）

### M061–M070: FSDP + NCCL后端
- [ ] FSDP compatibility layer全链路断点
- [ ] NCCL-backend all_reduce collector深化
- [ ] backward hook注册和梯度桶管理
- [ ] mixed-precision与分布式的交互断点

### M071–M080: viz + plot_panels 重构
- [ ] plot_panels.py 全面断点 (当前仅8个)
- [ ] 增加 ASCII art 输出模式 (无 matplotlib 依赖)
- [ ] 面板间数据流断点
- [ ] 图表配置驱动化改写

---

## Claude #5 任务指引（M081–M100）

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

## Claude #6 任务指引（M101–M120）

### M101–M110: 集成测试
- [ ] 为每对里程碑创建 `tests/test_port_mXXX_mXXX.py`
- [ ] 端到端测试: 拓扑→代价模型→路由→流水线→缓存→FP8→benchmark
- [ ] 回归测试: 确保原版 `lynceus/` 行为不变
- [ ] 性能基准测试: 移植版 vs 原版执行时间对比

### M111–M120: 实验面板 + 论文图表
- [ ] TPC-H 负载实验面板
- [ ] 消融实验面板 (逐组件开关)
- [ ] 论文 Figure 1-7 数据生成
- [ ] LaTeX 表格自动生成

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
