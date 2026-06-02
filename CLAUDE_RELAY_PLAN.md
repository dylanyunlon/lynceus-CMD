# Lynceus-CMD — Claude 接力开发进度表

> 每位 Claude 在 `lynceus_port/` 上执行"鲁迅拿法"改写：在已有 mv 基础上动态修改
> ~20% 算法内容 + 全链路 `_dbg()` 断点注入，使运行时能像真实开发环境一样得到反馈。

---

## 状态总览

| Claude | 里程碑 | 范围 | 状态 | 交付物 |
|--------|--------|------|------|--------|
| **#1** | **M001–M020** | port层全量改写+深度增强（39模块） | ✅ **已完成** | 2 commits, 覆盖率108.5% |
| #2 | M021–M040 | GPU kernel深化 + 分布式sync/collector断点 | 🔲 待启动 | — |
| #3 | M041–M060 | Mixed-precision optimizer + Auto-sharding算法改写 | 🔲 待启动 | — |
| #4 | M061–M080 | FSDP compat + integrations层二次改写 | 🔲 待启动 | — |
| #5 | M081–M100 | 端到端实验配置 + 可视化图表 + configs | 🔲 待启动 | — |
| #6 | M101–M120 | 集成测试 + 实验面板 + 论文图表生成 | 🔲 待启动 | — |

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

---

## Claude #2 任务指引（M021–M040）

### M021–M025: GPU kernel cost estimation 深化
**目标文件**: `lynceus_port/gpu_cost_kernel.py` (594行, 已有14个_dbg)

待做：
- [ ] `GPUKernelEstimator.estimate_seq_scan`: 深化内存层级模型 (L1/L2/HBM分层)
- [ ] `estimate_btree_lookup`: BTree遍历代价每层断点
- [ ] `estimate_hash_probe`: hash碰撞链长度+探测代价断点
- [ ] `estimate_hash_build`: 建表过程bucket分配断点
- [ ] `compare_cpu_vs_gpu`: 对比决策推理链断点
- [ ] `estimate_sort`: radix-sort pass级断点
- [ ] 算法改写 ~20%: SM占用率模型或内存带宽计算公式改写

### M026–M030: 分布式模块深化
**目标文件**:
- `distributed/sync.py` (392行, 7个_dbg)
- `distributed/collector.py` (631行, 7个_dbg)

待做：
- [ ] sync: push/pull通信原语每步断点, 参数聚合过程
- [ ] collector: all_reduce收集器断点, ring拓扑模拟
- [ ] 网络延迟模拟断点
- [ ] 算法改写: gradient compression or communication overlap

### M031–M035: 分布式优化器 + FSDP深化
**目标文件**:
- `distributed/optimizer.py` (349行, 6个_dbg)
- `distributed/fsdp_compat.py` (615行, 10个_dbg)

待做：
- [ ] optimizer: 融合优化器步骤断点 (loss scale/grad clip/param update)
- [ ] fsdp: shard参数管理断点, all-gather/reduce-scatter模拟
- [ ] 收敛检测断点深化
- [ ] 算法改写: Adam→AdamW或内循环等价数学变换

### M036–M040: integrations 二次改写
**目标文件**: `integrations/` 下的 par2qo_* 和 videx_* 文件

待做：
- [ ] 各 bridge 文件的关键路径断点从 ~5 提升到 ~15
- [ ] par2qo_robustness: 鲁棒性检测算法改写
- [ ] videx_histogram: 直方图合并算法改写
- [ ] tabular_bridge: 表数据索引策略改写

---

## Claude #3 任务指引（M041–M060）

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
