# Lynceus-CMD — Claude 开发进度追踪

> 每位 Claude 完成一批里程碑后在此记录，后续 Claude 接力推进。
> 原始里程碑定义见 `DEVELOPMENT_PLAN.md`。

---

## 基础建设阶段（Claude #1–#10，已完成 M001–M020）

| Claude # | 里程碑 | 状态 | 交付物 |
|----------|--------|------|--------|
| #1 | M001–M002 | ✅ 完成 | `schema.py`, `cost_model.py`, `topology.py` |
| #2 | M003–M004 | ✅ 完成 | `router.py`, `strategies/` |
| #3 | M005–M006 | ✅ 完成 | `benchmark.py`, `data_writer.py` |
| #4 | M007–M008 | ✅ 完成 | `par2qo_bridge.py` (薄桥接) |
| #5 | M009–M010 | ✅ 完成 | `videx_bridge.py` (薄桥接) |
| #6 | M011–M012 | ✅ 完成 | `tabular_bridge.py` (薄桥接) |
| #7 | M013–M014 | ✅ 完成 | `topology.py` 增强 |
| #8 | M015–M016 | ✅ 完成 | `pipeline_scheduler.py` |
| #9 | M017–M018 | ✅ 完成 | `cache_manager.py` |
| #10 | M019–M020 | ✅ 完成 | `fp8_stats.py` |

> ⚠️ 审计发现：#4–#6 的桥接文件内容过少，upstream 源码利用率不足 10%。
> 后续 Claude 必须深度移植 upstream，每文件每行都用上。

---

## 深度移植阶段（从 upstream 大量借鉴代码，改造 ~20%）

| 序号 | 里程碑范围 | 状态 | 核心任务 | 交付物 |
|------|-----------|------|----------|--------|
| **第一位 Claude** | **M007–M012 深度移植** | **✅ 完成** | PAR2QO 四件套 + VIDEX 四件套，共 4,057 行 | `par2qo_robustness.py`(832), `par2qo_querylets.py`(531), `par2qo_plan_cache.py`(323), `par2qo_utils.py`(384), `videx_metadata.py`(494), `videx_utils.py`(483), `videx_ndv_estimator.py`(522), `videx_histogram_utils.py`(488) |
| **第二位 Claude** | **M013–M016 深度移植** | 🔲 待做 | NCCL 拓扑 + Megatron 流水线深度移植 | 从 `upstream/tabular/src/` 移植 B-tree/hash 索引核心到 `core/`，增强 `topology.py` 用 Dijkstra 多跳路径(INV-4)，重写 `pipeline_scheduler.py` 用真实 Megatron bubble 公式(INV-1,INV-2) |
| **第三位 Claude** | **M017–M020 深度移植** | 🔲 待做 | vLLM 缓存 + DeepSeek FP8 深度移植 | 重写 `cache_manager.py` 用显式 `table_name` 字段(INV-3)，修复 `fp8_stats.py` 的 orig==0 和 NaN 兜底(INV-5,INV-6)，移植 `upstream/videx` 的直方图/NDV 到 `videx_histogram.py` |
| **第四位 Claude** | **M021–M026 深度移植** | 🔲 待做 | CUTLASS GPU kernel + BytePS/APEX 分布式 | 深化 `gpu_cost_kernel.py` (CUTLASS gemm_universal 模式)，重写 `distributed/sync.py` (BytePS push_pull 异步)，重写 `distributed/optimizer.py` (APEX DistributedFusedAdam 融合) |
| **第五位 Claude** | **M027–M032 深度移植** | 🔲 待做 | JAX 分片 + NCCL all_reduce + FSDP 兼容 | 深化 `sharding.py` (JAX pjit with_sharding_constraint)，重写 `distributed/collector.py` (NCCL all_reduce backend)，重写 `distributed/fsdp_compat.py` (FairScale FSDP 全分片) |
| **第六位 Claude** | **M033–M038 实验+论文** | 🔲 待做 | 端到端实验配置 + 可视化 + 消融面板 | 创建 `configs/`、`scripts/run_benchmark.py`，深化 `viz/plot_panels.py`，补齐 `tests/` 集成测试，生成论文级实验数据 (2000 steps × 3 seeds × 5 methods) |

---

## 第一位 Claude 详细记录

**日期**: 2026-05-30
**任务**: 从 upstream 深度移植 PAR2QO + VIDEX 核心算法到 `lynceus/integrations/`
**方法**: `cp` 源文件 → 改造 ~20% 算法 → 加断点调试输出
**改动摘要**:

### PAR2QO 移植 (4 文件, 2,070 行)
- `par2qo_robustness.py` (832行): 保留 rqo() 三阶段流水线 (sensitivity→sampling→selection)；用 Halton 替换 SALib/sobol；去 psycopg2 换 GPU/CPU 双路成本模型
- `par2qo_querylets.py` (531行): IMDB 模板改 TPC-H schema；加 `/*+GPU*/` 设备标注和 INV-3 合规的 `table_name` 字段
- `par2qo_plan_cache.py` (323行): 静态字典改 LRU 类；加设备偏好过滤和冷/热命中率追踪
- `par2qo_utils.py` (384行): 保留 card/list_multiply/clean/hint builder；加 selectivity_perturbation 和 batch_cost_summary

### VIDEX 移植 (4 文件, 1,987 行)
- `videx_metadata.py` (494行): 保留 VidexDBTaskStats/VidexTableStats/merge；Pydantic→dataclass；加 DeviceTableProfile + TableMetadataRegistry
- `videx_utils.py` (483行): 保留 BTreeKeyOp/RangeCond/IndexRangeCond/GT_Table_Return；加 selectivity() 和 GpuIndexCostAnnotation
- `videx_ndv_estimator.py` (522行): 保留全部13种估计方法 (Goodman/Jackknife/Chao/GEE/Sichel/MoM/Bootstrap/HT/Shlosser/ChaoLee/SmoothedJK/ErrorBound/Ada)；去 numpy/scipy；GEE 加有限总体修正
- `videx_histogram_utils.py` (488行): 保留 build_histogram/sort_validate/merge_sorted/fit_c；block_sample 改确定性步幅；桶边界对齐 distinct value；validate_error 用 q-error

### 断点调试覆盖
所有模块提供 `┌─ STATE DUMP` / `│ 过程日志` / `└─ 结果总结` 格式输出，运行时可看到：
- RQO: 每个 plan 的 E[penalty]、σ、P(incur)、worst sample
- NDV: 13种方法并排对比表
- 直方图: ASCII 柱状图 + q-error 验证
- 缓存: hit/miss/evict 流水 + 命中率统计
