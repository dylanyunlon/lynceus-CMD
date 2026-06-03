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

---

## 第二位 Claude 详细记录 (integrations/ 深度改写)

**日期**: 2026-06-02
**里程碑**: M036–M040 (integrations 二次改写)
**任务**: 14个集成模块的~20%算法等价改写 + 调试桩密度加倍 + _dbg递归bug批量修复
**改动**: 14 files, +809 / -472 lines (净增337行)

### 关键改写
1. **videx_histogram.py**: 二分查找粗定位+线性微调 O(log n)、Laplace 平滑 KL 散度、EMD 归一化[0,1]、partition/maxsplit 防冒号误拆
2. **par2qo_robustness.py**: scrambled Halton 黄金比例哈希、Winitzki(2008) erfinv、双侧 clamp
3. **tabular_bridge.py**: cache-line 8B 对齐 fanout、Amdahl 线程模型、TLB miss 代价、Robin Hood 探测
4. **videx_ndv_estimator.py**: 5种估计器改写(Goodman/JK混合/Sichel ln-space/Bootstrap log-space/Ada 5档)
5. **par2qo_bridge.py**: softmax 重加权 + Welford 在线方差
6. **par2qo_plan_cache.py**: 2Q 驱逐策略(扫描前25%找最少访问项)
7. **videx_histogram_utils.py**: reservoir 采样混合 + galloping merge O(n log m)
8. **par2qo_cost.py**: cache miss 模型 + 动态 B-tree 深度
9. **par2qo_querylets.py**: 同表多谓词 sqrt 相关性修正
10. **videx_utils.py**: Zipf 分布选择率估计
11. **videx_cost_model.py**: 对数空间选择率乘积防下溢

### Bug 修复
- 9 个文件 `_dbg()` 自递归无限递归 → 全部修复(integrations内)
- 每文件分配唯一 3 字母 _MOD_TAG

### 调试桩: ~160 → ~330+ (翻倍), 全部函数入口覆盖

---

## 第三位 Claude 详细记录 (GPU kernel + 分布式模块深化)

**日期**: 2026-06-03
**里程碑**: M022–M035
**任务**: GPU kernel 6函数深度改写 + 分布式4模块算法改写 + sharding改写 + _dbg递归修复
**改动**: 6 files, 净增~180行, 调试桩 ~50→~145

### 关键改写
1. **gpu_cost_kernel.py**: L1/L2/HBM 三层内存模型; SM低占用率带宽惩罚; B-tree每层缓存区分+warp divergence因子; hash atomic contention; CPU/GPU 10% hysteresis防抖; PCIe Gen4/Gen5自适应
2. **distributed/sync.py**: ring allreduce congestion模型(p>8有效带宽衰减); 通信-计算30%流水线重叠
3. **distributed/optimizer.py**: Adam→AdamW weight decay解耦; gradient clipping max_norm=1.0; loss scale NaN/Inf溢出检测跳过
4. **distributed/collector.py**: cal_rel_error加SMAPE双模式+q-error诊断
5. **distributed/fsdp_compat.py**: kl_divergence加对称KL选项
6. **sharding.py**: auto_shard round-robin热参数+负载均衡CV检测

### Bug 修复
- gpu_cost_kernel.py 和 sharding.py 的 _dbg 自递归: 已修复 (全项目0残留)
