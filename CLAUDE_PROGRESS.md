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

---

## 第四位 Claude 详细记录 (核心模块深化)

**日期**: 2026-06-03
**里程碑**: M041–M060
**任务**: fp8_stats/cache_manager/schema/topology/cost_model/pipeline_scheduler/benchmark 7文件改写
**改动**: 7 files, +169/-63 行 (净增106行)

### 关键改写
1. **fp8_stats.py** (281→304): quantize_column stochastic rounding 自动对比(选SNR更高者); INV-6 严格E4M3优先不降格E5M2
2. **cache_manager.py** (347→377): acquire 从简单LRU改为扫描前25%找频率最低的(2Q策略); _admit 加ghost缓存追踪被驱逐key
3. **schema.py** (387→404): Welford 加 Kahan 补偿防浮点精度损失; 收敛检测(尾部10步CV<5%)
4. **topology.py** (398→414): NUMA 亲和性折扣(同NUMA内-30%代价); 大数据量>10MB拥塞带宽衰减
5. **cost_model.py** (454→473): margin-based 推荐——赢者需领先≥5%,否则偏向数据所在设备
6. **pipeline_scheduler.py** (454→439): 合并重复的_compute_bubble_ratio_standalone; bubble公式加micro-batch粒度 (p-1)/(m+p-1)
7. **benchmark.py** (439→455): workload phase shift 3阶段查询分布漂移; 热表Zipf权重每500步轮转

### 验证: 39/39 py_compile 全项目通过

---

## 第五位 Claude 详细记录 (viz+策略+data_writer深化)

**日期**: 2026-06-03
**里程碑**: M061–M080
**任务**: plot_panels/adaptive/cost_driven/data_writer 4文件改写
**改动**: 4 files, +110/-59 行 (净增51行)

### 关键改写
1. **plot_panels.py** (746→766): build_benchmark_report 加自动ASCII sparkline渲染+panel汇总统计; detect_trend_change 修复0.495指数bug→0.5+自适应窗口
2. **strategies/adaptive.py** (220→207): UCB1→UCB1-Tuned(方差项); Gini系数负载均衡; 删除dead code残留
3. **strategies/cost_driven.py** (153→157): PAR2QO方差追踪改为EMA退化(窗口100)
4. **data_writer.py** (725→734): measure_with_median 加预热轮+P50/P95/P99百分位

### 验证: 39/39 py_compile 全项目通过


---

## 第六位 Claude 详细记录 (核心三文件算法深化)

**日期**: 2026-06-03
**里程碑**: M101–M106
**任务**: cost_model/schema/benchmark 三个最薄文件的算法级实质改写 + 调试探针注入
**改动**: 3 files, 算法改写 12 处, 调试探针 38 处

### 关键改写
1. **cost_model.py** (441→468, +6.1%):
   - CPUCostModel: I/O 代价从 if/elif 二元切换 → Mackert-Lohman 非线性混合模型 (effective_page_cost = seq·√sel + rand·(1-√sel))
   - CPU sort: 系数从 2.0·n·logn → 1.39·n·logn (3-way merge sort, Knuth TAOCP)
   - GPUCostModel sort: bitonic O(n·log²n/P) → radix sort O(n·w/P) (CUB DeviceRadixSort, pass = key_width/radix_bits)
   - GPU L2 cache hit 建模: 数据 < 40MB 时有效 HBM 带宽翻倍
   - CostModelEngine.recommend: 计算 runner-up + margin (置信度指标)
   - estimate_on_device: LRU 近似缓存命中衰减 (同 table+device 重复访问 io×0.7)

2. **schema.py** (372→350, -5.9%, 算法密度更高):
   - compute_statistics: 两遍求 mean+var → Welford 单遍在线算法 (数值更稳定)
   - SeedCurve.append: 加 EMA 在线更新用于实时收敛检测
   - get_transfer_cost: Dijkstra+heapq → Bellman-Ford (支持未来负权边, 提前终止优化)

3. **benchmark.py** (426→440, +3.3%):
   - generate_query_sequence 表选择: uniform → Zipf 分布 (w_i = 1/i^α, 新增 zipf_alpha 参数)
   - run_benchmark total_cost: sum() → Kahan 补偿求和 (误差 O(n·ε) → O(ε))
   - run_cumulative_benchmark 前缀和: 线性累加 → Kahan 补偿累加
   - main 统计输出: 只看 final_mean → IQR (p25/p50/p75) 分位数

### 调试探针: 38 处 (_dbg/_snapshot/_Timer/dump_state/dump_topology)

---

## Claude #7: 算法深化 — Consistent Hashing / Thompson Sampling / FPL+Hedge

| Milestone | 文件 | 改动 |
|-----------|------|------|
| M311 | lynceus/router.py | _HashRing: Karger 1997一致性哈希, 150虚拟节点, MD5+二分查找 |
| M312 | lynceus/router.py | route_one: strategy推荐 + hash ring负载均衡加权融合 |
| M313 | lynceus/router.py | route_batch: locality-aware batching, 按table_names分组 |
| M314 | lynceus/router.py | tournament: 全对决Elo (Bradley-Terry), sigmoid连续得分 |
| M315 | lynceus/strategies/adaptive.py | Thompson Sampling: Beta(α,β)后验采样选device |
| M316 | lynceus/strategies/adaptive.py | Gumbel-Max trick: 数值稳定的离散概率采样 |
| M317 | lynceus/strategies/adaptive.py | Bayesian online update: Welford法维护running mean/var |
| M318 | lynceus/strategies/cost_driven.py | FPL: Follow the Perturbed Leader, 指数噪声扰动 |
| M319 | lynceus/strategies/cost_driven.py | Hedge: Multiplicative Weights Update, w*=exp(-η·loss) |
| M320 | lynceus/strategies/cost_driven.py | _dbg_regret_trace: 累积regret和权重追踪 |

**统计**: +453/-114 lines, 3 files changed
**算法参考**: Karger 1997 (consistent hashing), Kalai & Vempala 2005 (FPL), Freund & Schapire 1997 (Hedge)
**验证**: py_compile 40/40, benchmark运行正常 (fpl_decision/thompson_select checkpoints confirmed)

### 开发计划 (6位Claude分工)

| Claude | Milestones | 任务 | 状态 |
|--------|-----------|------|------|
| #7 (当前) | M311-M320 | router/strategies算法改写 (Consistent Hashing + Thompson Sampling + FPL/Hedge) | ✅ |
| #8 (下一位) | M321-M330 | integrations/ 深度移植 (par2qo diagram.py + plan_reduction算法) | 待开始 |
| #9 | M331-M340 | upstream kepler trainer/evaluation 算法移植 | 待开始 |
| #10 | M341-M350 | distributed/ 通信算法深化 (BytePS/NCCL模拟) | 待开始 |
| #11 | M351-M360 | 端到端实验运行 + 论文数据生成 | 待开始 |
| #12 | M361-M370 | 最终集成测试 + 消融实验 | 待开始 |
