# Lynceus 移植增强报告

## 概要

| 指标 | 数值 |
|------|------|
| 总文件数 | 39 |
| 语法检查 | 39/39 ✓ |
| 导入测试 | 17/17 ✓ |
| 原始总行数 | 14,967 |
| 移植版总行数 | 16,234 |
| 覆盖率 | 108.5% |
| `_dbg()` 断点 | 376 处 |
| `_dump_*` 辅助函数 | 53 处 |

## 调试系统

环境变量 `LYNCEUS_DEBUG=1`（默认开启）控制输出。

格式：`[模块·标签] 消息`，输出到 stderr。

模块缩写：
- SCH: schema
- COS: cost_model
- ROU: router
- BEK: benchmark
- TOP: topology
- PIP: pipeline_scheduler
- SHA: sharding
- CAC: cache_manager
- DWR: data_writer
- STR: strategies
- INT: integrations
- DST: distributed

## 改写内容 (~20%)

### 算法改写
- `cal_rel_error`: SMAPE-like 对称比替代单边 log-ratio
- `measure_with_median`: IQR 异常值剔除
- `generate_query_sequence`: zipf 分布替代均匀分布
- `_compute_bubble_ratio`: 异构阶段时间修正公式
- `optimize_shard_placement`: 拓扑感知分片优化
- `_compute_load_balance_loss`: CV 变异系数衡量负载均衡

### 新增工具类
- `BenchmarkDiagnostics`: 策略对比、误差直方图、收敛检测、路由分布
- `_self_test()` 入口: data_writer 全链路自测

### 架构溯源
每个模块 docstring 包含完整的上游参考：
- PAR2QO (parse.py, postgres.py, utility.py, diagram.py)
- Megatron-LM (pipeline schedules, distributed optimizer)
- DeepSeek (Gate.forward, TopKGate balance loss)
- NCCL (ncclMemoryPool, nccl_tuner)

## 文件清单

所有 39 个文件均通过语法检查且行数 ≥ 原始版。
