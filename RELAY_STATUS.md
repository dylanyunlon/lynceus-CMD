# Lynceus-CMD Relay Development Plan

## Current State (Session 15, 2026-06-08)
- Total integration files: 111
- Upstream .py files: 197
- Coverage: ~56.3%
- Last benchmark: 2000 steps x 3 seeds
- Mean latency: 4144.7µs (steady: 4116.7µs, 12% improvement over warmup)

## Benchmark Results (Paper Table Data)
| Metric | Value |
|--------|-------|
| Mean Latency | 4144.7 µs |
| P50 Latency | 3369.7 µs |
| P99 Latency | 13001.6 µs |
| Warmup (0-100) | 4675.7 µs |
| Steady (100+) | 4116.7 µs |
| Cache Hit Rate | 99.9% |
| Improvement | 12.0% |

## Server Environment (ags1)
- Conda env: walking3 (Python 3.10, PyTorch 2.4.1, CUDA 12.1)
- To run: `conda activate walking3 && bash run_lynceus.sh`

## Claude Worker Plan

| Claude # | Milestones | Scope | Status |
|----------|-----------|-------|--------|
| 1st (指挥官, 当前) | M001-M196 | Core engine, par2qo, videx, kepler foundation | DONE |
| 2nd | M197-M210 | kepler pipeline depth (param_gen, training exec, query_utils) | DISPATCHING |
| 3rd | M211-M225 | par2qo carver advanced (robustness verification 全系列) | QUEUED |
| 4th | M226-M240 | videx histogram + ndv_estimator + test suites | QUEUED |
| 5th | M241-M250 | E2E pipeline打通 + benchmark vs SOTA对比 | QUEUED |
| 6th | M251-M260 | 性能优化 + 论文tex实验数据填充 | QUEUED |

## Sub-Model Dispatch Protocol
1. `git clone https://github.com/dylanyunlon/claude-hk-config.git /tmp/claude-hk-config`
2. Extract cookie: `grep -oP "\-b '\K[^']+" /tmp/claude-hk-config/raw_curl.txt`
3. Use `dispatch_m197_m204.py` or `claude_hk_chat.sh` → claude-opus-4-6
4. Model: claude-opus-4-6, effort=medium
5. If truncated, send `Continue`
6. Push: `git push origin main` (PAT configured)

## Key Uncovered Files (72 remaining, by size)
1. param_gen_test_output.py (2046L) → kepler_param_generation.py
2. pg_execute_training_data_queries_test.py (1428L) → kepler_training_execution.py扩展
3. videx_utils.py (1013L) → videx_primitives.py扩展
4. pg_execute_training_data_queries.py (797L) → kepler_training_execution.py合并
5. videx_service.py (691L) → videx_strategy.py扩展
6. query_utils.py (558L) → kepler_sql_features.py扩展
7. database_simulator.py (394L) → kepler_db_simulator.py扩展
8. pg_execute_explain_tools.py (404L) → kepler_plan_candidates.py扩展

## Git Config
- Author: dylanyunlon <dogechat@163.com>
- Branch: main only (no feature branches)
- No file suffixes: no v2, port, bridge, engine, utils, base, compat
- PAT: configured in remote URL
