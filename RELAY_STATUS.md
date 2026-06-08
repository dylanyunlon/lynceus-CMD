# Lynceus-CMD Relay Development Plan

## Current State (Session 14, 2026-06-08)
- Total integration files: 109
- Upstream .py files: 196
- Coverage: ~55.6%
- Last commit: 18b3e84

## Claude Worker Plan

| Claude # | Milestones | Scope | Status |
|----------|-----------|-------|--------|
| 1st (指挥官) | M001-M193 | Core engine, par2qo, videx, kepler foundation | DONE |
| 2nd | M194-M210 | kepler pipeline depth + videx model expansion | NEXT |
| 3rd | M211-M225 | par2qo carver advanced (param_gen, robustness) | QUEUED |
| 4th | M226-M240 | videx histogram + test suites | QUEUED |
| 5th | M241-M250 | E2E pipeline + benchmark generation | QUEUED |
| 6th | M251-M260 | Optimization + paper experiment data | QUEUED |

## File Suffix Cleanup (Session 14)
18 files renamed to remove misleading suffixes:
- `_bridge` → `_connector` or domain-specific
- `_engine` → domain-specific  
- `_utils` → `_common` or domain-specific
- `_base` → domain-specific
- `_compat` → simplified

## Key Upstream Files Still Uncovered (by size)
1. prep_query_template.py (2402L) — par2qo template preparation
2. param_gen_test_output.py (2046L) — kepler parameter generation
3. param_gen_test_output_robustness.py (1758L) — kepler robustness
4. robustness.py (1609L) — par2qo robustness testing
5. videx_histogram.py (1520L) — videx histogram core
6. pg_execute_training_data_queries_test.py (1428L) — kepler training data
7. videx_metadata.py (1305L) — videx metadata management
8. querylets.py (1217L) — par2qo query decomposition
9. videx_utils.py (1013L) — videx core utilities

## Environment Reference
- Conda env: Python 3.10, PyTorch 2.4.1, CUDA 12.1
- Target: conda+GPU environment per llm4walking_run.sh pattern
- Dependencies: pure python + numpy (no DB, no pydantic)

## Git Config
- Author: dylanyunlon <dogechat@163.com>
- Branch: main only (no feature branches)
- No file suffixes: no v2, port, bridge, engine, utils, base, compat
