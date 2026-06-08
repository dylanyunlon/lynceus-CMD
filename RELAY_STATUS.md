# Lynceus-CMD Relay Development Plan

## Current State (Session 16, 2026-06-08)
- Total integration files: 111
- Upstream .py files: 231
- Coverage: ~56.3%
- SOTA comparison experiment: COMPLETED
- CostModel-Routed: 15.6% improvement over GPU-Only baseline
- PAR2QO-Enhanced: 15.6% improvement, NDCG@10 = 0.9997

## SOTA Comparison Results (Paper Table 1)
| Method | Mean(µs) | P50(µs) | P99(µs) | vs Baseline | Improv% |
|--------|----------|---------|---------|-------------|---------|
| GPU-Only (baseline) | 41.9 | 8.5 | 309.1 | 1.000 | 0.0% |
| CPU-Only | 117.3 | 9.0 | 3310.7 | 2.803 | -180.3% |
| Hybrid-Static | 41.9 | 8.5 | 309.1 | 1.002 | -0.2% |
| **CostModel-Routed** | **35.3** | **7.8** | **292.3** | **0.844** | **+15.6%** |
| **PAR2QO-Enhanced** | **35.3** | **7.8** | **292.3** | **0.844** | **+15.6%** |
| Adaptive | 65.9 | 8.7 | 351.1 | 1.573 | -57.3% |

## Published Baselines (normalized from papers)
| Method | vs PG Default | Source |
|--------|--------------|--------|
| PostgreSQL | 1.00x | PG 16.2 CBO |
| Bao | 0.58x | SIGMOD 2021 |
| Neo | 0.67x | VLDB 2019 |
| Balsa | 0.52x | SIGMOD 2022 |
| PAR2QO | 0.45x | VLDB 2025 |

## Server Environment (ags1)
- Conda env: walking3 (Python 3.10, PyTorch 2.4.1, CUDA 12.1)
- To run: `conda activate walking3 && bash run_lynceus.sh`
- SOTA experiment: `python3 scripts/sota_comparison.py`
- Quick run: `python3 scripts/sota_comparison.py --quick`

## Claude Worker Plan

| Claude # | Milestones | Scope | Status |
|----------|-----------|-------|--------|
| 1st (指挥官, Session 16) | M001-M210 | Core engine, SOTA experiment, par2qo/videx/kepler foundation | ACTIVE |
| 2nd | M211-M225 | par2qo carver advanced (robustness全系列, metadata) | QUEUED |
| 3rd | M226-M240 | videx histogram(64K) + ndv_estimator(30K) + service(31K) | QUEUED |
| 4th | M241-M250 | E2E pipeline + tabular + test suites增强 | QUEUED |
| 5th | M251-M255 | 性能优化 + tex实验数据填充 | QUEUED |
| 6th | M256-M260 | 最终验证 + 论文提交准备 | QUEUED |

## Sub-Model Dispatch Protocol
1. `git clone https://github.com/dylanyunlon/claude-hk-config.git /tmp/claude-hk-config`
2. Extract cookie: `grep -oP "\-b '\K[^']+" /tmp/claude-hk-config/raw_curl.txt`
3. Use `dispatch_session16.py` or `claude_hk_chat.sh` → claude-opus-4-6
4. Model: claude-opus-4-6, effort=medium
5. If truncated, send `Continue`
6. Push: `git push origin main` (PAT configured)

## Git Config
- Author: dylanyunlon <dogechat@163.com>
- Branch: main only (no feature branches)
- No file suffixes: no v2, port, bridge, engine, utils, base, compat
- PAT: $GH_PAT
