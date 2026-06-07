# Lynceus-CMD Claude Relay Development Plan

## Overview

This project uses a relay development model where multiple Claude sessions
(Claude #1 through Claude #6+) progressively port, extend, and harden the
lynceus-CMD system. Each Claude session is assigned a milestone range (Mxxx)
and specific deliverables.

**Key Rules:**
- All work on `main` branch directly — NO feature branches, NO v2/v3/port suffixes
- Author: `dylanyunlon <dogechat@163.com>`
- Push key: use the provided GitHub PAT
- Algorithm changes (~20% rewrite): real algorithmic modifications, NOT string/docstring changes
- Every function gets `_dbg()` instrumentation printing all data/struct state
- Sub-model delegation via `claude_hk_chat.sh` → Opus 4.6 on claude.hk.cn
- Clone `dylanyunlon/claude-hk-config` for cookie/org sync before sub-model calls

---

## Claude #1 (Current Session #8): M101-M120 ✅ COMPLETED

**Scope:** Upstream algorithm porting batch 2 — par2qo + videx deep ports

### Deliverables:
- [x] M101: `par2qo_cardinality.py` — Bayesian posterior mean, Laplace+α smoothing, SHA256 trailer
- [x] M102: `par2qo_divergence.py` — Jensen-Shannon divergence, Hellinger distance, k-medoids PAM
- [x] M103: `par2qo_pqo_engine.py` — Wilson CI, IQR epsilon clamp, Halton QMC, CVaR tail-risk
- [x] M104: `par2qo_plan_inspector.py` — Amdahl parallel cost, Jaccard plan similarity, convex hull
- [x] M105: `par2qo_error_profiler.py` — SMAPE, Silverman KDE bandwidth, Welford online variance
- [x] M106: `par2qo_db_mutator.py` — exponential decay window, stratified random, consistent hashing
- [x] M107: `par2qo_template_engine.py` — AST pattern hash, Pareto frontier pruning, freq-weighted LRU
- [x] M108: `videx_service_engine.py` — circuit breaker, M/M/1 queueing, cubic spline histogram
- [x] M109: `videx_stats_analyzer.py` — Kolmogorov-Smirnov, compressed sensing, Theil-Sen regression
- [x] M110: `videx_model_inference.py` — PLM4NDV with L1/Lasso, AdaNDV exponential backoff, Good-Turing
- [x] M111: Updated `__init__.py` with full module registry
- [x] M112: CLAUDE_RELAY_PLAN.md with relay schedule
- [x] M113-M120: Reserved (commit, push, sub-model dispatch)

---

## Claude #2: M121-M140

**Scope:** Remaining upstream videx files — MySQL integration, environment, and database adapters

### Deliverables:
- M121: `videx_mysql_adapter.py` ← videx_mysql_utils.py (265L) + mysql_command.py (252L)
  - Algorithm: connection pooling with adaptive sizing, prepared statement cache with LFU eviction
- M122: `videx_env_manager.py` ← rds_env.py (391L) + videx_build_env.py (254L) + start_videx_server.py (45L)
  - Algorithm: environment fingerprinting with locality-sensitive hashing, config convergence detection
- M123: `videx_metadata_extended.py` ← meta.py (363L) + db_variable.py (237L) + common_operation.py (123L)
  - Algorithm: metadata versioning with vector clocks, diff-based incremental sync
- M124: `videx_histogram_engine.py` ← ndv_estimator.py (743L) + histogram_utils.py (538L)
  - Algorithm: streaming histogram merge (Ben-Haim & Tom-Tov), wavelet-based compression
- M125: `videx_sample_manager.py` ← sample_info.py (108L) + sample_file_info.py (82L) + explain_result.py (97L)
  - Algorithm: stratified progressive sampling, Horvitz-Thompson estimator for unequal probability
- M126: `videx_logging_telemetry.py` ← videx_logging.py (200L) + exceptions.py (104L)
  - Algorithm: adaptive log sampling (reservoir), anomaly detection on error rate via CUSUM
- M127: `videx_pydantic_bridge.py` ← pydantic_utils.py (58L) + sqlbrain_constants.py (23L)
  - Algorithm: schema evolution with backward compatibility validation
- M128-M130: Integration tests for all videx modules
- M131-M135: Cross-module smoke tests (par2qo → videx pipeline)
- M136-M140: Documentation, performance benchmarks

---

## Claude #3: M141-M160

**Scope:** Remaining par2qo utility files + tabular C++ bridge expansion

### Deliverables:
- M141: `par2qo_diagram_cost.py` ← diagram_best_cost.py (152L) + diagram_nearest.py (96L)
  - Algorithm: k-nearest with VP-tree spatial index, cost surface interpolation via RBF
- M142: `par2qo_data_transform.py` ← dict2json.py (13L) + gen_error_list.py (6L) + prepend.py (72L) + trans_pqo_combination_to_csv.py (166L)
  - Algorithm: streaming JSON serialization, columnar CSV with run-length encoding
- M143: `tabular_btree_engine.py` — expand tabular_bridge.py with full B-tree operations
  - Algorithm: write-optimized B-epsilon tree, fractional cascading for multi-level search
- M144: `tabular_hash_engine.py` — cuckoo hashing, Robin Hood probing
  - Algorithm: bucket-level Bloom filter, adaptive resizing with hysteresis
- M145-M150: `lynceus/core/` enhancements — routing engine, cost model fusion
- M151-M155: `lynceus/distributed/` — multi-node coordination, gossip protocol state sync
- M156-M160: End-to-end integration test suite

---

## Claude #4: M161-M180

**Scope:** Strategies module and visualization

### Deliverables:
- M161-M165: `lynceus/strategies/` — adaptive strategy selection based on workload classification
  - Algorithm: multi-armed bandit (UCB1) for strategy exploration, Thompson sampling
- M166-M170: `lynceus/viz/` — real-time cost surface visualization, plan comparison dashboards
- M171-M175: `run_lynceus.sh` enhancement — experiment orchestration, result collection
- M176-M180: Performance regression tests, CI integration

---

## Claude #5: M181-M200

**Scope:** Hardening, optimization, and production readiness

### Deliverables:
- M181-M185: Memory optimization — object pooling, lazy initialization, mmap for large histograms
- M186-M190: Concurrency — thread-safe caches, lock-free queues for request pipeline
- M191-M195: Error handling — graceful degradation, circuit breaker patterns throughout
- M196-M200: Benchmark suite — TPC-H, JOB, DSB workload runners

---

## Claude #6: M201-M220

**Scope:** Advanced features and research extensions

### Deliverables:
- M201-M205: Learned index structures — neural network cost model, gradient-based plan optimization
- M206-M210: Workload forecasting — ARIMA/Prophet for query pattern prediction
- M211-M215: Multi-objective optimization — Pareto-optimal plan selection across latency/throughput/memory
- M216-M220: Paper-ready experiments and result collection

---

## Sub-Model Delegation Protocol

1. Clone `dylanyunlon/claude-hk-config` for cookie/org config
2. Use `claude_hk_chat.sh` with Opus 4.6 model
3. Send task prompt as attachment with milestone range
4. If response truncated (413KB+), send "Continue" to resume
5. Verify output, commit with author `dylanyunlon <dogechat@163.com>`
6. Push to main directly

## File Organization

```
lynceus-CMD/
├── lynceus/
│   ├── core/           # Routing engine, cost model fusion
│   ├── distributed/    # Multi-node coordination
│   ├── integrations/   # Upstream ports (THIS IS THE MAIN WORK)
│   ├── strategies/     # Adaptive strategy selection
│   └── viz/            # Visualization
├── upstream/
│   ├── par2qo/         # Source: query optimization
│   ├── videx/          # Source: virtual index advisor
│   └── tabular/        # Source: B-tree/hash C++
├── CLAUDE_RELAY_PLAN.md
├── CLAUDE_PROGRESS.md
├── claude_hk_chat.sh
└── run_lynceus.sh
```
