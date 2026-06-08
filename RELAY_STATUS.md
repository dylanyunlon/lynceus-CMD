# Lynceus-CMD 接力开发进度

## 当前状态 (2026-06-08)

**总计**: 93 integration files / 175 upstream源文件 = **53.1% coverage**

---

## 第一位Claude (指挥官) — M001-M176 ✅ 已完成

**状态**: Session 1-10 全部完成
**产出**:
- 93个integration模块
- 核心引擎(router, schema, topology, sharding, cost_model)
- par2qo全系列 (cardinality, divergence, PQO engine, plan inspector, error profiler等)
- videx全系列 (service engine, stats analyzer, model inference, histogram等)
- kepler model_trainer 6模块 (via Opus 4.6)
- kepler training_data_collection_pipeline 部分 (plan_candidates, training_execution, verify_robustness)
- dispatch基础设施 (dispatch_to_opus.py, dispatch_loop.py, claude_hk_chat.sh)

---

## 第二位Claude — M177-M195

**范围**: kepler pipeline剩余 + videx model/analyze + carver脚本
**待移植文件**:
- M177: kepler_hint_extractor.py ← pg_plan_hint_extractor.py + query_text_utils.py + query_plan_utils.py + query_utils.py
- M178: kepler_param_gen_pipeline.py ← parameter_generator.py + param_gen_new.py + param_PQO_files_generate.py
- M179: kepler_evaluate_visualize.py ← evaluate.py + evaluate_both.py + evaluate_cost.py + evaluate_pqo.py + end_visualize*.py
- M180: kepler_sngp_model.py ← sngp_multihead_model.py
- M181: kepler_model_server.py ← model_server.py + model_server_main.py + query_parsing_utils.py
- M182: videx_model_innodb.py ← videx_model_innodb.py + videx_strategy.py + videx_model_example.py
- M183: videx_analyze_tools.py ← analyze_delete_rows.py + analyze_linear_distribution.py + analyze_trace_utils.py
- M184-M187: kepler pipeline remaining (distributed_execution_main, generate_plan_costs, etc.)
- M188-M191: carver numbered scripts (0_generate_parameter through 10_evaluate_cost)
- M192-M195: par2qo prep_query_template (2402L大文件) + remaining utility files

---

## 第三位Claude — M196-M210

**范围**: videx深度移植 + tabular扩展
- M196-M199: videx核心 (videx_histogram.py 1520L, videx_service.py 691L, videx_utils.py 1013L, videx_metadata.py 1305L)
- M200-M203: videx common + databases (db_variable, exceptions, pydantic_utils, sample_info, mysql_command, explain_result, common_operation)
- M204-M207: videx env + build (rds_env.py 391L, videx_build_env.py 254L, start_videx_server.py, statistics_info.py, estimate_stats_length.py)
- M208-M210: tabular config扩展 (config.py 354L + C++桥接增强)

---

## 第四位Claude — M211-M225

**范围**: videx测试套件 + kepler测试移植
- M211-M217: videx test (test_desc_index, test_info_low, test_mulcol_ndv, test_rec_in_ranges, test_records_in_range 827L, test_videx_utils 525L)
- M218-M221: kepler data_management tests (database_simulator_test, workload_test, test_util)
- M222-M225: kepler examples (active_copy.py, active_learning_main, active_learning_single) + integration tests

---

## 第五位Claude — M226-M240

**范围**: 端到端pipeline打通 + benchmark自动化
- M226-M230: carver robustness scripts (8_verify_robustness系列, 99_check_all, 99_plan_content, 99_testing_query_valid)
- M231-M235: carver高级评估 (6_metadata系列, 7_best_performance系列)
- M236-M238: run_lynceus.sh增强 (实验编排, 自动化)
- M239-M240: benchmark suite (TPC-H, JOB workload runners)

---

## 第六位Claude — M241-M260

**范围**: 性能优化 + 论文实验 + 文档
- M241-M245: 内存优化 (对象池, 延迟初始化, mmap)
- M246-M250: 并发优化 (线程安全缓存, 无锁队列)
- M251-M255: 论文实验复现 + 数据收集
- M256-M260: 完整文档 + API reference + README更新

---

## 子模型调度协议

1. `git clone https://github.com/dylanyunlon/claude-hk-config.git /tmp/claude-hk-config`
2. 从 `raw_curl.txt` 提取cookie和org
3. 用 `dispatch_loop.py` 或 `claude_hk_chat.sh` 调用 claude-opus-4-6
4. 模型选 `claude-opus-4-6`, effort=`high`
5. 截断时发 `Continue` 续传
6. 推送: `git push origin main` (PAT已配置在git remote url中)
7. 作者: `dylanyunlon <dogechat@163.com>`
8. **不开新分支, 不加v2/port后缀**

## Git AM 使用

```bash
# 应用本轮patches
git am session10_patches.patch
```
