# Lynceus-CMD 接力开发进度

## 当前状态 (2026-06-08 Session 11)

**总计**: 96 integration files / 182 upstream源文件 = **52.7%** coverage
(远程有96个: 94 from sessions 1-10 + M178 from parallel worker + M179-M180 from session 11)

---

## 第一位Claude (指挥官) — M001-M180 ✅ 进行中

**状态**: Session 1-10 完成(M001-M177), Session 11 进行中(M178-M183+)
**产出**:
- 96个integration模块 (截至M180)
- 核心引擎(router, schema, topology, sharding, cost_model)
- par2qo全系列
- videx全系列
- kepler model_trainer 系列
- kepler training_data_collection_pipeline 系列
- 子模型调度基础设施 (claude_hk_chat.sh, dispatch_worker.sh, dispatch_session11.py)

**Session 11 完成**:
- M178: kepler_sngp_multihead.py (788行) — SNGP+multihead numpy实现
- M179: kepler_hint_extractor.py (937行) — PG plan hints + query text utils  
- M180: kepler_param_pipeline.py (1051行) — 参数生成pipeline

**Session 11 待完成** (API限额恢复后继续):
- M181: par2qo_model_test.py ← 3_model_test*.py (3个文件)
- M182: par2qo_evaluate_suite.py ← 4_evaluate*.py (3个文件)
- M183: par2qo_visualize_suite.py ← 5_visualize*.py (3个文件)
- M184: kepler_model_server.py ← model_server.py + model_server_main.py
- M185: kepler_e2e_eval_main.py ← e2e_evaluation_main.py + e2e_evaluation.py
- M186-M190: par2qo carver batch scripts (0_param_PQO*, 0_mixture_PQO*, etc.)

---

## 第二位Claude — M191-M210

**范围**: kepler pipeline剩余 + videx深度
**待移植文件**:
- M191-M195: kepler training_data_collection_pipeline (evaluate.py 13970B, evaluate_both.py 19957B, end_visualize系列)
- M196-M200: videx核心 (db_variable 17937B, common_operation 5106B, explain_result, mysql_command, pydantic_utils)
- M201-M205: videx databases (mysql_command 12847B, adandv_model_infer 14174B, config 23089B, process_info 9312B, rds_env 4893B)
- M206-M210: videx build + 环境 (videx_container_entrypoint, start_videx_server, statistics_info, estimate_stats_length, sample_info)

---

## 第三位Claude — M211-M225

**范围**: par2qo carver高级脚本
- M211-M215: carver评估 (10_evaluate_cost, 10_visualize_cost, 6_metadata系列, 7_best_performance系列)
- M216-M220: carver验证 (8_verify_robustness系列, 9_verify_visualize, 99_check/plan/testing)
- M221-M225: carver高级生成 (1_generate_plan_candidates, 2_execute_training_data, 0_query_metadata_analysis)

---

## 第四位Claude — M226-M240

**范围**: videx测试套件 + kepler测试
- M226-M232: videx test (test_desc_index, test_info_low, test_mulcol_ndv, test_rec_in_ranges, test_records_in_range 827L, test_videx_utils 525L, test_main)
- M233-M237: kepler data_management tests
- M238-M240: kepler examples (active_copy, active_learning_main, active_learning_single)

---

## 第五位Claude — M241-M250

**范围**: 端到端pipeline + benchmark
- M241-M244: carver robustness完整系列
- M245-M248: par2qo prep_query_template大文件移植 + 工具链
- M249-M250: benchmark suite (TPC-H, JOB workload)

---

## 第六位Claude — M251-M260

**范围**: 优化 + 文档
- M251-M254: 内存优化 (对象池, mmap, lazy init)
- M255-M258: 论文实验复现
- M259-M260: 完整文档 + API reference

---

## 子模型调度协议

1. `git clone https://github.com/dylanyunlon/lynceus-CMD.git /home/user/lynceus`
2. 读取upstream源码 (cat)
3. 创建integration文件 (cat heredoc)
4. 规则: 鲁迅式拿法mv+20%算法改动, TF→numpy, DB→内存模拟, _debug_snapshot(), __main__自测
5. 不开新分支, 不加port/v2后缀
6. git push origin main (PAT已配置)
7. 截断时发 Continue
8. API限额: 60次/重置周期, 每次任务约消耗2-4次
