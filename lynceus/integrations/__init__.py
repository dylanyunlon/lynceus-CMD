"""
Lynceus integrations — bridges to upstream PAR2QO, VIDEX, and Tabular.

Ported modules (upstream → lynceus):
  PAR2QO (batch 1, Sessions #1-#7):
    robustness.py      → par2qo_robustness.py   (robust query optimization)
    querylets.py        → par2qo_querylets.py    (SQL template fragments)
    cached_robust_plan_dict.py → par2qo_plan_cache.py (plan caching)
    utility.py          → par2qo_toolkit.py        (cardinality & cost helpers)
    diagram.py          → par2qo_parametric.py       (plan diagram / selection)
    postgres.py         → par2qo_cost.py         (cost estimation bridge)

  PAR2QO (batch 2, Session #8 / M101-M120):
    prep_cardinality.py → par2qo_cardinality.py  (Bayesian cardinality w/ Laplace smoothing)
    kl.py + plan_reduction*.py → par2qo_divergence.py (Jensen-Shannon, k-medoids)
    pqo_method.py + par2qo_run.py → par2qo_pqo.py (PQO with CVaR, Halton QMC)
    plan_inspection.py + diagram*.py → par2qo_plan_inspector.py (Amdahl, convex hull)
    gen_real_error*.py + prep_sel*.py → par2qo_error_profiler.py (SMAPE, Welford, KDE)
    db_sliding.py + db_random.py + db_cat*.py → par2qo_db_mutator.py (exp decay, consistent hash)
    prep_query_template.py + prep_plan_set.py → par2qo_template.py (AST hash, Pareto)

  VIDEX (batch 1, Sessions #1-#7):
    videx_strategy.py   → videx_index_advisor.py        (virtual index strategy)
    videx_histogram.py  → videx_histogram.py     (histogram engine)
    videx_metadata.py   → videx_metadata.py      (table/column metadata)
    videx_primitives.py      → videx_primitives.py         (B-tree range utilities)
    ndv_estimator.py    → videx_ndv_estimator.py (NDV estimation methods)
    histogram_utils.py  → videx_histogram_transform.py (histogram construction)
    videx_model_innodb.py → videx_costing.py  (InnoDB cost model)

  VIDEX (batch 2, Session #8 / M101-M120):
    videx_evaluator.py + videx_strategy.py → videx_evaluator.py (circuit breaker, M/M/1)
    analyze_*.py + estimate_stats*.py → videx_stats_analyzer.py (KS test, Theil-Sen)
    plm4ndv_model_infer.py + adandv*.py → videx_model_inference.py (L1 PLM, Good-Turing)

  VIDEX (batch 3, M121-M123):
    videx_mysql_utils.py + mysql_command.py → videx_mysql_adapter.py (LFU cache, connection pooling)
    rds_env.py + videx_build_env.py → videx_env_manager.py (LSH fingerprinting, convergence detect)
    meta.py + db_variable.py + common_operation.py → videx_metadata_extended.py (vector clocks)

  PAR2QO (batch 3, Session #8 / M128-M129):
    diagram_best_cost.py + diagram_nearest.py + dict2json.py → par2qo_plan_selector.py (Thompson, VP-tree)
    prepend.py + trans_pqo_combination_to_csv.py → par2qo_ingestion.py (HLL sketch, tokenizer)

  Integration Tests (M130):
    test_integration_pipeline.py — 10 cross-module pipeline tests (all passing)

  Tabular:
    hash_table_common.h → tabular_featurizer.py      (index build cost)
"""
