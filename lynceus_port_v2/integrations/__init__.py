"""
Lynceus integrations — bridges to upstream PAR2QO, VIDEX, and Tabular.

Ported modules (upstream → lynceus):
  PAR2QO:
    robustness.py      → par2qo_robustness.py   (robust query optimization)
    querylets.py        → par2qo_querylets.py    (SQL template fragments)
    cached_robust_plan_dict.py → par2qo_plan_cache.py (plan caching)
    utility.py          → par2qo_utils.py        (cardinality & cost helpers)
    diagram.py          → par2qo_bridge.py       (plan diagram / selection)
    postgres.py         → par2qo_cost.py         (cost estimation bridge)

  VIDEX:
    videx_strategy.py   → videx_bridge.py        (virtual index strategy)
    videx_histogram.py  → videx_histogram.py     (histogram engine)
    videx_metadata.py   → videx_metadata.py      (table/column metadata)
    videx_utils.py      → videx_utils.py         (B-tree range utilities)
    ndv_estimator.py    → videx_ndv_estimator.py (NDV estimation methods)
    histogram_utils.py  → videx_histogram_utils.py (histogram construction)
    videx_model_innodb.py → videx_cost_model.py  (InnoDB cost model)

  Tabular:
    hash_table_common.h → tabular_bridge.py      (index build cost)
"""
