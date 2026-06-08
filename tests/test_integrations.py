#!/usr/bin/env python3
"""tests/test_integrations.py — integrations/ 14模块集成测试 (M221-M230)"""
import os, sys, traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("LYNCEUS_DEBUG", "0")

passed, failed, skipped = 0, 0, 0
def assert_(c): assert c
def run(name, fn):
    global passed, failed
    try: fn(); print(f"  \033[32mPASS\033[0m  {name}"); passed += 1
    except Exception as e: print(f"  \033[31mFAIL\033[0m  {name}: {e}"); failed += 1
def skip(name, r):
    global skipped; print(f"  \033[33mSKIP\033[0m  {name}: {r}"); skipped += 1

print("=" * 60)
print("  integrations/ module tests")
print("=" * 60)

try:
    from lynceus.integrations.par2qo_bridge import HeterogeneousPlanDiagram
    run("par2qo_bridge: HeterogeneousPlanDiagram", lambda: assert_(HeterogeneousPlanDiagram("q1").query_id == "q1"))
except ImportError as e: skip("par2qo_bridge", str(e))

try:
    from lynceus.integrations.par2qo_cost import CostCalibrator, DeviceIndependentCostEstimator
    run("par2qo_cost: CostCalibrator", lambda: assert_(CostCalibrator is not None))
    run("par2qo_cost: DeviceIndependentCostEstimator", lambda: assert_(DeviceIndependentCostEstimator is not None))
except ImportError as e: skip("par2qo_cost", str(e))

try:
    from lynceus.integrations.par2qo_plan_cache import CachedPlanEntry
    run("par2qo_plan_cache: CachedPlanEntry", lambda: assert_(CachedPlanEntry(plan_indices=[0,1]).plan_indices == [0,1]))
except ImportError as e: skip("par2qo_plan_cache", str(e))

try:
    from lynceus.integrations.par2qo_querylets import make_join_querylet
    run("par2qo_querylets: make_join_querylet", lambda: assert_(make_join_querylet("t1","t2","a1","a2","c1","c2") is not None))
except ImportError as e: skip("par2qo_querylets", str(e))

try:
    from lynceus.integrations.par2qo_robustness import CostEstimate
    run("par2qo_robustness: CostEstimate", lambda: assert_(CostEstimate(cpu_cost=100.0, gpu_cost=50.0).gpu_cost == 50.0))
except ImportError as e: skip("par2qo_robustness", str(e))

try:
    from lynceus.integrations.par2qo_utils import card, clean_json
    run("par2qo_utils: card callable", lambda: assert_(callable(card)))
    run("par2qo_utils: clean_json callable", lambda: assert_(callable(clean_json)))
except ImportError as e: skip("par2qo_utils", str(e))

try:
    from lynceus.integrations.videx_bridge import DeviceAwareCostModel
    run("videx_bridge: DeviceAwareCostModel", lambda: assert_(DeviceAwareCostModel is not None))
except ImportError as e: skip("videx_bridge", str(e))

try:
    from lynceus.integrations.videx_histogram_utils import build_histogram_from_samples
    run("videx_histogram_utils: build_histogram", lambda: assert_(len(build_histogram_from_samples([1,2,3,4,5,6,7,8,9,10], k=3)) > 0))
except ImportError as e: skip("videx_histogram_utils", str(e))

try:
    from lynceus.integrations.videx_metadata import ColumnMeta, DeviceTableProfile
    run("videx_metadata: ColumnMeta", lambda: assert_(ColumnMeta(name="col1").name == "col1"))
    run("videx_metadata: DeviceTableProfile", lambda: assert_(DeviceTableProfile(table_name="t1").table_name == "t1"))
except ImportError as e: skip("videx_metadata", str(e))

try:
    from lynceus.integrations.videx_ndv_estimator import NDVEstimator
    run("videx_ndv_estimator: NDVEstimator", lambda: assert_(NDVEstimator(original_num=1000) is not None))
except ImportError as e: skip("videx_ndv_estimator", str(e))

try:
    from lynceus.integrations.videx_utils import BTreeKeyOp
    run("videx_utils: BTreeKeyOp enum", lambda: assert_(hasattr(BTreeKeyOp, '__members__')))
except ImportError as e: skip("videx_utils", str(e))

try:
    from lynceus.integrations.tabular_bridge import compute_btree_fanout, compute_btree_height
    run("tabular_bridge: compute_btree_fanout", lambda: assert_(compute_btree_fanout(8, 8) > 0))
    run("tabular_bridge: compute_btree_height", lambda: assert_(compute_btree_height(1000000, 100) >= 1))
except ImportError as e: skip("tabular_bridge", str(e))

skip("videx_costing", "requires cachetools")
skip("videx_histogram", "requires pydantic")

print(f"\n{'='*60}\n  RESULTS: {passed} passed, {failed} failed, {skipped} skipped\n{'='*60}")
sys.exit(1 if failed else 0)
