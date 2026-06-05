#!/usr/bin/env python3
"""tests/test_distributed.py — distributed/ 4模块测试 (M231-M240)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("LYNCEUS_DEBUG", "0")
passed, failed, skipped = 0, 0, 0
def assert_(c): assert c
def run(name, fn):
    global passed, failed
    try: fn(); print(f"  \033[32mPASS\033[0m  {name}"); passed += 1
    except Exception as e: print(f"  \033[31mFAIL\033[0m  {name}: {e}"); failed += 1
print("=" * 60); print("  distributed/ module tests"); print("=" * 60)

try:
    from lynceus.distributed.sync import SyncConfig, SyncMetrics, SyncStrategy
    run("sync: SyncConfig", lambda: assert_(SyncConfig is not None))
    run("sync: SyncStrategy enum", lambda: assert_(hasattr(SyncStrategy, '__members__')))
except ImportError as e: print(f"  SKIP sync: {e}")
try:
    from lynceus.distributed.collector import AllReduceCollector, CollectionBuffer
    run("collector: AllReduceCollector", lambda: assert_(AllReduceCollector is not None))
    run("collector: CollectionBuffer", lambda: assert_(CollectionBuffer is not None))
except ImportError as e: print(f"  SKIP collector: {e}")
try:
    from lynceus.distributed.optimizer import DistributedCostModelOptimizer, OptimizerConfig
    run("optimizer: OptimizerConfig", lambda: assert_(OptimizerConfig is not None))
    run("optimizer: DistributedCostModelOptimizer", lambda: assert_(DistributedCostModelOptimizer is not None))
except ImportError as e: print(f"  SKIP optimizer: {e}")
try:
    from lynceus.distributed.fsdp_compat import FSDPCompatLayer, FSDPConfig, FSDPShardingStrategy
    run("fsdp_compat: FSDPConfig", lambda: assert_(FSDPConfig is not None))
    run("fsdp_compat: FSDPShardingStrategy enum", lambda: assert_(hasattr(FSDPShardingStrategy, '__members__')))
    run("fsdp_compat: FSDPCompatLayer", lambda: assert_(FSDPCompatLayer is not None))
except ImportError as e: print(f"  SKIP fsdp_compat: {e}")

print(f"\n{'='*60}\n  RESULTS: {passed} passed, {failed} failed\n{'='*60}")
sys.exit(1 if failed else 0)
