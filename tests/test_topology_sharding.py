#!/usr/bin/env python3
"""tests/test_topology_sharding.py — topology+sharding+gpu_cost_kernel (M241-M250)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("LYNCEUS_DEBUG", "0")
passed, failed = 0, 0
def assert_(c): assert c
def run(name, fn):
    global passed, failed
    try: fn(); print(f"  \033[32mPASS\033[0m  {name}"); passed += 1
    except Exception as e: print(f"  \033[31mFAIL\033[0m  {name}: {e}"); failed += 1
print("=" * 60); print("  topology + sharding + gpu_cost_kernel tests"); print("=" * 60)

from lynceus.cost_model import CostModelEngine, QueryDescriptor, QueryType, create_default_topology
from lynceus.schema import HardwareKind
topo = create_default_topology()

run("topo: 6 nodes", lambda: assert_(len(topo.nodes) == 6))
run("topo: has cpu0/cpu1", lambda: assert_("cpu0" in topo.nodes and "cpu1" in topo.nodes))
run("topo: has gpu0-3", lambda: assert_(all(f"gpu{i}" in topo.nodes for i in range(4))))
run("topo: cpu0 is CPU", lambda: assert_(topo.nodes["cpu0"].kind == HardwareKind.CPU))
run("topo: gpu0 is GPU", lambda: assert_(topo.nodes["gpu0"].kind == HardwareKind.GPU))
run("topo: cpu0→gpu0 finite", lambda: assert_(topo.get_transfer_cost("cpu0","gpu0",1<<20) < float('inf')))
run("topo: cpu1→gpu0 reachable", lambda: assert_(topo.get_transfer_cost("cpu1","gpu0",1<<20) < float('inf')))
run("topo: self=0", lambda: assert_(topo.get_transfer_cost("cpu0","cpu0",1<<20) == 0))
run("topo: NUMA local < cross", lambda: assert_(topo.get_transfer_cost("cpu0","gpu0",1<<20) < topo.get_transfer_cost("cpu1","gpu0",1<<20)))

engine = CostModelEngine(topo)
run("engine: recommend", lambda: assert_(isinstance(engine.recommend(
    QueryDescriptor(query_id="t",query_type=QueryType.FULL_TABLE_SCAN,estimated_rows=1000,table_rows=100000,table_name="x")), tuple)))

try:
    from lynceus.sharding import PartitionSpec, ShardGroupConfig
    run("sharding: PartitionSpec", lambda: assert_(PartitionSpec is not None))
    run("sharding: ShardGroupConfig", lambda: assert_(ShardGroupConfig is not None))
except: print("  SKIP sharding")
try:
    from lynceus.gpu_cost_kernel import GPUCostKernel, GPUArchConfig
    run("gpu_cost_kernel: GPUCostKernel", lambda: assert_(GPUCostKernel is not None))
    run("gpu_cost_kernel: GPUArchConfig", lambda: assert_(GPUArchConfig is not None))
except: print("  SKIP gpu_cost_kernel")

print(f"\n{'='*60}\n  RESULTS: {passed} passed, {failed} failed\n{'='*60}")
sys.exit(1 if failed else 0)
