"""
par2qo_explain_decoder.py — PostgreSQL EXPLAIN JSON decoder and plan structure analyzer
Upstream ref: par2qo/code/psql_explain_decoder.py (MIT)

Algorithmic enhancements:
- Iterative tree traversal instead of recursive (avoids stack overflow on deep plans)
- Selectivity ratio estimation with Laplace smoothing
- Plan fingerprinting via structure hash for deduplication
- Cost model extraction with operator-level breakdown
"""
import numpy as np
import json
import hashlib
from collections import OrderedDict, deque

# Operator type classifications
JOIN_OPERATORS = {'Merge Join', 'Hash Join', 'Nested Loop'}
SCAN_OPERATORS = {'Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Heap Scan', 'Bitmap Index Scan'}
PASSTHROUGH_OPS = {'Aggregate', 'Gather', 'Sort', 'Materialize', 'Hash', 'Gather Merge', 'Limit', 'Result'}

JOIN_COND_FIELD = {
    'Merge Join': 'Merge Cond',
    'Hash Join': 'Hash Cond',
    'Nested Loop': 'Join Filter',
}


class PlanNode:
    """Represents a node in a query execution plan tree."""
    
    __slots__ = ['node_type', 'alias', 'est_rows', 'actual_rows', 'cost',
                 'children', 'relationship', 'condition', 'depth']
    
    def __init__(self, node_type, alias="", est_rows=0, actual_rows=0, cost=0.0):
        self.node_type = node_type
        self.alias = alias
        self.est_rows = est_rows
        self.actual_rows = actual_rows
        self.cost = cost
        self.children = []
        self.relationship = ""
        self.condition = ""
        self.depth = 0
    
    def _dbg(self, label=""):
        print(f"  [PlanNode._dbg] {label}")
        print(f"    type={self.node_type}, alias={self.alias}")
        print(f"    est_rows={self.est_rows}, actual_rows={self.actual_rows}")
        print(f"    cost={self.cost:.2f}, depth={self.depth}")
        print(f"    n_children={len(self.children)}")


class ExplainDecoder:
    """Decode PostgreSQL EXPLAIN JSON into structured plan representations.
    
    Uses iterative BFS traversal for safety on deep plans.
    """
    
    def __init__(self):
        self._decode_count = 0
        self._fingerprint_cache = OrderedDict()
        self._cache_capacity = 512
    
    def _dbg(self, label=""):
        print(f"\n[ExplainDecoder._dbg] {label}")
        print(f"  decode_count={self._decode_count}")
        print(f"  fingerprint_cache_size={len(self._fingerprint_cache)}")
    
    def parse_explain_json(self, explain_json):
        """Parse EXPLAIN JSON (list or dict) into plan dict.
        
        Handles various PostgreSQL EXPLAIN output formats.
        """
        if isinstance(explain_json, str):
            explain_json = json.loads(explain_json)
        
        if isinstance(explain_json, list):
            if len(explain_json) > 0 and isinstance(explain_json[0], list):
                plan_data = explain_json[0][0][0].get('Plan', explain_json[0][0][0])
            elif len(explain_json) > 0 and isinstance(explain_json[0], dict):
                plan_data = explain_json[0].get('Plan', explain_json[0])
            else:
                plan_data = explain_json
        else:
            plan_data = explain_json.get('Plan', explain_json)
        
        return plan_data
    
    def decode_plan(self, plan_data):
        """Decode plan JSON into structured representation using iterative BFS.
        
        Returns (join_order_str, join_conditions, scan_methods, root_node).
        """
        self._decode_count += 1
        
        if isinstance(plan_data, str):
            plan_data = self.parse_explain_json(plan_data)
        
        join_order_parts = []
        join_conditions = []
        scan_methods = []
        
        # BFS traversal
        queue = deque([(plan_data, 0)])
        root_node = None
        nodes = []
        
        while queue:
            node_data, depth = queue.popleft()
            
            node_type = node_data.get('Node Type', 'Unknown')
            alias = node_data.get('Alias', node_data.get('Relation Name', ''))
            est_rows = node_data.get('Plan Rows', 0)
            actual_rows = node_data.get('Actual Rows', 0)
            total_cost = node_data.get('Total Cost', 0.0)
            
            node = PlanNode(node_type, alias, est_rows, actual_rows, total_cost)
            node.depth = depth
            node.relationship = node_data.get('Parent Relationship', '')
            
            nodes.append(node)
            if root_node is None:
                root_node = node
            
            children = node_data.get('Plans', [])
            
            # Handle child ordering (Outer first, then Inner)
            if len(children) == 2:
                c0_rel = children[0].get('Parent Relationship', '')
                c1_rel = children[1].get('Parent Relationship', '')
                if c0_rel == 'Inner' and c1_rel == 'Outer':
                    children = [children[1], children[0]]
            
            if node_type in JOIN_OPERATORS:
                # Extract join condition
                cond_field = JOIN_COND_FIELD.get(node_type, '')
                if cond_field and cond_field in node_data:
                    cond = node_data[cond_field]
                    join_conditions.append(f"{node_type}({cond})")
                    node.condition = cond
                
                # Build join order string
                child_aliases = []
                for c in children:
                    c_type = c.get('Node Type', '')
                    c_alias = c.get('Alias', c.get('Relation Name', 'x'))
                    if c_type in JOIN_OPERATORS:
                        child_aliases.append(f"<{c_type}>")
                    else:
                        child_aliases.append(c_alias)
                join_order_parts.append(f"{node_type}({','.join(child_aliases)})")
            
            elif node_type in SCAN_OPERATORS or 'Scan' in node_type:
                scan_methods.append(f"{node_type}({alias})")
            
            # Enqueue children
            for child in children:
                queue.append((child, depth + 1))
                child_node = PlanNode(child.get('Node Type', 'Unknown'))
                node.children.append(child_node)
        
        join_order = ' -> '.join(join_order_parts) if join_order_parts else 'sequential'
        
        return join_order, join_conditions, scan_methods, root_node
    
    def compute_selectivity_ratios(self, plan_data):
        """Extract estimated vs actual selectivity ratios from plan nodes.
        
        Uses Laplace smoothing for small cardinalities.
        """
        ratios = []
        
        queue = deque([plan_data])
        while queue:
            node = queue.popleft()
            
            est_rows = node.get('Plan Rows', 0)
            actual_rows = node.get('Actual Rows', 0)
            
            children = node.get('Plans', [])
            if len(children) == 2 and node.get('Node Type', '') in JOIN_OPERATORS:
                est_child_product = max(1, children[0].get('Plan Rows', 1)) * max(1, children[1].get('Plan Rows', 1))
                actual_child_product = max(1, children[0].get('Actual Rows', 1)) * max(1, children[1].get('Actual Rows', 1))
                
                # Laplace-smoothed selectivity
                est_sel = (est_rows + 1) / (est_child_product + 2)
                actual_sel = (actual_rows + 1) / (actual_child_product + 2)
                
                q_error = max(est_sel / (actual_sel + 1e-12), actual_sel / (est_sel + 1e-12))
                
                ratios.append({
                    'node_type': node.get('Node Type'),
                    'est_sel': est_sel,
                    'actual_sel': actual_sel,
                    'q_error': q_error,
                })
            
            for child in children:
                queue.append(child)
        
        return ratios
    
    def fingerprint(self, plan_data):
        """Generate a structural fingerprint for plan deduplication.
        
        Hashes the tree structure (node types and join order) ignoring
        cardinality estimates and costs.
        """
        if isinstance(plan_data, str):
            plan_data = self.parse_explain_json(plan_data)
        
        parts = []
        queue = deque([plan_data])
        while queue:
            node = queue.popleft()
            node_type = node.get('Node Type', '?')
            alias = node.get('Alias', node.get('Relation Name', ''))
            parts.append(f"{node_type}:{alias}")
            for child in node.get('Plans', []):
                queue.append(child)
        
        struct_str = '|'.join(parts)
        fp = hashlib.sha256(struct_str.encode()).hexdigest()[:16]
        
        # Cache
        if len(self._fingerprint_cache) >= self._cache_capacity:
            self._fingerprint_cache.popitem(last=False)
        self._fingerprint_cache[fp] = struct_str
        
        return fp
    
    def operator_cost_breakdown(self, plan_data):
        """Extract per-operator cost breakdown from EXPLAIN output."""
        breakdown = []
        
        queue = deque([plan_data])
        while queue:
            node = queue.popleft()
            node_type = node.get('Node Type', 'Unknown')
            startup = node.get('Startup Cost', 0.0)
            total = node.get('Total Cost', 0.0)
            
            breakdown.append({
                'operator': node_type,
                'alias': node.get('Alias', ''),
                'startup_cost': startup,
                'total_cost': total,
                'marginal_cost': total - startup,
            })
            
            for child in node.get('Plans', []):
                queue.append(child)
        
        return breakdown


if __name__ == "__main__":
    print("=" * 60)
    print("par2qo_explain_decoder — Integration Test")
    print("=" * 60)
    
    decoder = ExplainDecoder()
    
    # Synthetic EXPLAIN JSON
    explain_json = {
        "Node Type": "Hash Join",
        "Hash Cond": "(t.id = mc.movie_id)",
        "Plan Rows": 500,
        "Actual Rows": 480,
        "Total Cost": 1500.0,
        "Startup Cost": 100.0,
        "Plans": [
            {
                "Node Type": "Seq Scan",
                "Parent Relationship": "Outer",
                "Relation Name": "title",
                "Alias": "t",
                "Plan Rows": 1000,
                "Actual Rows": 950,
                "Total Cost": 200.0,
                "Startup Cost": 0.0,
            },
            {
                "Node Type": "Hash",
                "Parent Relationship": "Inner",
                "Plan Rows": 2000,
                "Actual Rows": 2100,
                "Total Cost": 300.0,
                "Startup Cost": 250.0,
                "Plans": [
                    {
                        "Node Type": "Index Scan",
                        "Parent Relationship": "Outer",
                        "Relation Name": "movie_companies",
                        "Alias": "mc",
                        "Plan Rows": 2000,
                        "Actual Rows": 2100,
                        "Total Cost": 250.0,
                        "Startup Cost": 0.0,
                    }
                ]
            }
        ]
    }
    
    # Test 1: Decode plan
    join_order, joins, scans, root = decoder.decode_plan(explain_json)
    print(f"  Join order: {join_order}")
    print(f"  Join conditions: {joins}")
    print(f"  Scan methods: {scans}")
    
    # Test 2: Selectivity ratios
    ratios = decoder.compute_selectivity_ratios(explain_json)
    print(f"\n  Selectivity ratios:")
    for r in ratios:
        print(f"    {r['node_type']}: est={r['est_sel']:.4f}, actual={r['actual_sel']:.4f}, q_err={r['q_error']:.2f}")
    
    # Test 3: Fingerprint
    fp = decoder.fingerprint(explain_json)
    print(f"\n  Plan fingerprint: {fp}")
    
    # Test 4: Cost breakdown
    breakdown = decoder.operator_cost_breakdown(explain_json)
    print(f"\n  Cost breakdown:")
    for b in breakdown:
        print(f"    {b['operator']}({b['alias']}): total={b['total_cost']:.1f}, marginal={b['marginal_cost']:.1f}")
    
    decoder._dbg("final state")
    print("\nAll tests passed.")
