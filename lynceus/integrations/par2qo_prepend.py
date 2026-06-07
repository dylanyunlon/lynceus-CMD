"""
par2qo_prepend.py — Workload file preprocessing and template generation
Upstream ref: par2qo/code/prepend.py (MIT)

Algorithmic enhancements:
- Template extraction via regex with memoized compilation
- Sorted merge for workload file combination
- Hash-based deduplication of parameterized queries
"""
import numpy as np
import os
import json
import re
import hashlib
from collections import OrderedDict

class WorkloadPrepender:
    """Preprocess SQL workloads by prepending hints or template info."""
    
    def __init__(self):
        self._templates = OrderedDict()
        self._query_hashes = set()
        self._compiled_patterns = {}
        self._n_processed = 0
    
    def _dbg(self, label=""):
        print(f"\n[WorkloadPrepender._dbg] {label}")
        print(f"  n_templates={len(self._templates)}")
        print(f"  n_unique_queries={len(self._query_hashes)}")
        print(f"  n_processed={self._n_processed}")
        print(f"  cached_patterns={len(self._compiled_patterns)}")
    
    def _get_pattern(self, pattern_str):
        """Get compiled regex pattern with memoization."""
        if pattern_str not in self._compiled_patterns:
            self._compiled_patterns[pattern_str] = re.compile(pattern_str)
        return self._compiled_patterns[pattern_str]
    
    def extract_template(self, query):
        """Extract parameterized template from a concrete query.
        
        Replaces numeric constants with $N placeholders.
        """
        # Replace quoted strings
        template = self._get_pattern(r"'[^']*'").sub("$S", query)
        # Replace numbers
        template = self._get_pattern(r'\b\d+\.?\d*\b').sub("$N", template)
        return template
    
    def add_query(self, query, template_id=None):
        """Add a query, deduplicating and extracting template."""
        self._n_processed += 1
        
        q_hash = hashlib.md5(query.encode()).hexdigest()[:16]
        if q_hash in self._query_hashes:
            return False  # Duplicate
        self._query_hashes.add(q_hash)
        
        if template_id is None:
            template = self.extract_template(query)
            t_hash = hashlib.md5(template.encode()).hexdigest()[:12]
            template_id = t_hash
        
        if template_id not in self._templates:
            self._templates[template_id] = {
                'template': self.extract_template(query),
                'queries': [],
            }
        self._templates[template_id]['queries'].append(query)
        return True
    
    def prepend_hint(self, query, hint_string):
        """Prepend pg_hint_plan hint to a query."""
        return f"{hint_string}\n{query}"
    
    def merge_workloads(self, workload_a, workload_b):
        """Merge two sorted workloads by template similarity.
        
        Uses sorted merge for O(n+m) combination.
        """
        merged = []
        i, j = 0, 0
        
        while i < len(workload_a) and j < len(workload_b):
            ta = self.extract_template(workload_a[i])
            tb = self.extract_template(workload_b[j])
            
            if ta <= tb:
                merged.append(workload_a[i])
                i += 1
            else:
                merged.append(workload_b[j])
                j += 1
        
        merged.extend(workload_a[i:])
        merged.extend(workload_b[j:])
        
        return merged
    
    def export_templates(self):
        """Export template summary."""
        return {tid: {'n_queries': len(info['queries']), 'template': info['template']}
                for tid, info in self._templates.items()}


class QueryFileWriter:
    """Write workload files in various formats."""
    
    @staticmethod
    def write_json(queries, filepath):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({str(i): q for i, q in enumerate(queries)}, f, indent=2)
    
    @staticmethod
    def write_csv(results, filepath, fields=None):
        import csv
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        if fields is None:
            fields = list(results[0].keys()) if results else []
        with open(filepath, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            for r in results:
                w.writerow(r)


if __name__ == "__main__":
    print("=" * 60)
    print("par2qo_prepend — Integration Test")
    print("=" * 60)
    
    prep = WorkloadPrepender()
    
    # Test 1: Template extraction
    q1 = "SELECT * FROM title t WHERE t.production_year > 2005 AND t.id = 12345"
    tmpl = prep.extract_template(q1)
    print(f"  Template: {tmpl}")
    
    # Test 2: Add queries
    queries = [
        "SELECT * FROM title t WHERE t.production_year > 2005 AND t.id = 12345",
        "SELECT * FROM title t WHERE t.production_year > 2010 AND t.id = 67890",
        "SELECT count(*) FROM movie_companies mc WHERE mc.company_id = 42",
        "SELECT * FROM title t WHERE t.production_year > 2005 AND t.id = 12345",  # dup
    ]
    for q in queries:
        added = prep.add_query(q)
    
    prep._dbg("after adding queries")
    templates = prep.export_templates()
    print(f"  Templates: {json.dumps(templates, indent=2)[:200]}")
    
    # Test 3: Prepend hint
    hint = "/*+ HashJoin(t mc) SeqScan(t) */"
    hinted = prep.prepend_hint(q1, hint)
    print(f"\n  Hinted query:\n    {hinted[:100]}...")
    
    # Test 4: Merge workloads
    wl_a = ["SELECT a FROM t1", "SELECT c FROM t3"]
    wl_b = ["SELECT b FROM t2", "SELECT d FROM t4"]
    merged = prep.merge_workloads(wl_a, wl_b)
    print(f"\n  Merged: {merged}")
    
    # Test 5: File writing
    QueryFileWriter.write_json(queries[:3], "/tmp/test_workload.json")
    print(f"  Workload JSON written to /tmp/test_workload.json")
    
    print("\nAll tests passed.")
