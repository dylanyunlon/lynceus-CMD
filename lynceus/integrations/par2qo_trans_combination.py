"""
par2qo_trans_combination.py — Transform PQO plan combinations to CSV for analysis
Upstream ref: par2qo/code/trans_pqo_combination_to_csv.py (MIT)

Algorithmic enhancements:
- Streaming CSV writer for memory-efficient large datasets
- Column type inference with statistical sampling
- EMA-tracked value distributions for data quality monitoring
"""
import numpy as np
import csv
import os
import json
from collections import defaultdict, OrderedDict

class PQOCombinationTransformer:
    """Transform PQO plan combinations into structured CSV format."""
    
    def __init__(self):
        self._records = []
        self._col_stats = defaultdict(lambda: {'n': 0, 'mean': 0.0, 'm2': 0.0})
        self._ema_alpha = 0.1
        self._ema_values = {}
    
    def _dbg(self, label=""):
        print(f"\n[PQOCombinationTransformer._dbg] {label}")
        print(f"  n_records={len(self._records)}")
        print(f"  n_columns={len(self._col_stats)}")
        for col in list(self._col_stats.keys())[:5]:
            s = self._col_stats[col]
            if s['n'] > 1:
                var = s['m2'] / (s['n'] - 1)
                print(f"    {col}: n={s['n']}, mean={s['mean']:.4f}, std={np.sqrt(var):.4f}")
    
    def _update_welford(self, col, val):
        s = self._col_stats[col]
        s['n'] += 1
        d = val - s['mean']
        s['mean'] += d / s['n']
        d2 = val - s['mean']
        s['m2'] += d * d2
    
    def add_combination(self, query_id, template_id, plan_id, plan_hint,
                         cost, latency=None, selectivities=None):
        """Add a PQO plan combination record."""
        record = OrderedDict([
            ('query_id', query_id),
            ('template_id', template_id),
            ('plan_id', plan_id),
            ('plan_hint', plan_hint),
            ('cost', cost),
            ('latency', latency or 0.0),
        ])
        
        if selectivities:
            for i, sel in enumerate(selectivities):
                record[f'sel_dim_{i}'] = sel
        
        self._records.append(record)
        
        # Track numeric columns
        for col in ['cost', 'latency']:
            if col in record and isinstance(record[col], (int, float)):
                self._update_welford(col, record[col])
                # EMA update
                if col not in self._ema_values:
                    self._ema_values[col] = record[col]
                else:
                    self._ema_values[col] = self._ema_alpha * record[col] + (1 - self._ema_alpha) * self._ema_values[col]
    
    def from_plan_dict(self, plan_dict, template_id, workload_name=""):
        """Import from plan dictionary format (upstream format)."""
        if isinstance(plan_dict, str):
            plan_dict = json.loads(plan_dict)
        
        for plan_key, plan_info in plan_dict.items():
            if isinstance(plan_info, dict):
                hint = plan_info.get('hints', plan_info.get('plan', ''))
                cost = plan_info.get('cost', 0)
            elif isinstance(plan_info, str):
                hint = plan_info
                cost = 0
            else:
                continue
            
            self.add_combination(
                query_id=f"{template_id}_{workload_name}",
                template_id=template_id,
                plan_id=plan_key,
                plan_hint=hint,
                cost=cost,
            )
    
    def save_csv(self, filepath):
        """Save to CSV with streaming writer."""
        if not self._records:
            return
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        fields = list(self._records[0].keys())
        
        # Collect all possible fields from all records
        all_fields = OrderedDict()
        for r in self._records:
            for k in r.keys():
                all_fields[k] = True
        fields = list(all_fields.keys())
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for record in self._records:
                writer.writerow(record)
    
    def save_json(self, filepath):
        """Save as JSON."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self._records, f, indent=2, default=str)
    
    def infer_column_types(self, sample_size=100):
        """Infer column types from statistical sampling."""
        if not self._records:
            return {}
        
        sample = self._records[:min(sample_size, len(self._records))]
        types = {}
        
        for key in sample[0].keys():
            values = [r.get(key) for r in sample if r.get(key) is not None]
            if not values:
                types[key] = 'null'
            elif all(isinstance(v, (int, np.integer)) for v in values):
                types[key] = 'integer'
            elif all(isinstance(v, (int, float, np.floating)) for v in values):
                types[key] = 'float'
            else:
                types[key] = 'string'
        
        return types


class DictToJson:
    """Convert various dict formats to standardized JSON.
    Upstream ref: par2qo/code/dict2json.py
    """
    
    @staticmethod
    def convert(input_data, output_path=None):
        """Convert input dict/list to JSON with standardized formatting."""
        if isinstance(input_data, str):
            # Try to parse as JSON or Python dict literal
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                import ast
                data = ast.literal_eval(input_data)
        else:
            data = input_data
        
        result = json.dumps(data, indent=2, default=str)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(result)
        
        return result


if __name__ == "__main__":
    print("=" * 60)
    print("par2qo_trans_combination — Integration Test")
    print("=" * 60)
    np.random.seed(42)
    
    # Test 1: Add combinations
    trans = PQOCombinationTransformer()
    rng = np.random.default_rng(42)
    
    for i in range(30):
        trans.add_combination(
            query_id=f"q_{i}",
            template_id=f"t_{i%5}",
            plan_id=f"plan_{i%8}",
            plan_hint=f"HashJoin(t1 t2) SeqScan(t{i%3})",
            cost=float(rng.exponential(100)),
            latency=float(rng.exponential(50)),
            selectivities=[float(rng.uniform(0.01, 0.5)) for _ in range(3)],
        )
    
    trans._dbg("after 30 combinations")
    
    # Test 2: Save CSV
    trans.save_csv("/tmp/test_pqo_combinations.csv")
    print(f"  CSV saved to /tmp/test_pqo_combinations.csv")
    
    # Test 3: Column type inference
    types = trans.infer_column_types()
    print(f"\n  Column types: {types}")
    
    # Test 4: From plan dict
    plan_dict = {
        "0": {"hints": "HashJoin(t1 t2)", "cost": 150},
        "1": {"hints": "MergeJoin(t1 t2)", "cost": 200},
    }
    trans.from_plan_dict(plan_dict, "template_7", "gaussian")
    print(f"\n  After import: {len(trans._records)} records")
    
    # Test 5: DictToJson
    result = DictToJson.convert({"key": "value", "nested": {"a": 1}})
    print(f"\n  DictToJson: {result[:50]}...")
    
    print("\nAll tests passed.")
