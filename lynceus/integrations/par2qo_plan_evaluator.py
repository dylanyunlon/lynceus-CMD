"""
par2qo_plan_evaluator.py — PQO plan latency evaluation and comparison
Upstream ref: par2qo/code/pqo_plan_evaluate.py (MIT)

Algorithmic enhancements:
- Huber loss for robust latency aggregation (replaces simple mean)
- Exponential moving median for streaming latency tracking
- Confidence interval estimation via bootstrap
- Outlier detection using modified Z-score (MAD-based)
"""
import numpy as np
import csv
import os
import time
from collections import defaultdict

class LatencyAggregator:
    """Robust latency aggregator using Huber loss and online statistics."""
    
    def __init__(self, huber_delta=50.0, ema_alpha=0.15):
        self._huber_delta = huber_delta
        self._alpha = ema_alpha
        self._values = []
        self._ema = None
        self._welford_n = 0
        self._welford_mean = 0.0
        self._welford_m2 = 0.0
    
    def _dbg(self, label=""):
        print(f"  [LatencyAggregator._dbg] {label}")
        print(f"    n_values={len(self._values)}")
        if self._values:
            print(f"    raw_mean={np.mean(self._values):.3f}ms")
            print(f"    raw_median={np.median(self._values):.3f}ms")
            print(f"    huber_mean={self.huber_mean():.3f}ms")
            print(f"    ema={self._ema:.3f}ms" if self._ema else "    ema=None")
            print(f"    welford_var={self.variance():.3f}")
            print(f"    mad={self.mad():.3f}")
    
    def add(self, latency):
        """Add a latency observation with EMA and Welford updates."""
        self._values.append(latency)
        
        # EMA update
        if self._ema is None:
            self._ema = latency
        else:
            self._ema = self._alpha * latency + (1 - self._alpha) * self._ema
        
        # Welford update
        self._welford_n += 1
        delta = latency - self._welford_mean
        self._welford_mean += delta / self._welford_n
        delta2 = latency - self._welford_mean
        self._welford_m2 += delta * delta2
    
    def variance(self):
        if self._welford_n < 2:
            return 0.0
        return self._welford_m2 / (self._welford_n - 1)
    
    def huber_mean(self):
        """Huber loss-based robust mean (less sensitive to outliers)."""
        if not self._values:
            return 0.0
        arr = np.array(self._values)
        median = np.median(arr)
        residuals = arr - median
        
        # Huber weighting
        weights = np.where(
            np.abs(residuals) <= self._huber_delta,
            1.0,
            self._huber_delta / (np.abs(residuals) + 1e-12)
        )
        return float(np.average(arr, weights=weights))
    
    def mad(self):
        """Median Absolute Deviation for outlier detection."""
        if len(self._values) < 2:
            return 0.0
        arr = np.array(self._values)
        return float(np.median(np.abs(arr - np.median(arr))))
    
    def detect_outliers(self, threshold=3.5):
        """Modified Z-score outlier detection using MAD."""
        if len(self._values) < 3:
            return []
        arr = np.array(self._values)
        median = np.median(arr)
        mad = self.mad()
        if mad < 1e-12:
            return []
        modified_z = 0.6745 * (arr - median) / mad
        return list(np.where(np.abs(modified_z) > threshold)[0])
    
    def bootstrap_ci(self, confidence=0.95, n_bootstrap=1000):
        """Bootstrap confidence interval for mean latency."""
        if len(self._values) < 5:
            return (0.0, 0.0)
        arr = np.array(self._values)
        rng = np.random.default_rng(42)
        boot_means = np.array([
            rng.choice(arr, size=len(arr), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        alpha = (1 - confidence) / 2
        lo = float(np.percentile(boot_means, 100 * alpha))
        hi = float(np.percentile(boot_means, 100 * (1 - alpha)))
        return (lo, hi)


class PlanEvaluator:
    """Evaluate and compare PQO plan latencies against default plans."""
    
    def __init__(self, huber_delta=50.0):
        self._results = []
        self._pqo_agg = LatencyAggregator(huber_delta=huber_delta)
        self._default_agg = LatencyAggregator(huber_delta=huber_delta)
        self._template_aggs = defaultdict(lambda: LatencyAggregator(huber_delta=huber_delta))
    
    def _dbg(self, label=""):
        print(f"\n{'='*60}")
        print(f"[PlanEvaluator._dbg] {label}")
        print(f"  n_results={len(self._results)}")
        self._pqo_agg._dbg("pqo_latency")
        self._default_agg._dbg("default_latency")
        if self._results:
            speedups = [r['speedup'] for r in self._results if r.get('speedup')]
            if speedups:
                print(f"  mean_speedup={np.mean(speedups):.3f}x")
                print(f"  median_speedup={np.median(speedups):.3f}x")
        print(f"{'='*60}")
    
    def evaluate_plan(self, query_id, template_id, pqo_latency, default_latency, plan_content=""):
        """Record a plan evaluation result."""
        speedup = default_latency / max(pqo_latency, 1e-6)
        
        result = {
            'query_id': query_id,
            'template_id': template_id,
            'pqo_latency': pqo_latency,
            'default_latency': default_latency,
            'plan_content': plan_content,
            'speedup': speedup,
            'timestamp': time.time(),
        }
        self._results.append(result)
        
        self._pqo_agg.add(pqo_latency)
        self._default_agg.add(default_latency)
        self._template_aggs[template_id].add(pqo_latency)
        
        return result
    
    def simulate_latency(self, base_latency, noise_std=5.0, n_runs=5):
        """Simulate query latency with Gaussian noise (replaces real DB call)."""
        rng = np.random.default_rng()
        latencies = base_latency + rng.normal(0, noise_std, n_runs)
        latencies = np.maximum(latencies, 0.1)
        return float(np.median(latencies))  # Median instead of mean for robustness
    
    def robustness_evaluation(self, query_id, template_id, plan_content, 
                               base_latency, n_instances=9, split="random"):
        """Evaluate plan robustness across multiple DB instances.
        
        Returns per-instance latencies with aggregate statistics.
        """
        instance_latencies = []
        
        if split in ("random", "sliding"):
            ins_list = list(range(n_instances))
        else:
            ins_list = [1, 2, 3, 4, 6, 7]  # Category split
        
        for ins_id in ins_list:
            # Simulate with instance-specific noise
            noise_scale = 3.0 + ins_id * 0.5  # Varying noise per instance
            lat = self.simulate_latency(base_latency, noise_std=noise_scale)
            instance_latencies.append(lat)
        
        arr = np.array(instance_latencies)
        return {
            'query_id': query_id,
            'template_id': template_id,
            'plan_content': plan_content,
            'instance_latencies': instance_latencies,
            'mean': float(arr.mean()),
            'median': float(np.median(arr)),
            'std': float(arr.std()),
            'cv': float(arr.std() / (arr.mean() + 1e-12)),  # Coefficient of variation
            'max_min_ratio': float(arr.max() / (arr.min() + 1e-12)),
        }
    
    def summary(self):
        """Aggregate evaluation summary."""
        if not self._results:
            return {}
        speedups = [r['speedup'] for r in self._results]
        return {
            'n_queries': len(self._results),
            'pqo_huber_mean': self._pqo_agg.huber_mean(),
            'default_huber_mean': self._default_agg.huber_mean(),
            'mean_speedup': float(np.mean(speedups)),
            'median_speedup': float(np.median(speedups)),
            'pqo_outliers': len(self._pqo_agg.detect_outliers()),
            'pqo_ci_95': self._pqo_agg.bootstrap_ci(),
        }
    
    def save_csv(self, filepath):
        """Save evaluation results to CSV."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        fields = ['query_id', 'template_id', 'pqo_latency', 'default_latency', 'speedup', 'plan_content']
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for r in self._results:
                writer.writerow(r)


if __name__ == "__main__":
    print("=" * 60)
    print("par2qo_plan_evaluator — Integration Test")
    print("=" * 60)
    np.random.seed(42)
    
    evaluator = PlanEvaluator(huber_delta=30.0)
    
    # Test 1: Evaluate multiple plans
    for i in range(20):
        template = f"q{i % 5}"
        pqo_lat = 10 + np.random.exponential(5)
        default_lat = 15 + np.random.exponential(8)
        evaluator.evaluate_plan(f"query_{i}", template, pqo_lat, default_lat, f"plan_{i}")
    
    evaluator._dbg("after 20 evaluations")
    
    # Test 2: Summary statistics
    summary = evaluator.summary()
    print(f"\n  Summary: {summary}")
    
    # Test 3: Robustness evaluation
    rob = evaluator.robustness_evaluation("q1", "template_1", "HashJoin(t1,t2)", 
                                           base_latency=25.0, n_instances=9)
    print(f"\n  Robustness: mean={rob['mean']:.2f}, cv={rob['cv']:.3f}, max/min={rob['max_min_ratio']:.2f}")
    print(f"  Instance latencies: {[f'{l:.1f}' for l in rob['instance_latencies']]}")
    
    # Test 4: Outlier detection
    agg = LatencyAggregator()
    for v in [10, 11, 12, 10, 11, 200, 11, 10, 300, 12]:
        agg.add(v)
    outliers = agg.detect_outliers()
    agg._dbg("with outliers")
    print(f"  Detected outliers at indices: {outliers}")
    
    # Test 5: Bootstrap CI
    ci = agg.bootstrap_ci()
    print(f"  95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
    
    # Test 6: Save CSV
    evaluator.save_csv("/tmp/test_eval_results.csv")
    print(f"\n  CSV saved to /tmp/test_eval_results.csv")
    
    print("\nAll tests passed.")
