"""
test_integration_pipeline — Cross-module integration tests for Lynceus.

M130: Validates end-to-end pipeline functionality across:
  - par2qo modules (cardinality → divergence → plan selection)
  - videx modules (metadata → histogram → stats → inference)
  - cross-system (par2qo plan selection ↔ videx cost model)
"""
import os
import sys
import time
import math
import random
import traceback

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))

def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[integ_test] {tag}: {items}")


class IntegrationTestRunner:
    """Run integration tests across lynceus modules."""
    
    def __init__(self):
        self._results = []
        self._start_time = None
    
    def run_all(self):
        """Run all integration tests."""
        self._start_time = time.time()
        
        tests = [
            ("par2qo_cardinality_pipeline", self.test_par2qo_cardinality_pipeline),
            ("par2qo_divergence_to_pqo", self.test_par2qo_divergence_to_pqo),
            ("par2qo_plan_selection_flow", self.test_par2qo_plan_selection_flow),
            ("videx_metadata_to_histogram", self.test_videx_metadata_to_histogram),
            ("videx_stats_to_inference", self.test_videx_stats_to_inference),
            ("videx_env_lifecycle", self.test_videx_env_lifecycle),
            ("videx_logging_pipeline", self.test_videx_logging_pipeline),
            ("cross_module_cost_estimation", self.test_cross_module_cost_estimation),
            ("serialization_roundtrip", self.test_serialization_roundtrip),
            ("data_utils_pipeline", self.test_data_utils_pipeline),
        ]
        
        for name, test_fn in tests:
            try:
                test_fn()
                self._results.append((name, "PASS", None))
                _dbg("test_pass", name=name)
            except Exception as e:
                self._results.append((name, "FAIL", str(e)))
                _dbg("test_fail", name=name, error=str(e))
        
        elapsed = time.time() - self._start_time
        return self._format_results(elapsed)
    
    # ── Test implementations ─────────────────────────────────────
    
    def test_par2qo_cardinality_pipeline(self):
        """Test cardinality estimation → error profiling pipeline."""
        from lynceus.integrations.par2qo_cardinality import (
            ori_cardest, _geo_mean, prep_basic_sensitive_rel_id
        )
        from lynceus.integrations.par2qo_error_profiler import (
            cal_rel_error, WelfordAccumulator, KernelDensityEstimator,
            reservoir_sample
        )
        
        # Test cardinality functions
        random.seed(42)
        all_ids, base_ids, join_ids, interleaved = prep_basic_sensitive_rel_id(5, 3)
        assert len(all_ids) == 8, f"Expected 8 all_ids, got {len(all_ids)}"
        assert len(base_ids) == 5
        assert len(join_ids) == 3
        
        # Welford online accumulator
        acc = WelfordAccumulator()
        true_cards = [random.randint(100, 10000) for _ in range(50)]
        for tc in true_cards:
            acc.update(float(tc))
        assert acc.count == 50
        assert acc.mean > 0
        assert acc.variance >= 0
        
        # KDE estimation — requires data at construction
        errors = [random.gauss(0, 1) for _ in range(100)]
        kde = KernelDensityEstimator(errors)
        log_scores = kde.score_samples([0.0])
        # score_samples returns log-density; exp() gives actual density
        density = math.exp(log_scores[0])
        assert density > 0, f"KDE density at mean should be positive: {density}"
        
        # Reservoir sampling
        items = list(range(100))
        sample = reservoir_sample(items, 10)
        assert len(sample) == 10
    
    def test_par2qo_divergence_to_pqo(self):
        """Test divergence computation → PQO engine integration."""
        from lynceus.integrations.par2qo_divergence import (
            js_divergence, js_distance, hellinger_distance,
            k_medoids_greedy, reduce_plans_by_opt_range
        )
        from lynceus.integrations.par2qo_pqo import (
            _wilson_interval, _halton_samples, PQOMethod
        )
        
        # Create plan cost distributions
        random.seed(123)
        plans = []
        for _ in range(5):
            costs = [max(0.01, random.gauss(100, 20)) for _ in range(30)]
            total = sum(costs)
            plans.append([c / total for c in costs])
        
        # Compute divergences
        div_matrix = []
        for i in range(len(plans)):
            row = []
            for j in range(len(plans)):
                d = js_divergence(plans[i], plans[j])
                row.append(d)
            div_matrix.append(row)
        
        # Self-divergence should be ~0
        for i in range(len(plans)):
            assert div_matrix[i][i] < 0.01, f"Self-JSD too high: {div_matrix[i][i]}"
        
        # JS distance
        dist = js_distance(plans[0], plans[1])
        assert 0 <= dist <= 1.0, f"JS distance out of range: {dist}"
        
        # Wilson CI returns (center, low, high)
        ci = _wilson_interval(45, 50)
        center, low, high = ci
        assert low < center < high, f"Invalid CI order: {ci}"
        assert 0.7 < low, f"CI lower bound unexpected: {low}"
        
        # Halton quasi-random
        samples = _halton_samples(10, 3)
        assert len(samples) == 10
        assert all(0 <= s <= 1 for row in samples for s in row)
    
    def test_par2qo_plan_selection_flow(self):
        """Test plan selection with Thompson Sampling + VP-tree."""
        from lynceus.integrations.par2qo_plan_selector import (
            BestCostSelector, NearestSelector, VPTree, PlanCacheSerializer
        )
        
        # Create synthetic plans and selectivity samples
        random.seed(55)
        plans = [f"plan_{i}" for i in range(10)]
        sel_samples = [[random.random() for _ in range(3)] for _ in range(20)]
        sample_to_plan = {i: i % len(plans) for i in range(20)}
        
        # BestCost selector
        best = BestCostSelector(plans, explore_budget=5)
        for _ in range(10):
            pid, plan = best.select([0.5, 0.3, 0.7])
            assert 0 <= pid < len(plans)
            best.update_result(pid, random.random(), 0.5)
        
        # Nearest selector with VP-tree
        nearest = NearestSelector(sel_samples, sample_to_plan, plans)
        query = [0.45, 0.35, 0.65]
        pid, plan = nearest.select(query)
        assert pid in range(len(plans))
        
        # Plan cache
        cache = PlanCacheSerializer()
        cache.store("test_key", {"plan": "plan_0", "cost": 42.5})
        loaded = cache.load("test_key")
        assert loaded["plan"] == "plan_0"
        assert loaded["cost"] == 42.5
    
    def test_videx_metadata_to_histogram(self):
        """Test metadata → histogram construction pipeline."""
        from lynceus.integrations.videx_metadata_extended import (
            TableMeta, ColumnMeta, IndexMeta
        )
        from lynceus.integrations.videx_histogram_builder import (
            StreamingHistogram, HybridNDVEstimator, HistogramBuilder
        )
        
        # Create table metadata
        tbl = TableMeta("testdb", "orders", row_count=50000)
        tbl.add_column(ColumnMeta("order_id", "bigint", is_pk=True))
        tbl.add_column(ColumnMeta("amount", "decimal"))
        tbl.add_column(ColumnMeta("status", "varchar", character_max_length=20))
        tbl.add_index(IndexMeta("pk_orders", ["order_id"], is_unique=True))
        
        assert tbl.has_column("amount")
        assert not tbl.has_column("nonexistent")
        
        # Build histogram from synthetic data
        random.seed(77)
        amounts = [random.lognormvariate(4, 1) for _ in range(1000)]
        
        hist = StreamingHistogram(max_bins=50)
        for a in amounts:
            hist.add(a)
        
        q50 = hist.quantile(0.5)
        assert q50 > 0, f"Median should be positive: {q50}"
        
        # NDV estimation
        statuses = random.choices(["pending", "shipped", "delivered", "cancelled"],
                                  weights=[30, 40, 20, 10], k=500)
        ndv_est = HybridNDVEstimator()
        est = ndv_est.estimate(statuses)
        assert 3 <= est <= 10, f"NDV estimate out of range: {est}"
    
    def test_videx_stats_to_inference(self):
        """Test stats analysis → model inference pipeline."""
        from lynceus.integrations.videx_stats_analyzer import (
            ks_test_uniform, StatisticsInfo, test_linearity
        )
        from lynceus.integrations.videx_model_inference import (
            PLM4NDVModel, AdaNDVEstimator, GoodTuringModel,
            ensemble_ndv_predict
        )
        
        random.seed(88)
        
        # KS test for uniformity — returns (is_uniform, statistic)
        uniform_data = [random.random() for _ in range(100)]
        ks_result = ks_test_uniform(uniform_data)
        assert isinstance(ks_result, tuple), f"ks_test should return tuple: {type(ks_result)}"
        is_uniform, statistic = ks_result
        assert isinstance(is_uniform, bool)
        assert 0 <= statistic <= 1.0
        
        # Stats info — requires (table_name, column_name)
        stats = StatisticsInfo("bench_table", "col_a")
        stats.update(ndv=500, null_count=10, min_val=0, max_val=10000, avg_len=8.5)
        assert stats.ndv == 500
        assert stats.null_count == 10
        
        # PLM4NDV — train then predict
        plm = PLM4NDVModel()
        sample_sizes = [100, 200, 500, 1000]
        observed_ndvs = [50, 90, 200, 400]
        plm.train(sample_sizes, observed_ndvs)
        ndv = plm.predict(2000)
        assert ndv > 0, f"PLM4NDV should give positive estimate: {ndv}"
    
    def test_videx_env_lifecycle(self):
        """Test environment manager lifecycle."""
        from lynceus.integrations.videx_env_manager import EnvManager
        
        env = EnvManager(default_db="production")
        
        ddl = "CREATE TABLE users (`id` INT, `name` VARCHAR(100), `email` VARCHAR(255), KEY `idx_name` (`name`))"
        schema = env.register_table("production", ddl)
        
        assert schema["table_name"] == "users"
        assert len(schema["columns"]) >= 3
        assert len(schema["indexes"]) >= 1
        
        env.update_config({"host": "db1", "port": "3306"})
        env.update_config({"host": "db1", "port": "3306", "charset": "utf8"})
        
        meta = env.get_table_meta("production", "users")
        assert meta is not None
    
    def test_videx_logging_pipeline(self):
        """Test logging and telemetry pipeline."""
        from lynceus.integrations.videx_logging_telemetry import (
            AdaptiveLogger, ErrorRateMonitor, TraceIdGenerator
        )
        
        logger = AdaptiveLogger(reservoir_size=20)
        monitor = ErrorRateMonitor(target_rate=0.05)
        tracer = TraceIdGenerator(node_id=1)
        
        random.seed(99)
        for i in range(100):
            trace = tracer.generate()
            assert len(trace) == 16, f"Trace ID wrong length: {len(trace)}"
            
            is_error = random.random() < 0.03
            if is_error:
                logger.error(f"request-{i} failed", trace=trace)
            else:
                logger.info(f"request-{i} ok", trace=trace)
            monitor.record(is_error)
        
        assert monitor.current_rate < 0.1, f"Error rate too high: {monitor.current_rate}"
    
    def test_cross_module_cost_estimation(self):
        """Test par2qo cost → videx metadata cross-integration."""
        from lynceus.integrations.par2qo_plan_inspector import (
            decode_explain, compare_plans
        )
        from lynceus.integrations.videx_metadata_extended import (
            IndexMeta
        )
        
        # decode_explain embeds Amdahl cost model
        plan_json = {"Plan": {"Node Type": "Seq Scan", "Total Cost": 100,
                              "Plan Rows": 1000, "Plan Width": 40}}
        decoded = decode_explain(plan_json)
        assert isinstance(decoded, dict), f"decode should return dict: {type(decoded)}"
        
        # compare_plans returns float (Jaccard similarity)
        plan_a = {"Plan": {"Node Type": "Seq Scan", "Total Cost": 100}}
        plan_b = {"Plan": {"Node Type": "Seq Scan", "Total Cost": 100}}
        sim = compare_plans(plan_a, plan_b)
        assert isinstance(sim, float), f"compare_plans should return float: {type(sim)}"
        assert 0 <= sim <= 1.0, f"Similarity out of range: {sim}"
        
        # Index selectivity
        idx = IndexMeta("idx_composite", ["col_a", "col_b", "col_c"])
        sel = idx.estimate_selectivity(
            {"col_a": 1000, "col_b": 100, "col_c": 10},
            total_rows=100000
        )
        assert 0 < sel < 1.0, f"Selectivity out of range: {sel}"
    
    def test_serialization_roundtrip(self):
        """Test serialization bridge round-trip."""
        from lynceus.integrations.videx_schema import (
            SerializationBridge, SchemaEvolver, TypeRegistry
        )
        
        bridge = SerializationBridge()
        data = {"name": "test", "values": [1, 2, 3], "nested": {"key": "val"}}
        
        serialized = bridge.serialize(data)
        deserialized = bridge.deserialize(serialized)
        
        assert deserialized["name"] == "test"
        assert deserialized["values"] == [1, 2, 3]
        
        # Type registry
        registry = TypeRegistry()
        assert registry.classify("varchar(255)") == "string"
        assert registry.classify("bigint") == "numeric"
        assert not registry.is_supported("GEOMETRY")
        assert registry.is_indexable("int")
    
    def test_data_utils_pipeline(self):
        """Test data ingestion utilities."""
        from lynceus.integrations.par2qo_ingestion import (
            HyperLogLogSketch, PQOCombinationTokenizer, FrequencyAggregator
        )
        
        # HyperLogLog
        hll = HyperLogLogSketch(precision=12)
        random.seed(42)
        unique_items = set()
        for _ in range(1000):
            item = f"item_{random.randint(1, 300)}"
            hll.add(item)
            unique_items.add(item)
        
        est = hll.estimate()
        actual = len(unique_items)
        error = abs(est - actual) / actual
        assert error < 0.2, f"HLL error too high: {error:.2%} (est={est}, actual={actual})"
        
        # Tokenizer
        tokenizer = PQOCombinationTokenizer()
        line = '["value1", "value2", "value3"],'
        tokens = tokenizer.tokenize_line(line)
        assert len(tokens) == 3
        assert tokens[0] == "value1"
        
        # Frequency aggregator
        agg = FrequencyAggregator()
        rows = [["a", "x"], ["b", "x"], ["a", "y"], ["a", "x"]]
        agg.add_data(rows, {0: "col_a", 1: "col_b"})
        freqs = agg.get_frequencies("col_a")
        assert freqs.get("a", 0) == 3
    
    # ── Result formatting ────────────────────────────────────────
    
    def _format_results(self, elapsed):
        passed = sum(1 for _, status, _ in self._results if status == "PASS")
        failed = sum(1 for _, status, _ in self._results if status == "FAIL")
        
        lines = [
            f"\n{'='*60}",
            f"INTEGRATION TEST RESULTS — {elapsed:.2f}s",
            f"{'='*60}",
        ]
        
        for name, status, error in self._results:
            icon = "✓" if status == "PASS" else "✗"
            line = f"  {icon} {name}"
            if error:
                line += f" — {error[:60]}"
            lines.append(line)
        
        lines.append(f"\n  {passed} passed, {failed} failed out of {len(self._results)}")
        lines.append(f"{'='*60}\n")
        
        return "\n".join(lines)


# Entry point
def run_tests():
    runner = IntegrationTestRunner()
    return runner.run_all()


if __name__ == "__main__":
    print(run_tests())
