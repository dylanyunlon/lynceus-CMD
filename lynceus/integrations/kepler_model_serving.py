"""
Kepler Model Serving Module
Pure numpy implementation with debug tracing.
"""

import numpy as np
import time
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

_DEBUG = False


def _dbg(fn_name: str, msg: str):
    if _DEBUG:
        print(f"[DEBUG][{fn_name}] {msg}")


# ---------------------------------------------------------------------------
# Serving Configuration
# ---------------------------------------------------------------------------
@dataclass
class ServingConfig:
    """Runtime configuration for the model server."""
    batch_size: int = 32
    timeout_ms: float = 500.0
    confidence_threshold: float = 0.7
    max_queue_size: int = 1024
    model_version: str = "v0.0.0"

    def __post_init__(self):
        fn = "ServingConfig.__post_init__"
        _dbg(fn, f"batch_size={self.batch_size}, timeout_ms={self.timeout_ms}, "
                  f"confidence_threshold={self.confidence_threshold}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be > 0")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")


# ---------------------------------------------------------------------------
# Request Logger
# ---------------------------------------------------------------------------
class RequestLogger:
    """Tracks request latencies and counts for monitoring."""

    def __init__(self, histogram_bins: int = 20):
        fn = "RequestLogger.__init__"
        self._latencies: List[float] = []
        self._counts: Dict[str, int] = defaultdict(int)
        self._errors: List[Tuple[float, str]] = []
        self._histogram_bins = histogram_bins
        _dbg(fn, f"initialised with histogram_bins={histogram_bins}")

    def log_request(self, endpoint: str, latency_ms: float,
                    success: bool = True, error_msg: str = ""):
        fn = "RequestLogger.log_request"
        _dbg(fn, f"endpoint={endpoint}, latency_ms={latency_ms:.2f}, success={success}")
        self._latencies.append(latency_ms)
        self._counts[endpoint] += 1
        if not success:
            self._errors.append((time.time(), error_msg))
            self._counts[f"{endpoint}_errors"] += 1

    def get_stats(self) -> Dict[str, Any]:
        fn = "RequestLogger.get_stats"
        _dbg(fn, f"total requests logged={len(self._latencies)}")
        if not self._latencies:
            return {"total_requests": 0, "mean_latency_ms": 0.0,
                    "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0,
                    "error_count": len(self._errors), "counts_by_endpoint": dict(self._counts)}

        arr = np.array(self._latencies)
        stats = {
            "total_requests": len(self._latencies),
            "mean_latency_ms": float(np.mean(arr)),
            "std_latency_ms": float(np.std(arr)),
            "min_latency_ms": float(np.min(arr)),
            "max_latency_ms": float(np.max(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "error_count": len(self._errors),
            "counts_by_endpoint": dict(self._counts),
        }
        _dbg(fn, f"stats computed: mean={stats['mean_latency_ms']:.2f}, "
                  f"p95={stats['p95_ms']:.2f}")
        return stats

    def latency_histogram(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (counts, bin_edges) for a histogram of latencies.
        counts: (n_bins,), bin_edges: (n_bins+1,)
        """
        fn = "RequestLogger.latency_histogram"
        _dbg(fn, f"computing histogram, n_latencies={len(self._latencies)}")
        if not self._latencies:
            empty_counts = np.zeros(self._histogram_bins, dtype=np.int64)
            empty_edges = np.linspace(0, 1, self._histogram_bins + 1)
            return empty_counts, empty_edges

        arr = np.array(self._latencies)
        counts, edges = np.histogram(arr, bins=self._histogram_bins)
        _dbg(fn, f"histogram range=[{edges[0]:.2f}, {edges[-1]:.2f}], "
                  f"peak_bin_count={counts.max()}")
        return counts, edges


# ---------------------------------------------------------------------------
# Query Parser
# ---------------------------------------------------------------------------
class QueryParser:
    """Extracts and normalises parameters from SQL-like query strings."""

    # Patterns for common WHERE-clause extractions
    _PARAM_PATTERN = re.compile(
        r"(\w+)\s*(=|>=|<=|>|<|!=|LIKE|IN)\s*['\"]?([^'\";\s,\)]+)['\"]?",
        re.IGNORECASE,
    )
    _IN_PATTERN = re.compile(
        r"(\w+)\s+IN\s*\(([^)]+)\)", re.IGNORECASE
    )

    @staticmethod
    def parse_sql_params(sql_text: str) -> List[Dict[str, Any]]:
        """
        Parse parameter bindings from a SQL WHERE clause.
        Returns list of dicts: [{column, operator, value}, ...]
        """
        fn = "QueryParser.parse_sql_params"
        _dbg(fn, f"sql_text length={len(sql_text)}")

        params: List[Dict[str, Any]] = []

        # Handle IN (...) specially
        for m in QueryParser._IN_PATTERN.finditer(sql_text):
            col = m.group(1)
            raw_vals = m.group(2)
            vals = [v.strip().strip("'\"") for v in raw_vals.split(",")]
            params.append({"column": col, "operator": "IN", "value": vals})
            _dbg(fn, f"IN-clause: column={col}, values={vals}")

        # Standard comparisons
        # Remove IN-clause regions to avoid double-matching
        cleaned = QueryParser._IN_PATTERN.sub("", sql_text)
        for m in QueryParser._PARAM_PATTERN.finditer(cleaned):
            col, op, val = m.group(1), m.group(2).upper(), m.group(3)
            # Skip SQL keywords accidentally captured
            if col.upper() in ("SELECT", "FROM", "WHERE", "AND", "OR",
                               "ORDER", "GROUP", "HAVING", "LIMIT", "SET",
                               "INSERT", "UPDATE", "DELETE", "JOIN", "TABLE"):
                continue
            params.append({"column": col, "operator": op, "value": val})
            _dbg(fn, f"param: {col} {op} {val}")

        _dbg(fn, f"total params extracted={len(params)}")
        return params

    @staticmethod
    def normalize_params(raw_params: List[Dict[str, Any]],
                         metadata: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """
        Convert parsed params into a numeric feature vector using metadata.

        metadata schema per column:
            {"type": "numeric"|"categorical",
             "min": float, "max": float,          # for numeric
             "categories": ["a","b",...]}          # for categorical

        Returns: np.ndarray of shape (n_features,)
        """
        fn = "QueryParser.normalize_params"
        _dbg(fn, f"raw_params count={len(raw_params)}, metadata keys={list(metadata.keys())}")

        features: List[float] = []
        for p in raw_params:
            col = p["column"]
            val = p["value"]
            if col not in metadata:
                _dbg(fn, f"column '{col}' not in metadata, skipping")
                continue

            meta = metadata[col]
            if meta["type"] == "numeric":
                v = float(val) if not isinstance(val, list) else float(val[0])
                lo, hi = meta["min"], meta["max"]
                normed = (v - lo) / (hi - lo + 1e-12)
                normed = np.clip(normed, 0.0, 1.0)
                features.append(float(normed))
                _dbg(fn, f"numeric {col}: raw={v}, normed={normed:.4f}")

            elif meta["type"] == "categorical":
                cats = meta["categories"]
                target = val if isinstance(val, str) else str(val)
                one_hot = [1.0 if c == target else 0.0 for c in cats]
                features.extend(one_hot)
                _dbg(fn, f"categorical {col}: target={target}, one_hot len={len(one_hot)}")

        arr = np.array(features, dtype=np.float64)
        _dbg(fn, f"feature vector shape={arr.shape}")
        return arr


# ---------------------------------------------------------------------------
# Model Server
# ---------------------------------------------------------------------------
class ModelServer:
    """
    Loads a numpy-serialised model and serves predictions.

    Model file format (npz):
        weights: (d, n_plans)  â linear projection
        biases:  (n_plans,)
        plan_names: (n_plans,)  â string labels
    """

    def __init__(self, config: Optional[ServingConfig] = None):
        fn = "ModelServer.__init__"
        self.config = config or ServingConfig()
        self.weights: Optional[np.ndarray] = None
        self.biases: Optional[np.ndarray] = None
        self.plan_names: Optional[np.ndarray] = None
        self._loaded = False
        self.logger = RequestLogger()
        _dbg(fn, f"server created, config={self.config}")

    def load_model(self, path: str) -> None:
        """Load model weights from an .npz file."""
        fn = "ModelServer.load_model"
        _dbg(fn, f"loading model from '{path}'")
        t0 = time.time()
        data = np.load(path, allow_pickle=True)
        self.weights = data["weights"]
        self.biases = data["biases"]
        self.plan_names = data.get("plan_names", np.arange(self.biases.shape[0]))
        self._loaded = True
        elapsed = (time.time() - t0) * 1000
        _dbg(fn, f"loaded in {elapsed:.1f}ms â weights shape={self.weights.shape}, "
                  f"n_plans={len(self.biases)}")

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

    def predict(self, params: np.ndarray) -> int:
        """
        Predict a single plan_id from a feature vector.
        params: (d,) float array
        returns: int plan index
        """
        fn = "ModelServer.predict"
        self._ensure_loaded()
        _dbg(fn, f"params shape={params.shape}")
        t0 = time.time()

        logits = params @ self.weights + self.biases  # (n_plans,)
        plan_id = int(np.argmax(logits))
        latency = (time.time() - t0) * 1000
        self.logger.log_request("predict", latency)
        _dbg(fn, f"plan_id={plan_id}, latency={latency:.2f}ms")
        return plan_id

    def predict_batch(self, params_list: np.ndarray) -> np.ndarray:
        """
        Batch prediction.
        params_list: (n, d)
        returns: (n,) plan indices
        """
        fn = "ModelServer.predict_batch"
        self._ensure_loaded()
        _dbg(fn, f"batch shape={params_list.shape}")
        t0 = time.time()

        n = params_list.shape[0]
        bs = self.config.batch_size
        results = []

        for start in range(0, n, bs):
            end = min(start + bs, n)
            batch = params_list[start:end]
            logits = batch @ self.weights + self.biases  # (bs, n_plans)
            preds = np.argmax(logits, axis=1)
            results.append(preds)
            _dbg(fn, f"micro-batch [{start}:{end}] done")

        all_preds = np.concatenate(results)
        latency = (time.time() - t0) * 1000
        self.logger.log_request("predict_batch", latency)
        _dbg(fn, f"total predictions={len(all_preds)}, latency={latency:.2f}ms")
        return all_preds

    def get_confidence(self, params: np.ndarray) -> Dict[str, float]:
        """
        Return softmax confidence distribution over plans.
        params: (d,)
        returns: dict mapping plan_name -> probability
        """
        fn = "ModelServer.get_confidence"
        self._ensure_loaded()
        _dbg(fn, f"params shape={params.shape}")
        t0 = time.time()

        logits = params @ self.weights + self.biases
        # Numerically stable softmax
        shifted = logits - np.max(logits)
        exp_l = np.exp(shifted)
        probs = exp_l / (np.sum(exp_l) + 1e-12)

        conf = {}
        for i, p in enumerate(probs):
            name = str(self.plan_names[i]) if self.plan_names is not None else str(i)
            conf[name] = float(p)

        latency = (time.time() - t0) * 1000
        self.logger.log_request("get_confidence", latency)
        max_conf = max(conf.values())
        _dbg(fn, f"max_confidence={max_conf:.4f}, "
                  f"above_threshold={max_conf >= self.config.confidence_threshold}, "
                  f"latency={latency:.2f}ms")
        return conf


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _DEBUG = True
    import tempfile, os

    # --- test ServingConfig ---
    cfg = ServingConfig(batch_size=16, timeout_ms=200, confidence_threshold=0.8)
    print(f"Config: {cfg}")

    # --- test QueryParser ---
    sql = "SELECT * FROM plans WHERE cpu_cores >= 4 AND region = 'us-east' AND tier IN ('free','pro')"
    params = QueryParser.parse_sql_params(sql)
    print(f"\nParsed params: {params}")

    metadata = {
        "cpu_cores": {"type": "numeric", "min": 1, "max": 64},
        "region": {"type": "categorical", "categories": ["us-east", "us-west", "eu"]},
    }
    feat = QueryParser.normalize_params(params, metadata)
    print(f"Normalised features: {feat}")

    # --- test ModelServer with synthetic model ---
    d, n_plans = 4, 3
    weights = np.random.randn(d, n_plans).astype(np.float64)
    biases = np.zeros(n_plans, dtype=np.float64)
    plan_names = np.array(["small", "medium", "large"])

    tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
    np.savez(tmp.name, weights=weights, biases=biases, plan_names=plan_names)
    tmp.close()

    server = ModelServer(config=cfg)
    server.load_model(tmp.name)

    x_single = np.random.randn(d)
    pid = server.predict(x_single)
    print(f"\nSingle predict: plan_id={pid}")

    conf = server.get_confidence(x_single)
    print(f"Confidence: {conf}")

    x_batch = np.random.randn(50, d)
    batch_preds = server.predict_batch(x_batch)
    print(f"Batch predict: {batch_preds.shape}, unique plans={np.unique(batch_preds)}")

    # --- test RequestLogger ---
    stats = server.logger.get_stats()
    print(f"\nServer stats: {json.dumps(stats, indent=2, default=str)}")
    counts, edges = server.logger.latency_histogram()
    print(f"Histogram bins={len(counts)}, edge range=[{edges[0]:.2f}, {edges[-1]:.2f}]")

    os.unlink(tmp.name)
    print("\nAll tests passed.")
