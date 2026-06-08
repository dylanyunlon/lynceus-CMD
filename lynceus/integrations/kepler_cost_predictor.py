"""
M196: kepler_cost_predictor — MLP Query Cost Predictor with Online Learning
Upstream: kepler model training infrastructure (~400L across multiple files)
Algorithm changes (20%):
  - Xavier/Glorot initialization (upstream: random)
  - GELU activation (upstream: ReLU)
  - Online SGD with momentum for incremental learning
  - Welford stats for loss tracking
"""
import math
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)
_DBG = True
def _dbg(tag, **kw):
    if _DBG: print(f"  [dbg:{tag}] { {k: repr(v)[:80] for k,v in kw.items()} }")


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit activation."""
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x: np.ndarray) -> np.ndarray:
    """Approximate GELU derivative."""
    cdf = 0.5 * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
    pdf = np.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    return cdf + x * pdf


class WelfordLoss:
    __slots__ = ("_n", "_mean", "_m2")
    def __init__(self):
        self._n = 0; self._mean = 0.0; self._m2 = 0.0
    def update(self, v):
        self._n += 1
        d = v - self._mean
        self._mean += d / self._n
        self._m2 += d * (v - self._mean)
    @property
    def mean(self): return self._mean
    @property
    def std(self): return math.sqrt(self._m2/self._n) if self._n > 1 else 0.0
    def snapshot(self): return {"n": self._n, "mean": round(self._mean,6), "std": round(self.std,6)}


class FeatureExtractor:
    """Extract feature vector from query plan metadata."""
    
    FEATURE_NAMES = [
        "num_tables", "num_joins", "estimated_rows_log", "num_predicates",
        "has_group_by", "has_order_by", "has_limit", "has_subquery",
        "max_index_depth", "selectivity_product_log", "total_columns",
        "estimated_cost_log", "join_types_encoded", "access_types_encoded",
        "has_aggregation", "num_sorts",
    ]
    
    def extract(self, plan: Dict[str, Any]) -> np.ndarray:
        features = np.zeros(len(self.FEATURE_NAMES))
        features[0] = plan.get("num_tables", 1)
        features[1] = plan.get("num_joins", 0)
        features[2] = math.log1p(plan.get("estimated_rows", 1))
        features[3] = plan.get("num_predicates", 0)
        features[4] = float(plan.get("has_group_by", False))
        features[5] = float(plan.get("has_order_by", False))
        features[6] = float(plan.get("has_limit", False))
        features[7] = float(plan.get("has_subquery", False))
        features[8] = plan.get("max_index_depth", 0)
        features[9] = math.log1p(max(plan.get("selectivity_product", 0), 1e-12))
        features[10] = plan.get("total_columns", 1)
        features[11] = math.log1p(plan.get("estimated_cost", 1))
        features[12] = {"nested_loop": 0, "hash_join": 1, "merge_join": 2}.get(
            plan.get("join_type", "nested_loop"), 0)
        features[13] = {"ALL": 0, "index": 1, "range": 2, "ref": 3, "eq_ref": 4, "const": 5}.get(
            plan.get("access_type", "ALL"), 0)
        features[14] = float(plan.get("has_aggregation", False))
        features[15] = plan.get("num_sorts", 0)
        _dbg("extract", n_features=len(features), nonzero=int(np.count_nonzero(features)))
        return features
    
    @property
    def dim(self) -> int:
        return len(self.FEATURE_NAMES)


class CostPredictor:
    """2-layer MLP for query cost prediction with online learning."""
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, 
                 lr: float = 0.001, momentum: float = 0.9):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.momentum = momentum
        
        # Xavier initialization
        scale1 = math.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = math.sqrt(2.0 / (hidden_dim + 1))
        
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, 1) * scale2
        self.b2 = np.zeros(1)
        
        # Momentum buffers
        self._vW1 = np.zeros_like(self.W1)
        self._vb1 = np.zeros_like(self.b1)
        self._vW2 = np.zeros_like(self.W2)
        self._vb2 = np.zeros_like(self.b2)
        
        # Feature normalization (online mean/std)
        self._feat_mean = np.zeros(input_dim)
        self._feat_var = np.ones(input_dim)
        self._feat_n = 0
        
        self._loss_tracker = WelfordLoss()
        self._train_count = 0
        _dbg("CostPredictor.__init__", input=input_dim, hidden=hidden_dim)
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._feat_mean) / np.sqrt(self._feat_var + 1e-8)
    
    def _update_norm_stats(self, x: np.ndarray):
        self._feat_n += 1
        delta = x - self._feat_mean
        self._feat_mean += delta / self._feat_n
        delta2 = x - self._feat_mean
        self._feat_var += (delta * delta2 - self._feat_var) / self._feat_n
    
    def predict(self, x: np.ndarray) -> float:
        xn = self._normalize(x)
        h = gelu(xn @ self.W1 + self.b1)
        out = float((h @ self.W2 + self.b2)[0])
        return max(out, 0.0)  # cost is non-negative
    
    def train_step(self, x: np.ndarray, target: float) -> float:
        """Single online SGD step with momentum. Returns loss."""
        self._update_norm_stats(x)
        xn = self._normalize(x).reshape(1, -1)
        
        # Forward
        z1 = xn @ self.W1 + self.b1
        h = gelu(z1)
        out = (h @ self.W2 + self.b2)[0, 0]
        
        loss = (out - target) ** 2
        
        # Backward
        d_out = 2 * (out - target)
        dW2 = h.T * d_out
        db2 = np.array([d_out])
        dh = (self.W2 * d_out).T
        dz1 = dh * gelu_derivative(z1)
        dW1 = xn.T @ dz1
        db1 = dz1[0]
        
        # Gradient clipping
        max_norm = 1.0
        for g in [dW1, db1, dW2, db2]:
            gnorm = np.linalg.norm(g)
            if gnorm > max_norm:
                g *= max_norm / gnorm
        
        # SGD with momentum
        self._vW1 = self.momentum * self._vW1 - self.lr * dW1
        self._vb1 = self.momentum * self._vb1 - self.lr * db1
        self._vW2 = self.momentum * self._vW2 - self.lr * dW2
        self._vb2 = self.momentum * self._vb2 - self.lr * db2
        
        self.W1 += self._vW1
        self.b1 += self._vb1
        self.W2 += self._vW2
        self.b2 += self._vb2
        
        self._loss_tracker.update(loss)
        self._train_count += 1
        
        if self._train_count % 100 == 0:
            _dbg("train_step", n=self._train_count, loss=round(loss,4),
                 mean_loss=round(self._loss_tracker.mean,4))
        
        return loss
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        preds = np.array([self.predict(X[i]) for i in range(len(X))])
        mse = np.mean((preds - y) ** 2)
        mae = np.mean(np.abs(preds - y))
        mape = np.mean(np.abs((preds - y) / np.maximum(y, 1e-6))) * 100
        r2 = 1 - np.sum((y - preds)**2) / max(np.sum((y - np.mean(y))**2), 1e-12)
        return {"mse": float(mse), "mae": float(mae), "mape": float(mape), "r2": float(r2)}
    
    def _debug_snapshot(self) -> Dict[str, Any]:
        return {
            "input_dim": self.input_dim, "hidden_dim": self.hidden_dim,
            "train_count": self._train_count, "loss": self._loss_tracker.snapshot(),
            "W1_norm": round(float(np.linalg.norm(self.W1)), 4),
            "W2_norm": round(float(np.linalg.norm(self.W2)), 4),
        }


if __name__ == "__main__":
    print("=== M196 kepler_cost_predictor self-test ===")
    
    fe = FeatureExtractor()
    plan = {"num_tables": 3, "num_joins": 2, "estimated_rows": 50000,
            "num_predicates": 4, "has_order_by": True, "access_type": "range",
            "estimated_cost": 1200, "total_columns": 12}
    features = fe.extract(plan)
    assert len(features) == 16
    
    # Test predictor
    pred = CostPredictor(input_dim=16, hidden_dim=24, lr=0.01)
    
    # Generate synthetic training data
    rng = np.random.RandomState(42)
    X_train = rng.randn(500, 16)
    # Target: nonlinear function of features
    y_train = np.abs(X_train[:, 0] * 100 + X_train[:, 2] * 50 + 
                     X_train[:, 0] * X_train[:, 2] * 20 + rng.randn(500) * 10)
    
    # Train
    for epoch in range(3):
        for i in range(len(X_train)):
            pred.train_step(X_train[i], y_train[i])
    
    # Evaluate
    metrics = pred.evaluate(X_train[:50], y_train[:50])
    snap = pred._debug_snapshot()
    
    print(f"  Train count: {snap['train_count']}")
    print(f"  Loss: {snap['loss']}")
    print(f"  Eval: MSE={metrics['mse']:.1f}, MAE={metrics['mae']:.1f}, R2={metrics['r2']:.3f}")
    assert snap["train_count"] == 1500
    print("  All tests passed!")
    print(f"  Lines: {sum(1 for _ in open(__file__))}")
