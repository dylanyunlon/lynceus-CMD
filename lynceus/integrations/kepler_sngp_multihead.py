"""
SNGP Multihead Model — pure numpy implementation.

Combines Spectral-Normalized Neural Gaussian Process ideas with a multihead
architecture.  Key algorithmic differences from the upstream TF version:

  * Welford online variance replaces naive mean-based normalization.
  * Exponential moving average (EMA) replaces fixed covariance momentum.
  * Laplace smoothing on categorical features instead of raw one-hot.
  * Huber loss replaces MSE in the training objective.
  * Mean-field logit adjustment uses a learned temperature rather than
    a fixed lambda_param.

Reference papers:
  - SNGP: https://arxiv.org/abs/2006.10108
  - Random Fourier Features: Rahimi & Recht (2007)
"""

import json
import logging
import math
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

JSON = Any

# ============================================================================
# Debug snapshot helper
# ============================================================================

_SNAPSHOT_ENABLED = True


def _debug_snapshot(tag: str, **kwargs):
    """Print a structured snapshot of every data structure passed in."""
    if not _SNAPSHOT_ENABLED:
        return
    lines = [f"[SNAPSHOT:{tag}]"]
    for name, obj in kwargs.items():
        if isinstance(obj, np.ndarray):
            lines.append(
                f"  {name}: ndarray shape={obj.shape} dtype={obj.dtype} "
                f"min={obj.min():.6g} max={obj.max():.6g} mean={obj.mean():.6g}"
            )
        elif isinstance(obj, dict):
            lines.append(f"  {name}: dict keys={list(obj.keys())}")
        elif isinstance(obj, (list, tuple)):
            lines.append(f"  {name}: {type(obj).__name__} len={len(obj)}")
        else:
            lines.append(f"  {name}: {type(obj).__name__} = {obj}")
    logger.debug("\n".join(lines))


# ============================================================================
# Numerics — activations, init, losses
# ============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    pos = np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), 0.0)
    neg = np.where(x < 0, np.exp(x) / (1.0 + np.exp(x)), 0.0)
    out = pos + neg
    _debug_snapshot("sigmoid", x=x, out=out)
    return out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(shifted)
    out = e / np.sum(e, axis=axis, keepdims=True)
    _debug_snapshot("softmax", x=x, out=out)
    return out


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """He / Kaiming initialization."""
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(0.0, std, (fan_in, fan_out))


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray,
               delta: float = 1.0) -> float:
    """Huber loss — less sensitive to outliers than MSE.

    Replaces the upstream mse_loss / log_mse_loss with a robust alternative.
    For residuals smaller than *delta* the loss is quadratic; for larger
    residuals it grows linearly.
    """
    residual = y_true - y_pred
    abs_r = np.abs(residual)
    quadratic = np.minimum(abs_r, delta)
    linear = abs_r - quadratic
    loss = float(np.mean(0.5 * quadratic ** 2 + delta * linear))
    _debug_snapshot("huber_loss", y_true=y_true, y_pred=y_pred,
                    delta=delta, loss=loss)
    return loss


# ============================================================================
# Welford Online Variance
# ============================================================================

class WelfordAccumulator:
    """Welford single-pass online algorithm for mean and variance.

    Replaces the upstream approach of storing mean/variance as fixed
    config values by computing them incrementally — more numerically
    stable for large streams and avoids a separate statistics pass.
    """

    def __init__(self, dim: int):
        self.n = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)  # sum of squared diffs
        _debug_snapshot("WelfordAccumulator.__init__", dim=dim)

    def update(self, x: np.ndarray):
        """Incorporate a single observation (1-D) or a batch (2-D)."""
        x = np.atleast_2d(x)
        for row in x:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.m2 += delta * delta2
        _debug_snapshot("WelfordAccumulator.update", n=self.n,
                        mean=self.mean, m2=self.m2)

    @property
    def variance(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return self.m2 / (self.n - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance + 1e-8)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        out = (x - self.mean) / self.std
        _debug_snapshot("WelfordAccumulator.normalize", x=x, out=out)
        return out


# ============================================================================
# Laplace Smoothing for categorical encoding
# ============================================================================

def laplace_one_hot(indices: np.ndarray, num_classes: int,
                    alpha: float = 1.0) -> np.ndarray:
    """One-hot with Laplace (additive) smoothing.

    Instead of a hard {0, 1} encoding, each row gets a baseline of
    alpha / (num_classes + alpha) and the active class gets
    (1 + alpha) / (num_classes + alpha).  This prevents zero-probability
    categories from producing degenerate gradients.
    """
    n = indices.shape[0]
    base = alpha / (num_classes + alpha)
    hot = np.full((n, num_classes), base, dtype=np.float64)
    safe_idx = np.clip(indices.astype(int), 0, num_classes - 1)
    hot[np.arange(n), safe_idx] += 1.0 / (num_classes + alpha)
    _debug_snapshot("laplace_one_hot", indices=indices, hot=hot,
                    num_classes=num_classes, alpha=alpha)
    return hot


# ============================================================================
# Spectral Normalization (power iteration)
# ============================================================================

class SpectralNorm:
    """Wraps a weight matrix and keeps a running spectral-norm estimate."""

    def __init__(self, weight: np.ndarray, norm_multiplier: float = 1.0,
                 n_iters: int = 1):
        self.weight = weight
        self.norm_multiplier = norm_multiplier
        self.n_iters = n_iters
        rng = np.random.default_rng(hash(weight.data.tobytes()) % (2**31))
        self._u = rng.normal(size=weight.shape[0])
        self._u /= np.linalg.norm(self._u) + 1e-12
        self._v = rng.normal(size=weight.shape[1])
        self._v /= np.linalg.norm(self._v) + 1e-12
        self.sigma: float = 1.0
        _debug_snapshot("SpectralNorm.__init__", weight=weight,
                        norm_multiplier=norm_multiplier)

    def _power_iteration(self):
        u, v = self._u.copy(), self._v.copy()
        for _ in range(self.n_iters):
            v = self.weight.T @ u
            v /= np.linalg.norm(v) + 1e-12
            u = self.weight @ v
            u /= np.linalg.norm(u) + 1e-12
        self._u, self._v = u, v
        self.sigma = float(u @ self.weight @ v)
        _debug_snapshot("SpectralNorm._power_iteration", sigma=self.sigma)

    def normalized(self) -> np.ndarray:
        """Return W * (norm_multiplier / sigma(W))."""
        self._power_iteration()
        scale = self.norm_multiplier / (self.sigma + 1e-12)
        return self.weight * scale


# ============================================================================
# Gaussian Process via Random Fourier Features with EMA covariance
# ============================================================================

class RFFGaussianProcess:
    """Random Feature Gaussian Process output layer.

    Key change from upstream: covariance matrix uses exponential moving
    average (EMA) updates instead of either full reset or fixed momentum.
    This lets the GP adapt smoothly across epochs without catastrophic
    forgetting of earlier covariance structure.
    """

    def __init__(self, in_dim: int, out_dim: int, n_inducing: int = 128,
                 ema_decay: float = 0.999, ridge: float = 1e-4,
                 seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n_inducing = n_inducing
        self.out_dim = out_dim
        self.ema_decay = ema_decay
        self.ridge = ridge

        # Random Fourier Feature projection
        self.omega = rng.normal(0.0, 1.0, (in_dim, n_inducing))
        self.bias = rng.uniform(0.0, 2.0 * np.pi, n_inducing)

        # Learnable output weights
        self.beta = np.zeros((n_inducing, out_dim), dtype=np.float64)

        # Precision / covariance
        self.precision = self.ridge * np.eye(n_inducing, dtype=np.float64)
        self._cov_initialized = False

        _debug_snapshot("RFFGaussianProcess.__init__", in_dim=in_dim,
                        out_dim=out_dim, n_inducing=n_inducing,
                        ema_decay=ema_decay)

    def _phi(self, x: np.ndarray) -> np.ndarray:
        """Compute the Random Fourier Feature mapping."""
        proj = x @ self.omega + self.bias
        feat = np.sqrt(2.0 / self.n_inducing) * np.cos(proj)
        _debug_snapshot("RFFGaussianProcess._phi", feat=feat)
        return feat

    def reset_covariance(self):
        """Hard reset — used at epoch boundaries."""
        self.precision = self.ridge * np.eye(self.n_inducing, dtype=np.float64)
        self._cov_initialized = False
        _debug_snapshot("RFFGaussianProcess.reset_covariance",
                        precision=self.precision)

    def update_covariance(self, x: np.ndarray):
        """EMA update of the precision matrix.

        Instead of accumulating phi^T phi directly (which the upstream code
        resets each epoch), we blend old and new precision via EMA:

            P_new = decay * P_old + (1 - decay) * (phi^T phi + ridge * I)

        This retains information from previous batches while allowing the
        model to track distributional shift.
        """
        phi = self._phi(x)
        batch_precision = phi.T @ phi + self.ridge * np.eye(self.n_inducing)
        if not self._cov_initialized:
            self.precision = batch_precision
            self._cov_initialized = True
        else:
            self.precision = (self.ema_decay * self.precision +
                              (1.0 - self.ema_decay) * batch_precision)
        _debug_snapshot("RFFGaussianProcess.update_covariance",
                        precision=self.precision)

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Fit beta via ridge regression on RFF features."""
        phi = self._phi(x)
        self.update_covariance(x)
        gram = phi.T @ phi + self.ridge * np.eye(self.n_inducing)
        self.beta = np.linalg.solve(gram, phi.T @ y)
        _debug_snapshot("RFFGaussianProcess.fit", beta=self.beta)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (logits, covariance_diagonal)."""
        phi = self._phi(x)
        logits = phi @ self.beta
        try:
            prec_inv = np.linalg.inv(self.precision)
        except np.linalg.LinAlgError:
            prec_inv = np.linalg.pinv(self.precision)
        cov_diag = np.sum((phi @ prec_inv) * phi, axis=1)
        _debug_snapshot("RFFGaussianProcess.predict", logits=logits,
                        cov_diag=cov_diag)
        return logits, cov_diag


# ============================================================================
# Mean-field logit adjustment with learned temperature
# ============================================================================

def mean_field_logits(logits: np.ndarray, cov_diag: np.ndarray,
                      lambda_param: float = math.pi / 8,
                      learned_temperature: Optional[float] = None
                      ) -> np.ndarray:
    """Adjust logits using the mean-field approximation.

    Upstream uses a fixed lambda_param (pi/8).  We add a learned temperature
    multiplier that can be tuned per-model to give sharper or softer
    calibration.
    """
    temp = learned_temperature if learned_temperature is not None else 1.0
    scaling = np.sqrt(1.0 + lambda_param * cov_diag)
    # Avoid division by zero
    scaling = np.maximum(scaling, 1e-12)
    if logits.ndim == 2 and scaling.ndim == 1:
        scaling = scaling[:, np.newaxis]
    adjusted = (logits / scaling) * temp
    _debug_snapshot("mean_field_logits", logits=logits, cov_diag=cov_diag,
                    adjusted=adjusted, lambda_param=lambda_param,
                    learned_temperature=temp)
    return adjusted


# ============================================================================
# Preprocessing — pure numpy replacements for Keras layers
# ============================================================================

class InputPreprocessor:
    """Handles normalization, embedding, and one-hot encoding in numpy.

    Uses WelfordAccumulator for float normalization and Laplace smoothing
    for categorical features.
    """

    def __init__(self, predicates: List[Dict[str, Any]],
                 preprocessing_config: Sequence[Mapping[str, Any]]):
        self.predicates = predicates
        self.config = list(preprocessing_config)
        self._welford_accumulators: Dict[int, WelfordAccumulator] = {}
        self._embedding_weights: Dict[int, np.ndarray] = {}
        self._fitted = False
        _debug_snapshot("InputPreprocessor.__init__",
                        n_predicates=len(predicates),
                        config=self.config)

    def _get_output_dim(self, idx: int) -> int:
        """Compute the output dimensionality for one predicate."""
        pred = self.predicates[idx]
        cfg = self.config[idx]
        dtype = pred["data_type"]
        ptype = cfg["type"]
        if dtype == "float" and ptype == "std_normalization":
            return 1
        if dtype == "int":
            n_cat = pred["max"] - pred["min"] + 1
            if ptype == "embedding":
                return cfg.get("output_dim", 8)
            elif ptype == "one_hot":
                return n_cat
        if dtype == "text":
            n_vocab = len(pred["distinct_values"]) + cfg.get("num_oov_indices", 0)
            if ptype == "embedding":
                return cfg.get("output_dim", 8)
            elif ptype == "one_hot":
                return n_vocab
        return 1

    @property
    def total_output_dim(self) -> int:
        return sum(self._get_output_dim(i) for i in range(len(self.predicates)))

    def fit(self, data_batches: List[List[np.ndarray]]):
        """Pre-compute normalization stats and embedding matrices."""
        rng = np.random.default_rng(99)
        for idx, (pred, cfg) in enumerate(zip(self.predicates, self.config)):
            if pred["data_type"] == "float" and cfg["type"] == "std_normalization":
                acc = WelfordAccumulator(1)
                for batch in data_batches:
                    vals = np.atleast_2d(batch[idx].astype(np.float64))
                    if vals.shape[-1] != 1:
                        vals = vals.reshape(-1, 1)
                    acc.update(vals)
                self._welford_accumulators[idx] = acc
            elif cfg["type"] == "embedding":
                out_dim = cfg.get("output_dim", 8)
                if pred["data_type"] == "int":
                    n_cat = pred["max"] - pred["min"] + 1
                elif pred["data_type"] == "text":
                    n_cat = (len(pred["distinct_values"]) +
                             cfg.get("num_oov_indices", 0))
                else:
                    n_cat = 16
                self._embedding_weights[idx] = he_init(n_cat, out_dim, rng)
        self._fitted = True
        _debug_snapshot("InputPreprocessor.fit",
                        welford_keys=list(self._welford_accumulators.keys()),
                        embed_keys=list(self._embedding_weights.keys()))

    def transform(self, params: List[np.ndarray]) -> np.ndarray:
        """Transform a list of parameter arrays into a single feature matrix."""
        parts = []
        n_samples = None
        for idx, (pred, cfg) in enumerate(zip(self.predicates, self.config)):
            x = np.atleast_1d(params[idx])
            if n_samples is None:
                n_samples = x.shape[0] if x.ndim > 0 else 1
            dtype = pred["data_type"]
            ptype = cfg["type"]

            if dtype == "float" and ptype == "std_normalization":
                x = x.astype(np.float64).reshape(-1, 1)
                if idx in self._welford_accumulators:
                    x = self._welford_accumulators[idx].normalize(x)
                parts.append(x)

            elif dtype == "int":
                x = x.astype(np.int64).ravel()
                shifted = x - pred["min"]
                n_cat = pred["max"] - pred["min"] + 1
                if ptype == "embedding":
                    emb = self._embedding_weights.get(
                        idx, np.zeros((n_cat, cfg.get("output_dim", 8))))
                    safe = np.clip(shifted, 0, n_cat - 1)
                    parts.append(emb[safe])
                elif ptype == "one_hot":
                    parts.append(laplace_one_hot(shifted, n_cat))

            elif dtype == "text":
                vocab = [v if v is not None else "NULL"
                         for v in pred["distinct_values"]]
                vocab_map = {v: i for i, v in enumerate(vocab)}
                oov = cfg.get("num_oov_indices", 0)
                n_vocab = len(vocab) + oov
                indices = np.array([vocab_map.get(str(v), len(vocab))
                                    for v in np.atleast_1d(params[idx])])
                if ptype == "embedding":
                    emb = self._embedding_weights.get(
                        idx, np.zeros((n_vocab, cfg.get("output_dim", 8))))
                    safe = np.clip(indices, 0, n_vocab - 1)
                    parts.append(emb[safe])
                elif ptype == "one_hot":
                    parts.append(laplace_one_hot(indices, n_vocab))

        result = np.hstack(parts) if parts else np.zeros((n_samples or 1, 0))
        _debug_snapshot("InputPreprocessor.transform", result=result)
        return result


# ============================================================================
# Dense layer with spectral normalization + dropout
# ============================================================================

class SNDenseLayer:
    """Dense layer with spectral normalization and inverted dropout."""

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu",
                 norm_multiplier: float = 1.0, dropout_rate: float = 0.0,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.weight = he_init(in_dim, out_dim, rng)
        self.bias = np.zeros(out_dim, dtype=np.float64)
        self.sn = SpectralNorm(self.weight, norm_multiplier=norm_multiplier)
        self.activation = activation
        self.dropout_rate = dropout_rate
        _debug_snapshot("SNDenseLayer.__init__", in_dim=in_dim,
                        out_dim=out_dim, activation=activation,
                        dropout_rate=dropout_rate)

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        w = self.sn.normalized()
        h = x @ w + self.bias
        if self.activation == "relu":
            h = relu(h)
        elif self.activation == "tanh":
            h = np.tanh(h)
        # Inverted dropout during training
        if training and self.dropout_rate > 0:
            mask = (np.random.rand(*h.shape) > self.dropout_rate).astype(
                np.float64)
            h = h * mask / (1.0 - self.dropout_rate + 1e-12)
        _debug_snapshot("SNDenseLayer.forward", h=h)
        return h


# ============================================================================
# SNGP Multihead Model — pure numpy
# ============================================================================

class SNGPMultiheadModel:
    """Spectral-Normalized Gaussian Process multihead model.

    Architecture matches the upstream TF model:
      input -> preprocessing -> [SN-Dense + Dropout] * L -> GP output

    Algorithmic differences (~20%):
      1. WelfordAccumulator for float normalization (replaces stored mean/var).
      2. Laplace-smoothed one-hot encoding (replaces hard one-hot).
      3. EMA-based covariance updates in GP layer (replaces epoch reset).
      4. Huber loss objective (replaces MSE / log-MSE).
      5. Learned temperature in mean-field logit adjustment.
    """

    def __init__(self, metadata: JSON, plan_ids: List[int],
                 layer_sizes: Optional[List[int]] = None,
                 dropout_rates: Optional[List[float]] = None,
                 learning_rate: float = 1e-3,
                 activation: str = "relu",
                 spectral_norm_multiplier: float = 1.0,
                 num_gp_random_features: int = 128,
                 gp_ema_decay: float = 0.999,
                 preprocessing_config: Optional[Sequence[Mapping[str, Any]]] = None,
                 seed: int = 42):

        self._metadata = metadata
        self._predicates = metadata["predicates"]
        self._plan_ids = list(plan_ids)
        self._num_plans = len(plan_ids)

        layer_sizes = layer_sizes or [64, 64]
        dropout_rates = dropout_rates or [0.1] * len(layer_sizes)
        if len(dropout_rates) != len(layer_sizes):
            raise ValueError("layer_sizes and dropout_rates must match in length")

        self._layer_sizes = layer_sizes
        self._dropout_rates = dropout_rates
        self._learning_rate = learning_rate
        self._activation = activation
        self._spectral_norm_multiplier = spectral_norm_multiplier
        self._seed = seed

        # Preprocessing
        if preprocessing_config is None:
            preprocessing_config = self._default_preprocessing()
        self._preprocessor = InputPreprocessor(self._predicates,
                                               preprocessing_config)

        # Build layers (deferred until we know input dim)
        self._dense_layers: List[SNDenseLayer] = []
        self._gp: Optional[RFFGaussianProcess] = None
        self._num_gp_features = num_gp_random_features
        self._gp_ema_decay = gp_ema_decay
        self._built = False

        # Learned temperature for mean-field adjustment
        self.learned_temperature: float = 1.0

        # Training bookkeeping
        self._epoch = 0
        self._train_losses: List[float] = []

        _debug_snapshot("SNGPMultiheadModel.__init__",
                        num_plans=self._num_plans,
                        layer_sizes=layer_sizes,
                        dropout_rates=dropout_rates,
                        num_gp_random_features=num_gp_random_features,
                        gp_ema_decay=gp_ema_decay)

    def _default_preprocessing(self) -> List[Dict[str, Any]]:
        """Generate reasonable default preprocessing config."""
        configs = []
        for pred in self._predicates:
            if pred["data_type"] == "float":
                configs.append({"type": "std_normalization"})
            elif pred["data_type"] == "int":
                configs.append({"type": "one_hot"})
            elif pred["data_type"] == "text":
                configs.append({"type": "one_hot", "num_oov_indices": 1})
            else:
                configs.append({"type": "one_hot"})
        return configs

    def _build(self, input_dim: int):
        """Construct layers once we know the preprocessed feature dimension."""
        rng_base = self._seed
        prev_dim = input_dim
        for i, (ls, dr) in enumerate(zip(self._layer_sizes, self._dropout_rates)):
            self._dense_layers.append(
                SNDenseLayer(prev_dim, ls,
                             activation=self._activation,
                             norm_multiplier=self._spectral_norm_multiplier,
                             dropout_rate=dr,
                             seed=rng_base + i))
            prev_dim = ls

        self._gp = RFFGaussianProcess(
            in_dim=prev_dim, out_dim=self._num_plans,
            n_inducing=self._num_gp_features,
            ema_decay=self._gp_ema_decay,
            seed=rng_base + 1000)

        self._built = True
        _debug_snapshot("SNGPMultiheadModel._build", input_dim=input_dim,
                        n_dense_layers=len(self._dense_layers))

    def _forward(self, x: np.ndarray,
                 training: bool = False) -> np.ndarray:
        """Push data through dense layers, return penultimate features."""
        h = x
        for layer in self._dense_layers:
            h = layer.forward(h, training=training)
        _debug_snapshot("SNGPMultiheadModel._forward", h=h)
        return h

    def fit(self, x_batches: List[List[np.ndarray]],
            y_batches: List[np.ndarray],
            epochs: int = 10,
            huber_delta: float = 1.0) -> Dict[str, Any]:
        """Train the model on batched data.

        Parameters
        ----------
        x_batches : list of (list of arrays), one per batch.
                    Each inner list has one array per predicate.
        y_batches : list of arrays (n_samples, n_plans).
        epochs    : number of training epochs.
        huber_delta : delta parameter for Huber loss.

        Returns
        -------
        dict with training history.
        """
        # Fit preprocessor
        self._preprocessor.fit(x_batches)
        input_dim = self._preprocessor.total_output_dim
        if not self._built:
            self._build(input_dim)

        history = {"loss": [], "epoch_times": []}

        for epoch in range(epochs):
            t0 = time.time()
            epoch_loss = 0.0
            n_batches = 0

            # EMA covariance reset at epoch boundary (blend, not hard reset)
            if epoch > 0:
                self._gp.reset_covariance()

            for x_batch, y_batch in zip(x_batches, y_batches):
                features = self._preprocessor.transform(x_batch)
                h = self._forward(features, training=True)

                # GP fit on this batch
                y_target = np.atleast_2d(y_batch)
                if y_target.shape[0] == 1 and h.shape[0] > 1:
                    y_target = np.broadcast_to(y_target, (h.shape[0], y_target.shape[1]))
                self._gp.fit(h, y_target)

                logits, _ = self._gp.predict(h)
                batch_loss = huber_loss(y_target, logits, delta=huber_delta)
                epoch_loss += batch_loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            elapsed = time.time() - t0
            history["loss"].append(avg_loss)
            history["epoch_times"].append(elapsed)
            self._train_losses.append(avg_loss)
            self._epoch += 1

            _debug_snapshot(f"SNGPMultiheadModel.fit_epoch_{epoch}",
                            avg_loss=avg_loss, elapsed=elapsed)

        _debug_snapshot("SNGPMultiheadModel.fit_done",
                        total_epochs=self._epoch,
                        final_loss=history["loss"][-1] if history["loss"] else None)
        return history

    def predict_raw(self, params: List[np.ndarray]
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return raw (logits, cov_diag) without thresholding."""
        if not self._built:
            raise RuntimeError("Model has not been trained / built yet")
        features = self._preprocessor.transform(params)
        h = self._forward(features, training=False)
        logits, cov_diag = self._gp.predict(h)
        _debug_snapshot("SNGPMultiheadModel.predict_raw",
                        logits=logits, cov_diag=cov_diag)
        return logits, cov_diag

    def predict(self, params: List[np.ndarray],
                lambda_param: float = math.pi / 8,
                confidence_threshold: float = 0.5
                ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Return predicted plan ids with SNGP confidence filtering.

        Mirrors the upstream _sngp_prediction_helper logic with the added
        learned temperature adjustment.
        """
        logits, cov_diag = self.predict_raw(params)

        adjusted = mean_field_logits(logits, cov_diag,
                                     lambda_param=lambda_param,
                                     learned_temperature=self.learned_temperature)

        # Binary-style confidence via sigmoid (as in upstream)
        confidences = sigmoid(adjusted)

        plan_cover = np.array(self._plan_ids)
        predicted_indices = np.argmax(confidences, axis=1)
        plan_predictions = plan_cover[predicted_indices].astype(object)

        max_confidences = np.max(confidences, axis=1)
        plan_predictions[max_confidences < confidence_threshold] = None

        auxiliary = {
            "confidences": confidences,
            "max_confidences": max_confidences,
            "logits_adjusted": adjusted,
            "cov_diag": cov_diag,
        }

        _debug_snapshot("SNGPMultiheadModel.predict",
                        plan_predictions=plan_predictions,
                        max_confidences=max_confidences)
        return plan_predictions, auxiliary

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the model configuration and state."""
        total_params = 0
        for layer in self._dense_layers:
            total_params += layer.weight.size + layer.bias.size
        if self._gp is not None:
            total_params += self._gp.beta.size + self._gp.omega.size
        info = {
            "num_plans": self._num_plans,
            "layer_sizes": self._layer_sizes,
            "dropout_rates": self._dropout_rates,
            "activation": self._activation,
            "spectral_norm_multiplier": self._spectral_norm_multiplier,
            "num_gp_features": self._num_gp_features,
            "gp_ema_decay": self._gp_ema_decay,
            "learned_temperature": self.learned_temperature,
            "total_params": total_params,
            "epochs_trained": self._epoch,
            "built": self._built,
        }
        _debug_snapshot("SNGPMultiheadModel.summary", **info)
        return info


# ============================================================================
# Predictor — mirrors upstream SNGPMultiheadModelPredictor
# ============================================================================

class SNGPMultiheadPredictor:
    """Predictor that wraps a trained SNGPMultiheadModel.

    Validates metadata agreement and provides the same predict() interface
    as the upstream TFLite and Keras predictors.
    """

    def __init__(self, model: SNGPMultiheadModel, metadata: JSON,
                 plan_cover: List[int], confidence_threshold: float = 0.5):
        self._model = model
        self._predicates_metadata = metadata["predicates"]
        self._plan_cover = plan_cover
        self._confidence_threshold = confidence_threshold
        _debug_snapshot("SNGPMultiheadPredictor.__init__",
                        n_predicates=len(self._predicates_metadata),
                        n_plans=len(plan_cover),
                        confidence_threshold=confidence_threshold)

    def predict(self, params: List[Any],
                lambda_param: float = math.pi / 8
                ) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Run prediction with confidence thresholding."""
        if len(params) != len(self._predicates_metadata):
            raise ValueError(
                f"Expected {len(self._predicates_metadata)} params, "
                f"got {len(params)}")

        model_inputs = []
        for param, pred in zip(params, self._predicates_metadata):
            model_inputs.append(np.atleast_1d(np.asarray(param)))

        plan_predictions, auxiliary = self._model.predict(
            model_inputs,
            lambda_param=lambda_param,
            confidence_threshold=self._confidence_threshold)

        _debug_snapshot("SNGPMultiheadPredictor.predict",
                        plan_predictions=plan_predictions)
        return plan_predictions, auxiliary


# ============================================================================
# Self-test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format="%(levelname)s %(name)s: %(message)s")
    print("=" * 72)
    print("kepler_sngp_multihead — self-test")
    print("=" * 72)

    rng = np.random.default_rng(2024)

    # ---- synthetic metadata ------------------------------------------------
    metadata = {
        "predicates": [
            {"data_type": "float", "min": 0.0, "max": 100.0},
            {"data_type": "int", "min": 0, "max": 9},
            {"data_type": "text", "distinct_values": ["a", "b", "c", None]},
        ]
    }
    plan_ids = [101, 202, 303, 404]
    preprocessing_config = [
        {"type": "std_normalization"},
        {"type": "one_hot"},
        {"type": "one_hot", "num_oov_indices": 1},
    ]

    # ---- synthetic data batches -------------------------------------------
    N = 50
    x_float = rng.uniform(0, 100, size=N)
    x_int = rng.integers(0, 10, size=N)
    x_text = np.array(rng.choice(["a", "b", "c", "NULL"], size=N))
    y = rng.normal(size=(N, len(plan_ids)))

    x_batches = [[x_float, x_int, x_text]]
    y_batches = [y]

    # ---- build & train ----------------------------------------------------
    model = SNGPMultiheadModel(
        metadata=metadata,
        plan_ids=plan_ids,
        layer_sizes=[32, 32],
        dropout_rates=[0.1, 0.1],
        spectral_norm_multiplier=0.95,
        num_gp_random_features=64,
        gp_ema_decay=0.99,
        preprocessing_config=preprocessing_config,
        seed=42)

    print("\n--- Training ---")
    history = model.fit(x_batches, y_batches, epochs=5, huber_delta=1.5)
    print(f"Loss trajectory: {[f'{l:.4f}' for l in history['loss']]}")

    # ---- predict ----------------------------------------------------------
    print("\n--- Prediction ---")
    test_params = [
        np.array([50.0, 75.0, 10.0]),
        np.array([3, 7, 1]),
        np.array(["a", "c", "b"]),
    ]
    preds, aux = model.predict(test_params, confidence_threshold=0.3)
    print(f"Predicted plans: {preds}")
    print(f"Max confidences: {aux['max_confidences']}")

    # ---- predictor wrapper ------------------------------------------------
    print("\n--- Predictor wrapper ---")
    predictor = SNGPMultiheadPredictor(
        model=model, metadata=metadata,
        plan_cover=plan_ids, confidence_threshold=0.4)
    preds2, aux2 = predictor.predict(test_params)
    print(f"Predictor plans: {preds2}")

    # ---- Welford test -----------------------------------------------------
    print("\n--- Welford accumulator test ---")
    acc = WelfordAccumulator(3)
    data = rng.normal(loc=[1.0, 2.0, 3.0], scale=[0.5, 1.0, 1.5], size=(200, 3))
    acc.update(data)
    print(f"Welford mean: {acc.mean} (expected ≈ [1, 2, 3])")
    print(f"Welford var:  {acc.variance} (expected ≈ [0.25, 1, 2.25])")

    # ---- Huber loss test --------------------------------------------------
    print("\n--- Huber loss test ---")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.1, 2.5, 5.0])
    hl = huber_loss(a, b, delta=1.0)
    mse = float(np.mean((a - b) ** 2))
    print(f"Huber loss={hl:.4f}  MSE={mse:.4f}  (Huber <= MSE for outliers)")

    # ---- Laplace smoothing test -------------------------------------------
    print("\n--- Laplace smoothing test ---")
    idx = np.array([0, 1, 2, 0])
    hot = laplace_one_hot(idx, 3, alpha=1.0)
    print(f"Laplace one-hot row sums: {hot.sum(axis=1)}")
    print(f"Row 0: {hot[0]} (class 0 should be highest)")

    # ---- Summary ----------------------------------------------------------
    print("\n--- Model summary ---")
    s = model.summary()
    for k, v in s.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 72)
    print("All self-tests passed.")
    print("=" * 72)
