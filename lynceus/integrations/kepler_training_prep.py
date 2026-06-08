"""
kepler_training_prep â Training / data-preparation utilities for Lynceus.

Ported from upstream/kepler/code/trainer_util.py (~160 lines).
Algorithm changes (~20%):
  - get_np_type: returns numpy dtype objects, not tf.dtypes
  - cast_columns: in-place column casting with overflow saturation
  - get_sample_weight: Bayesian-smoothed inverse-frequency weights
  - normalize: Welford online mean/var for single-pass numerical stability
  - one_hot: sparse-to-dense with optional smoothing (label smoothing)
  - train_test_split: stratified-aware shuffle split
"""
import os
import numpy as np
from typing import Optional, Tuple, Dict, Union, List

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[kepler_util] {tag}: {items}")


# ââ dtype mapping ââââââââââââââââââââââââââââââââââââââââââââââââ
_NP_TYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8":    np.int8,
    "int16":   np.int16,
    "int32":   np.int32,
    "int64":   np.int64,
    "uint8":   np.uint8,
    "uint16":  np.uint16,
    "uint32":  np.uint32,
    "bool":    np.bool_,
}


def get_np_type(type_str: str) -> np.dtype:
    """Map a string dtype name to a numpy dtype, falling back to float64."""
    dtype = _NP_TYPE_MAP.get(type_str.lower().strip(), np.float64)
    _dbg("get_np_type", requested=type_str, resolved=dtype)
    return np.dtype(dtype)


# ââ Column casting with saturation âââââââââââââââââââââââââââââââ
def cast_columns(data: np.ndarray,
                 col_indices: Union[List[int], None] = None,
                 target_dtype: str = "float32") -> np.ndarray:
    """
    Cast selected columns to *target_dtype* with overflow saturation.

    Values outside the representable range of the target type are clamped
    to its min/max instead of wrapping or raising.
    """
    out = data.copy()
    dt = get_np_type(target_dtype)
    info = np.finfo(dt) if np.issubdtype(dt, np.floating) else np.iinfo(dt)

    if col_indices is None:
        col_indices = list(range(data.shape[1]))

    for c in col_indices:
        col = out[:, c].astype(np.float64)
        col = np.clip(col, float(info.min), float(info.max))
        out[:, c] = col.astype(dt)

    _dbg("cast_columns", n_cols=len(col_indices), dtype=str(dt),
         range=(float(info.min), float(info.max)))
    return out


# ââ Bayesian-smoothed inverse-frequency sample weights âââââââââââ
def get_sample_weight(labels: np.ndarray,
                      *,
                      smooth_alpha: float = 1.0) -> np.ndarray:
    """
    Compute per-sample weights as smoothed inverse class frequency.

    Uses Bayesian Laplace smoothing:  w_c = N_total / (K * (count_c + Î±))
    where K is the number of classes and Î± is the smoothing constant.
    This avoids infinite weights for rare classes and zero weights for
    majority classes.
    """
    labels = np.asarray(labels).ravel()
    classes, counts = np.unique(labels, return_counts=True)
    n = labels.size
    k = classes.size

    freq = {c: cnt for c, cnt in zip(classes, counts)}
    weights = np.empty(n, dtype=np.float64)
    for i, lab in enumerate(labels):
        weights[i] = n / (k * (freq[lab] + smooth_alpha))

    # normalise so mean weight == 1
    weights /= (np.mean(weights) + 1e-12)

    _dbg("get_sample_weight", n=n, k=k, alpha=smooth_alpha,
         w_min=round(float(np.min(weights)), 4),
         w_max=round(float(np.max(weights)), 4))
    return weights


# ââ Welford single-pass normalisation ââââââââââââââââââââââââââââ
class WelfordNormalizer:
    """Online mean/variance tracker using Welford's algorithm."""

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.n: int = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)
        _dbg("WelfordNorm.__init__", n_features=n_features)

    def partial_fit(self, X: np.ndarray) -> "WelfordNormalizer":
        """Incrementally update statistics with a new batch."""
        X = np.asarray(X, dtype=np.float64)
        for row in X:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.m2 += delta * delta2
        _dbg("welford_partial_fit", n=self.n)
        return self

    @property
    def var(self) -> np.ndarray:
        return self.m2 / max(self.n, 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-12)


def normalize(X: np.ndarray,
              *,
              mean: Optional[np.ndarray] = None,
              std: Optional[np.ndarray] = None,
              eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalisation.  If mean/std are None they are computed from X
    using Welford's single-pass algorithm for numerical stability.

    Returns (X_normed, mean, std).
    """
    X = np.asarray(X, dtype=np.float64)

    if mean is None or std is None:
        norm = WelfordNormalizer(X.shape[1])
        norm.partial_fit(X)
        mean = norm.mean
        std = norm.std

    std_safe = np.where(std < eps, 1.0, std)
    X_normed = (X - mean) / std_safe

    _dbg("normalize", shape=X.shape,
         mean_range=(round(float(np.min(mean)), 4), round(float(np.max(mean)), 4)),
         std_range=(round(float(np.min(std)), 4), round(float(np.max(std)), 4)))
    return X_normed, mean, std


# ââ One-hot encoding with optional label smoothing âââââââââââââââ
def one_hot(labels: np.ndarray,
            num_classes: Optional[int] = None,
            *,
            smooth: float = 0.0) -> np.ndarray:
    """
    Convert integer labels to one-hot matrix.

    With smooth > 0, applies label smoothing:
        on-value  = 1 - smooth
        off-value = smooth / (K - 1)
    """
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if num_classes is None:
        num_classes = int(np.max(labels)) + 1

    n = labels.size
    mat = np.zeros((n, num_classes), dtype=np.float64)

    if smooth > 0.0 and num_classes > 1:
        off_val = smooth / (num_classes - 1)
        on_val = 1.0 - smooth
        mat[:] = off_val
        mat[np.arange(n), labels] = on_val
    else:
        mat[np.arange(n), labels] = 1.0

    _dbg("one_hot", n=n, K=num_classes, smooth=smooth,
         row_sums_mean=round(float(np.mean(np.sum(mat, axis=1))), 6))
    return mat


# ââ Stratified train/test split ââââââââââââââââââââââââââââââââââ
def train_test_split(X: np.ndarray,
                     y: np.ndarray,
                     *,
                     test_frac: float = 0.2,
                     seed: int = 42,
                     stratify: bool = False
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Shuffle-split into train/test.  If stratify=True the class proportions
    in y are approximately preserved in both splits.
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]

    if not stratify:
        idx = rng.permutation(n)
        cut = int(n * (1 - test_frac))
        tr, te = idx[:cut], idx[cut:]
    else:
        y_flat = np.asarray(y).ravel()
        tr_list, te_list = [], []
        for cls in np.unique(y_flat):
            cls_idx = np.where(y_flat == cls)[0]
            rng.shuffle(cls_idx)
            cut = max(1, int(len(cls_idx) * (1 - test_frac)))
            tr_list.append(cls_idx[:cut])
            te_list.append(cls_idx[cut:])
        tr = np.concatenate(tr_list)
        te = np.concatenate(te_list)
        rng.shuffle(tr)
        rng.shuffle(te)

    _dbg("train_test_split", n=n, test_frac=test_frac,
         n_train=len(tr), n_test=len(te), stratify=stratify)
    return X[tr], X[te], y[tr], y[te]


# ââ Mini-batch iterator âââââââââââââââââââââââââââââââââââââââââ
def minibatch_iter(X: np.ndarray,
                   y: np.ndarray,
                   *,
                   batch_size: int = 64,
                   shuffle: bool = True,
                   seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Yield (X_batch, y_batch) tuples for one epoch."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    idx = rng.permutation(n) if shuffle else np.arange(n)
    batches = []
    for start in range(0, n, batch_size):
        sel = idx[start:start + batch_size]
        batches.append((X[sel], y[sel]))
    _dbg("minibatch_iter", n=n, batch_size=batch_size,
         n_batches=len(batches), shuffle=shuffle)
    return batches
