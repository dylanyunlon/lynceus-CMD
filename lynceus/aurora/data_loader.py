"""
Data loading utilities for D2STGNN Aurora spatio-temporal traffic forecasting.
Pure NumPy implementation.
"""

import numpy as np
import os

_DEBUG = os.environ.get("AURORA_DEBUG", "0") == "1"


def _dbg(*args, **kwargs):
    if _DEBUG:
        print("[DEBUG data_loader]", *args, **kwargs)


class StandardScaler:
    """Per-feature z-score normalization."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self._fitted = False

    def fit(self, data):
        """Fit scaler on data of shape (..., C) or (N_samples, T, N, C)."""
        axes = tuple(range(data.ndim - 1))
        self.mean_ = np.mean(data, axis=axes, keepdims=False)
        self.std_ = np.std(data, axis=axes, keepdims=False)
        self.std_ = np.where(self.std_ < 1e-8, 1.0, self.std_)
        self._fitted = True
        _dbg(f"Scaler fit: mean={self.mean_}, std={self.std_}")
        return self

    def transform(self, data):
        assert self._fitted, "Scaler not fitted yet."
        return (data - self.mean_) / self.std_

    def inverse_transform(self, data):
        assert self._fitted, "Scaler not fitted yet."
        return data * self.std_ + self.mean_

    def fit_transform(self, data):
        return self.fit(data).transform(data)


def sliding_window(data, in_len, out_len):
    """
    Create sliding-window input/output pairs.

    Parameters
    ----------
    data : np.ndarray, shape (T, N, C)
    in_len : int â number of input time steps
    out_len : int â number of output (prediction) time steps

    Returns
    -------
    X : np.ndarray, shape (num_samples, in_len, N, C)
    Y : np.ndarray, shape (num_samples, out_len, N, C)
    """
    T = data.shape[0]
    total = in_len + out_len
    if T < total:
        raise ValueError(f"Data length {T} < in_len+out_len={total}")
    num_samples = T - total + 1
    X = np.zeros((num_samples, in_len) + data.shape[1:], dtype=data.dtype)
    Y = np.zeros((num_samples, out_len) + data.shape[1:], dtype=data.dtype)
    for i in range(num_samples):
        X[i] = data[i : i + in_len]
        Y[i] = data[i + in_len : i + total]
    _dbg(f"sliding_window: {num_samples} samples, X={X.shape}, Y={Y.shape}")
    return X, Y


class DataIterator:
    """Mini-batch iterator over (X, Y) arrays."""

    def __init__(self, X, Y, batch_size=32, shuffle=True):
        assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = X.shape[0]

    def __len__(self):
        return int(np.ceil(self.n / self.batch_size))

    def __iter__(self):
        indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, self.n, self.batch_size):
            end = min(start + self.batch_size, self.n)
            idx = indices[start:end]
            yield self.X[idx], self.Y[idx]


def generate_synthetic_traffic(n_nodes=207, n_steps=2000, n_features=2, seed=42):
    """
    Generate synthetic METR-LA-style traffic data.

    Produces an AR(1) process with spatial correlation driven by a random
    adjacency matrix, plus time-of-day and day-of-week features.

    Parameters
    ----------
    n_nodes : int
    n_steps : int
    n_features : int â number of channels (speed + occupancy by default)
    seed : int

    Returns
    -------
    data : np.ndarray, shape (n_steps, n_nodes, n_features)
    adj  : np.ndarray, shape (n_nodes, n_nodes) â weighted adjacency
    """
    rng = np.random.RandomState(seed)
    _dbg(f"Generating synthetic traffic: nodes={n_nodes}, steps={n_steps}")

    # --- Build random adjacency (sparse-ish, symmetric) ---
    sparsity = min(10 / n_nodes, 0.5)
    raw = rng.rand(n_nodes, n_nodes)
    mask = (raw < sparsity).astype(np.float64)
    mask = np.maximum(mask, mask.T)
    np.fill_diagonal(mask, 0)
    weights = rng.rand(n_nodes, n_nodes) * 0.5 + 0.5
    weights = (weights + weights.T) / 2.0
    adj = mask * weights

    # Row-normalize for diffusion
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum < 1e-8, 1.0, row_sum)
    adj_norm = adj / row_sum

    # --- AR(1) with spatial diffusion ---
    data = np.zeros((n_steps, n_nodes, n_features), dtype=np.float64)
    data[0] = rng.randn(n_nodes, n_features) * 5.0 + 60.0  # mean ~60 mph

    ar_coeff = 0.85
    diffusion_strength = 0.1
    noise_scale = 2.0

    for t in range(1, n_steps):
        noise = rng.randn(n_nodes, n_features) * noise_scale
        temporal = ar_coeff * data[t - 1]
        spatial = diffusion_strength * (adj_norm @ data[t - 1])
        # Time-of-day effect (period ~288 for 5-min intervals in a day)
        tod = np.sin(2.0 * np.pi * t / 288.0) * 5.0
        data[t] = temporal + spatial + noise + tod

    # Clamp to realistic range
    data = np.clip(data, 0.0, 120.0)

    _dbg(f"Synthetic data: shape={data.shape}, adj nnz={int(np.sum(adj > 0))}")
    return data, adj


class AuroraDataset:
    """
    Spatio-temporal traffic dataset with train/val/test splits.

    Parameters
    ----------
    data : np.ndarray, shape (T, N, C)  â full time series
    adj : np.ndarray or None, shape (N, N)
    seq_len : int â input sequence length
    pred_len : int â prediction horizon
    train_ratio : float
    val_ratio : float
    """

    def __init__(
        self,
        data,
        adj=None,
        seq_len=12,
        pred_len=12,
        train_ratio=0.7,
        val_ratio=0.15,
    ):
        assert data.ndim == 3, f"Expected (T, N, C), got {data.shape}"
        self.raw_data = data.astype(np.float64)
        self.adj = adj
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_nodes = data.shape[1]
        self.n_features = data.shape[2]

        T = data.shape[0]
        t_train = int(T * train_ratio)
        t_val = t_train + int(T * val_ratio)

        self._train_data = self.raw_data[:t_train]
        self._val_data = self.raw_data[t_train:t_val]
        self._test_data = self.raw_data[t_val:]

        _dbg(
            f"AuroraDataset: T={T}, train={self._train_data.shape[0]}, "
            f"val={self._val_data.shape[0]}, test={self._test_data.shape[0]}"
        )

        # Fit scaler on training data
        self.scaler = StandardScaler()
        self.scaler.fit(self._train_data)

        # Normalize
        train_norm = self.scaler.transform(self._train_data)
        val_norm = self.scaler.transform(self._val_data)
        test_norm = self.scaler.transform(self._test_data)

        # Build sliding windows
        self.train_X, self.train_Y = sliding_window(train_norm, seq_len, pred_len)
        self.val_X, self.val_Y = sliding_window(val_norm, seq_len, pred_len)
        self.test_X, self.test_Y = sliding_window(test_norm, seq_len, pred_len)

        _dbg(
            f"Samples: train={self.train_X.shape[0]}, "
            f"val={self.val_X.shape[0]}, test={self.test_X.shape[0]}"
        )

    @property
    def train(self):
        return self.train_X, self.train_Y

    @property
    def val(self):
        return self.val_X, self.val_Y

    @property
    def test(self):
        return self.test_X, self.test_Y

    def get_iterator(self, split="train", batch_size=32, shuffle=None):
        if split == "train":
            X, Y = self.train_X, self.train_Y
            if shuffle is None:
                shuffle = True
        elif split == "val":
            X, Y = self.val_X, self.val_Y
            if shuffle is None:
                shuffle = False
        elif split == "test":
            X, Y = self.test_X, self.test_Y
            if shuffle is None:
                shuffle = False
        else:
            raise ValueError(f"Unknown split: {split}")
        return DataIterator(X, Y, batch_size=batch_size, shuffle=shuffle)

    @classmethod
    def from_csv(cls, filepath, n_nodes=None, seq_len=12, pred_len=12, **kwargs):
        """
        Load from CSV. Expects shape (T, N) or (T, N*C).
        If n_nodes given, reshapes columns to (T, N, C).
        """
        _dbg(f"Loading CSV: {filepath}")
        raw = np.genfromtxt(filepath, delimiter=",", skip_header=1)
        if raw.ndim == 1:
            raw = raw.reshape(-1, 1)
        T, cols = raw.shape
        if n_nodes is not None and cols > n_nodes:
            C = cols // n_nodes
            raw = raw[:, : n_nodes * C].reshape(T, n_nodes, C)
        else:
            raw = raw[:, :, np.newaxis]  # (T, N, 1)
        return cls(raw, seq_len=seq_len, pred_len=pred_len, **kwargs)

    @classmethod
    def from_synthetic(cls, n_nodes=207, n_steps=2000, seq_len=12, pred_len=12, **kwargs):
        """Convenience: generate synthetic data and wrap in dataset."""
        data, adj = generate_synthetic_traffic(n_nodes=n_nodes, n_steps=n_steps)
        return cls(data, adj=adj, seq_len=seq_len, pred_len=pred_len, **kwargs)
