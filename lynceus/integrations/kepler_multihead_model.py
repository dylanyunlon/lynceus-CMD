"""
Multihead MLP with Spectral Normalization and GP Layer.
Pure numpy implementation.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import time
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _dbg(tag: str, **kw):
    parts = [f"[DBG:{tag}]"]
    for k, v in kw.items():
        if isinstance(v, np.ndarray):
            parts.append(f"{k}.shape={v.shape} dtype={v.dtype}")
        else:
            parts.append(f"{k}={v}")
    logger.debug(" ".join(parts))


# ---------------------------------------------------------------------------
# Numeric utilities
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    _dbg("softmax", x=x, axis=axis)
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    out = e / np.sum(e, axis=axis, keepdims=True)
    _dbg("softmax_out", out=out)
    return out


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax."""
    _dbg("log_softmax", x=x)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    out = shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    _dbg("log_softmax_out", out=out)
    return out


def relu(x: np.ndarray) -> np.ndarray:
    _dbg("relu", x=x)
    return np.maximum(0.0, x)


def he_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """He / Kaiming initialisation."""
    std = np.sqrt(2.0 / fan_in)
    w = rng.normal(0.0, std, (fan_in, fan_out))
    _dbg("he_init", fan_in=fan_in, fan_out=fan_out, w=w)
    return w


# ---------------------------------------------------------------------------
# Spectral Normalization (power-iteration)
# ---------------------------------------------------------------------------

class SpectralNorm:
    """Wraps a weight matrix and keeps a running estimate of its spectral norm."""

    def __init__(self, weight: np.ndarray, n_power_iters: int = 1):
        _dbg("SpectralNorm.__init__", weight=weight, n_power_iters=n_power_iters)
        self.weight = weight
        self.n_power_iters = n_power_iters
        self._u = np.random.randn(weight.shape[0])
        self._u /= np.linalg.norm(self._u) + 1e-12
        self._v = np.random.randn(weight.shape[1])
        self._v /= np.linalg.norm(self._v) + 1e-12
        self.sigma: float = 1.0

    def _power_iteration(self):
        _dbg("SpectralNorm._power_iteration", iters=self.n_power_iters)
        u, v = self._u.copy(), self._v.copy()
        for _ in range(self.n_power_iters):
            v = self.weight.T @ u
            v /= np.linalg.norm(v) + 1e-12
            u = self.weight @ v
            u /= np.linalg.norm(u) + 1e-12
        self._u, self._v = u, v
        self.sigma = float(u @ self.weight @ v)
        _dbg("SpectralNorm._power_iteration_done", sigma=self.sigma)

    def normalized(self) -> np.ndarray:
        """Return W / sigma(W)."""
        self._power_iteration()
        w_hat = self.weight / (self.sigma + 1e-12)
        _dbg("SpectralNorm.normalized", w_hat=w_hat)
        return w_hat


# ---------------------------------------------------------------------------
# Gaussian Process layer via Random Fourier Features
# ---------------------------------------------------------------------------

class GPLayer:
    """
    Approximate GP output using Random Fourier Features (RFF).
    Produces mean and variance estimates.
    """

    def __init__(self, in_dim: int, n_features: int = 128,
                 length_scale: float = 1.0, seed: int = 42):
        _dbg("GPLayer.__init__", in_dim=in_dim, n_features=n_features,
             length_scale=length_scale)
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.omega = rng.normal(0.0, 1.0 / length_scale, (in_dim, n_features))
        self.bias = rng.uniform(0.0, 2.0 * np.pi, n_features)
        self.beta: Optional[np.ndarray] = None
        self.precision: Optional[np.ndarray] = None
        self._ridge = 1e-4

    def _phi(self, x: np.ndarray) -> np.ndarray:
        """Random Fourier Feature map."""
        proj = x @ self.omega + self.bias
        feat = np.sqrt(2.0 / self.n_features) * np.cos(proj)
        _dbg("GPLayer._phi", feat=feat)
        return feat

    def fit(self, x: np.ndarray, y: np.ndarray):
        """Bayesian linear regression on RFF features."""
        _dbg("GPLayer.fit", x=x, y=y)
        phi = self._phi(x)
        gram = phi.T @ phi + self._ridge * np.eye(self.n_features)
        self.precision = gram
        self.beta = np.linalg.solve(gram, phi.T @ y)
        _dbg("GPLayer.fit_done", beta=self.beta)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, variance) for each sample."""
        _dbg("GPLayer.predict", x=x)
        if self.beta is None:
            raise RuntimeError("GPLayer has not been fit yet")
        phi = self._phi(x)
        mean = phi @ self.beta
        prec_inv = np.linalg.inv(self.precision)
        var = np.sum((phi @ prec_inv) * phi, axis=1, keepdims=True)
        _dbg("GPLayer.predict_out", mean=mean, var=var)
        return mean, var


# ---------------------------------------------------------------------------
# Single MLP head
# ---------------------------------------------------------------------------

class _MLPHead:
    """One classification head with optional spectral-norm layers."""

    def __init__(self, input_dim: int, hidden: int, n_classes: int,
                 use_sn: bool = True, seed: int = 0):
        _dbg("_MLPHead.__init__", input_dim=input_dim, hidden=hidden,
             n_classes=n_classes, seed=seed)
        rng = np.random.default_rng(seed)
        self.w1 = he_init(input_dim, hidden, rng)
        self.b1 = np.zeros(hidden)
        self.w2 = he_init(hidden, hidden, rng)
        self.b2 = np.zeros(hidden)
        self.w3 = he_init(hidden, n_classes, rng)
        self.b3 = np.zeros(n_classes)
        self.sn1 = SpectralNorm(self.w1) if use_sn else None
        self.sn2 = SpectralNorm(self.w2) if use_sn else None

    def forward(self, x: np.ndarray) -> np.ndarray:
        _dbg("_MLPHead.forward", x=x)
        w1 = self.sn1.normalized() if self.sn1 else self.w1
        h = relu(x @ w1 + self.b1)
        w2 = self.sn2.normalized() if self.sn2 else self.w2
        h = relu(h @ w2 + self.b2)
        logits = h @ self.w3 + self.b3
        _dbg("_MLPHead.forward_logits", logits=logits)
        return logits

    def features(self, x: np.ndarray) -> np.ndarray:
        """Return penultimate-layer features."""
        w1 = self.sn1.normalized() if self.sn1 else self.w1
        h = relu(x @ w1 + self.b1)
        w2 = self.sn2.normalized() if self.sn2 else self.w2
        h = relu(h @ w2 + self.b2)
        _dbg("_MLPHead.features", h=h)
        return h


# ---------------------------------------------------------------------------
# Multihead MLP
# ---------------------------------------------------------------------------

class MultiheadMLP:
    """
    Ensemble of MLP heads with a shared GP uncertainty layer.

    Parameters
    ----------
    input_dim : int
    hidden : int
    n_heads : int
    n_classes : int
    gp_features : int  â number of random Fourier features for the GP layer
    """

    def __init__(self, input_dim: int, hidden: int, n_heads: int,
                 n_classes: int, gp_features: int = 128, seed: int = 42):
        _dbg("MultiheadMLP.__init__", input_dim=input_dim, hidden=hidden,
             n_heads=n_heads, n_classes=n_classes)
        self.input_dim = input_dim
        self.hidden = hidden
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.heads: List[_MLPHead] = []
        for i in range(n_heads):
            self.heads.append(
                _MLPHead(input_dim, hidden, n_classes, use_sn=True, seed=seed + i)
            )
        self.gp = GPLayer(hidden, n_features=gp_features, seed=seed + 1000)
        self._gp_fitted = False

    # --- public API ---------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass â averaged softmax probabilities  (N, n_classes).
        """
        _dbg("MultiheadMLP.forward", x=x)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        all_probs = np.stack([softmax(h.forward(x)) for h in self.heads], axis=0)
        mean_probs = np.mean(all_probs, axis=0)
        _dbg("MultiheadMLP.forward_out", mean_probs=mean_probs)
        return mean_probs

    def predict_with_uncertainty(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        pred : (N,) int   â predicted class indices
        unc  : (N,) float â combined uncertainty score
        """
        _dbg("MultiheadMLP.predict_with_uncertainty", x=x)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[np.newaxis, :]

        all_logits = [h.forward(x) for h in self.heads]
        all_probs = np.stack([softmax(lg) for lg in all_logits], axis=0)
        mean_probs = np.mean(all_probs, axis=0)

        # epistemic: variance across heads (mutual information proxy)
        epistemic = np.mean(np.var(all_probs, axis=0), axis=-1)

        # aleatoric: entropy of the mean prediction
        aleatoric = -np.sum(mean_probs * np.log(mean_probs + 1e-12), axis=-1)

        # GP variance (if fitted)
        gp_var = np.zeros(x.shape[0])
        if self._gp_fitted:
            feats = self.heads[0].features(x)
            _, v = self.gp.predict(feats)
            gp_var = v.ravel()

        unc = epistemic + aleatoric + gp_var
        pred = np.argmax(mean_probs, axis=-1)
        _dbg("MultiheadMLP.predict_with_uncertainty_done", pred=pred, unc=unc)
        return pred, unc

    def fit_gp(self, x_train: np.ndarray, y_train: np.ndarray):
        """Fit the GP layer on penultimate features of head-0."""
        _dbg("MultiheadMLP.fit_gp", x_train=x_train, y_train=y_train)
        feats = self.heads[0].features(np.asarray(x_train, dtype=np.float64))
        targets = np.asarray(y_train, dtype=np.float64)
        if targets.ndim == 1:
            targets = targets[:, np.newaxis]
        self.gp.fit(feats, targets)
        self._gp_fitted = True

    def summary(self) -> Dict[str, Any]:
        """Model summary dict."""
        total_params = 0
        for h in self.heads:
            total_params += h.w1.size + h.b1.size
            total_params += h.w2.size + h.b2.size
            total_params += h.w3.size + h.b3.size
        info = {
            "input_dim": self.input_dim,
            "hidden": self.hidden,
            "n_heads": self.n_heads,
            "n_classes": self.n_classes,
            "total_params": total_params,
            "gp_fitted": self._gp_fitted,
        }
        _dbg("MultiheadMLP.summary", **info)
        return info


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    rng = np.random.default_rng(0)
    N, D, C = 64, 10, 3
    X = rng.normal(size=(N, D))
    y = rng.integers(0, C, size=N)

    model = MultiheadMLP(input_dim=D, hidden=32, n_heads=4, n_classes=C)
    probs = model.forward(X)
    print("probs shape:", probs.shape)

    model.fit_gp(X, y)
    preds, uncs = model.predict_with_uncertainty(X)
    print("preds:", preds[:10])
    print("uncertainty:", uncs[:10])
    print(model.summary())
