"""
kepler_model_base â Numpy MLP model base for Lynceus.

Ported from upstream/kepler/code/model.py (~180 lines of Keras model).
Algorithm changes (~20%):
  - _xavier_init: uses Glorot-uniform with gain correction for GELU
  - _gelu: exact GELU via erf instead of Keras approximation
  - forward: fused dropout mask with inverted scaling at train time
  - predict: deterministic (no dropout), supports batch chunking
  - _clip_grads: global norm clipping to avoid exploding gradients
"""
import os
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[kepler_model] {tag}: {items}")


# ââ Activation helpers âââââââââââââââââââââââââââââââââââââââââââ
def _gelu(x: np.ndarray) -> np.ndarray:
    """Exact GELU activation via erf (not the tanh approximation)."""
    from scipy.special import erf  # lazy import to keep top-level light
    out = 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
    _dbg("_gelu", in_mean=float(np.mean(x)), out_mean=float(np.mean(out)))
    return out


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of exact GELU."""
    from scipy.special import erf
    sqrt2 = np.sqrt(2.0)
    cdf = 0.5 * (1.0 + erf(x / sqrt2))
    pdf = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    return cdf + x * pdf


def _relu(x: np.ndarray) -> np.ndarray:
    out = np.maximum(0.0, x)
    _dbg("_relu", frac_active=float(np.mean(x > 0)))
    return out


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


_ACTIVATIONS = {
    "gelu": (_gelu, _gelu_grad),
    "relu": (_relu, _relu_grad),
}


# ââ Xavier / Glorot initialisation âââââââââââââââââââââââââââââââ
def _xavier_init(fan_in: int, fan_out: int,
                 rng: np.random.RandomState,
                 gain: float = 1.0) -> np.ndarray:
    """Glorot-uniform with optional gain correction (e.g. sqrt(2) for ReLU)."""
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    W = rng.uniform(-limit, limit, size=(fan_in, fan_out))
    _dbg("xavier_init", fan_in=fan_in, fan_out=fan_out,
         limit=round(limit, 6), W_std=round(float(np.std(W)), 6))
    return W


# ââ Dropout mask âââââââââââââââââââââââââââââââââââââââââââââââââ
def _dropout_mask(shape: Tuple[int, ...],
                  rate: float,
                  rng: np.random.RandomState) -> np.ndarray:
    """Inverted dropout mask: zeros dropped units, scales kept units by 1/(1-p)."""
    if rate <= 0.0 or rate >= 1.0:
        return np.ones(shape)
    keep = 1.0 - rate
    mask = (rng.rand(*shape) < keep).astype(np.float64) / keep
    _dbg("dropout_mask", shape=shape, rate=rate,
         frac_kept=float(np.mean(mask > 0)))
    return mask


# ââ Layer container ââââââââââââââââââââââââââââââââââââââââââââââ
class _DenseLayer:
    """Single dense layer with optional bias."""
    __slots__ = ("W", "b", "dW", "db", "z_cache", "a_cache", "mask_cache")

    def __init__(self, W: np.ndarray, b: np.ndarray):
        self.W = W
        self.b = b
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None
        self.z_cache: Optional[np.ndarray] = None
        self.a_cache: Optional[np.ndarray] = None
        self.mask_cache: Optional[np.ndarray] = None


# ââ MLP model ââââââââââââââââââââââââââââââââââââââââââââââââââââ
class NumpyMLP:
    """
    Pure-numpy MLP with Xavier init, GELU/ReLU activation, and inverted dropout.

    Parameters
    ----------
    layer_dims : list[int]
        Sequence of layer widths including input and output dims.
        e.g. [64, 128, 64, 1] gives 3 weight matrices.
    activation : str
        'gelu' or 'relu'.
    dropout_rate : float
        Probability of dropping a unit during training (0 = no dropout).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self,
                 layer_dims: List[int],
                 activation: str = "gelu",
                 dropout_rate: float = 0.1,
                 seed: int = 42):
        assert len(layer_dims) >= 2, "Need at least input + output dims"
        assert activation in _ACTIVATIONS, f"Unknown activation: {activation}"

        self.layer_dims = list(layer_dims)
        self.activation_name = activation
        self._act_fn, self._act_grad = _ACTIVATIONS[activation]
        self.dropout_rate = dropout_rate
        self.rng = np.random.RandomState(seed)

        gain = 1.0 if activation == "gelu" else np.sqrt(2.0)

        self.layers: List[_DenseLayer] = []
        for i in range(len(layer_dims) - 1):
            fan_in, fan_out = layer_dims[i], layer_dims[i + 1]
            W = _xavier_init(fan_in, fan_out, self.rng, gain=gain)
            b = np.zeros((1, fan_out))
            self.layers.append(_DenseLayer(W, b))

        _dbg("MLP.__init__", dims=layer_dims, activation=activation,
             dropout=dropout_rate, n_layers=len(self.layers),
             total_params=self.param_count())

    # ââ helpers ââââââââââââââââââââââââââââââââââââââââââââââ
    def param_count(self) -> int:
        return sum(l.W.size + l.b.size for l in self.layers)

    def get_params(self) -> List[Dict[str, np.ndarray]]:
        """Return a snapshot of all parameters."""
        return [{"W": l.W.copy(), "b": l.b.copy()} for l in self.layers]

    def set_params(self, params: List[Dict[str, np.ndarray]]) -> None:
        """Load parameters from snapshot."""
        for layer, p in zip(self.layers, params):
            layer.W = p["W"].copy()
            layer.b = p["b"].copy()
        _dbg("set_params", n_layers=len(params))

    # ââ forward pass âââââââââââââââââââââââââââââââââââââââââ
    def forward(self, X: np.ndarray, *, training: bool = True) -> np.ndarray:
        """
        Forward pass through all layers.

        Hidden layers use activation + dropout (if training).
        Output layer is linear (no activation, no dropout).
        """
        X = np.asarray(X, dtype=np.float64)
        a = X
        _dbg("forward_start", batch=a.shape[0], features=a.shape[1],
             training=training)

        for i, layer in enumerate(self.layers):
            z = a @ layer.W + layer.b
            layer.z_cache = z

            is_output = (i == len(self.layers) - 1)
            if is_output:
                # linear output â no activation, no dropout
                a_next = z
                layer.mask_cache = None
            else:
                a_next = self._act_fn(z)
                if training and self.dropout_rate > 0:
                    mask = _dropout_mask(a_next.shape, self.dropout_rate, self.rng)
                    a_next = a_next * mask
                    layer.mask_cache = mask
                else:
                    layer.mask_cache = None

            layer.a_cache = a  # input to this layer
            a = a_next

            _dbg(f"forward_L{i}", shape=z.shape, is_output=is_output,
                 z_mean=round(float(np.mean(z)), 6),
                 a_mean=round(float(np.mean(a)), 6))

        return a

    # ââ backward pass ââââââââââââââââââââââââââââââââââââââââ
    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Backward pass â populates layer.dW and layer.db.

        Parameters
        ----------
        loss_grad : array of shape (batch, output_dim)
            dL/d(output) from the loss function.
        """
        delta = loss_grad
        _dbg("backward_start", delta_shape=delta.shape,
             delta_norm=round(float(np.linalg.norm(delta)), 6))

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            a_in = layer.a_cache          # input to this layer
            n = a_in.shape[0]

            is_output = (i == len(self.layers) - 1)
            if not is_output:
                # apply dropout mask first (inverted scaling already baked in)
                if layer.mask_cache is not None:
                    delta = delta * layer.mask_cache
                # then activation gradient
                delta = delta * self._act_grad(layer.z_cache)

            layer.dW = (a_in.T @ delta) / n
            layer.db = np.mean(delta, axis=0, keepdims=True)

            _dbg(f"backward_L{i}", dW_norm=round(float(np.linalg.norm(layer.dW)), 6),
                 db_norm=round(float(np.linalg.norm(layer.db)), 6))

            if i > 0:
                delta = delta @ layer.W.T

    # ââ gradient clipping ââââââââââââââââââââââââââââââââââââ
    def clip_grads(self, max_norm: float = 5.0) -> float:
        """Global-norm gradient clipping. Returns the original norm."""
        all_grads = []
        for l in self.layers:
            if l.dW is not None:
                all_grads.append(l.dW.ravel())
                all_grads.append(l.db.ravel())
        if not all_grads:
            return 0.0
        global_vec = np.concatenate(all_grads)
        gnorm = float(np.linalg.norm(global_vec))
        if gnorm > max_norm:
            scale = max_norm / (gnorm + 1e-12)
            for l in self.layers:
                if l.dW is not None:
                    l.dW *= scale
                    l.db *= scale
        _dbg("clip_grads", gnorm=round(gnorm, 6), max_norm=max_norm,
             clipped=gnorm > max_norm)
        return gnorm

    # ââ SGD update âââââââââââââââââââââââââââââââââââââââââââ
    def sgd_step(self, lr: float = 1e-3) -> None:
        """Vanilla SGD parameter update."""
        for i, layer in enumerate(self.layers):
            if layer.dW is None:
                continue
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db
        _dbg("sgd_step", lr=lr)

    # ââ prediction (no dropout) ââââââââââââââââââââââââââââââ
    def predict(self, X: np.ndarray, *, batch_size: int = 512) -> np.ndarray:
        """Deterministic prediction with optional batch chunking."""
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        if n <= batch_size:
            out = self.forward(X, training=False)
            _dbg("predict", n=n, out_shape=out.shape)
            return out

        parts = []
        for start in range(0, n, batch_size):
            chunk = X[start:start + batch_size]
            parts.append(self.forward(chunk, training=False))
        out = np.concatenate(parts, axis=0)
        _dbg("predict_chunked", n=n, n_chunks=len(parts), out_shape=out.shape)
        return out

    # ââ serialization ââââââââââââââââââââââââââââââââââââââââ
    def state_dict(self) -> Dict[str, Any]:
        """Serialize model to a plain dict (numpy arrays inside)."""
        return {
            "layer_dims": self.layer_dims,
            "activation": self.activation_name,
            "dropout_rate": self.dropout_rate,
            "params": self.get_params(),
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any], seed: int = 42) -> "NumpyMLP":
        """Reconstruct model from state dict."""
        model = cls(d["layer_dims"], activation=d["activation"],
                    dropout_rate=d["dropout_rate"], seed=seed)
        model.set_params(d["params"])
        _dbg("from_state_dict", dims=d["layer_dims"])
        return model
