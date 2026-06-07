"""Temporal attention components for D2STGNN Aurora variant. Pure numpy."""

import numpy as np

_DEBUG = True


def _dbg(name: str, **kwargs):
    if _DEBUG:
        parts = [f"[DBG {name}]"]
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                parts.append(f"{k}={v.shape} Î¼={v.mean():.4f} Ï={v.std():.4f}")
            else:
                parts.append(f"{k}={v}")
        print(" | ".join(parts))


def _softmax(x, axis=-1):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def _dropout(x, rate, training=True):
    """Inverted dropout."""
    if not training or rate <= 0.0:
        return x
    mask = (np.random.rand(*x.shape) > rate).astype(x.dtype)
    return x * mask / (1.0 - rate)


def _layer_norm_1d(x, gamma, beta, eps=1e-5):
    """Layer norm over last dimension."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xn = (x - mu) / np.sqrt(var + eps)
    return gamma * xn + beta


def temporal_decay_weights(T, alpha=0.1):
    """Exponential decay weights biasing toward recent time steps.

    Returns shape (T,) where index T-1 (most recent) has weight 1.0 and
    earlier steps decay by exp(-alpha * distance).
    """
    positions = np.arange(T, dtype=np.float64)
    weights = np.exp(-alpha * (T - 1 - positions))
    weights /= weights.sum() + 1e-12
    _dbg("temporal_decay_weights", T=T, alpha=alpha, weights=weights)
    return weights


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding:
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model, max_len=512):
        self.d_model = d_model
        self.max_len = max_len
        self.pe = self._build_table(d_model, max_len)
        _dbg("PositionalEncoding.__init__", d_model=d_model, max_len=max_len,
             pe=self.pe)

    @staticmethod
    def _build_table(d_model, max_len):
        pe = np.zeros((max_len, d_model), dtype=np.float64)
        pos = np.arange(max_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div[:d_model // 2])  # handle odd d_model
        return pe

    def forward(self, T):
        """Return positional encoding of shape (T, D)."""
        out = self.pe[:T]
        _dbg("PositionalEncoding.forward", T=T, out=out)
        return out


# ---------------------------------------------------------------------------
# Temporal Attention
# ---------------------------------------------------------------------------

class TemporalAttention:
    """Multi-head scaled dot-product self-attention over the time axis.

    Input shapes  Q, K, V : (B, T, N, D)
    Output shape           : (B, T, N, D)
    Attention is computed per-node across time steps.
    """

    def __init__(self, d_model, n_heads, dropout=0.1, causal=True):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        self.causal = causal
        self.training = True

        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model).astype(np.float64) * scale
        self.W_K = np.random.randn(d_model, d_model).astype(np.float64) * scale
        self.W_V = np.random.randn(d_model, d_model).astype(np.float64) * scale
        self.W_O = np.random.randn(d_model, d_model).astype(np.float64) * scale
        self.b_Q = np.zeros(d_model, dtype=np.float64)
        self.b_K = np.zeros(d_model, dtype=np.float64)
        self.b_V = np.zeros(d_model, dtype=np.float64)
        self.b_O = np.zeros(d_model, dtype=np.float64)

        # layer norm params
        self.ln_gamma = np.ones(d_model, dtype=np.float64)
        self.ln_beta = np.zeros(d_model, dtype=np.float64)

        self.pos_enc = PositionalEncoding(d_model)

        n_params = 4 * d_model * d_model + 4 * d_model + 2 * d_model
        _dbg("TemporalAttention.__init__", d_model=d_model, n_heads=n_heads,
             causal=causal, param_count=n_params)

    def _causal_mask(self, T):
        """Lower-triangular causal mask (T, T). 0 = masked, 1 = attend."""
        mask = np.tril(np.ones((T, T), dtype=np.float64))
        _dbg("TemporalAttention._causal_mask", T=T, mask=mask)
        return mask

    def forward(self, Q, K, V):
        """Scaled dot-product multi-head attention.

        Q, K, V : (B, T, N, D)
        Returns : (B, T, N, D)
        """
        B, T, N, D = Q.shape
        _dbg("TemporalAttention.forward.input", Q=Q, K=K, V=V)

        # add positional encoding along time axis
        pe = self.pos_enc.forward(T)  # (T, D)
        Q = Q + pe[None, :, None, :]
        K = K + pe[None, :, None, :]

        # project  â reshape to (B*N, T, D) for matmul convenience
        def proj(x, W, b):
            # x: (B,T,N,D) -> (B,N,T,D) -> (B*N,T,D)
            xt = x.transpose(0, 2, 1, 3).reshape(B * N, T, D)
            return xt @ W.T + b

        q = proj(Q, self.W_Q, self.b_Q)  # (B*N, T, D)
        k = proj(K, self.W_K, self.b_K)
        v = proj(V, self.W_V, self.b_V)

        # split heads -> (B*N, n_heads, T, d_k)
        def split_heads(x):
            return x.reshape(B * N, T, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)  # (B*N,H,T,T)
        _dbg("TemporalAttention.forward.scores_raw", scores=scores)

        if self.causal:
            mask = self._causal_mask(T)  # (T, T)
            scores = np.where(mask[None, None, :, :] == 1, scores, -1e9)

        attn = _softmax(scores, axis=-1)
        attn = _dropout(attn, self.dropout, self.training)
        _dbg("TemporalAttention.forward.attn_weights", attn=attn)

        # weighted sum
        out = attn @ v  # (B*N, H, T, d_k)
        # merge heads
        out = out.transpose(0, 2, 1, 3).reshape(B * N, T, D)
        out = out @ self.W_O.T + self.b_O  # (B*N, T, D)

        # reshape back to (B, T, N, D)
        out = out.reshape(B, N, T, D).transpose(0, 2, 1, 3)

        # residual + layer norm
        out = _layer_norm_1d(out + Q, self.ln_gamma, self.ln_beta)
        _dbg("TemporalAttention.forward.output", out=out)
        return out


# ---------------------------------------------------------------------------
# Gated Temporal Convolution
# ---------------------------------------------------------------------------

class GatedTemporalConv:
    """1-D dilated temporal convolution with GLU gating.

    Input / output shape: (B, T, N, C).
    Convolution runs along the T axis independently per node.
    """

    def __init__(self, channels, kernel_size=3, dilation=1):
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation

        fan_in = channels * kernel_size
        std = np.sqrt(2.0 / fan_in)
        # Two sets of weights for GLU (signal + gate)
        self.W_sig = np.random.randn(kernel_size, channels, channels).astype(np.float64) * std
        self.b_sig = np.zeros(channels, dtype=np.float64)
        self.W_gate = np.random.randn(kernel_size, channels, channels).astype(np.float64) * std
        self.b_gate = np.zeros(channels, dtype=np.float64)

        # layer norm
        self.ln_gamma = np.ones(channels, dtype=np.float64)
        self.ln_beta = np.zeros(channels, dtype=np.float64)

        n_params = 2 * kernel_size * channels * channels + 2 * channels + 2 * channels
        _dbg("GatedTemporalConv.__init__", channels=channels, kernel_size=kernel_size,
             dilation=dilation, param_count=n_params)

    def _conv1d(self, x, W, b):
        """Causal dilated 1-D conv along T axis.

        x : (B, T, N, C)
        W : (K, C_in, C_out)
        b : (C_out,)
        Returns: (B, T, N, C_out)
        """
        B, T, N, C = x.shape
        K = self.kernel_size
        d = self.dilation
        # causal padding
        pad_len = (K - 1) * d
        xp = np.pad(x, ((0, 0), (pad_len, 0), (0, 0), (0, 0)), mode='constant')
        out = np.zeros((B, T, N, W.shape[2]), dtype=x.dtype)
        for i in range(K):
            tap = i * d
            out += np.einsum('btnc,cd->btnd', xp[:, pad_len - tap: pad_len - tap + T], W[i])
        out += b
        return out

    def forward(self, x):
        """x: (B, T, N, C) -> (B, T, N, C)."""
        _dbg("GatedTemporalConv.forward.input", x=x)
        sig = self._conv1d(x, self.W_sig, self.b_sig)
        gate = self._conv1d(x, self.W_gate, self.b_gate)

        # GLU: sig * sigmoid(gate)
        gate_act = 1.0 / (1.0 + np.exp(-gate))
        out = sig * gate_act
        _dbg("GatedTemporalConv.forward.glu", sig=sig, gate_act=gate_act)

        # residual + layer norm
        out = _layer_norm_1d(out + x, self.ln_gamma, self.ln_beta)
        _dbg("GatedTemporalConv.forward.output", out=out)
        return out
