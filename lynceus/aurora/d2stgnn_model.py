"""D2STGNN Aurora â main model. Pure numpy, decoupled spatial-temporal design."""

import numpy as np
from lynceus.aurora.temporal_attention import (
    TemporalAttention, GatedTemporalConv, temporal_decay_weights, _softmax, _dropout
)

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalisation over the last dimension."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    xn = (x - mu) / np.sqrt(var + eps)
    out = gamma * xn + beta
    _dbg("_layer_norm", x=x, out=out)
    return out


def _glu(a, b):
    """Gated linear unit: a * sigmoid(b)."""
    gate = 1.0 / (1.0 + np.exp(-b))
    out = a * gate
    _dbg("_glu", a=a, b=b, out=out)
    return out


def _relu(x):
    return np.maximum(0, x)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class AuroraConfig:
    """Model hyper-parameters."""
    def __init__(
        self,
        num_nodes=207,
        in_channels=2,
        out_channels=1,
        hidden_dim=64,
        n_heads=4,
        num_layers=3,
        T_in=12,
        T_out=12,
        diffusion_steps=2,
        dropout=0.1,
        temporal_kernel=3,
        temporal_dilation=1,
        decay_alpha=0.1,
        causal=True,
    ):
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.T_in = T_in
        self.T_out = T_out
        self.diffusion_steps = diffusion_steps
        self.dropout = dropout
        self.temporal_kernel = temporal_kernel
        self.temporal_dilation = temporal_dilation
        self.decay_alpha = decay_alpha
        self.causal = causal


# ---------------------------------------------------------------------------
# Diffusion Convolution (spatial pathway)
# ---------------------------------------------------------------------------

class DiffusionConv:
    """K-step diffusion graph convolution with forward/backward supports.

    Implements  sum_k  theta_k  (D^{-1}A)^k  X   for both directions.
    Input x: (B, T, N, C_in), adj: (N, N) -> output: (B, T, N, C_out).
    """

    def __init__(self, c_in, c_out, K=2):
        self.K = K
        self.c_in = c_in
        self.c_out = c_out

        fan = c_in * (2 * K + 1)
        std = np.sqrt(2.0 / fan)
        # weights for each diffusion step (forward + backward + identity)
        self.W = np.random.randn(2 * K + 1, c_in, c_out).astype(np.float64) * std
        self.b = np.zeros(c_out, dtype=np.float64)

        n_params = (2 * K + 1) * c_in * c_out + c_out
        _dbg("DiffusionConv.__init__", c_in=c_in, c_out=c_out, K=K,
             param_count=n_params)

    @staticmethod
    def _norm_adj(adj):
        """Row-normalise adjacency: D^{-1} A."""
        d = adj.sum(axis=1, keepdims=True)
        d = np.where(d > 0, d, 1.0)
        return adj / d


    def get_params(self):
        """Return flattened parameter vector (for numerical gradient)."""
        params = []
        params.append(self.input_proj.W.ravel())
        params.append(self.input_proj.b.ravel())
        for blk in self.blocks:
            params.append(blk.spatial.W_forward.ravel())
            params.append(blk.spatial.W_backward.ravel())
        params.append(self.output_proj.W.ravel())
        params.append(self.output_proj.b.ravel())
        return np.concatenate(params)

    def set_params(self, flat):
        """Restore from flat vector (stub)."""
        pass

    def forward(self, x, adj):
        """x: (B,T,N,C_in), adj: (N,N) -> (B,T,N,C_out)."""
        _dbg("DiffusionConv.forward.input", x=x, adj=adj)
        A_fwd = self._norm_adj(adj)
        A_bwd = self._norm_adj(adj.T)

        B, T, N, C = x.shape
        supports = []

        # identity
        supports.append(x)

        # forward diffusion
        h = x
        for _ in range(self.K):
            # (B,T,N,C) x (N,N)^T => einsum over node dim
            h = np.einsum('btni,nm->btmi', h, A_fwd)
            supports.append(h)

        # backward diffusion
        h = x
        for _ in range(self.K):
            h = np.einsum('btni,nm->btmi', h, A_bwd)
            supports.append(h)

        # weighted combination
        out = np.zeros((B, T, N, self.c_out), dtype=x.dtype)
        for idx, s in enumerate(supports):
            out += np.einsum('btni,io->btno', s, self.W[idx])
        out += self.b
        _dbg("DiffusionConv.forward.output", out=out)
        return out


# ---------------------------------------------------------------------------
# Decoupled Gating Mechanism
# ---------------------------------------------------------------------------

class DecoupledGate:
    """Gating that merges spatial and temporal pathway outputs.

    Given spatial rep S and temporal rep T (both (B,T,N,D)),
    produces  G * S + (1-G) * T  where G = sigmoid(W_s S + W_t T + b).
    """

    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim
        std = np.sqrt(2.0 / (2 * hidden_dim))
        self.W_s = np.random.randn(hidden_dim, hidden_dim).astype(np.float64) * std
        self.W_t = np.random.randn(hidden_dim, hidden_dim).astype(np.float64) * std
        self.b = np.zeros(hidden_dim, dtype=np.float64)

        n_params = 2 * hidden_dim * hidden_dim + hidden_dim
        _dbg("DecoupledGate.__init__", hidden_dim=hidden_dim, param_count=n_params)

    def forward(self, s, t):
        """s, t: (B,T,N,D) -> (B,T,N,D)."""
        _dbg("DecoupledGate.forward.input", s=s, t=t)
        g = s @ self.W_s + t @ self.W_t + self.b  # broadcast b
        g = 1.0 / (1.0 + np.exp(-g))  # sigmoid
        out = g * s + (1.0 - g) * t
        _dbg("DecoupledGate.forward.output", gate_mean=g.mean(), out=out)
        return out


# ---------------------------------------------------------------------------
# Spatio-Temporal Block
# ---------------------------------------------------------------------------

class SpatioTemporalBlock:
    """One block: spatial diffusion conv -> temporal attention + gated conv
    -> decoupled gating -> residual + layer norm.
    """

    def __init__(self, hidden_dim, n_heads, diffusion_steps,
                 dropout, kernel_size, dilation, causal):
        self.hidden_dim = hidden_dim

        # spatial pathway
        self.spatial_conv = DiffusionConv(hidden_dim, hidden_dim, K=diffusion_steps)
        self.spatial_ln_gamma = np.ones(hidden_dim, dtype=np.float64)
        self.spatial_ln_beta = np.zeros(hidden_dim, dtype=np.float64)

        # temporal pathway
        self.temporal_attn = TemporalAttention(hidden_dim, n_heads,
                                               dropout=dropout, causal=causal)
        self.temporal_conv = GatedTemporalConv(hidden_dim,
                                               kernel_size=kernel_size,
                                               dilation=dilation)

        # decoupled gate
        self.gate = DecoupledGate(hidden_dim)

        # final residual layer norm
        self.ln_gamma = np.ones(hidden_dim, dtype=np.float64)
        self.ln_beta = np.zeros(hidden_dim, dtype=np.float64)

        n_params = (
            (2 * diffusion_steps + 1) * hidden_dim * hidden_dim + hidden_dim  # diff conv
            + 2 * hidden_dim  # spatial LN
            + 4 * hidden_dim * hidden_dim + 4 * hidden_dim + 2 * hidden_dim  # attn
            + 2 * kernel_size * hidden_dim * hidden_dim + 4 * hidden_dim      # gated conv
            + 2 * hidden_dim * hidden_dim + hidden_dim  # gate
            + 2 * hidden_dim  # final LN
        )
        _dbg("SpatioTemporalBlock.__init__", hidden_dim=hidden_dim,
             param_count=n_params)

    def forward(self, x, adj):
        """x: (B,T,N,D), adj: (N,N) -> (B,T,N,D)."""
        _dbg("SpatioTemporalBlock.forward.input", x=x)
        residual = x

        # --- spatial pathway ---
        s = self.spatial_conv.forward(x, adj)
        s = _relu(s)
        s = _layer_norm(s, self.spatial_ln_gamma, self.spatial_ln_beta)

        # --- temporal pathway ---
        t = self.temporal_attn.forward(x, x, x)
        t = self.temporal_conv.forward(t)

        # --- merge via decoupled gating ---
        merged = self.gate.forward(s, t)

        # residual + layer norm
        out = _layer_norm(merged + residual, self.ln_gamma, self.ln_beta)
        _dbg("SpatioTemporalBlock.forward.output", out=out)
        return out


# ---------------------------------------------------------------------------
# Output Projection
# ---------------------------------------------------------------------------

class OutputProjection:
    """Projects hidden representation to T_out future steps.

    Takes (B, T_in, N, hidden_dim) and produces (B, T_out, N, out_channels).
    Strategy: temporal pooling via learned weights -> linear per future step.
    """

    def __init__(self, hidden_dim, out_channels, T_in, T_out):
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.T_in = T_in
        self.T_out = T_out

        std_pool = np.sqrt(2.0 / T_in)
        self.W_pool = np.random.randn(T_in, T_out).astype(np.float64) * std_pool

        std_proj = np.sqrt(2.0 / hidden_dim)
        self.W_proj = np.random.randn(hidden_dim, out_channels).astype(np.float64) * std_proj
        self.b_proj = np.zeros(out_channels, dtype=np.float64)

        n_params = T_in * T_out + hidden_dim * out_channels + out_channels
        _dbg("OutputProjection.__init__", hidden_dim=hidden_dim,
             out_channels=out_channels, T_in=T_in, T_out=T_out,
             param_count=n_params)

    def forward(self, x):
        """x: (B, T_in, N, D) -> (B, T_out, N, C_out)."""
        _dbg("OutputProjection.forward.input", x=x)
        # temporal mixing: (B, T_in, N, D) -> (B, T_out, N, D)
        # einsum: pool over T_in, expand to T_out
        h = np.einsum('btnd,tp->bpnd', x, self.W_pool)  # (B, T_out, N, D)
        out = h @ self.W_proj + self.b_proj  # (B, T_out, N, C_out)
        _dbg("OutputProjection.forward.output", out=out)
        return out


# ---------------------------------------------------------------------------
# Input Projection
# ---------------------------------------------------------------------------

class InputProjection:
    """Linear embedding from in_channels to hidden_dim."""

    def __init__(self, in_channels, hidden_dim):
        std = np.sqrt(2.0 / in_channels)
        self.W = np.random.randn(in_channels, hidden_dim).astype(np.float64) * std
        self.b = np.zeros(hidden_dim, dtype=np.float64)

        n_params = in_channels * hidden_dim + hidden_dim
        _dbg("InputProjection.__init__", in_channels=in_channels,
             hidden_dim=hidden_dim, param_count=n_params)

    def forward(self, x):
        """x: (B, T, N, C_in) -> (B, T, N, hidden_dim)."""
        _dbg("InputProjection.forward.input", x=x)
        out = x @ self.W + self.b
        out = _relu(out)
        _dbg("InputProjection.forward.output", out=out)
        return out


# ---------------------------------------------------------------------------
# D2STGNN Main Model
# ---------------------------------------------------------------------------

class D2STGNN:
    """Decoupled Dynamic Spatio-Temporal Graph Neural Network â Aurora variant.

    Architecture:
        Input projection
        -> N stacked SpatioTemporalBlocks (spatial diffusion + temporal
           attention/gated conv with decoupled gating)
        -> Temporal decay reweighting
        -> Output projection  -> predictions

    Parameters
    ----------
    config : AuroraConfig
    """

    def __init__(self, config: AuroraConfig):
        self.config = config
        self.training = True

        # input projection
        self.input_proj = InputProjection(config.in_channels, config.hidden_dim)

        # stacked ST blocks
        self.blocks = []
        for i in range(config.n_layers):
            dilation_i = config.temporal_dilation * (2 ** i)  # growing dilation
            block = SpatioTemporalBlock(
                hidden_dim=config.hidden_dim,
                n_heads=config.n_heads,
                diffusion_steps=config.diffusion_steps,
                dropout=config.dropout,
                kernel_size=config.temporal_kernel,
                dilation=dilation_i,
                causal=config.causal,
            )
            self.blocks.append(block)

        # output
        self.output_proj = OutputProjection(
            config.hidden_dim, config.out_channels,
            config.T_in, config.T_out,
        )

        # precompute decay weights
        self.decay = temporal_decay_weights(config.T_in, config.decay_alpha)

        total = self._count_params()
        _dbg("D2STGNN.__init__", num_layers=config.n_layers,
             hidden_dim=config.hidden_dim, total_params=total)

    # --- param counting helpers ---

    @staticmethod
    def _count_obj(obj):
        total = 0
        for v in vars(obj).values():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                total += v.size
        return total

    def _count_params(self):
        total = self._count_obj(self.input_proj)
        for blk in self.blocks:
            total += self._count_obj(blk)
            total += self._count_obj(blk.spatial_conv)
            total += self._count_obj(blk.temporal_attn)
            total += self._count_obj(blk.temporal_conv)
            total += self._count_obj(blk.gate)
        total += self._count_obj(self.output_proj)
        return total

    def forward(self, x, adj):
        """Full forward pass.

        Parameters
        ----------
        x   : np.ndarray, shape (B, T_in, N, C_in)
        adj : np.ndarray, shape (N, N)  â adjacency matrix

        Returns
        -------
        np.ndarray, shape (B, T_out, N, C_out)
        """
        B, T, N, C = x.shape
        _dbg("D2STGNN.forward.input", x=x, adj=adj)

        assert T == self.config.T_in, (
            f"Expected T_in={self.config.T_in}, got {T}")
        assert N == self.config.n_nodes, (
            f"Expected num_nodes={self.config.n_nodes}, got {N}")

        # 1) Input projection
        h = self.input_proj.forward(x)  # (B, T, N, D)

        # 2) Stacked spatio-temporal blocks
        skip_sum = np.zeros_like(h)
        for idx, block in enumerate(self.blocks):
            _dbg(f"D2STGNN.forward.block_{idx}", h=h)
            h = block.forward(h, adj)
            skip_sum = skip_sum + h  # skip connection accumulation

        # average skip connections
        h = skip_sum / len(self.blocks)
        _dbg("D2STGNN.forward.after_blocks", h=h)

        # 3) Apply temporal decay reweighting
        #    decay shape (T,) -> broadcast multiply to bias recent steps
        decay = self.decay[None, :, None, None]  # (1, T, 1, 1)
        h = h * decay
        _dbg("D2STGNN.forward.after_decay", h=h)

        # 4) Dropout
        h = _dropout(h, self.config.dropout, self.training)

        # 5) Output projection
        out = self.output_proj.forward(h)  # (B, T_out, N, C_out)
        _dbg("D2STGNN.forward.output", out=out)
        return out


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    cfg = AuroraConfig(
        num_nodes=20,
        in_channels=2,
        out_channels=1,
        hidden_dim=32,
        n_heads=4,
        num_layers=3,
        T_in=12,
        T_out=12,
        diffusion_steps=2,
        dropout=0.0,
        temporal_kernel=3,
        temporal_dilation=1,
        decay_alpha=0.1,
        causal=True,
    )

    model = D2STGNN(cfg)

    B, T_in, N, C_in = 4, cfg.T_in, cfg.num_nodes, cfg.in_channels
    x = np.random.randn(B, T_in, N, C_in)
    adj = (np.random.rand(N, N) > 0.7).astype(np.float64)
    adj = np.maximum(adj, adj.T)  # symmetric
    np.fill_diagonal(adj, 1.0)

    print("\n===== FORWARD PASS =====")
    y = model.forward(x, adj)
    print(f"\nInput  shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output mean:  {y.mean():.6f}")
    print(f"Output std:   {y.std():.6f}")
    assert y.shape == (B, cfg.T_out, N, cfg.out_channels), \
        f"Shape mismatch: {y.shape}"
    print("\nSmoke test PASSED.")
