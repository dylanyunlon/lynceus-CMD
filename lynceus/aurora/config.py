"""AuroraConfig â D2STGNN Aurora variant configuration."""

import os
import numpy as np
from dataclasses import dataclass, field

_DEBUG = os.environ.get("AURORA_DEBUG", "0") == "1"


def _dbg(tag: str, **kw):
    if _DEBUG:
        parts = [f"[DBG:{tag}]"]
        for k, v in kw.items():
            if isinstance(v, np.ndarray):
                parts.append(f"{k}:shape={v.shape},Î¼={v.mean():.4f},Ï={v.std():.4f}")
            else:
                parts.append(f"{k}={v}")
        print(" | ".join(parts))


@dataclass
class AuroraConfig:
    """Full configuration for the D2STGNN Aurora spatio-temporal forecaster."""

    # graph / spatial
    n_nodes: int = 207
    in_channels: int = 2
    out_channels: int = 2
    hidden_dim: int = 64

    # architecture
    n_layers: int = 3
    K_diffusion: int = 2
    n_heads: int = 4

    # temporal window
    seq_len: int = 12
    pred_len: int = 12

    # regularisation & training
    dropout: float = 0.1
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    patience: int = 10
    seed: int = 42


    # D2STGNN model params (aliases/extras)
    temporal_dilation: int = 1
    diffusion_steps: int = 2
    temporal_kernel: int = 3
    causal: bool = True
    decay_alpha: float = 0.1

    def __post_init__(self):
        """Apply environment-variable overrides after dataclass init."""
        self.epochs = int(os.environ.get("EPOCHS", str(self.epochs)))
        self.n_nodes = int(os.environ.get("AURORA_NODES", str(self.n_nodes)))
        self.hidden_dim = int(os.environ.get("AURORA_HIDDEN", str(self.hidden_dim)))
        _dbg("AuroraConfig.__post_init__",
             epochs=self.epochs, n_nodes=self.n_nodes,
             hidden_dim=self.hidden_dim, overrides_applied=True)
        # Aliases for model compatibility
        self.T_in = self.seq_len
        self.T_out = self.pred_len
        self.num_nodes = self.n_nodes


    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def rng(self) -> np.random.Generator:
        """Return a seeded numpy Generator."""
        _dbg("AuroraConfig.rng", seed=self.seed)
        return np.random.default_rng(self.seed)

    def head_dim(self) -> int:
        """Dimension per attention head."""
        assert self.hidden_dim % self.n_heads == 0, (
            f"hidden_dim ({self.hidden_dim}) must be divisible by n_heads ({self.n_heads})"
        )
        d = self.hidden_dim // self.n_heads
        _dbg("AuroraConfig.head_dim", head_dim=d)
        return d

    def summary(self) -> str:
        """One-line human-readable summary."""
        s = (f"Aurora(N={self.n_nodes}, h={self.hidden_dim}, "
             f"L={self.n_layers}, K={self.K_diffusion}, heads={self.n_heads}, "
             f"seq={self.seq_len}â{self.pred_len}, bs={self.batch_size}, "
             f"lr={self.lr}, drop={self.dropout}, epochs={self.epochs})")
        _dbg("AuroraConfig.summary", text=s)
        return s


def make_default_config(**overrides) -> AuroraConfig:
    """Factory that creates a config, optionally patching fields."""
    cfg = AuroraConfig(**overrides)
    _dbg("make_default_config", summary=cfg.summary())
    return cfg


if __name__ == "__main__":
    cfg = make_default_config()
    print(cfg.summary())
    print(f"head_dim = {cfg.head_dim()}")
