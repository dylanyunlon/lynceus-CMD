"""Graph convolution primitives for D2STGNN Aurora.

Every public function carries a _dbg() call that fires when AURORA_DEBUG=1.
All operations are pure NumPy â no framework dependency.
"""

import os
import numpy as np

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


# ------------------------------------------------------------------
# Laplacian & Chebyshev basis
# ------------------------------------------------------------------

def compute_laplacian(A: np.ndarray) -> np.ndarray:
    """Normalised Laplacian  L = I - D^{-1/2} A D^{-1/2}.

    Parameters
    ----------
    A : (N, N) adjacency matrix (non-negative, may be asymmetric).

    Returns
    -------
    L : (N, N) symmetric normalised Laplacian.
    """
    _dbg("compute_laplacian.enter", A=A)
    A_sym = 0.5 * (A + A.T)
    d = A_sym.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    N = A.shape[0]
    L = np.eye(N) - D_inv_sqrt @ A_sym @ D_inv_sqrt
    _dbg("compute_laplacian.exit", L=L)
    return L


def chebyshev_basis(L: np.ndarray, K: int) -> list[np.ndarray]:
    """First K Chebyshev polynomials T_0 â¦ T_{K-1} of the Laplacian.

    Uses the recurrence  T_k(L) = 2Â·LÂ·T_{k-1}(L) - T_{k-2}(L)
    with scaled LÌ = 2L/Î»_max - I  (Î»_max â 2 for normalised Laplacian).

    Parameters
    ----------
    L : (N, N) normalised Laplacian.
    K : number of basis matrices to return.

    Returns
    -------
    list of K arrays each (N, N).
    """
    _dbg("chebyshev_basis.enter", L=L, K=K)
    N = L.shape[0]
    # Rescale into [-1, 1] spectrum
    lam_max = 2.0  # tight upper bound for normalised Laplacian
    L_tilde = (2.0 / lam_max) * L - np.eye(N)

    basis: list[np.ndarray] = []
    T_prev = np.eye(N)  # T_0
    basis.append(T_prev.copy())
    if K == 1:
        _dbg("chebyshev_basis.exit", n_basis=len(basis))
        return basis

    T_curr = L_tilde.copy()  # T_1
    basis.append(T_curr.copy())

    for k in range(2, K):
        T_next = 2.0 * L_tilde @ T_curr - T_prev
        basis.append(T_next.copy())
        T_prev, T_curr = T_curr, T_next
        _dbg("chebyshev_basis.iter", k=k, T_next=T_next)

    _dbg("chebyshev_basis.exit", n_basis=len(basis))
    return basis


# ------------------------------------------------------------------
# Transition matrices for diffusion convolution
# ------------------------------------------------------------------

def _transition_matrices(A: np.ndarray):
    """Forward and backward random-walk transition matrices."""
    d_fwd = A.sum(axis=1, keepdims=True)
    P_fwd = np.where(d_fwd > 0, A / d_fwd, 0.0)

    d_bwd = A.sum(axis=0, keepdims=True)
    P_bwd = np.where(d_bwd > 0, A / d_bwd, 0.0)

    _dbg("_transition_matrices", P_fwd=P_fwd, P_bwd=P_bwd)
    return P_fwd, P_bwd


# ------------------------------------------------------------------
# Core convolution operations
# ------------------------------------------------------------------

def diffusion_conv(X: np.ndarray, A: np.ndarray, K: int) -> np.ndarray:
    """K-order diffusion convolution on adjacency A.

    Collects power-iteration supports from both forward and backward
    transition matrices and concatenates them along the channel axis.

    Parameters
    ----------
    X : (B, N, C) node feature tensor.
    A : (N, N) adjacency matrix.
    K : diffusion order (number of hops).

    Returns
    -------
    out : (B, N, C * 2K) concatenated diffusion features.
    """
    _dbg("diffusion_conv.enter", X=X, A=A, K=K)
    B, N, C = X.shape
    P_fwd, P_bwd = _transition_matrices(A)

    supports: list[np.ndarray] = []

    # Forward diffusion powers
    Z_fwd = X.copy()  # (B, N, C) â P^0 X = X
    for k in range(K):
        Z_fwd = np.einsum("mn,bnc->bmc", P_fwd, Z_fwd)
        supports.append(Z_fwd.copy())
        _dbg("diffusion_conv.fwd", k=k, Z_fwd=Z_fwd)

    # Backward diffusion powers
    Z_bwd = X.copy()
    for k in range(K):
        Z_bwd = np.einsum("mn,bnc->bmc", P_bwd, Z_bwd)
        supports.append(Z_bwd.copy())
        _dbg("diffusion_conv.bwd", k=k, Z_bwd=Z_bwd)

    out = np.concatenate(supports, axis=-1)  # (B, N, C*2K)
    _dbg("diffusion_conv.exit", out=out)
    return out


def gcn_layer(X: np.ndarray, A: np.ndarray,
              W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Single GCN layer with residual connection.

    Y = Ï(Ã Â· X Â· W + b) + X_proj
    where Ã = DÌ^{-1/2} Ã DÌ^{-1/2},  Ã = A + I.

    Parameters
    ----------
    X : (B, N, C_in)
    A : (N, N) adjacency.
    W : (C_in, C_out) weight matrix.
    b : (C_out,) bias.

    Returns
    -------
    Y : (B, N, C_out)
    """
    _dbg("gcn_layer.enter", X=X, A=A, W=W, b=b)
    N = A.shape[0]
    C_in, C_out = W.shape

    # Self-loop augmented adjacency
    A_tilde = A + np.eye(N)
    d = A_tilde.sum(axis=1)
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    # Message passing + linear
    H = np.einsum("mn,bnc->bmc", A_hat, X)  # (B, N, C_in)
    Z = H @ W + b  # (B, N, C_out)

    # ReLU activation
    Z = np.maximum(Z, 0.0)

    # Residual: project X if dimensions mismatch
    if C_in == C_out:
        residual = X
    else:
        # Simple linear projection for residual path
        W_res = np.eye(C_in, C_out)
        residual = X @ W_res
        _dbg("gcn_layer.residual_proj", W_res=W_res)

    Y = Z + residual
    _dbg("gcn_layer.exit", Y=Y)
    return Y


def multi_scale_gcn(X: np.ndarray, A: np.ndarray,
                    weights: list[tuple[np.ndarray, np.ndarray]],
                    K_list: list[int]) -> np.ndarray:
    """Multi-scale graph convolution at different diffusion orders.

    For each scale k in K_list, a separate diffusion convolution is
    performed, followed by a learned linear projection. Outputs are
    summed across scales.

    Parameters
    ----------
    X       : (B, N, C_in) input features.
    A       : (N, N) adjacency.
    weights : list of (W_k, b_k) per scale.
              W_k : (C_in * 2k, C_out),  b_k : (C_out,).
    K_list  : list of diffusion orders, e.g. [1, 2, 3].

    Returns
    -------
    out : (B, N, C_out) aggregated multi-scale features.
    """
    _dbg("multi_scale_gcn.enter", X=X, A=A, n_scales=len(K_list), K_list=K_list)
    assert len(weights) == len(K_list), "Need one (W, b) pair per scale."

    accumulated = None
    for idx, K in enumerate(K_list):
        W_k, b_k = weights[idx]
        D = diffusion_conv(X, A, K)      # (B, N, C_in * 2K)
        Z = D @ W_k + b_k               # (B, N, C_out)
        Z = np.maximum(Z, 0.0)          # ReLU
        _dbg("multi_scale_gcn.scale", idx=idx, K=K, Z=Z)
        accumulated = Z if accumulated is None else accumulated + Z

    _dbg("multi_scale_gcn.exit", out=accumulated)
    return accumulated


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    os.environ["AURORA_DEBUG"] = "1"
    rng = np.random.default_rng(0)
    B, N, C = 2, 10, 4

    A = (rng.random((N, N)) > 0.7).astype(np.float64)
    np.fill_diagonal(A, 0)
    X = rng.standard_normal((B, N, C))

    print("=== Laplacian ===")
    L = compute_laplacian(A)

    print("\n=== Chebyshev basis K=3 ===")
    basis = chebyshev_basis(L, K=3)

    print("\n=== Diffusion conv K=2 ===")
    D = diffusion_conv(X, A, K=2)
    print(f"diffusion output shape: {D.shape}")

    print("\n=== GCN layer ===")
    W = rng.standard_normal((C, 8)) * 0.1
    b = np.zeros(8)
    Y = gcn_layer(X, A, W, b)
    print(f"GCN output shape: {Y.shape}")

    print("\n=== Multi-scale GCN ===")
    K_list = [1, 2]
    weights = [
        (rng.standard_normal((C * 2 * 1, 8)) * 0.1, np.zeros(8)),
        (rng.standard_normal((C * 2 * 2, 8)) * 0.1, np.zeros(8)),
    ]
    M = multi_scale_gcn(X, A, weights, K_list)
    print(f"Multi-scale output shape: {M.shape}")
