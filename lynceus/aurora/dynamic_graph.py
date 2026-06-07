"""Dynamic graph construction module for D2STGNN Aurora.

Learns a soft adjacency from dual node embeddings and supports
KNN-graph construction and adaptive thresholding.
Pure NumPy.  Every public method includes a _dbg() call.
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


def _softmax_rows(Z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stable."""
    Z_shifted = Z - Z.max(axis=-1, keepdims=True)
    e = np.exp(Z_shifted)
    return e / e.sum(axis=-1, keepdims=True)


def _topk_mask(M: np.ndarray, k: int) -> np.ndarray:
    """Return a boolean mask that is True for the top-k values per row."""
    N = M.shape[0]
    mask = np.zeros_like(M, dtype=bool)
    for i in range(N):
        idx = np.argpartition(M[i], -k)[-k:]
        mask[i, idx] = True
    return mask


class DynamicGraphLearner:
    """Learns a sparse, directed adjacency from dual node embeddings.

    The adjacency is computed as
        A = sparsify_top_k( softmax( E1 Â· E2^T ) )

    where E1, E2 â R^{NÃd} are learnable node embeddings.

    Parameters
    ----------
    n_nodes   : number of graph nodes N.
    embed_dim : dimensionality d of each node embedding.
    top_k     : how many neighbours to keep per node.
    seed      : random seed for reproducibility.
    """

    def __init__(self, n_nodes: int, embed_dim: int,
                 top_k: int = 20, seed: int = 42):
        _dbg("DynamicGraphLearner.__init__",
             n_nodes=n_nodes, embed_dim=embed_dim, top_k=top_k, seed=seed)
        self.n_nodes = n_nodes
        self.embed_dim = embed_dim
        self.top_k = min(top_k, n_nodes)
        self.seed = seed

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(embed_dim)
        self.E1 = rng.standard_normal((n_nodes, embed_dim)) * scale
        self.E2 = rng.standard_normal((n_nodes, embed_dim)) * scale
        _dbg("DynamicGraphLearner.__init__.done", E1=self.E1, E2=self.E2)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, node_features: np.ndarray | None = None) -> np.ndarray:
        """Compute sparse soft adjacency  A = topk( softmax(E1 Â· E2^T) ).

        Parameters
        ----------
        node_features : (B, N, C) or (N, C) optional.  Currently unused
                        beyond debug logging (future: gating / modulation).

        Returns
        -------
        A : (N, N) sparse soft adjacency.
        """
        _dbg("DynamicGraphLearner.forward.enter",
             node_features=node_features if node_features is not None else "None")

        logits = self.E1 @ self.E2.T  # (N, N)
        _dbg("DynamicGraphLearner.forward.logits", logits=logits)

        A_soft = _softmax_rows(logits)  # (N, N)
        _dbg("DynamicGraphLearner.forward.softmax", A_soft=A_soft)

        # Sparsify: keep only top-k per row
        mask = _topk_mask(A_soft, self.top_k)
        A = np.where(mask, A_soft, 0.0)

        # Re-normalise rows so they still sum to 1
        row_sums = A.sum(axis=1, keepdims=True)
        A = np.where(row_sums > 0, A / row_sums, 0.0)

        _dbg("DynamicGraphLearner.forward.exit", A=A,
             nnz=int(np.count_nonzero(A)),
             density=float(np.count_nonzero(A)) / (self.n_nodes ** 2))
        return A

    # ------------------------------------------------------------------
    # KNN graph from feature similarity
    # ------------------------------------------------------------------

    def knn_graph(self, features: np.ndarray, k: int) -> np.ndarray:
        """Build a k-nearest-neighbour adjacency from feature cosine similarity.

        Parameters
        ----------
        features : (N, C) node feature matrix.
        k        : number of neighbours.

        Returns
        -------
        A_knn : (N, N) binary adjacency (symmetric).
        """
        _dbg("DynamicGraphLearner.knn_graph.enter", features=features, k=k)
        N = features.shape[0]
        k = min(k, N - 1)

        # Cosine similarity
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        F_normed = features / norms
        sim = F_normed @ F_normed.T  # (N, N)
        np.fill_diagonal(sim, -np.inf)  # exclude self

        _dbg("DynamicGraphLearner.knn_graph.sim", sim=sim)

        # Top-k neighbours per node
        A_knn = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            idx = np.argpartition(sim[i], -k)[-k:]
            A_knn[i, idx] = 1.0

        # Make symmetric (undirected KNN)
        A_knn = np.maximum(A_knn, A_knn.T)
        _dbg("DynamicGraphLearner.knn_graph.exit", A_knn=A_knn,
             nnz=int(A_knn.sum()))
        return A_knn

    # ------------------------------------------------------------------
    # Adaptive threshold
    # ------------------------------------------------------------------

    def adaptive_threshold(self, similarities: np.ndarray,
                           percentile: float = 90.0) -> np.ndarray:
        """Threshold a similarity matrix at a data-driven percentile.

        Parameters
        ----------
        similarities : (N, N) pairwise similarity scores.
        percentile   : keep edges above this percentile (0â100).

        Returns
        -------
        A_thresh : (N, N) binary adjacency after thresholding.
        """
        _dbg("adaptive_threshold.enter", similarities=similarities,
             percentile=percentile)

        # Compute threshold from upper triangle (avoid double-counting)
        iu = np.triu_indices_from(similarities, k=1)
        vals = similarities[iu]
        thresh = np.percentile(vals, percentile)
        _dbg("adaptive_threshold.thresh", thresh=thresh,
             val_min=float(vals.min()), val_max=float(vals.max()))

        A_thresh = (similarities >= thresh).astype(np.float64)
        np.fill_diagonal(A_thresh, 0.0)  # no self-loops

        _dbg("adaptive_threshold.exit", A_thresh=A_thresh,
             nnz=int(A_thresh.sum()))
        return A_thresh

    # ------------------------------------------------------------------
    # Embedding update (simple SGD step)
    # ------------------------------------------------------------------

    def update_embeddings(self, grad: dict[str, np.ndarray],
                          lr: float = 0.001) -> None:
        """Apply a gradient-descent step on the node embeddings.

        Parameters
        ----------
        grad : dict with keys ``'E1'`` and ``'E2'``, each (N, d).
        lr   : learning rate.
        """
        _dbg("update_embeddings.enter", lr=lr,
             grad_E1=grad["E1"], grad_E2=grad["E2"])

        self.E1 -= lr * grad["E1"]
        self.E2 -= lr * grad["E2"]

        _dbg("update_embeddings.exit", E1=self.E1, E2=self.E2)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"DynamicGraphLearner(N={self.n_nodes}, d={self.embed_dim}, "
                f"top_k={self.top_k})")


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    os.environ["AURORA_DEBUG"] = "1"
    rng = np.random.default_rng(7)
    N, D, C = 20, 8, 4

    learner = DynamicGraphLearner(n_nodes=N, embed_dim=D, top_k=5, seed=7)
    print(learner)

    print("\n=== Forward (learned adjacency) ===")
    A = learner.forward()
    print(f"Adjacency shape: {A.shape}, nnz: {np.count_nonzero(A)}")

    print("\n=== KNN graph from features ===")
    feats = rng.standard_normal((N, C))
    A_knn = learner.knn_graph(feats, k=4)
    print(f"KNN adj shape: {A_knn.shape}, edges: {int(A_knn.sum())}")

    print("\n=== Adaptive threshold ===")
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    sim = (feats / np.maximum(norms, 1e-8)) @ (feats / np.maximum(norms, 1e-8)).T
    A_at = learner.adaptive_threshold(sim, percentile=85)
    print(f"Thresholded adj shape: {A_at.shape}, edges: {int(A_at.sum())}")

    print("\n=== Embedding update ===")
    fake_grad = {
        "E1": rng.standard_normal((N, D)) * 0.01,
        "E2": rng.standard_normal((N, D)) * 0.01,
    }
    learner.update_embeddings(fake_grad, lr=0.01)
    print("Embeddings updated.")
