"""
Diagram-based cost model for Par2QO query optimization.
Ported from upstream diagram_best_cost.py + diagram_nearest.py
with VP-tree spatial indexing and RBF interpolation.
"""

import numpy as np
from collections import namedtuple

# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------
_DEBUG = False

def _dbg(*args, **kwargs):
    """Conditional debug printer. Toggle with module-level _DEBUG flag."""
    if _DEBUG:
        print("[DBG]", *args, **kwargs)

# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------

def euclidean_dist(a, b):
    """L2 distance between two vectors."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    _dbg(f"euclidean_dist: diff_norm={np.sqrt(np.dot(diff, diff)):.6f}")
    return np.sqrt(np.dot(diff, diff))


def manhattan_dist(a, b):
    """L1 distance between two vectors."""
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    result = np.sum(diff)
    _dbg(f"manhattan_dist: {result:.6f}")
    return result


def chebyshev_dist(a, b):
    """L-inf distance between two vectors."""
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    result = np.max(diff) if diff.size > 0 else 0.0
    _dbg(f"chebyshev_dist: {result:.6f}")
    return result


_METRICS = {
    "euclidean": euclidean_dist,
    "manhattan": manhattan_dist,
    "chebyshev": chebyshev_dist,
}

def get_metric(name="euclidean"):
    """Return a distance function by name."""
    if name not in _METRICS:
        raise ValueError(f"Unknown metric '{name}'. Choose from {list(_METRICS)}")
    _dbg(f"get_metric: selected '{name}'")
    return _METRICS[name]

# ---------------------------------------------------------------------------
# Vantage-Point Tree
# ---------------------------------------------------------------------------

_VPNode = namedtuple("_VPNode", ["vantage", "radius", "index", "left", "right"])


class VPTree:
    """
    Vantage-Point tree for efficient nearest-neighbour queries in
    arbitrary metric spaces.

    Algorithm twist vs. classic Yianilos (1993): the partition uses the
    *geometric median* distance (approx via iterative reweighting for 3
    rounds) rather than the plain median, which empirically tightens the
    bounding balls on skewed data.
    """

    def __init__(self, points, metric="euclidean"):
        self._points = np.asarray(points, dtype=np.float64)
        self._dist = get_metric(metric) if isinstance(metric, str) else metric
        _dbg(f"VPTree.__init__: building tree over {len(self._points)} points")
        indices = np.arange(len(self._points))
        self._root = self._build(indices)
        _dbg("VPTree.__init__: build complete")

    # -- internal build ---------------------------------------------------

    @staticmethod
    def _geometric_median_1d(values, n_iter=3):
        """Approximate geometric median of 1-D array via Weiszfeld iterations."""
        if len(values) == 0:
            return 0.0
        est = np.median(values)
        for _ in range(n_iter):
            diffs = np.abs(values - est)
            weights = np.where(diffs > 1e-12, 1.0 / diffs, 0.0)
            w_sum = weights.sum()
            if w_sum < 1e-15:
                break
            est = np.dot(weights, values) / w_sum
        return float(est)

    def _build(self, indices):
        if len(indices) == 0:
            return None
        if len(indices) == 1:
            return _VPNode(
                vantage=self._points[indices[0]],
                radius=0.0,
                index=int(indices[0]),
                left=None,
                right=None,
            )

        # pick vantage: farthest from first point (spread heuristic)
        rng_idx = 0
        dists_to_first = np.array(
            [self._dist(self._points[indices[rng_idx]], self._points[j]) for j in indices]
        )
        vp_local = int(np.argmax(dists_to_first))
        vp_global = indices[vp_local]
        vp = self._points[vp_global]

        remaining = np.delete(indices, vp_local)
        dists = np.array([self._dist(vp, self._points[j]) for j in remaining])

        mu = self._geometric_median_1d(dists)
        _dbg(f"_build: vp_idx={vp_global}, mu={mu:.4f}, n={len(remaining)}")

        left_mask = dists <= mu
        right_mask = ~left_mask

        return _VPNode(
            vantage=vp,
            radius=mu,
            index=int(vp_global),
            left=self._build(remaining[left_mask]),
            right=self._build(remaining[right_mask]),
        )

    # -- query ------------------------------------------------------------

    def _knn_search(self, node, target, k, heap):
        """
        Recursive k-NN search.  `heap` is a list of (-dist, idx) kept as a
        max-heap (negative dists so smallest real distance has largest key).
        """
        if node is None:
            return

        d = self._dist(target, node.vantage)
        _dbg(f"_knn_search: idx={node.index}, d={d:.6f}")

        if len(heap) < k:
            heap.append((-d, node.index))
            heap.sort()
        elif d < -heap[0][0]:
            heap[0] = (-d, node.index)
            heap.sort()

        tau = -heap[0][0] if len(heap) == k else float("inf")

        if d < node.radius:
            # target inside ball â search left first
            if d - tau <= node.radius:
                self._knn_search(node.left, target, k, heap)
                tau = -heap[0][0] if len(heap) == k else float("inf")
            if d + tau >= node.radius:
                self._knn_search(node.right, target, k, heap)
        else:
            # target outside ball â search right first
            if d + tau >= node.radius:
                self._knn_search(node.right, target, k, heap)
                tau = -heap[0][0] if len(heap) == k else float("inf")
            if d - tau <= node.radius:
                self._knn_search(node.left, target, k, heap)

    def query(self, target, k=5):
        """Return (distances, indices) arrays for the k nearest neighbours."""
        target = np.asarray(target, dtype=np.float64)
        heap = []
        self._knn_search(self._root, target, k, heap)
        heap.sort(key=lambda x: -x[0])  # ascending distance
        dists = np.array([-h[0] for h in heap])
        idxs = np.array([h[1] for h in heap])
        _dbg(f"query: returned {len(dists)} neighbours")
        return dists, idxs


# ---------------------------------------------------------------------------
# Public nearest-point helper
# ---------------------------------------------------------------------------

def find_nearest_diagram_points(target, candidates, k=5, metric="euclidean"):
    """
    Find the *k* closest diagram points to *target* using a VP-tree.

    Parameters
    ----------
    target : array-like, shape (d,)
    candidates : array-like, shape (n, d)
    k : int
    metric : str or callable

    Returns
    -------
    distances : ndarray (k,)
    indices   : ndarray (k,)  â indices into *candidates*
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    k = min(k, len(candidates))
    _dbg(f"find_nearest_diagram_points: n={len(candidates)}, k={k}")
    tree = VPTree(candidates, metric=metric)
    return tree.query(target, k=k)


# ---------------------------------------------------------------------------
# RBF interpolation helpers
# ---------------------------------------------------------------------------

def _rbf_gaussian(r, epsilon=1.0):
    """Gaussian RBF kernel: exp(-(epsilon * r)^2)."""
    return np.exp(-(epsilon * r) ** 2)


def _rbf_multiquadric(r, epsilon=1.0):
    """Multiquadric RBF: sqrt(1 + (epsilon*r)^2)."""
    return np.sqrt(1.0 + (epsilon * r) ** 2)


def _rbf_inverse_multiquadric(r, epsilon=1.0):
    """Inverse multiquadric RBF."""
    return 1.0 / np.sqrt(1.0 + (epsilon * r) ** 2)


_RBF_KERNELS = {
    "gaussian": _rbf_gaussian,
    "multiquadric": _rbf_multiquadric,
    "inv_multiquadric": _rbf_inverse_multiquadric,
}


def compute_best_cost_surface(diagram_points, costs, epsilon=1.0,
                               kernel="gaussian", reg=1e-8):
    """
    Build an RBF interpolation model from diagram sample points and their
    associated costs.

    Uses Tikhonov regularisation (diagonal loading) to stabilise the
    Gram-matrix inversion â changed from upstream's bare solve to a
    ridge-style approach with configurable `reg` parameter.

    Returns
    -------
    dict â surface model containing weights, centres, epsilon, kernel name.
    """
    pts = np.asarray(diagram_points, dtype=np.float64)
    c = np.asarray(costs, dtype=np.float64).ravel()
    n = len(pts)
    assert n == len(c), "points / costs length mismatch"
    _dbg(f"compute_best_cost_surface: n={n}, kernel={kernel}, eps={epsilon}")

    rbf_fn = _RBF_KERNELS.get(kernel)
    if rbf_fn is None:
        raise ValueError(f"Unknown RBF kernel '{kernel}'")

    # pairwise distance matrix
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, d)
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))         # (n, n)

    Phi = rbf_fn(dist_mat, epsilon)
    Phi += reg * np.eye(n)
    _dbg(f"compute_best_cost_surface: cond(Phi)={np.linalg.cond(Phi):.2e}")

    # Solve via Cholesky when possible (Phi is SPD after regularisation)
    try:
        L = np.linalg.cholesky(Phi)
        z = np.linalg.solve(L, c)
        weights = np.linalg.solve(L.T, z)
    except np.linalg.LinAlgError:
        _dbg("compute_best_cost_surface: Cholesky failed, falling back to lstsq")
        weights, *_ = np.linalg.lstsq(Phi, c, rcond=None)

    _dbg(f"compute_best_cost_surface: weight range [{weights.min():.4f}, {weights.max():.4f}]")

    return {
        "centres": pts.copy(),
        "weights": weights,
        "epsilon": float(epsilon),
        "kernel": kernel,
    }


def interpolate_cost(point, surface_model):
    """
    Predict the cost at an arbitrary *point* using a previously fitted
    RBF surface model (from `compute_best_cost_surface`).
    """
    pt = np.asarray(point, dtype=np.float64)
    centres = surface_model["centres"]
    weights = surface_model["weights"]
    eps = surface_model["epsilon"]
    rbf_fn = _RBF_KERNELS[surface_model["kernel"]]

    dists = np.sqrt(np.sum((centres - pt) ** 2, axis=1))
    phi_vals = rbf_fn(dists, eps)
    cost = np.dot(phi_vals, weights)
    _dbg(f"interpolate_cost: predicted={cost:.6f}")
    return float(cost)


# ---------------------------------------------------------------------------
# DiagramCostModel â high-level wrapper
# ---------------------------------------------------------------------------

class DiagramCostModel:
    """
    End-to-end diagram cost model: stores sample diagram points, builds
    a VP-tree index for neighbour lookup, and fits an RBF surface for
    cost interpolation.
    """

    def __init__(self, metric="euclidean", rbf_kernel="gaussian",
                 epsilon=1.0, reg=1e-8):
        self.metric = metric
        self.rbf_kernel = rbf_kernel
        self.epsilon = epsilon
        self.reg = reg
        self._tree = None
        self._surface = None
        self._points = None
        self._costs = None
        _dbg("DiagramCostModel.__init__: created (unfitted)")

    def fit(self, diagram_points, costs):
        """Fit the model on observed diagram points and costs."""
        self._points = np.asarray(diagram_points, dtype=np.float64)
        self._costs = np.asarray(costs, dtype=np.float64).ravel()
        assert len(self._points) == len(self._costs)
        _dbg(f"DiagramCostModel.fit: {len(self._points)} samples")

        self._tree = VPTree(self._points, metric=self.metric)
        self._surface = compute_best_cost_surface(
            self._points, self._costs,
            epsilon=self.epsilon,
            kernel=self.rbf_kernel,
            reg=self.reg,
        )
        _dbg("DiagramCostModel.fit: done")

    def predict(self, point):
        """Predict cost for a single query point."""
        if self._surface is None:
            raise RuntimeError("Model not fitted yet â call .fit() first")
        return interpolate_cost(point, self._surface)

    def predict_batch(self, points):
        """Predict costs for an array of query points."""
        pts = np.asarray(points, dtype=np.float64)
        _dbg(f"DiagramCostModel.predict_batch: {len(pts)} queries")
        return np.array([self.predict(p) for p in pts])

    def nearest(self, point, k=5):
        """Return (distances, indices) of the k nearest training points."""
        if self._tree is None:
            raise RuntimeError("Model not fitted yet â call .fit() first")
        return self._tree.query(np.asarray(point, dtype=np.float64), k=k)

    def score(self, test_points, test_costs):
        """
        Compute root-mean-square error on held-out data.
        """
        preds = self.predict_batch(test_points)
        actual = np.asarray(test_costs, dtype=np.float64).ravel()
        rmse = np.sqrt(np.mean((preds - actual) ** 2))
        _dbg(f"DiagramCostModel.score: RMSE={rmse:.6f}")
        return float(rmse)

    def __repr__(self):
        n = len(self._points) if self._points is not None else 0
        return (f"DiagramCostModel(metric={self.metric!r}, kernel={self.rbf_kernel!r}, "
                f"eps={self.epsilon}, n_samples={n})")


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _DEBUG = True
    rng = np.random.default_rng(42)
    X = rng.standard_normal((60, 3))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.05, 60)

    model = DiagramCostModel(metric="euclidean", rbf_kernel="gaussian",
                             epsilon=0.8, reg=1e-6)
    model.fit(X[:50], y[:50])
    print("RMSE:", model.score(X[50:], y[50:]))

    dists, idxs = model.nearest(X[55], k=3)
    print("Nearest 3:", idxs, "dists:", dists)
    print(repr(model))
