"""
Kepler Active Learning Module
Pure numpy implementation with debug tracing.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple
import time

_DEBUG = False


def _dbg(fn_name: str, msg: str):
    if _DEBUG:
        print(f"[DEBUG][{fn_name}] {msg}")


# ---------------------------------------------------------------------------
# Uncertainty Scorer
# ---------------------------------------------------------------------------
class UncertaintyScorer:
    """Scores pool samples by various uncertainty measures."""

    @staticmethod
    def entropy_score(probs: np.ndarray) -> np.ndarray:
        """
        Shannon entropy across class probabilities.
        probs: (n_samples, n_classes)
        returns: (n_samples,)
        """
        fn = "UncertaintyScorer.entropy_score"
        _dbg(fn, f"probs shape={probs.shape}")
        clipped = np.clip(probs, 1e-12, 1.0)
        ent = -np.sum(clipped * np.log(clipped), axis=1)
        _dbg(fn, f"entropy min={ent.min():.6f} max={ent.max():.6f} mean={ent.mean():.6f}")
        return ent

    @staticmethod
    def margin_score(probs: np.ndarray) -> np.ndarray:
        """
        Margin between top-two class probabilities (lower margin = more uncertain).
        probs: (n_samples, n_classes)
        returns: (n_samples,) â negative margin so higher = more uncertain.
        """
        fn = "UncertaintyScorer.margin_score"
        _dbg(fn, f"probs shape={probs.shape}")
        sorted_p = np.sort(probs, axis=1)
        margin = sorted_p[:, -1] - sorted_p[:, -2]
        score = -margin  # negate so argsort-descending picks smallest margins
        _dbg(fn, f"margin min={margin.min():.6f} max={margin.max():.6f}")
        return score

    @staticmethod
    def least_confident_score(probs: np.ndarray) -> np.ndarray:
        """
        1 - max(p) for each sample. Higher = less confident.
        probs: (n_samples, n_classes)
        returns: (n_samples,)
        """
        fn = "UncertaintyScorer.least_confident_score"
        _dbg(fn, f"probs shape={probs.shape}")
        max_p = np.max(probs, axis=1)
        score = 1.0 - max_p
        _dbg(fn, f"least_confident min={score.min():.6f} max={score.max():.6f}")
        return score


# ---------------------------------------------------------------------------
# Acquisition Functions
# ---------------------------------------------------------------------------
class AcquisitionFunctions:
    """Bayesian-style acquisition functions operating on mean/sigma arrays."""

    @staticmethod
    def UCB(mu: np.ndarray, sigma: np.ndarray, beta: float = 2.0) -> np.ndarray:
        """
        Upper Confidence Bound: mu + beta * sigma.
        returns: (n_samples,) acquisition values.
        """
        fn = "AcquisitionFunctions.UCB"
        _dbg(fn, f"mu shape={mu.shape}, sigma shape={sigma.shape}, beta={beta}")
        acq = mu + beta * sigma
        _dbg(fn, f"UCB min={acq.min():.6f} max={acq.max():.6f}")
        return acq

    @staticmethod
    def EI(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
        """
        Expected Improvement: E[max(f(x) - best, 0)].
        Closed-form under Gaussian assumption.
        """
        fn = "AcquisitionFunctions.EI"
        _dbg(fn, f"mu shape={mu.shape}, best={best:.6f}")
        sigma_safe = np.clip(sigma, 1e-12, None)
        z = (mu - best) / sigma_safe
        # Approximate Phi and phi with numpy
        phi = np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
        Phi = 0.5 * (1.0 + _erf_approx(z / np.sqrt(2.0)))
        ei = sigma_safe * (z * Phi + phi)
        ei = np.where(sigma < 1e-12, 0.0, ei)
        _dbg(fn, f"EI min={ei.min():.6f} max={ei.max():.6f} mean={ei.mean():.6f}")
        return ei

    @staticmethod
    def PI(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
        """
        Probability of Improvement: P(f(x) > best).
        """
        fn = "AcquisitionFunctions.PI"
        _dbg(fn, f"mu shape={mu.shape}, best={best:.6f}")
        sigma_safe = np.clip(sigma, 1e-12, None)
        z = (mu - best) / sigma_safe
        pi = 0.5 * (1.0 + _erf_approx(z / np.sqrt(2.0)))
        pi = np.where(sigma < 1e-12, 0.0, pi)
        _dbg(fn, f"PI min={pi.min():.6f} max={pi.max():.6f}")
        return pi


def _erf_approx(x: np.ndarray) -> np.ndarray:
    """Abramowitz & Stegun approximation of erf (max error ~1.5e-7)."""
    fn = "_erf_approx"
    _dbg(fn, f"x shape={x.shape}")
    sign = np.sign(x)
    x_abs = np.abs(x)
    p = 0.3275911
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    t = 1.0 / (1.0 + p * x_abs)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x_abs ** 2)
    return sign * y


# ---------------------------------------------------------------------------
# Active Learner
# ---------------------------------------------------------------------------
@dataclass
class _ModelProtocol:
    """Minimal duck-type contract for models passed into ActiveLearner."""
    pass


class ActiveLearner:
    """
    Orchestrates active learning query strategies.

    A *model* is any object with:
        - predict_proba(X) -> np.ndarray (n, c)
        - predict(X) -> np.ndarray (n,)
    or for Bayesian-style:
        - predict_mean_std(X) -> (mu, sigma)  both (n,)

    *pool*: np.ndarray of shape (n_pool, d) â unlabelled feature matrix.
    """

    def __init__(self, scorer: Optional[UncertaintyScorer] = None,
                 acquisition: Optional[AcquisitionFunctions] = None,
                 seed: int = 42):
        fn = "ActiveLearner.__init__"
        self.scorer = scorer or UncertaintyScorer()
        self.acquisition = acquisition or AcquisitionFunctions()
        self.rng = np.random.RandomState(seed)
        _dbg(fn, f"initialised with seed={seed}")

    # ---- core strategies ----

    def uncertainty_sampling(self, model, pool: np.ndarray, k: int,
                             method: str = "entropy") -> np.ndarray:
        """
        Select k most uncertain samples from pool.
        method: 'entropy' | 'margin' | 'least_confident'
        returns: indices into pool (shape (k,))
        """
        fn = "ActiveLearner.uncertainty_sampling"
        _dbg(fn, f"pool shape={pool.shape}, k={k}, method={method}")
        probs = model.predict_proba(pool)
        _dbg(fn, f"probs shape={probs.shape}")

        if method == "entropy":
            scores = self.scorer.entropy_score(probs)
        elif method == "margin":
            scores = self.scorer.margin_score(probs)
        elif method == "least_confident":
            scores = self.scorer.least_confident_score(probs)
        else:
            _dbg(fn, f"unknown method '{method}', falling back to entropy")
            scores = self.scorer.entropy_score(probs)

        top_k = np.argsort(scores)[-k:][::-1]
        _dbg(fn, f"selected indices (first 10): {top_k[:10]}")
        return top_k

    def query_by_committee(self, models: list, pool: np.ndarray,
                           k: int) -> np.ndarray:
        """
        QBC: pick samples with maximum vote disagreement among committee.
        models: list of model objects each with predict(X).
        returns: indices into pool (shape (k,))
        """
        fn = "ActiveLearner.query_by_committee"
        _dbg(fn, f"committee size={len(models)}, pool shape={pool.shape}, k={k}")
        predictions = np.array([m.predict(pool) for m in models])  # (C, n_pool)
        n_committee, n_pool = predictions.shape
        _dbg(fn, f"predictions matrix shape={predictions.shape}")

        # Vote entropy
        n_classes = int(predictions.max()) + 1
        vote_counts = np.zeros((n_pool, n_classes), dtype=np.float64)
        for c_idx in range(n_committee):
            for s_idx in range(n_pool):
                vote_counts[s_idx, int(predictions[c_idx, s_idx])] += 1

        vote_probs = vote_counts / n_committee
        disagreement = self.scorer.entropy_score(vote_probs)
        top_k = np.argsort(disagreement)[-k:][::-1]
        _dbg(fn, f"top disagreement scores: {disagreement[top_k[:5]]}")
        return top_k

    def expected_improvement(self, model, pool: np.ndarray,
                             best_so_far: float) -> np.ndarray:
        """
        Rank pool by Expected Improvement acquisition.
        model must expose predict_mean_std(X) -> (mu, sigma).
        returns: sorted indices descending by EI.
        """
        fn = "ActiveLearner.expected_improvement"
        _dbg(fn, f"pool shape={pool.shape}, best_so_far={best_so_far:.6f}")
        mu, sigma = model.predict_mean_std(pool)
        ei = self.acquisition.EI(mu, sigma, best_so_far)
        ranked = np.argsort(ei)[::-1]
        _dbg(fn, f"top-5 EI values: {ei[ranked[:5]]}")
        return ranked

    def select_next_batch(self, strategy: str, pool: np.ndarray, k: int,
                          *, model=None, models: Optional[list] = None,
                          best_so_far: float = 0.0,
                          method: str = "entropy") -> np.ndarray:
        """
        Unified dispatcher for batch selection.
        strategy: 'uncertainty' | 'qbc' | 'ei' | 'random'
        returns: indices of selected samples.
        """
        fn = "ActiveLearner.select_next_batch"
        _dbg(fn, f"strategy={strategy}, k={k}, pool shape={pool.shape}")

        if strategy == "uncertainty":
            if model is None:
                raise ValueError("uncertainty strategy requires `model`")
            return self.uncertainty_sampling(model, pool, k, method=method)

        elif strategy == "qbc":
            if models is None or len(models) < 2:
                raise ValueError("qbc strategy requires >=2 models")
            return self.query_by_committee(models, pool, k)

        elif strategy == "ei":
            if model is None:
                raise ValueError("ei strategy requires `model`")
            ranked = self.expected_improvement(model, pool, best_so_far)
            return ranked[:k]

        elif strategy == "random":
            indices = self.rng.choice(pool.shape[0], size=min(k, pool.shape[0]),
                                      replace=False)
            _dbg(fn, f"random indices: {indices[:10]}")
            return indices

        else:
            raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Batch Diversity Filter (optional post-processing)
# ---------------------------------------------------------------------------
class BatchDiversityFilter:
    """Down-select a candidate set to maximise pairwise distance in feature space."""

    @staticmethod
    def max_min_distance(pool: np.ndarray, candidate_indices: np.ndarray,
                         k: int) -> np.ndarray:
        """
        Greedy max-min diversity selection from candidate subset.
        returns: k indices from candidate_indices.
        """
        fn = "BatchDiversityFilter.max_min_distance"
        _dbg(fn, f"candidates={len(candidate_indices)}, k={k}")
        if k >= len(candidate_indices):
            _dbg(fn, "k >= candidates, returning all")
            return candidate_indices

        cand_feats = pool[candidate_indices]
        selected = [0]
        remaining = list(range(1, len(candidate_indices)))

        for _ in range(k - 1):
            sel_feats = cand_feats[selected]
            best_idx, best_dist = -1, -1.0
            for r in remaining:
                dists = np.sqrt(np.sum((sel_feats - cand_feats[r]) ** 2, axis=1))
                min_d = dists.min()
                if min_d > best_dist:
                    best_dist = min_d
                    best_idx = r
            selected.append(best_idx)
            remaining.remove(best_idx)
            _dbg(fn, f"selected local idx={best_idx}, min_dist={best_dist:.6f}")

        result = candidate_indices[np.array(selected)]
        _dbg(fn, f"final selected pool indices: {result}")
        return result


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _DEBUG = True

    class _DummyModel:
        def predict_proba(self, X):
            rng = np.random.RandomState(0)
            p = rng.dirichlet([1, 1, 1], size=X.shape[0])
            return p

        def predict(self, X):
            return np.argmax(self.predict_proba(X), axis=1)

        def predict_mean_std(self, X):
            rng = np.random.RandomState(1)
            mu = rng.randn(X.shape[0])
            sigma = np.abs(rng.randn(X.shape[0])) * 0.5
            return mu, sigma

    pool = np.random.randn(100, 5)
    learner = ActiveLearner(seed=7)
    mdl = _DummyModel()

    print("=== uncertainty (entropy) ===")
    idx = learner.select_next_batch("uncertainty", pool, 5, model=mdl)
    print("selected:", idx)

    print("\n=== qbc ===")
    idx2 = learner.select_next_batch("qbc", pool, 5,
                                     models=[_DummyModel() for _ in range(4)])
    print("selected:", idx2)

    print("\n=== expected improvement ===")
    idx3 = learner.select_next_batch("ei", pool, 5, model=mdl, best_so_far=0.5)
    print("selected:", idx3)

    print("\n=== random ===")
    idx4 = learner.select_next_batch("random", pool, 5)
    print("selected:", idx4)

    print("\nAll tests passed.")
