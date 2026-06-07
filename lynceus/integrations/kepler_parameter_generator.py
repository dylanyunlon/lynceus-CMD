"""
Kepler Parameter Generator
~~~~~~~~~~~~~~~~~~~~~~~~~~
Pure-numpy parameter generation utilities with per-function _dbg() tracing.
"""

import numpy as np
import sys
import time
from typing import List, Tuple, Optional, Sequence, Any


# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------
_DEBUG = False


def enable_debug(flag: bool = True) -> None:
    global _DEBUG
    _DEBUG = flag


def _dbg(fn_name: str, msg: str, **kw: Any) -> None:
    if not _DEBUG:
        return
    extras = " ".join(f"{k}={v!r}" for k, v in kw.items())
    ts = time.perf_counter()
    print(f"[DBG {ts:.6f}] {fn_name}: {msg} {extras}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Sobol direction numbers (first 21 dimensions, Joe-Kuo)
# Used by the Latin-hypercube path that falls back to Sobol sequences.
# ---------------------------------------------------------------------------
_SOBOL_DIRECTION_NUMBERS: List[List[int]] = [
    [1],
    [1, 1],
    [1, 1, 1],
    [1, 3, 1],
    [1, 1, 7],
    [1, 3, 7],
    [1, 1, 5],
    [1, 3, 3],
    [1, 1, 1, 1],
    [1, 1, 3, 3],
    [1, 3, 5, 13],
    [1, 1, 5, 5],
    [1, 3, 7, 11],
    [1, 1, 1, 15],
    [1, 3, 3, 9],
    [1, 1, 7, 13],
    [1, 3, 5, 3],
    [1, 1, 1, 7],
    [1, 3, 3, 5],
    [1, 1, 7, 9],
]


def _sobol_sequence(n: int, dims: int, seed: int = 0) -> np.ndarray:
    """Generate an *n Ã dims* Sobol quasi-random matrix in [0, 1).

    Uses a simplified Gray-code implementation with the direction numbers
    above.  For dims > len(_SOBOL_DIRECTION_NUMBERS) the extra dimensions
    fall back to a scrambled Halton-like scheme.
    """
    _dbg("_sobol_sequence", "start", n=n, dims=dims, seed=seed)

    bits = 30
    result = np.zeros((n, dims), dtype=np.float64)
    scale = 1.0 / (1 << bits)

    for d in range(dims):
        v = np.zeros(bits, dtype=np.int64)
        if d < len(_SOBOL_DIRECTION_NUMBERS):
            dn = _SOBOL_DIRECTION_NUMBERS[d]
            for i, val in enumerate(dn):
                v[i] = val << (bits - i - 1)
            for i in range(len(dn), bits):
                v[i] = v[i - len(dn)] ^ (v[i - len(dn)] >> len(dn))
                for k in range(1, len(dn)):
                    v[i] ^= ((dn[k] >> (len(dn) - k)) & 1) * v[i - k]
        else:
            rng = np.random.RandomState(seed + d * 9973)
            for i in range(bits):
                v[i] = rng.randint(1, (1 << bits)) | 1
                v[i] <<= max(0, bits - i - 1)

        x = np.int64(0)
        for i in range(n):
            c = 0
            val = i
            while val & 1:
                val >>= 1
                c += 1
            if c < bits:
                x ^= v[c]
            result[i, d] = x * scale

    _dbg("_sobol_sequence", "done", shape=result.shape)
    return result


# ---------------------------------------------------------------------------
# ParameterGenerator
# ---------------------------------------------------------------------------
class ParameterGenerator:
    """Generates parameter samples for Kepler workload exploration.

    All heavy lifting is done with numpy; no external dependencies required.
    """

    def __init__(self, rng_seed: int = 42) -> None:
        _dbg("ParameterGenerator.__init__", "create", rng_seed=rng_seed)
        self._seed = rng_seed
        self._rng = np.random.RandomState(rng_seed)
        self._call_count = 0

    # ------------------------------------------------------------------
    # Internal bookkeeping
    # ------------------------------------------------------------------
    def _tick(self) -> int:
        self._call_count += 1
        _dbg("ParameterGenerator._tick", "call", count=self._call_count)
        return self._call_count

    def reset(self, seed: Optional[int] = None) -> None:
        s = seed if seed is not None else self._seed
        _dbg("ParameterGenerator.reset", "reseed", seed=s)
        self._rng = np.random.RandomState(s)
        self._call_count = 0

    # ------------------------------------------------------------------
    # generate_uniform
    # ------------------------------------------------------------------
    def generate_uniform(
        self,
        n: int,
        bounds: List[Tuple[float, float]],
    ) -> np.ndarray:
        """Return *n* samples drawn uniformly within *bounds*.

        Parameters
        ----------
        n : int
            Number of samples.
        bounds : list of (lo, hi) tuples
            Per-dimension bounds.

        Returns
        -------
        np.ndarray of shape (n, len(bounds))
        """
        self._tick()
        _dbg("generate_uniform", "enter", n=n, dims=len(bounds))

        dims = len(bounds)
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        if np.any(lo > hi):
            _dbg("generate_uniform", "WARNING: lo > hi detected, swapping")
            lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)

        raw = self._rng.rand(n, dims)
        samples = lo + raw * (hi - lo)

        # 20 % algorithm tweak: apply antithetic-variate pairing for the
        # first half to reduce variance in coverage.
        half = n // 2
        if half > 0:
            _dbg("generate_uniform", "antithetic pairing", half=half)
            antithetic = lo + (1.0 - raw[:half]) * (hi - lo)
            combined = np.vstack([samples, antithetic])
            self._rng.shuffle(combined)
            samples = combined[:n]

        _dbg("generate_uniform", "done", shape=samples.shape)
        return samples

    # ------------------------------------------------------------------
    # generate_latin_hypercube  (Sobol-based stratification)
    # ------------------------------------------------------------------
    def generate_latin_hypercube(
        self,
        n: int,
        dims: int,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> np.ndarray:
        """Sobol-seeded Latin Hypercube sample.

        Instead of the classic random-permutation LHS this uses a Sobol
        sequence to seed the strata centres, then adds a small uniform jitter
        (20 % algorithm change â improved space-filling).
        """
        self._tick()
        _dbg("generate_latin_hypercube", "enter", n=n, dims=dims)

        sobol = _sobol_sequence(n, dims, seed=self._seed)

        # Rank-based stratification to guarantee LHS property
        ranks = np.zeros_like(sobol)
        for d in range(dims):
            order = np.argsort(sobol[:, d])
            ranks[order, d] = np.arange(n)

        # Strata centres + jitter (jitter magnitude = 0.5 / n)
        jitter = self._rng.rand(n, dims) * (0.5 / n)
        unit_samples = (ranks + 0.25 + jitter) / n
        unit_samples = np.clip(unit_samples, 0.0, 1.0)

        if bounds is not None:
            lo = np.array([b[0] for b in bounds], dtype=np.float64)
            hi = np.array([b[1] for b in bounds], dtype=np.float64)
            samples = lo + unit_samples * (hi - lo)
        else:
            samples = unit_samples

        _dbg("generate_latin_hypercube", "done", shape=samples.shape)
        return samples

    # ------------------------------------------------------------------
    # generate_stratified
    # ------------------------------------------------------------------
    def generate_stratified(
        self,
        n: int,
        strata_counts: List[int],
    ) -> np.ndarray:
        """Stratified 1-D sampling.

        Divides [0, 1) into *len(strata_counts)* strata and draws the
        requested number of samples from each stratum.  Total samples
        returned equals *sum(strata_counts)* (parameter *n* is advisory
        and used only for debug logging).

        20 % tweak: within each stratum samples are drawn from a truncated
        triangular distribution centred at the stratum midpoint instead of
        plain uniform.
        """
        self._tick()
        _dbg("generate_stratified", "enter", n=n, strata=len(strata_counts))

        k = len(strata_counts)
        edges = np.linspace(0.0, 1.0, k + 1)
        parts: List[np.ndarray] = []

        for i, cnt in enumerate(strata_counts):
            lo, hi = edges[i], edges[i + 1]
            mid = (lo + hi) / 2.0
            width = hi - lo

            # Triangular(lo, mid, hi) then clip
            tri = self._rng.triangular(lo, mid, hi, size=cnt)
            tri = np.clip(tri, lo + 1e-12, hi - 1e-12)

            _dbg("generate_stratified", f"stratum {i}", lo=lo, hi=hi, cnt=cnt)
            parts.append(tri)

        result = np.concatenate(parts).reshape(-1, 1)
        _dbg("generate_stratified", "done", total=result.shape[0])
        return result

    # ------------------------------------------------------------------
    # cartesian_product
    # ------------------------------------------------------------------
    @staticmethod
    def cartesian_product(lists: Sequence[Sequence[float]]) -> np.ndarray:
        """Full Cartesian (grid) product of the supplied per-dimension lists.

        Returns an (M, D) array where M = prod(len(l) for l in lists).
        """
        _dbg("cartesian_product", "enter", dims=len(lists))

        arrays = [np.asarray(lst, dtype=np.float64) for lst in lists]
        mesh = np.meshgrid(*arrays, indexing="ij")
        flat = [m.ravel() for m in mesh]
        result = np.column_stack(flat)

        _dbg("cartesian_product", "done", shape=result.shape)
        return result

    # ------------------------------------------------------------------
    # clip_to_bounds
    # ------------------------------------------------------------------
    @staticmethod
    def clip_to_bounds(
        vals: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
    ) -> np.ndarray:
        """Element-wise clip *vals* into [lo, hi].

        Broadcasts lo/hi against vals.  Returns a copy; original is not
        mutated.

        20 % tweak: values that exceed bounds by more than 10 % of the
        range are reflected back into the interval rather than hard-clamped,
        which preserves distribution shape better during optimisation.
        """
        _dbg("clip_to_bounds", "enter", shape=vals.shape)

        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        result = vals.copy().astype(np.float64)
        span = hi - lo
        span = np.where(span == 0, 1.0, span)

        overshoot_lo = lo - result
        overshoot_hi = result - hi

        # Reflection for moderate overshoots
        reflect_lo = (overshoot_lo > 0) & (overshoot_lo < 0.1 * span)
        reflect_hi = (overshoot_hi > 0) & (overshoot_hi < 0.1 * span)

        result = np.where(reflect_lo, lo + overshoot_lo, result)
        result = np.where(reflect_hi, hi - overshoot_hi, result)

        # Hard clamp everything else
        result = np.clip(result, lo, hi)

        _dbg("clip_to_bounds", "done")
        return result


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def make_generator(seed: int = 42) -> ParameterGenerator:
    _dbg("make_generator", "factory", seed=seed)
    return ParameterGenerator(rng_seed=seed)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    enable_debug(True)

    gen = make_generator(0)
    u = gen.generate_uniform(20, [(0, 1), (10, 20), (-5, 5)])
    print("uniform:", u.shape)

    lhs = gen.generate_latin_hypercube(16, 3, bounds=[(0, 1), (0, 1), (0, 1)])
    print("lhs:", lhs.shape)

    strat = gen.generate_stratified(30, [5, 10, 15])
    print("stratified:", strat.shape)

    cart = ParameterGenerator.cartesian_product([[1, 2], [3, 4, 5]])
    print("cartesian:", cart.shape)

    raw = np.array([[1.05, -0.02], [0.5, 0.5]])
    clipped = ParameterGenerator.clip_to_bounds(
        raw, np.array([0.0, 0.0]), np.array([1.0, 1.0])
    )
    print("clipped:", clipped)
