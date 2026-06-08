# Author: dylanyunlon <dogechat@163.com>
# M197 – ported from upstream param_gen_test_output.py (2046 lines)
# Sobol replaces random sampling; Welford online statistics; _dbg() breakpoints
"""
Kepler parameter generation pipeline for query optimization.

Generates, validates and splits parameter sets for SQL template queries
using quasi-random Sobol sequences instead of pseudo-random sampling,
and Welford's online algorithm for running statistics on cardinality
distributions.
"""

import collections
import csv
import json
import logging
import os
import re
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Sobol quasi-random engine (replaces random throughout)
# ---------------------------------------------------------------------------

class _SobolEngine:
    """Sobol low-discrepancy sequence generator using direction numbers."""

    _DIRECTION_NUMBERS = np.array([
        1 << 31, 1 << 30, 1 << 29, 1 << 28, 1 << 27,
        1 << 26, 1 << 25, 1 << 24, 1 << 23, 1 << 22,
        1 << 21, 1 << 20, 1 << 19, 1 << 18, 1 << 17,
        1 << 16, 1 << 15, 1 << 14, 1 << 13, 1 << 12,
        1 << 11, 1 << 10, 1 << 9,  1 << 8,  1 << 7,
        1 << 6,  1 << 5,  1 << 4,  1 << 3,  1 << 2,
        1 << 1,  1 << 0,
    ], dtype=np.uint64)

    def __init__(self, dimension: int = 1, seed: int = 0):
        self.dimension = max(1, dimension)
        self._index = seed & 0xFFFF
        self._state = np.zeros(self.dimension, dtype=np.uint64)

    def _rightmost_zero(self, n: int) -> int:
        """Position of the rightmost zero bit in *n*."""
        pos = 0
        while n & 1:
            n >>= 1
            pos += 1
        return min(pos, len(self._DIRECTION_NUMBERS) - 1)

    def draw(self, n: int = 1) -> np.ndarray:
        """Return *n* quasi-random samples in [0, 1)^dim."""
        out = np.empty((n, self.dimension), dtype=np.float64)
        for i in range(n):
            c = self._rightmost_zero(self._index)
            v = self._DIRECTION_NUMBERS[c]
            for d in range(self.dimension):
                self._state[d] ^= v
            out[i] = self._state / (2.0 ** 32)
            self._index += 1
        return out

    def draw_flat(self, n: int = 1) -> np.ndarray:
        """Single-dimension draw returning a 1-D array."""
        return self.draw(n)[:, 0] if self.dimension == 1 else self.draw(n)


def _sobol_choice(population, k: int, seed: int = 0):
    """Select *k* items from *population* using a Sobol sequence (with replacement)."""
    pop = list(population)
    if not pop:
        return []
    eng = _SobolEngine(dimension=1, seed=seed)
    indices = (eng.draw_flat(k) * len(pop)).astype(int)
    indices = np.clip(indices, 0, len(pop) - 1)
    return [pop[idx] for idx in indices]


def _sobol_weighted_choice(population, weights, k: int, seed: int = 0):
    """Weighted sampling via Sobol-driven inverse-CDF."""
    pop = list(population)
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() == 0:
        w = np.ones_like(w)
    cdf = np.cumsum(w / w.sum())
    eng = _SobolEngine(dimension=1, seed=seed)
    u = eng.draw_flat(k)
    indices = np.searchsorted(cdf, u, side="right")
    indices = np.clip(indices, 0, len(pop) - 1)
    return [pop[idx] for idx in indices]


def _sobol_shuffle(lst, seed: int = 0):
    """Shuffle *lst* in-place using Sobol-keyed Schwartzian transform."""
    if len(lst) <= 1:
        return lst
    eng = _SobolEngine(dimension=1, seed=seed)
    keys = eng.draw_flat(len(lst))
    order = np.argsort(keys)
    tmp = [lst[i] for i in order]
    lst[:] = tmp
    return lst


# ---------------------------------------------------------------------------
# Welford online statistics (replaces ad-hoc mean / var bookkeeping)
# ---------------------------------------------------------------------------

class WelfordAccumulator:
    """Welford's online algorithm for running mean and variance."""

    __slots__ = ("_n", "_mean", "_m2")

    def __init__(self):
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._m2 += delta * delta2

    def update_batch(self, xs) -> None:
        for x in xs:
            self.update(float(x))

    @property
    def count(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        return self._m2 / self._n if self._n > 1 else 0.0

    @property
    def sample_variance(self) -> float:
        return self._m2 / (self._n - 1) if self._n > 1 else 0.0

    @property
    def std(self) -> float:
        return float(np.sqrt(self.variance))

    def __repr__(self) -> str:
        return (f"WelfordAccumulator(n={self._n}, mean={self._mean:.4f}, "
                f"std={self.std:.4f})")


# ---------------------------------------------------------------------------
# Debug infrastructure
# ---------------------------------------------------------------------------

_DBG_ENABLED = os.environ.get("LYNCEUS_DBG", "0") == "1"


def _dbg(tag: str = "", **kwargs):
    """Conditional breakpoint / trace hook.

    When LYNCEUS_DBG=1 is set in the environment the function prints the
    tag and keyword arguments then drops into pdb.  Otherwise it is a no-op.
    """
    if not _DBG_ENABLED:
        return
    frame = sys._getframe(1)
    loc = f"{frame.f_code.co_filename}:{frame.f_lineno}"
    parts = [f"[DBG {tag}] {loc}"]
    for k, v in kwargs.items():
        parts.append(f"  {k}={v!r}")
    print("\n".join(parts), file=sys.stderr)
    try:
        import pdb
        pdb.set_trace()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Constants (matching upstream IMDB domain)
# ---------------------------------------------------------------------------

EXTRA_TABLE = ["it_miidx", "it_mi", "it_pi"]

available_it_for_mi = (
    ["genres"] * 13 + ["budget"] * 2 + ["release dates"] * 20 + ["countries"] * 10
)
available_it_for_pi = ["mini biography"] * 3 + ["trivia"] * 2 + ["height"] * 1
available_it_for_miidx = (
    ["top 250 rank"] * 2 + ["bottom 10 rank"] * 2 + ["rating"] * 16 + ["votes"] * 11
)

cct_for_cc_subject_id = (
    ["IN ('cast', 'crew')"] * 4
    + ["= 'cast'"] * 12
    + ["!= 'complete+verified'"] * 2
)
cct_for_cc_status_id = (
    ["= 'complete'"] * 3
    + ["LIKE '%complete%'"] * 7
    + [" = 'complete+verified'"] * 9
)

INT_TABLE_DICT = {
    "t": {
        "production_year": {"min": 1880, "max": 2019},
        "episode_nr": {"min": 1, "max": 91821},
    }
}


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------

def check_required_values(metadata_file):
    """
    Check if the required variables have values. If any of them is missing, raise an error.

    Args:
    metadata_file: Path to the metadata file.

    Raises:
    ValueError: If any of the required variables is missing.
    """
    if not metadata_file:
        raise ValueError("Metadata file (_METADATA_FILE) is missing or not provided.")
    print("All required values are provided.")


def escape_single_quotes(param):
    """
    Escapes single quotes in a string parameter by replacing each single quote with two single quotes.    
    Args:
        param: The parameter to escape. Can be any type, but only strings will be modified.
        
    Returns:
        If param is a string, returns a new string with all single quotes doubled.
        Otherwise, returns the original param unchanged.
        
    Examples:
        >>> escape_single_quotes("O'Reilly")
        "O''Reilly"
        >>> escape_single_quotes(123)
        123
    """
    if isinstance(param, str):
        return param.replace("'", "''")
    return param


def save_metadata_to_csv(output_dir, metadata, query_id):
    """
    Saves the metadata to a CSV file in the output directory with the query_id as filename.
    
    This function creates a two-column CSV file (Key, Value) where each row represents
    a key-value pair from the metadata dictionary.
    
    Args:
        output_dir (str): The directory path where the metadata CSV file will be saved.
        metadata (dict): A dictionary containing the metadata information to save.
        query_id (str): Identifier used to name the output CSV file.
    
    Returns:
        None: The function doesn't return a value but prints confirmation message.
    
    Example:
        >>> metadata = {"timestamp": "2023-04-01", "status": "complete"}
        >>> save_metadata_to_csv("/path/to/output", metadata, "query123")
        Metadata saved to /path/to/output/query123.csv
    """
    metadata_file = os.path.join(output_dir, f"{query_id}.csv")
    _dbg("save_meta", path=metadata_file, n_keys=len(metadata))
    with open(metadata_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Value"])
        for key, value in metadata.items():
            writer.writerow([key, value])
    print(f"Metadata saved to {metadata_file}")


# ---------------------------------------------------------------------------
# Cardinality bucket helpers
# ---------------------------------------------------------------------------

def create_buckets(data, num_buckets):
    """
    Step 2: Divides data points into a specified number of buckets based on the second value of each data point.
    
    This function distributes data items into buckets by value range. If all values are identical,
    it distributes items evenly across buckets. Otherwise, it creates buckets of equal value ranges
    and assigns items to the appropriate bucket based on their value.
    
    Args:
        data (list): List of tuples where each tuple's last element is the value used for bucketing.
        num_buckets (int): Number of buckets to create.
        
    Returns:
        list: A list of bucket lists, where each bucket contains data points assigned to it.
    """
    second_values = np.array([d[-1] for d in data], dtype=np.float64)
    _dbg("create_buckets", n_data=len(data), n_buckets=num_buckets)

    # Welford stats on cardinality values
    wf = WelfordAccumulator()
    wf.update_batch(second_values)

    min_value = float(second_values.min())
    max_value = float(second_values.max())
    print(f"!!! Cardinality range of current column [{min_value}, {max_value}] "
          f"(welford: mean={wf.mean:.2f}, std={wf.std:.2f})")

    interval_width = (max_value - min_value) / num_buckets
    buckets: list = [[] for _ in range(num_buckets)]

    if interval_width == 0:
        # Sobol-driven round-robin when all values identical
        for i, item in enumerate(data):
            buckets[i % num_buckets].append(item)
    else:
        # Use numpy vectorised bucketing for speed
        indices = np.floor((second_values - min_value) / interval_width).astype(int)
        indices = np.clip(indices, 0, num_buckets - 1)
        for idx, item in zip(indices, data):
            buckets[idx].append(item)

    return buckets


def sample_from_buckets(buckets, total_samples, seed: int = 0):
    """
    Step 3: Samples data points from multiple buckets with Sobol replacement to create a representative dataset.
    
    This function distributes the total number of samples evenly across non-empty buckets and 
    selects items from each bucket using Sobol quasi-random sequences. It ensures the final 
    sample size matches the requested total by distributing any remainder samples across 
    buckets and truncating if necessary.
    
    Args:
        buckets (list): List of bucket lists, where each bucket contains data points.
        total_samples (int): Total number of samples to collect across all buckets.
        seed (int): Sobol seed for reproducibility.
        
    Returns:
        list: A combined list of sampled data points from all buckets, with exactly
                total_samples number of items.
    """
    non_empty_buckets = [b for b in buckets if b]
    num_ne = len(non_empty_buckets)
    if num_ne == 0:
        return []
    samples_per = total_samples // num_ne
    remainder = total_samples % num_ne

    _dbg("sample_from_buckets", total=total_samples, non_empty=num_ne)

    sampled_data = []
    sub_seed = seed
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        num_samples = samples_per + (1 if i < remainder else 0)
        sampled_data.extend(_sobol_choice(bucket, num_samples, seed=sub_seed))
        sub_seed += num_samples

    return sampled_data[:total_samples]


# ---------------------------------------------------------------------------
# CDF implementation for range predicates
# ---------------------------------------------------------------------------

def get_range_param_and_cardinality(data, operator):
    """
    Generate param and cumulative cardinality adjustments based on the operator.
    
    Args:
    - data: The input list of tuples (param, cardinality).
    - operator: The operator type ('>', '>=', '<', '<=', '=').
    
    Returns:
    - A list of tuples (param, cardinality), where `param` is the value,
      and `cardinality` is the cumulative count based on the operator.
      
    Example 1:
    - Input: 
        - data = [(1, 10), (2, 20), (3, 30), (4, 40)]
        - operator = "<" -> "< param"
    - Output:
        - data = [(1, 0), (2, 10), (3, 30), (4, 60)]  # Cumulative without including the current
    
    Example 2:
    - Input: 
        - data = [(1, 10), (2, 20), (3, 30), (4, 40)]
        - operator = ">=" -> ">= param"
    - Output:
        - data = [(1, 100), (2, 90), (3, 70), (4, 40)]
    
    """
    data_sorted = sorted(data, key=lambda x: x[0])
    _dbg("range_cdf", operator=operator, n=len(data_sorted))

    # Welford tracker on raw cardinalities
    wf = WelfordAccumulator()
    for _, card in data_sorted:
        wf.update(float(card))

    # Use numpy prefix-sum for CDF computation (replaces manual loop)
    params = [p for p, _ in data_sorted]
    cards = np.array([c for _, c in data_sorted], dtype=np.float64)

    if operator == ">":
        rev_cumsum = np.cumsum(cards[::-1])[::-1]
        # shift: exclude current value
        adjusted = np.zeros_like(rev_cumsum)
        adjusted[:-1] = rev_cumsum[1:]
        return list(zip(params, adjusted.tolist()))

    elif operator == ">=":
        rev_cumsum = np.cumsum(cards[::-1])[::-1]
        return list(zip(params, rev_cumsum.tolist()))

    elif operator == "<":
        cumsum = np.cumsum(cards)
        adjusted = np.zeros_like(cumsum)
        adjusted[1:] = cumsum[:-1]
        return list(zip(params, adjusted.tolist()))

    elif operator == "<=":
        cumsum = np.cumsum(cards)
        return list(zip(params, cumsum.tolist()))

    elif operator == "=":
        return list(zip(params, cards.tolist()))

    return list(zip(params, cards.tolist()))


# ---------------------------------------------------------------------------
# IN-operator sampling
# ---------------------------------------------------------------------------

def sample_in_operator(buckets, total_samples, num_samples_per_bucket: int = 5,
                       shuffle: bool = True, seed_start: int = 2024):
    """
    Creates IN
"""
