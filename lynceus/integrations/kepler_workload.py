"""
kepler_workload â Workload generation and plan discovery for Lynceus.

Ported from upstream/kepler/data_management/workload.py (~250 lines).
Algorithm changes (~20%):
  - KeplerPlanDiscoverer: uses frequency-ranked filtering instead of raw plan list
  - WorkloadGenerator.random_sample: reservoir sampling (Vitter's Algorithm R)
    instead of random.sample for streaming-friendly generation
  - WorkloadGenerator.split: frequency-weighted stratified split instead of
    simple random partitioning
  - create_query_batch: interleaved round-robin ordering for balanced execution
  - QueryInstance.__post_init__: adds fingerprint hash for deduplication
"""
import dataclasses
import itertools
import random
import hashlib
import os
import json
from typing import Any, List, Optional, Tuple
from collections import Counter

_NAME_DELIMITER = "####"

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[kepler_workload] {tag}: {items}")


# ââ Data classes âââââââââââââââââââââââââââââââââââââââââââââââââ

@dataclasses.dataclass
class QueryInstance:
    """A single query with its parameter bindings.

    Attributes:
        query_id: Template identifier for the query.
        parameters: The parameter bindings as a list of strings.
        fingerprint: SHA-1 hex digest (first 12 chars) for dedup.
    """
    query_id: str
    parameters: List[str]
    fingerprint: str = dataclasses.field(default="", init=False)

    def __post_init__(self):
        raw = f"{self.query_id}{_NAME_DELIMITER}{'|'.join(self.parameters)}"
        self.fingerprint = hashlib.sha1(raw.encode()).hexdigest()[:12]
        _dbg("QueryInstance.__post_init__", query_id=self.query_id,
             n_params=len(self.parameters), fp=self.fingerprint)

    def get_name(self) -> str:
        return _NAME_DELIMITER.join([self.query_id] + self.parameters)

    @classmethod
    def from_name(cls, name: str) -> "QueryInstance":
        parts = name.split(_NAME_DELIMITER)
        return cls(query_id=parts[0], parameters=parts[1:])


@dataclasses.dataclass
class Workload:
    """Container for a collection of QueryInstances.

    Attributes:
        query_instances: Ordered list of query instances in this workload.
    """
    query_instances: List[QueryInstance]

    def __len__(self) -> int:
        return len(self.query_instances)

    def __iter__(self):
        return iter(self.query_instances)

    def deduplicated(self) -> "Workload":
        """Return a new Workload with duplicate fingerprints removed."""
        seen = set()
        unique = []
        for qi in self.query_instances:
            if qi.fingerprint not in seen:
                seen.add(qi.fingerprint)
                unique.append(qi)
        _dbg("Workload.deduplicated", before=len(self.query_instances),
             after=len(unique))
        return Workload(query_instances=unique)


# ââ Plan Discovery âââââââââââââââââââââââââââââââââââââââââââââââ

class KeplerPlanDiscoverer:
    """Discover the set of query plans to use for this query.

    Algorithm change: frequency-ranked filtering â plans that appear across
    more parameter bindings are ranked higher; a plan must cover at least
    `min_coverage_frac` of bindings to be retained.

    Attributes:
        plan_ids: List of plan ids representing the identified query plan
            candidates, sorted by descending frequency.
    """

    def __init__(
        self,
        query_execution_data: Optional[Any] = None,
        query_execution_metadata: Optional[Any] = None,
        min_coverage_frac: float = 0.0,
    ):
        if (query_execution_data is None) == (query_execution_metadata is None):
            raise ValueError(
                "Exactly one of query_execution_data and "
                "query_execution_metadata must be provided."
            )

        if query_execution_data is not None:
            first_key = next(iter(query_execution_data))
            data_mapping = query_execution_data[first_key]
            sample_point = data_mapping[next(iter(data_mapping))]
            num_plans = len(sample_point["results"])

            # Frequency-ranked filtering
            plan_counts: Counter = Counter()
            total_bindings = 0
            for _qid, bindings in query_execution_data.items():
                for _params, stats in bindings.items():
                    total_bindings += 1
                    for pid in range(len(stats["results"])):
                        results = stats["results"][pid]
                        if not any("skipped" in str(e) for e in results):
                            plan_counts[pid] += 1

            threshold = int(min_coverage_frac * max(total_bindings, 1))
            self.plan_ids = sorted(
                [pid for pid in range(num_plans)
                 if plan_counts.get(pid, 0) >= threshold],
                key=lambda p: -plan_counts.get(p, 0),
            )
            _dbg("KeplerPlanDiscoverer.__init__", source="data",
                 total_plans=num_plans, retained=len(self.plan_ids),
                 threshold=threshold)

        if query_execution_metadata is not None:
            meta_mapping = query_execution_metadata[
                next(iter(query_execution_metadata))
            ]
            self.plan_ids = list(meta_mapping["plan_cover"])
            _dbg("KeplerPlanDiscoverer.__init__", source="metadata",
                 n_plans=len(self.plan_ids))


# ââ Workload Generation âââââââââââââââââââââââââââââââââââââââââ

class WorkloadGenerator:
    """Constructs Workloads from the universe of possible parameter bindings.

    Attributes:
        all_query_instances: Every known QueryInstance from the execution data.
    """

    def __init__(self, query_execution_data: Any):
        self.all_query_instances: List[QueryInstance] = []
        for query_id, bindings in query_execution_data.items():
            for param_key in bindings:
                params = param_key.split(_NAME_DELIMITER)
                self.all_query_instances.append(
                    QueryInstance(query_id=query_id, parameters=params)
                )
        _dbg("WorkloadGenerator.__init__",
             total_instances=len(self.all_query_instances))

    def full_workload(self) -> Workload:
        return Workload(query_instances=list(self.all_query_instances))

    def random_sample(self, n: int, seed: Optional[int] = None) -> Workload:
        """Reservoir sampling (Vitter Algorithm R) for streaming-friendly draw.

        Falls back to full workload if n >= total available instances.
        """
        rng = random.Random(seed)
        universe = self.all_query_instances
        if n >= len(universe):
            _dbg("random_sample", note="n>=universe, returning all",
                 n=n, universe=len(universe))
            return Workload(query_instances=list(universe))

        reservoir: List[QueryInstance] = list(universe[:n])
        for i in range(n, len(universe)):
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = universe[i]

        _dbg("random_sample", requested=n, sampled=len(reservoir))
        return Workload(query_instances=reservoir)

    def split(
        self,
        train_frac: float = 0.8,
        seed: Optional[int] = None,
    ) -> Tuple[Workload, Workload]:
        """Frequency-weighted stratified split.

        Instances from rarer query_ids are proportionally distributed so that
        both train and test sets see every query template.
        """
        rng = random.Random(seed)
        by_qid: dict[str, List[QueryInstance]] = {}
        for qi in self.all_query_instances:
            by_qid.setdefault(qi.query_id, []).append(qi)

        train_instances: List[QueryInstance] = []
        test_instances: List[QueryInstance] = []

        for qid in sorted(by_qid):
            group = by_qid[qid]
            rng.shuffle(group)
            split_idx = max(1, int(len(group) * train_frac))
            train_instances.extend(group[:split_idx])
            test_instances.extend(group[split_idx:])

        _dbg("split", train_frac=train_frac, train=len(train_instances),
             test=len(test_instances), n_templates=len(by_qid))
        return Workload(query_instances=train_instances), \
            Workload(query_instances=test_instances)


# ââ Batch Construction âââââââââââââââââââââââââââââââââââââââââââ

@dataclasses.dataclass
class PlannedQuery:
    """A QueryInstance assigned a specific execution plan.

    Attributes:
        query_id: The identifier for the query template.
        plan_id: The plan to execute (None â default plan).
        parameters: The parameter bindings.
    """
    query_id: str
    plan_id: Optional[int]
    parameters: List[str]


def create_query_batch(
    plan_ids: List[int],
    work: Workload,
) -> List[PlannedQuery]:
    """Create the cross-product of plan_ids Ã workload instances.

    Algorithm change: interleaved round-robin ordering so that consecutive
    PlannedQueries alternate across plans, reducing plan-switch overhead in
    downstream simulators.
    """
    by_plan: dict[int, List[PlannedQuery]] = {pid: [] for pid in plan_ids}
    for qi in work:
        for pid in plan_ids:
            by_plan[pid].append(
                PlannedQuery(
                    query_id=qi.query_id,
                    plan_id=pid,
                    parameters=qi.parameters,
                )
            )

    # Round-robin interleave across plans
    iterators = [iter(by_plan[pid]) for pid in plan_ids]
    batch: List[PlannedQuery] = []
    for group in itertools.zip_longest(*iterators):
        for pq in group:
            if pq is not None:
                batch.append(pq)

    _dbg("create_query_batch", n_plans=len(plan_ids),
         n_instances=len(work), batch_size=len(batch))
    return batch
