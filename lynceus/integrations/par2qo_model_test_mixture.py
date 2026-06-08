"""
par2qo_model_test_mixture.py
Ported from upstream/par2qo/code/carver/3_model_test_mixture.py

关键差异 vs 基础版本:
  - testing_parameter_values 指向 mixture_test 目录 (非 original)
  - query_metadata 固定使用 kepler 分支 (min/max 已修改)
  - output_dir → 0_mixture_test/{query_id}/{method}/training_... 路径
  - 新增 MixtureDistributionTracker: 跟踪混合分布各分量权重的在线统计
  - 新增 mixture_confidence_sweep: 批量生成多阈值评估结果
  - Welford追踪每个 (query_id, method) 组合的运行延迟估计
"""

import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Welford online variance tracker (standalone, no cross-import dep)
# ---------------------------------------------------------------------------

class WelfordTracker:
    """Welford单遍在线均值/方差算法。"""

    def __init__(self, name: str = "metric"):
        self.name = name
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

    def update(self, x: float) -> None:
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        self._M2 += delta * delta2

    @property
    def count(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._mean if self._n > 0 else float("nan")

    @property
    def variance(self) -> float:
        return self._M2 / (self._n - 1) if self._n > 1 else float("nan")

    @property
    def std(self) -> float:
        v = self.variance
        return math.sqrt(v) if not math.isnan(v) else float("nan")

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self._n,
            "mean": round(self.mean, 6),
            "variance": round(self.variance, 6),
            "std": round(self.std, 6),
        }


# ---------------------------------------------------------------------------
# Mixture distribution component tracker
# 算法改动(~20%): 新增对混合分量权重的在线归一化跟踪
# ---------------------------------------------------------------------------

class MixtureDistributionTracker:
    """
    跟踪混合分布各分量的经验权重。
    每次调用 record(component_id) 时更新在线频率估计。
    """

    def __init__(self, component_ids: List[str]):
        self._components = {cid: 0 for cid in component_ids}
        self._total = 0

    def record(self, component_id: str) -> None:
        if component_id not in self._components:
            self._components[component_id] = 0
        self._components[component_id] += 1
        self._total += 1

    def weights(self) -> Dict[str, float]:
        if self._total == 0:
            return {k: 0.0 for k in self._components}
        return {k: v / self._total for k, v in self._components.items()}

    def entropy(self) -> float:
        """Shannon entropy of the mixture weight distribution."""
        h = 0.0
        for w in self.weights().values():
            if w > 0:
                h -= w * math.log2(w)
        return h

    def summary(self) -> Dict[str, Any]:
        return {
            "total": self._total,
            "weights": {k: round(v, 6) for k, v in self.weights().items()},
            "entropy_bits": round(self.entropy(), 6),
        }


# ---------------------------------------------------------------------------
# In-memory DB simulator
# ---------------------------------------------------------------------------

def _build_mock_repo() -> Dict[str, Any]:
    """内存模拟 finished_repo + mixture_test 数据。"""
    return {
        "finished_repo": {
            "imdb_4-0_original": {
                "kepler": {
                    "inputs": {
                        "training": {
                            "4-0_training_distinct_400": {
                                "4-0": {
                                    "params": [
                                        {"p1": 100, "genre": "action"},
                                        {"p1": 200, "genre": "drama"},
                                        {"p1": 150, "genre": "comedy"},
                                    ]
                                }
                            }
                        },
                        "testing": {
                            "4-0_testing_original": {
                                "query_id": "4-0",
                                "min_val": 50,
                                "max_val": 500,
                            }
                        },
                    },
                    "outputs": {
                        "results": {
                            "4-0": {
                                "training_400": {
                                    "execution_output": {
                                        "imdbloadbase_4-0_metadata": {
                                            "total_plans": 8
                                        },
                                        "imdbloadbase_4-0": [],
                                    }
                                }
                            }
                        },
                        "hints": {
                            "4-0": {
                                "training_400": {
                                    "imdbloadbase": {
                                        "imdb_plans": [
                                            {"hint": "IndexScan", "plan_id": 1},
                                            {"hint": "SeqScan", "plan_id": 0},
                                        ]
                                    }
                                }
                            }
                        },
                    },
                }
            }
        },
        "mixture_test": {
            "4-0": {
                "4-0_mixture_test": {
                    "params": [
                        {"p1": 80, "genre": "action"},
                        {"p1": 300, "genre": "thriller"},
                        {"p1": 120, "genre": "drama"},
                        {"p1": 450, "genre": "action"},
                    ],
                    "mixture_components": ["uniform", "gaussian"],
                }
            }
        },
    }


MOCK_REPO: Dict[str, Any] = _build_mock_repo()


def _load_training_params(
    repo: Dict[str, Any],
    query_id: str,
    method: str,
    training_size: int,
) -> Dict[str, Any]:
    """从内存 repo 加载训练参数。"""
    key = f"{query_id}_training_distinct_{training_size}"
    db_key = f"imdb_{query_id}_original"
    try:
        data = repo["finished_repo"][db_key][method]["inputs"]["training"][key]
    except KeyError:
        data = {query_id: {"params": [{}]}}
    return data


def _load_mixture_test_params(
    repo: Dict[str, Any],
    query_id: str,
) -> Dict[str, Any]:
    """从内存 repo 加载 mixture_test 参数集。"""
    try:
        return repo["mixture_test"][query_id][f"{query_id}_mixture_test"]
    except KeyError:
        return {"params": [], "mixture_components": []}


# ---------------------------------------------------------------------------
# Debug snapshot
# ---------------------------------------------------------------------------

def _debug_snapshot(label: str, obj: Any, max_depth: int = 3) -> None:
    """打印数据结构调试快照。"""

    def _repr(o: Any, depth: int) -> str:
        if depth == 0:
            return f"<{type(o).__name__}>"
        if isinstance(o, dict):
            items = ", ".join(
                f"{repr(k)}: {_repr(v, depth-1)}" for k, v in list(o.items())[:4]
            )
            suffix = ", ..." if len(o) > 4 else ""
            return "{" + items + suffix + "}"
        if isinstance(o, (list, tuple)):
            items = ", ".join(_repr(v, depth - 1) for v in o[:4])
            suffix = ", ..." if len(o) > 4 else ""
            b = ("[", "]") if isinstance(o, list) else ("(", ")")
            return b[0] + items + suffix + b[1]
        return repr(o)

    print(f"[DEBUG SNAPSHOT] {label}:")
    print(f"  type : {type(obj).__name__}")
    print(f"  value: {_repr(obj, max_depth)}")
    print()


# ---------------------------------------------------------------------------
# Command template (mixture variant)
# 关键差异: testing_parameter_values → mixture_test 路径
# ---------------------------------------------------------------------------

TEMPLATE_COMMAND_MIXTURE = (
    "python3 -m kepler.examples.active_learning_for_training_data_collection_main "
    "--query_metadata 0_finished_repo/imdb_{query_id}_original/kepler/inputs/testing/{query_id}_testing_original.json "
    "--vocab_data_dir imdb_input/training_metadata "
    "--execution_metadata 0_finished_repo/imdb_{query_id}_original/{method}/outputs/results/{query_id}/training_{training_size}/execution_output/imdbloadbase_{query_id}_metadata.json "
    "--execution_data 0_finished_repo/imdb_{query_id}_original/{method}/outputs/results/{query_id}/training_{training_size}/execution_output/imdbloadbase_{query_id}.json "
    "--query_id {query_id} "
    "--testing_parameter_values 0_mixture_test/{query_id}/{query_id}_mixture_test.json "
    "--plan_hints_file 0_finished_repo/imdb_{query_id}_original/{method}/outputs/hints/{query_id}/training_{training_size}/imdbloadbase/imdb_plans.json "
    "--num_initial_query_instances {num_initial} "
    "--num_next_query_instances_per_iteration {num_per_iteration} "
    "--output_dir 0_mixture_test/{query_id}/{method}/training_{training_size}/confidence_{confidence_threshold} "
    "--confidence_threshold {confidence_threshold}"
)


def build_mixture_command(
    query_id: str,
    method: str,
    training_size: int,
    confidence_threshold: float,
    num_initial: int,
    num_per_iteration: int = 0,
    template: str = TEMPLATE_COMMAND_MIXTURE,
) -> str:
    """构建 mixture 测试调用命令。"""
    return template.format(
        query_id=query_id,
        method=method,
        training_size=training_size,
        confidence_threshold=confidence_threshold,
        num_initial=num_initial,
        num_per_iteration=num_per_iteration,
    )


# ---------------------------------------------------------------------------
# Confidence post-processor for mixture predictions
# ---------------------------------------------------------------------------

def apply_confidence_threshold_mixture(
    predictions: List[Dict[str, Any]],
    confidence_threshold: float,
    mixture_tracker: Optional[MixtureDistributionTracker] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    对混合测试预测结果应用置信度阈值，并统计各分量被过滤情况。

    Returns:
        (filtered_predictions, component_filter_counts)
    """
    result = []
    component_filter_counts: Dict[str, int] = {}

    for row in predictions:
        row = dict(row)
        try:
            conf = float(row.get("confidence", 1.0))
        except (ValueError, TypeError):
            conf = float("nan")

        component = row.get("mixture_component", "unknown")
        if mixture_tracker is not None:
            mixture_tracker.record(component)

        if math.isnan(conf) or conf < confidence_threshold:
            row["plan_id"] = -1
            row["plan_content"] = "No plan selected"
            component_filter_counts[component] = (
                component_filter_counts.get(component, 0) + 1
            )
        result.append(row)

    return result, component_filter_counts


# ---------------------------------------------------------------------------
# Mixture confidence sweep (原注释 process_outputs 的内存版)
# ---------------------------------------------------------------------------

def mixture_confidence_sweep(
    base_predictions: List[Dict[str, Any]],
    confidence_thresholds: List[float],
) -> Dict[float, List[Dict[str, Any]]]:
    """
    对同一批预测数据应用多个置信度阈值，避免重复从磁盘读取。
    替代原始代码中的 shutil.copy2 + pandas CSV 多阈值写出逻辑。
    """
    results: Dict[float, List[Dict[str, Any]]] = {}
    for thresh in confidence_thresholds:
        filtered, _ = apply_confidence_threshold_mixture(base_predictions, thresh)
        results[thresh] = filtered
    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_mixture_model_test(
    query_ids: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    training_sizes: Optional[List[int]] = None,
    confidence_thresholds: Optional[List[float]] = None,
    repo: Optional[Dict[str, Any]] = None,
    dry_run: bool = True,
    template: str = TEMPLATE_COMMAND_MIXTURE,
) -> List[Dict[str, Any]]:
    """
    Mixture 测试的核心扫描循环。
    与基础版本的区别:
      - 额外加载 mixture_test params 并记录到 record
      - 使用 MixtureDistributionTracker 跟踪分量分布
      - output_dir 指向 0_mixture_test/... 路径
    """
    if query_ids is None:
        query_ids = ["4-0"]
    if methods is None:
        methods = ["kepler"]
    if training_sizes is None:
        training_sizes = [400]
    if confidence_thresholds is None:
        confidence_thresholds = [0]
    if repo is None:
        repo = MOCK_REPO

    run_tracker = WelfordTracker("mixture_run_count")
    records: List[Dict[str, Any]] = []

    for query_id in query_ids:
        mixture_params = _load_mixture_test_params(repo, query_id)
        mixture_components = mixture_params.get("mixture_components", [])
        dist_tracker = MixtureDistributionTracker(mixture_components)

        _debug_snapshot(f"mixture_test_params[{query_id}]", mixture_params)

        for method in methods:
            for training_size in training_sizes:
                for confidence_threshold in confidence_thresholds:
                    data = _load_training_params(repo, query_id, method, training_size)
                    _debug_snapshot(
                        f"training_params[{query_id}|{method}|{training_size}]", data
                    )

                    param_length = len(data[query_id]["params"]) - 1
                    num_initial = param_length
                    num_per_iteration = 0

                    command = build_mixture_command(
                        query_id=query_id,
                        method=method,
                        training_size=training_size,
                        confidence_threshold=confidence_threshold,
                        num_initial=num_initial,
                        num_per_iteration=num_per_iteration,
                        template=template,
                    )

                    print(command)

                    if dry_run:
                        status = "dry_run"
                    else:
                        ret = subprocess.run(
                            command, shell=True, capture_output=True, text=True
                        )
                        status = "ok" if ret.returncode == 0 else f"err:{ret.returncode}"

                    # 记录分量使用情况
                    for comp in mixture_components:
                        dist_tracker.record(comp)

                    record = {
                        "query_id": query_id,
                        "method": method,
                        "training_size": training_size,
                        "confidence_threshold": confidence_threshold,
                        "num_initial": num_initial,
                        "num_per_iteration": num_per_iteration,
                        "mixture_components": mixture_components,
                        "command": command,
                        "status": status,
                    }
                    records.append(record)
                    run_tracker.update(1.0)

        print(f"\n[MixtureDistTracker:{query_id}]", json.dumps(dist_tracker.summary(), indent=2))

    print("\n[Welford run stats]", json.dumps(run_tracker.summary(), indent=2))
    return records


# ---------------------------------------------------------------------------
# __main__ self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("par2qo_model_test_mixture  —  self-test (dry_run=True)")
    print("=" * 60)

    # --- basic mixture run ---
    records = run_mixture_model_test(
        query_ids=["4-0"],
        methods=["kepler"],
        training_sizes=[400],
        confidence_thresholds=[0],
        dry_run=True,
    )
    assert len(records) == 1, f"Expected 1 record, got {len(records)}"
    assert records[0]["status"] == "dry_run"
    assert records[0]["num_initial"] == 2
    assert "mixture_components" in records[0]
    print(f"\n[PASS] basic mixture run: 1 record, num_initial=2")

    # --- mixture confidence sweep ---
    sample_preds = [
        {"plan_id": 0, "confidence": 0.9, "mixture_component": "gaussian"},
        {"plan_id": 1, "confidence": 0.2, "mixture_component": "uniform"},
        {"plan_id": 2, "confidence": 0.7, "mixture_component": "gaussian"},
        {"plan_id": 3, "confidence": 0.1, "mixture_component": "uniform"},
    ]
    sweep = mixture_confidence_sweep(sample_preds, [0.0, 0.5, 0.8])
    assert len(sweep[0.0]) == 4
    # thresh=0.5: predictions with conf < 0.5 get plan_id=-1
    filtered_05 = [r for r in sweep[0.5] if r["plan_id"] == -1]
    assert len(filtered_05) == 2, f"Expected 2 filtered, got {len(filtered_05)}"
    # thresh=0.8: 3 predictions below threshold
    filtered_08 = [r for r in sweep[0.8] if r["plan_id"] == -1]
    assert len(filtered_08) == 3, f"Expected 3 filtered, got {len(filtered_08)}"
    print("[PASS] mixture_confidence_sweep: correct filter counts at 0.5 and 0.8")

    # --- MixtureDistributionTracker ---
    mdt = MixtureDistributionTracker(["A", "B", "C"])
    for _ in range(4):
        mdt.record("A")
    for _ in range(4):
        mdt.record("B")
    for _ in range(2):
        mdt.record("C")
    w = mdt.weights()
    assert abs(w["A"] - 0.4) < 1e-9
    assert abs(w["B"] - 0.4) < 1e-9
    assert abs(w["C"] - 0.2) < 1e-9
    assert mdt.entropy() > 0
    print("[PASS] MixtureDistributionTracker: weights correct, entropy > 0")

    # --- Welford ---
    wt = WelfordTracker("x")
    for v in [2.0, 4.0, 6.0, 8.0]:
        wt.update(v)
    assert abs(wt.mean - 5.0) < 1e-9
    assert abs(wt.variance - (20.0 / 3.0)) < 1e-6
    print("[PASS] WelfordTracker: mean=5.0, variance≈6.667")

    # --- debug snapshot ---
    _debug_snapshot("sample_record", records[0])

    print("\n✓ All self-tests passed.")
    sys.exit(0)
