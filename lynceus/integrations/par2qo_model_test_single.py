"""
par2qo_model_test_single.py
Ported from upstream/par2qo/code/carver/3_model_test_single.py

关键差异 vs 基础版本:
  - 针对 DSB (Decision Support Benchmark) 工作负载，非 IMDB
  - 使用 active_learning_for_training_data_collection_main_single 入口
  - query_metadata 指向 original_template_gaussian 目录
  - testing_parameter_values 使用 _testing_original_single.json
  - output_dir 使用 evaluation_single 子目录
  - 默认 method = 'gaussian'; query_ids 为多查询 DSB 集合
  - 新增 SingleQueryProfiler: 跟踪每个查询的独立性能特征
  - 新增 gaussian_param_estimator: 对训练参数做高斯参数估计 (μ, σ)
  - Welford追踪各查询的 num_initial 分布
"""

import json
import math
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Welford online variance tracker
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
# SingleQueryProfiler
# 算法改动(~20%): 新增每个 query_id 的独立性能特征追踪
# ---------------------------------------------------------------------------

class SingleQueryProfiler:
    """
    为每个查询独立维护一个 WelfordTracker，跟踪 num_initial 的分布。
    支持多方法、多 training_size 的横向比较。
    """

    def __init__(self):
        self._trackers: Dict[str, WelfordTracker] = {}

    def record(self, query_id: str, num_initial: int) -> None:
        if query_id not in self._trackers:
            self._trackers[query_id] = WelfordTracker(f"num_initial[{query_id}]")
        self._trackers[query_id].update(float(num_initial))

    def summary_all(self) -> Dict[str, Dict[str, Any]]:
        return {qid: t.summary() for qid, t in self._trackers.items()}

    def top_k_by_mean(self, k: int = 3) -> List[Tuple[str, float]]:
        """返回 num_initial 均值最大的 k 个查询。"""
        items = [(qid, t.mean) for qid, t in self._trackers.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:k]


# ---------------------------------------------------------------------------
# Gaussian parameter estimator
# 算法改动: 对数值型训练参数做高斯 (μ, σ) 估计，辅助后续特征工程
# ---------------------------------------------------------------------------

def gaussian_param_estimator(
    params: List[Dict[str, Any]],
    numeric_keys: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    对训练参数列表中每个数值型字段估计高斯参数 (μ, σ)。
    使用 Welford 算法避免两遍扫描。

    Args:
        params: 训练参数列表，每个元素为 dict
        numeric_keys: 要估计的字段名；None 时自动检测数值字段

    Returns:
        {field: {"mean": μ, "std": σ, "count": n}}
    """
    if not params:
        return {}

    # 自动检测数值字段
    if numeric_keys is None:
        numeric_keys = []
        for k, v in params[0].items():
            try:
                float(v)
                numeric_keys.append(k)
            except (TypeError, ValueError):
                pass

    trackers: Dict[str, WelfordTracker] = {k: WelfordTracker(k) for k in numeric_keys}

    for p in params:
        for k in numeric_keys:
            try:
                val = float(p.get(k, float("nan")))
                if not math.isnan(val):
                    trackers[k].update(val)
            except (TypeError, ValueError):
                pass

    return {
        k: {
            "mean": round(t.mean, 6),
            "std": round(t.std, 6),
            "count": t.count,
        }
        for k, t in trackers.items()
    }


# ---------------------------------------------------------------------------
# In-memory DB simulator (DSB workload)
# ---------------------------------------------------------------------------

_DSB_QUERY_IDS = [
    "013", "018", "019", "025", "027", "040",
    "050", "072", "084", "085", "091", "099", "100",
]

def _build_mock_dsb_repo() -> Dict[str, Any]:
    """构建 DSB 工作负载的内存模拟数据。"""
    repo: Dict[str, Any] = {}
    for qid in _DSB_QUERY_IDS:
        repo[qid] = {
            "gaussian": {
                "inputs": {
                    "training": {
                        f"{qid}_training_distinct_50": {
                            qid: {
                                "params": [
                                    {"val_1": float(i * 10 + int(qid[:3])), "val_2": float(i)}
                                    for i in range(6)
                                ]
                            }
                        }
                    },
                    "testing": {
                        f"{qid}_testing_original_single": {
                            "query_id": qid,
                            "distribution": "gaussian",
                            "single_param": True,
                        }
                    },
                    "frequency": {
                        f"{qid}_train_50_freq": {
                            qid: {"freq": [1.0 / 5] * 5}
                        }
                    },
                },
                "outputs": {
                    "results": {
                        qid: {
                            "training_50": {
                                "execution_output": {
                                    f"dsb_{qid}_metadata": {"total_plans": 6},
                                    f"dsb_{qid}": [],
                                }
                            }
                        }
                    },
                    "hints": {
                        qid: {
                            "training_50": {
                                "dsb": {
                                    "dsb_plans": [
                                        {"hint": f"plan_{k}", "plan_id": k}
                                        for k in range(3)
                                    ]
                                }
                            }
                        }
                    },
                    "evaluation_single": {},
                },
            }
        }
    return repo


MOCK_DSB_REPO: Dict[str, Any] = _build_mock_dsb_repo()


def _load_dsb_training_params(
    repo: Dict[str, Any],
    query_id: str,
    method: str,
    training_size: int,
) -> Dict[str, Any]:
    """从内存 DSB repo 加载训练参数。"""
    key = f"{query_id}_training_distinct_{training_size}"
    try:
        data = repo[query_id][method]["inputs"]["training"][key]
    except KeyError:
        data = {query_id: {"params": [{}]}}
    return data


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
# Command template (single / DSB variant)
# 关键差异:
#   - main_single 入口 (非 main)
#   - original_template_gaussian 路径
#   - _testing_original_single.json
#   - evaluation_single 输出目录
# ---------------------------------------------------------------------------

TEMPLATE_COMMAND_SINGLE = (
    "python3 -m kepler.examples.active_learning_for_training_data_collection_main_single "
    "--query_metadata dsb_cardrange_workload/original_template_gaussian/{query_id}.json "
    "--vocab_data_dir dsb_training_metadata "
    "--execution_metadata dsb_cardrange_workload/dsb_{query_id}_original/{method}/outputs/results/{query_id}/training_{training_size}/execution_output/dsb_{query_id}_metadata.json "
    "--execution_data dsb_cardrange_workload/dsb_{query_id}_original/{method}/outputs/results/{query_id}/training_{training_size}/execution_output/dsb_{query_id}.json "
    "--query_id {query_id} "
    "--testing_parameter_values dsb_cardrange_workload/dsb_{query_id}_original/{method}/inputs/testing/{query_id}_testing_original_single.json "
    "--plan_hints_file dsb_cardrange_workload/dsb_{query_id}_original/{method}/outputs/hints/{query_id}/training_{training_size}/dsb/dsb_plans.json "
    "--num_initial_query_instances {num_initial} "
    "--num_next_query_instances_per_iteration {num_per_iteration} "
    "--output_dir dsb_cardrange_workload/dsb_{query_id}_original/{method}/outputs/evaluation_single/{query_id}/training_{training_size}/confidence_{confidence_threshold} "
    "--frequency_file dsb_cardrange_workload/dsb_{query_id}_original/{method}/inputs/frequency/{query_id}_train_{training_size}_freq.json "
    "--confidence_threshold {confidence_threshold}"
)


def build_single_command(
    query_id: str,
    method: str,
    training_size: int,
    confidence_threshold: float,
    num_initial: int,
    num_per_iteration: int = 0,
    template: str = TEMPLATE_COMMAND_SINGLE,
) -> str:
    """构建 single-parameter DSB 测试调用命令。"""
    return template.format(
        query_id=query_id,
        method=method,
        training_size=training_size,
        confidence_threshold=confidence_threshold,
        num_initial=num_initial,
        num_per_iteration=num_per_iteration,
    )


# ---------------------------------------------------------------------------
# Confidence post-processor for single-param predictions
# ---------------------------------------------------------------------------

def apply_confidence_threshold_single(
    predictions: List[Dict[str, Any]],
    confidence_threshold: float,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    对单参数预测结果应用置信度阈值。

    Returns:
        (filtered_predictions, n_rejected)
    """
    result = []
    n_rejected = 0
    for row in predictions:
        row = dict(row)
        try:
            conf = float(row.get("confidence", 1.0))
        except (ValueError, TypeError):
            conf = float("nan")
        if math.isnan(conf) or conf < confidence_threshold:
            row["plan_id"] = -1
            row["plan_content"] = "No plan selected"
            n_rejected += 1
        result.append(row)
    return result, n_rejected


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_single_model_test(
    query_ids: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    training_sizes: Optional[List[int]] = None,
    confidence_thresholds: Optional[List[float]] = None,
    repo: Optional[Dict[str, Any]] = None,
    dry_run: bool = True,
    template: str = TEMPLATE_COMMAND_SINGLE,
) -> List[Dict[str, Any]]:
    """
    Single-parameter DSB 测试的核心扫描循环。
    与基础版本的区别:
      - 面向 DSB query_ids (多查询集合)
      - 使用 main_single 入口
      - SingleQueryProfiler 跟踪各查询特征
      - gaussian_param_estimator 计算参数分布
      - evaluation_single 输出路径
    """
    if query_ids is None:
        query_ids = _DSB_QUERY_IDS
    if methods is None:
        methods = ["gaussian"]
    if training_sizes is None:
        training_sizes = [50]
    if confidence_thresholds is None:
        confidence_thresholds = [0]
    if repo is None:
        repo = MOCK_DSB_REPO

    run_tracker = WelfordTracker("single_run_count")
    profiler = SingleQueryProfiler()
    records: List[Dict[str, Any]] = []

    for query_id in query_ids:
        for method in methods:
            for training_size in training_sizes:
                for confidence_threshold in confidence_thresholds:
                    data = _load_dsb_training_params(repo, query_id, method, training_size)
                    _debug_snapshot(
                        f"dsb_params[{query_id}|{method}|{training_size}]", data
                    )

                    raw_params = data[query_id]["params"]
                    param_length = len(raw_params) - 1
                    num_initial = param_length
                    num_per_iteration = 0

                    # 高斯参数估计
                    gauss_stats = gaussian_param_estimator(raw_params)

                    profiler.record(query_id, num_initial)

                    command = build_single_command(
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

                    record = {
                        "query_id": query_id,
                        "method": method,
                        "training_size": training_size,
                        "confidence_threshold": confidence_threshold,
                        "num_initial": num_initial,
                        "num_per_iteration": num_per_iteration,
                        "gauss_stats": gauss_stats,
                        "command": command,
                        "status": status,
                    }
                    records.append(record)
                    run_tracker.update(1.0)

    print("\n[Welford run stats]", json.dumps(run_tracker.summary(), indent=2))
    print("\n[SingleQueryProfiler top-3]", json.dumps(profiler.top_k_by_mean(3), indent=2))
    return records


# ---------------------------------------------------------------------------
# __main__ self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("par2qo_model_test_single  —  self-test (dry_run=True)")
    print("=" * 60)

    # --- small subset test ---
    test_qids = ["013", "018", "019"]
    records = run_single_model_test(
        query_ids=test_qids,
        methods=["gaussian"],
        training_sizes=[50],
        confidence_thresholds=[0],
        dry_run=True,
    )
    assert len(records) == 3, f"Expected 3 records, got {len(records)}"
    for r in records:
        assert r["status"] == "dry_run"
        assert r["num_initial"] == 5  # 6 params - 1
        assert "gauss_stats" in r
        assert "val_1" in r["gauss_stats"]
    print(f"\n[PASS] small subset: 3 records, num_initial=5, gauss_stats populated")

    # --- gaussian_param_estimator ---
    sample_params = [
        {"val_1": 10.0, "val_2": 1.0},
        {"val_1": 20.0, "val_2": 3.0},
        {"val_1": 30.0, "val_2": 5.0},
    ]
    stats = gaussian_param_estimator(sample_params)
    assert "val_1" in stats and "val_2" in stats
    assert abs(stats["val_1"]["mean"] - 20.0) < 1e-9
    assert abs(stats["val_2"]["mean"] - 3.0) < 1e-9
    print("[PASS] gaussian_param_estimator: val_1 mean=20.0, val_2 mean=3.0")

    # --- SingleQueryProfiler ---
    sp = SingleQueryProfiler()
    sp.record("013", 5)
    sp.record("013", 7)
    sp.record("018", 3)
    summary = sp.summary_all()
    assert abs(summary["013"]["mean"] - 6.0) < 1e-9
    top = sp.top_k_by_mean(1)
    assert top[0][0] == "013"
    print("[PASS] SingleQueryProfiler: mean=6.0 for 013, top_k correct")

    # --- confidence filter single ---
    preds = [
        {"plan_id": 0, "confidence": 0.9},
        {"plan_id": 1, "confidence": 0.1},
        {"plan_id": 2, "confidence": "NaN"},
    ]
    filtered, n_rej = apply_confidence_threshold_single(preds, 0.5)
    assert n_rej == 2
    assert filtered[0]["plan_id"] == 0
    assert filtered[1]["plan_id"] == -1
    assert filtered[2]["plan_id"] == -1
    print("[PASS] apply_confidence_threshold_single: n_rejected=2")

    # --- Welford ---
    wt = WelfordTracker("t")
    for v in [5.0, 5.0, 5.0, 5.0]:
        wt.update(v)
    assert abs(wt.mean - 5.0) < 1e-9
    assert abs(wt.variance) < 1e-9
    print("[PASS] WelfordTracker: constant series, variance≈0")

    # --- debug snapshot ---
    _debug_snapshot("sample_record", records[0])

    print("\n✓ All self-tests passed.")
    sys.exit(0)
