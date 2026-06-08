"""
par2qo_model_test.py
Ported from upstream/par2qo/code/carver/3_model_test.py

鲁迅式移植:
  - 保留核心逻辑骨架 (command构建 + 参数扫描循环)
  - DB文件I/O → dict内存模拟
  - os.system() → 捕获式执行 + dry-run模式
  - 加 _debug_snapshot() 打印数据结构状态
  - 加 Welford在线方差跟踪执行次数
  - 加 __main__ 自测
"""

import os
import json
import subprocess
import math
import sys
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Welford online variance tracker
# ---------------------------------------------------------------------------

class WelfordTracker:
    """Welford单遍在线均值/方差算法，避免数值不稳定的两遍法。"""

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
# In-memory DB simulator (replaces file I/O)
# ---------------------------------------------------------------------------

def _build_mock_repo() -> Dict[str, Any]:
    """构建内存模拟的 finished_repo 数据结构，替代真实磁盘目录。"""
    return {
        "imdb_4-0_original": {
            "kepler": {
                "inputs": {
                    "training": {
                        "4-0_training_distinct_400": {
                            "4-0": {
                                "params": [
                                    {"param_1": 100, "param_2": "action"},
                                    {"param_1": 200, "param_2": "drama"},
                                    {"param_1": 150, "param_2": "comedy"},
                                ]
                            }
                        }
                    },
                    "testing": {
                        "4-0_testing_original": {
                            "query_id": "4-0",
                            "predicates": ["param_1 > $1", "genre = $2"],
                        }
                    },
                    "frequency": {
                        "4-0_train_400_freq": {"4-0": {"freq": [0.3, 0.5, 0.2]}}
                    },
                },
                "outputs": {
                    "results": {
                        "4-0": {
                            "training_400": {
                                "execution_output": {
                                    "imdbloadbase_4-0_metadata": {
                                        "total_plans": 12,
                                        "default_plan": 0,
                                    },
                                    "imdbloadbase_4-0": [
                                        {"plan_id": 0, "cost": 1500},
                                        {"plan_id": 1, "cost": 1200},
                                    ],
                                }
                            }
                        }
                    },
                    "hints": {
                        "4-0": {
                            "training_400": {
                                "imdbloadbase": {
                                    "imdb_plans": [
                                        {"hint": "IndexScan(title)", "plan_id": 1},
                                        {"hint": "SeqScan(title)", "plan_id": 0},
                                    ]
                                }
                            }
                        }
                    },
                    "evaluation": {},
                },
            }
        }
    }


MOCK_REPO: Dict[str, Any] = _build_mock_repo()


def _load_training_params(
    repo: Dict[str, Any],
    query_id: str,
    method: str,
    training_size: int,
) -> Dict[str, Any]:
    """从内存 repo 中加载训练参数，模拟 open(training_param_file) + json.load()。"""
    key = f"{query_id}_training_distinct_{training_size}"
    db_key = f"imdb_{query_id}_original"
    try:
        data = repo[db_key][method]["inputs"]["training"][key]
    except KeyError:
        # 降级：返回含最小结构的占位数据
        data = {query_id: {"params": [{}]}}
    return data


# ---------------------------------------------------------------------------
# Debug snapshot
# ---------------------------------------------------------------------------

def _debug_snapshot(label: str, obj: Any, max_depth: int = 3) -> None:
    """打印数据结构的调试快照，类似 pprint 但附带类型信息。"""

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
            bracket = ("[", "]") if isinstance(o, list) else ("(", ")")
            return bracket[0] + items + suffix + bracket[1]
        return repr(o)

    print(f"[DEBUG SNAPSHOT] {label}:")
    print(f"  type : {type(obj).__name__}")
    print(f"  value: {_repr(obj, max_depth)}")
    print()


# ---------------------------------------------------------------------------
# Command builder (核心逻辑保留，路径改为内存可验证的字符串)
# ---------------------------------------------------------------------------

TEMPLATE_COMMAND = (
    "python3 -m kepler.examples.active_learning_for_training_data_collection_main "
    "--query_metadata 0_finished_repo/imdb_{query_id}_original/{method}/inputs/testing/{query_id}_testing_original.json "
    "--vocab_data_dir imdb_input/training_metadata "
    "--execution_metadata 0_finished_repo/imdb_{query_id}_original/{method}/outputs/results/{query_id}/training_{training_size}/execution_output/imdbloadbase_{query_id}_metadata.json "
    "--execution_data 0_finished_repo/imdb_{query_id}_original/{method}/outputs/results/{query_id}/training_{training_size}/execution_output/imdbloadbase_{query_id}.json "
    "--query_id {query_id} "
    "--testing_parameter_values 0_finished_repo/imdb_{query_id}_original/{method}/inputs/testing/{query_id}_testing_original.json "
    "--plan_hints_file 0_finished_repo/imdb_{query_id}_original/{method}/outputs/hints/{query_id}/training_{training_size}/imdbloadbase/imdb_plans.json "
    "--num_initial_query_instances {num_initial} "
    "--num_next_query_instances_per_iteration {num_per_iteration} "
    "--output_dir 0_finished_repo/imdb_{query_id}_original/{method}/outputs/evaluation/{query_id}/training_{training_size}/confidence_{confidence_threshold} "
    "--frequency_file 0_finished_repo/imdb_{query_id}_original/{method}/inputs/frequency/{query_id}_train_{training_size}_freq.json "
    "--confidence_threshold {confidence_threshold}"
)


def build_command(
    query_id: str,
    method: str,
    training_size: int,
    confidence_threshold: float,
    num_initial: int,
    num_per_iteration: int = 0,
    template: str = TEMPLATE_COMMAND,
) -> str:
    """构建 kepler 调用命令字符串。"""
    return template.format(
        query_id=query_id,
        method=method,
        training_size=training_size,
        confidence_threshold=confidence_threshold,
        num_initial=num_initial,
        num_per_iteration=num_per_iteration,
    )


# ---------------------------------------------------------------------------
# Confidence post-processor (原注释代码中的 process_outputs 移植)
# ---------------------------------------------------------------------------

def apply_confidence_threshold(
    predictions: List[Dict[str, Any]],
    confidence_threshold: float,
) -> List[Dict[str, Any]]:
    """
    对预测结果列表应用置信度阈值过滤。
    替代原 pandas CSV 处理逻辑，改为纯 dict 操作。
    """
    result = []
    for row in predictions:
        row = dict(row)
        try:
            conf = float(row.get("confidence", 1.0))
        except (ValueError, TypeError):
            conf = float("nan")
        if math.isnan(conf) or conf < confidence_threshold:
            row["plan_id"] = -1
            row["plan_content"] = "No plan selected"
        result.append(row)
    return result


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_model_test(
    query_ids: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    training_sizes: Optional[List[int]] = None,
    confidence_thresholds: Optional[List[float]] = None,
    repo: Optional[Dict[str, Any]] = None,
    dry_run: bool = True,
    template: str = TEMPLATE_COMMAND,
) -> List[Dict[str, Any]]:
    """
    核心扫描循环，移植自 upstream 3_model_test.py。

    改动:
      - 文件 I/O → 内存 repo dict
      - os.system() → dry_run 模式 (真实执行时改为 subprocess)
      - 收集并返回每次运行的元数据列表

    Returns:
        List of run-record dicts with keys: query_id, method, training_size,
        confidence_threshold, num_initial, command, status
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

    tracker = WelfordTracker("run_count")
    records: List[Dict[str, Any]] = []

    for query_id in query_ids:
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

                    command = build_command(
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
                        "command": command,
                        "status": status,
                    }
                    records.append(record)
                    tracker.update(1.0)

    print("\n[Welford stats]", json.dumps(tracker.summary(), indent=2))
    return records


# ---------------------------------------------------------------------------
# __main__ self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("par2qo_model_test  —  self-test (dry_run=True)")
    print("=" * 60)

    # --- basic run ---
    records = run_model_test(
        query_ids=["4-0"],
        methods=["kepler"],
        training_sizes=[400],
        confidence_thresholds=[0],
        dry_run=True,
    )
    assert len(records) == 1, "Expected 1 record"
    assert records[0]["status"] == "dry_run"
    assert records[0]["num_initial"] == 2  # 3 params - 1

    print("\n[PASS] basic run: 1 record, status=dry_run, num_initial=2")

    # --- multi-sweep ---
    records2 = run_model_test(
        query_ids=["4-0"],
        methods=["kepler"],
        training_sizes=[400],
        confidence_thresholds=[0, 0.5],
        dry_run=True,
    )
    assert len(records2) == 2, f"Expected 2 records, got {len(records2)}"
    print("[PASS] multi-sweep: 2 records for 2 confidence thresholds")

    # --- confidence filter ---
    sample_preds = [
        {"plan_id": 1, "confidence": 0.9, "plan_content": "HashJoin"},
        {"plan_id": 2, "confidence": 0.3, "plan_content": "NestedLoop"},
        {"plan_id": 3, "confidence": "bad", "plan_content": "MergeJoin"},
    ]
    filtered = apply_confidence_threshold(sample_preds, confidence_threshold=0.5)
    assert filtered[0]["plan_id"] == 1
    assert filtered[1]["plan_id"] == -1
    assert filtered[2]["plan_id"] == -1
    print("[PASS] confidence filter: high-conf kept, low-conf/bad → plan_id=-1")

    # --- Welford ---
    wt = WelfordTracker("latency_ms")
    for v in [10.0, 20.0, 30.0]:
        wt.update(v)
    assert wt.count == 3
    assert abs(wt.mean - 20.0) < 1e-9
    assert abs(wt.variance - 100.0) < 1e-9
    print("[PASS] WelfordTracker: count=3, mean=20.0, variance=100.0")

    # --- debug snapshot ---
    _debug_snapshot("sample_record", records[0])

    print("\n✓ All self-tests passed.")
    sys.exit(0)
