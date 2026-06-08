"""
lynceus/data_writer.py — Benchmark data output writer.

算法改动:
    1. cal_rel_error → symmetric MAPE + log_ratio 双返回
    2. measure_with_median → IQR outlier rejection
    3. write_json → p50/p90/p99 latency percentile
    4. clean_benchmark_data → regex pattern matching
"""
from __future__ import annotations
import math
import time
import json
import re
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

logger = logging.getLogger(__name__)

# ─── String Parsing (from par2qo parse.py) ──────────────────────────────────

def split_string(s: str, delimiter: str) -> List[str]:
    parts = s.split(delimiter)
    result: List[str] = []
    for k in range(len(parts)):
        result.append(parts[k])
        if k < len(parts) - 1:
            result.append(delimiter)
    return result

def parse_string(tokens: List[str], delimiter: str) -> List[str]:
    result: List[str] = []
    for token in tokens:
        if delimiter in token:
            result.extend(split_string(token, delimiter))
        else:
            result.append(token)
    return result

# ─── Record Type Mapping ────────────────────────────────────────────────────

record_type_names = ['seq_scan', 'index_scan', 'hash_probe', 'hash_build',
                     'transfer', 'pipeline', 'allreduce']
record_format_dict = {
    'seq_scan': 'SequentialScan', 'index_scan': 'IndexScan',
    'hash_probe': 'HashProbe', 'hash_build': 'HashBuild',
    'transfer': 'DataTransfer', 'pipeline': 'PipelineStage',
    'allreduce': 'AllReduceSync',
}

# ─── Data Sanitisation ──────────────────────────────────────────────────────

# 改动: 编译 regex 模式做噪声匹配 (原版只用关键字字符串 in)
_NOISE_PATTERNS = [
    re.compile(r'^\s*DEBUG\b', re.IGNORECASE),
    re.compile(r'^\s*WARNING\b', re.IGNORECASE),
    re.compile(r'^[=\-]{4,}'),           # 分隔线
    re.compile(r'^\s*INTERNAL\b'),
    re.compile(r'^\s*#.*$'),             # 注释行
    re.compile(r'^\s*//.*$'),            # C-style注释
    re.compile(r'^\[.*\]\s*$'),          # bare bracket lines like "[end]"
]

def clean_benchmark_data(raw_lines: List[str],
                         del_keys: Optional[List[str]] = None,
                         extra_patterns: Optional[List[re.Pattern]] = None
                         ) -> List[str]:
    """改动: 用预编译 regex 做噪声过滤, 比原版逐字符串 in 检查更精确。

    原版: for key in del_keys: if key in line → delete
    问题: "WARNING" 会误删包含 "WARNING" 子串的合法数据行
    新版: 用 ^\\s*WARNING\\b anchor 匹配行首, 不误伤内容
    """
    patterns = list(_NOISE_PATTERNS)
    if extra_patterns:
        patterns.extend(extra_patterns)
    if del_keys:
        for key in del_keys:
            patterns.append(re.compile(re.escape(key)))

    cleaned = []
    for line in raw_lines:
        should_delete = any(p.search(line) for p in patterns)
        if not should_delete:
            line = line.replace('+', '').strip()
            if line:
                cleaned.append(line)
    return cleaned

# ─── Cost List Extraction ────────────────────────────────────────────────────

def get_cost_list_from_json(json_str: str,
                            debug_print: bool = True) -> Tuple[List[float], List[float]]:
    predicted = []
    actual = []
    try:
        records = json.loads(json_str)
        if not isinstance(records, list):
            records = [records]
        for rec in records:
            if 'predicted_cost' in rec:
                predicted.append(float(rec['predicted_cost']))
            if 'actual_cost' in rec:
                actual.append(float(rec['actual_cost']))
    except json.JSONDecodeError as e:
        if debug_print:
            print(f"  [data_writer] WARNING: JSON parse error: {e}")
    if debug_print:
        print(f"  [data_writer] Extracted {len(predicted)} predicted, "
              f"{len(actual)} actual costs")
    return predicted, actual

# ─── Relative Error ──────────────────────────────────────────────────────────

def cal_rel_error(true_val: float, est_val: float) -> float:
    """改动: 返回 symmetric MAPE 代替非对称 log-ratio。

    原版: if true > est: log(true/est) else: -log(est/true)
    问题: 非对称, 且 true=0 或 est=0 时未定义
    新版: sMAPE = 2*|true-est| / (|true|+|est|+eps)
    范围 [0, 2], 对称, 0处有 epsilon 保护
    """
    eps = 1e-9
    if math.isnan(true_val) or math.isnan(est_val):
        return 0.0
    numerator = 2.0 * abs(true_val - est_val)
    denominator = abs(true_val) + abs(est_val) + eps
    return numerator / denominator


def cal_log_ratio(true_val: float, est_val: float) -> float:
    """保留原版 log-ratio 做辅助指标, 有符号。"""
    if true_val <= 0 or est_val <= 0:
        if true_val <= 0 and est_val <= 0:
            return 0.0
        return float('inf') if true_val <= 0 else float('-inf')
    if true_val > est_val:
        return math.log(true_val / est_val)
    else:
        return -math.log(est_val / true_val)


# ─── Output Format Types ────────────────────────────────────────────────────

class OutputFormat(Enum):
    JSON = auto()
    CSV = auto()
    LOG = auto()


# ─── Benchmark Record ───────────────────────────────────────────────────────

@dataclass
class BenchmarkRecord:
    record_id: int
    operation: str
    predicted_cost_us: float = 0.0
    actual_cost_us: float = 0.0
    routing_decision: str = "CPU"
    device: str = ""
    n_rows: int = 0
    data_bytes: int = 0
    rel_error: float = 0.0
    log_ratio: float = 0.0         # 改动: 保留 log_ratio
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.actual_cost_us > 0 and self.predicted_cost_us > 0:
            self.rel_error = cal_rel_error(self.actual_cost_us, self.predicted_cost_us)
            self.log_ratio = cal_log_ratio(self.actual_cost_us, self.predicted_cost_us)

    def dump_debug(self, prefix: str = "") -> str:
        lines = [
            f"{prefix}╔══ Record #{self.record_id} ═══════",
            f"{prefix}║ op={self.operation} pred={self.predicted_cost_us:.1f}µs "
            f"act={self.actual_cost_us:.1f}µs",
            f"{prefix}║ sMAPE={self.rel_error:.4f} log_ratio={self.log_ratio:.4f} "
            f"→ {self.routing_decision}",
            f"{prefix}╚═══════════════════════",
        ]
        return "\n".join(lines)


# ─── Experiment Metadata ─────────────────────────────────────────────────────

@dataclass
class ExperimentMeta:
    experiment_id: str = "exp_001"
    workload_name: str = "default"
    template_id: int = 0
    n_samples: int = 100
    tolerance: float = 0.2
    blend_factor: float = 0.5
    gpu_arch: str = "A100"
    n_workers: int = 1
    sharding_strategy: str = "FULL_SHARD"
    costing_version: str = "v1.0"
    def dump_debug(self, prefix: str = "") -> str:
        return (f"{prefix}Experiment({self.experiment_id}): "
                f"{self.workload_name} N={self.n_samples} "
                f"tol={self.tolerance} arch={self.gpu_arch}")


# ─── Timing with IQR Outlier Rejection ───────────────────────────────────────

def _percentile(sorted_vals: List[float], p: float) -> float:
    """无 numpy 的 percentile: 线性插值。"""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    k = (n - 1) * p
    f = math.floor(k)
    c = min(f + 1, n - 1)
    if f == c:
        return sorted_vals[int(f)]
    return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)


def measure_with_median(func, *args, times: int = 5,
                        debug_print: bool = True, **kwargs) -> float:
    """改动: IQR outlier rejection 后取中位数。

    原版: sort → 取 middle 元素
    新版: 先算 Q1/Q3/IQR, 过滤掉 > Q3+1.5*IQR 或 < Q1-1.5*IQR 的试验,
    再对剩余值取中位数。
    效果: OS 调度/GC 抖动产生的极端值不影响结果。
    """
    latencies = []
    for trial in range(times):
        start = time.time()
        func(*args, **kwargs)
        elapsed_us = (time.time() - start) * 1e6
        latencies.append(elapsed_us)
        if debug_print and trial == 0:
            print(f"  [data_writer] Trial 0: {elapsed_us:.1f}µs")

    latencies.sort()
    n = len(latencies)

    if n >= 5:
        # IQR outlier rejection
        q1 = _percentile(latencies, 0.25)
        q3 = _percentile(latencies, 0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        filtered = [v for v in latencies if lo <= v <= hi]
        if not filtered:
            filtered = latencies  # 不能全删
    else:
        filtered = latencies

    filtered.sort()
    m = len(filtered)
    if m % 2 == 1:
        median = filtered[m // 2]
    else:
        median = (filtered[m // 2 - 1] + filtered[m // 2]) / 2

    if debug_print:
        rejected = n - len(filtered)
        print(f"  [data_writer] Median of {n} trials: {median:.1f}µs "
              f"(rejected {rejected} outliers, "
              f"min={filtered[0]:.1f}, max={filtered[-1]:.1f})")
    return median


# ─── Batch Cost Collection ───────────────────────────────────────────────────

def collect_all_costs(records: List[BenchmarkRecord],
                      debug_print: bool = True) -> List[float]:
    costs = []
    for rec in records:
        costs.append(rec.actual_cost_us)
        if debug_print and len(costs) % 50 == 0:
            print(f"  [data_writer] Collected {len(costs)}/{len(records)} costs")
    return costs


# ─── Config Record Generation ───────────────────────────────────────────────

def gen_config_record(operation: str, device: str,
                      params: Optional[Dict[str, Any]] = None) -> str:
    fmt_name = record_format_dict.get(operation, operation)
    parts = [f"/* CostModelConfig: {fmt_name} */"]
    parts.append(f"  device = {device}")
    if params:
        for k, v in sorted(params.items()):
            parts.append(f"  {k} = {v}")
    parts.append("/* end */")
    return "\n".join(parts)


# ─── Data Writer ─────────────────────────────────────────────────────────────

class DataWriter:
    def __init__(self, output_dir: str = "./results",
                 experiment: Optional[ExperimentMeta] = None,
                 debug_print: bool = True):
        self._output_dir = output_dir
        self._experiment = experiment or ExperimentMeta()
        self._records: List[BenchmarkRecord] = []
        self._record_counter = 0
        self._debug = debug_print
        self._log_file: Optional[str] = None
        self._write_count = 0
        if debug_print:
            print(f"\n[data_writer] Initialized DataWriter")
            print(f"  output_dir = {output_dir}")
            print(f"  {self._experiment.dump_debug()}")

    def init_log(self, debug_print: Optional[bool] = None) -> str:
        dp = debug_print if debug_print is not None else self._debug
        exp = self._experiment
        log_dir = os.path.join(self._output_dir, "log", exp.workload_name)
        log_filename = (f"{exp.experiment_id}_t{exp.template_id}"
                       f"_{exp.workload_name}_b{exp.blend_factor}"
                       f"_N{exp.n_samples}.log")
        self._log_file = os.path.join(log_dir, log_filename)
        if dp:
            print(f"  [data_writer] Log file: {self._log_file}")
        return self._log_file

    def add_record(self, operation: str, predicted_us: float, actual_us: float,
                   routing: str = "CPU", device: str = "", n_rows: int = 0,
                   data_bytes: int = 0,
                   metadata: Optional[Dict[str, Any]] = None) -> BenchmarkRecord:
        self._record_counter += 1
        record = BenchmarkRecord(
            record_id=self._record_counter, operation=operation,
            predicted_cost_us=predicted_us, actual_cost_us=actual_us,
            routing_decision=routing, device=device,
            n_rows=n_rows, data_bytes=data_bytes,
            metadata=metadata or {})
        self._records.append(record)
        if self._debug and self._record_counter % 10 == 0:
            print(f"  [data_writer] Record #{self._record_counter}: {operation} "
                  f"sMAPE={record.rel_error:.3f} log_r={record.log_ratio:.3f}")
        return record

    def write_json(self, filepath: Optional[str] = None,
                   debug_print: Optional[bool] = None) -> str:
        """改动: summary 加 p50/p90/p99 latency percentile。

        原版: 只有 mean_rel_error 和 GPU/CPU 计数
        新版: 加 latency_percentiles, 更接近 SLA 评估需求
        """
        dp = debug_print if debug_print is not None else self._debug
        if filepath is None:
            exp = self._experiment
            filepath = os.path.join(
                self._output_dir,
                f"{exp.experiment_id}_{exp.workload_name}_N{exp.n_samples}.json")

        # 改动: 计算 percentile
        actual_costs = sorted(r.actual_cost_us for r in self._records)
        pctiles = {}
        if actual_costs:
            for p_name, p_val in [("p50", 0.5), ("p90", 0.9), ("p99", 0.99)]:
                pctiles[p_name] = _percentile(actual_costs, p_val)

        errors = [abs(r.rel_error) for r in self._records]
        output = {
            "experiment": {
                "id": self._experiment.experiment_id,
                "workload": self._experiment.workload_name,
                "template_id": self._experiment.template_id,
                "n_samples": self._experiment.n_samples,
                "tolerance": self._experiment.tolerance,
                "blend_factor": self._experiment.blend_factor,
                "gpu_arch": self._experiment.gpu_arch,
                "n_workers": self._experiment.n_workers,
                "sharding": self._experiment.sharding_strategy,
                "costing_version": self._experiment.costing_version,
            },
            "summary": {
                "n_records": len(self._records),
                "mean_smape": sum(errors) / max(1, len(errors)),
                "gpu_routed": sum(1 for r in self._records if r.routing_decision == "GPU"),
                "cpu_routed": sum(1 for r in self._records if r.routing_decision == "CPU"),
                "latency_percentiles_us": pctiles,
            },
            "records": [
                {
                    "id": r.record_id, "operation": r.operation,
                    "predicted_cost": r.predicted_cost_us,
                    "actual_cost": r.actual_cost_us,
                    "smape": r.rel_error, "log_ratio": r.log_ratio,
                    "routing": r.routing_decision, "device": r.device,
                    "n_rows": r.n_rows, "data_bytes": r.data_bytes,
                    "timestamp": r.timestamp, "metadata": r.metadata,
                }
                for r in self._records
            ],
        }
        self._write_count += 1
        if dp:
            print(f"\n  [data_writer] write_json #{self._write_count}: {filepath}")
            print(f"    records: {len(self._records)}, pctiles: {pctiles}")
        return json.dumps(output, indent=2)

    def write_csv(self, filepath: Optional[str] = None,
                  debug_print: Optional[bool] = None) -> str:
        dp = debug_print if debug_print is not None else self._debug
        if filepath is None:
            exp = self._experiment
            filepath = os.path.join(
                self._output_dir,
                f"{exp.experiment_id}_{exp.workload_name}_N{exp.n_samples}.csv")
        header = "id,operation,predicted_us,actual_us,smape,log_ratio,routing,device,n_rows,data_bytes"
        lines = [header]
        for r in self._records:
            lines.append(
                f"{r.record_id},{r.operation},{r.predicted_cost_us:.2f},"
                f"{r.actual_cost_us:.2f},{r.rel_error:.4f},{r.log_ratio:.4f},"
                f"{r.routing_decision},{r.device},{r.n_rows},{r.data_bytes}")
        self._write_count += 1
        if dp:
            print(f"\n  [data_writer] write_csv #{self._write_count}: {len(self._records)} rows")
        return "\n".join(lines)

    def write_log(self, debug_print: Optional[bool] = None) -> str:
        dp = debug_print if debug_print is not None else self._debug
        lines = []
        total_predicted = 0.0
        total_actual = 0.0
        better_count = 0
        for r in self._records:
            total_predicted += r.predicted_cost_us
            total_actual += r.actual_cost_us
            if abs(r.rel_error) < self._experiment.tolerance:
                better_count += 1
            line = (f"Record {r.record_id}-{self._experiment.workload_name}-"
                    f"{self._experiment.blend_factor}: "
                    f"{r.operation} on {r.device}: "
                    f"pred={r.predicted_cost_us:.1f}µs, "
                    f"act={r.actual_cost_us:.1f}µs, "
                    f"sMAPE={r.rel_error:.4f}, log_r={r.log_ratio:.4f}, "
                    f"routing={r.routing_decision}, "
                    f"{better_count}/{r.record_id} within tolerance")
            lines.append(line)
        n = len(self._records)
        if n > 0:
            summary = (f"SUMMARY: Predicted avg: {total_predicted/n:.1f}µs, "
                      f"Actual avg: {total_actual/n:.1f}µs, "
                      f"Ratio: {total_predicted/max(0.001, total_actual):.3f}, "
                      f"{better_count}/{n} within tol={self._experiment.tolerance}")
            lines.append(summary)
        self._write_count += 1
        if dp and lines:
            print(f"\n  [data_writer] write_log #{self._write_count}")
            print(f"    {lines[-1]}")
        return "\n".join(lines)

    def dump_state(self) -> str:
        n = len(self._records)
        errors = [abs(r.rel_error) for r in self._records]
        mean_err = sum(errors) / max(1, n)
        max_err = max(errors) if errors else 0.0
        gpu_count = sum(1 for r in self._records if r.routing_decision == "GPU")
        lines = [
            f"DataWriter: {n} records, {self._write_count} writes",
            f"  mean_sMAPE={mean_err:.4f}, max_sMAPE={max_err:.4f}",
            f"  GPU/CPU = {gpu_count}/{n - gpu_count}",
        ]
        if self._records:
            lines.append("  last 3:")
            for r in self._records[-3:]:
                lines.append(f"    #{r.record_id}: {r.operation} "
                           f"sMAPE={r.rel_error:.3f} → {r.routing_decision}")
        return "\n".join(lines)
