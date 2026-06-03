"""
lynceus_port_v3/data_writer.py — Benchmark data output writer (v3-ported).

v3 变更:
    - cal_rel_error: 对称相对误差 → SMAPE (对称百分比绝对误差)
      原版用 log-ratio: log(true/est),  不对称 (同幅偏大偏小误差不等)
      v3: 2*|true-est|/(|true|+|est|),  范围 [0,2], 对称, 无 log(0) 问题
    - measure_with_median: 增加 IQR 异常值剔除 (剔除 > Q3+1.5*IQR)
    - write_json: summary 增加 SMAPE 分位数 (p50/p90/p95)
    - DataWriter.add_record: 在线计算 running mean/variance (Welford)
      而不是在 dump 时遍历全部记录
    - collect_all_costs: 增加可选的 filter_outliers 参数
"""

from __future__ import annotations

import math
import time
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto


# --- port_v3: debug instrumentation ---
try:
    from ._debug import dbg, snapshot, timing, checkpoint
except ImportError:
    from lynceus_port_v3._debug import dbg, snapshot, timing, checkpoint

logger = logging.getLogger(__name__)


# ─── String Parsing Utilities ────────────────────────────────────────────────

def split_string(s: str, delimiter: str) -> List[str]:
    """Split string and interleave delimiter tokens."""
    parts = s.split(delimiter)
    result: List[str] = []
    for k in range(len(parts)):
        result.append(parts[k])
        if k < len(parts) - 1:
            result.append(delimiter)
    return result


def parse_string(tokens: List[str], delimiter: str) -> List[str]:
    """Apply split_string recursively to a token list."""
    result: List[str] = []
    for token in tokens:
        if delimiter in token:
            result.extend(split_string(token, delimiter))
        else:
            result.append(token)
    return result


# ─── Hint/Record Type Mapping ────────────────────────────────────────────────

record_type_names = ['seq_scan', 'index_scan', 'hash_probe', 'hash_build',
                     'transfer', 'pipeline', 'allreduce']
record_format_dict = {
    'seq_scan':     'SequentialScan',
    'index_scan':   'IndexScan',
    'hash_probe':   'HashProbe',
    'hash_build':   'HashBuild',
    'transfer':     'DataTransfer',
    'pipeline':     'PipelineStage',
    'allreduce':    'AllReduceSync',
}


# ─── Data Sanitisation ──────────────────────────────────────────────────────

def clean_benchmark_data(raw_lines: List[str],
                         del_keys: Optional[List[str]] = None) -> List[str]:
    """Clean benchmark output lines by removing noise."""
    if del_keys is None:
        del_keys = ['DEBUG', 'WARNING', '====', '----', 'INTERNAL']

    cleaned = []
    for line in raw_lines:
        should_delete = False
        for key in del_keys:
            if key in line:
                should_delete = True
                break
        if not should_delete:
            line = line.replace('+', '').strip()
            if line:
                cleaned.append(line)
    return cleaned


# ─── Cost List Extraction ────────────────────────────────────────────────────

def get_cost_list_from_json(json_str: str,
                            debug_print: bool = True) -> Tuple[List[float], List[float]]:
    """Extract predicted and actual cost lists from benchmark JSON."""
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


# ─── Relative Error Computation ──────────────────────────────────────────────
# v3 变更: 从 log-ratio 改为 SMAPE (对称百分比绝对误差)
# 原版: log(true/est) — 不对称, log(0)问题
# v3:   2*|true-est|/(|true|+|est|) — 对称, 范围[0,2], 无除零

def cal_rel_error(true_val: float, est_val: float) -> float:
    """SMAPE-based symmetric relative error.

    v3 变更: 替代 log-ratio, 使得正负偏差的惩罚对称.
    范围 [0, 2], 其中 0=完全匹配, 2=完全不匹配.
    """
    abs_true = abs(true_val)
    abs_est = abs(est_val)
    denom = abs_true + abs_est
    if denom < 1e-12:
        return 0.0
    return 2.0 * abs(true_val - est_val) / denom


# ─── Output Format Types ────────────────────────────────────────────────────

class OutputFormat(Enum):
    JSON = auto()
    CSV = auto()
    LOG = auto()


# ─── Benchmark Record ───────────────────────────────────────────────────────

@dataclass
class BenchmarkRecord:
    """One benchmark measurement."""
    record_id: int
    operation: str
    predicted_cost_us: float = 0.0
    actual_cost_us: float = 0.0
    routing_decision: str = "CPU"
    device: str = ""
    n_rows: int = 0
    data_bytes: int = 0
    rel_error: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.actual_cost_us > 0 and self.predicted_cost_us > 0:
            self.rel_error = cal_rel_error(self.actual_cost_us, self.predicted_cost_us)

    def dump_debug(self, prefix: str = "") -> str:
        lines = [
            f"{prefix}╔══ BenchmarkRecord #{self.record_id} ═════════════════",
            f"{prefix}║ operation    = {self.operation} ({record_format_dict.get(self.operation, '?')})",
            f"{prefix}║ predicted    = {self.predicted_cost_us:,.1f}µs",
            f"{prefix}║ actual       = {self.actual_cost_us:,.1f}µs",
            f"{prefix}║ rel_error    = {self.rel_error:.4f}",
            f"{prefix}║ routing      = {self.routing_decision}",
            f"{prefix}║ device       = {self.device}",
            f"{prefix}║ n_rows       = {self.n_rows:,}",
            f"{prefix}║ data_bytes   = {self.data_bytes:,}",
            f"{prefix}║ timestamp    = {self.timestamp:.3f}",
            f"{prefix}║ metadata     = {self.metadata}",
            f"{prefix}╚══════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


# ─── Experiment Metadata ─────────────────────────────────────────────────────

@dataclass
class ExperimentMeta:
    """Experiment-level metadata."""
    experiment_id: str = "exp_001"
    workload_name: str = "default"
    template_id: int = 0
    n_samples: int = 100
    tolerance: float = 0.2
    blend_factor: float = 0.5
    gpu_arch: str = "H100"              # v3: 更新到 H100
    n_workers: int = 1
    sharding_strategy: str = "FULL_SHARD"
    cost_model_version: str = "v3.0"    # v3

    def dump_debug(self, prefix: str = "") -> str:
        lines = [
            f"{prefix}╔══ ExperimentMeta ═══════════════════════════════",
            f"{prefix}║ experiment_id  = {self.experiment_id}",
            f"{prefix}║ workload       = {self.workload_name}",
            f"{prefix}║ template_id    = {self.template_id}",
            f"{prefix}║ n_samples      = {self.n_samples}",
            f"{prefix}║ tolerance      = {self.tolerance}",
            f"{prefix}║ blend_factor   = {self.blend_factor}",
            f"{prefix}║ gpu_arch       = {self.gpu_arch}",
            f"{prefix}║ n_workers      = {self.n_workers}",
            f"{prefix}║ sharding       = {self.sharding_strategy}",
            f"{prefix}║ cm_version     = {self.cost_model_version}",
            f"{prefix}╚═══════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


# ─── Timing Utility ──────────────────────────────────────────────────────────
# v3 变更: 增加 IQR 异常值剔除

def measure_with_median(func, *args, times: int = 5,
                        debug_print: bool = True,
                        trim_outliers: bool = True,
                        **kwargs) -> float:
    """Run a function N times and return median latency.

    v3 变更: trim_outliers=True 时, 先用 IQR 法剔除异常试次,
    再取中位数. 防止偶尔的 GC/context-switch 尖刺污染结果.
    """
    latencies = []
    for trial in range(times):
        start = time.time()
        func(*args, **kwargs)
        elapsed_us = (time.time() - start) * 1e6
        latencies.append(elapsed_us)

        if debug_print and trial == 0:
            print(f"  [data_writer] Trial 0: {elapsed_us:.1f}µs")

    # v3: IQR 异常值剔除
    if trim_outliers and len(latencies) >= 5:
        s = sorted(latencies)
        q1 = s[len(s) // 4]
        q3 = s[3 * len(s) // 4]
        iqr = q3 - q1
        upper_fence = q3 + 1.5 * iqr
        latencies = [x for x in latencies if x <= upper_fence]
        if not latencies:
            latencies = s  # 全部被剔除则恢复

    # Median without numpy
    latencies.sort()
    n = len(latencies)
    if n % 2 == 1:
        median = latencies[n // 2]
    else:
        median = (latencies[n // 2 - 1] + latencies[n // 2]) / 2

    if debug_print:
        print(f"  [data_writer] Median of {times} trials: {median:.1f}µs "
              f"(min={latencies[0]:.1f}, max={latencies[-1]:.1f})")

    return median


# ─── Batch Cost Collection ───────────────────────────────────────────────────
# v3 变更: 增加可选的 filter_outliers 参数

def collect_all_costs(records: List[BenchmarkRecord],
                      debug_print: bool = True,
                      filter_outliers: bool = False) -> List[float]:
    """Collect actual costs from all benchmark records.

    v3: filter_outliers=True 时用 MAD (median absolute deviation) 过滤.
    """
    costs = [rec.actual_cost_us for rec in records]

    if filter_outliers and len(costs) >= 10:
        s = sorted(costs)
        median_val = s[len(s) // 2]
        abs_devs = sorted(abs(c - median_val) for c in costs)
        mad = abs_devs[len(abs_devs) // 2]
        threshold = median_val + 5.0 * max(mad, 1e-6)
        costs = [c for c in costs if c <= threshold]

    if debug_print:
        print(f"  [data_writer] Collected {len(costs)} costs")

    return costs


# ─── Hint/Config Generation ─────────────────────────────────────────────────

def gen_config_record(operation: str, device: str,
                      params: Optional[Dict[str, Any]] = None) -> str:
    """Generate a cost model configuration record string."""
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
    """Writes benchmark results to structured output files.

    v3 变更:
      - add_record 内部用 Welford 在线算法维护 running mean/var
      - write_json summary 增加 SMAPE 分位数 (p50/p90/p95)
    """

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
        # v3: Welford 在线统计
        self._welford_n = 0
        self._welford_mean = 0.0
        self._welford_m2 = 0.0

        if debug_print:
            print(f"\n[data_writer] Initialized DataWriter")
            print(f"  output_dir = {output_dir}")
            print(self._experiment.dump_debug("  "))

    def init_log(self, debug_print: Optional[bool] = None) -> str:
        """Initialise log file."""
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
        """Add a benchmark record.

        v3: 同时用 Welford 在线更新 error 统计.
        """
        self._record_counter += 1
        record = BenchmarkRecord(
            record_id=self._record_counter,
            operation=operation,
            predicted_cost_us=predicted_us,
            actual_cost_us=actual_us,
            routing_decision=routing,
            device=device,
            n_rows=n_rows,
            data_bytes=data_bytes,
            metadata=metadata or {},
        )
        self._records.append(record)

        # v3: Welford 在线更新
        self._welford_n += 1
        delta = abs(record.rel_error) - self._welford_mean
        self._welford_mean += delta / self._welford_n
        delta2 = abs(record.rel_error) - self._welford_mean
        self._welford_m2 += delta * delta2

        if self._debug and self._record_counter % 10 == 0:
            std = math.sqrt(self._welford_m2 / self._welford_n) if self._welford_n > 1 else 0.0
            print(f"  [data_writer] Record #{self._record_counter}: {operation} "
                  f"pred={predicted_us:.1f}µs actual={actual_us:.1f}µs "
                  f"err={record.rel_error:.3f} running_mean={self._welford_mean:.4f}±{std:.4f}")

        return record

    def _error_quantiles(self) -> Dict[str, float]:
        """v3: 计算 SMAPE 误差分位数."""
        if not self._records:
            return {"p50": 0.0, "p90": 0.0, "p95": 0.0}
        errors = sorted(abs(r.rel_error) for r in self._records)
        n = len(errors)
        return {
            "p50": errors[int(n * 0.50)],
            "p90": errors[min(n - 1, int(n * 0.90))],
            "p95": errors[min(n - 1, int(n * 0.95))],
        }

    def write_json(self, filepath: Optional[str] = None,
                   debug_print: Optional[bool] = None) -> str:
        """Write records to JSON.

        v3: summary 增加 error_quantiles.
        """
        dp = debug_print if debug_print is not None else self._debug

        if filepath is None:
            exp = self._experiment
            filepath = os.path.join(
                self._output_dir,
                f"{exp.experiment_id}_{exp.workload_name}_N{exp.n_samples}.json"
            )

        quantiles = self._error_quantiles()

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
                "cost_model_version": self._experiment.cost_model_version,
            },
            "summary": {
                "n_records": len(self._records),
                "mean_rel_error": self._welford_mean,  # v3: 直接用在线计算的
                "gpu_routed": sum(1 for r in self._records if r.routing_decision == "GPU"),
                "cpu_routed": sum(1 for r in self._records if r.routing_decision == "CPU"),
                "error_quantiles": quantiles,  # v3 新增
            },
            "records": [
                {
                    "id": r.record_id,
                    "operation": r.operation,
                    "predicted_cost": r.predicted_cost_us,
                    "actual_cost": r.actual_cost_us,
                    "rel_error": r.rel_error,
                    "routing": r.routing_decision,
                    "device": r.device,
                    "n_rows": r.n_rows,
                    "data_bytes": r.data_bytes,
                    "timestamp": r.timestamp,
                    "metadata": r.metadata,
                }
                for r in self._records
            ],
        }

        self._write_count += 1

        if dp:
            print(f"\n  [data_writer] write_json #{self._write_count}: {filepath}")
            print(f"    records: {len(self._records)}")
            print(f"    mean_rel_error: {output['summary']['mean_rel_error']:.4f}")
            print(f"    error p50/p90/p95: {quantiles['p50']:.4f}/{quantiles['p90']:.4f}/{quantiles['p95']:.4f}")
            print(f"    GPU/CPU split: {output['summary']['gpu_routed']}/{output['summary']['cpu_routed']}")

        return json.dumps(output, indent=2)

    def write_csv(self, filepath: Optional[str] = None,
                  debug_print: Optional[bool] = None) -> str:
        """Write records to CSV format."""
        dp = debug_print if debug_print is not None else self._debug

        if filepath is None:
            exp = self._experiment
            filepath = os.path.join(
                self._output_dir,
                f"{exp.experiment_id}_{exp.workload_name}_N{exp.n_samples}.csv"
            )

        header = "id,operation,predicted_us,actual_us,rel_error,routing,device,n_rows,data_bytes"
        lines = [header]
        for r in self._records:
            lines.append(
                f"{r.record_id},{r.operation},{r.predicted_cost_us:.2f},"
                f"{r.actual_cost_us:.2f},{r.rel_error:.4f},{r.routing_decision},"
                f"{r.device},{r.n_rows},{r.data_bytes}"
            )

        csv_text = "\n".join(lines)
        self._write_count += 1

        if dp:
            print(f"\n  [data_writer] write_csv #{self._write_count}: {filepath}")
            print(f"    rows: {len(self._records)}")

        return csv_text

    def write_log(self, debug_print: Optional[bool] = None) -> str:
        """Write structured log."""
        dp = debug_print if debug_print is not None else self._debug

        lines = []
        total_predicted = 0.0
        total_actual = 0.0
        better_count = 0

        for r in self._records:
            total_predicted += r.predicted_cost_us
            total_actual += r.actual_cost_us
            is_close = abs(r.rel_error) < self._experiment.tolerance
            if is_close:
                better_count += 1

            line = (f"Record {r.record_id}-{self._experiment.workload_name}-"
                    f"{self._experiment.blend_factor}: "
                    f"{r.operation} on {r.device}: "
                    f"predicted={r.predicted_cost_us:.1f}µs, "
                    f"actual={r.actual_cost_us:.1f}µs, "
                    f"err={r.rel_error:.4f}, "
                    f"routing={r.routing_decision}, "
                    f"{better_count}/{r.record_id} within tolerance")
            lines.append(line)

        n = len(self._records)
        if n > 0:
            summary = (f"SUMMARY: Predicted avg: {total_predicted/n:.1f}µs, "
                      f"Actual avg: {total_actual/n:.1f}µs, "
                      f"Ratio: {total_predicted/max(0.001, total_actual):.3f}, "
                      f"{better_count}/{n} within tolerance={self._experiment.tolerance}")
            lines.append(summary)

        log_text = "\n".join(lines)
        self._write_count += 1

        if dp:
            print(f"\n  [data_writer] write_log #{self._write_count}")
            if lines:
                print(f"    {lines[-1]}")

        return log_text

    def dump_state(self) -> str:
        """Full state dump."""
        n = len(self._records)
        std = math.sqrt(self._welford_m2 / max(1, self._welford_n)) if self._welford_n > 1 else 0.0
        gpu_count = sum(1 for r in self._records if r.routing_decision == "GPU")

        lines = [
            "╔══ DataWriter State ═══════════════════════════════════",
            f"║ output_dir     = {self._output_dir}",
            f"║ n_records      = {n}",
            f"║ write_count    = {self._write_count}",
            f"║ log_file       = {self._log_file or '(not initialised)'}",
            f"║ mean_abs_error = {self._welford_mean:.4f} (online)",
            f"║ std_abs_error  = {std:.4f} (online)",
            f"║ GPU/CPU        = {gpu_count}/{n - gpu_count}",
            "║",
            self._experiment.dump_debug("║ "),
        ]
        if self._records:
            lines.append("║")
            lines.append("║ ── Last 5 Records ──")
            for r in self._records[-5:]:
                lines.append(f"║   #{r.record_id}: {r.operation} "
                           f"pred={r.predicted_cost_us:.0f} act={r.actual_cost_us:.0f} "
                           f"err={r.rel_error:.3f} → {r.routing_decision}")
        lines.append("╚════════════════════════════════════════════════════════")
        return "\n".join(lines)
