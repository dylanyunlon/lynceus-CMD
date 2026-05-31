"""
lynceus_port/data_writer.py — 移植版基准数据输出.

改写 ≈ 20%:
  - cal_rel_error: 增加对称比 (SMAPE-like) 替代单边 log-ratio
  - measure_with_median: 增加 IQR 异常值剔除 (下四分位-上四分位过滤)
  - DataWriter: 增加 rolling window 滑动窗口误差追踪
  - dump_calibration_chart: 断点 ASCII 误差分布图
  - 全链路 print-trace
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

from . import _dbg

logger = logging.getLogger(__name__)


# ─── String Parsing Utilities ────────────────────────────────────────────────

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
    """★ 改写: 对称相对误差 (SMAPE-like) — 替代原始单边 log-ratio.

    SMAPE = |true - est| / ((|true| + |est|) / 2)
    对预测过高/过低给予对称惩罚, 避免 log(0) 和方向偏倚.
    """
    if true_val <= 0 and est_val <= 0:
        return 0.0
    denom = (abs(true_val) + abs(est_val)) / 2.0
    if denom <= 0:
        return float('inf')
    if math.isnan(true_val) or math.isnan(est_val):
        return 0.0
    return abs(true_val - est_val) / denom


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
    experiment_id: str = "exp_001"
    workload_name: str = "default"
    template_id: int = 0
    n_samples: int = 100
    tolerance: float = 0.2
    blend_factor: float = 0.5
    gpu_arch: str = "A100"
    n_workers: int = 1
    sharding_strategy: str = "FULL_SHARD"
    cost_model_version: str = "v1.0"

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

def measure_with_median(func, *args, times: int = 5,
                        debug_print: bool = True,
                        iqr_filter: bool = True,   # ★ 改写: IQR 异常值剔除
                        **kwargs) -> float:
    """★ 改写: 增加 IQR 过滤 — 去除 < Q1-1.5*IQR 和 > Q3+1.5*IQR 的异常值."""
    latencies = []
    for trial in range(times):
        start = time.time()
        func(*args, **kwargs)
        elapsed_us = (time.time() - start) * 1e6
        latencies.append(elapsed_us)
        if debug_print and trial == 0:
            print(f"  [data_writer] Trial 0: {elapsed_us:.1f}µs")

    latencies.sort()

    # ★ IQR 过滤
    if iqr_filter and len(latencies) >= 5:
        q1_idx = len(latencies) // 4
        q3_idx = 3 * len(latencies) // 4
        q1, q3 = latencies[q1_idx], latencies[q3_idx]
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = [v for v in latencies if lo <= v <= hi]
        if filtered:
            latencies = filtered
            if debug_print:
                print(f"  [data_writer] IQR filter: kept {len(filtered)}/{times} trials")

    n = len(latencies)
    if n % 2 == 1:
        median = latencies[n // 2]
    else:
        median = (latencies[n // 2 - 1] + latencies[n // 2]) / 2

    if debug_print:
        print(f"  [data_writer] Median of {n} trials: {median:.1f}µs "
              f"(min={latencies[0]:.1f}, max={latencies[-1]:.1f})")
    return median


# ─── Batch Cost Collection ───────────────────────────────────────────────────

def collect_all_costs(records: List[BenchmarkRecord],
                      debug_print: bool = True) -> List[float]:
    costs = []
    for rec in records:
        costs.append(rec.actual_cost_us)
        if debug_print and len(costs) % 50 == 0:
            print(f"  [data_writer] Collected {len(costs)}/{len(records)} costs "
                  f"(latest: {rec.actual_cost_us:.1f}µs)")
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
    # ★ 改写: 滑动窗口大小
    ROLLING_WINDOW: int = 50

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
        # ★ 改写: 滑动窗口误差追踪
        self._rolling_errors: List[float] = []

        if debug_print:
            print(f"\n[data_writer] Initialized DataWriter")
            print(f"  output_dir = {output_dir}")
            print(self._experiment.dump_debug("  "))

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

        # ★ 改写: 滑动窗口
        self._rolling_errors.append(abs(record.rel_error))
        if len(self._rolling_errors) > self.ROLLING_WINDOW:
            self._rolling_errors.pop(0)

        if self._debug and self._record_counter % 10 == 0:
            rolling_avg = (sum(self._rolling_errors) /
                          max(1, len(self._rolling_errors)))
            print(f"  [data_writer] Record #{self._record_counter}: {operation} "
                  f"pred={predicted_us:.1f}µs actual={actual_us:.1f}µs "
                  f"err={record.rel_error:.3f} rolling_err={rolling_avg:.3f}")

        return record

    def write_json(self, filepath: Optional[str] = None,
                   debug_print: Optional[bool] = None) -> str:
        dp = debug_print if debug_print is not None else self._debug
        if filepath is None:
            exp = self._experiment
            filepath = os.path.join(
                self._output_dir,
                f"{exp.experiment_id}_{exp.workload_name}_N{exp.n_samples}.json"
            )
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
                "mean_rel_error": (sum(abs(r.rel_error) for r in self._records)
                                   / max(1, len(self._records))),
                "gpu_routed": sum(1 for r in self._records if r.routing_decision == "GPU"),
                "cpu_routed": sum(1 for r in self._records if r.routing_decision == "CPU"),
            },
            "records": [
                {
                    "id": r.record_id, "operation": r.operation,
                    "predicted_cost": r.predicted_cost_us,
                    "actual_cost": r.actual_cost_us,
                    "rel_error": r.rel_error, "routing": r.routing_decision,
                    "device": r.device, "n_rows": r.n_rows,
                    "data_bytes": r.data_bytes, "timestamp": r.timestamp,
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
            print(f"    GPU/CPU split: {output['summary']['gpu_routed']}/{output['summary']['cpu_routed']}")
        return json.dumps(output, indent=2)

    def write_csv(self, filepath: Optional[str] = None,
                  debug_print: Optional[bool] = None) -> str:
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
        dp = debug_print if debug_print is not None else self._debug
        lines = []
        total_predicted = total_actual = 0.0
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

    # ★ 改写: ASCII 误差分布图 (断点辅助)
    def dump_calibration_chart(self) -> str:
        """断点辅助: ASCII 误差直方图 — 快速检查标定质量."""
        if not self._records:
            return "(no records)"
        errors = [abs(r.rel_error) for r in self._records]
        buckets = [0] * 10  # 0-0.1, 0.1-0.2, ... 0.9-1.0+
        for e in errors:
            idx = min(9, int(e * 10))
            buckets[idx] += 1
        total = len(errors)
        lines = ["┌── Calibration Error Distribution ──"]
        for i, cnt in enumerate(buckets):
            lo = i * 0.1
            hi = (i + 1) * 0.1 if i < 9 else float('inf')
            bar = "█" * max(1, int(cnt / max(1, total) * 40))
            lines.append(f"│ {lo:.1f}-{hi:.1f}: {bar} ({cnt}/{total})")
        lines.append(f"│ mean={sum(errors)/total:.4f}, "
                     f"max={max(errors):.4f}, "
                     f"p50={sorted(errors)[total//2]:.4f}")
        lines.append("└──────────────────────────────────")
        return "\n".join(lines)

    def dump_state(self) -> str:
        n = len(self._records)
        errors = [abs(r.rel_error) for r in self._records]
        mean_err = sum(errors) / max(1, n)
        max_err = max(errors) if errors else 0.0
        gpu_count = sum(1 for r in self._records if r.routing_decision == "GPU")
        rolling_avg = (sum(self._rolling_errors) /
                      max(1, len(self._rolling_errors))) if self._rolling_errors else 0.0
        lines = [
            "╔══ DataWriter State ═══════════════════════════════════",
            f"║ output_dir     = {self._output_dir}",
            f"║ n_records      = {n}",
            f"║ write_count    = {self._write_count}",
            f"║ log_file       = {self._log_file or '(not initialised)'}",
            f"║ mean_abs_error = {mean_err:.4f}",
            f"║ max_abs_error  = {max_err:.4f}",
            f"║ rolling_err    = {rolling_avg:.4f} (window={self.ROLLING_WINDOW})",
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
