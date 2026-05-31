"""
lynceus_port/data_writer.py — 移植版 — Benchmark data output writer.

Architecture references (ported/adapted from):
  - PAR2QO parse.py (par2qo/code/parse.py:1-101)
    → split_string(), parse_string() — recursive string splitting/parsing
    → gen_join_hints(), gen_scan_hints(), gen_final_hint() — structured
      hint generation from parsed query plans
    → join_type / hint_type_dict mapping tables
  - PAR2QO postgres.py (par2qo/code/postgres.py:1-178)
    → get_real_latency() — benchmark execution with median-of-N timing
    → get_plan_cost_simple() — cost extraction from EXPLAIN JSON
    → get_plan_cost() — full cost extraction with cardinality injection
    → get_all_plan_cost() — batch cost collection across plan candidates
    → DropBufferCache() — cache clearing between benchmark runs
  - PAR2QO utility.py (par2qo/code/utility.py:1-497)
    → clean() — JSON file sanitisation (strip noise lines, normalise format)
    → get_cost_list() — parse cost/latency from benchmark output JSON
    → cal_rel_error() — relative error between predicted and observed
    → evenly_sample_card() — generate evenly spaced sample points
  - PAR2QO diagram.py (par2qo/code/diagram.py:1-565)
    → saveModeltoCache() / loadModelFromCache — JSON model serialisation
    → initLogFile() — structured logging to per-experiment log paths

Modifications from upstream references (~20% original):
  - Removed: psycopg2 database connections, PostgreSQL EXPLAIN parsing
  - Removed: pg_hint_plan integration, ml_cardest injection
  - Removed: numpy/matplotlib/tqdm dependencies
  - Added:   Lynceus cost-model benchmark data formatting
  - Added:   Multi-format output (JSON, CSV, structured log)
  - Added:   Experiment metadata tracking (hardware config, routing decisions)
  - Added:   Comprehensive debug dump at each write stage
  - Changed: SQL query hints → cost model configuration records
  - Changed: plan cost extraction → cost model prediction vs observed logging

Design:
  The data writer serialises benchmark results from cost model evaluation:
  predicted vs actual execution costs, routing decisions (GPU vs CPU),
  hardware utilisation, and calibration quality metrics. This is the
  output side of the benchmark pipeline (benchmark.py collects;
  data_writer.py exports for analysis).
"""

from __future__ import annotations

import math
import time
import json
import logging
import sys

def _dbg(tag, msg):
    print(f"[DBG][{tag}] {msg}", file=sys.stderr)

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

logger = logging.getLogger(__name__)


# ─── String Parsing Utilities ────────────────────────────────────────────────
# Directly adapted from par2qo parse.py split_string / parse_string (lines 1-12).
# In PAR2QO these recursively split SQL hint strings by delimiters;
# here we use them for structured benchmark output formatting.

def split_string(s: str, delimiter: str) -> List[str]:
    """Split string and interleave delimiter tokens.

    Ported from par2qo/code/parse.py:split_string (line 1).

    Original:
        tmp = tmp.split(d)
        tmp_list = []
        for k in range(len(tmp)):
            tmp_list.append(tmp[k])
            if k < len(tmp) - 1:
                tmp_list.append(d)
        return tmp_list

    Used for tokenising structured benchmark output.
    """
    parts = s.split(delimiter)
    result: List[str] = []
    for k in range(len(parts)):
        result.append(parts[k])
        if k < len(parts) - 1:
            result.append(delimiter)
    return result


def parse_string(tokens: List[str], delimiter: str) -> List[str]:
    """Apply split_string recursively to a token list.

    Ported from par2qo/code/parse.py:parse_string (line 10).

    Original:
        final_list = []
        for i in range(len(new_list)):
            if d in new_list[i]:
                tmp_list = split_string(new_list[i], d)
                final_list += tmp_list
            else:
                final_list.append(new_list[i])
        return final_list
    """
    result: List[str] = []
    for token in tokens:
        if delimiter in token:
            result.extend(split_string(token, delimiter))
        else:
            result.append(token)
    return result


# ─── Hint/Record Type Mapping ────────────────────────────────────────────────
# Adapted from par2qo parse.py join_type/hint_type_dict (lines 14-18).
# In PAR2QO these mapped SQL join types to pg_hint_plan names;
# here we map cost model record types to output format keys.

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
# Adapted from par2qo utility.py clean() (line 69-87).
# Original cleaned EXPLAIN JSON output by stripping noise lines;
# we clean benchmark data output for consistent formatting.

def clean_benchmark_data(raw_lines: List[str],
                         del_keys: Optional[List[str]] = None) -> List[str]:
    """Clean benchmark output lines by removing noise.

    Ported from par2qo/code/utility.py:clean (line 69).

    Original:
        del_line_key = ['QUERY PLAN', 'row)', '----']
        for i in del_line_key:
            if i in line: need_to_del = 1
        if need_to_del == 0:
            line = line.replace('+', '')
            line = line.strip() + '\\n'
            new_f.write(line)

    Lynceus: applied to benchmark output instead of EXPLAIN output.
    """
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
# Adapted from par2qo utility.py get_cost_list() (line 90-115).
# Original parsed EXPLAIN JSON for 'Total Cost' and 'Actual Total Time';
# we parse Lynceus benchmark JSON for predicted vs observed costs.

def get_cost_list_from_json(json_str: str,
                            debug_print: bool = True) -> Tuple[List[float], List[float]]:
    """Extract predicted and actual cost lists from benchmark JSON.

    Ported from par2qo/code/utility.py:get_cost_list (line 90).

    Original:
        if "Total Cost" in line:
            cost = line.split(':')[1].split(',')[0]
            est_cost_list.append(float(cost))
        if "Actual Total Time" in line:
            cost = line.split(':')[1].split(',')[0]
            actual_cost_list.append(float(cost))

    Lynceus: extracts predicted_cost and actual_cost from JSON records.
    """
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
# Directly from par2qo utility.py cal_rel_error (line 260-267).

def cal_rel_error(true_val: float, est_val: float) -> float:
    """Log-ratio relative error.

    Ported from par2qo/code/utility.py:cal_rel_error (line 260).

    Original:
        if true > est:
            error = math.log(true / est)
        else:
            error = - math.log(est / true)

    Lynceus: added zero/NaN guards (INV-5).
    """
    if true_val <= 0 or est_val <= 0:
        if true_val <= 0 and est_val <= 0:
            return 0.0
        return float('inf') if true_val <= 0 else float('-inf')
    if math.isnan(true_val) or math.isnan(est_val):
        return 0.0
    if true_val > est_val:
        return math.log(true_val / est_val)
    else:
        return -math.log(est_val / true_val)


# ─── Output Format Types ────────────────────────────────────────────────────

class OutputFormat(Enum):
    """Output format for benchmark data."""
    JSON = auto()
    CSV = auto()
    LOG = auto()           # structured log (like par2qo logging output)


# ─── Benchmark Record ───────────────────────────────────────────────────────
# Adapted from par2qo diagram.py's per-query output_result records
# and postgres.py's get_real_latency return structure.

@dataclass
class BenchmarkRecord:
    """One benchmark measurement — analogous to one row in par2qo's
    output_result list: [sql_id, para_sql, plan_list[robust_plan_id]].

    Extended with Lynceus-specific fields for cost model evaluation.
    """
    record_id: int
    operation: str                    # from record_type_names
    # Cost model evaluation
    predicted_cost_us: float = 0.0    # cost model prediction
    actual_cost_us: float = 0.0       # observed execution time
    routing_decision: str = "CPU"     # CPU or GPU
    # Hardware context
    device: str = ""
    n_rows: int = 0
    data_bytes: int = 0
    # Calibration quality
    rel_error: float = 0.0
    # Metadata
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived fields."""
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
# Adapted from par2qo diagram.py __init__ metadata:
#   db_name, workload_name, query_id, template_id, N, b, tolerance

@dataclass
class ExperimentMeta:
    """Experiment-level metadata — mirrors par2qo Diagram.__init__ params."""
    experiment_id: str = "exp_001"
    workload_name: str = "default"    # par2qo: workload_name
    template_id: int = 0             # par2qo: template_id
    n_samples: int = 100             # par2qo: N
    tolerance: float = 0.2           # par2qo: tolerance
    blend_factor: float = 0.5        # par2qo: b
    # Lynceus-specific
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
# Adapted from par2qo postgres.py get_real_latency (lines 10-72):
# Original ran N trials, collected latencies, returned median.
# We provide a timing wrapper that does the same for cost model evaluation.

def measure_with_median(func, *args, times: int = 5,
                        debug_print: bool = True, **kwargs) -> float:
    """Run a function N times and return median latency.

    Adapted from par2qo/code/postgres.py:get_real_latency (line 10).

    Original:
        for i in range(times):
            cursor_.execute(to_execute_)
            cur_latency = query_plan[0][0][0]['Plan']['Actual Total Time']
            latency_list.append(cur_latency)
        return np.median(np.array(latency_list))

    Lynceus: generalised to any callable, no numpy.
    """
    latencies = []
    for trial in range(times):
        start = time.time()
        func(*args, **kwargs)
        elapsed_us = (time.time() - start) * 1e6
        latencies.append(elapsed_us)

        if debug_print and trial == 0:
            print(f"  [data_writer] Trial 0: {elapsed_us:.1f}µs")

    # Median without numpy: sort and pick middle
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
# Adapted from par2qo postgres.py get_all_plan_cost (lines 163-178):
# Original iterated over plan candidates and collected costs.

def collect_all_costs(records: List[BenchmarkRecord],
                      debug_print: bool = True) -> List[float]:
    """Collect actual costs from all benchmark records.

    Adapted from par2qo/code/postgres.py:get_all_plan_cost (line 163).

    Original:
        opt_cost_list = []
        for hint_id in range(len(cur_plan_list)):
            cost_with_hint, ... = get_plan_cost(...)
            opt_cost_list.append(cost_with_hint)
        return opt_cost_list
    """
    costs = []
    for rec in records:
        costs.append(rec.actual_cost_us)
        if debug_print and len(costs) % 50 == 0:
            print(f"  [data_writer] Collected {len(costs)}/{len(records)} costs "
                  f"(latest: {rec.actual_cost_us:.1f}µs)")
    return costs


# ─── Hint/Config Generation ─────────────────────────────────────────────────
# Adapted from par2qo parse.py gen_final_hint (lines 77-86):
# Original assembled SQL optimizer hints from join/scan components;
# we assemble cost model configuration records.

def gen_config_record(operation: str, device: str,
                      params: Optional[Dict[str, Any]] = None) -> str:
    """Generate a cost model configuration record string.

    Adapted from par2qo/code/parse.py:gen_final_hint (line 77).

    Original:
        result = '/*+'
        for i in gen_scan_hints(scan_mtd):
            result += '\\n' + i
        join_hints, leading = gen_join_hints(str)
        for i in join_hints:
            result += '\\n' + i
        result += '\\n' + leading
        result += ' */'

    Lynceus: generates cost model config instead of SQL hints.
    """
    fmt_name = record_format_dict.get(operation, operation)
    parts = [f"/* CostModelConfig: {fmt_name} */"]
    parts.append(f"  device = {device}")
    if params:
        for k, v in sorted(params.items()):
            parts.append(f"  {k} = {v}")
    parts.append("/* end */")
    return "\n".join(parts)


# ─── Data Writer ─────────────────────────────────────────────────────────────
# Main class: orchestrates writing benchmark results to files.

class DataWriter:
    """Writes benchmark results to structured output files.

    Adapted from par2qo diagram.py's model serialisation flow:
      - saveModeltoCache → write_json
      - initLogFile → init_log
      - output_result.append → add_record

    Plus utility.py's clean() for output sanitisation and
    postgres.py's get_real_latency for timing metadata.
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

        if debug_print:
            print(f"\n[data_writer] Initialized DataWriter")
            print(f"  output_dir = {output_dir}")
            print(self._experiment.dump_debug("  "))

    def init_log(self, debug_print: Optional[bool] = None) -> str:
        """Initialise log file — mirrors par2qo diagram.py initLogFile().

        Original:
            filename = './log/on-base/' + self.db_name + '/diagram/' +
                str(self.query_id) + '-' + str(self.template_id) + '_' +
                self.workload_name + '_workload_b' + str(self.b) +
                '_N' + str(self.N) + '.log'
            logging.basicConfig(filename=filename, level=logging.INFO)
        """
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
        """Add a benchmark record — like par2qo's output_result.append()."""
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

        if self._debug and self._record_counter % 10 == 0:
            print(f"  [data_writer] Record #{self._record_counter}: {operation} "
                  f"pred={predicted_us:.1f}µs actual={actual_us:.1f}µs "
                  f"err={record.rel_error:.3f}")

        return record

    def write_json(self, filepath: Optional[str] = None,
                   debug_print: Optional[bool] = None) -> str:
        """Write records to JSON — mirrors par2qo diagram.py saveModeltoCache().

        Original pattern:
            filename = f'./reuse/{db_name}/diagram/{query_id}-...json'
            with open(filename, "w") as f:
                json.dump(model_data, f, indent=2)
        """
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
            print(f"    GPU/CPU split: {output['summary']['gpu_routed']}/{output['summary']['cpu_routed']}")

        # Return serialised JSON (actual file write would go here)
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
        """Write structured log — mirrors par2qo logging.info output.

        Adapted from par2qo diagram_best_cost.py evaluate() logging:
            output_string = f"Q{query_id}-{workload_name}..." +
                f"Robust plan is {robust_plan_id}: {latency}, ..."
            logging.info(output_string)
        """
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

            # Format mirrors par2qo's per-query logging
            line = (f"Record {r.record_id}-{self._experiment.workload_name}-"
                    f"{self._experiment.blend_factor}: "
                    f"{r.operation} on {r.device}: "
                    f"predicted={r.predicted_cost_us:.1f}µs, "
                    f"actual={r.actual_cost_us:.1f}µs, "
                    f"err={r.rel_error:.4f}, "
                    f"routing={r.routing_decision}, "
                    f"{better_count}/{r.record_id} within tolerance")
            lines.append(line)

        # Summary (mirrors par2qo's final summary line)
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
                print(f"    {lines[-1]}")  # print summary

        return log_text

    def dump_state(self) -> str:
        """Full state dump for breakpoint inspection."""
        n = len(self._records)
        errors = [abs(r.rel_error) for r in self._records]
        mean_err = sum(errors) / max(1, n)
        max_err = max(errors) if errors else 0.0
        gpu_count = sum(1 for r in self._records if r.routing_decision == "GPU")

        lines = [
            "╔══ DataWriter State ═══════════════════════════════════",
            f"║ output_dir     = {self._output_dir}",
            f"║ n_records      = {n}",
            f"║ write_count    = {self._write_count}",
            f"║ log_file       = {self._log_file or '(not initialised)'}",
            f"║ mean_abs_error = {mean_err:.4f}",
            f"║ max_abs_error  = {max_err:.4f}",
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
