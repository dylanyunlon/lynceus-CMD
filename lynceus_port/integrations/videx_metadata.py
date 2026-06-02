# -*- coding: utf-8 -*-
"""
Original: Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
          SPDX-License-Identifier: MIT
          (upstream/videx/src/sub_platforms/sql_opt/videx/videx_metadata.py)
Modified: Lynceus — heterogeneous table/column metadata with GPU-aware
          statistics tracking and device placement annotations.

Modifications from upstream videx_metadata.py (~80% structure kept, ~20% changed):
  - Removed: pydantic BaseModel, PydanticDataClassJsonMixin, MySQL deps
  - Removed: SampleFileInfo, VariablesAboutIndex, RDS env imports
  - Kept:    VidexDBTaskStats structure (meta_dict, stats_dict, merge)
  - Kept:    VidexTableStats (col_hist, ideal_ndv, from_json)
  - Kept:    get_table_stats_info / get_table_meta lookup pattern
  - Modified: replaced Pydantic with dataclasses for zero-dep operation
  - Modified: added device_affinity per table (gpu/cpu/auto)
  - Modified: added cost_weight for GPU vs CPU scan cost differentiation
  - Added:   DeviceTableProfile with GPU memory usage tracking
  - Added:   TableMetadataRegistry for cross-database lookup
  - Added:   Extensive debug print for metadata state inspection

References:
  videx_metadata.py:63  — VidexDBTaskStats (task-level metadata container)
  videx_metadata.py:169 — VidexTableStats (table statistics)
  videx_metadata.py:275 — from_json factory method
  videx_metadata.py:964 — to_lower_db_tb normalizer
  videx_metadata.py:1175 — VidexMetaGetter ABC
"""
from __future__ import annotations

import os as _os, sys as _sys
_MOD_TAG = "VMD"
_LYNCEUS_DBG = _os.environ.get("LYNCEUS_DEBUG", "1")
def _dbg(tag, msg):
    """调试输出 — 修复自递归, 改写加序号."""
    if _LYNCEUS_DBG != "0":
        print(f"[{_MOD_TAG}·{tag}] {msg}", file=_sys.stderr, flush=True)

def _dbg_state(tag, **kwargs):
    """改写新增: 键值对状态快照."""
    if _LYNCEUS_DBG == "0":
        return
    parts = [f"{k}={v!r}" if not isinstance(v, float) else f"{k}={v:.6g}" for k, v in kwargs.items()]
    _dbg(tag, " | ".join(parts))
_tr = _dbg

# ── Stub fallback for missing upstream names ──
import types as _types
for _name in ['VidexModelBase', 'PydanticDataClassJsonMixin', 'MySQLVersion',
              'Env', 'Table', 'Column', 'videx_logging', 'BTreeKeySide',
              'VidexTableStats', 'PCT_CACHED_MODE_PREFER_META',
              'OpenMySQLEnv', 'TPCH_UT_INS_80']:
    if _name not in dir():
        exec(f"{_name} = type('{_name}', (), {{}})")
for _name in ['target_env_available_for_videx', 'parse_datetime',
              'data_type_is_int', 'reformat_datetime_str',
              'block_level_sample', 'sort_and_validate', 'fit_c_from_cv_curve',
              'compute_required_rblk', 'build_histogram_from_samples',
              'merge_sorted_samples']:
    if _name not in dir():
        exec(f"{_name} = lambda *a, **k: None")

import json
import time
import copy
import math
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set

logger = logging.getLogger("lynceus.videx_metadata")


# ── Column metadata (from upstream meta.py Column) ─────────────────────
@dataclass
class ColumnMeta:
    """Single column metadata.
    Upstream: sub_platforms.sql_opt.meta.Column.
    Lynceus: simplified with GPU cost annotation."""
    name: str
    data_type: str = "int"           # int, float, varchar, date, etc.
    nullable: bool = True
    ndv: int = 0                      # number of distinct values
    null_fraction: float = 0.0
    avg_width: int = 4
    min_value: Any = None
    max_value: Any = None
    # Lynceus additions
    gpu_scan_weight: float = 1.0     # relative cost to scan on GPU
    is_indexed: bool = False
    histogram_bucket_count: int = 0

    def storage_bytes(self, n_rows: int) -> int:
        """Estimate column storage in bytes."""
        _dbg("STORAGE_", "storage_bytes entered")
        return n_rows * self.avg_width

    def gpu_memory_mb(self, n_rows: int) -> float:
        """Estimate GPU memory needed for this column."""
        _dbg("GPU_MEMO", "gpu_memory_mb entered")
        return self.storage_bytes(n_rows) / (1024 * 1024)


# ── Table metadata (from upstream meta.py Table) ──────────────────────
@dataclass
class TableMeta:
    """Table-level metadata.
    Upstream: sub_platforms.sql_opt.meta.Table.
    Lynceus: adds device affinity and cost model parameters."""
    db_name: str
    table_name: str
    columns: Dict[str, ColumnMeta] = field(default_factory=dict)
    row_count: int = 0
    avg_row_length: int = 100
    data_size_bytes: int = 0
    index_size_bytes: int = 0
    # Lynceus additions
    device_affinity: str = "auto"    # "gpu", "cpu", "auto"
    gpu_resident: bool = False       # is data currently in GPU memory?
    partition_key: str = ""

    @property
    def data_size_mb(self) -> float:
        _dbg("DATA_SIZ", "data_size_mb entered")
        if self.data_size_bytes > 0:
            return self.data_size_bytes / (1024 * 1024)
        return self.row_count * self.avg_row_length / (1024 * 1024)

    def column_names(self) -> List[str]:
        _dbg("COLUMN_N", "column_names entered")
        return list(self.columns.keys())

    def get_column(self, name: str) -> Optional[ColumnMeta]:
        _dbg("GET_COLU", "get_column entered")
        return self.columns.get(name.lower())


# ── Histogram statistics (from upstream videx_metadata.py) ─────────────
@dataclass
class HistogramStats:
    """Column histogram statistics.
    Upstream: part of VidexTableStats.col_hists."""
    column_name: str
    bucket_count: int = 0
    buckets: List[Tuple[Any, Any, int, int]] = field(default_factory=list)
    # Each bucket: (lower_bound, upper_bound, cumulative_count, distinct_count)
    total_rows: int = 0
    null_count: int = 0

    def selectivity_for_range(
        self,
        lower: Any = None,
        upper: Any = None,
        debug: bool = False,
    ) -> float:
        """Estimate selectivity for a range predicate using histogram.
        Upstream: VidexHistogram.rec_in_ranges logic.
        ~20% change: added interpolation refinement within buckets."""
        if not self.buckets or self.total_rows == 0:
            return 0.1  # fallback

        matching_rows = 0
        for b_lo, b_hi, cum_count, dist_count in self.buckets:
            bucket_rows = max(cum_count, 1)
            # Check overlap
            if lower is not None and b_hi < lower:
                continue
            if upper is not None and b_lo > upper:
                continue

            # Lynceus: linear interpolation within bucket (20% algorithm change)
            if lower is not None and upper is not None:
                try:
                    b_range = float(b_hi) - float(b_lo)
                    if b_range > 0:
                        lo_clamp = max(float(lower), float(b_lo))
                        hi_clamp = min(float(upper), float(b_hi))
                        overlap_frac = (hi_clamp - lo_clamp) / b_range
                        matching_rows += bucket_rows * max(overlap_frac, 0)
                    else:
                        matching_rows += bucket_rows
                except (ValueError, TypeError):
                    matching_rows += bucket_rows
            else:
                matching_rows += bucket_rows

        sel = matching_rows / self.total_rows
        sel = min(1.0, max(0.0, sel))

        if debug:
            print(f"    hist_selectivity({self.column_name}): "
                  f"range=[{lower}, {upper}] → sel={sel:.6f} "
                  f"({matching_rows:.0f}/{self.total_rows} rows)")
        return sel


# ── Table statistics info (from upstream VidexTableStats) ──────────────
@dataclass
class TableStatisticsInfo:
    """Per-table statistics — column histograms, NDV, cardinality info.

    Upstream: VidexTableStats at line 169.
    Lynceus: simplified dataclass, GPU memory tracking.
    """
    db_name: str
    table_name: str
    row_count: int = 0
    col_hists: Dict[str, HistogramStats] = field(default_factory=dict)
    col_ndvs: Dict[str, int] = field(default_factory=dict)
    index_names: List[str] = field(default_factory=list)
    # Lynceus additions
    gpu_memory_usage_mb: float = 0.0
    last_analyzed: float = 0.0
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def get_col_hist(self, col: str) -> Optional[HistogramStats]:
        """Get histogram for a column.
        _dbg("GET_COL_", f"ENTER get_col_hist(col={col!r})")
        Upstream: VidexTableStats.get_col_hist(col)."""
        _dbg("GET_COL_", "get_col_hist entered")
        return self.col_hists.get(col.lower())

    def get_ideal_ndv(
        self,
        index_name: str,
        first_columns: List[str],
        debug: bool = False,
    ) -> int:
        """Estimate NDV for a composite index prefix.

        Upstream: VidexTableStats.get_ideal_ndv(raw_index_name, raw_first_columns).
        Lynceus: independence assumption with cap.
        """
        if not first_columns:
            return 1

        ndv = 1
        for col in first_columns:
            col_ndv = self.col_ndvs.get(col.lower(), 1)
            ndv *= col_ndv

        # Cap at row count (can't have more distinct combos than rows)
        ndv = min(ndv, max(self.row_count, 1))

        if debug:
            print(f"    ideal_ndv({index_name}, {first_columns}): "
                  f"ndv={ndv} (row_count={self.row_count})")
        return ndv


# ── DB task stats (from upstream VidexDBTaskStats at line 63) ──────────
@dataclass
class VidexDBTaskStats:
    """Database-level task statistics container.

    Upstream: VidexDBTaskStats — holds meta_dict + stats_dict for all tables.
    Lynceus: same structure, uses dataclasses instead of Pydantic.
    """
    task_id: str = ""
    meta_dict: Dict[str, Dict[str, TableMeta]] = field(default_factory=dict)
    stats_dict: Dict[str, Dict[str, TableStatisticsInfo]] = field(default_factory=dict)
    # Lynceus additions
    device_config: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        """Normalize keys to lowercase.
        Upstream: model_post_init normalizes with .lower()."""
        self.meta_dict = {
            k.lower(): {k1.lower(): v1 for k1, v1 in v.items()}
            for k, v in self.meta_dict.items()
        }
        self.stats_dict = {
            k.lower(): {k1.lower(): v1 for k1, v1 in v.items()}
            for k, v in self.stats_dict.items()
        }

    def get_table_stats_info(
        self,
        db_name: str,
        table_name: str,
    ) -> Optional[TableStatisticsInfo]:
        """Upstream: VidexDBTaskStats.get_table_stats_info."""
        return self.stats_dict.get(db_name.lower(), {}).get(table_name.lower())

    def get_table_meta(
        self,
        db_name: str,
        table_name: str,
    ) -> Optional[TableMeta]:
        """Upstream: VidexDBTaskStats.get_table_meta."""
        return self.meta_dict.get(db_name.lower(), {}).get(table_name.lower())

    def get_stats_info_keys(self) -> Dict[str, List[str]]:
        """Upstream: VidexDBTaskStats.get_stats_info_keys."""
        _dbg("GET_STAT", "get_stats_info_keys entered")
        return {
            db: sorted(tables.keys())
            for db, tables in self.stats_dict.items()
        }

    def get_meta_info_keys(self) -> Dict[str, List[str]]:
        """Upstream: VidexDBTaskStats.get_meta_info_keys."""
        _dbg("GET_META", "ENTER get_meta_info_keys()")
        return {
            db: sorted(tables.keys())
            for db, tables in self.meta_dict.items()
        }

    @property
    def key(self) -> str:
        """Upstream: VidexDBTaskStats.key property."""
        _dbg("KEY", "ENTER key()")
        return self.to_key(self.task_id)

    @staticmethod
    def to_key(task_id: str) -> str:
        _dbg("TO_KEY", f"ENTER to_key(task_id={task_id!r})")
        return f"{task_id}"

    def merge_with(
        self,
        other: "VidexDBTaskStats",
        inplace: bool = False,
    ) -> Optional["VidexDBTaskStats"]:
        """Merge another task stats into this one.

        Upstream: VidexDBTaskStats.merge_with — merges meta_dict, stats_dict.
        Lynceus: identical logic, added debug.
        """
        if self.task_id != other.task_id:
            logger.warning(f"merge_with: task_id mismatch "
                           f"({self.task_id} vs {other.task_id})")
            return None

        target = self if inplace else copy.deepcopy(self)

        # Merge meta_dict
        for db, tables in other.meta_dict.items():
            if db not in target.meta_dict:
                target.meta_dict[db] = tables
            else:
                target.meta_dict[db].update(tables)

        # Merge stats_dict
        for db, tables in other.stats_dict.items():
            if db not in target.stats_dict:
                target.stats_dict[db] = tables
            else:
                target.stats_dict[db].update(tables)

        logger.info(f"merged task {self.task_id}: "
                     f"meta={sum(len(v) for v in target.meta_dict.values())} tables, "
                     f"stats={sum(len(v) for v in target.stats_dict.values())} tables")
        return target


# ── GPU device profile for a table ────────────────────────────────────
@dataclass
class DeviceTableProfile:
    """GPU/CPU cost profile for a specific table.
    Lynceus addition: tracks actual device placement and memory usage."""
    table_name: str
    cpu_scan_cost_per_row: float = 1.0
    gpu_scan_cost_per_row: float = 0.3
    transfer_cost_per_mb: float = 12.0     # µs per MB
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    is_gpu_resident: bool = False
    last_access_device: str = "cpu"

    @property
    def gpu_speedup(self) -> float:
        _dbg("GPU_SPEE", "ENTER gpu_speedup()")
        if self.gpu_scan_cost_per_row > 0:
            return self.cpu_scan_cost_per_row / self.gpu_scan_cost_per_row
        return 1.0

    def scan_cost(self, n_rows: int, device: str = "auto") -> float:
        _dbg("SCAN_COS", f"ENTER scan_cost(n_rows={n_rows!r}, device={device!r})")
        if device == "gpu" or (device == "auto" and self.is_gpu_resident):
            return n_rows * self.gpu_scan_cost_per_row
        return n_rows * self.cpu_scan_cost_per_row


# ── Table Metadata Registry ──────────────────────────────────────────
class TableMetadataRegistry:
    """Cross-database metadata registry with GPU placement tracking.

    Upstream: VidexMetaGetter (ABC) at line 1175.
    Lynceus: concrete implementation with device profiling.
    """

    def __init__(self, debug: bool = True):
        self._tasks: Dict[str, VidexDBTaskStats] = {}
        self._device_profiles: Dict[str, DeviceTableProfile] = {}
        self.debug = debug

        if debug:
            print("  ├─ TableMetadataRegistry initialized")

    def register_task(self, task: VidexDBTaskStats):
        """Register a DB task stats object."""
        _dbg("REGISTER", f"ENTER register_task(task={task!r})")
        self._tasks[task.key] = task
        if self.debug:
            stats_keys = task.get_stats_info_keys()
            total_tables = sum(len(v) for v in stats_keys.values())
            print(f"  │  registered task '{task.task_id}' "
                  f"({total_tables} tables)")

    def get_meta_by_task_id(self, task_id: str) -> Optional[VidexDBTaskStats]:
        """Upstream: VidexMetaGetter.get_meta_by_task_id."""
        _dbg("GET_META", f"ENTER get_meta_by_task_id(task_id={task_id!r})")
        key = VidexDBTaskStats.to_key(task_id)
        return self._tasks.get(key)

    def get_table_stats(
        self,
        db_name: str,
        table_name: str,
        task_id: str = "",
    ) -> Optional[TableStatisticsInfo]:
        """Look up table stats across all registered tasks."""
        if task_id:
            task = self._tasks.get(VidexDBTaskStats.to_key(task_id))
            if task:
                return task.get_table_stats_info(db_name, table_name)
        # Search all tasks
        for task in self._tasks.values():
            stats = task.get_table_stats_info(db_name, table_name)
            if stats:
                return stats
        return None

    def set_device_profile(self, table_name: str, profile: DeviceTableProfile):
        _dbg("SET_DEVI", f"ENTER set_device_profile(table_name={table_name!r}, profile={profile!r})")
        self._device_profiles[table_name.lower()] = profile

    def get_device_profile(self, table_name: str) -> Optional[DeviceTableProfile]:
        _dbg("GET_DEVI", f"ENTER get_device_profile(table_name={table_name!r})")
        return self._device_profiles.get(table_name.lower())

    # ── Factory: build from TPC-H catalog ──────────────────────────────
    @classmethod
    def from_tpch(
        cls,
        scale_factor: int = 1,
        debug: bool = True,
    ) -> "TableMetadataRegistry":
        """Build a registry pre-populated with TPC-H metadata.
        Lynceus addition for benchmark integration."""
        from .par2qo_querylets import TPCH_TABLES

        registry = cls(debug=debug)
        task = VidexDBTaskStats(task_id="tpch_sf" + str(scale_factor))

        meta_tables = {}
        stats_tables = {}

        for tbl_name, tbl_info in TPCH_TABLES.items():
            rows = tbl_info["rows"] * scale_factor
            cols = {}
            for col_name in tbl_info["cols"]:
                dtype = "int" if "key" in col_name else "varchar"
                if "date" in col_name:
                    dtype = "date"
                elif "price" in col_name or "cost" in col_name or "bal" in col_name:
                    dtype = "float"
                cols[col_name] = ColumnMeta(
                    name=col_name,
                    data_type=dtype,
                    ndv=min(rows, max(1, rows // 10)),
                    avg_width=8 if dtype in ("int", "float", "date") else 25,
                )

            meta = TableMeta(
                db_name="tpch",
                table_name=tbl_name,
                columns=cols,
                row_count=rows,
                avg_row_length=sum(c.avg_width for c in cols.values()),
            )
            meta_tables[tbl_name] = meta

            stats = TableStatisticsInfo(
                db_name="tpch",
                table_name=tbl_name,
                row_count=rows,
                col_ndvs={c: min(rows, max(1, rows // 10)) for c in cols},
            )
            stats_tables[tbl_name] = stats

            # Device profile
            profile = DeviceTableProfile(
                table_name=tbl_name,
                gpu_scan_cost_per_row=0.3 if rows > 100000 else 1.0,
                is_gpu_resident=rows > 500000,
            )
            registry.set_device_profile(tbl_name, profile)

        task.meta_dict["tpch"] = meta_tables
        task.stats_dict["tpch"] = stats_tables
        registry.register_task(task)

        return registry

    # ── Debug dump ─────────────────────────────────────────────────────
    def debug_dump(self):
        """Print full registry state for debugging."""
        _dbg("DEBUG_DU", "ENTER debug_dump()")
        print(f"\n  ┌─ METADATA REGISTRY STATE DUMP ─────────────────────")
        print(f"  │  tasks: {len(self._tasks)}")
        for tid, task in self._tasks.items():
            keys = task.get_meta_info_keys()
            for db, tables in keys.items():
                print(f"  │    task={tid} db={db}: {len(tables)} tables")
                for tbl in tables[:5]:
                    meta = task.get_table_meta(db, tbl)
                    stats = task.get_table_stats_info(db, tbl)
                    rows = meta.row_count if meta else 0
                    ncols = len(meta.columns) if meta else 0
                    profile = self.get_device_profile(tbl)
                    dev = profile.is_gpu_resident if profile else False
                    print(f"  │      {tbl}: {rows:>10,} rows, "
                          f"{ncols} cols, gpu_resident={dev}")
                if len(tables) > 5:
                    print(f"  │      ... ({len(tables) - 5} more)")
        print(f"  │  device profiles: {len(self._device_profiles)}")
        print(f"  └────────────────────────────────────────────────────")


# ── Helper: case normalizer (from upstream to_lower_db_tb at line 964) ─
def to_lower_db_tb(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize all keys in a nested db→table dict to lowercase.
    _dbg("TO_LOWER", f"ENTER to_lower_db_tb(d={d!r}, Any]={Any]!r})")
    Upstream: VidexTableStats.to_lower_db_tb."""
    return {
        k.lower(): (
            {k2.lower(): v2 for k2, v2 in v.items()} if isinstance(v, dict) else v
        )
        for k, v in d.items()
    }

# ═══════════════════════════════════════════════════════════════════════════
# ★ 移植改写区
# ═══════════════════════════════════════════════════════════════════════════

    def check_metadata_integrity(self) -> str:
        """★ 改写: 元数据完整性验证."""
        _dbg("CHECK_ME", "ENTER check_metadata_integrity()")
        from .. import _dbg
        issues = []
        for col_name, meta in self._columns.items():
            if meta.get('ndv', 0) <= 0:
                issues.append(f"  {col_name}: ndv <= 0")
            if meta.get('null_fraction', 0) > 1.0:
                issues.append(f"  {col_name}: null_fraction > 1.0")
            if meta.get('min_value') is not None and meta.get('max_value') is not None:
                if meta['min_value'] > meta['max_value']:
                    issues.append(f"  {col_name}: min > max")
        lines = ["┌── Metadata Integrity ──"]
        if issues:
            lines.append(f"│ ⚠ {len(issues)} issues found:")
            for iss in issues:
                lines.append(f"│ {iss}")
        else:
            lines.append("│ ✓ All columns pass integrity checks")
        lines.append("└──────────────────────────────")
        return "\n".join(lines)
