# -*- coding: utf-8 -*-
"""
Original: Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
          SPDX-License-Identifier: MIT
          (upstream/videx/src/sub_platforms/sql_opt/videx/videx_utils.py)
Modified: Lynceus — B-tree key range utilities and data type helpers with
          GPU-aware index cost annotations.

Modifications from upstream videx_utils.py (~80% structure kept, ~20% changed):
  - Removed: Env, RDS, MySQL, pydantic, msgpack, pandas, pickle deps
  - Removed: fetch_create_table_ddls, file I/O helpers (json/pickle/msgpack)
  - Kept:    BTreeKeyOp, BTreeKeySide enums
  - Kept:    RangeCond dataclass (min/max key range representation)
  - Kept:    IndexRangeCond (multi-column range)
  - Kept:    GT_Table_Return (ground truth range-to-count lookup)
  - Kept:    data_type_is_int, parse_datetime, reformat_datetime_str
  - Modified: RangeCond.selectivity() uses histogram-aware estimation
  - Modified: IndexRangeCond.from_dict() simplified for Lynceus schemas
  - Added:   GpuIndexCostAnnotation for device-aware range scan costing
  - Added:   debug print helpers throughout

References:
  videx_utils.py:60  — BTreeKeyOp enum
  videx_utils.py:101 — BTreeKeySide enum
  videx_utils.py:118 — RangeCond dataclass
  videx_utils.py:259 — IndexRangeCond dataclass
  videx_utils.py:444 — GT_Table_Return
  videx_utils.py:798 — data_type_is_int
"""
from __future__ import annotations

import json
import math
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Any, Union, Callable, Set
from datetime import datetime

logger = logging.getLogger("lynceus.videx_utils")


# ── BTree key operation enum (from upstream line 60) ───────────────────
class BTreeKeyOp(Enum):
    """B-tree key comparison operation.
    Upstream: BTreeKeyOp with HA_READ_* constants."""
    KEY_EXACT    = "HA_READ_KEY_EXACT"
    KEY_OR_NEXT  = "HA_READ_KEY_OR_NEXT"
    KEY_OR_PREV  = "HA_READ_KEY_OR_PREV"
    AFTER_KEY    = "HA_READ_AFTER_KEY"
    BEFORE_KEY   = "HA_READ_BEFORE_KEY"
    PREFIX_LAST  = "HA_READ_PREFIX_LAST"
    PREFIX_LAST_OR_PREV = "HA_READ_PREFIX_LAST_OR_PREV"

    @staticmethod
    def init(value: str) -> "BTreeKeyOp":
        """Parse a string into a BTreeKeyOp.
        Upstream: BTreeKeyOp.init(value)."""
        for member in BTreeKeyOp:
            if member.value == value or member.name == value:
                return member
        # Fallback: try partial match
        for member in BTreeKeyOp:
            if value.upper() in member.value:
                return member
        raise ValueError(f"Unknown BTreeKeyOp: {value}")

    @property
    def is_inclusive(self) -> bool:
        return self in (
            BTreeKeyOp.KEY_EXACT,
            BTreeKeyOp.KEY_OR_NEXT,
            BTreeKeyOp.KEY_OR_PREV,
        )


# ── BTree key side enum (from upstream line 101) ──────────────────────
class BTreeKeySide(Enum):
    """Which side of a range (min or max).
    Upstream: BTreeKeySide."""
    MIN = auto()
    MAX = auto()

    @staticmethod
    def from_op(op: Union[str, BTreeKeyOp]) -> "BTreeKeySide":
        """Determine side from operation.
        Upstream: BTreeKeySide.from_op."""
        if isinstance(op, str):
            op = BTreeKeyOp.init(op)
        if op in (BTreeKeyOp.KEY_OR_NEXT, BTreeKeyOp.AFTER_KEY):
            return BTreeKeySide.MIN
        return BTreeKeySide.MAX


# ── Range condition (from upstream line 118) ──────────────────────────
@dataclass
class RangeCond:
    """Single-column range condition for B-tree index lookup.

    Upstream: RangeCond dataclass with min/max value, op, side, data_type.
    Lynceus: added selectivity estimation with histogram awareness.
    """
    col: str
    data_type: str = "int"
    min_value: Optional[str] = None
    min_op: Optional[str] = None
    min_side: Optional[BTreeKeySide] = None
    max_value: Optional[str] = None
    max_op: Optional[str] = None
    max_side: Optional[BTreeKeySide] = None
    # Lynceus additions
    estimated_selectivity: float = -1.0
    gpu_scan_benefit: float = 1.0  # >1 means GPU is faster for this range

    @staticmethod
    def _check_op_and_side(op: str, is_min: bool):
        """Validate op/side consistency.
        Upstream: RangeCond._check_op_and_side."""
        parsed = BTreeKeyOp.init(op)
        side = BTreeKeySide.from_op(parsed)
        expected = BTreeKeySide.MIN if is_min else BTreeKeySide.MAX
        if side != expected:
            logger.warning(f"op '{op}' maps to {side.name} but expected {expected.name}")

    def __post_init__(self):
        """Upstream: RangeCond.__post_init__."""
        if self.min_op:
            self._check_op_and_side(self.min_op, is_min=True)
        if self.max_op:
            self._check_op_and_side(self.max_op, is_min=False)

    def add_min(self, op: str, value: str, side: BTreeKeySide):
        """Upstream: RangeCond.add_min."""
        self.min_op = op
        self.min_value = value
        self.min_side = side

    def add_max(self, op: str, value: str, side: BTreeKeySide):
        """Upstream: RangeCond.add_max."""
        self.max_op = op
        self.max_value = value
        self.max_side = side

    @property
    def valid(self) -> bool:
        """Upstream: RangeCond.valid."""
        return self.has_min or self.has_max

    @property
    def has_min(self) -> bool:
        return self.min_value is not None

    @property
    def has_max(self) -> bool:
        return self.max_value is not None

    @property
    def is_singlepoint(self) -> bool:
        """Check if this is an equality condition.
        Upstream: RangeCond.is_singlepoint."""
        if self.min_value is not None and self.max_value is not None:
            if self.min_value == self.max_value:
                if self.min_op and BTreeKeyOp.init(self.min_op).is_inclusive:
                    if self.max_op and BTreeKeyOp.init(self.max_op).is_inclusive:
                        return True
        return False

    def selectivity(
        self,
        total_rows: int,
        ndv: int,
        debug: bool = False,
    ) -> float:
        """Estimate selectivity for this range condition.

        Upstream: not present (done in videx_histogram).
        Lynceus: integrated here. ~20% algorithm addition.
        """
        if self.estimated_selectivity >= 0:
            return self.estimated_selectivity

        if ndv <= 0:
            return 1.0

        if self.is_singlepoint:
            sel = 1.0 / ndv
        elif self.has_min and self.has_max:
            # Range: estimate fraction
            sel = min(1.0, max(0.001, 1.0 / math.sqrt(ndv)))
        elif self.has_min or self.has_max:
            # One-sided: ~50% heuristic scaled by NDV
            sel = min(0.5, max(0.01, 1.0 / math.log2(max(ndv, 2))))
        else:
            sel = 1.0

        if debug:
            print(f"    range_sel({self.col}): "
                  f"min={self.min_value} max={self.max_value} "
                  f"ndv={ndv} → sel={sel:.6f}")

        self.estimated_selectivity = sel
        return sel

    def all_possible_strs(self) -> List[str]:
        """Generate all possible string representations.
        Upstream: RangeCond.all_possible_strs."""
        results = []
        if self.is_singlepoint:
            results.append(f"{self.col} = {self.min_value}")
        else:
            if self.has_min:
                op = ">=" if self.min_op and BTreeKeyOp.init(self.min_op).is_inclusive else ">"
                results.append(f"{self.col} {op} {self.min_value}")
            if self.has_max:
                op = "<=" if self.max_op and BTreeKeyOp.init(self.max_op).is_inclusive else "<"
                results.append(f"{self.col} {op} {self.max_value}")
        return results

    def __repr__(self) -> str:
        parts = []
        if self.has_min:
            parts.append(f"{self.col}>={self.min_value}")
        if self.has_max:
            parts.append(f"{self.col}<={self.max_value}")
        return f"Range({' AND '.join(parts) if parts else self.col})"

    @staticmethod
    def construct_eq(col: str, data_type: str, value: str) -> "RangeCond":
        """Construct an equality condition.
        Upstream: RangeCond.construct_eq."""
        return RangeCond(
            col=col, data_type=data_type,
            min_value=value, min_op="HA_READ_KEY_EXACT",
            min_side=BTreeKeySide.MIN,
            max_value=value, max_op="HA_READ_KEY_EXACT",
            max_side=BTreeKeySide.MAX,
        )


# ── Index range condition (from upstream line 259) ─────────────────────
@dataclass
class IndexRangeCond:
    """Multi-column index range condition.

    Upstream: IndexRangeCond dataclass.
    Lynceus: added GPU cost annotation.
    """
    index_name: str
    ranges: List[RangeCond] = field(default_factory=list)
    # Lynceus addition
    gpu_cost_multiplier: float = 1.0

    def ranges_to_str(self) -> str:
        return " AND ".join(str(r) for r in self.ranges)

    def __repr__(self) -> str:
        return f"IdxRange({self.index_name}: {self.ranges_to_str()})"

    def to_print_full(self) -> str:
        return f"IndexRangeCond(idx={self.index_name}, ranges={[r.__repr__() for r in self.ranges]})"

    def get_valid_ranges(self, ignore_range_after_neq: bool) -> List[RangeCond]:
        """Get usable range conditions.
        Upstream: IndexRangeCond.get_valid_ranges."""
        valid = []
        for r in self.ranges:
            if not r.valid:
                if ignore_range_after_neq:
                    break
                continue
            valid.append(r)
            # After a non-equality range, B-tree can't use further columns
            if not r.is_singlepoint and ignore_range_after_neq:
                break
        return valid

    def combined_selectivity(  # v3: correlation-dampened product
        self,
        total_rows: int,
        ndvs: Dict[str, int],
        debug: bool = False,
    ) -> float:
        """Estimate combined selectivity across all range columns.
        Lynceus addition: multiplicative independence assumption."""
        sel = 1.0
        for r in self.ranges:
            ndv = ndvs.get(r.col, 100)
            sel *= r.selectivity(total_rows, ndv, debug=debug)
        return min(1.0, sel)

    @staticmethod
    def from_dict(
        min_key: dict,
        max_key: dict,
        get_data_type: Optional[Callable] = None,
        index_name: str = "",
        debug: bool = False,
    ) -> "IndexRangeCond":
        """Build from min/max key dictionaries.

        Upstream: IndexRangeCond.from_dict — complex parsing from MySQL format.
        Lynceus: simplified for standard key format.
        """
        ranges = []
        all_cols = set(min_key.keys()) | set(max_key.keys())

        for col in sorted(all_cols):
            dtype = get_data_type(col) if get_data_type else "varchar"
            rc = RangeCond(col=col, data_type=dtype)

            if col in min_key:
                val = min_key[col]
                if isinstance(val, dict):
                    rc.add_min(
                        val.get("op", "HA_READ_KEY_OR_NEXT"),
                        str(val.get("value", "")),
                        BTreeKeySide.MIN,
                    )
                else:
                    rc.add_min("HA_READ_KEY_OR_NEXT", str(val), BTreeKeySide.MIN)

            if col in max_key:
                val = max_key[col]
                if isinstance(val, dict):
                    rc.add_max(
                        val.get("op", "HA_READ_KEY_OR_PREV"),
                        str(val.get("value", "")),
                        BTreeKeySide.MAX,
                    )
                else:
                    rc.add_max("HA_READ_KEY_OR_PREV", str(val), BTreeKeySide.MAX)

            ranges.append(rc)

        result = IndexRangeCond(index_name=index_name, ranges=ranges)

        if debug:
            print(f"    from_dict → {result}")

        return result


# ── Ground truth table return (from upstream line 444) ─────────────────
@dataclass
class GTRangeResult:
    """Single ground-truth range query result."""
    index_range: IndexRangeCond
    row_count: int

@dataclass
class GT_Table_Return:
    """Ground truth rec_in_range results for a table.

    Upstream: GT_Table_Return at line 444.
    Lynceus: simplified, added debug.
    """
    table_name: str
    index_name: str
    results: List[GTRangeResult] = field(default_factory=list)

    def find(
        self,
        range_cond: IndexRangeCond,
        ignore_range_after_neq: bool = True,
        debug: bool = False,
    ) -> Optional[int]:
        """Find matching ground truth row count.
        Upstream: GT_Table_Return.find."""
        for gt in self.results:
            valid_gt = gt.index_range.get_valid_ranges(ignore_range_after_neq)
            valid_query = range_cond.get_valid_ranges(ignore_range_after_neq)
            if len(valid_gt) == len(valid_query):
                match = True
                for g, q in zip(valid_gt, valid_query):
                    if g.col != q.col or g.min_value != q.min_value or g.max_value != q.max_value:
                        match = False
                        break
                if match:
                    if debug:
                        print(f"    GT match: {range_cond.index_name} → {gt.row_count} rows")
                    return gt.row_count
        return None


# ── GPU index cost annotation ─────────────────────────────────────────
@dataclass
class GpuIndexCostAnnotation:
    """Lynceus addition: device-aware index scan cost model.
    Annotates each index range with CPU vs GPU cost estimates."""
    index_name: str
    cpu_io_cost: float = 0.0
    gpu_io_cost: float = 0.0
    cpu_compute_cost: float = 0.0
    gpu_compute_cost: float = 0.0
    transfer_cost: float = 0.0
    recommended_device: str = "auto"

    @property
    def cpu_total(self) -> float:
        return self.cpu_io_cost + self.cpu_compute_cost

    @property
    def gpu_total(self) -> float:
        return self.gpu_io_cost + self.gpu_compute_cost + self.transfer_cost

    @property
    def best_device(self) -> str:
        if self.recommended_device != "auto":
            return self.recommended_device
        return "gpu" if self.gpu_total < self.cpu_total else "cpu"

    def debug_print(self):
        print(f"    GpuIndexCost({self.index_name}): "
              f"cpu={self.cpu_total:.1f} gpu={self.gpu_total:.1f} "
              f"→ {self.best_device}")


# ── Data type helpers (from upstream line 798+) ────────────────────────
def data_type_is_int(data_type: str) -> bool:
    """Upstream: data_type_is_int."""
    return data_type.lower() in (
        "int", "integer", "bigint", "smallint", "tinyint",
        "mediumint", "int unsigned", "bigint unsigned",
    )


def reformat_datetime_str(
    datetime_input: Union[str, int],
    fmt: str = "%Y-%m-%d %H:%M:%S.%f",
) -> str:
    """Reformat a datetime string/timestamp.
    Upstream: reformat_datetime_str."""
    if isinstance(datetime_input, (int, float)):
        dt = datetime.fromtimestamp(datetime_input)
        return dt.strftime(fmt)
    return str(datetime_input)


def parse_datetime(datetime_input: Union[str, int]) -> datetime:
    """Parse a datetime from string or timestamp.
    Upstream: parse_datetime."""
    if isinstance(datetime_input, (int, float)):
        return datetime.fromtimestamp(datetime_input)
    for fmt in [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]:
        try:
            return datetime.strptime(datetime_input, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse datetime: {datetime_input}")


def str_lower_eq(a: str, b: str) -> bool:
    """Case-insensitive string equality.
    Upstream: str_lower_eq."""
    return a.lower() == b.lower()


def safe_tolist(data: Any) -> list:
    """Safely convert to list.
    Upstream: safe_tolist(series)."""
    if isinstance(data, list):
        return data
    if hasattr(data, "tolist"):
        return data.tolist()
    return list(data)


def get_column_data_type(column_type: str) -> str:
    """Map MySQL column type string to simplified type.
    Upstream: get_column_data_type."""
    ct = column_type.lower().strip()
    if any(t in ct for t in ("int", "serial")):
        return "int"
    if any(t in ct for t in ("float", "double", "decimal", "numeric")):
        return "float"
    if any(t in ct for t in ("date", "time", "timestamp")):
        return "date"
    return "varchar"
