# -*- coding: utf-8 -*-
"""
Original: Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
          SPDX-License-Identifier: MIT
          (upstream/videx/src/sub_platforms/sql_opt/meta.py)
          (upstream/videx/src/sub_platforms/sql_opt/common/db_variable.py)
          (upstream/videx/src/sub_platforms/sql_opt/databases/mysql/common_operation.py)
Modified: Lynceus M123 — Extended metadata model with vector clocks for
          multi-node metadata consistency tracking.

Modifications from upstream (~80% structure kept, ~20% changed):
  - Merged: meta.py (Table/Column/Index/IndexColumn dataclasses) +
            db_variable.py (MysqlVariable, VariablesAboutIndex) +
            common_operation.py (mapping_index_columns, correct_df_type)
  - Kept:   TableId, Column, OrderColumn, IndexType, IndexColumn, Index, Table
  - Kept:   VariableScope, MysqlVariable, SingleValueVariable, MultiValueVariable
  - Kept:   VariablesAboutIndex (optimizer_switch, sort_buffer_size, etc.)
  - Kept:   mapping_index_columns, patch_index_invisible
  - Kept:   correct_df_type_by_mysql_type, replace_illegal_value
  - Kept:   parse_sample_data_to_dataframe, mysql_to_pandas_type
  - Kept:   OpTypeName, JoinItem, JsonMultiValueItem, get_table_uk
  - Modified: Table and Column carry vector clocks for distributed consistency
  - Modified: VariablesAboutIndex tracks variable versions via vector stamps
  - Added:   VectorClock — Lamport-style vector clock for metadata versioning
  - Added:   MetadataVersionTracker — tracks per-table version vectors
  - Added:   _dbg() diagnostic hooks on every function

References:
  meta.py:13       — clean_int helper
  meta.py:28       — TableId
  meta.py:49       — Column
  meta.py:106      — IndexType enum
  meta.py:113      — IndexColumn
  meta.py:188      — Index
  meta.py:204      — Table
  meta.py:261      — OpTypeName
  meta.py:322      — get_table_uk
  meta.py:341      — mysql_to_pandas_type
  db_variable.py:14  — VariableScope
  db_variable.py:20  — MysqlVariable
  db_variable.py:68  — SingleValueVariable
  db_variable.py:84  — MultiValueVariable
  db_variable.py:112 — VariablesAboutIndex
  common_operation.py:19  — parse_from_expression
  common_operation.py:26  — mapping_index_columns
  common_operation.py:44  — patch_index_invisible
  common_operation.py:60  — replace_illegal_value
  common_operation.py:68  — correct_df_type_by_mysql_type
  common_operation.py:109 — parse_sample_data_to_dataframe
"""
from __future__ import annotations

import copy
import hashlib
import logging
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("lynceus.videx_metadata_extended")

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[meta_ext] {tag}: {items}")


# ═══════════════════════════════════════════════════════════════════
#  Vector Clock (algorithm addition ~20%)
# ═══════════════════════════════════════════════════════════════════

class VectorClock:
    """Lamport-style vector clock for distributed metadata versioning.

    Algorithm addition: upstream metadata has no versioning. This adds
    vector clocks so that metadata updates from multiple nodes (e.g.
    parallel VIDEX workers) can be causally ordered and conflicts
    detected via the happens-before partial order.

    Each node is identified by a string ID. The clock is a dict mapping
    node IDs to monotonically increasing counters.
    """

    def __init__(self, node_id: str = "default"):
        self._clocks: Dict[str, int] = {node_id: 0}
        self._node_id = node_id
        _dbg("VectorClock_init", node_id=node_id)

    def tick(self) -> "VectorClock":
        """Increment this node's clock component."""
        self._clocks[self._node_id] = self._clocks.get(self._node_id, 0) + 1
        _dbg("vc_tick", node=self._node_id,
             val=self._clocks[self._node_id])
        return self

    def merge(self, other: "VectorClock") -> "VectorClock":
        """Merge with another vector clock (element-wise max), then tick."""
        _dbg("vc_merge", self_nodes=len(self._clocks),
             other_nodes=len(other._clocks))
        for node, ts in other._clocks.items():
            self._clocks[node] = max(self._clocks.get(node, 0), ts)
        self.tick()
        return self

    def happens_before(self, other: "VectorClock") -> bool:
        """Check if self causally precedes other (self < other)."""
        at_least_one_less = False
        for node in set(list(self._clocks.keys()) + list(other._clocks.keys())):
            s = self._clocks.get(node, 0)
            o = other._clocks.get(node, 0)
            if s > o:
                _dbg("vc_not_before", node=node, self_val=s, other_val=o)
                return False
            if s < o:
                at_least_one_less = True
        result = at_least_one_less
        _dbg("vc_happens_before", result=result)
        return result

    def is_concurrent(self, other: "VectorClock") -> bool:
        """Check if two clocks are concurrent (neither happens-before)."""
        concurrent = (not self.happens_before(other)
                      and not other.happens_before(self))
        _dbg("vc_concurrent", result=concurrent)
        return concurrent

    @property
    def snapshot(self) -> Dict[str, int]:
        return dict(self._clocks)

    @property
    def node_id(self) -> str:
        return self._node_id

    def __repr__(self):
        return f"VC({self._clocks})"


# ═══════════════════════════════════════════════════════════════════
#  Metadata Version Tracker
# ═══════════════════════════════════════════════════════════════════

class MetadataVersionTracker:
    """Tracks vector clock versions per table for consistency.

    Algorithm addition: gives each table metadata entry a vector clock
    so that concurrent updates from different VIDEX workers can be
    detected and resolved (latest-wins or conflict-flag).
    """

    def __init__(self, node_id: str = "default"):
        self._node_id = node_id
        self._versions: Dict[str, VectorClock] = {}
        _dbg("MetaVersionTracker_init", node=node_id)

    def stamp(self, table_key: str) -> VectorClock:
        """Tick the clock for a given table and return the new version."""
        _dbg("version_stamp", key=table_key)
        if table_key not in self._versions:
            self._versions[table_key] = VectorClock(self._node_id)
        vc = self._versions[table_key].tick()
        return vc

    def merge_remote(self, table_key: str,
                     remote_vc: VectorClock) -> VectorClock:
        """Merge a remote vector clock for a table."""
        _dbg("version_merge_remote", key=table_key,
             remote=remote_vc.snapshot)
        if table_key not in self._versions:
            self._versions[table_key] = VectorClock(self._node_id)
        self._versions[table_key].merge(remote_vc)
        return self._versions[table_key]

    def get_version(self, table_key: str) -> Optional[VectorClock]:
        _dbg("get_version", key=table_key)
        return self._versions.get(table_key)

    def is_outdated(self, table_key: str,
                    remote_vc: VectorClock) -> bool:
        """Check if local version is outdated (happens-before remote)."""
        local = self._versions.get(table_key)
        if local is None:
            _dbg("version_no_local", key=table_key)
            return True
        result = local.happens_before(remote_vc)
        _dbg("version_outdated", key=table_key, result=result)
        return result

    def all_versions(self) -> Dict[str, Dict[str, int]]:
        return {k: v.snapshot for k, v in self._versions.items()}


# ═══════════════════════════════════════════════════════════════════
#  clean_int helper (from meta.py:13)
# ═══════════════════════════════════════════════════════════════════

def clean_int(value) -> Optional[int]:
    """Convert various types to int or None. Upstream: clean_int."""
    _dbg("clean_int", value=value, type=type(value).__name__)
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════════
#  TableId (from meta.py:28)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TableId:
    """Composite key identifying a table. Upstream: TableId."""
    db_name: Optional[str] = None
    table_name: Optional[str] = None

    def __hash__(self):
        return hash(f"{self.db_name}.{self.table_name}")

    def __eq__(self, other):
        if not other:
            return False
        return (self.db_name == other.db_name
                and self.table_name == other.table_name)

    def __lt__(self, other):
        if self.db_name < other.db_name:
            return True
        if self.db_name == other.db_name:
            return self.table_name < other.table_name
        return False


# ═══════════════════════════════════════════════════════════════════
#  Column (from meta.py:49)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Column:
    """Table column metadata. Upstream: Column(BaseModel).
    Modified: carries a vector clock version stamp."""
    name: Optional[str] = None
    table: Optional[str] = None
    db: Optional[str] = None
    ordinal_position: Optional[int] = None
    is_nullable: Optional[Union[str, bool]] = None
    data_type: Optional[str] = None
    character_maximum_length: Optional[int] = None
    character_octet_length: Optional[int] = None
    numeric_precision: Optional[int] = None
    numeric_scale: Optional[int] = None
    datetime_precision: Optional[int] = None
    character_set_name: Optional[str] = None
    collation_name: Optional[str] = None
    column_type: Optional[str] = None
    column_key: Optional[str] = None
    default: Optional[str] = None
    unsigned: Optional[bool] = None
    is_pk: Optional[bool] = None
    is_sharding_key: Optional[bool] = None
    auto_increment: bool = False
    invisible: bool = False
    alias: Optional[str] = None
    enum_candidates: Optional[List[str]] = None
    # Vector clock stamp (algorithm addition)
    _version: Optional[Dict[str, int]] = field(default=None, repr=False)

    def __eq__(self, other):
        _dbg("Column_eq", self=str(self), other=str(other))
        return (self.db == other.db and self.table == other.table
                and self.name == other.name)

    @property
    def enum_values(self) -> Optional[List[str]]:
        _dbg("Column_enum_values", name=self.name)
        if self.enum_candidates and len(self.enum_candidates) > 0:
            return self.enum_candidates
        if self.data_type == 'enum' and self.column_type:
            self.enum_candidates = (
                self.column_type.split('(')[1].split(')')[0].split(',')
            )
        return self.enum_candidates

    def stamp_version(self, vc: VectorClock):
        """Attach a vector clock snapshot to this column."""
        _dbg("Column_stamp", name=self.name, vc=vc.snapshot)
        self._version = vc.snapshot

    def __str__(self):
        return f"{self.db}.{self.table}.{self.name}"


# ═══════════════════════════════════════════════════════════════════
#  OrderColumn (from meta.py:94)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OrderColumn(Column):
    """Column with sort direction. Upstream: OrderColumn."""
    asc: bool = True

    @classmethod
    def build_from_column(cls, column: Column, asc: bool):
        _dbg("OrderColumn_build", col=str(column), asc=asc)
        if column is None:
            return None
        oc = cls()
        for attr in ('name', 'table', 'db', 'ordinal_position', 'is_nullable',
                     'data_type', 'character_maximum_length',
                     'character_octet_length', 'numeric_precision',
                     'numeric_scale', 'datetime_precision',
                     'character_set_name', 'collation_name', 'column_type',
                     'column_key', 'auto_increment'):
            setattr(oc, attr, getattr(column, attr, None))
        oc.asc = asc
        return oc


# ═══════════════════════════════════════════════════════════════════
#  IndexType (from meta.py:106)
# ═══════════════════════════════════════════════════════════════════

class IndexType(Enum):
    """Index classification. Upstream: IndexType enum."""
    PRIMARY = 'PRIMARY'
    UNIQUE = 'UNIQUE'
    NORMAL = 'NORMAL'
    FOREIGN_KEY = 'FOREIGN_KEY'


# ═══════════════════════════════════════════════════════════════════
#  IndexColumn (from meta.py:113)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class IndexColumn:
    """Column within an index with cardinality info.
    Upstream: IndexColumn(BaseModel)."""
    name: Optional[str] = None
    cardinality: Optional[int] = None
    sub_part: int = 0
    expression: Optional[str] = None
    collation: str = 'asc'
    column_ref: Optional[Column] = field(default=None, repr=False)
    table_id: Optional[TableId] = None

    @property
    def is_desc(self) -> bool:
        return self.collation == 'desc'

    @classmethod
    def from_column(cls, column: Column, collation: str = 'asc',
                    sub_part: int = 0,
                    expression: Optional[str] = None) -> Optional["IndexColumn"]:
        _dbg("IndexColumn_from_column", col=str(column), collation=collation)
        if column is None:
            return None
        ic = cls(name=column.name)
        ic.column_ref = column
        ic.table_id = TableId(db_name=column.db, table_name=column.table)
        if sub_part == 0 and column.data_type and column.data_type.upper() in ('TEXT', 'LONGTEXT'):
            ic.sub_part = 255
        else:
            ic.sub_part = sub_part
        ic.expression = expression
        ic.collation = collation
        return ic

    @classmethod
    def simple_column(cls, column_name: str, db_name: str,
                      table_name: str, collation: str = 'asc',
                      sub_part: int = 0,
                      expression: Optional[str] = None) -> "IndexColumn":
        _dbg("IndexColumn_simple", col=column_name, db=db_name,
             table=table_name)
        col = Column(name=column_name, table=table_name, db=db_name,
                     data_type='varchar')
        return cls.from_column(col, collation, sub_part, expression)

    @property
    def db_name(self) -> Optional[str]:
        if self.table_id is not None:
            return self.table_id.db_name
        if self.column_ref is not None:
            return self.column_ref.db
        return None

    @property
    def table_name(self) -> Optional[str]:
        if self.table_id is not None:
            return self.table_id.table_name
        if self.column_ref is not None:
            return self.column_ref.table
        return None

    def __eq__(self, other):
        _dbg("IndexColumn_eq", self_name=self.name,
             other_name=getattr(other, 'name', '?'))
        return (self.db_name == other.db_name
                and self.table_name == other.table_name
                and self.name == other.name
                and self.expression == other.expression
                and self.sub_part == other.sub_part
                and self.collation == other.collation)


# ═══════════════════════════════════════════════════════════════════
#  Index (from meta.py:188)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Index:
    """Index metadata. Upstream: Index(IndexBasicInfo)."""
    type: Optional[IndexType] = None
    db_name: Optional[str] = None
    table_name: Optional[str] = None
    name: Optional[str] = None
    is_unique: bool = False
    is_visible: bool = True
    index_type: Optional[str] = None
    columns: List[IndexColumn] = field(default_factory=list)

    def get_column_names(self) -> List[str]:
        _dbg("Index_get_column_names", idx=self.name)
        return [c.name for c in self.columns]

    @property
    def db(self):
        return self.db_name

    @property
    def table(self):
        return self.table_name


# ═══════════════════════════════════════════════════════════════════
#  Table (from meta.py:204) — with vector clock
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Table:
    """Table metadata with vector clock versioning.
    Upstream: Table(BaseModel).
    Modified: adds _vc field for distributed consistency."""
    name: Optional[str] = None
    db: Optional[str] = None
    engine: Optional[str] = None
    row_format: Optional[str] = None
    rows: Optional[int] = None
    avg_row_length: Optional[int] = None
    data_length: Optional[int] = None
    index_length: Optional[int] = None
    data_free: Optional[int] = None
    auto_increment: Optional[int] = None
    create_time: Optional[int] = None
    update_time: Optional[int] = None
    check_time: Optional[int] = None
    collation: Optional[str] = None
    charset: Optional[str] = None
    comment: Optional[str] = None
    ddl: Optional[str] = None
    table_size: Optional[int] = None
    table_type: Optional[str] = None
    create_options: Optional[str] = None
    columns: List[Column] = field(default_factory=list)
    indexes: List[Index] = field(default_factory=list)
    cluster_index_size: Optional[int] = None
    other_index_sizes: Optional[int] = None
    # Vector clock version (algorithm addition)
    _vc: Optional[VectorClock] = field(default=None, repr=False)

    def __post_init__(self):
        _dbg("Table_init", name=self.name, db=self.db)
        if self.indexes:
            for index in self.indexes:
                if index.db_name is None:
                    index.db_name = self.db
                if index.table_name is None:
                    index.table_name = self.name

    @property
    def table_id(self) -> TableId:
        return TableId(db_name=self.db, table_name=self.name)

    def support_optimize(self) -> Tuple[bool, str]:
        _dbg("support_optimize", name=self.name, engine=self.engine)
        if self.engine and str(self.engine).lower() != "innodb":
            return False, f"engine is {self.engine}"
        if not self.indexes:
            return False, f"table {self.name} has no pk"
        pk = next((i for i in self.indexes if i.type == IndexType.PRIMARY),
                  None)
        if pk is None:
            return False, f"table {self.name} has no pk"
        return True, ""

    def stamp_version(self, vc: VectorClock):
        """Attach a vector clock to this table metadata entry."""
        _dbg("Table_stamp", name=self.name, vc=vc.snapshot)
        self._vc = vc
        # Propagate to columns
        for col in self.columns:
            col.stamp_version(vc)

    @property
    def version_snapshot(self) -> Optional[Dict[str, int]]:
        return self._vc.snapshot if self._vc else None


# ═══════════════════════════════════════════════════════════════════
#  OpTypeName (from meta.py:261)
# ═══════════════════════════════════════════════════════════════════

class OpTypeName(Enum):
    """SQL operation type classification.
    Upstream: OpTypeName enum with func_type/func_name properties."""
    EQ_FUNC = (("EQ_FUNC", "EQUAL_FUNC"), ())
    RANGE_FUNC = (("GT_FUNC", "GE_FUNC", "LT_FUNC", "LE_FUNC", "NE_FUNC"), ())
    BETWEEN_FUNC = (("BETWEEN",), ())
    LIKE_FUNC = (("LIKE_FUNC",), ("like", "regexp", "regexp_like"))
    IN_FUNC = (("IN_FUNC",), ("<in_optimizer>",))
    IS_FUNC = (("ISNULL_FUNC", "ISNOTNULL_FUNC"), ())
    MULT_EQUAL_FUNC = (("MULT_EQUAL_FUNC",), ())
    CONSTANT_FUNC = (("TRUE_FUNC", "FALSE_FUNC"), ())
    JSON_FUNC = (("MEMBER_OF_FUNC", "JSON_CONTAINS", "JSON_OVERLAPS"), ())

    @property
    def func_type(self):
        return self.value[0]

    @property
    def func_name(self):
        return self.value[1]

    @classmethod
    def build_from_name(cls, name: str):
        _dbg("OpTypeName_from_name", name=name)
        return cls.__members__.get(name)

    @classmethod
    def build_from_values(cls, values: List[str]):
        _dbg("OpTypeName_from_values", values=values[:3])
        for member in cls:
            if values[0][0] in member.value[0]:
                return member
        return None


# ═══════════════════════════════════════════════════════════════════
#  JoinItem (from meta.py:293)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class JoinItem:
    """Join condition representation. Upstream: JoinItem."""
    operation: Optional[str] = None
    op_type: Optional[OpTypeName] = None
    left: Optional[Column] = None
    right: Optional[Column] = None

    def __str__(self):
        return f"{self.left} {self.operation} {self.right}"


# ═══════════════════════════════════════════════════════════════════
#  JsonMultiValueItem (from meta.py:305)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class JsonMultiValueItem:
    """JSON multi-value index expression. Upstream: JsonMultiValueItem."""
    column_func_str: Optional[str] = None
    array_type: Optional[str] = None
    column: Optional[Column] = None

    @property
    def index_expression(self) -> str:
        _dbg("json_mv_index_expr", func=self.column_func_str)
        return f"(CAST({self.column_func_str} AS {self.array_type} ARRAY))"


# ═══════════════════════════════════════════════════════════════════
#  get_table_uk (from meta.py:322)
# ═══════════════════════════════════════════════════════════════════

def get_table_uk(table_meta: Table) -> List[List[str]]:
    """Return all unique key column lists for a table.
    Upstream: get_table_uk."""
    _dbg("get_table_uk", table=table_meta.name)
    uks = []
    for idx in table_meta.indexes:
        if idx.type == IndexType.UNIQUE:
            uk_cols = [c.name for c in idx.columns]
            uks.append(uk_cols)
    _dbg("get_table_uk_done", n_uks=len(uks))
    return uks


# ═══════════════════════════════════════════════════════════════════
#  mysql_to_pandas_type (from meta.py:341)
# ═══════════════════════════════════════════════════════════════════

def mysql_to_pandas_type(mysql_type: str) -> str:
    """Convert MySQL data type string to pandas dtype.
    Upstream: mysql_to_pandas_type."""
    _dbg("mysql_to_pandas_type", mysql_type=mysql_type)
    t = mysql_type.lower()
    if 'bigint' in t and 'unsigned' in t:
        return 'uint64'
    elif 'bigint' in t:
        return 'int64'
    elif 'int' in t and 'unsigned' in t:
        return 'uint32'
    elif 'int' in t:
        return 'int32'
    elif 'varchar' in t or 'text' in t:
        return 'object'
    elif 'double' in t:
        return 'float64'
    elif 'float' in t:
        return 'float32'
    elif 'date' in t or 'datetime' in t:
        return 'datetime64[ns]'
    else:
        return 'object'


# ═══════════════════════════════════════════════════════════════════
#  Variable system (from db_variable.py)
# ═══════════════════════════════════════════════════════════════════

class VariableScope(Enum):
    """Variable scope. Upstream: VariableScope enum."""
    SESSION = "SESSION"
    GLOBAL = "GLOBAL"
    BOTH = "BOTH"


@dataclass
class MysqlVariable:
    """Base MySQL variable descriptor.
    Upstream: MysqlVariable(BaseModel).
    Modified: carries vector clock version stamp."""
    name: str = ""
    scope: VariableScope = VariableScope.BOTH
    version: List[str] = field(default_factory=list)
    dynamic: bool = False
    read_only: bool = False
    need_set: bool = True
    is_update: bool = False
    # Vector clock stamp for variable version tracking
    _var_version: Optional[Dict[str, int]] = field(default=None, repr=False)

    def set_value(self, val):
        _dbg("MysqlVar_set_value", name=self.name, val=val)
        raise NotImplementedError

    def get_value(self, key=None):
        _dbg("MysqlVar_get_value", name=self.name, key=key)
        raise NotImplementedError

    def stamp_version(self, vc: VectorClock):
        _dbg("MysqlVar_stamp", name=self.name, vc=vc.snapshot)
        self._var_version = vc.snapshot

    def generate_set_statements(self, version_str: str) -> List[str]:
        _dbg("MysqlVar_gen_set_stmts", name=self.name, version=version_str)
        ret = []
        if (self.need_set and self.is_update
                and version_str in self.version):
            if self.scope in (VariableScope.GLOBAL, VariableScope.BOTH):
                value = self.get_value()
                if value and value != "":
                    if isinstance(self, MultiValueVariable):
                        value = f'"{value}"'
                    ret.append(f"global {self.name}={value}")
        return ret


@dataclass
class SingleValueVariable(MysqlVariable):
    """Single-value MySQL variable. Upstream: SingleValueVariable."""
    value: Optional[Union[str, int]] = None

    def set_value(self, val):
        _dbg("SingleVar_set", name=self.name, val=val)
        if val is None or val == "":
            return
        self.is_update = True
        self.value = val

    def get_value(self, key=None):
        _dbg("SingleVar_get", name=self.name)
        if not self.is_update:
            return ""
        return self.value


@dataclass
class MultiValueVariable(MysqlVariable):
    """Multi-value MySQL variable (comma-separated k=v pairs).
    Upstream: MultiValueVariable."""
    fields_data: Dict[str, str] = field(default_factory=dict)

    def set_value(self, val):
        _dbg("MultiVar_set", name=self.name, val=val)
        if val is None or val == "":
            return
        self.is_update = True
        for item in val.split(","):
            parts = item.split("=", 1)
            if len(parts) == 2:
                self.fields_data[parts[0]] = parts[1]

    def get_value(self, key: Optional[str] = None):
        _dbg("MultiVar_get", name=self.name, key=key)
        if not self.is_update:
            return ""
        if key is None:
            return ",".join(f"{k}={v}" for k, v in self.fields_data.items())
        return self.fields_data.get(key, "")


DEFAULT_INNODB_PAGE_SIZE = 16384


@dataclass
class VariablesAboutIndex:
    """Collection of MySQL variables relevant to index optimization.

    Upstream: VariablesAboutIndex(BaseModel) with ~15 variable fields.
    Modified: each variable carries vector clock stamps via
    stamp_all_versions().
    """
    optimizer_switch: MultiValueVariable = field(default_factory=lambda:
        MultiValueVariable(name="optimizer_switch", scope=VariableScope.BOTH,
                           version=["mysql5.7", "mysql8.0", "mariadb11.8"],
                           dynamic=True, need_set=True))
    sort_buffer_size: SingleValueVariable = field(default_factory=lambda:
        SingleValueVariable(name="sort_buffer_size", scope=VariableScope.BOTH,
                            version=["mysql5.7", "mysql8.0", "mariadb11.8"],
                            dynamic=True, need_set=False))
    join_buffer_size: SingleValueVariable = field(default_factory=lambda:
        SingleValueVariable(name="join_buffer_size", scope=VariableScope.BOTH,
                            version=["mysql5.7", "mysql8.0", "mariadb11.8"],
                            dynamic=True, need_set=True))
    tmp_table_size: SingleValueVariable = field(default_factory=lambda:
        SingleValueVariable(name="tmp_table_size", scope=VariableScope.BOTH,
                            version=["mysql5.7", "mysql8.0", "mariadb11.8"],
                            dynamic=True, need_set=True))
    max_heap_table_size: SingleValueVariable = field(default_factory=lambda:
        SingleValueVariable(name="max_heap_table_size",
                            scope=VariableScope.BOTH,
                            version=["mysql5.7", "mysql8.0", "mariadb11.8"],
                            dynamic=True, need_set=True))
    innodb_large_prefix: SingleValueVariable = field(default_factory=lambda:
        SingleValueVariable(name="innodb_large_prefix",
                            scope=VariableScope.GLOBAL,
                            version=["mysql5.7"],
                            dynamic=True, need_set=True))
    max_seeks_for_key: SingleValueVariable = field(default_factory=lambda:
        SingleValueVariable(name="max_seeks_for_key",
                            scope=VariableScope.BOTH,
                            version=["mysql5.7", "mysql8.0", "mariadb11.8"],
                            dynamic=True, need_set=True))
    eq_range_index_dive_limit: SingleValueVariable = field(default_factory=lambda:
        SingleValueVariable(name="eq_range_index_dive_limit",
                            scope=VariableScope.BOTH,
                            version=["mysql5.7", "mysql8.0", "mariadb11.8"],
                            dynamic=True, need_set=True))
    optimizer_prune_level: SingleValueVariable = field(default_factory=lambda:
        SingleValueVariable(name="optimizer_prune_level",
                            scope=VariableScope.BOTH,
                            version=["mysql5.7", "mysql8.0", "mariadb11.8"],
                            dynamic=True, need_set=True))
    optimizer_search_depth: SingleValueVariable = field(default_factory=lambda:
        SingleValueVariable(name="optimizer_search_depth",
                            scope=VariableScope.BOTH,
                            version=["mysql5.7", "mysql8.0", "mariadb11.8"],
                            dynamic=True, need_set=True))
    range_optimizer_max_mem_size: SingleValueVariable = field(default_factory=lambda:
        SingleValueVariable(name="range_optimizer_max_mem_size",
                            scope=VariableScope.BOTH,
                            version=["mysql5.7", "mysql8.0"],
                            dynamic=True, need_set=True))
    version: SingleValueVariable = field(default_factory=lambda:
        SingleValueVariable(name="version", scope=VariableScope.BOTH,
                            version=["mysql5.7", "mysql8.0", "mariadb11.8"],
                            dynamic=False, read_only=True, need_set=False))
    innodb_page_size: SingleValueVariable = field(default_factory=lambda:
        SingleValueVariable(name="innodb_page_size",
                            scope=VariableScope.GLOBAL,
                            version=["mysql5.7", "mysql8.0", "mariadb11.8"],
                            dynamic=False, read_only=True, need_set=False))
    innodb_buffer_pool_size: SingleValueVariable = field(default_factory=lambda:
        SingleValueVariable(name="innodb_buffer_pool_size",
                            scope=VariableScope.GLOBAL,
                            version=["mysql5.7", "mysql8.0", "mariadb11.8"],
                            dynamic=False, read_only=True, need_set=False))
    myisam_max_sort_file_size: SingleValueVariable = field(default_factory=lambda:
        SingleValueVariable(name="myisam_max_sort_file_size",
                            scope=VariableScope.GLOBAL,
                            version=["mysql5.7", "mysql8.0", "mariadb11.8"],
                            dynamic=False, read_only=True, need_set=False))

    def get_all_attributes(self):
        _dbg("VariablesAboutIndex_get_all")
        return iter(self.__dict__.items())

    def stamp_all_versions(self, vc: VectorClock):
        """Stamp all variable fields with the given vector clock.
        Algorithm addition for distributed variable tracking."""
        _dbg("stamp_all_versions", vc=vc.snapshot)
        for attr_name, attr_val in self.__dict__.items():
            if isinstance(attr_val, MysqlVariable):
                attr_val.stamp_version(vc)


# ═══════════════════════════════════════════════════════════════════
#  mapping_index_columns (from common_operation.py:26)
# ═══════════════════════════════════════════════════════════════════

def mapping_index_columns(table: Table):
    """Link index columns to their Column objects.
    Upstream: mapping_index_columns(table)."""
    _dbg("mapping_index_columns", table=table.name,
         n_cols=len(table.columns), n_idx=len(table.indexes))
    column_dict = {col.name: col for col in table.columns}
    for index in table.indexes:
        for ic in index.columns:
            col_name = ic.name
            if not col_name:
                if ic.expression:
                    col_name = _parse_column_from_expression(
                        ic.expression.replace("\\'", "'")
                    )
                    ic.name = col_name
                else:
                    raise ValueError(
                        f"Table [{table.name}] index [{index.name}] "
                        f"has empty column name"
                    )
            if col_name in column_dict:
                ic.column_ref = column_dict[col_name]
            else:
                _dbg("mapping_col_miss", idx=index.name, col=col_name)


def _parse_column_from_expression(expression: str) -> Optional[str]:
    """Extract column name from SQL expression.
    Upstream: parse_from_expression using sqlglot."""
    _dbg("parse_col_from_expr", expr=expression[:60])
    # Lightweight parser without sqlglot dependency
    # Looks for backtick-wrapped identifiers or bare identifiers
    match = re.search(r'`(\w+)`', expression)
    if match:
        return match.group(1)
    # Try simple column reference pattern
    match = re.search(r'(?:^|\.)(\w+)(?:\s|$|->)', expression)
    if match:
        return match.group(1)
    return None


# ═══════════════════════════════════════════════════════════════════
#  patch_index_invisible (from common_operation.py:44)
# ═══════════════════════════════════════════════════════════════════

def patch_index_invisible(table: Table):
    """Update index visibility from DDL.
    Upstream: patch_index_invisible."""
    _dbg("patch_index_invisible", table=table.name)
    if not table.ddl:
        return
    # Reversed regex pattern to find invisible indexes
    reg_pattern = r'ELBISIVNI 00008.*?\(\s+`(.*?)`\s+YEK'
    re_match = re.findall(reg_pattern, table.ddl[::-1])
    invisible_indexes = [x[::-1] for x in re_match]
    if not invisible_indexes:
        return
    for index in table.indexes:
        if index.name in invisible_indexes:
            _dbg("index_invisible", db=index.db, table=index.table,
                 name=index.name)
            index.is_visible = False


# ═══════════════════════════════════════════════════════════════════
#  Data type correction (from common_operation.py:60-123)
# ═══════════════════════════════════════════════════════════════════

def replace_illegal_value(data, expected_pd_type: str):
    """Replace illegal date values with epoch defaults.
    Upstream: replace_illegal_value."""
    _dbg("replace_illegal_value", dtype=expected_pd_type)
    if expected_pd_type == 'date':
        data = data.replace('0000-00-00', '1970-01-01')
    elif expected_pd_type == 'datetime':
        data = data.replace('0000-00-00 00:00:00', '1970-01-01 00:00:00')
    return data


def correct_df_type_by_mysql_type(df_sample_raw, table_meta: Table):
    """Correct DataFrame column types based on MySQL metadata.
    Upstream: correct_df_type_by_mysql_type."""
    _dbg("correct_df_type", table=table_meta.name,
         n_cols=len(table_meta.columns))
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        _dbg("correct_df_type_skip", reason="pandas/numpy not available")
        return df_sample_raw

    col_meta_dict = {col.name.lower(): col for col in table_meta.columns
                     if col.name}
    for col in df_sample_raw.columns:
        col_meta = col_meta_dict.get(col.lower())
        if col_meta is None:
            _dbg("correct_df_col_skip", col=col, reason="not in meta")
            continue
        exist_dtype = str(df_sample_raw[col].dtype)
        expected_dtype = mysql_to_pandas_type(col_meta.column_type or col_meta.data_type or "")
        if exist_dtype == expected_dtype:
            continue
        try:
            df_sample_raw[col] = replace_illegal_value(
                df_sample_raw[col],
                (col_meta.data_type or "").lower(),
            )
            if (col_meta.data_type or "").lower() in ('datetime', 'timestamp'):
                df_sample_raw[col] = pd.to_datetime(
                    df_sample_raw[col], format='mixed'
                )
            else:
                df_sample_raw[col] = df_sample_raw[col].astype(expected_dtype)
        except Exception as e:
            _dbg("correct_df_type_err", col=col, err=str(e))
            if ('int' in expected_dtype
                    and (df_sample_raw[col].hasnans
                         or np.inf in df_sample_raw[col].values
                         or -np.inf in df_sample_raw[col].values)):
                try:
                    nullable = expected_dtype.replace('uint', 'UInt').replace('int', 'Int')
                    df_sample_raw[col] = df_sample_raw[col].astype(nullable)
                except Exception as e2:
                    _dbg("correct_df_fallback_err", col=col, err=str(e2))
    return df_sample_raw


def parse_sample_data_to_dataframe(data: List[Dict[str, str]],
                                   table_meta: Table):
    """Convert sampled row dicts into a DataFrame with correct types.
    Upstream: parse_sample_data_to_dataframe."""
    _dbg("parse_sample_to_df", n_rows=len(data) if data else 0)
    if data is None:
        try:
            import pandas as pd
            return pd.DataFrame({})
        except ImportError:
            return {}
    df_dict: Dict[str, list] = {}
    for row in data:
        for col, val in row.items():
            if col not in df_dict:
                df_dict[col] = []
            df_dict[col].append(val)
    try:
        import pandas as pd
        ret = pd.DataFrame(df_dict)
        return correct_df_type_by_mysql_type(ret, table_meta)
    except ImportError:
        _dbg("parse_sample_no_pandas")
        return df_dict
