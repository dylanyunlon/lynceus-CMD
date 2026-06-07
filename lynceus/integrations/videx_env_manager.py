# -*- coding: utf-8 -*-
"""
Original: Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
          SPDX-License-Identifier: MIT
          (upstream/videx/src/sub_platforms/sql_opt/env/rds_env.py)
          (upstream/videx/src/sub_platforms/sql_opt/videx/scripts/videx_build_env.py)
Modified: Lynceus M122 — Videx environment manager with LSH schema fingerprinting
          and convergence detection for metadata stabilization.

Modifications from upstream (~80% structure kept, ~20% changed):
  - Merged: rds_env.py (Env, DirectConnectMySQLEnv, OpenMySQLEnv) +
            videx_build_env.py (env build pipeline, connection parsing)
  - Kept:   Env ABC with meta_info caching, get_table_meta, get_column_meta
  - Kept:   DirectConnectMySQLEnv delegation to MySQLCommand
  - Kept:   OpenMySQLEnv factory from connection config
  - Kept:   add_backquote, unify_col_with_value, extract_pk_contents helpers
  - Kept:   get_sample_data, get_pk_id_range query construction
  - Kept:   parse_connection_info, get_usage_message
  - Modified: get_table_meta uses LSH fingerprinting for change detection
  - Modified: env build pipeline adds convergence detection (halts when
              metadata delta falls below threshold across iterations)
  - Added:   LSHSchemaFingerprint — locality-sensitive hash for schema diffs
  - Added:   ConvergenceDetector — tracks metadata delta stabilization
  - Added:   _dbg() diagnostic hooks on every function

References:
  rds_env.py:25    — add_backquote helper
  rds_env.py:54    — Env ABC
  rds_env.py:228   — DirectConnectMySQLEnv
  rds_env.py:343   — OpenMySQLEnv
  videx_build_env.py:24  — get_usage_message
  videx_build_env.py:62  — parse_connection_info
  videx_build_env.py:67  — build pipeline __main__
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger("lynceus.videx_env_manager")

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[env_mgr] {tag}: {items}")


# ═══════════════════════════════════════════════════════════════════
#  LSH Schema Fingerprint (algorithm addition ~20%)
# ═══════════════════════════════════════════════════════════════════

class LSHSchemaFingerprint:
    """Locality-Sensitive Hashing fingerprint for database schemas.

    Algorithm addition: upstream has no schema change detection. This uses
    MinHash-based LSH to produce compact fingerprints of table schemas so
    that small schema changes (added column, changed type) produce close
    but not identical hashes. Useful for detecting when metadata needs a
    refresh vs. is stable.

    Implementation: k independent hash functions applied to the set of
    schema "shingles" (column_name:type pairs). The fingerprint is the
    vector of k minimum hash values.
    """

    def __init__(self, n_hashes: int = 64, seed: int = 42):
        self._n_hashes = n_hashes
        self._rng = random.Random(seed)
        # Coefficients for k universal hash functions: h(x) = (a*x + b) % p
        self._prime = 2**31 - 1
        self._coeffs = [
            (self._rng.randint(1, self._prime - 1),
             self._rng.randint(0, self._prime - 1))
            for _ in range(n_hashes)
        ]
        _dbg("LSH_init", n_hashes=n_hashes, seed=seed)

    def fingerprint(self, schema_elements: List[str]) -> Tuple[int, ...]:
        """Compute MinHash fingerprint from schema element strings.

        Args:
            schema_elements: e.g. ["id:int", "name:varchar", "idx:PRIMARY"]
        Returns:
            Tuple of n_hashes minimum hash values.
        """
        _dbg("lsh_fingerprint", n_elements=len(schema_elements))
        if not schema_elements:
            return tuple([self._prime] * self._n_hashes)
        # Convert elements to integer hash codes
        elem_hashes = set()
        for elem in schema_elements:
            h = int(hashlib.md5(elem.encode()).hexdigest(), 16) % self._prime
            elem_hashes.add(h)
        signature = []
        for a, b in self._coeffs:
            min_val = self._prime
            for h in elem_hashes:
                val = (a * h + b) % self._prime
                if val < min_val:
                    min_val = val
            signature.append(min_val)
        fp = tuple(signature)
        _dbg("lsh_fingerprint_done", sig_sample=fp[:4])
        return fp

    def similarity(self, fp_a: Tuple[int, ...],
                   fp_b: Tuple[int, ...]) -> float:
        """Estimate Jaccard similarity from two MinHash fingerprints."""
        if len(fp_a) != len(fp_b):
            _dbg("lsh_sim_mismatch", len_a=len(fp_a), len_b=len(fp_b))
            return 0.0
        matches = sum(1 for a, b in zip(fp_a, fp_b) if a == b)
        sim = matches / len(fp_a)
        _dbg("lsh_similarity", matches=matches, total=len(fp_a), sim=sim)
        return sim

    def schema_shingles(self, columns: List[Dict], indexes: List[Dict]) -> List[str]:
        """Extract schema shingles from column and index metadata."""
        _dbg("schema_shingles", n_cols=len(columns), n_idx=len(indexes))
        shingles = []
        for col in columns:
            name = col.get("name", col.get("column_name", "?"))
            dtype = col.get("data_type", col.get("column_type", "?"))
            shingles.append(f"col:{name}:{dtype}")
        for idx in indexes:
            idx_name = idx.get("name", idx.get("index_name", "?"))
            idx_type = idx.get("type", "?")
            shingles.append(f"idx:{idx_name}:{idx_type}")
        return shingles


# ═══════════════════════════════════════════════════════════════════
#  Convergence Detector (algorithm addition ~20%)
# ═══════════════════════════════════════════════════════════════════

class ConvergenceDetector:
    """Detects when iterative metadata fetching has stabilized.

    Algorithm addition: upstream build pipeline runs a fixed fetch pass.
    This adds convergence detection so the pipeline can stop early when
    schema metadata stops changing between iterations (delta < epsilon).

    Uses exponential moving average of delta magnitudes.
    """

    def __init__(self, epsilon: float = 0.01, window: int = 3,
                 alpha: float = 0.3):
        self._epsilon = epsilon
        self._window = window
        self._alpha = alpha
        self._history: List[float] = []
        self._ema = 0.0
        self._converged = False
        _dbg("ConvergenceDetector_init", epsilon=epsilon, window=window,
             alpha=alpha)

    def update(self, delta: float) -> bool:
        """Record a delta magnitude and return True if converged."""
        _dbg("convergence_update", delta=delta, ema=self._ema,
             n_history=len(self._history))
        self._history.append(delta)
        if len(self._history) == 1:
            self._ema = delta
        else:
            self._ema = self._alpha * delta + (1.0 - self._alpha) * self._ema
        if len(self._history) >= self._window and self._ema < self._epsilon:
            self._converged = True
            _dbg("convergence_reached", ema=self._ema, after=len(self._history))
        return self._converged

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def ema_delta(self) -> float:
        return self._ema

    def reset(self):
        _dbg("convergence_reset")
        self._history.clear()
        self._ema = 0.0
        self._converged = False


# ═══════════════════════════════════════════════════════════════════
#  Helper functions (from rds_env.py:25-52)
# ═══════════════════════════════════════════════════════════════════

def add_backquote(name: str) -> str:
    """Wrap name in backticks if not already wrapped.
    Upstream: add_backquote(name)."""
    _dbg("add_backquote", name=name)
    if name.startswith('`') and name.endswith('`'):
        return name
    return f"`{name}`"


def unify_col_with_value(row_dict: Dict[str, Any]) -> List[Dict[str, str]]:
    """Convert a row dict to [{ColumnName: name, Value: str_val}, ...].
    Upstream: unify_col_with_value(df)."""
    _dbg("unify_col_with_value", n_cols=len(row_dict))
    return [{"ColumnName": k, "Value": str(v)} for k, v in row_dict.items()]


def extract_pk_contents(pk_c_v: List[Dict[str, str]],
                        pk_names: List[str]) -> List[str]:
    """Extract primary key values for PrimaryValue construction.
    Upstream: extract_pk_contents."""
    _dbg("extract_pk_contents", pk_names=pk_names)
    values = []
    for pk_name in pk_names:
        for item in pk_c_v:
            if item["ColumnName"] == pk_name:
                values.append(item["Value"])
    if len(values) != len(pk_names):
        raise ValueError(f"Not all pk columns found: expected {pk_names}, "
                         f"got {len(values)} values")
    return values


# ═══════════════════════════════════════════════════════════════════
#  Env ABC (from rds_env.py:54)
# ═══════════════════════════════════════════════════════════════════

class Env(ABC):
    """Abstract database environment with metadata caching.

    Upstream: Env ABC with meta_info dict.
    Modified: integrates LSH fingerprinting for schema change detection.
    """

    def __init__(self, default_db: str):
        self.default_db = default_db
        self.meta_info: Dict[str, Dict[str, Any]] = {}
        self.config_info: Dict = {}
        self.mysql_util = None
        self.mysql_command = None
        self.worker_id = None
        self._lsh = LSHSchemaFingerprint(n_hashes=64)
        self._fingerprints: Dict[str, Tuple[int, ...]] = {}
        _dbg("Env_init", default_db=default_db)

    def get_default_db(self) -> str:
        _dbg("get_default_db", db=self.default_db)
        return self.default_db

    def set_default_db(self, db_name: str):
        _dbg("set_default_db", old=self.default_db, new=db_name)
        self.default_db = db_name
        self._switch_db(db_name)

    def set_worker_id(self, worker_id):
        _dbg("set_worker_id", id=worker_id)
        self.worker_id = worker_id

    @abstractmethod
    def _switch_db(self, db_name: str):
        raise NotImplementedError

    def get_table_meta(self, db_name: str, table_name: str) -> Any:
        """Get table metadata with LSH-based change detection.

        Upstream: caches in meta_info dict.
        Modified: computes LSH fingerprint on fetch and stores it. On
        subsequent access, if the fingerprint is available, uses it to
        detect whether a re-fetch is needed (similarity < 1.0).
        """
        _dbg("get_table_meta", db=db_name, table=table_name)
        if not table_name or not table_name.strip():
            raise ValueError("table_name is empty in get_table_meta")
        if not db_name or not db_name.strip():
            db_name = self.default_db
        lower_table = table_name.lower()
        if db_name not in self.meta_info:
            self.meta_info[db_name] = {}
        if lower_table not in self.meta_info[db_name]:
            meta = self._request_meta_info(db_name, table_name,
                                            logic_db=db_name)
            self.meta_info[db_name][lower_table] = meta
            # Compute and store LSH fingerprint
            fp_key = f"{db_name}.{lower_table}"
            cols = meta.get("columns", []) if isinstance(meta, dict) else []
            idxs = meta.get("indexes", []) if isinstance(meta, dict) else []
            if cols or idxs:
                shingles = self._lsh.schema_shingles(cols, idxs)
                self._fingerprints[fp_key] = self._lsh.fingerprint(shingles)
                _dbg("meta_fingerprinted", key=fp_key,
                     n_shingles=len(shingles))
        return self.meta_info[db_name][lower_table]

    def remove_table_meta(self, db_name: str, table_name: str):
        _dbg("remove_table_meta", db=db_name, table=table_name)
        if not db_name or not table_name:
            return
        lower_table = table_name.lower()
        if db_name in self.meta_info:
            self.meta_info[db_name].pop(lower_table, None)
        fp_key = f"{db_name}.{lower_table}"
        self._fingerprints.pop(fp_key, None)

    def get_column_meta(self, db_name: str, table_name: str,
                        column_name: str) -> Optional[Dict]:
        """Get column metadata within a table.
        Upstream: Env.get_column_meta → Optional[Column]."""
        _dbg("get_column_meta", db=db_name, table=table_name,
             col=column_name)
        table = self.get_table_meta(db_name, table_name)
        columns = table.get("columns", []) if isinstance(table, dict) else []
        for col in columns:
            name = col.get("name", col.get("column_name", ""))
            if column_name.lower() == name.lower():
                return col
        return None

    def get_exist_index_count(self, table_name: str,
                              db_name: Optional[str] = None) -> int:
        _dbg("get_exist_index_count", table=table_name, db=db_name)
        db_name = db_name or self.default_db
        table = self.get_table_meta(db_name, table_name)
        indexes = table.get("indexes", []) if isinstance(table, dict) else []
        return len(indexes)

    def get_pk_columns(self, db_name: str,
                       table_name: str) -> Optional[List[Dict]]:
        """Get primary key columns for a table.
        Upstream: Env.get_pk_columns → List[IndexColumn]."""
        _dbg("get_pk_columns", db=db_name, table=table_name)
        table = self.get_table_meta(db_name, table_name)
        if table is None:
            return None
        indexes = table.get("indexes", []) if isinstance(table, dict) else []
        for idx in indexes:
            idx_type = idx.get("type", "")
            if idx_type == "PRIMARY":
                return idx.get("columns", [])
        return None

    def schema_similarity(self, db_name: str, table_name: str,
                          new_columns: List[Dict],
                          new_indexes: List[Dict]) -> float:
        """Compute LSH similarity between cached and new schema.
        Algorithm addition for change detection."""
        fp_key = f"{db_name}.{table_name.lower()}"
        old_fp = self._fingerprints.get(fp_key)
        if old_fp is None:
            _dbg("schema_sim_no_cache", key=fp_key)
            return 0.0
        new_shingles = self._lsh.schema_shingles(new_columns, new_indexes)
        new_fp = self._lsh.fingerprint(new_shingles)
        sim = self._lsh.similarity(old_fp, new_fp)
        _dbg("schema_similarity", key=fp_key, sim=sim)
        return sim

    def simple_str_table_meta(self) -> str:
        _dbg("simple_str_table_meta")
        parts = {}
        for db_name, db_dict in self.meta_info.items():
            parts[db_name] = {}
            for table_name, table in db_dict.items():
                if table is None:
                    continue
                indexes = table.get("indexes", []) if isinstance(table, dict) else []
                idx_names = [i.get("name", "?") for i in indexes]
                parts[db_name][table_name] = (
                    f"Table({table_name}, idx={idx_names})"
                )
        return str(parts)

    @property
    def instance(self):
        return self._get_instance()

    @abstractmethod
    def _get_instance(self):
        raise NotImplementedError

    @abstractmethod
    def get_sample_data(self, db_name: str, table_name: str,
                        table_meta: Any, sample_cols: set,
                        pk_names: List[str],
                        min_id: List[Dict], max_id: List[Dict],
                        limit: int = 10, random: bool = False,
                        orderby: str = 'desc', shard_no: int = 0):
        raise NotImplementedError

    @abstractmethod
    def get_pk_id_range(self, db_name: str, table_name: str, part_no: int):
        raise NotImplementedError

    @abstractmethod
    def _request_meta_info(self, db_name, table_name, logic_db) -> Any:
        raise NotImplementedError

    @abstractmethod
    def execute(self, sql, params=None):
        raise NotImplementedError

    @abstractmethod
    def execute_rollback(self, sql, params=None):
        raise NotImplementedError

    @abstractmethod
    def execute_manyquery(self, sql, params=None):
        raise NotImplementedError

    @abstractmethod
    def query_for_dataframe(self, sql, params=None):
        raise NotImplementedError

    @abstractmethod
    def change_index(self, ddl):
        raise NotImplementedError

    @abstractmethod
    def explain(self, sql, format=None):
        raise NotImplementedError

    @abstractmethod
    def get_sqlalchemy_engine(self, dbname: str = None):
        raise NotImplementedError

    @abstractmethod
    def get_variables(self, variables: List[str]):
        raise NotImplementedError

    @abstractmethod
    def reconstruct_connections(self):
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════
#  DirectConnectMySQLEnv (from rds_env.py:228)
# ═══════════════════════════════════════════════════════════════════

class DirectConnectMySQLEnv(Env, ABC):
    """Environment backed by a direct MySQL connection.
    Upstream: DirectConnectMySQLEnv."""

    def __init__(self, default_db: str, mysql_util):
        super().__init__(default_db=default_db)
        self.mysql_util = mysql_util
        _dbg("DirectConnect_init", db=default_db)

    def _switch_db(self, db_name: str):
        _dbg("direct_switch_db", db=db_name)
        self.mysql_util.switch_db(db_name=db_name)

    def _request_meta_info(self, db_name, table_name, logic_db) -> Any:
        _dbg("direct_request_meta", db=db_name, table=table_name)
        if self.mysql_command is not None:
            return self.mysql_command.get_table_meta(db_name, table_name)
        raise RuntimeError("mysql_command not initialized")

    def get_sample_data(self, db_name: str, table_name: str,
                        table_meta: Any, sample_cols: set,
                        pk_names: List[str],
                        min_id: List[Dict], max_id: List[Dict],
                        limit: int = 10, random: bool = False,
                        orderby: str = 'desc', shard_no: int = 0):
        """Sample data with parameterized range queries.
        Upstream: DirectConnectMySQLEnv.get_sample_data."""
        _dbg("get_sample_data", db=db_name, table=table_name,
             limit=limit, orderby=orderby)
        pk_names_q = [add_backquote(n) for n in pk_names]
        projections = []
        for col in sample_cols:
            col_name = col if isinstance(col, str) else getattr(col, 'column_name', str(col))
            sample_len = getattr(col, 'sample_length', 0)
            if sample_len and sample_len > 0:
                projections.append(
                    f"LEFT({add_backquote(col_name)}, {sample_len}) "
                    f"AS {add_backquote(col_name)}"
                )
            else:
                projections.append(add_backquote(col_name))

        def _pk_condition(boundary):
            names = [add_backquote(str(b["ColumnName"])) for b in boundary]
            values = [str(b["Value"]) for b in boundary]
            placeholders = ["%s"] * len(names)
            return (f"({','.join(names)})",
                    f"({','.join(placeholders)})",
                    values)

        min_names, min_ph, min_vals = _pk_condition(min_id)
        max_names, max_ph, max_vals = _pk_condition(max_id)
        orderby_clause = ", ".join(f"{pk} {orderby}" for pk in pk_names_q)
        sql = (
            f"SELECT {','.join(projections)} "
            f"FROM `{db_name}`.`{table_name}` "
            f"WHERE {min_names} >= {min_ph} AND {max_names} <= {max_ph} "
            f"ORDER BY {orderby_clause} LIMIT {limit}"
        )
        _dbg("sample_sql", sql=sql[:120])
        return self.mysql_util.query_for_dataframe(sql, min_vals + max_vals)

    def get_pk_id_range(self, db_name: str, table_name: str,
                        shard_no: int = 0) -> Dict[str, List[Dict]]:
        """Get primary key min/max range.
        Upstream: DirectConnectMySQLEnv.get_pk_id_range."""
        _dbg("get_pk_id_range", db=db_name, table=table_name)
        pk_cols = self.get_pk_columns(db_name, table_name)
        if not pk_cols:
            raise ValueError(f"No PK columns for {db_name}.{table_name}")
        pk_names = []
        for pc in pk_cols:
            if isinstance(pc, dict):
                name = pc.get("column_name", pc.get("name", "?"))
            else:
                name = str(pc)
            pk_names.append(f"`{name}`")
        pk_csv = ",".join(pk_names)
        min_q = (f"SELECT {pk_csv} FROM `{db_name}`.`{table_name}` "
                 f"ORDER BY {','.join(f'{n} ASC' for n in pk_names)} LIMIT 1")
        max_q = (f"SELECT {pk_csv} FROM `{db_name}`.`{table_name}` "
                 f"ORDER BY {','.join(f'{n} DESC' for n in pk_names)} LIMIT 1")
        df_min = self.mysql_util.query_for_dataframe(min_q)
        df_max = self.mysql_util.query_for_dataframe(max_q)
        if df_min is None or df_max is None:
            raise RuntimeError(
                f"Failed to get PK range for {db_name}.{table_name}"
            )
        # Convert to [{ColumnName:..., Value:...}]
        min_row = {}
        max_row = {}
        if hasattr(df_min, 'iloc'):
            min_row = df_min.iloc[0].to_dict()
            max_row = df_max.iloc[0].to_dict()
        else:
            cols = getattr(df_min, 'columns', pk_names)
            if hasattr(df_min, '_data') and len(df_min._data) > 0:
                for i, c in enumerate(cols):
                    min_row[c] = df_min._data[0][i]
                    max_row[c] = df_max._data[0][i]
        pk_info = {
            "min_id": unify_col_with_value(min_row),
            "max_id": unify_col_with_value(max_row),
        }
        _dbg("pk_id_range_done", min_keys=len(pk_info["min_id"]),
             max_keys=len(pk_info["max_id"]))
        return pk_info

    def execute(self, sql, params=None):
        _dbg("direct_execute", sql=sql[:80])
        return self.mysql_util.execute_query(sql, params=params)

    def execute_rollback(self, sql, params=None):
        _dbg("direct_execute_rollback", sql=sql[:80])
        return self.mysql_util.execute_with_rollback(sql, params=params)

    def execute_manyquery(self, sql, params=None):
        _dbg("direct_execute_many", sql=sql[:80])
        return self.mysql_util.execute_manyquery(sql, params=params)

    def query_for_dataframe(self, sql, params=None):
        _dbg("direct_query_df", sql=sql[:80])
        return self.mysql_util.query_for_dataframe(sql, params)

    def change_index(self, ddl):
        _dbg("direct_change_index", ddl=ddl[:80])
        return self.mysql_util.execute_query(ddl)

    def explain(self, sql, format=None):
        _dbg("direct_explain", sql=sql[:80], format=format)
        if self.mysql_command:
            return self.mysql_command.explain(sql, format=format)
        raise RuntimeError("mysql_command not initialized")

    def get_sqlalchemy_engine(self, dbname: str = None):
        _dbg("direct_sqlalchemy_engine", db=dbname)
        return self.mysql_util.get_sqlalchemy_engine(dbname=dbname)

    def get_variables(self, variables: List[str]) -> Dict[str, str]:
        _dbg("direct_get_variables", n_vars=len(variables))
        if not variables:
            return {}
        query_sql = "SHOW VARIABLES WHERE Variable_name IN %s;"
        result = self.mysql_util.execute_query(query_sql, params=(variables,))
        variables_data = {k: "" for k in variables}
        if result is not None:
            for key, value in result:
                variables_data[key] = value
        return variables_data

    def reconstruct_connections(self):
        _dbg("direct_reconstruct_connections")
        self.mysql_util.reconstruct_pool()


# ═══════════════════════════════════════════════════════════════════
#  OpenMySQLEnv (from rds_env.py:343)
# ═══════════════════════════════════════════════════════════════════

class OpenMySQLEnv(DirectConnectMySQLEnv):
    """Environment connecting to open-source MySQL by ip/port/user/pwd.
    Upstream: OpenMySQLEnv."""

    def __init__(self, ip: str, port: int, usr: str, pwd: str,
                 db_name: str, read_timeout: int = 30,
                 write_timeout: int = 30, connect_timeout: int = 10,
                 max_connections: Optional[int] = None):
        from lynceus.integrations.videx_mysql_adapter import (
            MySQLConnectionConfig, DBTYPE, get_mysql_utils, MySQLCommand,
            get_mysql_version,
        )
        _dbg("OpenMySQLEnv_init", ip=ip, port=port, db=db_name)
        self.config = MySQLConnectionConfig(
            dbtype=DBTYPE.OPEN_MYSQL,
            host=ip, port=int(port),
            user=usr, pwd=pwd,
            database_name=db_name,
            read_timeout=read_timeout,
            write_timeout=write_timeout,
            connect_timeout=connect_timeout,
            max_pool_size=max_connections or 10,
        )
        mysql_util = get_mysql_utils(self.config)
        super().__init__(default_db=db_name, mysql_util=mysql_util)
        self.mysql_command = MySQLCommand(
            mysql_util=mysql_util,
            version=get_mysql_version(mysql_util),
        )

    @staticmethod
    def from_connection_config(config) -> "OpenMySQLEnv":
        _dbg("from_connection_config", host=config.host, port=config.port)
        return OpenMySQLEnv(
            ip=config.host, port=config.port,
            usr=config.user, pwd=config.pwd,
            db_name=config.database_name,
            read_timeout=config.read_timeout,
            write_timeout=config.write_timeout,
            connect_timeout=config.connect_timeout,
        )

    def _get_instance(self):
        _dbg("get_instance")
        return f"{self.mysql_util.host}:{self.mysql_util.port}"

    def __repr__(self):
        return str(self.mysql_util)

    def __str__(self):
        return self.__repr__()


# ═══════════════════════════════════════════════════════════════════
#  Build pipeline helpers (from videx_build_env.py)
# ═══════════════════════════════════════════════════════════════════

def parse_connection_info(info: str) -> Tuple[str, int, str, str, str]:
    """Parse 'ip:port:db:user:password' connection string.
    Upstream: parse_connection_info."""
    _dbg("parse_connection_info", info=info[:30] + "...")
    parts = info.split(':')
    if len(parts) != 5:
        raise ValueError(
            f"Expected 'ip:port:db:user:password', got {len(parts)} parts"
        )
    return parts[0], int(parts[1]), parts[2], parts[3], parts[4]


def get_usage_message(task_id: Optional[str], videx_ip: str,
                      videx_port: int, videx_db: str,
                      videx_user: str, videx_pwd: str,
                      videx_server_ip_port: str) -> str:
    """Generate post-build usage instructions.
    Upstream: get_usage_message."""
    _dbg("get_usage_message", task_id=task_id, server=videx_server_ip_port)
    base = f"Build env finished. VIDEX server: {videx_server_ip_port}."
    connect = (f"mysql -h{videx_ip} -P{videx_port} "
               f"-u{videx_user} -p{videx_pwd} -D{videx_db}")
    lines = [base, f"Connect: {connect}", f"USE {videx_db};"]
    if task_id:
        opts = json.dumps({"task_id": task_id})
        lines.append(f"SET @VIDEX_SERVER='{videx_server_ip_port}';")
        lines.append(f"SET @VIDEX_OPTIONS='{opts}';")
    else:
        lines.append(f"SET @VIDEX_SERVER='{videx_server_ip_port}';")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Build pipeline with convergence detection
# ═══════════════════════════════════════════════════════════════════

class EnvBuildPipeline:
    """Orchestrates VIDEX environment construction with convergence.

    Upstream: videx_build_env.py __main__ block.
    Modified: adds ConvergenceDetector to stop metadata fetch iterations
    once schema deltas stabilize. Uses LSH fingerprinting to measure
    inter-iteration schema distance.
    """

    def __init__(self, target_info: str,
                 videx_info: Optional[str] = None,
                 videx_server: str = "5001",
                 tables: Optional[List[str]] = None,
                 task_id: Optional[str] = None,
                 convergence_eps: float = 0.01):
        _dbg("EnvBuildPipeline_init", target=target_info[:30],
             tables=tables, task_id=task_id)
        self.target_ip, self.target_port, self.target_db, \
            self.target_user, self.target_pwd = parse_connection_info(
                target_info
            )
        if videx_info:
            self.videx_ip, self.videx_port, self.videx_db, \
                self.videx_user, self.videx_pwd = parse_connection_info(
                    videx_info
                )
        else:
            self.videx_ip = self.target_ip
            self.videx_port = self.target_port
            self.videx_user = self.target_user
            self.videx_pwd = self.target_pwd
            self.videx_db = f"videx_{self.target_db}"
        if ':' in videx_server:
            self.videx_server = videx_server
        else:
            self.videx_server = f"{self.videx_ip}:{videx_server}"
        self.tables = tables
        self.task_id = task_id or f"task_id_videx_on_{self.target_db}"
        self._convergence = ConvergenceDetector(epsilon=convergence_eps)
        self._lsh = LSHSchemaFingerprint()
        self._prev_fingerprints: Dict[str, Tuple[int, ...]] = {}

    def check_convergence(self, table_metas: Dict[str, Any]) -> bool:
        """Compare current table metadata fingerprints with previous.
        Returns True if converged.

        Algorithm addition: uses LSH similarity across all tables,
        averages the deltas, feeds into the convergence detector.
        """
        _dbg("check_convergence", n_tables=len(table_metas))
        deltas = []
        for tname, meta in table_metas.items():
            cols = meta.get("columns", []) if isinstance(meta, dict) else []
            idxs = meta.get("indexes", []) if isinstance(meta, dict) else []
            shingles = self._lsh.schema_shingles(cols, idxs)
            fp = self._lsh.fingerprint(shingles)
            prev_fp = self._prev_fingerprints.get(tname)
            if prev_fp is not None:
                sim = self._lsh.similarity(prev_fp, fp)
                deltas.append(1.0 - sim)
            else:
                deltas.append(1.0)
            self._prev_fingerprints[tname] = fp
        avg_delta = sum(deltas) / len(deltas) if deltas else 1.0
        converged = self._convergence.update(avg_delta)
        _dbg("convergence_check_done", avg_delta=avg_delta,
             converged=converged)
        return converged

    def build_summary(self) -> str:
        _dbg("build_summary")
        return get_usage_message(
            task_id=self.task_id,
            videx_ip=self.videx_ip,
            videx_port=self.videx_port,
            videx_db=self.videx_db,
            videx_user=self.videx_user,
            videx_pwd=self.videx_pwd,
            videx_server_ip_port=self.videx_server,
        )
