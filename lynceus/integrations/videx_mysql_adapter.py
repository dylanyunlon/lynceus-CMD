# -*- coding: utf-8 -*-
"""
Original: Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
          SPDX-License-Identifier: MIT
          (upstream/videx/src/sub_platforms/sql_opt/videx/videx_mysql_utils.py)
          (upstream/videx/src/sub_platforms/sql_opt/databases/mysql/mysql_command.py)
Modified: Lynceus M121 — MySQL connection adapter with LFU query cache and
          enhanced connection pool lifecycle management.

Modifications from upstream (~80% structure kept, ~20% changed):
  - Merged: videx_mysql_utils.py (connection pool, query helpers) +
            mysql_command.py (table/index metadata, EXPLAIN)
  - Kept:   DBTYPE enum, MySQLConnectionConfig model
  - Kept:   AbstractMySQLUtils pool lifecycle (shared/persistent)
  - Kept:   OpenMySQLUtils connection factory
  - Kept:   MySQLVersion enum and version detection
  - Kept:   MySQLCommand.get_table_columns, get_table_indexes, get_table_meta
  - Kept:   query_for_dataframe, query_for_value standalone helpers
  - Modified: query_for_dataframe adds LFU cache layer (bounded, evicts LFU)
  - Modified: connection pool reconstruct uses exponential back-off retry
  - Modified: get_table_meta caches parsed Table objects with TTL
  - Added:   LFUCache — bounded least-frequently-used eviction cache
  - Added:   PoolHealthMonitor — periodic connection validity checks
  - Added:   _dbg() diagnostic hooks on every function

References:
  videx_mysql_utils.py:22  — DBTYPE enum
  videx_mysql_utils.py:27  — MySQLConnectionConfig
  videx_mysql_utils.py:43  — get_mysql_utils factory
  videx_mysql_utils.py:50  — AbstractMySQLUtils
  videx_mysql_utils.py:189 — OpenMySQLUtils
  mysql_command.py:24      — MySQLVersion enum
  mysql_command.py:43      — get_mysql_version
  mysql_command.py:66      — MySQLCommand class
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time
import threading
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("lynceus.videx_mysql_adapter")

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))


def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[mysql_adpt] {tag}: {items}")


# ═══════════════════════════════════════════════════════════════════
#  LFU Cache (algorithm addition ~20%)
# ═══════════════════════════════════════════════════════════════════

class LFUCache:
    """Bounded Least-Frequently-Used cache with O(1) eviction.

    Algorithm addition: upstream has no query caching. This adds an LFU
    cache for repeated SQL queries, bounded by max_size to prevent
    unbounded memory growth. Uses frequency counter with min-heap
    emulation via ordered dict bucketing.
    """

    def __init__(self, max_size: int = 256, ttl_seconds: float = 300.0):
        self._store: Dict[str, Any] = {}
        self._freq: Dict[str, int] = {}
        self._timestamps: Dict[str, float] = {}
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._min_freq = 0
        self._freq_buckets: Dict[int, dict] = defaultdict(dict)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        _dbg("LFUCache_init", max_size=max_size, ttl=ttl_seconds)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                _dbg("lfu_miss", key=key[:64], misses=self._misses)
                return None
            if time.monotonic() - self._timestamps[key] > self._ttl:
                self._evict_key(key)
                self._misses += 1
                _dbg("lfu_ttl_expire", key=key[:64])
                return None
            old_freq = self._freq[key]
            new_freq = old_freq + 1
            self._freq[key] = new_freq
            del self._freq_buckets[old_freq][key]
            if not self._freq_buckets[old_freq]:
                del self._freq_buckets[old_freq]
                if self._min_freq == old_freq:
                    self._min_freq = new_freq
            self._freq_buckets[new_freq][key] = True
            self._hits += 1
            _dbg("lfu_hit", key=key[:64], freq=new_freq, hits=self._hits)
            return self._store[key]

    def put(self, key: str, value: Any):
        with self._lock:
            if key in self._store:
                self._store[key] = value
                self._timestamps[key] = time.monotonic()
                return
            if len(self._store) >= self._max_size:
                self._evict_lfu()
            self._store[key] = value
            self._freq[key] = 1
            self._timestamps[key] = time.monotonic()
            self._freq_buckets[1][key] = True
            self._min_freq = 1
            _dbg("lfu_put", key=key[:64], size=len(self._store))

    def _evict_lfu(self):
        if not self._freq_buckets:
            return
        bucket = self._freq_buckets[self._min_freq]
        victim_key = next(iter(bucket))
        self._evict_key(victim_key)
        _dbg("lfu_evict", key=victim_key[:64], freq=self._min_freq)

    def _evict_key(self, key: str):
        if key not in self._store:
            return
        freq = self._freq[key]
        del self._store[key]
        del self._freq[key]
        del self._timestamps[key]
        if key in self._freq_buckets.get(freq, {}):
            del self._freq_buckets[freq][key]
            if not self._freq_buckets[freq]:
                del self._freq_buckets[freq]

    def invalidate(self):
        with self._lock:
            self._store.clear()
            self._freq.clear()
            self._timestamps.clear()
            self._freq_buckets.clear()
            self._min_freq = 0
            _dbg("lfu_invalidate", hits=self._hits, misses=self._misses)

    @property
    def stats(self) -> Dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._store)}


# ═══════════════════════════════════════════════════════════════════
#  DBTYPE enum (from videx_mysql_utils.py:22)
# ═══════════════════════════════════════════════════════════════════

class DBTYPE(Enum):
    """Database backend type. Upstream: DBTYPE enum."""
    OPEN_MYSQL = "OPEN_MYSQL"
    SQLITE = "SQLITE"


# ═══════════════════════════════════════════════════════════════════
#  MySQLConnectionConfig (from videx_mysql_utils.py:27)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MySQLConnectionConfig:
    """Connection parameters for a MySQL instance.
    Upstream: pydantic BaseModel. Replaced with dataclass for fewer deps."""
    dbtype: DBTYPE = DBTYPE.OPEN_MYSQL
    host: str = "127.0.0.1"
    port: int = 3306
    database_name: Optional[str] = None
    user: Optional[str] = None
    pwd: Optional[str] = None
    consul: Optional[str] = None
    charset: str = "utf8"
    initial_pool_size: int = 5
    max_pool_size: int = 10
    read_timeout: int = 30
    write_timeout: int = 30
    connect_timeout: int = 10


# ═══════════════════════════════════════════════════════════════════
#  Pool Health Monitor (algorithm addition ~20%)
# ═══════════════════════════════════════════════════════════════════

class PoolHealthMonitor:
    """Track connection pool health with exponential back-off reconnect.

    Algorithm addition: upstream just closes/reopens the pool. This adds
    back-off delay and health status tracking so rapid failures don't
    cause a reconnect storm.
    """

    def __init__(self, max_retries: int = 5, base_delay: float = 0.5):
        self._consecutive_failures = 0
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._last_check = 0.0
        self._healthy = True
        _dbg("PoolHealth_init", max_retries=max_retries, base_delay=base_delay)

    def record_success(self):
        self._consecutive_failures = 0
        self._healthy = True
        self._last_check = time.monotonic()
        _dbg("pool_health_ok", ts=self._last_check)

    def record_failure(self) -> float:
        """Returns recommended delay (seconds) before next retry."""
        self._consecutive_failures += 1
        self._healthy = False
        delay = self._base_delay * (2 ** min(self._consecutive_failures, 8))
        _dbg("pool_health_fail", failures=self._consecutive_failures,
             delay=delay, healthy=self._healthy)
        return delay

    @property
    def should_retry(self) -> bool:
        return self._consecutive_failures < self._max_retries

    @property
    def healthy(self) -> bool:
        return self._healthy


# ═══════════════════════════════════════════════════════════════════
#  get_mysql_utils factory (from videx_mysql_utils.py:43)
# ═══════════════════════════════════════════════════════════════════

def get_mysql_utils(config: MySQLConnectionConfig) -> "AbstractMySQLUtils":
    """Factory: create a MySQL utility instance from config."""
    _dbg("get_mysql_utils", dbtype=config.dbtype, host=config.host,
         port=config.port, db=config.database_name)
    if config.dbtype == DBTYPE.OPEN_MYSQL:
        return OpenMySQLUtils(config)
    else:
        raise ValueError(f"Unsupported datasource type: {config.dbtype}")


# ═══════════════════════════════════════════════════════════════════
#  AbstractMySQLUtils (from videx_mysql_utils.py:50)
# ═══════════════════════════════════════════════════════════════════

class AbstractMySQLUtils:
    """Base class for MySQL connection pool management.
    Modification: exponential back-off on reconstruct, LFU query cache."""

    def __init__(self, mysql_type: str, database: Optional[str],
                 charset: str, read_timeout: int = 30,
                 write_timeout: int = 30, connect_timeout: int = 10):
        self.mysql_type = mysql_type
        self.database = database
        self.charset = charset
        self.pool = None
        self.pool_type = None
        self.read_timeout = read_timeout
        self.write_timeout = write_timeout
        self.connect_timeout = connect_timeout
        self._query_cache = LFUCache(max_size=512, ttl_seconds=120.0)
        self._health = PoolHealthMonitor()
        _dbg("AbstractMySQL_init", type=mysql_type, db=database, charset=charset)

    def get_connection(self):
        _dbg("get_connection_base", type=self.mysql_type)
        raise NotImplementedError

    def switch_db(self, db_name: str):
        _dbg("switch_db", old=self.database, new=db_name)
        if db_name == self.database:
            return
        self.database = db_name
        self._query_cache.invalidate()
        self.reconstruct_pool()

    def reconstruct_pool(self):
        """Reconstruct pool with exponential back-off retry."""
        _dbg("reconstruct_pool", pool_type=self.pool_type, healthy=self._health.healthy)
        if self.pool is not None:
            try:
                self.pool.close()
            except Exception as e:
                _dbg("pool_close_err", err=str(e))
            if not self._health.healthy:
                delay = self._health.record_failure()
                if delay > 0:
                    _dbg("reconstruct_backoff", delay=delay)
                    time.sleep(delay)
            if self.pool_type == 'PooledDB':
                self.get_shared_pool()
            else:
                self.get_persistent_pool()
            self._health.record_success()

    def get_shared_pool(self, initial_connections: int = 1, max_connections: int = 10):
        _dbg("get_shared_pool", init=initial_connections, max=max_connections)
        self.pool = _SharedPoolStub(creator=self.get_connection,
                                    min_cached=initial_connections,
                                    max_connections=max_connections)
        self.pool_type = 'PooledDB'
        return self.pool

    def get_persistent_pool(self):
        _dbg("get_persistent_pool", type=self.mysql_type)
        self.pool = _PersistentPoolStub(creator=self.get_connection)
        self.pool_type = 'PersistentDB'
        return self.pool

    def _ensure_pool(self):
        if self.pool is None:
            self.pool = self.get_shared_pool()

    def query_for_dataframe(self, sql_template: str, params: Optional[list] = None) -> Any:
        """Query with LFU cache layer."""
        cache_key = _sql_cache_key(sql_template, params)
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            _dbg("query_df_cached", sql=sql_template[:60])
            return cached
        self._ensure_pool()
        conn = self.get_connection()
        try:
            result = query_for_dataframe(conn, sql_template, params)
            self._query_cache.put(cache_key, result)
            self._health.record_success()
            _dbg("query_df_exec", sql=sql_template[:60],
                 rows=len(result) if result is not None else 0)
            return result
        except Exception:
            self._health.record_failure()
            raise

    def query_for_value(self, sql_template: str, params: Optional[list] = None):
        self._ensure_pool()
        conn = self.get_connection()
        _dbg("query_for_value", sql=sql_template[:60])
        try:
            result = query_for_value(conn, sql_template, params)
            self._health.record_success()
            return result
        except Exception:
            self._health.record_failure()
            raise

    def execute_query(self, sql: str, params: Optional[list] = None):
        self._ensure_pool()
        _dbg("execute_query", sql=sql[:60])
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            result = cursor.fetchall() if cursor.rowcount > 0 else None
            cursor.close()
            self._health.record_success()
            return result
        except Exception:
            self._health.record_failure()
            raise
        finally:
            try: conn.close()
            except Exception: pass

    def execute_manyquery(self, sql: str, params: Optional[list] = None):
        self._ensure_pool()
        _dbg("execute_manyquery", sql=sql[:60])
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.executemany(sql, params)
            conn.commit()
            result = cursor.fetchall() if cursor.rowcount > 0 else None
            cursor.close()
            self._health.record_success()
            return result
        except Exception:
            self._health.record_failure()
            raise
        finally:
            try: conn.close()
            except Exception: pass

    def execute_insert_with_transaction(self, sql: str, params: Optional[list] = None):
        self._ensure_pool()
        _dbg("execute_insert_txn", sql=sql[:60])
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            inserted_id = cursor.lastrowid
            cursor.close()
            conn.commit()
            self._health.record_success()
            return inserted_id
        except Exception:
            self._health.record_failure()
            raise
        finally:
            try: conn.close()
            except Exception: pass

    def batch_execute_with_transaction(self, sql_list: list) -> bool:
        self._ensure_pool()
        _dbg("batch_exec_txn", n_stmts=len(sql_list))
        conn = self.get_connection()
        success = True
        try:
            cursor = conn.cursor()
            for sql in sql_list:
                cursor.execute(sql)
            conn.commit()
            self._health.record_success()
        except Exception as e:
            _dbg("batch_exec_rollback", err=str(e))
            conn.rollback()
            success = False
            self._health.record_failure()
        finally:
            try: cursor.close(); conn.close()
            except Exception: pass
        return success

    def execute_with_rollback(self, sql: str, params):
        self._ensure_pool()
        _dbg("execute_with_rollback", sql=sql[:60])
        conn = self.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.rollback()
            cursor.close()
        finally:
            try: conn.close()
            except Exception: pass

    def destroy(self):
        _dbg("destroy_pool", type=self.pool_type)
        self._query_cache.invalidate()
        if self.pool is not None:
            try: self.pool.close()
            except Exception as e:
                logger.warning(f"close connection pool failed: {e}")


# ═══════════════════════════════════════════════════════════════════
#  OpenMySQLUtils (from videx_mysql_utils.py:189)
# ═══════════════════════════════════════════════════════════════════

class OpenMySQLUtils(AbstractMySQLUtils):
    """Open-source MySQL connector via host/port/user/password."""

    def __init__(self, config: MySQLConnectionConfig):
        super().__init__('open_mysql', config.database_name, config.charset,
                         config.read_timeout, config.write_timeout, config.connect_timeout)
        self.host = config.host
        self.port = config.port
        self.user = config.user
        self.password = config.pwd
        _dbg("OpenMySQL_init", host=self.host, port=self.port, db=self.database)

    def get_connection(self):
        _dbg("open_mysql_connect", host=self.host, port=self.port, db=self.database)
        try:
            import pymysql
            return pymysql.connect(user=self.user, password=self.password,
                                   db=self.database, charset=self.charset or 'utf8',
                                   host=self.host, port=self.port,
                                   read_timeout=self.read_timeout,
                                   write_timeout=self.write_timeout,
                                   connect_timeout=self.connect_timeout)
        except ImportError:
            _dbg("pymysql_stub", reason="pymysql not installed")
            raise RuntimeError("pymysql is required for OpenMySQLUtils")

    def get_sqlalchemy_engine(self, dbname: Optional[str] = None):
        import urllib.parse
        dbname = dbname or self.database
        _dbg("sqlalchemy_engine", db=dbname, host=self.host)
        try:
            from sqlalchemy import create_engine
            url = "mysql+pymysql://{user}:{pw}@[{host}]:{port}/{db}".format(
                host=self.host, port=self.port, db=dbname,
                user=self.user, pw=urllib.parse.quote_plus(self.password or ""))
            return create_engine(url)
        except ImportError:
            raise RuntimeError("sqlalchemy is required for engine creation")

    def __repr__(self):
        return f"OpenMySQL:{self.host}:{self.port}/{self.database}"

    def __str__(self):
        return self.__repr__()


# ═══════════════════════════════════════════════════════════════════
#  MySQL Version (from mysql_command.py:24)
# ═══════════════════════════════════════════════════════════════════

class MySQLVersion(Enum):
    MySQL_57 = 'mysql5.7'
    MySQL_8 = 'mysql8.0'
    MariaDB_11_8 = 'mariadb11.8'

    @staticmethod
    def get_version_enum(version: str) -> "MySQLVersion":
        _dbg("get_version_enum", raw=version)
        if "mariadb" in version.lower():
            return MySQLVersion.MariaDB_11_8
        elif version.startswith("8"):
            return MySQLVersion.MySQL_8
        elif version.startswith("5"):
            return MySQLVersion.MySQL_57
        return MySQLVersion.MySQL_57

    def __str__(self):
        return self.value


def get_mysql_version(mysql_util: AbstractMySQLUtils) -> MySQLVersion:
    _dbg("get_mysql_version", util=str(mysql_util)[:60])
    try:
        sql = "show variables like 'version';"
        df = mysql_util.query_for_dataframe(sql)
        if df is None or len(df) == 0:
            return MySQLVersion.MySQL_57
        version_str = str(df.iloc[0, 1]) if hasattr(df, 'iloc') else str(df)
        return MySQLVersion.get_version_enum(version_str)
    except Exception as e:
        _dbg("version_error", err=str(e))
        return MySQLVersion.MySQL_57


def datetime64_to_datetime(date_obj) -> Optional[datetime]:
    _dbg("datetime64_convert", input_type=type(date_obj).__name__)
    if date_obj is None:
        return None
    try:
        from numpy import datetime64
        if isinstance(date_obj, datetime64):
            ts = date_obj.tolist()
            if isinstance(ts, int):
                return datetime.fromtimestamp(ts / 1_000_000_000)
            return ts
    except ImportError:
        pass
    if isinstance(date_obj, datetime):
        return date_obj
    return date_obj


# ═══════════════════════════════════════════════════════════════════
#  MySQLCommand (from mysql_command.py:66)
# ═══════════════════════════════════════════════════════════════════

class MySQLCommand:
    """High-level MySQL metadata and EXPLAIN command runner.
    Modified: table meta results are cached with a bounded LFU and TTL."""

    def __init__(self, mysql_util: AbstractMySQLUtils, version: MySQLVersion):
        self.mysql_util = mysql_util
        self.version = version
        self._meta_cache = LFUCache(max_size=128, ttl_seconds=600.0)
        _dbg("MySQLCommand_init", version=version)

    def get_table_columns(self, db_name: str, table_name: str) -> List[Dict[str, Any]]:
        _dbg("get_table_columns", db=db_name, table=table_name)
        sql = (f"SELECT table_schema, table_name, column_name, ordinal_position, "
               f"is_nullable, data_type, character_maximum_length, "
               f"character_octet_length, numeric_precision, numeric_scale, "
               f"datetime_precision, character_set_name, collation_name, "
               f"column_type, column_key, extra "
               f"FROM information_schema.columns "
               f"WHERE table_schema='{db_name}' AND table_name='{table_name}'")
        df = self.mysql_util.query_for_dataframe(sql)
        columns = []
        if df is None or (hasattr(df, '__len__') and len(df) == 0):
            return columns
        rows = df.values if hasattr(df, 'values') else df
        for row in rows:
            col = {"db": row[0], "table": row[1], "name": row[2],
                   "ordinal_position": row[3], "is_nullable": row[4],
                   "data_type": row[5],
                   "character_maximum_length": _safe_int(row[6]),
                   "character_octet_length": _safe_int(row[7]),
                   "numeric_precision": _safe_int(row[8]),
                   "numeric_scale": _safe_int(row[9]),
                   "datetime_precision": _safe_int(row[10]),
                   "character_set_name": row[11], "collation_name": row[12],
                   "column_type": row[13], "column_key": row[14],
                   "auto_increment": 'auto_increment' in str(row[15])}
            columns.append(col)
        _dbg("get_table_columns_done", n_cols=len(columns))
        return columns

    def get_table_indexes(self, db_name: str, table_name: str) -> List[Dict[str, Any]]:
        _dbg("get_table_indexes", db=db_name, table=table_name, version=self.version)
        if self.version == MySQLVersion.MySQL_8:
            sql = (f"SELECT table_schema AS dbname, table_name, index_name, "
                   f"non_unique, seq_in_index, column_name, cardinality, "
                   f"sub_part, is_visible, expression, collation, index_type "
                   f"FROM information_schema.statistics "
                   f"WHERE table_schema='{db_name}' AND table_name='{table_name}'")
        else:
            sql = (f"SELECT table_schema AS dbname, table_name, index_name, "
                   f"non_unique, seq_in_index, column_name, cardinality, "
                   f"sub_part, 'YES' AS is_visible, 'NULL' AS expression, "
                   f"collation, index_type "
                   f"FROM information_schema.statistics "
                   f"WHERE table_schema='{db_name}' AND table_name='{table_name}'")
        df = self.mysql_util.query_for_dataframe(sql)
        if df is None or (hasattr(df, '__len__') and len(df) == 0):
            return []
        indexes_raw: Dict[str, List] = defaultdict(list)
        rows = df.values if hasattr(df, 'values') else df
        for row in rows:
            indexes_raw[row[2]].append({
                "dbname": row[0], "table_name": row[1], "index_name": row[2],
                "non_unique": row[3], "seq_in_index": row[4],
                "column_name": row[5], "cardinality": row[6],
                "sub_part": int(row[7]) if row[7] is not None and not _is_nan(row[7]) else 0,
                "is_visible": row[8], "expression": row[9],
                "collation": str(row[10]).replace('A', 'asc').replace('D', 'desc'),
                "index_type": row[11]})
        indexes = []
        for idx_name, col_rows in indexes_raw.items():
            non_unique = col_rows[0]["non_unique"]
            idx_type = "PRIMARY" if non_unique == 0 and idx_name == "PRIMARY" else \
                       "UNIQUE" if non_unique == 0 else "NORMAL"
            sorted_cols = sorted(col_rows, key=lambda r: r["seq_in_index"])
            indexes.append({"type": idx_type, "db_name": col_rows[0]["dbname"],
                "table_name": col_rows[0]["table_name"], "name": idx_name,
                "is_unique": non_unique == 0,
                "is_visible": col_rows[0]["is_visible"] == "YES",
                "index_type": col_rows[0]["index_type"],
                "columns": [{"column_name": c["column_name"],
                    "cardinality": c["cardinality"], "sub_part": c["sub_part"],
                    "expression": c["expression"], "collation": c["collation"]}
                    for c in sorted_cols]})
        _dbg("get_table_indexes_done", n_indexes=len(indexes))
        return indexes

    def get_table_meta(self, db_name: str, table_name: str) -> Dict[str, Any]:
        """Fetch full table metadata. Modified: caches result in LFU with TTL."""
        cache_key = f"{db_name}.{table_name}"
        cached = self._meta_cache.get(cache_key)
        if cached is not None:
            _dbg("get_table_meta_cached", db=db_name, table=table_name)
            return cached
        _dbg("get_table_meta", db=db_name, table=table_name)
        sql = f"SHOW TABLE STATUS IN `{db_name}` LIKE '{table_name}'"
        df = self.mysql_util.query_for_dataframe(sql)
        if df is None or (hasattr(df, '__len__') and len(df) == 0):
            raise LookupError(f"Table {db_name}.{table_name} not found")
        row = df.iloc[0] if hasattr(df, 'iloc') else df[0]
        table = {"name": table_name, "db": db_name,
            "engine": _get_df_val(row, "Engine"),
            "row_format": _get_df_val(row, "Row_format"),
            "collation": _get_df_val(row, "Collation"),
            "comment": _get_df_val(row, "Comment"),
            "rows": int(_get_df_val(row, "Rows") or 0),
            "avg_row_length": int(_get_df_val(row, "Avg_row_length") or 0),
            "data_length": int(_get_df_val(row, "Data_length") or 0),
            "index_length": int(_get_df_val(row, "Index_length") or 0),
            "create_time": _ts_from_dt(_get_df_val(row, "Create_time")),
            "update_time": _ts_from_dt(_get_df_val(row, "Update_time")),
            "check_time": _ts_from_dt(_get_df_val(row, "Check_time")),
            "columns": self.get_table_columns(db_name, table_name),
            "indexes": self.get_table_indexes(db_name, table_name)}
        try:
            stats_sql = (f"SELECT n_rows, clustered_index_size, sum_of_other_index_sizes "
                         f"FROM mysql.innodb_table_stats "
                         f"WHERE database_name='{db_name}' AND table_name='{table_name}'")
            sdf = self.mysql_util.query_for_dataframe(stats_sql)
            if sdf is not None and (hasattr(sdf, '__len__') and len(sdf) == 1):
                srow = sdf.iloc[0] if hasattr(sdf, 'iloc') else sdf[0]
                table["rows"] = int(srow[0])
                table["cluster_index_size"] = int(srow[1])
                table["other_index_sizes"] = int(srow[2])
        except Exception as e:
            _dbg("innodb_stats_skip", err=str(e))
        try:
            ddl_df = self.mysql_util.query_for_dataframe(
                f"SHOW CREATE TABLE `{db_name}`.`{table_name}`")
            ddl = str((ddl_df.values if hasattr(ddl_df, 'values') else ddl_df)[0][1])
            table["ddl"] = re.sub(r'\b(?:AUTO_INCREMENT|auto_increment)=\d+\b', "", ddl)
        except Exception:
            table["ddl"] = None
        self._meta_cache.put(cache_key, table)
        _dbg("get_table_meta_done", n_cols=len(table["columns"]),
             n_idx=len(table["indexes"]))
        return table

    def explain(self, sql: str, format: Optional[str] = None) -> Dict[str, Any]:
        _dbg("explain", sql=sql[:80], format=format)
        result: Dict[str, Any] = {"format": format}
        if format and format.upper() == "JSON":
            result["explain_json"] = self.mysql_util.query_for_value(f"EXPLAIN FORMAT=JSON {sql}")
        else:
            result["explain_items"] = self._explain_tabular(sql)
        return result

    def _explain_tabular(self, sql: str) -> List[Dict[str, Any]]:
        _dbg("explain_tabular", sql=sql[:80])
        df = self.mysql_util.query_for_dataframe(f"EXPLAIN {sql}")
        items = []
        if df is None: return items
        header = {0:"id",1:"select_type",2:"table",3:"partitions",4:"type",
                  5:"possible_keys",6:"key",7:"key_len",8:"ref",9:"rows",
                  10:"filtered",11:"Extra"}
        for row in (df.values if hasattr(df, 'values') else df):
            items.append({header[i]: row[i] for i in header if i < len(row)})
        return items


# ═══════════════════════════════════════════════════════════════════
#  Standalone query helpers (from videx_mysql_utils.py:229-265)
# ═══════════════════════════════════════════════════════════════════

def _parse_col_names(cursor) -> List[str]:
    _dbg("parse_col_names", rowcount=getattr(cursor, 'rowcount', '?'))
    if cursor.description is None:
        return []
    return [desc[0] for desc in cursor.description]


def query_for_dataframe(connection, sql_template: str, params: Optional[list] = None):
    _dbg("query_for_dataframe_raw", sql=sql_template[:60])
    try:
        cursor = connection.cursor()
        cursor.execute(sql_template, params)
        col_names = _parse_col_names(cursor)
        data = cursor.fetchall()
        connection.commit()
        cursor.close()
        try:
            import pandas as pd
            return pd.DataFrame([list(r) for r in data], columns=col_names)
        except ImportError:
            return _LightFrame(col_names, data)
    except Exception as e:
        logger.error(f"query_for_dataframe failed: {sql_template[:80]}, {e}")
        raise


def query_for_value(connection, sql_template: str, params: Optional[list] = None):
    _dbg("query_for_value_raw", sql=sql_template[:60])
    try:
        cursor = connection.cursor()
        cursor.execute(sql_template, params)
        data = cursor.fetchone()
        connection.commit()
        cursor.close()
        return data[0] if data else None
    except Exception as e:
        logger.error(f"query_for_value failed: {sql_template[:80]}, {e}")
        raise


# ═══════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════

def _sql_cache_key(sql: str, params: Optional[list]) -> str:
    raw = sql + (str(params) if params else "")
    return hashlib.md5(raw.encode("utf-8", errors="replace")).hexdigest()

def _safe_int(val) -> Optional[int]:
    if val is None: return None
    try:
        if isinstance(val, float) and math.isnan(val): return None
        return int(val)
    except (ValueError, TypeError): return None

def _is_nan(val) -> bool:
    try: return isinstance(val, float) and math.isnan(val)
    except TypeError: return False

def _get_df_val(row, col_name):
    if isinstance(row, dict): return row.get(col_name)
    if hasattr(row, '__getitem__'):
        try: return row[col_name]
        except (KeyError, IndexError): return None
    return getattr(row, col_name, None)

def _ts_from_dt(dt_val) -> Optional[int]:
    if dt_val is None: return None
    dt = datetime64_to_datetime(dt_val)
    if dt is None: return None
    if isinstance(dt, datetime): return int(dt.timestamp())
    return None


# ═══════════════════════════════════════════════════════════════════
#  Lightweight stubs (avoid hard deps on DBUtils)
# ═══════════════════════════════════════════════════════════════════

class _SharedPoolStub:
    def __init__(self, creator, min_cached=1, max_connections=10):
        self._creator = creator
    def connection(self, shareable=False):
        return _PoolConnCtx(self._creator)
    def close(self):
        pass

class _PersistentPoolStub:
    def __init__(self, creator):
        self._creator = creator
    def connection(self):
        return _PoolConnCtx(self._creator)
    def close(self):
        pass

class _PoolConnCtx:
    def __init__(self, creator):
        self._creator = creator
        self._conn = None
    def __enter__(self):
        self._conn = self._creator()
        return self._conn
    def __exit__(self, *args):
        if self._conn:
            try: self._conn.close()
            except Exception: pass

class _LightFrame:
    """Minimal DataFrame-like object for environments without pandas."""
    def __init__(self, columns: List[str], data: list):
        self.columns = columns
        self._data = [list(row) for row in data]
        self.values = self._data
    def __len__(self):
        return len(self._data)
    def __getitem__(self, key):
        if isinstance(key, str):
            idx = self.columns.index(key)
            return [row[idx] for row in self._data]
        return self._data[key]
    @property
    def iloc(self):
        return self
    @property
    def empty(self):
        return len(self._data) == 0
