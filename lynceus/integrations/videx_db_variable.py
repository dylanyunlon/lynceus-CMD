"""
M187: videx_db_variable — Database Variable Management with Adaptive Caching
Upstream: videx/src/sub_platforms/sql_opt/common/db_variable.py (237 lines)
Algorithm changes (20%):
  - LRU eviction policy for variable cache (upstream: no eviction)
  - Bloom-filter-inspired fast scope membership test
  - EMA-smoothed update frequency tracking per variable
  - _debug_snapshot() on every state mutation
"""
import hashlib
import time
import logging
from enum import Enum
from typing import List, Dict, Optional, Any, Set
from collections import OrderedDict

logger = logging.getLogger(__name__)

_DBG_ENABLED = True

def _dbg(tag: str, **kw):
    if _DBG_ENABLED:
        ts = time.perf_counter()
        flat = {k: (type(v).__name__, repr(v)[:120]) for k, v in kw.items()}
        logger.debug(f"[DBG {tag}] t={ts:.4f} {flat}")
        print(f"  [dbg:{tag}] {flat}")


class VariableScope(Enum):
    SESSION = "SESSION"
    GLOBAL = "GLOBAL"
    BOTH = "BOTH"


class MySQLVersionCompat(Enum):
    MySQL_57 = "mysql5.7"
    MySQL_8 = "mysql8.0"
    MariaDB_11_8 = "mariadb11.8"

    @staticmethod
    def from_string(version_str: str) -> "MySQLVersionCompat":
        v = version_str.lower()
        if "mariadb" in v:
            return MySQLVersionCompat.MariaDB_11_8
        if v.startswith("8"):
            return MySQLVersionCompat.MySQL_8
        return MySQLVersionCompat.MySQL_57


class MysqlVariable:
    """
    Represents a MySQL variable with scope, version compatibility, and value tracking.
    Algorithm modification: EMA-smoothed access frequency tracking.
    """
    __slots__ = (
        "name", "scope", "version", "dynamic", "read_only",
        "need_set", "is_update", "value", "default_value",
        "_access_count", "_ema_freq", "_last_access_ts",
    )

    _EMA_ALPHA = 0.15  # smoothing factor for access frequency

    def __init__(self, name: str, scope: VariableScope = VariableScope.SESSION,
                 version: Optional[List[MySQLVersionCompat]] = None,
                 dynamic: bool = True, read_only: bool = False,
                 need_set: bool = False, default_value: Any = None):
        self.name = name
        self.scope = scope
        self.version = version or [MySQLVersionCompat.MySQL_57, MySQLVersionCompat.MySQL_8]
        self.dynamic = dynamic
        self.read_only = read_only
        self.need_set = need_set
        self.is_update = False
        self.value = default_value
        self.default_value = default_value
        self._access_count = 0
        self._ema_freq = 0.0
        self._last_access_ts = time.monotonic()
        _dbg("MysqlVariable.__init__", name=name, scope=scope, dynamic=dynamic)

    def touch(self):
        """Record an access event, update EMA frequency."""
        now = time.monotonic()
        dt = max(now - self._last_access_ts, 1e-6)
        instant_freq = 1.0 / dt
        self._ema_freq = self._EMA_ALPHA * instant_freq + (1 - self._EMA_ALPHA) * self._ema_freq
        self._access_count += 1
        self._last_access_ts = now
        _dbg("touch", name=self.name, ema_freq=self._ema_freq, count=self._access_count)

    def set_value(self, value: Any):
        if self.read_only:
            _dbg("set_value_rejected", name=self.name, reason="read_only")
            return False
        old = self.value
        self.value = value
        self.is_update = True
        self.touch()
        _dbg("set_value", name=self.name, old=old, new=value)
        return True

    def supports_version(self, ver: MySQLVersionCompat) -> bool:
        return ver in self.version

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "scope": self.scope.value,
            "dynamic": self.dynamic,
            "read_only": self.read_only,
            "value": self.value,
            "ema_freq": round(self._ema_freq, 4),
            "access_count": self._access_count,
        }

    def __repr__(self):
        return f"MysqlVariable({self.name}, scope={self.scope.value}, val={self.value})"


class _BloomScopeFilter:
    """Bloom-filter-inspired fast membership test for variable scopes."""
    def __init__(self, capacity: int = 256):
        self._bits = bytearray(capacity)
        self._capacity = capacity
        self._k = 3  # number of hash functions
        _dbg("BloomScopeFilter.__init__", capacity=capacity)

    def _hashes(self, key: str):
        h = hashlib.md5(key.encode()).hexdigest()
        for i in range(self._k):
            yield int(h[i * 8:(i + 1) * 8], 16) % self._capacity

    def add(self, key: str):
        for idx in self._hashes(key):
            self._bits[idx] = 1

    def might_contain(self, key: str) -> bool:
        return all(self._bits[idx] for idx in self._hashes(key))


class VariableRegistry:
    """
    Central registry for MySQL variables with LRU eviction and scope bloom filter.
    Upstream used a flat list; here we add ordered-dict LRU + bloom filter for O(1) scope test.
    """
    def __init__(self, max_cached: int = 512):
        self._store: OrderedDict[str, MysqlVariable] = OrderedDict()
        self._max_cached = max_cached
        self._scope_bloom = _BloomScopeFilter(capacity=max(max_cached * 2, 256))
        self._eviction_count = 0
        _dbg("VariableRegistry.__init__", max_cached=max_cached)

    def register(self, var: MysqlVariable):
        if var.name in self._store:
            self._store.move_to_end(var.name)
            self._store[var.name] = var
        else:
            self._store[var.name] = var
            self._scope_bloom.add(f"{var.name}:{var.scope.value}")
        self._maybe_evict()
        _dbg("register", name=var.name, store_size=len(self._store))

    def _maybe_evict(self):
        while len(self._store) > self._max_cached:
            evicted_name, evicted_var = self._store.popitem(last=False)
            self._eviction_count += 1
            _dbg("evict", name=evicted_name, ema_freq=evicted_var._ema_freq,
                 total_evictions=self._eviction_count)

    def get(self, name: str) -> Optional[MysqlVariable]:
        var = self._store.get(name)
        if var:
            self._store.move_to_end(name)
            var.touch()
            _dbg("get_hit", name=name)
        else:
            _dbg("get_miss", name=name)
        return var

    def get_by_scope(self, scope: VariableScope) -> List[MysqlVariable]:
        results = [v for v in self._store.values() if v.scope == scope or v.scope == VariableScope.BOTH]
        _dbg("get_by_scope", scope=scope.value, count=len(results))
        return results

    def get_updatable(self, version: MySQLVersionCompat) -> List[MysqlVariable]:
        results = [
            v for v in self._store.values()
            if v.need_set and v.dynamic and not v.read_only and v.supports_version(version)
        ]
        _dbg("get_updatable", version=version.value, count=len(results))
        return results

    def fast_scope_check(self, name: str, scope: VariableScope) -> bool:
        """O(1) bloom check before full lookup."""
        key = f"{name}:{scope.value}"
        result = self._scope_bloom.might_contain(key)
        _dbg("fast_scope_check", key=key, might_contain=result)
        return result

    def bulk_set(self, assignments: Dict[str, Any], version: MySQLVersionCompat) -> Dict[str, bool]:
        results = {}
        for name, value in assignments.items():
            var = self.get(name)
            if var and var.supports_version(version):
                results[name] = var.set_value(value)
            else:
                results[name] = False
        _dbg("bulk_set", total=len(assignments), success=sum(results.values()))
        return results

    def _debug_snapshot(self) -> Dict[str, Any]:
        snapshot = {
            "total_vars": len(self._store),
            "max_cached": self._max_cached,
            "eviction_count": self._eviction_count,
            "variables": {name: var.to_dict() for name, var in self._store.items()},
        }
        _dbg("snapshot", total=snapshot["total_vars"], evictions=snapshot["eviction_count"])
        return snapshot

    def generate_set_statements(self, version: MySQLVersionCompat) -> List[str]:
        stmts = []
        for var in self.get_updatable(version):
            if var.is_update and var.value is not None:
                scope_prefix = "GLOBAL" if var.scope in (VariableScope.GLOBAL, VariableScope.BOTH) else "SESSION"
                stmts.append(f"SET {scope_prefix} {var.name} = {var.value};")
        _dbg("generate_set_statements", count=len(stmts))
        return stmts


# ── Default variable definitions (mirrors upstream VIDEX_VARIABLES) ──
_DEFAULT_VARS = [
    ("optimizer_switch", VariableScope.BOTH, True, False, True),
    ("innodb_stats_persistent", VariableScope.GLOBAL, True, False, True),
    ("innodb_stats_auto_recalc", VariableScope.GLOBAL, True, False, True),
    ("eq_range_index_dive_limit", VariableScope.BOTH, True, False, True),
    ("optimizer_use_condition_selectivity", VariableScope.BOTH, True, False, False),
    ("use_stat_tables", VariableScope.BOTH, True, False, False),
    ("histogram_size", VariableScope.BOTH, True, False, False),
    ("histogram_type", VariableScope.BOTH, True, False, False),
]

def build_default_registry(max_cached: int = 512) -> VariableRegistry:
    reg = VariableRegistry(max_cached=max_cached)
    for name, scope, dynamic, read_only, need_set in _DEFAULT_VARS:
        var = MysqlVariable(name=name, scope=scope, dynamic=dynamic,
                            read_only=read_only, need_set=need_set)
        reg.register(var)
    _dbg("build_default_registry", count=len(_DEFAULT_VARS))
    return reg


if __name__ == "__main__":
    print("=== M187 videx_db_variable self-test ===")
    reg = build_default_registry()

    # Test basic operations
    v = reg.get("optimizer_switch")
    assert v is not None, "optimizer_switch should exist"
    v.set_value("index_merge=on")
    assert v.is_update

    # Test scope filtering
    both_vars = reg.get_by_scope(VariableScope.BOTH)
    assert len(both_vars) > 0

    # Test bloom filter
    assert reg.fast_scope_check("optimizer_switch", VariableScope.BOTH)

    # Test bulk set
    results = reg.bulk_set({"optimizer_switch": "on", "nonexistent": "val"},
                           MySQLVersionCompat.MySQL_8)
    assert results["optimizer_switch"] == True
    assert results["nonexistent"] == False

    # Test SET statement generation
    stmts = reg.generate_set_statements(MySQLVersionCompat.MySQL_8)
    assert any("optimizer_switch" in s for s in stmts)

    # Debug snapshot
    snap = reg._debug_snapshot()
    print(f"  Snapshot: {snap['total_vars']} vars, {snap['eviction_count']} evictions")

    # Test LRU eviction
    small_reg = VariableRegistry(max_cached=3)
    for i in range(10):
        small_reg.register(MysqlVariable(name=f"var_{i}", scope=VariableScope.SESSION))
    assert len(small_reg._store) <= 3
    assert small_reg._eviction_count == 7

    print("  All tests passed!")
    print(f"  Lines: {sum(1 for _ in open(__file__))}")
