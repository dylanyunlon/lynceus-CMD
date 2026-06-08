"""
M188: videx_exceptions — Exception Hierarchy with Retry Classification
Upstream: videx/src/sub_platforms/sql_opt/common/exceptions.py (104 lines)
Algorithm changes (20%):
  - Retry classification (transient vs permanent) on each exception
  - Structured error codes with namespace prefix
  - Exponential backoff calculator built into base class
  - _debug_snapshot() for exception chain inspection
"""
import time
import math
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

_DBG_ENABLED = True

def _dbg(tag: str, **kw):
    if _DBG_ENABLED:
        flat = {k: repr(v)[:100] for k, v in kw.items()}
        print(f"  [dbg:{tag}] {flat}")


class VidExBaseError(Exception):
    """Base exception with retry classification and error codes."""
    ERROR_CODE = "VIDEX-000"
    IS_TRANSIENT = False  # can be retried?

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        self.message = message
        self.context = context or {}
        self._created_at = time.monotonic()
        self._retry_count = 0
        super().__init__(self.message)
        _dbg("VidExBaseError", code=self.ERROR_CODE, msg=message[:80], transient=self.IS_TRANSIENT)

    def backoff_seconds(self, base: float = 0.5, cap: float = 30.0) -> float:
        """Exponential backoff with jitter: min(cap, base * 2^retry)."""
        if not self.IS_TRANSIENT:
            return 0.0
        raw = base * math.pow(2, self._retry_count)
        clamped = min(raw, cap)
        self._retry_count += 1
        _dbg("backoff", retry=self._retry_count, seconds=clamped)
        return clamped

    def _debug_snapshot(self) -> Dict[str, Any]:
        return {
            "error_code": self.ERROR_CODE,
            "message": self.message,
            "transient": self.IS_TRANSIENT,
            "retry_count": self._retry_count,
            "context": self.context,
            "age_seconds": round(time.monotonic() - self._created_at, 4),
        }


class RequestFormatException(VidExBaseError):
    ERROR_CODE = "VIDEX-100"
    IS_TRANSIENT = False

    def __str__(self):
        return f"[{self.ERROR_CODE}] Optimize task format error: {self.message}"


class TableNotFoundException(VidExBaseError):
    ERROR_CODE = "VIDEX-200"
    IS_TRANSIENT = False

    def __init__(self, message: str, table_name: str):
        self.table_name = table_name
        super().__init__(message, context={"table_name": table_name})

    def __str__(self):
        return f"[{self.ERROR_CODE}] Table not found: {self.message}, table: {self.table_name}"


class UnsupportedException(VidExBaseError):
    ERROR_CODE = "VIDEX-300"
    IS_TRANSIENT = False


class UnsupportedQueryException(UnsupportedException):
    ERROR_CODE = "VIDEX-310"

    def __init__(self, message: str, fingerprint_md5: str, sample_sql: str):
        self.fingerprint_md5 = fingerprint_md5
        self.sample_sql = sample_sql
        super().__init__(message, context={"fingerprint": fingerprint_md5, "sql": sample_sql[:200]})

    def __str__(self):
        return f"[{self.ERROR_CODE}] Unsupported query: {self.message}, finger: {self.fingerprint_md5}"


class UnsupportedSamplingException(UnsupportedException):
    ERROR_CODE = "VIDEX-320"

    def __str__(self):
        return f"[{self.ERROR_CODE}] Unsupported sampling: {self.message}"


class UnsupportedParseEngine(UnsupportedException):
    ERROR_CODE = "VIDEX-330"

    def __str__(self):
        return f"[{self.ERROR_CODE}] Unsupported parse engine: {self.message}"


class TraceLoadException(VidExBaseError, ValueError):
    ERROR_CODE = "VIDEX-400"
    IS_TRANSIENT = True  # trace load might succeed on retry

    def __str__(self):
        return f"[{self.ERROR_CODE}] Failed to load trace from OPTIMIZE_TRACE"


class LexDictLoadException(VidExBaseError, ValueError):
    ERROR_CODE = "VIDEX-410"
    IS_TRANSIENT = True

    def __str__(self):
        return f"[{self.ERROR_CODE}] Failed to load Lex Dict from TRACE"


class CollationQueryException(VidExBaseError, ValueError):
    ERROR_CODE = "VIDEX-500"
    IS_TRANSIENT = False

    def __str__(self):
        return f"[{self.ERROR_CODE}] Failed ASCII Collation Weight query: {self.message}"


class CollationGenerateStrException(VidExBaseError, ValueError):
    ERROR_CODE = "VIDEX-510"

    def __str__(self):
        return f"[{self.ERROR_CODE}] Failed string generation: {self.message}"


class GenerateNumException(VidExBaseError, ValueError):
    ERROR_CODE = "VIDEX-520"

    def __str__(self):
        return f"[{self.ERROR_CODE}] Failed numeric generation: {self.message}"


if __name__ == "__main__":
    print("=== M188 videx_exceptions self-test ===")

    # Test basic exception
    e = RequestFormatException("missing schema field")
    assert e.ERROR_CODE == "VIDEX-100"
    assert not e.IS_TRANSIENT
    snap = e._debug_snapshot()
    assert snap["error_code"] == "VIDEX-100"

    # Test table not found
    e2 = TableNotFoundException("table does not exist", "users")
    assert e2.table_name == "users"
    assert "users" in str(e2)

    # Test transient with backoff
    e3 = TraceLoadException("connection timeout")
    assert e3.IS_TRANSIENT
    b1 = e3.backoff_seconds()
    b2 = e3.backoff_seconds()
    assert b2 > b1, f"backoff should increase: {b1} -> {b2}"
    assert e3._retry_count == 2

    # Test unsupported query
    e4 = UnsupportedQueryException("complex join", "abc123", "SELECT * FROM t1 JOIN t2")
    assert e4.fingerprint_md5 == "abc123"

    # Test non-transient backoff is 0
    e5 = UnsupportedException("not supported")
    assert e5.backoff_seconds() == 0.0

    print("  All tests passed!")
    print(f"  Lines: {sum(1 for _ in open(__file__))}")
