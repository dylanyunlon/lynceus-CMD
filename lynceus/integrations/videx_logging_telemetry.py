"""
videx_logging_telemetry — Logging and telemetry with anomaly detection for Lynceus.

Ported from:
  - upstream/videx/videx_logging.py (200 lines)
  - upstream/videx/common/exceptions.py (104 lines)

Algorithm changes (~20%):
  - AdaptiveLogger: reservoir-sampled log entries to avoid log flooding
  - ErrorRateMonitor: CUSUM anomaly detection on error rate
  - StructuredExceptions: exception taxonomy with severity scoring
  - TraceIdGenerator: snowflake-style distributed trace IDs
"""
import math
import os
import time
import random
import hashlib
from collections import deque, defaultdict

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))

def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[log_tel] {tag}: {items}")


# ── Reservoir-sampled logger ────────────────────────────────────
class AdaptiveLogger:
    """Logger with reservoir sampling to prevent log flooding.
    
    Algorithm change: upstream logs every message.
    Reservoir sampling keeps a representative sample when log volume
    exceeds threshold, with priority weighting for errors.
    """
    
    def __init__(self, reservoir_size=1000, flood_threshold=100):
        self.reservoir_size = reservoir_size
        self.flood_threshold = flood_threshold  # messages per second
        self._reservoir = []
        self._count = 0
        self._window_start = time.time()
        self._window_count = 0
        self._dropped = 0
        self._level_counts = defaultdict(int)
    
    def log(self, level, message, **context):
        """Log a message with adaptive sampling."""
        self._count += 1
        self._window_count += 1
        self._level_counts[level] += 1
        
        now = time.time()
        elapsed = now - self._window_start
        
        # Reset window every second
        if elapsed >= 1.0:
            rate = self._window_count / elapsed
            self._window_start = now
            self._window_count = 0
            
            if rate > self.flood_threshold:
                _dbg("log_throttle", rate=f"{rate:.1f}/s",
                     threshold=self.flood_threshold)
        
        # Priority: errors always kept, others reservoir-sampled
        entry = {
            "time": now, "level": level,
            "message": message, "context": context,
        }
        
        priority = {"ERROR": 3, "WARN": 2, "INFO": 1, "DEBUG": 0}.get(level, 0)
        
        if priority >= 2 or len(self._reservoir) < self.reservoir_size:
            self._reservoir.append(entry)
            if len(self._reservoir) > self.reservoir_size:
                # Evict lowest priority
                self._reservoir.sort(key=lambda e: {"ERROR": 3, "WARN": 2, "INFO": 1, "DEBUG": 0}.get(e["level"], 0))
                self._reservoir.pop(0)
        else:
            # Reservoir sampling with priority weighting
            j = random.randint(0, self._count - 1)
            if j < self.reservoir_size:
                self._reservoir[j] = entry
            else:
                self._dropped += 1
    
    def info(self, message, **ctx):
        self.log("INFO", message, **ctx)
    
    def error(self, message, **ctx):
        self.log("ERROR", message, **ctx)
    
    def warn(self, message, **ctx):
        self.log("WARN", message, **ctx)
    
    def debug(self, message, **ctx):
        self.log("DEBUG", message, **ctx)
    
    def dump_state(self):
        print(f"[AdaptiveLogger] total={self._count} reservoir={len(self._reservoir)} "
              f"dropped={self._dropped}")
        print(f"  levels: {dict(self._level_counts)}")


# ── CUSUM error rate anomaly detection ──────────────────────────
class ErrorRateMonitor:
    """Monitor error rate with CUSUM anomaly detection.
    
    Algorithm change: upstream has no anomaly detection.
    CUSUM (Cumulative Sum) detects shifts in error rate mean:
    S_t = max(0, S_{t-1} + (x_t - μ_0 - k))
    Alert when S_t > h (decision interval).
    """
    
    def __init__(self, target_rate=0.01, sensitivity=0.5, threshold=4.0,
                 window_size=100):
        self.target_rate = target_rate
        self.sensitivity = sensitivity
        self.threshold = threshold
        self.window_size = window_size
        
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._window = deque(maxlen=window_size)
        self._total_requests = 0
        self._total_errors = 0
        self._alerts = []
    
    def record(self, is_error):
        """Record a request outcome."""
        self._total_requests += 1
        if is_error:
            self._total_errors += 1
        
        x = 1.0 if is_error else 0.0
        self._window.append(x)
        
        # CUSUM update
        self._cusum_pos = max(0, self._cusum_pos + x - self.target_rate - self.sensitivity)
        self._cusum_neg = min(0, self._cusum_neg + x - self.target_rate + self.sensitivity)
        
        # Check for anomaly
        if self._cusum_pos > self.threshold:
            self._alerts.append({
                "time": time.time(),
                "type": "rate_increase",
                "cusum": self._cusum_pos,
                "current_rate": sum(self._window) / max(len(self._window), 1),
            })
            self._cusum_pos = 0  # Reset after alert
            _dbg("cusum_alert", type="increase",
                 rate=f"{self._alerts[-1]['current_rate']:.4f}")
    
    @property
    def current_rate(self):
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)
    
    @property
    def is_anomalous(self):
        return self._cusum_pos > self.threshold * 0.5
    
    def dump_state(self):
        print(f"[ErrorMonitor] total={self._total_requests} errors={self._total_errors} "
              f"rate={self.current_rate:.4f} cusum={self._cusum_pos:.3f} "
              f"alerts={len(self._alerts)}")


# ── Structured exceptions ───────────────────────────────────────
class LynceusException(Exception):
    """Base exception with severity scoring."""
    SEVERITY_LOW = 1
    SEVERITY_MEDIUM = 2
    SEVERITY_HIGH = 3
    SEVERITY_CRITICAL = 4
    
    def __init__(self, message, severity=2, context=None):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()

class TableNotFoundException(LynceusException):
    def __init__(self, db, table):
        super().__init__(f"Table not found: {db}.{table}",
                        severity=self.SEVERITY_MEDIUM,
                        context={"db": db, "table": table})

class UnsupportedException(LynceusException):
    def __init__(self, feature):
        super().__init__(f"Unsupported: {feature}",
                        severity=self.SEVERITY_LOW,
                        context={"feature": feature})

class ConnectionException(LynceusException):
    def __init__(self, host, port, reason=""):
        super().__init__(f"Connection failed: {host}:{port} — {reason}",
                        severity=self.SEVERITY_HIGH,
                        context={"host": host, "port": port})

class StatisticsStaleException(LynceusException):
    def __init__(self, table, age_seconds):
        super().__init__(f"Statistics stale for {table} (age: {age_seconds:.0f}s)",
                        severity=self.SEVERITY_MEDIUM,
                        context={"table": table, "age": age_seconds})


# ── Snowflake-style trace ID generator ───────────────────────────
class TraceIdGenerator:
    """Generate distributed trace IDs using snowflake pattern.
    
    Algorithm change: upstream uses simple incrementing counter.
    Snowflake: 41 bits timestamp + 10 bits node + 12 bits sequence.
    """
    
    def __init__(self, node_id=0, epoch=1700000000000):
        self.node_id = node_id & 0x3FF  # 10 bits
        self.epoch = epoch
        self._sequence = 0
        self._last_ts = 0
    
    def generate(self):
        ts = int(time.time() * 1000) - self.epoch
        
        if ts == self._last_ts:
            self._sequence = (self._sequence + 1) & 0xFFF
            if self._sequence == 0:
                ts += 1
        else:
            self._sequence = 0
        
        self._last_ts = ts
        
        trace_id = ((ts & 0x1FFFFFFFFFF) << 22) | (self.node_id << 12) | self._sequence
        hex_id = f"{trace_id:016x}"
        
        _dbg("trace_id", hex=hex_id, ts=ts, seq=self._sequence)
        return hex_id
