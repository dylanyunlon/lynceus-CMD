"""
videx_env_manager — Environment management with fingerprinting for Lynceus.

Ported from:
  - upstream/videx/env/rds_env.py (391 lines)
  - upstream/videx/videx_build_env.py (254 lines)
  - upstream/videx/start_videx_server.py (45 lines)

Algorithm changes (~20%):
  - EnvManager: locality-sensitive hashing for environment fingerprinting
  - ConfigConvergence: detect when config changes stabilize via CUSUM
  - DDLParser: incremental schema hash for change detection
  - ServerStartup: graceful startup with exponential backoff health checks
"""
import math
import os
import re
import time
import hashlib
from collections import defaultdict

_DBG = bool(os.environ.get("LYNCEUS_DBG", ""))

def _dbg(tag, **kw):
    if _DBG:
        items = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        print(f"[env_mgr] {tag}: {items}")


# ── Environment fingerprint via LSH ─────────────────────────────
class EnvironmentFingerprint:
    """Locality-sensitive hash for environment state.
    
    Algorithm change: upstream uses exact config comparison.
    LSH fingerprint enables approximate matching: two environments
    with similar configs get similar fingerprints (Jaccard + MinHash).
    """
    
    def __init__(self, n_hashes=64, seed=42):
        self.n_hashes = n_hashes
        self.seed = seed
        self._hash_params = self._init_hash_params()
    
    def _init_hash_params(self):
        """Initialize MinHash parameters."""
        import random
        random.seed(self.seed)
        LARGE_PRIME = 2**31 - 1
        return [(random.randint(1, LARGE_PRIME), random.randint(0, LARGE_PRIME))
                for _ in range(self.n_hashes)]
    
    def compute(self, config_dict):
        """Compute MinHash fingerprint of config dictionary."""
        # Shingle the config into feature set
        shingles = set()
        for key, value in sorted(config_dict.items()):
            shingles.add(f"{key}={value}")
            # Also add key-only shingles for partial matching
            shingles.add(f"key:{key}")
        
        if not shingles:
            return tuple([0] * self.n_hashes)
        
        # MinHash: for each hash function, find minimum hash of any shingle
        LARGE_PRIME = 2**31 - 1
        signature = []
        for a, b in self._hash_params:
            min_hash = float("inf")
            for shingle in shingles:
                h = hash(shingle) & 0xFFFFFFFF
                val = (a * h + b) % LARGE_PRIME
                min_hash = min(min_hash, val)
            signature.append(min_hash)
        
        fp = tuple(signature)
        _dbg("fingerprint", n_shingles=len(shingles),
             fp_preview=fp[:4])
        return fp
    
    def similarity(self, fp1, fp2):
        """Estimate Jaccard similarity from MinHash signatures."""
        matches = sum(1 for a, b in zip(fp1, fp2) if a == b)
        sim = matches / len(fp1)
        _dbg("fp_similarity", matches=matches, total=len(fp1),
             jaccard=f"{sim:.4f}")
        return sim


# ── Config convergence detection via CUSUM ───────────────────────
class ConfigConvergenceDetector:
    """Detect when configuration changes have stabilized.
    
    Algorithm change: upstream has no convergence detection.
    Uses CUSUM (Cumulative Sum) control chart to detect when
    the rate of config changes drops below a threshold.
    """
    
    def __init__(self, threshold=3.0, drift=0.5):
        self.threshold = threshold
        self.drift = drift
        self._cusum_pos = 0.0
        self._cusum_neg = 0.0
        self._change_times = []
        self._converged = False
    
    def record_change(self, magnitude=1.0):
        """Record a configuration change event."""
        self._change_times.append(time.time())
        self._cusum_pos = max(0, self._cusum_pos + magnitude - self.drift)
        self._cusum_neg = min(0, self._cusum_neg - magnitude + self.drift)
        
        if self._cusum_pos > self.threshold:
            self._converged = False
            _dbg("cusum_alert", cusum_pos=f"{self._cusum_pos:.3f}")
        
    def record_stable(self):
        """Record a stable tick (no changes)."""
        self._cusum_pos = max(0, self._cusum_pos - self.drift)
        self._cusum_neg = min(0, self._cusum_neg + self.drift)
        
        if abs(self._cusum_pos) < 0.1 and abs(self._cusum_neg) < 0.1:
            self._converged = True
    
    @property
    def is_converged(self):
        return self._converged
    
    def dump_state(self):
        print(f"[CUSUM] pos={self._cusum_pos:.3f} neg={self._cusum_neg:.3f} "
              f"converged={self._converged} changes={len(self._change_times)}")


# ── DDL Parser with incremental schema hash ─────────────────────
class DDLParser:
    """Parse and track DDL changes with incremental hashing.
    
    Algorithm change: upstream re-parses entire schema on each check.
    Incremental hash: only recompute hash for changed tables.
    """
    
    def __init__(self):
        self._table_hashes = {}
        self._schema_hash = None
    
    def parse_create_table(self, ddl):
        """Parse CREATE TABLE DDL and extract structure."""
        table_match = re.search(r"CREATE\s+TABLE\s+`?(\w+)`?", ddl, re.IGNORECASE)
        table_name = table_match.group(1) if table_match else "unknown"
        
        columns = []
        for m in re.finditer(r"`(\w+)`\s+(\w+(?:\([^)]+\))?)", ddl):
            columns.append({"name": m.group(1), "type": m.group(2)})
        
        indexes = []
        for m in re.finditer(r"(?:KEY|INDEX)\s+`?(\w+)`?\s*\(([^)]+)\)", ddl):
            idx_cols = [c.strip().strip("`") for c in m.group(2).split(",")]
            indexes.append({"name": m.group(1), "columns": idx_cols})
        
        engine_match = re.search(r"ENGINE=(\w+)", ddl, re.IGNORECASE)
        engine = engine_match.group(1) if engine_match else "InnoDB"
        
        result = {
            "table_name": table_name,
            "columns": columns,
            "indexes": indexes,
            "engine": engine,
        }
        
        # Update incremental hash
        table_hash = hashlib.sha256(ddl.encode()).hexdigest()[:16]
        changed = self._table_hashes.get(table_name) != table_hash
        self._table_hashes[table_name] = table_hash
        
        if changed:
            self._recompute_schema_hash()
        
        _dbg("parse_ddl", table=table_name, cols=len(columns),
             idxs=len(indexes), changed=changed)
        return result
    
    def _recompute_schema_hash(self):
        """Recompute schema hash from individual table hashes."""
        combined = "|".join(f"{k}:{v}" for k, v in sorted(self._table_hashes.items()))
        self._schema_hash = hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    @property
    def schema_hash(self):
        return self._schema_hash
    
    def has_table_changed(self, table_name, new_ddl):
        """Check if a table's schema has changed."""
        new_hash = hashlib.sha256(new_ddl.encode()).hexdigest()[:16]
        return self._table_hashes.get(table_name) != new_hash


# ── Environment manager ──────────────────────────────────────────
class EnvManager:
    """Manage database environments with fingerprinting and convergence detection."""
    
    def __init__(self, default_db="test"):
        self.default_db = default_db
        self.meta_info = {}  # db -> table_name -> schema
        self.config_info = {}
        self._fingerprinter = EnvironmentFingerprint()
        self._convergence = ConfigConvergenceDetector()
        self._ddl_parser = DDLParser()
        self._current_fp = None
        
        _dbg("env_init", db=default_db)
    
    def register_table(self, db_name, ddl):
        """Register a table from its DDL."""
        schema = self._ddl_parser.parse_create_table(ddl)
        if db_name not in self.meta_info:
            self.meta_info[db_name] = {}
        self.meta_info[db_name][schema["table_name"]] = schema
        self._convergence.record_change()
        return schema
    
    def update_config(self, config_dict):
        """Update environment configuration."""
        old_fp = self._current_fp
        self._current_fp = self._fingerprinter.compute(config_dict)
        self.config_info.update(config_dict)
        
        if old_fp is not None:
            sim = self._fingerprinter.similarity(old_fp, self._current_fp)
            if sim > 0.95:
                self._convergence.record_stable()
            else:
                self._convergence.record_change(magnitude=1 - sim)
    
    def is_stable(self):
        """Check if the environment configuration has stabilized."""
        return self._convergence.is_converged
    
    def get_table_meta(self, db_name, table_name):
        return self.meta_info.get(db_name, {}).get(table_name)
    
    def dump_state(self):
        print(f"[EnvManager] db={self.default_db}")
        print(f"  tables: {sum(len(v) for v in self.meta_info.values())}")
        print(f"  config keys: {len(self.config_info)}")
        print(f"  schema_hash: {self._ddl_parser.schema_hash}")
        self._convergence.dump_state()


# ── Server startup with exponential backoff health check ─────────
class ServerStartupManager:
    """Manage server startup with health check backoff.
    
    Algorithm change: upstream does immediate start.
    Exponential backoff health checks ensure dependencies are ready.
    """
    
    def __init__(self, port=5001, host="0.0.0.0", max_retries=10):
        self.port = port
        self.host = host
        self.max_retries = max_retries
        self._started = False
    
    def check_health(self, check_fn=None):
        """Run health checks with exponential backoff."""
        delay = 0.5
        for attempt in range(self.max_retries):
            if check_fn is None:
                _dbg("health_ok", attempt=attempt)
                return True
            
            try:
                if check_fn():
                    _dbg("health_ok", attempt=attempt)
                    return True
            except Exception as e:
                _dbg("health_fail", attempt=attempt, error=str(e)[:50],
                     next_delay=f"{delay:.1f}s")
            
            time.sleep(delay)
            delay = min(delay * 2, 30)
        
        return False
    
    def startup_info(self):
        return {
            "host": self.host,
            "port": self.port,
            "set_command": f"SET @VIDEX_SERVER='{self.host}:{self.port}';",
        }
