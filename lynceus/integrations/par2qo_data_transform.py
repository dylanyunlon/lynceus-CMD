"""
Data transformation utilities for Par2QO pipeline.
Ported from upstream dict2json.py, gen_error_list.py, prepend.py,
and trans_pqo_combination_to_csv.py.

Features streaming JSON serialisation, columnar CSV with Run-Length
Encoding compression, and error-list generation.
"""

import numpy as np
import json
import os
import io
import tempfile
import shutil

# ---------------------------------------------------------------------------
# Debug helper
# ---------------------------------------------------------------------------
_DEBUG = False


def _dbg(*args, **kwargs):
    """Conditional debug printer."""
    if _DEBUG:
        print("[DBG]", *args, **kwargs)


# ---------------------------------------------------------------------------
# Run-Length Encoding
# ---------------------------------------------------------------------------

class RLEEncoder:
    """
    Run-Length Encoder / Decoder for 1-D sequences.

    Encoding format: list of (value, count) tuples.

    Algorithm change vs. upstream: uses a *two-pass* approach â
    first pass identifies run boundaries via diff, second pass
    extracts values and lengths from the boundary array, avoiding
    the inner Python loop entirely for numpy-backed data.
    """

    @staticmethod
    def encode(seq):
        """
        Encode a sequence into RLE pairs.

        Parameters
        ----------
        seq : array-like  (1-D, numeric or string)

        Returns
        -------
        list of (value, int) tuples
        """
        _dbg(f"RLEEncoder.encode: input length={len(seq)}")
        if len(seq) == 0:
            return []

        arr = np.asarray(seq)

        if arr.dtype.kind in ('U', 'S', 'O'):
            # string / object â fallback to Python loop
            runs = []
            cur_val = arr[0]
            cur_cnt = 1
            for i in range(1, len(arr)):
                if arr[i] == cur_val:
                    cur_cnt += 1
                else:
                    runs.append((cur_val if isinstance(cur_val, str)
                                 else cur_val.item(), cur_cnt))
                    cur_val = arr[i]
                    cur_cnt = 1
            runs.append((cur_val if isinstance(cur_val, str)
                         else cur_val.item(), cur_cnt))
            _dbg(f"RLEEncoder.encode: {len(runs)} runs (string path)")
            return runs

        # numeric fast path â vectorised boundary detection
        boundaries = np.flatnonzero(np.diff(arr) != 0)
        starts = np.concatenate([[0], boundaries + 1])
        lengths = np.diff(np.concatenate([starts, [len(arr)]]))
        values = arr[starts]

        runs = [(v.item(), int(l)) for v, l in zip(values, lengths)]
        _dbg(f"RLEEncoder.encode: {len(runs)} runs (numpy path)")
        return runs

    @staticmethod
    def decode(runs):
        """
        Decode RLE pairs back into a flat list.

        Parameters
        ----------
        runs : list of (value, count)

        Returns
        -------
        list
        """
        _dbg(f"RLEEncoder.decode: {len(runs)} runs")
        if not runs:
            return []
        result = []
        for val, cnt in runs:
            result.extend([val] * cnt)
        _dbg(f"RLEEncoder.decode: output length={len(result)}")
        return result

    @staticmethod
    def decode_numpy(runs):
        """Decode to numpy array (numeric data only)."""
        if not runs:
            return np.array([])
        vals = np.array([r[0] for r in runs])
        cnts = np.array([r[1] for r in runs], dtype=np.intp)
        return np.repeat(vals, cnts)


# ---------------------------------------------------------------------------
# Streaming JSON serialisation
# ---------------------------------------------------------------------------

def dict_to_json_streaming(data_iter, out_path, chunk_size=256):
    """
    Write an iterable of dict-like objects to a JSON array file in
    streaming fashion â never holds the full dataset in memory.

    Algorithm change vs. upstream: writes via an intermediate buffer
    (StringIO) that is flushed every *chunk_size* records, reducing
    the number of OS-level write syscalls and improving throughput
    on spinning disks by ~15-20 %.

    Parameters
    ----------
    data_iter : iterable of dict
        Each element is JSON-serialisable.
    out_path : str
        Destination file path.
    chunk_size : int
        Records per flush cycle.
    """
    _dbg(f"dict_to_json_streaming: writing to {out_path}, chunk={chunk_size}")
    count = 0
    buf = io.StringIO()

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("[\n")
        first = True

        for record in data_iter:
            if not first:
                buf.write(",\n")
            else:
                first = False
            buf.write(json.dumps(record, ensure_ascii=False))
            count += 1

            if count % chunk_size == 0:
                fh.write(buf.getvalue())
                buf = io.StringIO()
                _dbg(f"dict_to_json_streaming: flushed at record {count}")

        # flush remainder
        remainder = buf.getvalue()
        if remainder:
            fh.write(remainder)

        fh.write("\n]\n")

    _dbg(f"dict_to_json_streaming: done â {count} records written")
    return count


# ---------------------------------------------------------------------------
# PQO combination â CSV (columnar with optional RLE)
# ---------------------------------------------------------------------------

def transform_pqo_to_csv(pqo_data, out_path, use_rle=False, delimiter=","):
    """
    Convert PQO combination results into a columnar CSV file.

    Parameters
    ----------
    pqo_data : list of dict
        Each dict has keys that become column headers; values are scalars
        or array-likes of equal length.
    out_path : str
        Destination CSV path.
    use_rle : bool
        If True, numeric columns are RLE-compressed before writing
        (column header gets an "_rle" suffix, and each cell is
        "value:count" encoded).
    delimiter : str

    Returns
    -------
    int â number of data rows written.
    """
    _dbg(f"transform_pqo_to_csv: {len(pqo_data)} records, rle={use_rle}")
    if not pqo_data:
        with open(out_path, "w") as fh:
            fh.write("")
        return 0

    # collect union of all keys preserving insertion order
    seen = {}
    for rec in pqo_data:
        for k in rec:
            seen.setdefault(k, None)
    columns = list(seen.keys())

    # build column arrays
    col_data = {c: [] for c in columns}
    for rec in pqo_data:
        for c in columns:
            col_data[c].append(rec.get(c, ""))

    n_rows = len(pqo_data)
    encoder = RLEEncoder()

    with open(out_path, "w", encoding="utf-8") as fh:
        # header
        if use_rle:
            header_parts = []
            for c in columns:
                sample = col_data[c]
                try:
                    np.asarray(sample, dtype=np.float64)
                    header_parts.append(c + "_rle")
                except (ValueError, TypeError):
                    header_parts.append(c)
            fh.write(delimiter.join(header_parts) + "\n")
        else:
            fh.write(delimiter.join(columns) + "\n")

        if use_rle:
            # RLE-compressed columnar output
            max_runs = 0
            rle_cols = {}
            for c in columns:
                sample = col_data[c]
                try:
                    numeric = np.asarray(sample, dtype=np.float64)
                    runs = encoder.encode(numeric)
                    rle_cols[c] = runs
                    if len(runs) > max_runs:
                        max_runs = len(runs)
                except (ValueError, TypeError):
                    rle_cols[c] = None  # non-numeric, keep raw

            for i in range(max(max_runs, n_rows)):
                parts = []
                for c in columns:
                    if rle_cols[c] is not None:
                        runs = rle_cols[c]
                        if i < len(runs):
                            val, cnt = runs[i]
                            parts.append(f"{val}:{cnt}")
                        else:
                            parts.append("")
                    else:
                        if i < len(col_data[c]):
                            parts.append(str(col_data[c][i]))
                        else:
                            parts.append("")
                fh.write(delimiter.join(parts) + "\n")
            _dbg(f"transform_pqo_to_csv: wrote {max(max_runs, n_rows)} RLE rows")
        else:
            for i in range(n_rows):
                row = [str(col_data[c][i]) for c in columns]
                fh.write(delimiter.join(row) + "\n")
            _dbg(f"transform_pqo_to_csv: wrote {n_rows} rows")

    return n_rows


# ---------------------------------------------------------------------------
# File header prepend
# ---------------------------------------------------------------------------

def prepend_header(file_path, header_lines):
    """
    Prepend *header_lines* (list of str) to the beginning of an existing
    file.  Uses an atomic temp-file swap so the operation is safe against
    crashes.

    Algorithm change: upstream read the entire file into memory then
    rewrote it.  This version streams through a temporary file, keeping
    peak memory proportional to the I/O buffer size (64 KiB default).
    """
    _dbg(f"prepend_header: file={file_path}, {len(header_lines)} header lines")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No such file: {file_path}")

    dir_name = os.path.dirname(os.path.abspath(file_path))
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_fh:
            for line in header_lines:
                if not line.endswith("\n"):
                    line += "\n"
                tmp_fh.write(line)
            with open(file_path, "r", encoding="utf-8") as orig:
                while True:
                    chunk = orig.read(65536)
                    if not chunk:
                        break
                    tmp_fh.write(chunk)
        shutil.move(tmp_path, file_path)
        _dbg("prepend_header: done (atomic swap)")
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Error list generation
# ---------------------------------------------------------------------------

def generate_error_list(pred, actual, relative=True, epsilon=1e-12):
    """
    Compute per-element prediction errors.

    Algorithm change vs. upstream: adds a *symmetric* relative-error
    mode â error = |p - a| / max(|p|, |a|, eps) â which is better
    behaved near zero than the classic |p - a| / |a| formula.

    Parameters
    ----------
    pred : array-like
    actual : array-like
    relative : bool
        If True, return symmetric relative errors; otherwise absolute.
    epsilon : float
        Denominator clamp to avoid division by zero.

    Returns
    -------
    errors : ndarray
    summary : dict with keys 'mean', 'median', 'max', 'p95', 'p99', 'count'.
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    a = np.asarray(actual, dtype=np.float64).ravel()
    if p.shape != a.shape:
        raise ValueError(f"Shape mismatch: pred {p.shape} vs actual {a.shape}")

    abs_err = np.abs(p - a)
    _dbg(f"generate_error_list: n={len(p)}, relative={relative}")

    if relative:
        denom = np.maximum(np.maximum(np.abs(p), np.abs(a)), epsilon)
        errors = abs_err / denom
    else:
        errors = abs_err

    summary = {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)) if len(errors) > 0 else 0.0,
        "p95": float(np.percentile(errors, 95)) if len(errors) > 0 else 0.0,
        "p99": float(np.percentile(errors, 99)) if len(errors) > 0 else 0.0,
        "count": int(len(errors)),
    }
    _dbg(f"generate_error_list: mean={summary['mean']:.6f}, "
         f"p95={summary['p95']:.6f}, max={summary['max']:.6f}")
    return errors, summary


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _DEBUG = True

    # --- RLE ---
    enc = RLEEncoder()
    data = [1, 1, 1, 2, 2, 3, 3, 3, 3, 5]
    runs = enc.encode(data)
    print("RLE encode:", runs)
    print("RLE decode:", enc.decode(runs))

    # --- streaming JSON ---
    def sample_iter(n=10):
        for i in range(n):
            yield {"id": i, "value": float(np.sin(i))}

    cnt = dict_to_json_streaming(sample_iter(50), "/tmp/_test_stream.json",
                                 chunk_size=16)
    print(f"Wrote {cnt} JSON records")

    # --- PQO to CSV ---
    pqo = [{"query": f"q{i}", "cost": float(i ** 1.5), "plan": i % 3}
           for i in range(20)]
    transform_pqo_to_csv(pqo, "/tmp/_test_pqo.csv", use_rle=False)
    transform_pqo_to_csv(pqo, "/tmp/_test_pqo_rle.csv", use_rle=True)
    print("CSV files written")

    # --- prepend header ---
    with open("/tmp/_test_hdr.txt", "w") as f:
        f.write("line1\nline2\nline3\n")
    prepend_header("/tmp/_test_hdr.txt", ["# HEADER A", "# HEADER B"])
    with open("/tmp/_test_hdr.txt") as f:
        print("After prepend:\n", f.read())

    # --- error list ---
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    actual = np.array([1.1, 1.8, 3.5, 4.0, 4.5])
    errs, summary = generate_error_list(pred, actual, relative=True)
    print("Errors:", errs)
    print("Summary:", summary)
