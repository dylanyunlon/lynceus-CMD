# -*- coding: utf-8 -*-
"""
Original: PAR2QO utility.py — helper functions for cardinality, JSON, cost
          parsing (upstream/par2qo/code/utility.py, Hap-Hugh/PAR2QO)
Modified: Lynceus — heterogeneous utility functions with GPU-aware cost parsing,
          column-oriented data transforms, and diagnostic print helpers.

Modifications from upstream utility.py (~80% structure kept, ~20% changed):
  - Removed: matplotlib, multiprocessing, numpy, tqdm imports
  - Removed: file I/O functions that hardcode PostgreSQL output format
  - Kept:    card() cardinality clamp
  - Kept:    list_multiply() element-wise product
  - Kept:    clean() JSON cleaner structure
  - Kept:    get_cost_list() cost extraction from plan JSON
  - Kept:    join_hint / yuxi helper string builders
  - Modified: cost parsing now extracts GPU/CPU annotations
  - Modified: JSON cleaner handles Lynceus-format plan output
  - Added:   batch_cost_summary() for multi-plan analysis
  - Added:   selectivity_perturbation() from error vectors
  - Added:   debug_print_table() for structured state dumps

References:
  PAR2QO utility.py:9   — card(a) cardinality floor
  PAR2QO utility.py:14  — list_multiply(a, b)
  PAR2QO utility.py:60  — clean(json_file, new_json_file)
  PAR2QO utility.py:82  — get_cost_list(json_file)
"""
from __future__ import annotations

import math
import json
import re
import time
import logging
from typing import List, Dict, Optional, Tuple, Any, Union

from .. import _dbg, _dump_obj, _snapshot, _Timer, LYNCEUS_DEBUG
_T = "PUT"


logger = logging.getLogger("lynceus.par2qo_utils")


# ── Cardinality helpers (from PAR2QO utility.py:9-20) ──────────────────

def card(a: Union[int, float, str]) -> int:
    """Clamp cardinality to at least 1.

    PAR2QO: card(a) — if int(a)==0 return 1 else int(a).
    Lynceus: identical logic, added type safety.
    """
    _dbg(_T, "card()")
    try:
        v = int(a)
    except (ValueError, TypeError):
        v = 0
    return max(v, 1)


def list_multiply(a: List[float], b: List[float]) -> List[float]:
    """Element-wise product of two lists.

    PAR2QO: list_multiply(a, b) — assert same length, zip-multiply.
    Lynceus: identical.
    """
    _dbg(_T, "list_multiply()")
    assert len(a) == len(b), (
        f"list_multiply: length mismatch ({len(a)} vs {len(b)})"
    )
    return [x * y for x, y in zip(a, b)]


def list_add(a: List[float], b: List[float]) -> List[float]:
    """Element-wise sum. Lynceus addition."""
    _dbg(_T, "list_add()")
    assert len(a) == len(b), f"list_add: length mismatch ({len(a)} vs {len(b)})"
    return [x + y for x, y in zip(a, b)]


def list_ratio(a: List[float], b: List[float], epsilon: float = 1e-12) -> List[float]:
    """Element-wise ratio a/b with epsilon guard. Lynceus addition."""
    _dbg(_T, "list_ratio()")
    assert len(a) == len(b), f"list_ratio: length mismatch ({len(a)} vs {len(b)})"
    return [x / max(y, epsilon) for x, y in zip(a, b)]


# ── Selectivity helpers ────────────────────────────────────────────────

def selectivity_from_cardinality(
    estimated: List[int],
    raw: List[int],
) -> List[float]:
    """Compute selectivity = estimated / raw for each relation.

    PAR2QO: [est_base_card[i]/raw_base_card[i] for i in range(n)].
    Lynceus: vectorised with safety.
    """
    _dbg(_T, "selectivity_from_cardinality()")
    result = []
    for e, r in zip(estimated, raw):
        result.append(card(e) / card(r))
    return result


def selectivity_perturbation(
    base_sel: List[float],
    join_sel: List[float],
    error: List[float],
    sensitive_dims: List[int],
    use_log_space: bool = True,
    debug: bool = False,
) -> Tuple[List[float], List[float]]:
    """Perturb selectivities by error vector on sensitive dimensions.

    PAR2QO: prep_sel(..., error, recentered_error, relation_list, ...).
    Lynceus: simplified interface. ~20% change: log-space perturbation default.
    """
    _dbg(_T, "selectivity_perturbation()")
    perturbed_base = list(base_sel)
    perturbed_join = list(join_sel)
    n_base = len(base_sel)

    for i, dim_id in enumerate(sensitive_dims):
        if i >= len(error):
            break

        if use_log_space:
            factor = math.exp(error[i])
        else:
            factor = 1.0 + error[i]

        if dim_id < n_base:
            old = perturbed_base[dim_id]
            perturbed_base[dim_id] = min(1.0, max(1e-8, old * factor))
            if debug:
                print(f"    perturb base[{dim_id}]: "
                      f"{old:.6f} → {perturbed_base[dim_id]:.6f} "
                      f"(err={error[i]:.3f}, factor={factor:.3f})")
        else:
            j = dim_id - n_base
            if j < len(perturbed_join):
                old = perturbed_join[j]
                perturbed_join[j] = min(1.0, max(1e-8, old * factor))
                if debug:
                    print(f"    perturb join[{j}]: "
                          f"{old:.6f} → {perturbed_join[j]:.6f} "
                          f"(err={error[i]:.3f}, factor={factor:.3f})")

    return perturbed_base, perturbed_join


# ── Hint string builders (from PAR2QO utility.py:21-49) ────────────────

def yuxi(i: int, order: List[str]) -> str:
    """Build pg_hint_plan-style alias reference.
    PAR2QO: yuxi(i, order) → ' (yuxi_N order[i]) '."""
    _dbg(_T, "yuxi()")
    return f" (yuxi_{i} {order[i]}) "


def yuxi_short(i: int, order: List[str]) -> str:
    """Short form alias reference.
    PAR2QO: yuxi_short(i, order)."""
    _dbg(_T, "yuxi_short()")
    return f" yuxi_{i} {order[i]}"


def yuxi_card(join_list: List[str], rows: int) -> str:
    """Cardinality hint for a join group.
    PAR2QO: yuxi_card(join_list, rows)."""
    _dbg(_T, "yuxi_card()")
    card_str = "Rows("
    for item in join_list:
        card_str += item
    return card_str + f" #{rows})"


def join_hint(join_list: List[str], mtd: Optional[str] = None) -> str:
    """Build a join hint from a list of table references.
    PAR2QO: join_hint(join_list, mtd)."""
    _dbg(_T, "join_hint()")
    if mtd is None:
        join_str = "("
    else:
        join_str = f"{mtd}("
    for item in join_list:
        join_str += item
    return join_str + ")"


def modify_query(sql: str, hint: str, explain: str = "") -> str:
    """Prepend hint to query for execution.
    PAR2QO: modify_query — simple concat.
    Lynceus: also strips device annotations for clean SQL."""
    # Strip existing device hints
    _dbg(_T, "modify_query()")
    cleaned = re.sub(r'/\*\+\s*(GPU|CPU)\s*\*/', '', sql)
    if explain:
        return f"{explain}\n{hint}\n{cleaned}"
    return f"{hint}\n{cleaned}"


# ── JSON/cost parsing (from PAR2QO utility.py:60-150) ──────────────────

def clean_json(raw_text: str, del_keys: Optional[List[str]] = None) -> str:
    """Clean PostgreSQL EXPLAIN JSON output into parseable JSON.

    PAR2QO: clean(json_file, new_json_file, del_line_key).
    Lynceus: operates on strings instead of files.
    """
    _dbg(_T, "clean_json()")
    if del_keys is None:
        del_keys = ["QUERY PLAN", "row)", "----"]

    lines = []
    for line in raw_text.split("\n"):
        skip = False
        for key in del_keys:
            if key in line:
                skip = True
                break
        if not skip:
            line = line.replace("+", "")
            lines.append(line.strip())
    return "\n".join(lines)


def get_cost_list_from_json(
    json_text: str,
    is_estimate: bool = True,
) -> Tuple[List[float], List[float]]:
    """Extract cost lists from plan JSON text.

    PAR2QO: get_cost_list(json_file, is_estimate) → reads file.
    Lynceus: parses string. Returns (estimated_costs, actual_costs).
    """
    _dbg(_T, "get_cost_list_from_json()")
    est_costs = []
    actual_costs = []
    in_plan = False

    for line in json_text.split("\n"):
        if line.strip() == "[":
            in_plan = True
            continue
        if in_plan and "Total Cost" in line:
            match = re.search(r'"Total Cost":\s*([\d.]+)', line)
            if match:
                est_costs.append(float(match.group(1)))
            in_plan = False
        if "Actual Total Time" in line:
            match = re.search(r'"Actual Total Time":\s*([\d.]+)', line)
            if match:
                actual_costs.append(float(match.group(1)))

    return est_costs, actual_costs


def parse_plan_costs(
    plan_json: Dict[str, Any],
    debug: bool = False,
) -> Dict[str, float]:
    """Parse cost breakdown from a single plan JSON object.

    PAR2QO: scattered across robustness.py / postgres.py.
    Lynceus: unified parser with device annotation extraction.
    Returns: {total_cost, startup_cost, plan_rows, plan_width, ...}
    """
    _dbg(_T, "parse_plan_costs()")
    result = {}
    plan = plan_json.get("Plan", plan_json)

    result["total_cost"] = plan.get("Total Cost", 0.0)
    result["startup_cost"] = plan.get("Startup Cost", 0.0)
    result["plan_rows"] = plan.get("Plan Rows", 0)
    result["plan_width"] = plan.get("Plan Width", 0)
    result["node_type"] = plan.get("Node Type", "Unknown")

    # Lynceus: extract device annotations if present
    output = plan.get("Output", [])
    if isinstance(output, list):
        for item in output:
            if isinstance(item, str) and "/*+GPU*/" in item:
                result["device_hint"] = "gpu"
            elif isinstance(item, str) and "/*+CPU*/" in item:
                result["device_hint"] = "cpu"

    if debug:
        print(f"    plan_cost: node={result['node_type']} "
              f"total={result['total_cost']:.1f} "
              f"rows={result['plan_rows']} "
              f"device={result.get('device_hint', 'auto')}")

    return result


# ── Batch cost summary ─────────────────────────────────────────────────

def batch_cost_summary(
    plans: List[Dict[str, Any]],
    plan_labels: Optional[List[str]] = None,
    debug: bool = True,
) -> Dict[str, Any]:
    """Summarize costs across multiple plans.

    PAR2QO: scattered print statements in rqo().
    Lynceus: structured summary with device breakdown.
    """
    _dbg(_T, "batch_cost_summary()")
    if plan_labels is None:
        plan_labels = [f"plan_{i}" for i in range(len(plans))]

    costs = []
    for i, plan_json in enumerate(plans):
        parsed = parse_plan_costs(plan_json, debug=False)
        costs.append(parsed["total_cost"])

    if not costs:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "std": 0}

    n = len(costs)
    mean = sum(costs) / n
    std = math.sqrt(sum((c - mean) ** 2 for c in costs) / max(n - 1, 1))
    best_idx = costs.index(min(costs))
    worst_idx = costs.index(max(costs))

    summary = {
        "count": n,
        "min": min(costs),
        "max": max(costs),
        "mean": mean,
        "std": std,
        "best_plan": plan_labels[best_idx],
        "worst_plan": plan_labels[worst_idx],
        "spread": max(costs) / max(min(costs), 1e-6),
    }

    if debug:
        print(f"\n  ┌─ COST SUMMARY ({n} plans) ─────────────────────────")
        print(f"  │  min={summary['min']:.1f} max={summary['max']:.1f} "
              f"mean={mean:.1f} σ={std:.1f}")
        print(f"  │  best={plan_labels[best_idx]} worst={plan_labels[worst_idx]} "
              f"spread={summary['spread']:.1f}×")
        for i, (label, cost) in enumerate(zip(plan_labels, costs)):
            bar = "█" * int(min(cost / max(max(costs), 1) * 30, 30))
            marker = " ★" if i == best_idx else ""
            print(f"  │  {label:>12s}: {cost:10.1f} {bar}{marker}")
        print(f"  └────────────────────────────────────────────────────")

    return summary


# ── Debug print helpers ────────────────────────────────────────────────

def debug_print_table(
    title: str,
    headers: List[str],
    rows: List[List[Any]],
    max_rows: int = 20,
):
    """Print a formatted debug table for state inspection."""
    _dbg(_T, "debug_print_table()")
    widths = [len(h) for h in headers]
    for row in rows[:max_rows]:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    sep = "  │  " + "─".join("─" * w for w in widths)
    print(f"\n  ┌─ {title} ─────────────────────────────────────────")

    # Header
    hdr = "  │  " + " ".join(f"{h:>{widths[i]}}" for i, h in enumerate(headers))
    print(hdr)
    print(sep)

    # Rows
    for r_idx, row in enumerate(rows):
        if r_idx >= max_rows:
            print(f"  │  ... ({len(rows) - max_rows} more rows)")
            break
        line = "  │  " + " ".join(
            f"{str(cell):>{widths[i]}}" for i, cell in enumerate(row)
        )
        print(line)

    print(f"  └────────────────────────────────────────────────────")


def debug_print_selectivities(
    base_sel: List[float],
    join_sel: List[float],
    label: str = "selectivities",
):
    """Print selectivity vectors for debugging."""
    _dbg(_T, "debug_print_selectivities()")
    print(f"\n  ┌─ {label} ──────────────────────────────────────────")
    print(f"  │  base ({len(base_sel)}):")
    for i, s in enumerate(base_sel):
        bar = "█" * int(min(s * 50, 50))
        print(f"  │    [{i:2d}] {s:.6f} {bar}")
    print(f"  │  join ({len(join_sel)}):")
    for j, s in enumerate(join_sel):
        bar = "█" * int(min(s * 50, 50))
        print(f"  │    [{j:2d}] {s:.6f} {bar}")
    print(f"  └────────────────────────────────────────────────────")
