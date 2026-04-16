#!/usr/bin/env python3
"""analyze_hlo_dump.py — agent-readable summary of an XLA_FLAGS=--xla_dump_to
directory.

Given <dir>/xla_dump/ populated by pf.setup_env(...), produces <dir>/hlo_summary.md
covering all four profiling categories:

    Memory        per-module peak live HBM; the 10 largest allocations overall
    Compute time  final fused HLO op counts + CUBLAS/CUDNN custom-calls
    Sharding      collective ops (all-gather, reduce-scatter, all-reduce,
                  collective-permute, all-to-all) with byte counts, plus every
                  "Involuntary full rematerialization" warning from XLA
    Compilation   module count + total on-disk size (proxy for compile effort)

Usage:
    python3 analyze_hlo_dump.py <artifacts_dir>
    python3 analyze_hlo_dump.py <artifacts_dir> --top 30 --module jit_gw_jax_main

The output is a Markdown file plus a hlo_summary.json for programmatic use.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


# ─────────────────────────────────────────────────────────────────────────
#  Filename routing
# ─────────────────────────────────────────────────────────────────────────

# XLA_FLAGS dump layout (post ~2024):
#   module_XXXX.<jit_name>.before_optimizations.txt
#   module_XXXX.<jit_name>.sm_XX.X_gpu_after_optimizations.txt
#   module_XXXX.<jit_name>.sm_XX.X_gpu_after_optimizations-memory-usage-report.txt
#   module_XXXX.<jit_name>.sm_XX.X_gpu_after_optimizations-buffer-assignment.txt
#   module_XXXX.<jit_name>.thunk_sequence.txt
#   module_XXXX.<jit_name>.ir-no-opt.ll / ir-with-opt.ll / ptx
#   module_XXXX.<jit_name>.autotune_results.pbtxt
#   module_XXXX.<jit_name>.gpu_target_config.pbtxt

RE_MODULE_NAME = re.compile(r"module_(\d+)\.([^.]+)\.")
RE_MEMORY_TOTAL = re.compile(r"^Total bytes used:\s+(\d+)")
RE_ALLOC = re.compile(
    r"(?P<cum>[\d.]+[KMG]?iB)\s*\(\s*\d+%\);\s+"
    r"(?P<total>[\d.]+[KMG]?iB|\s*0B);\s+allocation\s+\d+:\s+size\s+"
    r"(?P<size>[\d.]+[KMG]?iB|\s*0B),\s+(?P<rest>.*)"
)
RE_REMAT = re.compile(r"Involuntary full rematerialization.*", re.IGNORECASE)
RE_COLLECTIVE_OP = re.compile(
    r"^\s*%?[\w\-.]+\s+=\s+(?P<ty>[\w.()\[\],{} ]+)\s+"
    r"(?P<name>all-gather|all-reduce|reduce-scatter|collective-permute|"
    r"all-to-all|all-gather-start|collective-broadcast)"
    r"\((?P<args>.*?)\)"
)
RE_SHAPE_BYTES = re.compile(r"(?:[suf]|bf|c)(\d+)\[([\d,]+)\]")  # e.g. f32[64,64]


def _parse_bytes(s: str) -> int:
    """'1.50MiB' -> 1572864. '0B' -> 0. Accepts surrounding whitespace."""
    s = s.strip()
    if not s:
        return 0
    m = re.match(r"^([\d.]+)\s*(B|KiB|MiB|GiB|TiB)?$", s)
    if not m:
        return 0
    n = float(m.group(1))
    unit = m.group(2) or "B"
    scale = {"B": 1, "KiB": 1024, "MiB": 1024**2,
             "GiB": 1024**3, "TiB": 1024**4}[unit]
    return int(n * scale)


def _shape_bytes_from_type(type_str: str) -> int:
    """'f32[64,64,64]' -> number of bytes (ignoring batch, assuming contiguous)."""
    total = 0
    for dtype_bits, dims in RE_SHAPE_BYTES.findall(type_str):
        bits = int(dtype_bits)
        n_elem = 1
        for d in dims.split(","):
            d = d.strip()
            if d:
                n_elem *= int(d)
        total += n_elem * (bits // 8)
    return total


def _human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(x) < 1024.0:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} PiB"


# ─────────────────────────────────────────────────────────────────────────
#  Per-file parsers
# ─────────────────────────────────────────────────────────────────────────

def parse_memory_report(path: Path) -> dict:
    """Parse module_*.after_optimizations-memory-usage-report.txt."""
    text = path.read_text(errors="ignore")
    total = 0
    m = RE_MEMORY_TOTAL.search(text)
    if m:
        total = int(m.group(1))
    allocs: list[dict] = []
    for am in RE_ALLOC.finditer(text):
        allocs.append({
            "size": _parse_bytes(am.group("size")),
            "rest": am.group("rest").strip(),
        })
    allocs.sort(key=lambda a: a["size"], reverse=True)
    return {"path": str(path), "total_bytes": total, "allocations": allocs}


def parse_optimized_hlo(path: Path) -> dict:
    """Collect op counts + collective ops + remat warnings from the final HLO."""
    text = path.read_text(errors="ignore")
    op_counts: dict[str, int] = defaultdict(int)
    collectives: list[dict] = []
    custom_calls: list[dict] = []
    remats: list[str] = []

    # Look for custom-call targets: __cublas$gemm, __cudnn$..., cufft, etc.
    for m in re.finditer(r'custom_call_target="([^"]+)"', text):
        custom_calls.append({"target": m.group(1)})

    # Rematerialization warnings emitted as HLO comments
    for m in RE_REMAT.finditer(text):
        remats.append(m.group(0).strip())

    for line in text.splitlines():
        stripped = line.strip().lstrip("%")
        if "=" not in stripped:
            continue
        # Extract op name — the first token after ` = <shape> ` and before `(`
        # We look for patterns like "all-reduce(" after some prelude.
        for coll in ("all-gather", "reduce-scatter", "all-reduce",
                     "collective-permute", "all-to-all",
                     "all-gather-start", "collective-broadcast"):
            idx = stripped.find(coll + "(")
            if idx == -1:
                continue
            # Find the output shape/type just before the op name
            head = stripped[:idx].strip()
            after_eq = head.split("=", 1)[1].strip() if "=" in head else head
            op_type = after_eq  # e.g. "f32[8,16]{1,0}"
            nbytes = _shape_bytes_from_type(op_type)
            collectives.append({
                "op": coll,
                "output_type": op_type[:120],
                "output_bytes": nbytes,
            })
            op_counts[coll] += 1
            break
        else:
            # Track fusion/dot/convolution counts (compute-bound ops)
            for core in ("fusion", "dot", "convolution", "fft", "reduce",
                         "scatter", "gather", "concatenate", "reshape",
                         "transpose", "copy"):
                if re.search(rf"\b{core}\(", stripped):
                    op_counts[core] += 1
                    break

    return {
        "path": str(path),
        "op_counts": dict(op_counts),
        "collectives": collectives,
        "custom_calls": custom_calls,
        "remats": remats,
    }


def parse_buffer_assignment(path: Path) -> dict:
    """Parse buffer-assignment for peak live HBM + rematerialization details."""
    text = path.read_text(errors="ignore")
    remats = [m.group(0).strip() for m in RE_REMAT.finditer(text)]
    # Peak live memory: search for "Total bytes used: N"
    total = 0
    m = RE_MEMORY_TOTAL.search(text)
    if m:
        total = int(m.group(1))
    return {"path": str(path), "peak_bytes": total, "remats": remats}


# ─────────────────────────────────────────────────────────────────────────
#  Top-level driver
# ─────────────────────────────────────────────────────────────────────────

def scan(dump_dir: Path) -> dict:
    """Walk a dump dir and collect per-module + aggregate stats."""
    modules: dict[str, dict] = defaultdict(lambda: {
        "files": [], "memory": None, "hlo": None, "buffer": None,
    })

    for fp in sorted(dump_dir.iterdir()):
        if not fp.is_file():
            continue
        name = fp.name
        m = RE_MODULE_NAME.match(name)
        if not m:
            continue
        mod_id = f"module_{m.group(1)}.{m.group(2)}"
        modules[mod_id]["files"].append(name)

        if name.endswith("-memory-usage-report.txt"):
            modules[mod_id]["memory"] = parse_memory_report(fp)
        elif name.endswith("-buffer-assignment.txt"):
            modules[mod_id]["buffer"] = parse_buffer_assignment(fp)
        elif name.endswith("_gpu_after_optimizations.txt"):
            modules[mod_id]["hlo"] = parse_optimized_hlo(fp)
        elif name.endswith("after_optimizations.txt"):
            # Non-GPU dump fallback (CPU backend etc.)
            modules[mod_id]["hlo"] = parse_optimized_hlo(fp)

    # Aggregate
    total_modules = len(modules)
    total_peak_bytes = sum((v["memory"] or {}).get("total_bytes", 0)
                           for v in modules.values())
    agg_collectives: list[dict] = []
    agg_remats: list[dict] = []
    agg_custom_calls: dict[str, int] = defaultdict(int)
    agg_op_counts: dict[str, int] = defaultdict(int)
    for mod_id, v in modules.items():
        hlo = v.get("hlo") or {}
        for c in hlo.get("collectives", []):
            agg_collectives.append({"module": mod_id, **c})
        for r in hlo.get("remats", []):
            agg_remats.append({"module": mod_id, "warning": r})
        for cc in hlo.get("custom_calls", []):
            agg_custom_calls[cc["target"]] += 1
        for k, n in (hlo.get("op_counts") or {}).items():
            agg_op_counts[k] += n
        for r in (v.get("buffer") or {}).get("remats", []):
            agg_remats.append({"module": mod_id, "warning": r})

    return {
        "dump_dir": str(dump_dir),
        "total_modules": total_modules,
        "total_peak_bytes": total_peak_bytes,
        "agg_op_counts": dict(agg_op_counts),
        "agg_custom_calls": dict(agg_custom_calls),
        "agg_collectives": agg_collectives,
        "agg_remats": agg_remats,
        "modules": dict(modules),
    }


def render_markdown(summary: dict, top_n: int = 20) -> str:
    lines: list[str] = []
    lines.append(f"# HLO dump summary")
    lines.append("")
    lines.append(f"**Dump dir:** `{summary['dump_dir']}`")
    lines.append(f"**Modules dumped:** {summary['total_modules']}")
    lines.append(f"**Sum of per-module peak live HBM:** "
                 f"{_human_bytes(summary['total_peak_bytes'])} "
                 f"(upper bound; peaks happen at different times)")
    lines.append("")

    # ─────── Memory ───────
    lines.append("## Memory — largest modules by peak HBM")
    lines.append("")
    lines.append("| Module | Peak HBM | Top allocation |")
    lines.append("|---|---:|---|")
    items = sorted(summary["modules"].items(),
                   key=lambda kv: (kv[1]["memory"] or {}).get("total_bytes", 0),
                   reverse=True)
    for mod_id, v in items[:top_n]:
        mem = v.get("memory") or {}
        total = mem.get("total_bytes", 0)
        top_alloc = (mem.get("allocations") or [{"size": 0, "rest": ""}])[0]
        rest = top_alloc["rest"][:120].replace("|", "\\|")
        lines.append(f"| `{mod_id}` | {_human_bytes(total)} | "
                     f"{_human_bytes(top_alloc['size'])} — {rest} |")
    lines.append("")

    # ─────── Compute ───────
    lines.append("## Compute — aggregate op counts")
    lines.append("")
    op_items = sorted(summary["agg_op_counts"].items(),
                      key=lambda kv: kv[1], reverse=True)
    lines.append("| Op | Count |")
    lines.append("|---|---:|")
    for k, n in op_items:
        lines.append(f"| `{k}` | {n} |")
    lines.append("")

    if summary["agg_custom_calls"]:
        lines.append("### Custom calls (cuBLAS / cuDNN / cuFFT / etc.)")
        lines.append("")
        lines.append("| Target | Count |")
        lines.append("|---|---:|")
        for k, n in sorted(summary["agg_custom_calls"].items(),
                           key=lambda kv: kv[1], reverse=True):
            lines.append(f"| `{k}` | {n} |")
        lines.append("")

    # ─────── Sharding / communication ───────
    lines.append("## Sharding — collectives (largest by output bytes)")
    lines.append("")
    colls = sorted(summary["agg_collectives"],
                   key=lambda c: c.get("output_bytes", 0), reverse=True)
    if not colls:
        lines.append("_No collective ops found (single-device or pure-SPMD-free)._")
    else:
        lines.append("| Module | Op | Output bytes | Output type |")
        lines.append("|---|---|---:|---|")
        for c in colls[:top_n]:
            out_ty = c["output_type"][:80].replace("|", "\\|")
            lines.append(f"| `{c['module']}` | `{c['op']}` | "
                         f"{_human_bytes(c['output_bytes'])} | `{out_ty}` |")
    lines.append("")

    # ─────── Remats ───────
    lines.append("## Rematerialization warnings")
    lines.append("")
    if not summary["agg_remats"]:
        lines.append("_None._ 🎉")
    else:
        for r in summary["agg_remats"][:50]:
            w = r["warning"][:300].replace("\n", " ")
            lines.append(f"- **`{r['module']}`**: {w}")
    lines.append("")

    # ─────── Retrace groups ───────
    # Group module_XXXX.<fn> → <fn> so we can count how many module IDs each
    # jit function produced. Multiple module_IDs for the same function
    # directly signals shape polymorphism / argument churn, which is the
    # single most common compile-time waste in LORRAX.
    by_fn: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for mod_id, v in summary["modules"].items():
        # Strip leading "module_XXXX." to leave "jit_<name>"
        parts = mod_id.split(".", 1)
        fn = parts[1] if len(parts) == 2 else mod_id
        peak = (v.get("memory") or {}).get("total_bytes", 0)
        by_fn[fn].append((mod_id, peak))

    retrace_rows = sorted(
        (
            (fn, len(mods), max(p for _, p in mods), sum(p for _, p in mods))
            for fn, mods in by_fn.items()
        ),
        key=lambda r: (r[1], r[2]), reverse=True,
    )
    lines.append("## Retrace groups — jit() name → module count")
    lines.append("")
    lines.append("_Multiple modules per jit name mean XLA recompiled from "
                 "scratch under a new signature. >2 is worth investigating; "
                 ">5 is almost always a shape-polymorphism bug._")
    lines.append("")
    lines.append("| jit fn | #modules | max peak | Σ peak |")
    lines.append("|---|---:|---:|---:|")
    for fn, n_mods, mx_peak, sum_peak in retrace_rows[:top_n]:
        lines.append(f"| `{fn}` | {n_mods} | {_human_bytes(mx_peak)} | "
                     f"{_human_bytes(sum_peak)} |")
    lines.append("")

    # ─────── Files ───────
    lines.append("## Per-module file index")
    lines.append("")
    lines.append("_(for deeper inspection — each module writes up to 10 "
                 "files; the memory-usage-report is the most agent-readable)_")
    lines.append("")
    lines.append("| Module | Files |")
    lines.append("|---|---|")
    for mod_id, v in items[:top_n]:
        flist = ", ".join(v["files"][:3])
        if len(v["files"]) > 3:
            flist += f", … (+{len(v['files'])-3})"
        lines.append(f"| `{mod_id}` | {flist} |")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("artifacts_dir",
                    help="Path containing xla_dump/ (produced by pf.setup_env)")
    ap.add_argument("--top", type=int, default=20,
                    help="How many rows per table (default 20)")
    ap.add_argument("--out-md", default=None,
                    help="Output markdown path (default <dir>/hlo_summary.md)")
    ap.add_argument("--out-json", default=None,
                    help="Output JSON path (default <dir>/hlo_summary.json)")
    args = ap.parse_args()

    root = Path(args.artifacts_dir).resolve()
    dump = root / "xla_dump" if (root / "xla_dump").is_dir() else root
    if not dump.is_dir():
        print(f"[analyze_hlo_dump] no dump dir at {dump}", file=sys.stderr)
        return 1

    summary = scan(dump)
    md = render_markdown(summary, top_n=args.top)
    out_md = Path(args.out_md) if args.out_md else (root / "hlo_summary.md")
    out_json = Path(args.out_json) if args.out_json else (root / "hlo_summary.json")
    out_md.write_text(md)

    def _sanitize(o):
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items() if k != "files"}
        if isinstance(o, list):
            return [_sanitize(x) for x in o[:100]]
        return o
    out_json.write_text(json.dumps(_sanitize(summary), indent=2, default=str))

    print(f"[analyze_hlo_dump] {summary['total_modules']} modules, "
          f"peak-sum={_human_bytes(summary['total_peak_bytes'])}, "
          f"{len(summary['agg_collectives'])} collectives, "
          f"{len(summary['agg_remats'])} remat warnings")
    print(f"[analyze_hlo_dump] wrote {out_md}")
    print(f"[analyze_hlo_dump] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
