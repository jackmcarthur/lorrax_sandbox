#!/usr/bin/env python3
"""analyze_hlo_dump.py — agent-readable summary of an XLA_FLAGS=--xla_dump_to
directory.

Produces:

    <artifacts>/hlo_summary.md      Ranked tables: Memory, Sharding, Rematerialization,
                                     Retrace groups, Custom calls. Every row carries a
                                     source_file:line when XLA metadata preserved one.
    <artifacts>/hlo_summary.json    Same data, machine-readable.

    <artifacts>/memory_details.txt  Top-N modules' memory-usage-report concatenated with
                                     a one-line header each — the densest "what's eating
                                     memory here" view possible from static HLO.
    <artifacts>/collectives_details.txt   Each top collective with ~10 lines of HLO
                                     context (producing op, source_file:line) pulled
                                     from the optimized HLO.
    <artifacts>/remat_details.txt   Every rematerialization warning with 5 lines of
                                     surrounding HLO and its source_file:line.
    <artifacts>/retrace_details.txt Per jit name, the input signatures (arg types)
                                     that caused each new compiled module.

Usage:
    python3 analyze_hlo_dump.py <artifacts_dir> [--top 20]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


# ─────────────────────────────────────────────────────────────────────────
#  File routing
# ─────────────────────────────────────────────────────────────────────────

# XLA_FLAGS dump layout (post ~2024):
#   module_XXXX.<jit_name>.before_optimizations.txt
#   module_XXXX.<jit_name>.sm_XX.X_gpu_after_optimizations.txt
#   module_XXXX.<jit_name>.sm_XX.X_gpu_after_optimizations-memory-usage-report.txt
#   module_XXXX.<jit_name>.sm_XX.X_gpu_after_optimizations-buffer-assignment.txt
#   module_XXXX.<jit_name>.thunk_sequence.txt

RE_MODULE_NAME = re.compile(r"module_(\d+)\.([^.]+)\.")
RE_MEMORY_TOTAL = re.compile(r"^Total bytes used:\s+(\d+)")
RE_ALLOC = re.compile(
    r"(?P<cum>[\d.]+[KMG]?iB)\s*\(\s*\d+%\);\s+"
    r"(?P<total>[\d.]+[KMG]?iB|\s*0B);\s+allocation\s+\d+:\s+size\s+"
    r"(?P<size>[\d.]+[KMG]?iB|\s*0B),\s+(?P<rest>.*)"
)
RE_REMAT = re.compile(r"Involuntary full rematerialization.*", re.IGNORECASE)
# HLO metadata carrying the Python source that emitted the op, e.g.:
#   metadata={op_name="jit(foo)/jit(main)/..." source_file="w_isdf.py" source_line=324 ...}
RE_META_SOURCE = re.compile(
    r'source_file="(?P<file>[^"]+)"\s+source_line=(?P<line>\d+)'
)
RE_META_OPNAME = re.compile(r'op_name="([^"]+)"')
RE_SHAPE_BYTES = re.compile(r"(?:[suf]|bf|c)(\d+)\[([\d,]+)\]")
# Input type signature inside the before_optimizations.txt:
#   ENTRY %main.4 (Arg_0.1.0: f32[8,8]) -> ...
RE_ENTRY_SIG = re.compile(r"ENTRY\s+%[^\s]+\s*\((?P<sig>[^)]*)\)")


# ─────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────

def _parse_bytes(s: str) -> int:
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


def _shape_bytes(type_str: str) -> int:
    total = 0
    for dtype_bits, dims in RE_SHAPE_BYTES.findall(type_str):
        bits = int(dtype_bits)
        n = 1
        for d in dims.split(","):
            d = d.strip()
            if d:
                n *= int(d)
        total += n * (bits // 8)
    return total


def _hb(n: int | float | None) -> str:
    if n is None:
        return "-"
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(x) < 1024.0:
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} PiB"


def _short_source(file: str, line: str | int) -> str:
    """Shorten absolute path to something like .../w_isdf.py:324."""
    if not file:
        return ""
    p = file
    if "/sources/lorrax/src/" in p:
        p = p.split("/sources/lorrax/src/", 1)[1]
    elif "/lorrax/src/" in p:
        p = p.split("/lorrax/src/", 1)[1]
    return f"{p}:{line}"


def _extract_source(meta_text: str) -> tuple[str, str]:
    """Pull the first (file, line) pair out of an HLO metadata block."""
    m = RE_META_SOURCE.search(meta_text)
    if not m:
        return "", ""
    return m.group("file"), m.group("line")


# ─────────────────────────────────────────────────────────────────────────
#  Per-file parsers
# ─────────────────────────────────────────────────────────────────────────

def parse_memory_report(path: Path) -> dict:
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
    return {"path": str(path), "total_bytes": total, "allocations": allocs,
            "raw": text}


def parse_optimized_hlo(path: Path) -> dict:
    """Walk the optimized HLO once; extract collectives with source metadata,
    remat warnings with context, custom-call counts.

    Each line in the optimized HLO looks like:
        %all-gather.42 = c128[...] all-gather(%foo), ..., metadata={...}
    The metadata block is usually on the same line, delimited by {}.
    """
    lines = path.read_text(errors="ignore").splitlines()
    n = len(lines)
    collectives: list[dict] = []
    custom_calls: dict[str, int] = defaultdict(int)
    remats: list[dict] = []

    coll_names = ("all-gather-start", "all-gather", "reduce-scatter",
                  "all-reduce", "collective-permute",
                  "all-to-all", "collective-broadcast")

    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("%")
        # Collectives
        for cn in coll_names:
            idx = stripped.find(cn + "(")
            if idx == -1:
                continue
            head = stripped[:idx].strip()
            out_ty = head.split("=", 1)[1].strip() if "=" in head else head
            source_file, source_line = _extract_source(line)
            src = _short_source(source_file, source_line)
            collectives.append({
                "line_no": i + 1,
                "op": cn,
                "output_type": out_ty[:160],
                "output_bytes": _shape_bytes(out_ty),
                "source": src,
                "raw": line.rstrip(),
            })
            break
        # Custom calls
        for m in re.finditer(r'custom_call_target="([^"]+)"', line):
            custom_calls[m.group(1)] += 1
        # Remat warnings
        if RE_REMAT.search(line):
            # Grab 5 lines of context on each side
            ctx_before = lines[max(0, i-5):i]
            ctx_after = lines[i+1:min(n, i+6)]
            # Look for source metadata in the warning itself
            source_file, source_line = _extract_source(line)
            # Some warnings reference a source_file via HLO metadata format
            if not source_file:
                m = re.search(r'source_file="([^"]+)".*?source_line=(\d+)', line)
                if m:
                    source_file, source_line = m.group(1), m.group(2)
            remats.append({
                "line_no": i + 1,
                "warning": line.rstrip(),
                "source": _short_source(source_file, source_line),
                "context_before": [c.rstrip() for c in ctx_before],
                "context_after": [c.rstrip() for c in ctx_after],
            })

    return {
        "path": str(path),
        "collectives": collectives,
        "custom_calls": dict(custom_calls),
        "remats": remats,
        "raw_lines": lines,
    }


def parse_before_optimizations(path: Path) -> dict:
    """Pull the input type signature from the ENTRY line. Used for retrace
    attribution — each module's ENTRY sig shows the exact argument shapes."""
    text = path.read_text(errors="ignore")
    m = RE_ENTRY_SIG.search(text)
    sig = m.group("sig") if m else ""
    return {"path": str(path), "entry_sig": sig[:400]}


def parse_buffer_assignment(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    return {"path": str(path),
            "remats": [m.group(0).strip() for m in RE_REMAT.finditer(text)]}


# ─────────────────────────────────────────────────────────────────────────
#  Scan
# ─────────────────────────────────────────────────────────────────────────

def scan(dump_dir: Path) -> dict:
    modules: dict[str, dict] = defaultdict(lambda: {
        "files": [], "memory": None, "hlo": None, "buffer": None, "sig": None,
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
            modules[mod_id]["hlo"] = parse_optimized_hlo(fp)
        elif name.endswith(".before_optimizations.txt"):
            modules[mod_id]["sig"] = parse_before_optimizations(fp)

    total_modules = len(modules)
    total_peak_bytes = sum((v["memory"] or {}).get("total_bytes", 0)
                           for v in modules.values())
    agg_collectives: list[dict] = []
    agg_remats: list[dict] = []
    agg_custom: dict[str, int] = defaultdict(int)
    for mod_id, v in modules.items():
        hlo = v.get("hlo") or {}
        for c in hlo.get("collectives", []):
            agg_collectives.append({"module": mod_id, **c})
        for r in hlo.get("remats", []):
            agg_remats.append({"module": mod_id, **r})
        for cc, n in hlo.get("custom_calls", {}).items():
            agg_custom[cc] += n
        # Also catch remats dumped into buffer-assignment.txt
        for rw in (v.get("buffer") or {}).get("remats", []):
            agg_remats.append({"module": mod_id, "warning": rw, "source": "",
                               "context_before": [], "context_after": []})

    return {
        "dump_dir": str(dump_dir),
        "total_modules": total_modules,
        "total_peak_bytes": total_peak_bytes,
        "agg_custom_calls": dict(agg_custom),
        "agg_collectives": agg_collectives,
        "agg_remats": agg_remats,
        "modules": dict(modules),
    }


# ─────────────────────────────────────────────────────────────────────────
#  Markdown rendering
# ─────────────────────────────────────────────────────────────────────────

def render_markdown(summary: dict, top_n: int = 20,
                    details_link_base: str = "") -> str:
    L: list[str] = []
    L.append(f"# HLO dump summary")
    L.append("")
    L.append(f"**Dump dir:** `{summary['dump_dir']}`")
    L.append(f"**Modules dumped:** {summary['total_modules']}")
    L.append(f"**Sum of per-module peak live HBM:** "
             f"{_hb(summary['total_peak_bytes'])} "
             f"(upper bound; peaks occur at different times)")
    L.append("")
    L.append("_Companion files with richer context:_")
    L.append(f"- [`memory_details.txt`]({details_link_base}memory_details.txt) — "
             "top-N modules' memory-usage-report, concatenated")
    L.append(f"- [`collectives_details.txt`]({details_link_base}collectives_details.txt) — "
             "HLO context around each collective + source_file:line")
    L.append(f"- [`remat_details.txt`]({details_link_base}remat_details.txt) — "
             "every remat warning + nearby HLO lines")
    L.append(f"- [`retrace_details.txt`]({details_link_base}retrace_details.txt) — "
             "input signatures that caused each retrace")
    L.append("")

    # ─────── Memory ───────
    L.append("## Memory — largest modules by peak HBM")
    L.append("")
    L.append("| Module | Peak HBM | Top allocation |")
    L.append("|---|---:|---|")
    items = sorted(summary["modules"].items(),
                   key=lambda kv: (kv[1]["memory"] or {}).get("total_bytes", 0),
                   reverse=True)
    for mod_id, v in items[:top_n]:
        mem = v.get("memory") or {}
        total = mem.get("total_bytes", 0)
        top = (mem.get("allocations") or [{"size": 0, "rest": ""}])[0]
        rest = top["rest"][:120].replace("|", "\\|")
        L.append(f"| `{mod_id}` | {_hb(total)} | "
                 f"{_hb(top['size'])} — {rest} |")
    L.append("")

    # ─────── Sharding / collectives ───────
    L.append("## Sharding — collectives (largest by output bytes)")
    L.append("")
    colls = sorted(summary["agg_collectives"],
                   key=lambda c: c.get("output_bytes", 0), reverse=True)
    if not colls:
        L.append("_No collective ops found (single-device or pure-SPMD-free)._")
    else:
        L.append("| Module | Op | Output bytes | Source | Output type |")
        L.append("|---|---|---:|---|---|")
        for c in colls[:top_n]:
            out_ty = c["output_type"][:60].replace("|", "\\|")
            src = c.get("source", "").replace("|", "\\|")
            L.append(f"| `{c['module']}` | `{c['op']}` | "
                     f"{_hb(c['output_bytes'])} | "
                     f"`{src}` | `{out_ty}` |")
    L.append("")

    # ─────── Rematerialization ───────
    L.append("## Rematerialization warnings")
    L.append("")
    if not summary["agg_remats"]:
        L.append("_None._")
    else:
        L.append("| Module | Source | Warning |")
        L.append("|---|---|---|")
        for r in summary["agg_remats"][:50]:
            w = r["warning"][:240].replace("\n", " ").replace("|", "\\|")
            src = r.get("source", "") or "_(no metadata)_"
            L.append(f"| `{r['module']}` | `{src}` | {w} |")
    L.append("")

    # ─────── Retrace groups ───────
    by_fn: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for mod_id, v in summary["modules"].items():
        parts = mod_id.split(".", 1)
        fn = parts[1] if len(parts) == 2 else mod_id
        peak = (v.get("memory") or {}).get("total_bytes", 0)
        by_fn[fn].append((mod_id, peak))

    retrace_rows = sorted(
        ((fn, len(mods), max(p for _, p in mods), sum(p for _, p in mods))
         for fn, mods in by_fn.items()),
        key=lambda r: (r[1], r[2]), reverse=True,
    )
    L.append("## Retrace groups — jit() name → module count")
    L.append("")
    L.append("_More than 2 modules for the same jit name means XLA recompiled. "
             "Anything above 5 is almost always shape polymorphism — see "
             "`retrace_details.txt` for the signatures._")
    L.append("")
    L.append("| jit fn | #modules | max peak | Σ peak |")
    L.append("|---|---:|---:|---:|")
    for fn, n_mods, mx, sm in retrace_rows[:top_n]:
        if n_mods < 2 and len(retrace_rows) > top_n:
            break
        L.append(f"| `{fn}` | {n_mods} | {_hb(mx)} | {_hb(sm)} |")
    L.append("")

    # ─────── Custom calls ───────
    if summary["agg_custom_calls"]:
        L.append("## Custom calls (cuBLAS / cuDNN / cuFFT / etc.)")
        L.append("")
        L.append("| Target | Count |")
        L.append("|---|---:|")
        for k, n in sorted(summary["agg_custom_calls"].items(),
                           key=lambda kv: kv[1], reverse=True):
            L.append(f"| `{k}` | {n} |")
        L.append("")

    return "\n".join(L) + "\n"


# ─────────────────────────────────────────────────────────────────────────
#  Detail txt writers
# ─────────────────────────────────────────────────────────────────────────

def write_memory_details(summary: dict, out_path: Path, top_n: int) -> None:
    items = sorted(summary["modules"].items(),
                   key=lambda kv: (kv[1]["memory"] or {}).get("total_bytes", 0),
                   reverse=True)[:top_n]
    parts: list[str] = []
    parts.append("# memory_details.txt — top-{0} modules by peak HBM".format(top_n))
    parts.append("# Each section: memory-usage-report.txt verbatim (allocations")
    parts.append("# sorted by size, smaller boilerplate trimmed).")
    parts.append("")
    for mod_id, v in items:
        mem = v.get("memory") or {}
        if not mem:
            continue
        parts.append(f"────────────────────────────────────────────────────────────")
        parts.append(f"[{mod_id}]   peak HBM = {_hb(mem.get('total_bytes', 0))}")
        parts.append(f"────────────────────────────────────────────────────────────")
        raw = mem.get("raw", "")
        # Drop the noisy "Allocations sorted by size with their values:" footer
        cut = raw.find("Allocations sorted by size with their values:")
        if cut > 0:
            raw = raw[:cut].rstrip()
        parts.append(raw)
        parts.append("")
    out_path.write_text("\n".join(parts) + "\n")


def write_collectives_details(summary: dict, out_path: Path, top_n: int) -> None:
    colls = sorted(summary["agg_collectives"],
                   key=lambda c: c.get("output_bytes", 0), reverse=True)[:top_n]
    # For each collective, pull the HLO context (3 lines before, 3 after)
    lines_of: dict[str, list[str]] = {}
    for mod_id, v in summary["modules"].items():
        hlo = v.get("hlo") or {}
        lines_of[mod_id] = hlo.get("raw_lines", [])

    parts: list[str] = []
    parts.append("# collectives_details.txt — top-{0} collectives by output bytes".format(top_n))
    parts.append("# Each block: HLO line of the collective ± 3 lines of context")
    parts.append("# plus the Python source_file:line extracted from metadata.")
    parts.append("")
    for c in colls:
        mod_id = c["module"]
        src = c.get("source") or "(no source metadata)"
        parts.append(f"────────────────────────────────────────────────────────────")
        parts.append(f"[{mod_id}]  op={c['op']}  bytes={_hb(c['output_bytes'])}  src={src}")
        parts.append(f"────────────────────────────────────────────────────────────")
        L = lines_of.get(mod_id, [])
        idx = c["line_no"] - 1
        lo = max(0, idx - 3)
        hi = min(len(L), idx + 4)
        for i in range(lo, hi):
            marker = ">>" if i == idx else "  "
            # Strip metadata={...} block to keep line short
            ln = L[i].rstrip()
            ln = re.sub(r",\s*metadata=\{[^}]*\}", "", ln)
            ln = re.sub(r",\s*frontend_attributes=\{[^}]*\}", "", ln)
            parts.append(f"{marker} {ln}")
        parts.append("")
    out_path.write_text("\n".join(parts) + "\n")


def write_remat_details(summary: dict, out_path: Path) -> None:
    remats = summary["agg_remats"]
    parts: list[str] = []
    parts.append("# remat_details.txt — every involuntary rematerialization warning")
    parts.append("# Each block: warning + 5 lines before + 5 after from its HLO file")
    parts.append("")
    if not remats:
        parts.append("_No rematerialization warnings emitted — nothing to show._")
    for r in remats:
        parts.append(f"────────────────────────────────────────────────────────────")
        parts.append(f"[{r['module']}]  src={r.get('source','(none)')}")
        parts.append(f"────────────────────────────────────────────────────────────")
        parts.append(f">> {r['warning']}")
        if r.get("context_before") or r.get("context_after"):
            parts.append("   --- context before ---")
            for c in r.get("context_before", []):
                parts.append(f"   {c}")
            parts.append("   --- context after ---")
            for c in r.get("context_after", []):
                parts.append(f"   {c}")
        parts.append("")
    out_path.write_text("\n".join(parts) + "\n")


def write_retrace_details(summary: dict, out_path: Path, top_n: int) -> None:
    """For each jit name with >1 module, list the entry signatures of each
    module instance. Exposes shape polymorphism at a glance."""
    by_fn: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for mod_id, v in summary["modules"].items():
        parts = mod_id.split(".", 1)
        fn = parts[1] if len(parts) == 2 else mod_id
        peak = (v.get("memory") or {}).get("total_bytes", 0)
        sig = (v.get("sig") or {}).get("entry_sig", "")
        by_fn[fn].append((mod_id, peak, sig))

    rows = sorted(
        ((fn, mods) for fn, mods in by_fn.items() if len(mods) >= 2),
        key=lambda r: len(r[1]), reverse=True,
    )
    out: list[str] = []
    out.append("# retrace_details.txt — input signatures per jit name")
    out.append("# Each block: one jit name that retraced, with the ENTRY type")
    out.append("# signature for each compiled module instance.")
    out.append("#")
    out.append("# Diffing the signatures within a block reveals which arg shape")
    out.append("# changed between calls — the root cause of the retrace.")
    out.append("")
    for fn, mods in rows[:top_n]:
        mods_sorted = sorted(mods, key=lambda t: t[1], reverse=True)
        out.append(f"────────────────────────────────────────────────────────────")
        out.append(f"[{fn}]  {len(mods_sorted)} modules")
        out.append(f"────────────────────────────────────────────────────────────")
        for mid, peak, sig in mods_sorted:
            out.append(f"  {mid}  peak={_hb(peak):>10s}")
            if sig:
                out.append(f"    sig: {sig}")
        out.append("")
    if not rows:
        out.append("_No function retraced more than once._ 🎉")
    out_path.write_text("\n".join(out) + "\n")


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("artifacts_dir",
                    help="Path containing xla_dump/ (produced by pf.setup_env)")
    ap.add_argument("--top", type=int, default=20,
                    help="How many rows per table (default 20)")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    root = Path(args.artifacts_dir).resolve()
    dump = root / "xla_dump" if (root / "xla_dump").is_dir() else root
    if not dump.is_dir():
        print(f"[analyze_hlo_dump] no dump dir at {dump}", file=sys.stderr)
        return 1

    summary = scan(dump)

    # Core summary
    md = render_markdown(summary, top_n=args.top)
    out_md = Path(args.out_md) if args.out_md else (root / "hlo_summary.md")
    out_json = Path(args.out_json) if args.out_json else (root / "hlo_summary.json")
    out_md.write_text(md)

    # JSON (drop the 'raw_lines' and 'raw' heavy fields for size sanity)
    def _sanitize(o):
        if isinstance(o, dict):
            return {k: _sanitize(v) for k, v in o.items()
                    if k not in ("raw_lines", "raw", "files")}
        if isinstance(o, list):
            return [_sanitize(x) for x in o[:500]]
        return o
    out_json.write_text(json.dumps(_sanitize(summary), indent=2, default=str))

    # Detail txts (siblings of hlo_summary.md)
    write_memory_details(summary, root / "memory_details.txt", args.top)
    write_collectives_details(summary, root / "collectives_details.txt", args.top)
    write_remat_details(summary, root / "remat_details.txt")
    write_retrace_details(summary, root / "retrace_details.txt", args.top)

    print(f"[analyze_hlo_dump] {summary['total_modules']} modules, "
          f"peak-sum={_hb(summary['total_peak_bytes'])}, "
          f"{len(summary['agg_collectives'])} collectives, "
          f"{len(summary['agg_remats'])} remat warnings")
    print(f"[analyze_hlo_dump] wrote {out_md}")
    print(f"[analyze_hlo_dump] wrote {out_json}")
    print(f"[analyze_hlo_dump] wrote {root/'memory_details.txt'}")
    print(f"[analyze_hlo_dump] wrote {root/'collectives_details.txt'}")
    print(f"[analyze_hlo_dump] wrote {root/'remat_details.txt'}")
    print(f"[analyze_hlo_dump] wrote {root/'retrace_details.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
