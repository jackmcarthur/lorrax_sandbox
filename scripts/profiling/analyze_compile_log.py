#!/usr/bin/env python3
"""analyze_compile_log.py — agent-readable summary of a JAX compile log.

Consumes the file written by pf.attach_compile_log() (or a captured
JAX_LOG_COMPILES stderr). Produces:

  - Top-N slowest XLA compilations
  - Total compile time by function
  - Every TRACING CACHE MISS with its "because:" reason
  - Count of pjit tracing events

Usage:
    python3 analyze_compile_log.py <artifacts_dir>/compile.log
    python3 analyze_compile_log.py <artifacts_dir>   # picks up compile.log inside
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


RE_XLA_COMPILE = re.compile(
    r"Finished XLA compilation of jit\((?P<name>[^)]+)\) in (?P<secs>[\d.]+) sec"
)
RE_MLIR_CONV = re.compile(
    r"Finished jaxpr to MLIR module conversion jit\((?P<name>[^)]+)\) in "
    r"(?P<secs>[\d.]+) sec"
)
RE_TRACE = re.compile(
    r"Finished tracing \+ transforming (?P<name>\S+) for pjit in "
    r"(?P<secs>[\d.]+) sec"
)
RE_CACHE_MISS = re.compile(r"TRACING CACHE MISS at (?P<loc>\S+) .*? because:")
RE_PERSISTENT_MISS = re.compile(
    r"Persistent compilation cache miss for jit\((?P<name>[^)]+)\).*",
    re.IGNORECASE,
)


def parse(log_path: Path) -> dict:
    text = log_path.read_text(errors="ignore")

    xla: dict[str, list[float]] = defaultdict(list)
    mlir: dict[str, list[float]] = defaultdict(list)
    trace: dict[str, list[float]] = defaultdict(list)
    cache_misses: list[dict] = []
    persistent_misses: list[str] = []

    # Grab "because:" reasons, which span multiple lines
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = RE_XLA_COMPILE.search(line)
        if m:
            xla[m.group("name")].append(float(m.group("secs")))
            continue
        m = RE_MLIR_CONV.search(line)
        if m:
            mlir[m.group("name")].append(float(m.group("secs")))
            continue
        m = RE_TRACE.search(line)
        if m:
            trace[m.group("name")].append(float(m.group("secs")))
            continue
        m = RE_CACHE_MISS.search(line)
        if m:
            # Collect the "because:" block (up to 6 following lines
            # until the next WARNING line)
            reason: list[str] = []
            for j in range(i + 1, min(i + 8, len(lines))):
                nxt = lines[j].rstrip()
                if not nxt or nxt.startswith(("WARNING", "2026", "2025")):
                    break
                reason.append(nxt)
            cache_misses.append({
                "location": m.group("loc"),
                "reason": " ".join(r.strip() for r in reason)[:400],
            })
            continue
        m = RE_PERSISTENT_MISS.search(line)
        if m:
            persistent_misses.append(m.group("name"))

    return {
        "log_path": str(log_path),
        "xla": xla,
        "mlir": mlir,
        "trace": trace,
        "cache_misses": cache_misses,
        "persistent_misses": persistent_misses,
    }


def render(summary: dict, top_n: int = 30) -> str:
    lines: list[str] = []
    lines.append("# Compilation log summary")
    lines.append("")
    lines.append(f"**Log:** `{summary['log_path']}`")
    lines.append("")

    def _totals(d: dict[str, list[float]]) -> list[tuple[str, int, float, float]]:
        rows = []
        for name, secs in d.items():
            rows.append((name, len(secs), sum(secs), max(secs)))
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows

    xla_rows = _totals(summary["xla"])
    mlir_rows = _totals(summary["mlir"])
    trace_rows = _totals(summary["trace"])

    total_xla = sum(r[2] for r in xla_rows)
    total_mlir = sum(r[2] for r in mlir_rows)
    total_trace = sum(r[2] for r in trace_rows)

    lines.append("## Wall-clock totals across the run")
    lines.append("")
    lines.append("| Stage | Count | Total seconds | Max single |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| trace+transform | {sum(r[1] for r in trace_rows)} | "
                 f"{total_trace:.3f} | "
                 f"{max((r[3] for r in trace_rows), default=0):.3f} |")
    lines.append(f"| jaxpr→MLIR | {sum(r[1] for r in mlir_rows)} | "
                 f"{total_mlir:.3f} | "
                 f"{max((r[3] for r in mlir_rows), default=0):.3f} |")
    lines.append(f"| XLA compile | {sum(r[1] for r in xla_rows)} | "
                 f"{total_xla:.3f} | "
                 f"{max((r[3] for r in xla_rows), default=0):.3f} |")
    lines.append("")

    lines.append(f"## Top {top_n} XLA compilations by total time")
    lines.append("")
    lines.append("| jit() name | Count | Total s | Max s |")
    lines.append("|---|---:|---:|---:|")
    for name, n, tot, mx in xla_rows[:top_n]:
        lines.append(f"| `{name}` | {n} | {tot:.3f} | {mx:.3f} |")
    lines.append("")

    lines.append(f"## Top {top_n} pjit trace+transform by total time")
    lines.append("")
    lines.append("| function | Count | Total s | Max s |")
    lines.append("|---|---:|---:|---:|")
    for name, n, tot, mx in trace_rows[:top_n]:
        lines.append(f"| `{name}` | {n} | {tot:.3f} | {mx:.3f} |")
    lines.append("")

    lines.append("## Tracing cache misses")
    lines.append("")
    if not summary["cache_misses"]:
        lines.append("_None._")
    else:
        lines.append(f"Total: **{len(summary['cache_misses'])}** cache misses. "
                     "Each one is a retrace event — look at the **because** "
                     "line to find the root cause (new shape, new static arg, "
                     "new jaxpr, etc.).")
        lines.append("")
        by_loc: dict[str, list[str]] = defaultdict(list)
        for cm in summary["cache_misses"]:
            by_loc[cm["location"]].append(cm["reason"])
        loc_rows = sorted(by_loc.items(), key=lambda kv: len(kv[1]), reverse=True)
        lines.append("| Location | Misses | Sample reason |")
        lines.append("|---|---:|---|")
        for loc, reasons in loc_rows[:top_n]:
            reason = reasons[0][:140].replace("|", "\\|")
            lines.append(f"| `{loc}` | {len(reasons)} | {reason} |")
    lines.append("")

    lines.append("## Persistent cache misses")
    lines.append("")
    if not summary["persistent_misses"]:
        lines.append("_None._")
    else:
        counts: dict[str, int] = defaultdict(int)
        for n in summary["persistent_misses"]:
            counts[n] += 1
        for name, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"- `{name}`  ×{c}")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="compile.log file OR artifacts dir containing it")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    p = Path(args.path).resolve()
    if p.is_dir():
        p = p / "compile.log"
    if not p.is_file():
        print(f"[analyze_compile_log] no such file: {p}", file=sys.stderr)
        return 1

    summary = parse(p)
    md = render(summary, top_n=args.top)
    out_md = Path(args.out_md) if args.out_md else (p.parent / "compile_summary.md")
    out_json = Path(args.out_json) if args.out_json else (p.parent / "compile_summary.json")
    out_md.write_text(md)
    # JSON keeps lists — drop defaultdicts
    summary_json = {
        **summary,
        "xla": {k: v for k, v in summary["xla"].items()},
        "mlir": {k: v for k, v in summary["mlir"].items()},
        "trace": {k: v for k, v in summary["trace"].items()},
    }
    out_json.write_text(json.dumps(summary_json, indent=2, default=str))

    print(f"[analyze_compile_log] traces={sum(len(v) for v in summary['trace'].values())}  "
          f"compiles={sum(len(v) for v in summary['xla'].values())}  "
          f"cache-misses={len(summary['cache_misses'])}")
    print(f"[analyze_compile_log] wrote {out_md}")
    print(f"[analyze_compile_log] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
