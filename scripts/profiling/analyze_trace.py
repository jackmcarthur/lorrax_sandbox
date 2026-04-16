#!/usr/bin/env python3
"""analyze_trace.py — agent-readable summary of a JAX xprof / perfetto trace.

Reads ``<artifacts>/xprof/plugins/profile/<ts>/perfetto_trace.json.gz`` (the
Chrome-JSON aggregated trace that JAX emits on single-process runs). Writes:

    <artifacts>/trace_summary.md        Ranked tables:
                                          - Top kernels by total GPU time
                                          - H2D / D2H transfers by bytes + bandwidth
                                          - Async-copy overlap with compute
                                          - Sliding-window bandwidth saturation
                                          - Low-occupancy kernel offenders
    <artifacts>/trace_summary.json      Same data, machine-readable.
    <artifacts>/trace_details.txt       Dense per-event dump of the top-N copies
                                        and top-N compute kernels.

The trace JSON is Chrome-trace format (phase 'X' events with ts/dur in µs).
Each GPU event carries an ``args`` dict with:
  - memcpy_details   : "kind_src:pinned kind_dst:device size:<bytes> dest:<dev> async:<0/1>"
  - hlo_module       : e.g. jit_species_structure_factors
  - hlo_op           : e.g. fusion.3, all-gather-start.1
  - name             : Python path e.g. jit(...)/jit(main)/foo source_file=...
  - theoretical_occupancy_pct : occupancy hint (for compute kernels only)

Multi-process runs disable the perfetto aggregation (to avoid rank races) —
for those, each rank ships its own xplane.pb and this analyzer does not
handle them yet. The singletrack summary is the 80%-use case.

Usage:
    python3 analyze_trace.py <artifacts_dir>
    python3 analyze_trace.py <artifacts_dir> --top 30 --window-ms 100
    python3 analyze_trace.py <artifacts_dir> --trace <path-to-perfetto_trace.json.gz>
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────
#  Loaders
# ─────────────────────────────────────────────────────────────────────────

RE_MCD = re.compile(
    r"kind_src:(?P<src>\w+)\s+kind_dst:(?P<dst>\w+)\s+"
    r"size:(?P<size>\d+)(?:\s+dest:(?P<dest>\d+))?(?:\s+async:(?P<async_>\d+))?"
)


def _find_trace_json(root: Path) -> Path | None:
    """Locate a perfetto_trace.json.gz under <root>/xprof/...
    Single-process: <root>/xprof/plugins/profile/<ts>/perfetto_trace.json.gz
    Multi-process:  <root>/xprof/rank_0/plugins/profile/<ts>/perfetto_trace.json.gz

    Returns the most-recent rank-0 trace, None if none found.
    """
    candidates: list[Path] = []
    # Multi-process: prefer rank_0
    for base in (root / "xprof" / "rank_0", root / "xprof"):
        prof = base / "plugins" / "profile"
        if prof.is_dir():
            for d in sorted(prof.iterdir(),
                            key=lambda p: p.stat().st_mtime, reverse=True):
                p = d / "perfetto_trace.json.gz"
                if p.is_file():
                    candidates.append(p)
            if candidates:
                return candidates[0]
    return None


def load_trace(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def _hb(n: int | float | None) -> str:
    if n is None:
        return "-"
    x = float(n)
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(x) < 1024.0:
            return f"{x:.2f} {u}"
        x /= 1024.0
    return f"{x:.2f} PiB"


def _us_to_ms(us: float) -> float:
    return us / 1000.0


# ─────────────────────────────────────────────────────────────────────────
#  Extraction
# ─────────────────────────────────────────────────────────────────────────

def index_streams(events: list[dict]) -> tuple[dict, dict]:
    """Return (tid→name, gpu_streams) mappings.

    gpu_streams[(pid,tid)] = {"kind": one of compute|h2d|d2h|d2d|mixed,
                              "name": raw stream name}
    """
    tid_name: dict[tuple[int, int], str] = {}
    proc_name: dict[int, str] = {}
    for e in events:
        if e.get("ph") == "M" and e.get("name") == "process_name":
            proc_name[e["pid"]] = e["args"]["name"]
        elif e.get("ph") == "M" and e.get("name") == "thread_name":
            tid_name[(e["pid"], e["tid"])] = e["args"]["name"]

    gpu_streams: dict[tuple[int, int], dict] = {}
    for key, name in tid_name.items():
        has_h2d = "MemcpyH2D" in name
        has_d2h = "MemcpyD2H" in name
        has_d2d = "MemcpyD2D" in name
        has_compute = "Compute" in name
        if not (has_h2d or has_d2h or has_d2d or has_compute):
            continue
        if has_compute and not (has_h2d or has_d2h or has_d2d):
            kind = "compute"
        elif has_h2d and not (has_d2h or has_d2d or has_compute):
            kind = "h2d"
        elif has_d2h and not (has_h2d or has_d2d or has_compute):
            kind = "d2h"
        elif has_d2d and not (has_h2d or has_d2h or has_compute):
            kind = "d2d"
        else:
            kind = "mixed"
        gpu_streams[key] = {"kind": kind, "name": name}
    return tid_name, gpu_streams


def extract_gpu_events(events: list[dict], gpu_streams: dict) -> list[dict]:
    """Pull all phase-X events on known GPU streams, pre-parse memcpy args."""
    out = []
    for e in events:
        if e.get("ph") != "X":
            continue
        key = (e.get("pid"), e.get("tid"))
        s = gpu_streams.get(key)
        if not s:
            continue
        name = e.get("name", "")
        args = e.get("args") or {}
        mcd = args.get("memcpy_details", "")
        parsed = {}
        if mcd:
            m = RE_MCD.search(mcd)
            if m:
                parsed = {
                    "src_kind": m.group("src"),
                    "dst_kind": m.group("dst"),
                    "size": int(m.group("size")),
                    "async": int(m.group("async_") or "0"),
                }
        # Classify this event
        if "MemcpyH2D" in name:
            ev_kind = "h2d"
        elif "MemcpyD2H" in name:
            ev_kind = "d2h"
        elif "MemcpyD2D" in name:
            ev_kind = "d2d"
        else:
            ev_kind = "compute"
        out.append({
            "ts": float(e.get("ts", 0.0)),
            "dur": float(e.get("dur", 0.0)),
            "end": float(e.get("ts", 0.0)) + float(e.get("dur", 0.0)),
            "name": name,
            "pid": e.get("pid"),
            "tid": e.get("tid"),
            "stream_kind": s["kind"],
            "stream_name": s["name"],
            "ev_kind": ev_kind,
            "hlo_module": args.get("hlo_module", ""),
            "hlo_op": args.get("hlo_op", ""),
            "py_name": args.get("name", ""),
            "occupancy": args.get("theoretical_occupancy_pct", ""),
            **parsed,
        })
    return out


# ─────────────────────────────────────────────────────────────────────────
#  Analyses
# ─────────────────────────────────────────────────────────────────────────

def rank_kernels(events: list[dict], top_n: int) -> list[dict]:
    """Top compute kernels by total GPU time."""
    buckets: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "total_us": 0.0, "max_us": 0.0,
                 "hlo_module": "", "py_name": "", "occupancy": ""})
    for e in events:
        if e["ev_kind"] != "compute":
            continue
        key = e["hlo_op"] or e["name"]
        b = buckets[key]
        b["count"] += 1
        b["total_us"] += e["dur"]
        if e["dur"] > b["max_us"]:
            b["max_us"] = e["dur"]
        if e["hlo_module"]:
            b["hlo_module"] = e["hlo_module"]
        if e["py_name"]:
            b["py_name"] = e["py_name"]
        if e["occupancy"]:
            b["occupancy"] = e["occupancy"]
    rows = [{"op": k, **v} for k, v in buckets.items()]
    rows.sort(key=lambda r: r["total_us"], reverse=True)
    return rows[:top_n]


def rank_copies(events: list[dict], top_n: int) -> dict:
    """Aggregate H2D / D2H / D2D transfers by direction + per-event ranking."""
    totals: dict[str, dict] = {
        "h2d": {"count": 0, "bytes": 0, "us": 0.0},
        "d2h": {"count": 0, "bytes": 0, "us": 0.0},
        "d2d": {"count": 0, "bytes": 0, "us": 0.0},
    }
    tops: list[dict] = []
    for e in events:
        k = e["ev_kind"]
        if k not in totals:
            continue
        sz = int(e.get("size") or 0)
        totals[k]["count"] += 1
        totals[k]["bytes"] += sz
        totals[k]["us"] += e["dur"]
        if sz > 0:
            tops.append({
                "ev_kind": k,
                "bytes": sz,
                "us": e["dur"],
                "gbps": (sz / 1e9) / (e["dur"] / 1e6) if e["dur"] > 0 else 0.0,
                "ts": e["ts"],
                "async": e.get("async", 0),
                "hlo_op": e["hlo_op"],
                "py_name": e["py_name"],
            })
    tops.sort(key=lambda r: r["bytes"], reverse=True)
    # Effective bandwidth per direction (ignoring parallelism on multiple streams)
    for k in totals:
        t = totals[k]
        t["gbps"] = (t["bytes"] / 1e9) / (t["us"] / 1e6) if t["us"] > 0 else 0.0
    return {"totals": totals, "top_events": tops[:top_n]}


def overlap_analysis(events: list[dict]) -> dict:
    """For every H2D/D2H copy event, compute the fraction of its wall-time
    during which ≥1 compute kernel is running on a different stream.

    A well-overlapping async transfer has overlap_frac → 1.0. A stall
    (legitimate or accidental) shows overlap_frac → 0.0.

    Also reports aggregate D2H / H2D exposed (non-overlapped) time — this is
    what actually holds up forward progress if JAX is using async transfers
    to hide bandwidth.
    """
    copies = [e for e in events if e["ev_kind"] in ("h2d", "d2h")]
    computes = [(e["ts"], e["end"]) for e in events if e["ev_kind"] == "compute"]
    computes.sort()

    # Sorted start times for binary search
    import bisect
    starts = [c[0] for c in computes]

    def _compute_overlap(a: float, b: float) -> float:
        """Sum of compute interval overlaps with [a,b]."""
        total = 0.0
        # Find first compute whose end > a
        # We only have starts sorted; a lightweight sweep is fine since copy
        # counts are modest. Use starts to skip clearly-too-late intervals.
        hi = bisect.bisect_right(starts, b)
        for i in range(hi - 1, -1, -1):
            cs, ce = computes[i]
            if ce <= a:
                # all earlier compute intervals also end before a
                break
            lo = max(a, cs)
            hi2 = min(b, ce)
            if hi2 > lo:
                total += hi2 - lo
        return total

    per_event: list[dict] = []
    exposed = defaultdict(float)  # ev_kind -> us
    for c in copies:
        if c["dur"] <= 0:
            continue
        ov = _compute_overlap(c["ts"], c["end"])
        ov_frac = min(1.0, ov / c["dur"])
        exposed_us = c["dur"] - min(ov, c["dur"])
        exposed[c["ev_kind"]] += exposed_us
        per_event.append({
            "ev_kind": c["ev_kind"],
            "bytes": int(c.get("size") or 0),
            "us": c["dur"],
            "overlap_frac": ov_frac,
            "exposed_us": exposed_us,
            "ts": c["ts"],
            "async": c.get("async", 0),
            "hlo_op": c["hlo_op"],
        })

    # Aggregate fractions
    def _agg(kind: str) -> dict:
        evs = [e for e in per_event if e["ev_kind"] == kind]
        if not evs:
            return {"count": 0, "total_us": 0.0, "exposed_us": 0.0,
                    "overlap_frac": None}
        total_us = sum(e["us"] for e in evs)
        exp_us = sum(e["exposed_us"] for e in evs)
        return {
            "count": len(evs),
            "total_us": total_us,
            "exposed_us": exp_us,
            "overlap_frac": 1.0 - (exp_us / total_us) if total_us > 0 else None,
        }

    return {
        "h2d": _agg("h2d"),
        "d2h": _agg("d2h"),
        "per_event": per_event,
    }


def bandwidth_saturation(events: list[dict], window_us: float = 100_000.0
                         ) -> dict:
    """Sliding-window peak bandwidth per direction.

    Divides the trace into non-overlapping windows of ``window_us``; reports
    the window with the highest byte throughput for each direction.

    The A100 PCIe Gen4 x16 theoretical ceiling is ~32 GB/s per direction.
    Sustained >20 GB/s for >100 ms almost certainly means the copy channel
    is saturated — look at overlap_frac next to decide whether that blocks
    compute.
    """
    if not events:
        return {}
    t0 = min(e["ts"] for e in events)
    t1 = max(e["end"] for e in events)
    n_bins = max(1, int((t1 - t0) / window_us) + 1)
    bins_h2d = [0] * n_bins
    bins_d2h = [0] * n_bins
    for e in events:
        if e["ev_kind"] not in ("h2d", "d2h"):
            continue
        sz = int(e.get("size") or 0)
        if sz <= 0 or e["dur"] <= 0:
            continue
        # Distribute bytes across bins the event spans, weighted by overlap
        s, ed = e["ts"], e["end"]
        rate = sz / e["dur"]  # bytes per µs
        bi0 = int((s - t0) / window_us)
        bi1 = int((ed - t0) / window_us)
        for bi in range(max(0, bi0), min(n_bins, bi1 + 1)):
            bin_lo = t0 + bi * window_us
            bin_hi = bin_lo + window_us
            overlap = max(0.0, min(ed, bin_hi) - max(s, bin_lo))
            bytes_here = overlap * rate
            target = bins_h2d if e["ev_kind"] == "h2d" else bins_d2h
            target[bi] += int(bytes_here)

    def _peak(bins):
        if not bins:
            return {"peak_gbps": 0.0, "peak_bytes": 0, "t_peak_s": 0.0}
        idx = max(range(len(bins)), key=lambda i: bins[i])
        return {
            "peak_bytes": bins[idx],
            "peak_gbps": (bins[idx] / 1e9) / (window_us / 1e6),
            "t_peak_s": (idx * window_us) / 1e6,
        }

    return {
        "window_us": window_us,
        "n_bins": n_bins,
        "h2d": _peak(bins_h2d),
        "d2h": _peak(bins_d2h),
        "bins_h2d": bins_h2d,
        "bins_d2h": bins_d2h,
    }


def low_occupancy(events: list[dict], thresh_pct: float = 50.0,
                  top_n: int = 20) -> list[dict]:
    """Compute kernels whose reported theoretical occupancy is below thresh
    AND which consume meaningful total time — sorted by wasted time."""
    rows = []
    for e in events:
        if e["ev_kind"] != "compute" or not e["occupancy"]:
            continue
        try:
            occ = float(e["occupancy"])
        except Exception:
            continue
        if occ >= thresh_pct:
            continue
        rows.append({
            "hlo_op": e["hlo_op"],
            "occupancy": occ,
            "dur_us": e["dur"],
            "hlo_module": e["hlo_module"],
            "py_name": e["py_name"][:120],
        })
    rows.sort(key=lambda r: r["dur_us"], reverse=True)
    return rows[:top_n]


# ─────────────────────────────────────────────────────────────────────────
#  Rendering
# ─────────────────────────────────────────────────────────────────────────

def render_md(summary: dict, top_n: int) -> str:
    L: list[str] = []
    L.append("# Trace summary")
    L.append("")
    L.append(f"**Trace:** `{summary['trace_path']}`")
    L.append(f"**Duration:** {summary['duration_s']:.3f} s")
    L.append(f"**GPU streams:** {summary['n_compute_streams']} compute, "
             f"{summary['n_h2d_streams']} H2D, "
             f"{summary['n_d2h_streams']} D2H")
    L.append("")
    L.append("_Companion:_ [`trace_details.txt`](trace_details.txt) — "
             "dense per-event dump of the top copies + top kernels.")
    L.append("")

    # ─────── Transfer totals ───────
    L.append("## Host ↔ device transfers")
    L.append("")
    tot = summary["copies"]["totals"]
    L.append("| Direction | Count | Total bytes | Total time | Avg GB/s |")
    L.append("|---|---:|---:|---:|---:|")
    for k in ("h2d", "d2h", "d2d"):
        t = tot[k]
        L.append(f"| {k.upper()} | {t['count']} | {_hb(t['bytes'])} | "
                 f"{_us_to_ms(t['us']):.2f} ms | {t['gbps']:.2f} |")
    L.append("")
    L.append("_Avg GB/s is **sum(bytes) / sum(time)**: treats the channel as "
             "one stream. With multiple streams, instantaneous bandwidth can "
             "be higher; see the peak table below._")
    L.append("")

    # ─────── Overlap ───────
    L.append("## Async overlap — were copies hidden behind compute?")
    L.append("")
    ov = summary["overlap"]
    L.append("| Direction | Count | Total time | Exposed (non-overlapped) | Overlap frac |")
    L.append("|---|---:|---:|---:|---:|")
    for k in ("h2d", "d2h"):
        a = ov[k]
        if a["count"] == 0:
            L.append(f"| {k.upper()} | 0 | - | - | - |")
            continue
        of = a["overlap_frac"]
        of_s = f"{of:.3f}" if of is not None else "-"
        L.append(f"| {k.upper()} | {a['count']} | {_us_to_ms(a['total_us']):.2f} ms | "
                 f"{_us_to_ms(a['exposed_us']):.2f} ms | {of_s} |")
    L.append("")
    L.append("_overlap_frac = (total − exposed) / total. **Close to 1 is good** "
             "(copy happened while the GPU was busy with compute, so it's free). "
             "**Below ~0.3 means the copy is blocking the pipeline** — either "
             "the issuer is waiting on the data (legitimate stall) or the "
             "copy was dispatched too late (schedulable bug)._")
    L.append("")

    # ─────── Bandwidth saturation ───────
    bw = summary["bandwidth"]
    if bw:
        L.append(f"## Bandwidth saturation (window = {bw['window_us']/1000:.0f} ms)")
        L.append("")
        L.append("| Direction | Peak window bytes | Peak window GB/s | At t |")
        L.append("|---|---:|---:|---:|")
        for k in ("h2d", "d2h"):
            p = bw[k]
            L.append(f"| {k.upper()} | {_hb(p['peak_bytes'])} | "
                     f"{p['peak_gbps']:.2f} | {p['t_peak_s']:.2f} s |")
        L.append("")
        L.append("_A100 PCIe Gen4 x16 ≈ 32 GB/s/direction theoretical. "
                 "Sustained > ~20 GB/s in a window means the link is saturated; "
                 "combine with the overlap table above — saturated + low "
                 "overlap = real bottleneck._")
        L.append("")

    # ─────── Top kernels ───────
    L.append(f"## Top {top_n} GPU kernels by total time")
    L.append("")
    L.append("| Op | Count | Total ms | Max ms | Occupancy % | HLO module | Source |")
    L.append("|---|---:|---:|---:|---:|---|---|")
    for r in summary["kernels"][:top_n]:
        py = (r.get("py_name") or "").split(" source_file=")[0][:60]
        py = py.replace("|", "\\|")
        L.append(f"| `{r['op']}` | {r['count']} | "
                 f"{_us_to_ms(r['total_us']):.2f} | {_us_to_ms(r['max_us']):.2f} | "
                 f"{r.get('occupancy','') or '-'} | `{r['hlo_module'][:30]}` | "
                 f"`{py}` |")
    L.append("")

    # ─────── Low occupancy ───────
    lo = summary["low_occupancy"]
    if lo:
        L.append("## Low-occupancy compute kernels "
                 f"(theoretical < 50 %, ranked by wasted time)")
        L.append("")
        L.append("| Op | Occupancy | µs | Source |")
        L.append("|---|---:|---:|---|")
        for r in lo[:top_n]:
            py = (r.get("py_name") or "").split(" source_file=")[0][:80]
            L.append(f"| `{r['hlo_op']}` | {r['occupancy']:.1f} % | "
                     f"{r['dur_us']:.1f} | `{py}` |")
        L.append("")

    return "\n".join(L) + "\n"


def render_details(summary: dict, top_n: int) -> str:
    L: list[str] = []
    L.append("# trace_details.txt — dense per-event dump")
    L.append("")
    L.append("## Top copies (by bytes)")
    L.append(f"{'kind':<5s}  {'bytes':>10s}  {'us':>8s}  {'GB/s':>6s}  "
             f"{'async':>5s}  hlo_op / source")
    for e in summary["copies"]["top_events"][:top_n]:
        L.append(f"{e['ev_kind'].upper():<5s}  "
                 f"{_hb(e['bytes']):>10s}  "
                 f"{e['us']:>8.2f}  "
                 f"{e['gbps']:>6.2f}  "
                 f"{e.get('async',0):>5d}  "
                 f"{e['hlo_op']}  ::  {e['py_name'][:100]}")
    L.append("")
    L.append("## Top copies by EXPOSED time (not hidden by compute)")
    L.append(f"{'kind':<5s}  {'bytes':>10s}  {'us':>8s}  {'exposed':>8s}  "
             f"{'overlap':>7s}  hlo_op")
    top_exposed = sorted(summary["overlap"]["per_event"],
                         key=lambda r: r["exposed_us"], reverse=True)[:top_n]
    for e in top_exposed:
        L.append(f"{e['ev_kind'].upper():<5s}  "
                 f"{_hb(e['bytes']):>10s}  "
                 f"{e['us']:>8.2f}  "
                 f"{e['exposed_us']:>8.2f}  "
                 f"{e['overlap_frac']:>7.3f}  "
                 f"{e['hlo_op']}")
    L.append("")
    L.append("## Top compute kernels")
    L.append(f"{'total_ms':>9s}  {'count':>5s}  {'max_ms':>8s}  "
             f"{'occ%':>5s}  op  ::  source")
    for r in summary["kernels"][:top_n]:
        py = (r.get("py_name") or "").split(" source_file=")[0][:120]
        L.append(f"{_us_to_ms(r['total_us']):>9.2f}  "
                 f"{r['count']:>5d}  "
                 f"{_us_to_ms(r['max_us']):>8.2f}  "
                 f"{(r.get('occupancy','') or '-'):>5s}  "
                 f"{r['op']}  ::  {py}")
    return "\n".join(L) + "\n"


# ─────────────────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("artifacts_dir",
                    help="Path containing xprof/ (produced by pf.trace_profile)")
    ap.add_argument("--trace", default=None,
                    help="Override the trace JSON path")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--window-ms", type=float, default=100.0,
                    help="Sliding-window size for bandwidth analysis")
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-txt", default=None)
    args = ap.parse_args()

    root = Path(args.artifacts_dir).resolve()
    if args.trace:
        trace_path = Path(args.trace).resolve()
    else:
        trace_path = _find_trace_json(root)
    if trace_path is None or not trace_path.is_file():
        print(f"[analyze_trace] no perfetto_trace.json.gz under {root}; "
              "multi-process traces not yet supported — use single-process or "
              "pass --trace explicitly.", file=sys.stderr)
        return 1

    print(f"[analyze_trace] loading {trace_path} ({trace_path.stat().st_size/1e6:.1f} MB gz)")
    trace = load_trace(trace_path)
    events = trace.get("traceEvents", [])
    tid_name, gpu_streams = index_streams(events)
    gpu_events = extract_gpu_events(events, gpu_streams)

    if not gpu_events:
        print(f"[analyze_trace] no GPU-stream events parsed — abort.", file=sys.stderr)
        return 1
    t0 = min(e["ts"] for e in gpu_events)
    t1 = max(e["end"] for e in gpu_events)
    duration_s = (t1 - t0) / 1e6

    n_compute = sum(1 for s in gpu_streams.values() if s["kind"] == "compute")
    n_h2d = sum(1 for s in gpu_streams.values() if s["kind"] == "h2d")
    n_d2h = sum(1 for s in gpu_streams.values() if s["kind"] == "d2h")

    kernels = rank_kernels(gpu_events, args.top)
    copies = rank_copies(gpu_events, args.top)
    overlap = overlap_analysis(gpu_events)
    bandwidth = bandwidth_saturation(gpu_events, window_us=args.window_ms * 1000)
    low_occ = low_occupancy(gpu_events, thresh_pct=50.0, top_n=args.top)

    summary = {
        "trace_path": str(trace_path),
        "duration_s": duration_s,
        "n_compute_streams": n_compute,
        "n_h2d_streams": n_h2d,
        "n_d2h_streams": n_d2h,
        "kernels": kernels,
        "copies": copies,
        "overlap": {**overlap, "per_event": overlap["per_event"][:500]},
        "bandwidth": {**bandwidth, "bins_h2d": bandwidth.get("bins_h2d", [])[:500],
                      "bins_d2h": bandwidth.get("bins_d2h", [])[:500]},
        "low_occupancy": low_occ,
    }

    md = render_md(summary, args.top)
    txt = render_details(summary, args.top)
    out_md = Path(args.out_md) if args.out_md else (root / "trace_summary.md")
    out_json = Path(args.out_json) if args.out_json else (root / "trace_summary.json")
    out_txt = Path(args.out_txt) if args.out_txt else (root / "trace_details.txt")
    out_md.write_text(md)
    out_txt.write_text(txt)
    out_json.write_text(json.dumps(summary, indent=2, default=str))

    print(f"[analyze_trace] duration {duration_s:.2f}s, {len(gpu_events)} GPU events, "
          f"{copies['totals']['h2d']['count']} H2D, "
          f"{copies['totals']['d2h']['count']} D2H")
    print(f"[analyze_trace] wrote {out_md}")
    print(f"[analyze_trace] wrote {out_txt}")
    print(f"[analyze_trace] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
