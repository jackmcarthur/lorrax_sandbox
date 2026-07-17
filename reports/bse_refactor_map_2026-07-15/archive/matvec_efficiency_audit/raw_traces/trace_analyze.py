"""Kernel-level timeline from an xprof trace.json.gz (login-node, no GPU).

Aggregates device-side kernel durations by name and by category
(gemm / fft / collective / copy-transpose / elementwise / other), so a single
matvec call's time can be attributed. Also reports the busiest single kernels.

usage: trace_analyze.py <profile_logdir_or_trace.json.gz> [n_calls]
   n_calls: divide totals by this to get per-call numbers (default 10).
"""
import gzip, json, sys, re
from collections import Counter, defaultdict
from pathlib import Path

arg = sys.argv[1]
n_calls = int(sys.argv[2]) if len(sys.argv) > 2 else 10

p = Path(arg)
if p.is_dir():
    gz = sorted(p.rglob("*.trace.json.gz"))
    if not gz:
        print("no trace.json.gz under", p); sys.exit(1)
    trace_gz = gz[0]
else:
    trace_gz = p
print(f"### trace: {trace_gz}")

with gzip.open(trace_gz, "rt") as f:
    data = json.load(f)

events = data.get("traceEvents", [])

# Identify pids/tids: names of process/thread metadata to find GPU/device streams.
pid_name = {}
tid_name = {}
for ev in events:
    if ev.get("ph") == "M" and ev.get("name") == "process_name":
        pid_name[ev["pid"]] = ev.get("args", {}).get("name", "")
    if ev.get("ph") == "M" and ev.get("name") == "thread_name":
        tid_name[(ev.get("pid"), ev.get("tid"))] = ev.get("args", {}).get("name", "")

# device streams: process/thread names containing "stream" / "TensorFlow" / GPU markers.
def is_device(ev):
    pn = pid_name.get(ev.get("pid"), "")
    tn = tid_name.get((ev.get("pid"), ev.get("tid")), "")
    s = (pn + " " + tn).lower()
    return ("stream" in s) or ("gpu" in s and "compute" in s) or ("/device:gpu" in s) or ("xla ops" in s)

def classify(name):
    n = name.lower()
    if any(k in n for k in ["cublas", "gemm", "matmul", "dot"]):
        return "gemm"
    if "fft" in n:
        return "fft"
    if any(k in n for k in ["all-gather", "allgather", "collective-permute", "ppermute",
                            "reduce-scatter", "reducescatter", "all-reduce", "allreduce",
                            "nccl", "all-to-all"]):
        return "collective"
    if any(k in n for k in ["copy", "transpose", "bitcast"]):
        return "copy/transpose"
    if any(k in n for k in ["fusion", "add", "multiply", "subtract", "convert", "broadcast",
                            "reduce", "slice", "concatenate", "dynamic"]):
        return "elementwise/fusion"
    return "other"

by_name_dev = Counter()
by_cat_dev = Counter()
by_name_all = Counter()
dev_total = 0.0
found_device = False
for ev in events:
    if ev.get("ph") != "X":
        continue
    name = ev.get("name")
    dur = ev.get("dur", 0)
    if not isinstance(name, str):
        continue
    by_name_all[name] += dur
    if is_device(ev):
        found_device = True
        by_name_dev[name] += dur
        by_cat_dev[classify(name)] += dur
        dev_total += dur

# Fallback: if we couldn't identify device streams by metadata, use a heuristic —
# the streams whose names look like kernels (contain 'fusion'/'gemm'/'fft'/collective).
if not found_device:
    print("[warn] could not tag device streams via metadata; using name heuristic")
    for ev in events:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name")
        if not isinstance(name, str):
            continue
        if classify(name) != "other" or "%" in name:
            by_name_dev[name] += ev.get("dur", 0)
            by_cat_dev[classify(name)] += ev.get("dur", 0)
            dev_total += ev.get("dur", 0)

print(f"\n## Device-kernel time by CATEGORY (total {dev_total/1e3:.3f} ms over trace, "
      f"~{dev_total/1e3/n_calls:.4f} ms/call over {n_calls} calls)")
for cat, us in by_cat_dev.most_common():
    print(f"  {cat:22s} {us/1e3:9.3f} ms  {100*us/max(dev_total,1):5.1f}%   "
          f"{us/1e3/n_calls:8.4f} ms/call")

print(f"\n## Top 30 device kernels by total time")
for name, us in by_name_dev.most_common(30):
    short = name if len(name) < 70 else name[:67] + "..."
    print(f"  {us/1e3:9.3f} ms  {us/1e3/n_calls:8.4f}/call  [{classify(name):18s}] {short}")

print(f"\n## Process/thread names seen (for stream identification)")
for pid, nm in sorted(pid_name.items()):
    print(f"  pid {pid}: {nm}")
