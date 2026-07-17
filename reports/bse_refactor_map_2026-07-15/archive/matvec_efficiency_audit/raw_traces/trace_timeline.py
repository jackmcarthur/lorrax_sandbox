"""Per-collective / per-kernel device timeline from an xprof trace (login-node).

Emits device-stream ops (collectives, fft, gemm, big copies) in TIMESTAMP order
with inter-op GAPS, for measuring launch latency and comm/compute overlap. Also
detects concurrency: ops whose [ts, ts+dur) intervals overlap across streams.

usage: trace_timeline.py <profile_logdir_or_trace.json.gz> [--window N]
   --window N: only show the last N-th call window (default: whole trace).
"""
import gzip, json, sys, re
from pathlib import Path

arg = sys.argv[1]
p = Path(arg)
trace_gz = sorted(p.rglob("*.trace.json.gz"))[0] if p.is_dir() else p
print(f"### timeline: {trace_gz}")
with gzip.open(trace_gz, "rt") as f:
    data = json.load(f)
events = data.get("traceEvents", [])

pid_name, tid_name = {}, {}
for ev in events:
    if ev.get("ph") == "M" and ev.get("name") == "process_name":
        pid_name[ev["pid"]] = ev.get("args", {}).get("name", "")
    if ev.get("ph") == "M" and ev.get("name") == "thread_name":
        tid_name[(ev.get("pid"), ev.get("tid"))] = ev.get("args", {}).get("name", "")

def stream(ev):
    pn = pid_name.get(ev.get("pid"), "")
    tn = tid_name.get((ev.get("pid"), ev.get("tid")), "")
    return (pn + " | " + tn)

def is_device(ev):
    s = stream(ev).lower()
    return ("stream" in s) or ("xla ops" in s) or ("/device:gpu" in s and "compute" in s)

def cat(name):
    n = name.lower()
    if any(k in n for k in ["all-gather", "allgather"]): return "ALLGATHER"
    if any(k in n for k in ["collective-permute", "ppermute"]): return "PPERMUTE"
    if any(k in n for k in ["reduce-scatter", "reducescatter"]): return "REDUCE-SCATTER"
    if any(k in n for k in ["all-reduce", "allreduce"]): return "ALL-REDUCE"
    if "all-to-all" in n: return "ALL-TO-ALL"
    if "nccl" in n: return "NCCL"
    if "fft" in n: return "FFT"
    if any(k in n for k in ["cublas", "gemm", "matmul"]): return "GEMM"
    if "dot" in n: return "DOT"
    if any(k in n for k in ["copy", "transpose"]): return "COPY/TRANSPOSE"
    return None

# collect device ops of interest
ops = []
for ev in events:
    if ev.get("ph") != "X" or not isinstance(ev.get("name"), str):
        continue
    if not is_device(ev):
        continue
    c = cat(ev["name"])
    if c is None:
        continue
    ops.append(dict(ts=ev["ts"], dur=ev.get("dur", 0), cat=c,
                    name=ev["name"], stream=stream(ev)))

if not ops:
    print("[warn] no categorized device ops found; dumping stream names:")
    for pid, nm in sorted(pid_name.items()):
        print(f"  pid {pid}: {nm}")
        for (pp, tt), tn in sorted(tid_name.items()):
            if pp == pid:
                print(f"      tid {tt}: {tn}")
    sys.exit(0)

ops.sort(key=lambda o: o["ts"])
t0 = ops[0]["ts"]

# collectives + fft + gemm timeline
print(f"\n## Device op timeline (ts relative to first op, us). {len(ops)} ops.")
print(f"{'t_us':>10} {'dur_us':>9} {'gap_us':>8}  {'cat':16s} {'stream':28s} name")
prev_end = None
COMM = {"ALLGATHER", "PPERMUTE", "REDUCE-SCATTER", "ALL-REDUCE", "ALL-TO-ALL", "NCCL"}
comm_time = 0.0
fft_time = 0.0
gemm_time = 0.0
copy_time = 0.0
counts = {}
for o in ops:
    rel = o["ts"] - t0
    gap = "" if prev_end is None else f"{o['ts']-prev_end:.2f}"
    prev_end = o["ts"] + o["dur"]
    counts[o["cat"]] = counts.get(o["cat"], 0) + 1
    if o["cat"] in COMM: comm_time += o["dur"]
    elif o["cat"] == "FFT": fft_time += o["dur"]
    elif o["cat"] in ("GEMM", "DOT"): gemm_time += o["dur"]
    elif o["cat"] == "COPY/TRANSPOSE": copy_time += o["dur"]
    short = o["name"] if len(o["name"]) < 55 else o["name"][:52] + "..."
    st = o["stream"] if len(o["stream"]) < 28 else o["stream"][-28:]
    print(f"{rel:10.2f} {o['dur']:9.2f} {gap:>8}  {o['cat']:16s} {st:28s} {short}")

print(f"\n## Category totals (whole trace)")
tot = comm_time + fft_time + gemm_time + copy_time
for k, v in counts.items():
    print(f"  {k:16s} count={v}")
print(f"  comm_time={comm_time:.1f}us  fft={fft_time:.1f}us  gemm={gemm_time:.1f}us  "
      f"copy/transpose={copy_time:.1f}us")

# concurrency check: max number of simultaneously-active ops
edges = []
for o in ops:
    edges.append((o["ts"], 1))
    edges.append((o["ts"] + o["dur"], -1))
edges.sort()
cur = mx = 0
for _, d in edges:
    cur += d
    mx = max(mx, cur)
print(f"  max concurrent device ops (of interest): {mx}  "
      f"({'OVERLAP present' if mx > 1 else 'fully serialized'})")
