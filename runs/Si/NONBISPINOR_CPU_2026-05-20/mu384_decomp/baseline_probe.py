"""Measure RSS at each major import boundary on the CPU backend.

Goal: characterize the "framework floor" — what RSS does each process hold
just from imports, before any GW work starts? This goes into the +6.5 GB
decomposition as a fixed-cost row.
"""
import os
import sys
import psutil

proc = psutil.Process()

def rss_gb() -> float:
    return proc.memory_info().rss / 1e9

def vsize_gb() -> float:
    return proc.memory_info().vms / 1e9

def stamp(label: str) -> None:
    print(f"[baseline] {label:50s}  RSS={rss_gb():6.3f} GB  VSZ={vsize_gb():7.2f} GB",
          flush=True)

stamp("python startup (after psutil)")

import numpy as np
stamp("after numpy")

# Touch a LARGE BLAS call to actually fault in OpenBLAS thread arenas across
# all 8 worker threads (small matrices don't trigger multi-thread fanout).
A = np.random.rand(2048, 2048)
B = np.random.rand(2048, 2048)
for _ in range(3):
    C = A @ B
del A, B, C
stamp("after large numpy matmul (BLAS multi-thread warm)")

import jax
stamp("after jax")

import jax.numpy as jnp
stamp("after jax.numpy")

# Force device + compiler init with a non-trivial matmul
x = jnp.ones((1024, 1024))
y = (x @ x).block_until_ready()
del x, y
stamp("after jax matmul (XLA warm)")

import h5py
stamp("after h5py")

# LORRAX imports
import gw.gw_jax as _gw  # noqa: F401
stamp("after gw.gw_jax")

import common.isdf_fitting as _isdf  # noqa: F401
stamp("after common.isdf_fitting")

import common.cholesky_2d as _ch  # noqa: F401
import common.wfn_transforms as _wt  # noqa: F401
import common.load_wfns as _lw  # noqa: F401
stamp("after common.*")

# Thread + arena info
import threading
print(f"[baseline] active threads      = {threading.active_count()}", flush=True)
thr_names = [t.name for t in threading.enumerate()]
print(f"[baseline] thread names        = {thr_names}", flush=True)

# /proc/self/status: VmData, VmRSS, VmSize, VmPeak
try:
    keys = ("VmPeak", "VmSize", "VmRSS", "VmData", "RssAnon", "RssFile", "RssShmem", "Threads")
    with open("/proc/self/status") as fh:
        for line in fh:
            for k in keys:
                if line.startswith(k + ":"):
                    print(f"[baseline] status {line.rstrip()}", flush=True)
                    break
except Exception as e:
    print(f"[baseline] status parse failed: {e}", flush=True)

# smaps: per-VMA RSS attribution
try:
    cat_rss = {"heap": 0, "anon": 0, "file": 0, "stack": 0, "other": 0}
    cat_n = {k: 0 for k in cat_rss}
    cur_cat = "other"
    with open("/proc/self/smaps") as fh:
        for line in fh:
            # Header line: addr-addr perm offset dev inode [path]
            if line and line[0].isdigit() and "-" in line.split()[0]:
                parts = line.split()
                path = parts[-1] if len(parts) >= 6 else ""
                if path == "[heap]":
                    cur_cat = "heap"
                elif path.startswith("[stack"):
                    cur_cat = "stack"
                elif path in ("", "[anon]", "[anon:thread_stack]"):
                    cur_cat = "anon"
                else:
                    cur_cat = "file"
                cat_n[cur_cat] += 1
            elif line.startswith("Rss:"):
                rkb = int(line.split()[1])
                cat_rss[cur_cat] += rkb
    print(f"[baseline] smaps category    : "
          f"heap {cat_rss['heap']/1024:.1f} MB ({cat_n['heap']}m), "
          f"anon {cat_rss['anon']/1024:.1f} MB ({cat_n['anon']}m), "
          f"file {cat_rss['file']/1024:.1f} MB ({cat_n['file']}m), "
          f"stack {cat_rss['stack']/1024:.1f} MB ({cat_n['stack']}m), "
          f"other {cat_rss['other']/1024:.1f} MB ({cat_n['other']}m)", flush=True)
except Exception as e:
    print(f"[baseline] smaps parse failed: {e}", flush=True)

print(f"[baseline] FINAL RSS = {rss_gb():.3f} GB", flush=True)
