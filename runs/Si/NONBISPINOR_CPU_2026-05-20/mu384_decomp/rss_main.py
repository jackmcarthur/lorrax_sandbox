"""RSS-instrumented launcher for `python -m gw.gw_jax -i cohsex.in`.

Starts a background thread that polls /proc/self/status every 0.5s and
writes peak RSS + timeline to rss_timeline_rank<RANK>_<TAG>.txt at exit.

Usage:
    python -u rss_main.py [args passed to gw.gw_jax]

Reads:
    SLURM_PROCID    rank id used for output filename
    RSS_TAG         scenario tag (default, openblas1, mallocarena1, ...)
    RSS_INTERVAL_S  sample interval in seconds (default 0.5)
"""
import os
import sys
import threading
import time
import atexit

_RANK = int(os.environ.get("SLURM_PROCID", "0"))
_TAG = os.environ.get("RSS_TAG", "default")
_INT = float(os.environ.get("RSS_INTERVAL_S", "0.5"))
_OUT = f"rss_timeline_rank{_RANK}_{_TAG}.txt"

_t0 = time.perf_counter()
_timeline = []     # (t, rss_kb, vmsize_kb, threads, rss_anon_kb)
_peak_rss_kb = 0
_stop = threading.Event()

def _read_status():
    rss = vsz = thr = rss_anon = vmdata = 0
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmSize:"):
                    vsz = int(line.split()[1])
                elif line.startswith("Threads:"):
                    thr = int(line.split()[1])
                elif line.startswith("RssAnon:"):
                    rss_anon = int(line.split()[1])
                elif line.startswith("VmData:"):
                    vmdata = int(line.split()[1])
    except Exception:
        pass
    return rss, vsz, thr, rss_anon, vmdata

def _sampler():
    global _peak_rss_kb
    while not _stop.is_set():
        t = time.perf_counter() - _t0
        rss, vsz, thr, rssa, vmd = _read_status()
        _timeline.append((t, rss, vsz, thr, rssa, vmd))
        if rss > _peak_rss_kb:
            _peak_rss_kb = rss
        _stop.wait(_INT)

def _flush():
    global _peak_rss_kb
    _stop.set()
    # final sample
    t = time.perf_counter() - _t0
    rss, vsz, thr, rssa, vmd = _read_status()
    _timeline.append((t, rss, vsz, thr, rssa, vmd))
    if rss > _peak_rss_kb:
        _peak_rss_kb = rss
    try:
        with open(_OUT, "w") as fh:
            fh.write(f"# rank={_RANK} tag={_TAG} interval={_INT}s n_samples={len(_timeline)}\n")
            fh.write(f"# PEAK_RSS_GB = {_peak_rss_kb / (1024**2):.3f}\n")
            fh.write(f"# {'t (s)':>8s}  {'RSS (KB)':>12s}  {'VmSize KB':>12s}  "
                     f"{'thr':>5s}  {'RssAnon KB':>12s}  {'VmData KB':>12s}\n")
            for t, rss, vsz, thr, rssa, vmd in _timeline:
                fh.write(f"{t:>8.2f}  {rss:>12d}  {vsz:>12d}  {thr:>5d}  "
                         f"{rssa:>12d}  {vmd:>12d}\n")
    except Exception as e:
        print(f"[rss_main rank{_RANK}] flush failed: {e}", file=sys.stderr, flush=True)
    print(f"[rss_main rank{_RANK}] tag={_TAG} PEAK RSS = {_peak_rss_kb / (1024**2):.3f} GB "
          f"(file={_OUT})", flush=True)

_thr = threading.Thread(target=_sampler, name="rss-sampler", daemon=True)
_thr.start()
atexit.register(_flush)

# Now invoke gw.gw_jax with the passed args
import runpy
sys.argv = ["gw.gw_jax"] + sys.argv[1:]
runpy.run_module("gw.gw_jax", run_name="__main__", alter_sys=True)
