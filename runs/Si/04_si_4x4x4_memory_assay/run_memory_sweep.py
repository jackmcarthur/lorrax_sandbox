#!/usr/bin/env python3
"""First-principles GPU memory sweep for LORRAX ISDF/GW pipeline.

Runs the FULL zeta fitting pipeline (load_wfns -> CCT -> cholesky ->
r-chunk loop [load, pair density, ZCT, reshard, solve, gather]) with
_mem_report() hooks active at every stage boundary.

MUST be run in multi-process mode (1 GPU per process):
    srun --jobid=$JOBID --gres=gpu:4 -N 1 -n 4 $SHIFTER \
        --env=LORRAX_MEM_PROFILE=1 \
        --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
        --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
        python3 -u run_memory_sweep.py [--xprof] [--config CONFIG_ID]

Configurations are defined in SWEEP_CONFIGS below. Each config specifies
chunk sizes and (optionally) calculation parameters. The script runs the
full pipeline for each config, collecting _mem_report output and timing.

Output: sweep_results_<timestamp>.json with per-config, per-stage memory data.

When --xprof is passed, also captures XProf traces for each config under
./profiles/xprof/<config_id>-<timestamp>/.
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from io import StringIO
from contextlib import redirect_stdout

# Validate multi-process execution BEFORE importing JAX
# (JAX import triggers device initialization)
if "SLURM_PROCID" in os.environ:
    # We're running under srun — good
    pass
else:
    print("WARNING: Not running under srun. Memory measurements may be invalid.")
    print("  Use: srun -N 1 -n 4 ... python3 -u run_memory_sweep.py")

# Ensure mem profiling is active
os.environ["LORRAX_MEM_PROFILE"] = "1"

# Initialize JAX distributed BEFORE any other JAX calls.
# This is critical: without it, each srun task sees only its own GPU
# and thinks it's the sole process. Uses the same logic as gw_jax.py.
import jax

_DISTRIBUTED_SENTINEL = "JAX_DISTRIBUTED_INITIALIZED"

def _init_jax_distributed():
    if os.environ.get(_DISTRIBUTED_SENTINEL):
        return
    proc_count = int(os.environ.get("JAX_PROCESS_COUNT",
                     os.environ.get("JAX_NUM_PROCESSES",
                     os.environ.get("SLURM_NTASKS", "1"))))
    if proc_count > 1:
        try:
            jax.distributed.initialize()
            os.environ[_DISTRIBUTED_SENTINEL] = "1"
            return
        except Exception:
            pass
        # Fallback: explicit coordinator from SLURM
        coord = os.environ.get("JAX_COORDINATOR_ADDRESS")
        if coord is None:
            import subprocess
            nodelist = os.environ.get("SLURM_NODELIST")
            if nodelist:
                try:
                    result = subprocess.run(
                        ["scontrol", "show", "hostnames", nodelist],
                        capture_output=True, text=True, check=True)
                    first_host = result.stdout.strip().split("\n")[0]
                    coord = f"{first_host}:12355"
                except Exception:
                    pass
            if coord is None:
                host = (os.environ.get("SLURMD_NODENAME")
                        or os.environ.get("HOSTNAME") or "localhost")
                coord = f"{host}:12355"
        proc_id = int(os.environ.get("JAX_PROCESS_INDEX",
                      os.environ.get("SLURM_PROCID", "0")))
        jax.distributed.initialize(coordinator_address=coord,
                                   num_processes=proc_count,
                                   process_id=proc_id)
    os.environ[_DISTRIBUTED_SENTINEL] = "1"

_init_jax_distributed()

import jax.numpy as jnp
import numpy as np

# Verify multi-process execution
n_procs = jax.process_count()
n_devs = jax.device_count()
local_devs = jax.local_device_count()
proc_id = jax.process_index()

if proc_id == 0:
    print(f"Processes: {n_procs}, Total devices: {n_devs}, "
          f"Local devices per process: {local_devs}")
    print(f"Device: {jax.devices()[0].device_kind}")
    mem_limit = jax.local_devices()[0].memory_stats().get('bytes_limit', 0) / 1e9
    print(f"GPU memory limit: {mem_limit:.1f} GB")

    if n_procs == 1 and n_devs > 1:
        print("\n" + "!" * 70)
        print("ERROR: Single process with multiple GPUs detected!")
        print("This gives WRONG memory numbers. Use: srun -n <num_gpus>")
        print("!" * 70)
        sys.exit(1)

# Import LORRAX modules
from common.wfnreader import WFNReader
from common.symmetry_maps import SymMaps
from common.meta import Meta
from common import timing
from common.isdf_fitting import fit_zeta_chunked_to_h5
from common.load_wfns import load_centroids_band_chunked
from gw.gw_init import compute_optimal_chunks
import configparser


# ============================================================================
# Sweep configurations
# ============================================================================
# Each config is a dict of overrides applied to the base cohsex.in.
# The sweep script runs the full zeta fitting pipeline for each.
#
# Keys:
#   band_chunk: bands per FFT chunk (controls centroid load peak)
#   r_chunk: r-points per zeta chunk (controls pair/ZCT/solve peaks)
#   q_chunk: q-points per solve batch (controls solve peak)
#   n_centroids: number of ISDF centroids (if different from base)
#   nband: total bands (if different from base)
#   label: human-readable name for this config
#
# The sweep covers:
#   - Band chunk impact on centroid load peak (fixing r,q)
#   - R-chunk impact on pair/ZCT/solve peaks (fixing band,q)
#   - Q-chunk impact on solve peak (fixing band,r)

SWEEP_CONFIGS = [
    # --- Band chunk sweep (r_chunk=3456, q_chunk=1) ---
    {"id": "bc09_rc3456_qc01", "band_chunk": 9,  "r_chunk": 3456, "q_chunk": 1,
     "label": "small band chunk"},
    {"id": "bc18_rc3456_qc01", "band_chunk": 18, "r_chunk": 3456, "q_chunk": 1,
     "label": "medium band chunk"},
    {"id": "bc35_rc3456_qc01", "band_chunk": 35, "r_chunk": 3456, "q_chunk": 1,
     "label": "all bands in one chunk"},

    # --- R-chunk sweep (band_chunk=35, q_chunk=1) ---
    {"id": "bc35_rc1152_qc01", "band_chunk": 35, "r_chunk": 1152, "q_chunk": 1,
     "label": "small r-chunk (1/12 of n_rtot)"},
    # bc35_rc3456_qc01 already in band sweep above
    {"id": "bc35_rc6912_qc01", "band_chunk": 35, "r_chunk": 6912, "q_chunk": 1,
     "label": "large r-chunk (1/2 of n_rtot)"},
    {"id": "bc35_rc13824_qc01", "band_chunk": 35, "r_chunk": 13824, "q_chunk": 1,
     "label": "full r (1 chunk)"},

    # --- Q-chunk sweep (band_chunk=35, r_chunk=3456) ---
    {"id": "bc35_rc3456_qc04", "band_chunk": 35, "r_chunk": 3456, "q_chunk": 4,
     "label": "q_chunk=4"},
    {"id": "bc35_rc3456_qc16", "band_chunk": 35, "r_chunk": 3456, "q_chunk": 16,
     "label": "q_chunk=16"},
    {"id": "bc35_rc3456_qc64", "band_chunk": 35, "r_chunk": 3456, "q_chunk": 64,
     "label": "q_chunk=64 (all q at once)"},
]


def parse_mem_lines(output: str) -> list[dict]:
    """Parse [MEM ...] lines from stdout into structured records.

    Each line looks like:
      [MEM isdf | after CCT] used=1.234 peak=5.678 limit=38.1 GB
    or:
      [MEM load_wfns | centroid_load: bc[0] after FFT+extract] used=... peak=... GB
    """
    records = []
    # Use .+? (non-greedy) for the label group instead of [^\]]+
    # because labels can contain ] in chunk indices like chunk[0]
    pattern = re.compile(
        r'\[MEM\s+([^|]+)\|\s*(.+?)\]\s*'
        r'used=([0-9.]+)\s*peak=([0-9.]+)'
        r'(?:\s*limit=([0-9.]+))?'
    )
    for line in output.split('\n'):
        m = pattern.search(line)
        if m:
            records.append({
                'module': m.group(1).strip(),
                'label': m.group(2).strip(),
                'used_gb': float(m.group(3)),
                'peak_gb': float(m.group(4)),
                'limit_gb': float(m.group(5)) if m.group(5) else None,
            })
    return records


def run_one_config(cfg: dict, basedir: str, wfn: "WFNReader", sym: "SymMaps",
                   capture_xprof: bool = False) -> dict:
    """Run the full zeta fitting pipeline for one config, return memory data."""
    cfg_id = cfg["id"]
    band_chunk = cfg["band_chunk"]
    r_chunk = cfg["r_chunk"]
    q_chunk = cfg["q_chunk"]
    nband = cfg.get("nband", 35)
    n_centroids_cfg = cfg.get("n_centroids", 240)

    if jax.process_index() == 0:
        print(f"\n{'='*70}")
        print(f"CONFIG: {cfg_id} — {cfg.get('label', '')}")
        print(f"  band_chunk={band_chunk}, r_chunk={r_chunk}, q_chunk={q_chunk}")
        print(f"  nband={nband}, n_centroids={n_centroids_cfg}")
        print(f"{'='*70}")

    # Load cohsex.in for base parameters
    cohsex_path = os.path.join(basedir, "cohsex.in")
    config = configparser.ConfigParser()
    config.read(cohsex_path)
    params = dict(config["cohsex"])

    # Override parameters
    nval = int(params["nval"])
    ncond = nband - nval
    nspinor = 2
    fft_grid = tuple(int(x) for x in wfn.fft_grid)
    n_rtot = fft_grid[0] * fft_grid[1] * fft_grid[2]
    nk_tot = sym.nk_tot
    kgrid = np.array(wfn.kgrid)

    # Load centroids
    centroids_file = os.path.join(basedir, params["centroids_file"])
    centroid_frac = np.loadtxt(centroids_file, dtype=np.float64)
    n_rmu = len(centroid_frac)
    centroid_indices = np.round(centroid_frac * np.array(fft_grid)[None, :]).astype(np.int32)
    centroid_indices = centroid_indices % np.array(fft_grid)[None, :]

    # Build mesh (auto-detect from available devices)
    devices = np.array(jax.devices())
    n_dev = len(devices)
    # Try square-ish mesh
    p_x = int(np.sqrt(n_dev))
    while n_dev % p_x != 0:
        p_x -= 1
    p_y = n_dev // p_x
    mesh = jax.sharding.Mesh(devices.reshape(p_x, p_y), ("x", "y"))

    if jax.process_index() == 0:
        print(f"  mesh: {p_x}x{p_y}, nk={nk_tot}, n_rmu={n_rmu}, n_rtot={n_rtot}")

    # Build Meta-like object with all attributes fit_zeta_chunked_to_h5 needs
    meta = type("Meta", (), {
        "nk_tot": nk_tot,
        "nspinor": nspinor,
        "nspinor_wfnfile": 2,
        "fft_grid": fft_grid,
        "n_rtot": n_rtot,
        "n_rmu": n_rmu,
        "kgrid": kgrid,
        "memory_per_device_gb": 28,
        "b_id_0": 0,
        "b_id_3": nband,
        "b_id_4": nband,
    })()

    # Compute chunk sizes
    n_q = kgrid[0] * kgrid[1] * kgrid[2]
    num_r_chunks = max(1, (n_rtot + r_chunk - 1) // r_chunk)

    # Output file for this config (temp, will be deleted)
    output_h5 = os.path.join(basedir, "tmp", f"zeta_sweep_{cfg_id}.h5")
    os.makedirs(os.path.dirname(output_h5), exist_ok=True)

    # Optionally wrap in XProf trace
    xprof_dir = None
    if capture_xprof:
        xprof_dir = os.path.join(basedir, "profiles", "xprof", f"{cfg_id}")
        os.makedirs(xprof_dir, exist_ok=True)

    # Reset peak memory counter by reading it (it's monotonic, so we record
    # the baseline and subtract)
    gc.collect()
    baseline_stats = jax.local_devices()[0].memory_stats()
    baseline_peak = baseline_stats.get("peak_bytes_in_use", 0) / 1e9

    # Run the full pipeline, capturing stdout for MEM lines
    t_start = time.perf_counter()
    captured_output = StringIO()

    try:
        if capture_xprof and xprof_dir:
            ctx = jax.profiler.trace(xprof_dir)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            # Tee stdout: print AND capture
            import io

            class TeeWriter(io.TextIOBase):
                def __init__(self, *writers):
                    self.writers = writers
                def write(self, s):
                    for w in self.writers:
                        w.write(s)
                    return len(s)
                def flush(self):
                    for w in self.writers:
                        w.flush()

            old_stdout = sys.stdout
            tee = TeeWriter(old_stdout, captured_output)
            sys.stdout = tee

            try:
                psi_l_Y, psi_l_X, psi_r_Y, psi_r_X = fit_zeta_chunked_to_h5(
                    wfn=wfn,
                    sym=sym,
                    meta=meta,
                    centroid_indices=jnp.asarray(centroid_indices),
                    bispinor=False,
                    mesh_xy=mesh,
                    band_range_left=(0, nband),
                    band_range_right=(0, nband),
                    chunk_r=r_chunk,
                    output_file=output_h5,
                    band_chunk_size=band_chunk,
                    q_chunk_size=q_chunk,
                    use_gspace_cache=True,
                    isdf_pair_mode="spin_traced",
                )
                # Force completion
                psi_l_Y.block_until_ready()
                success = True
                error_msg = None
            except Exception as e:
                success = False
                error_msg = str(e)[:500]
                if jax.process_index() == 0:
                    print(f"  FAILED: {error_msg}")
            finally:
                sys.stdout = old_stdout

    except Exception as e:
        success = False
        error_msg = str(e)[:500]
        captured_output = StringIO()

    t_elapsed = time.perf_counter() - t_start

    # Parse memory records from captured output
    mem_records = parse_mem_lines(captured_output.getvalue())

    # Get final memory stats
    final_stats = jax.local_devices()[0].memory_stats()
    final_peak = final_stats.get("peak_bytes_in_use", 0) / 1e9

    # Clean up GPU memory for next config
    try:
        del psi_l_Y, psi_l_X, psi_r_Y, psi_r_X
    except NameError:
        pass
    gc.collect()
    jax.clear_caches()

    # Clean up temp file
    if os.path.exists(output_h5):
        try:
            os.remove(output_h5)
        except OSError:
            pass

    result = {
        "config_id": cfg_id,
        "config": cfg,
        "success": success,
        "error": error_msg,
        "elapsed_s": round(t_elapsed, 2),
        "n_processes": n_procs,
        "n_devices": n_dev,
        "mesh": f"{p_x}x{p_y}",
        "system": {
            "nk": nk_tot, "nband": nband, "nspinor": nspinor,
            "n_rmu": n_rmu, "n_rtot": n_rtot, "n_q": n_q,
            "fft_grid": list(fft_grid),
        },
        "chunks": {
            "band_chunk": band_chunk, "r_chunk": r_chunk,
            "q_chunk": q_chunk, "num_r_chunks": num_r_chunks,
        },
        "baseline_peak_gb": round(baseline_peak, 3),
        "final_peak_gb": round(final_peak, 3),
        "mem_records": mem_records,
        "xprof_dir": xprof_dir,
    }

    if jax.process_index() == 0:
        print(f"\n  Result: {'OK' if success else 'FAIL'}, "
              f"elapsed={t_elapsed:.1f}s, "
              f"peak={final_peak:.3f} GB, "
              f"{len(mem_records)} mem records captured")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xprof", action="store_true",
                        help="Capture XProf traces for each config")
    parser.add_argument("--config", type=str, default=None,
                        help="Run only this config ID (comma-separated for multiple)")
    parser.add_argument("--basedir", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Base directory with cohsex.in and WFN.h5")
    args = parser.parse_args()

    basedir = args.basedir

    # Load system once
    wfn_path = os.path.join(basedir, "WFN.h5")
    if not os.path.exists(wfn_path):
        print(f"ERROR: {wfn_path} not found")
        sys.exit(1)

    wfn = WFNReader(wfn_path)
    sym = SymMaps(wfn)

    if jax.process_index() == 0:
        print(f"\nSystem: nk={sym.nk_tot}, fft={wfn.fft_grid}, "
              f"kgrid={wfn.kgrid}")

    # Filter configs if requested
    configs = SWEEP_CONFIGS
    if args.config:
        requested = set(args.config.split(","))
        configs = [c for c in configs if c["id"] in requested]
        if not configs:
            print(f"ERROR: no configs match {args.config}")
            print(f"Available: {[c['id'] for c in SWEEP_CONFIGS]}")
            sys.exit(1)

    if jax.process_index() == 0:
        print(f"\nRunning {len(configs)} configurations:")
        for c in configs:
            print(f"  {c['id']}: {c.get('label', '')}")

    # Run sweep
    all_results = []
    for i, cfg in enumerate(configs):
        if jax.process_index() == 0:
            print(f"\n[{i+1}/{len(configs)}] Running {cfg['id']}...")

        result = run_one_config(cfg, basedir, wfn, sym,
                                capture_xprof=args.xprof)
        all_results.append(result)

        # Sync between configs
        jax.experimental.multihost_utils.sync_global_devices(f"sweep_config_{i}")

    # Write results (only rank 0)
    if jax.process_index() == 0:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        outfile = os.path.join(basedir, f"sweep_results_{timestamp}.json")
        with open(outfile, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults written to {outfile}")

        # Print summary table
        print(f"\n{'='*90}")
        print(f"SWEEP SUMMARY")
        print(f"{'='*90}")
        print(f"{'Config':<25s} {'OK':>3s} {'Time':>7s} "
              f"{'Peak':>7s} {'Records':>7s}")
        print(f"{'-'*90}")
        for r in all_results:
            ok = "Y" if r["success"] else "N"
            print(f"{r['config_id']:<25s} {ok:>3s} "
                  f"{r['elapsed_s']:>7.1f}s "
                  f"{r['final_peak_gb']:>6.2f}G "
                  f"{len(r['mem_records']):>7d}")

        # Per-stage peak summary for each config
        print(f"\n{'='*90}")
        print(f"PER-STAGE PEAK MEMORY (GB, device 0)")
        print(f"{'='*90}")
        for r in all_results:
            if not r["success"]:
                continue
            print(f"\n--- {r['config_id']} ---")
            # Group by stage and find max peak within each stage
            stages = {}
            for rec in r["mem_records"]:
                # Extract stage name from label
                label = rec["label"]
                stages[label] = {
                    "used": rec["used_gb"],
                    "peak": rec["peak_gb"],
                }
            for label, data in stages.items():
                print(f"  {label:<50s} used={data['used']:>6.3f} "
                      f"peak={data['peak']:>6.3f}")


if __name__ == "__main__":
    main()
