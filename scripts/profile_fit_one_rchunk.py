"""Probe: dump XLA HLO buffer-assignment from an AOT-compiled
fit_one_rchunk to see what XLA actually allocates vs my primitive
model.  No pf.setup_env — it has distributed-init issues under
bare lxrun on Perlmutter.  Uses XLA_FLAGS directly.
"""
import os
import sys
from pathlib import Path

OUT = Path("/pscratch/sd/j/jackm/lorrax_sandbox/reports/aot_memory_model_poc_2026-04-20/buffer_dumps")
OUT.mkdir(parents=True, exist_ok=True)

# MUST set before jax import
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "")
    + f" --xla_dump_to={OUT}"
    + " --xla_dump_hlo_as_text"
    + " --xla_dump_hlo_as_proto=false"
).strip()

sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")

import jax
jax.config.update("jax_enable_x64", True)

# Distributed init matches production's one-GPU-per-rank layout
cv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
n_local = len(cv.split(",")) if cv else 1
if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    jax.distributed.initialize(local_device_ids=list(range(n_local)))

import numpy as np
from jax.sharding import Mesh
from gw.aot_memory_model import (
    SysDims, MeshSpec, Knobs, get_kernel, aot_measure, load_fit,
)

devs = jax.devices()
if jax.process_index() == 0:
    print(f"\n========== n_devices = {len(devs)} ==========\n")

mesh = Mesh(np.asarray(devs[:4]).reshape(2, 2), ("x", "y"))

CONFIGS = {
    "mos2_3x3": dict(
        sys=SysDims(kgrid=(3, 3, 1), fft_grid=(80, 72, 8),
                    n_rmu=640, n_s=2, n_b=80, n_b_sum=160,
                    n_r=80 * 72 * 8),
        knobs=Knobs.of(chunk_r=46080, band_chunk=80),
    ),
    "si_4x4x4": dict(
        sys=SysDims(kgrid=(4, 4, 4), fft_grid=(24, 24, 24),
                    n_rmu=480, n_s=1, n_b=60, n_b_sum=120,
                    n_r=24 ** 3),
        knobs=Knobs.of(chunk_r=24 ** 3, band_chunk=60),
    ),
}

kernel = get_kernel("fit_one_rchunk")
mspec = MeshSpec(2, 2)


def section(title):
    if jax.process_index() == 0:
        print(f"\n{'='*72}\n{title}\n{'='*72}")


fit = load_fit("fit_one_rchunk", tag="current")

for name, cfg in CONFIGS.items():
    section(f"{name}")
    sys_ = cfg["sys"]
    knobs = cfg["knobs"]

    # primitive-level prediction
    if jax.process_index() == 0:
        print(f"\n[A] primitive prediction (γ={fit.gamma}):")
        total_raw = 0.0
        for fname, beta in zip(fit.feature_names, fit.coefs):
            T = kernel.PRIMITIVES[fname](sys_, knobs, mspec)
            contrib = beta * T
            total_raw += contrib
            cls = kernel.PRIMITIVE_CLASSES[fname]
            print(f"  β[{fname:12s}] = {beta:5.2f}  T = {T/1e6:>8.1f} MB  "
                  f"→ β·T = {contrib/1e9:>6.3f} GB  ({cls})")
        print(f"  intercept                           = "
              f"{fit.intercept/1e9:>6.3f} GB")
        raw = total_raw + fit.intercept
        print(f"  Σβ·T raw                            = {raw/1e9:>6.3f} GB")
        print(f"  Σβ·T · γ                            = "
              f"{raw*fit.gamma/1e9:>6.3f} GB")

    # compile + memory_analysis
    meas = aot_measure(kernel, sys_, knobs, mesh)
    if jax.process_index() == 0:
        print(f"\n[B] compiled.memory_analysis() (bytes per device):")
        for k in ("argument", "output", "temp", "alias", "total"):
            v = meas.get(k, 0)
            print(f"  {k:10s} = {v/1e9:>6.3f} GB")
        print(f"\n[C] cost_analysis flops = {meas['flops']/1e9:.1f} G")

if jax.process_index() == 0:
    print(f"\n\nHLO + buffer-assignment dumped under: {OUT}")
    # List what's there
    dumped = sorted(OUT.glob("*"))
    print(f"  {len(dumped)} files written")
    for p in dumped[:30]:
        print(f"    {p.name}")
