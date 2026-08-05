# LORRAX $HOME-dependency Migration — Fresh-Checkout Validation

**Date:** 2026-06-24
**Allocation:** SLURM job 54985309 (1 node nid001136, 4× A100-40GB), alive throughout.
**Branch validated:** `agent/dep-home-migration` (fresh `git clone` of `sources/lorrax_D`).
**Verdict:** **Migration is SAFE to finalize.** The relocated `$HOME/software` deps link
and run end-to-end; a real 4-GPU GW calc produced valid `eqp0.dat`. The only friction is
two doc/launch-recipe gaps in SKILL.md (below) — neither is caused by the relocation.

---

## (a) Did the fresh checkout + $HOME-dep linkage work end-to-end? — YES

- `git clone /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D /pscratch/sd/j/jackm/lorrax_freshtest_val`
  then `git checkout agent/dep-home-migration` — clean, branch is the clone default (origin/HEAD).
- In-container dep visibility (SKILL.md SHIFTER prefix, bind mounts to `$HOME/software`):
  all three present —
  - `/lorrax_phdf5/lib/libhdf5.so` ✓
  - `/lorrax_slate/lib/libmpi_gtl_cuda.so.0` ✓
  - `/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64/libcusolverMp.so{,.0,.0.7.2.0}` ✓
- **`ldd liblorrax_ffi.so` inside the container resolves EVERY relocated lib to a `$HOME`
  path — zero `/pscratch` paths, zero "not found":**
  - `libcusolverMp.so.0` → `/lorrax_nvhpc/0.7.2_cuda12.9/...` (HOME)
  - `libcublasmp.so.0`, `libcal.so.0` → `/lorrax_nvhpc/25.5_cuda12.9/...` (HOME) — note the
    run pulls cublasmp/cal from the **25.5** tree as well as cusolverMp from 0.7.2
  - `libhdf5_parallel_gnu_123.so.200` → `/lorrax_phdf5/...` (HOME)
  - `libslate.so.2 / libblaspp.so.2 / liblapackpp.so.2` → `/global/homes/.../software/slate/install/lib64`
  - `libmpi_gtl_cuda.so.0 / libsci_gnu_* / libxpmem.so.0` → `/lorrax_slate/...` (HOME)

## (b) Test calc — command and key output

Reused preprocessed inputs from `runs/MoS2/02_mos2_3x3_nosym/41_ffi_e2e_C` (MoS2 3×3,
80 bands, `use_ffi_io=true`, `do_screened=true`), copied (symlinks dereferenced) to
`/pscratch/sd/j/jackm/lorrax_freshtest_val_run`. PYTHONPATH pointed at the **fresh**
checkout `…/lorrax_freshtest_val/src`; `LORRAX_FFI_SO` pointed at the original built
`.so` (see issue 1).

```
source shifter_env.sh   # SKILL.md GWJAX prefix + LORRAX_FFI_SO + PYTHONPATH=fresh/src
srun --jobid=54985309 --gres=gpu:4 -N1 -n4 $SHIFTER \
    --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; \
             exec python3 -u -m gw.gw_jax -i $PWD/cohsex.in'
```

Key output (success):
- Banner: `Backend: GPU  Devices: 4  Mesh: 2×2  Processes: 4`
- `[lorrax cusolverMp] library 0.7.2, NCCL 2.26.3, comm path: NCCL, grid: 2x2` ← cuSolverMp
  loaded from `$HOME` and used (`path=cusolvermp_cholesky` for L_q = chol(C_q))
- phdf5 SlabIO wrote + collectively H5Fclose'd `zeta_q.h5`, `isdf_tensors_640.h5` ← FFI phdf5 from `$HOME`
- Produced: `eqp0.dat`, `eqp1.dat`, `eqp_g0w0.dat`, `sigma_diag.dat`, `sigma_freq_debug.dat`,
  `sigma_mnk.h5`, `WFN_qp.h5`. Total recorded 20.2 s.
- Sample value (`sigma_diag.dat`, k=0 n=0): `sigX=-40.032649  sigC=4.157183 + -0.005431i  sigXC=-35.875467`.

## (c) Issues / friction points

1. **Fresh clone has NO compiled `liblorrax_ffi.so` → a true newcomer cannot run without a
   build step.** `.gitignore` (lines 71-72) excludes `src/ffi/**/cpp/build/` and
   `src/ffi/**/*.so`, so the single runtime lib
   `src/ffi/common/cpp/build/liblorrax_ffi.so` is absent in the checkout. `ffi_loader.py`
   would raise `FileNotFoundError: Could not locate liblorrax_ffi*.so. Build with: bash
   src/ffi/common/cpp/build.sh`. **Cause: build artifact, not migration.** Out of scope to
   rebuild (needs the in-container NVHPC/Cray toolchain). I worked around it by setting
   `LORRAX_FFI_SO` to the original source's already-built `.so`, which is the realistic
   newcomer fallback. NB: the SKILL.md GWJAX prefix sets `PYTHONPATH` to the *original*
   source tree (`…/sources/lorrax_D/src`), which already contains a built `.so`, so the
   documented happy path never exercises a from-scratch FFI build.

2. **SKILL.md Step 6 GW command is missing per-rank GPU binding → "Duplicate GPU detected"
   crash.** First run (verbatim SKILL.md command, no `CUDA_VISIBLE_DEVICES`) died with:
   `XlaRuntimeError: INTERNAL: NCCL operation ncclGroupEnd() failed: invalid usage … 'Duplicate
   GPU detected : rank 1 and rank 0 both on CUDA device 3000'` (all 4 ranks grabbed device 0).
   Cause: `runtime.init_jax_distributed()` (src/runtime/__init__.py:137-141) derives
   `local_device_ids` from `CUDA_VISIBLE_DEVICES`; when it is unset every rank sees all 4
   GPUs and `jax.distributed.initialize()` runs with no args → each process owns device 0.
   The code's own docstring states the Perlmutter contract is one GPU per rank via
   `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`, but the SKILL.md `srun` recipe does not set it and
   even says *"Never set CUDA_VISIBLE_DEVICES … JAX auto-detects GPUs from SLURM"* (line 164).
   **Fix that worked:** wrap the python call: `bash -c 'export
   CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; exec python3 -u -m gw.gw_jax -i …'`. **Cause: stale
   launch recipe, NOT migration.** (Pre-existing — would have failed before the move too.)

## (d) Docs still wrong/stale in the updated SKILL.md

- **Step 6 GW launch command** omits the required `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`
  per-rank binding (issue 2). As written it crashes on Perlmutter. The "Never set
  CUDA_VISIBLE_DEVICES" pitfall note (line 164) directly contradicts
  `runtime.init_jax_distributed()` and should be corrected. (This pre-dates the migration.)
- The GWJAX intro claims the Shifter image is "JAX 0.7.2 / Python 3.12"; the running
  `nvcr.io/nvidia/jax:25.04-py3` + `isdf_site` overlay reports `jax 0.5.3.dev20260624`
  in-container. Minor cosmetic drift; harmless.
- The `$HOME/software` bind-mount paths, nvhpc `0.7.2_cuda12.9` lib64 path, phdf5/slate
  stage paths, and `LD_LIBRARY_PATH` (incl. `…/software/slate/install/lib64`, which is
  reached via shifter's auto-mount of `/global/homes`, not via `$VOL`) are all **correct**.

## (e) Verdict — SAFE TO FINALIZE (delete scratch originals)

The relocation to `$HOME/software` is functionally complete and correct: all four dep
families bind-mount, the FFI `.so` links exclusively against `$HOME` paths (verified by
`ldd`), and a real 4-GPU screened-GW calc ran to completion producing valid sigma/eqp0
output using cuSolverMp + phdf5 loaded from `$HOME`. The two issues found are both
pre-existing (gitignored FFI build artifact; stale GPU-binding recipe) and independent of
the dependency move — neither is a reason to keep the scratch copies.

Caveat noted but not blocking: only nvhpc `0.7.2_cuda12.9` and `25.5_cuda12.9` were copied
to `$HOME`; `0.7.0`/`0.8.0` remain only on scratch. The current production recipe uses
0.7.2 (cusolverMp) + 25.5 (cublasmp/cal), both present and exercised here. If any workflow
pins 0.7.0/0.8.0, copy those before deleting scratch; otherwise safe.
