# cuSOLVERMp FFI profiling orientation (2026-05-12)

## Summary

Oriented on `lorrax_D` branch `agent/zeta-ibz-header` for the cuSOLVERMp / cuBLASMp FFI overhead investigation. No LORRAX source changes were made in this pass.

The branch is 27 commits ahead of `origin/agent/zeta-ibz-header`; the source tree has one untracked `profile/` directory containing only `compile.log` and `pf_setup.json`. The sandbox top-level currently tracks `sources/lorrax_D` as a subproject at an older commit, so `git diff` at the sandbox level reports a subproject pointer change.

## Profiling workflow notes

The active cluster hooks are `lxalloc`, `lxrun`, `lxshell`, and `lxpre`; no `lxattach` hook was found under sandbox scripts, skills, or `sources/lorrax_D`. `lxrun` wraps `srun -> select_gpu.sh -> shifter -> in_container.sh -> user command`, with Cray MPICH, one GPU per rank, and `MPICH_GPU_SUPPORT_ENABLED=1` reasserted inside Shifter.

Canonical profile command for GWJAX remains:

```bash
cd <run_dir>
LORRAX_NGPU=4 lxrun python3 -u \
  /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/run_profiled.py \
  --out profile -m gw.gw_jax -i cohsex.in
```

Then run the three analyzers from the sandbox root. For Nsight Systems, prefer putting `nsys profile` inside the `lxrun` command so it profiles the containerized Python process rather than mostly the outer `srun`/Shifter startup. The suggested `--capture-range=cudaProfilerApi` requires explicit CUDA profiler start/stop hooks; none were found in current source, so use full-process capture or add a tiny bracketed hook before relying on that option.

## FFI hot paths

`src/ffi/cusolvermp/cpp/batched_potrf_ffi.cc` and `batched_potrs_ffi.cc` match the reported overhead pattern: per FFI call they bridge streams with pooled CUDA events, optionally copy aliased buffers, create matrix descriptors, query workspace sizes, loop over q slices, destroy descriptors, then bridge back to XLA.

The same setup-cost shape appears in `src/ffi/cublasmp/cpp/batched_gemm_ffi.cc`. `src/ffi/cublasmp/cpp/batched_w_solve_ffi.cc` already fuses a larger operation into one handler, but still creates descriptors and sizes workspaces per call. `src/ffi/cusolvermp/cpp/ctx.h` is the natural home for descriptor/workspace caches shared by these handlers.

## Recommended next edits

1. Add NVTX ranges around descriptor create, buffer-size query, per-q solver calls, and cross-stream waits. This is the smallest first patch and makes the Nsight trace answer which setup costs are real.
2. Add a descriptor + buffer-size cache on `LorraxCusolverMpCtx`, keyed by operation, dtype, global shape, block sizes, lld, and transpose/layout attributes. Apply first to `potrf`/`potrs`, then generalize to LU and cuBLASMp GEMM.
3. Profile MoS2 3x3 with Nsight Systems and NCCL debug after H1 to decide whether CUDA graph capture across the q loop is worth the risk.

## Sandbox bookkeeping

Logged two sandbox issues in `KNOWN_SANDBOX_ERRORS.md`: the active source path is `sources/lorrax_D` rather than documented `sources/lorrax`, and the recent MoS2 D variant directories named in reports lack per-variant manifests.
