# Known Sandbox Errors

This file tracks documentation errors, broken paths, incorrect instructions, and
other issues agents encounter while using this sandbox. It exists so that problems
are recorded immediately rather than rediscovered by the next agent.

**If you are an agent and you encounter an error caused by incorrect documentation,
a missing file, a wrong path, a stale reference, or any other sandbox infrastructure
problem: record it here immediately before continuing your work.** Use the format below.

Do not fix the documentation yourself unless the fix is trivial and obvious. Record
the issue here for human review. Fixing incorrect physics or code is your job; fixing
incorrect sandbox scaffolding is the human's job.

---

## Format

```
### YYYY-MM-DD: Short description
- **Where**: which file/doc had the error
- **What happened**: what you tried and what went wrong
- **Expected**: what should have happened
- **Workaround**: what you did instead (if anything)
```

---

## Issues

### 2026-05-16: bispinor 16-GPU retest spec uses `LORRAX_NGPU=16` (total) but lxrun reads it as per-node → gres error
- **Where**: bispinor IBZ 16-GPU retest task spec ("Run A" / "Run B" sequence), which sets `SLURM_JOBID=53054263 LORRAX_NGPU=16 lxrun python3 -u -m gw.gw_jax ...`.
- **What happened**: `lxrun` (defined in `modulefiles/lorrax_agent/1.0.lua:170-222`) treats `LORRAX_NGPU` as GPUs **per node** and builds `--gres="gpu:${LORRAX_NGPU}"`. With `LORRAX_NGPU=16` on Perlmutter (4 GPUs/node), srun rejected the step: `srun: error: Unable to create step for job 53054263: Invalid generic resource (gres) specification`. Run died in 2 s with no gw.out beyond the lxrun banner.
- **Expected**: For a 16-rank job on 4 nodes, callers should use `LORRAX_NNODES=4 LORRAX_NGPU=4` (total_ranks = nnodes × ngpu_per_node = 16). This matches the in-tree launch helper `runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/run_16gpu_v2.sh`.
- **Workaround**: Relaunched Run A with `LORRAX_NNODES=4 LORRAX_NGPU=4`. Task spec should be updated to specify per-node and node-count separately, or rename `LORRAX_NGPU` semantics to total-GPUs in the module.

### 2026-05-16: Shifter env passthrough — `LORRAX_FORCE_FULL_BZ=1` set in the shell does not reach the container
- **Where**: bispinor IBZ retest Run A spec instructs `export LORRAX_FORCE_FULL_BZ=1; lxrun python3 -u ...`.
- **What happened**: `lxrun` invokes the workload inside `shifter`, which only forwards env vars explicitly listed via `--env=` (see `LORRAX_SHIFTER` in the base `lorrax_B` module). A plain `export` from the calling shell is not seen by the in-container python — Run A would silently fall back to the IBZ cascade if it has been activated globally, defeating the "force full-BZ" guard.
- **Expected**: The spec should propagate guard env vars via `LORRAX_SHIFTER` (or document the shifter-env idiom). E.g. `LORRAX_SHIFTER="$LORRAX_SHIFTER --env=LORRAX_FORCE_FULL_BZ=1" lxrun ...`.
- **Workaround**: Used the `LORRAX_SHIFTER` extension idiom at launch.

### 2026-05-12: Active LORRAX tree is `sources/lorrax_D`, but top-level docs still reference `sources/lorrax`
- **Where**: `AGENTS.md` source-code table and non-negotiable rule 5; `skills/checkpoint/SKILL.md` git command examples.
- **What happened**: Following the documented `sources/lorrax` path failed because this sandbox checkout has `sources/lorrax_D` and `sources/lorrax_D_old`, but no `sources/lorrax` symlink.
- **Expected**: The docs should either point at the active `sources/lorrax_D` tree or provide a `sources/lorrax` symlink.
- **Workaround**: Treat `sources/lorrax_D` as the active LORRAX source for this session, matching the user's request to inspect "lorrax D".

### 2026-05-12: MoS2 D variant run directories lack per-variant manifests
- **Where**: `runs/MoS2/00_mos2_3x3_cohsex/D_perf_after_2026-05-12/` and `D_gflat_bispinor_shardmap_2026-05-12/`; top-level `AGENTS.md` says every run directory must have a `manifest.yaml`.
- **What happened**: Attempting to read the active variant manifests failed with `No such file or directory`; the parent `runs/MoS2/00_mos2_3x3_cohsex/manifest.yaml` still lists old steps as `pending`.
- **Expected**: Each D variant run should include a manifest or the parent manifest should track the D variants' states.
- **Workaround**: Use the corresponding report files and `profile_launch.log` / output files for D-run status until manifests are repaired.

### ~~2026-04-04: kin_ion.h5 not mentioned in BUILD_INPUTS or run setup docs~~ FIXED
- **Fix**: Added `kin_ion_file = kin_ion.h5` documentation to `skills/build_inputs/SKILL.md` Step 6 (GWJAX input), describing the file, its shape, how to generate it, and its dependencies.

### ~~2026-04-05: centroid.kmeans_isdf does not accept `-i cohsex.in`~~ FIXED
- **Fix**: Removed `-i cohsex.in` from both invocations in `skills/execute_workflow/SKILL.md` (Perlmutter step 5a and local step 5). The module reads `WFN.h5` from the CWD; correct invocation is `python3 -m centroid.kmeans_isdf $N_CENTROIDS --no-plot --seed 42`.

### ~~2026-05-14: `centroid.kmeans_isdf` has no `__main__` block; CLI lives in `centroid.kmeans_cli`~~ FIXED
- **Where**: invocations of `python3 -m centroid.kmeans_isdf 300 --no-plot --seed 42` (as in the lorrax_agent overlay `lxpre` shell function at `modulefiles/lorrax_agent/1.0.lua:278` and in `skills/execute_workflow/SKILL.md`).
- **What happened**: `python3 -m centroid.kmeans_isdf 300 --no-plot --seed 42` exits 0 with NO output written and no centroids file produced; the module has no `if __name__ == "__main__":` block.
- **Expected**: invocation should produce `centroids_frac_<N>.txt`. The CLI entrypoint is `centroid.kmeans_cli`, not `centroid.kmeans_isdf`. Also, `--no-plot` does not exist; the correct flag is the absence of `--plot` (which is opt-in, default False).
- **Fix (2026-05-14, lorrax_B `agent/trs-aware-sym-fix` commit 69ab42c + this sandbox edit)**: Updated every doc + CLI-invocation reference to `centroid.kmeans_cli` (in-tree: `pyproject.toml`, `AGENTS.md`, `README.md`, `docs/{CODEBASE,PHYSICS,ENVIRONMENT}_COMPREHENSIVE.md`, `docs/index.md`, `config/modulefiles/lorrax/0.1.0.lua`).  Sandbox: `modulefiles/lorrax_agent/1.0.lua` `lxpre`, `skills/build_inputs/SKILL.md`, `skills/execute_workflow/SKILL.md`, `skills/profiling_stack/SKILL.md`.  Algorithm-library references to `kmeans_isdf` (test imports of `weighted_kmeans_jax`, module-tree listings) are intentionally preserved.

### ~~2026-05-14: Bispinor V_q silently falls back to scalar V_q if `centroids_file_current` unset~~ FIXED
- **Where**: `sources/lorrax_B/src/gw/gw_init.py::fit_zeta` bispinor branch.
- **What happened**: For a `cfg.bispinor=True` run without `centroids_file_current = ...` in `cohsex.in`, the bispinor ζ-fit branch silently skipped, the downstream V_q orchestrator silently fell back to scalar V_q, and the run crashed much later with a misleading full-BZ vs IBZ shape mismatch (`ζ_L on disk has 36 q's; resolved IBZ has 8`).
- **Fix (2026-05-14, lorrax_B `agent/trs-aware-sym-fix` commit 69ab42c)**: Added a loud-fail guard at the bispinor entry — `fit_zeta` now raises `ValueError` with a clear message pointing the user at `centroid.kmeans_cli --density-mode current ...` if `cfg.bispinor=True` and `cfg.paths.centroids_file_current` is unset.

### ~~2026-05-14: `gw.kin_ion_io` calls non-existent `SymMaps.get_gvecs_kfull`~~ FIXED
- **Where**: `sources/lorrax_B/src/psp/dft_operators.py:142` (called from `sources/lorrax_B/src/gw/kin_ion_io.py:196` and `psp/get_dipole_mtxels.py`).
- **What happened**: Running `python3 -m gw.kin_ion_io -i cohsex.in` (and `psp.get_dipole_mtxels`) crashes with `AttributeError: 'SymMaps' object has no attribute 'get_gvecs_kfull'` after processing the first k-point.
- **Fix (2026-05-14, on `agent/trs-aware-sym-fix`)**: `psp/dft_operators.py::generate_gvectors_k` now dispatches through `WfnLoader.gvecs(k="full_bz")` + `ngk_valid(...)` (the post-P5 location of the moved API). Mirrors the cached pattern already in `psp/get_DFT_mtxels.py::_gvecs_full_cache`. Verified: `psp.get_dipole_mtxels` and `gw.kin_ion_io` both run end-to-end on MoS2 3×3 SOC (`runs/MoS2/03_mos2_3x3_soc_2026-05-14/00_lorrax/`).

### ~~2026-04-05: `uv run` from sandbox fails because editable LORRAX path points to missing `/pscratch` location~~ FIXED
- **Fix**: Changed `pyproject.toml` editable path from `"../lorrax"` to `"./sources/lorrax"`. Ran `uv sync` to rebuild the venv. LORRAX submodules (`gw`, `centroid`, `psp`) now import correctly via `uv run`.

### 2026-05-12: `scripts/profiling/pf.py` hangs multi-process JAX init on current Cray MPICH stack
- **Where**: `scripts/profiling/pf.py:66-107` (`_maybe_init_jax_distributed`).
- **What happened**: Running `lxrun python3 -u .../run_profiled.py --out profile -m gw.gw_jax -i cohsex.in` on any multi-rank allocation (LORRAX_NGPU>=2 OR LORRAX_NNODES>=2) hangs in `jax.distributed.initialize()`'s topology exchange and dies after 2 minutes with `Getting local topologies failed: GetKeyValue() timed out with key: cuda:local_topology/cuda/{1,2,3,...}`.
- **Expected**: pf.py's bootstrap should mirror the canonical `runtime.init_jax_distributed()` in `sources/lorrax_X/src/runtime/__init__.py:109-152`, which explicitly handles the Perlmutter case where each rank only sees one GPU via `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`. That code passes `local_device_ids=[0]` derived from CUDA_VISIBLE_DEVICES — and its docstring says verbatim: *"jax.distributed.initialize() with no args then hangs in the topology exchange because it assumes each process owns all local GPUs."*
- **Workaround**: Edited `scripts/profiling/pf.py` `_maybe_init_jax_distributed()` to delegate to `runtime.init_jax_distributed()` (the canonical implementation). One source of truth for multi-process JAX init across all LORRAX entrypoints.

### 2026-05-17: Round 4 task referenced expired SLURM JIDs 53075110/53075115 and misdiagnosed cache-leak site
- **Where**: Agent K Round-4 task message (memory_model_refit_2026-05-17 initiative).
- **What happened**:
  - Both JIDs (53075110, 53075115) had expired at session start; no active allocations under `$USER` or queue-wide; `lorrax_agent` overlay also failed to load (`module load lorrax_agent` → "module(s) are unknown"). Could not run live verification (V1/V3/V5 reruns, live_arrays probe of buffer count).
  - Task message says "Open `src/common/fft_helpers.py` and find the `make_flat_k_fft` cache (Agent G's report cites the exact site — it's a module-level dict)" — but `fft_helpers.py` has NO module-level dict for sphere/g_index buffers. The `_fft_workspace_cache` there caches *integer peak-byte counts*, not arrays. The real device-resident `(nk, nx, ny, nz) int32` g_index buffer is cached at `common/psi_G_store.py:174` (`self._g_index_dev`), and a fresh one is created per `fit_zeta_to_h5` call because a new `psi_G_store` is constructed each time. Prior agents (G, H) propagated the misdiagnosis ("make_flat_k_fft cache").
- **Expected**: Either an unexpired JID or instructions for spinning a fresh `lxalloc` were needed; and the cache fix should be applied at the real allocation site.
- **Workaround**: Proceeded with source-code commits (fix at real site in `psi_G_store.py`/`wfn_loader.py`, planner override threading, planner term refresh) + pytest verification; live verification deferred. Documented the diagnostic correction in commit 1's body.

## 2026-05-17 — device.memory_stats() returns None on JAX 0.8 / CUDA 12.9

On the Perlmutter shifter stack (JAX 0.8.x, CUDA 12.9), `jax.devices()[0].memory_stats()` returns `None` instead of the expected dict with `peak_bytes_in_use`. This silently makes any HBM probe that relies on it useless — earlier rounds (3, 5, 6) were measuring `jax.live_arrays()` byte-sum-divided-by-16 as a proxy, which conflates sharded vs replicated and is ~7× lower than nvidia-smi peak.

Workaround (commit `6ba1fad` on lorrax_B `agent/bispinor-ibz`): `_mem_probe` falls back to `nvidia-smi --query-gpu=memory.used` per-rank when `memory_stats()` is None. nvidia-smi is the only OOM-faithful metric on this stack.

Affected: anything that compares planner predictions to runtime HBM. Round 7 (`agent_n_faithfulness_audit.md`) used the nvidia-smi fallback throughout.

## 2026-05-17 — nvidia-smi `memory.used` is NOT the OOM-relevant peak under `XLA_PYTHON_CLIENT_ALLOCATOR=platform`

The sandbox default `XLA_PYTHON_CLIENT_ALLOCATOR=platform` (cudaMallocAsync) + `XLA_PYTHON_CLIENT_PREALLOCATE=false` releases pages back to the OS aggressively between operations. nvidia-smi samples at second-timescale and misses microsecond-scale in-jit peaks. Round 7's `agent_n_faithfulness_audit.md` mistakenly concluded the planner over-predicts by 7-8× based on this metric.

Round 8 `agent_o_allocator_audit.md` corrects this: under BFC + preallocated, true XLA-arena peak is 76.05 GB/dev at the same config where nvidia-smi-under-platform read 8.67 GB/dev. Planner predicts 66.41 — actually UNDER-predicts the true peak by 14%.

**Rule of thumb:** for OOM-relevant HWM measurement, either (a) run with `XLA_PYTHON_CLIENT_ALLOCATOR=default` + `XLA_PYTHON_CLIENT_PREALLOCATE=true` + `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` to access `device.memory_stats()`, or (b) trust the planner's static prediction as a lower-bound. Do NOT cite nvidia-smi peaks under platform allocator as a tight HBM measurement.

### 2026-05-19: WITHDRAWN — `nspinor=1` is an unsupported code path
- The prior 2026-05-18 entry described `unfold_psi` / `WfnLoader._ensure_phdf5_static` shape mismatches when a `noncolin=.false. lspinorb=.false.` Si WFN.h5 is loaded. Production LORRAX runs always use fully-relativistic pseudopotentials with `noncolin=.true.` (and typically `lspinorb=.true.`), giving `nspinor=2`. Non-bispinor in LORRAX means `bispinor=false` in `cohsex.in`, NOT `nspinor=1` at the QE level.
- The two `lorrax_A` commits (`8c18925`, `dc0b254`) that "fixed" the broadcast in the ns=1 codepath were reset out of `agent/si-nonbispinor-mu-sweep` on 2026-05-19. The unsupported path stays unsupported.

### 2026-05-19: WITHDRAWN — `cusolverMpPotrf status=7` is hbm40g + BFC@0.95, not a bug
- The prior 2026-05-18 entry framed this as a sandbox bug to be worked around by disabling cusolverMp. That was wrong:
  - The Si bispinor μ-sweep (`agent_t`, JID 53096549) ran on the same 2×2 mesh / BFC+0.95 / cusolverMp-on combination and completed cleanly, because it was on an `hbm80g` allocation (80 GB/GPU → 76 GB BFC pool, 4 GB free for NCCL/CAL).
  - The failed runs (JID 53097982) were on plain `lxalloc` = `hbm40g` (40 GB/GPU → 38 GB BFC pool, 2 GB free, NCCL user-buffer registration starves).
  - Both `sources/lorrax_B/docs/ENVIRONMENT_COMPREHENSIVE.md` §3.2 + §8.3 and the user's `feedback_lxalloc_gpu_constraint_mixes_hbm.md` memory already document the right answer.
- **Operating guideline**: For BFC + `MEM_FRACTION=0.95` measurements on a 2D mesh, use `salloc --constraint="gpu&hbm80g" -J lx-alloc-$USER`. Don't toggle `cusolvermp_*=off` — cuSOLVERMp is a shipping default of the community release.

## 2026-06-15 — JAX version triple-mismatch: pyproject pins >=0.9.0/cuda13, Shifter image ships ~0.8/cuda12, skill says 0.7.2

Three in-repo sources disagree on the JAX version, and current `main` (`e85be60`)
added a JAX-0.9-only symbol with no fallback. Surfaced while planning the rebase of
an old-main checkout to `e85be60` for the CrI3 6×6 GN-PPM study.

- `sources/lorrax_*/pyproject.toml` (lines ~9-10) pins **`jax[cuda13]>=0.9.0`** — a
  *build requirement* the running environment does not satisfy. This pin already
  existed at old main `0f355b7` (not introduced by the rebase).
- The Perlmutter module sets the Shifter image from
  `config/perlmutter/site_config.sh:32` → `LORRAX_IMAGE="nvcr.io/nvidia/jax:25.04-py3"`,
  a **CUDA-12** image shipping a JAX in roughly the **0.5–0.8** range — NOT 0.9, NOT
  cuda13. The `isdf_site` venv layered via PYTHONPATH deliberately contains no jax
  (only `jaxtyping`), so the container's JAX is authoritative.
- `skills/execute_workflow/SKILL.md:136` claims **JAX 0.7.2**; `docs/MEMORY_MODEL.md:922`
  (in lorrax source) says **JAX 0.8 / CUDA 12.9**. Three different numbers.

**The new landmine (added by the rebase to `e85be60`):** commit `c7e6695` introduced
`lax.pcast(...)` at `src/common/cholesky_2d.py:186` (old main had `jnp.zeros` there).
`lax.pcast` does not exist in JAX <0.9 → `AttributeError` at trace time **if that code
path executes**. It is reached only via the in-tree `sharded_cholesky` ζ-fit path,
which on a 2D mesh with the production default `cusolvermp_charge=auto`
(`isdf_fitting.py:943-953` → `cusolvermp_cholesky` FFI) is **not** taken. So CrI3 6×6
GN-PPM on 16 GPUs (4×4 mesh) is expected to run; the landmine only fires if a run
forces `cusolvermp_charge=off` or uses a 1D mesh. `ppm_sigma._to_host_np`
(`src/gw/ppm_sigma.py:240-248`) wraps its `process_allgather(tiled=False)` calls in
try/except → JAX-0.9 `tiled=` strictness degrades gracefully, not a crash.

**Guidance:** On the currently-pinned `jax:25.04-py3` container, do NOT force
`cusolvermp_charge=off` and do NOT run a 1D mesh on current main. Fully satisfying the
`>=0.9.0`/cuda13 pin requires bumping `LORRAX_IMAGE` to a JAX-0.9/CUDA-13 NGC build —
an environment task orthogonal to the git rebase. GPU memory-model / planner slot
counts are byte-identical between old and current main (the 9 rebase commits are
CPU-MPI + JAX-0.9 strictness only), so prior CrI3 planning numbers transfer 1:1.

## 2026-06-15 — QE/BGW module names in execute_workflow skill are stale (unversioned)

`skills/execute_workflow/SKILL.md:44` says `module load espresso berkeleygw`, but on
the current Perlmutter stack there is **no bare `espresso` or `berkeleygw` module** —
`module load espresso` silently swaps PrgEnv/compiler modules and leaves `pw.x` NOT on
PATH (`which pw.x` → not found). The actual modules are versioned:

- `espresso/7.5-libxc-7.0.0-gpu` (the `(D)` default) or `espresso/7.3.1-libxc-6.2.2-gpu`
  — also `*-cpu` variants.
- `berkeleygw/4.0-nvhpc-23.9` (the `(D)` default) or `berkeleygw/4.0-gcc-12.3`.

Use `module load espresso/7.5-libxc-7.0.0-gpu berkeleygw/4.0-nvhpc-23.9` for the
QE→BGW preprocessing steps (pw.x / pw2bgw.x / wfn2hdf.x). The skill's command should be
updated to a versioned form (or a note that the version must be appended).

Also note: a bash `module load X 2>&1 | tail -1` pipe runs `module` in a subshell, so
the PATH changes are LOST. Load modules without a trailing pipe in the same shell.

## 2026-06-15 — execute_workflow LORRAX preprocessing module names are wrong (`*_chunked`)

`skills/execute_workflow/SKILL.md` (Perlmutter §, steps 5b/5c) invokes
`python3 -m psp.get_dipole_mtxels_chunked` and `python3 -m gw.kin_ion_io_chunked`.
Neither runs: `No module named gw.kin_ion_io_chunked` / `psp.get_dipole_mtxels_chunked`.

In current main (`e85be60`), `src/gw/kin_ion_io_chunked.py` and
`src/psp/get_dipole_mtxels_chunked.py` exist **but have no `__main__`** (they are
library helpers), so `-m` fails. The runnable entrypoints are the non-`_chunked`
modules, which DO have `__main__`:
- `python3 -u -m gw.kin_ion_io -i cohsex.in`
- `python3 -u -m psp.get_dipole_mtxels -i cohsex.in`

This matches the `lorrax_agent` `lxpre` shell function (`modulefiles/lorrax_agent/1.0.lua`),
which correctly uses the non-`_chunked` names. Only the execute_workflow skill is stale.

## 2026-06-15 — centroid prune (pivoted_cholesky) cuFFT-scratch OOM under platform allocator

`python3 -m centroid.kmeans_cli <N> --seed 42` (the documented invocation) OOMs at the
pivoted-Cholesky prune step on an 80 GB A100, for ALL counts (observed at N=1500 and
N=6000):
```
Failed to create cuFFT batched plan with scratch allocator
Failed to allocate request for 16.50GiB ... on device ordinal 0   (batch_count: 984)
```
Root cause: with the container's default `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
(cudaMallocAsync), `jax.memory_stats()` returns `bytes_limit=None`, so
`gpu_utils.get_device_memory_gb()` falls back to nvidia-smi (~71 GB) and the prune's
`build_gram_q0_via_loadwfns` picks `band_chunk_size=64`, whose cuFFT plan needs ~16.5 GB
scratch that cudaMallocAsync can't satisfy alongside the Gram arrays.

Workaround (per the skill's `memory_per_device_gb = 28` cuFFT-OOM hint, applied at the
container level): append BFC + low mem-fraction env so the auto-detected budget drops to
~29 GB and band_chunk shrinks:
```
--env=XLA_PYTHON_CLIENT_ALLOCATOR=default
--env=XLA_PYTHON_CLIENT_PREALLOCATE=true
--env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.4   # bytes_limit≈32GB → budget≈29GB
```
The same BFC env is needed for `gw.kin_ion_io` / `psp.get_dipole_mtxels` to avoid the
same cuFFT-scratch OOM. Existing `centroids_frac_1500.txt` from prior CrI3 80 Ry runs
(same geometry) are reusable to sidestep regeneration at that count.

**Update 2026-06-16 (partial fix + a second, deeper OOM):** Added an opt-in
**`--prune-mem-gb`** flag to `kmeans_cli` (default 0 = legacy auto-detect; threads
`memory_per_device_gb` → `prune_candidates_by_pivoted_cholesky` →
`build_gram_q0_via_loadwfns`). `--prune-mem-gb 20` **fixes the cuFFT-scratch OOM**: the
prune's FFT chunk `cs ∝ budget` (the OOM's `cs=984` came from the 71 GB auto-budget). NB a
`band_chunk_size` override would NOT have worked — `load_centroids_band_chunked` sets
`cs = cs_budget` when `k_chunk_size is None`, ignoring `band_chunk_size`. **But fixing (1)
exposes a SECOND OOM:** at large candidate counts (μ≥6000 → ~9000 candidates) the prune's
Gram build needs **173 GB** (`prune_candidates_by_pivoted_cholesky` line 387) — the
candidate axis is unchunked. **Workaround for large μ: `--oversample 1.0`** (skips the prune
entirely — fine for OOM/memory tests where only the centroid *count* matters). Proper fix:
chunk the candidate axis in the Gram build. Both on `lorrax_A agent/cri3-ppm-maxbands`.

## 2026-06-16 — IBZ-cascade crashes the bispinor zeta fit (full-BZ Z_q vs IBZ L_q)

A bispinor `gw.gw_jax` run on current `main` (`e85be60`) with orbit-closed centroids dies in
the transverse zeta fit:
```
ValueError: B.shape[0]=9 != Nq=5
  ffi/cusolvermp/batched.py:200  batched_distributed_potrs
  common/isdf_fitting.py:1301    solve_zeta
```
The cusolverMp Cholesky back-solve gets a **full-BZ Z_q (9 q-points)** against an
**IBZ-factored L_q (5 q-points)**. The IBZ cascade activates on centroid orbit-closure
(`gw_init.py:959-961`); it was activated 2026-05-11, so bispinor runs from before then (e.g.
`runs/MoS2/D_60Ry_bispinor`, 2026-05-05) were immune and never exercised this path.

**Workaround:** `export LORRAX_FORCE_FULL_BZ=1` (full-BZ = identical physics, just disables the
symmetry optimization). Used for the milestone-A screened-bispinor validation
(`reports/bispinor_screened_a_validation_2026-06-16/`). Genuine LORRAX regression (the full-BZ
Z_q must be sliced to IBZ before potrs, or L_q unfolded to full BZ) — fix pending.

NB — **the plain `export` DID reach the container here**, empirically contradicting the
2026-05-16 "Shifter env passthrough" entry above: run 1 (no env) crashed IBZ-active at Nq=5;
run 2 (`export LORRAX_FORCE_FULL_BZ=1`, identical centroids) ran full-BZ (9 q). The only changed
input was the export, so on `lorrax_C` @ `e85be60` via `lxrun`, shifter forwards host env and no
`LORRAX_SHIFTER --env=` idiom was needed. (Possibly module/shifter-config-specific — the 2026-05-16
note was on `lorrax_B`. If a future full-BZ guard silently no-ops, fall back to the `--env=` idiom.)

## 2026-06-16 — lorrax module stack: don't pipe `module load` / `lxattach`; base modulepath

Two footguns when bringing up the `lorrax_agent` pool overlay (cost several wasted attempts):

1. **Never pipe `module load …` or `lxattach`.** `module load lorrax_C 2>&1 | tail -1` runs the
   module function in a **pipe subshell**, so its `setenv`/`export` never reach the parent shell
   → `LORRAX_ROOT` stays empty and the overlay errors "requires a base lorrax module". Same for
   `lxattach … | head` — its `export SLURM_JOBID` is lost, so `lxrun` then says "SLURM_JOBID not
   set". Run them bare (redirects like `2>/dev/null` are fine; pipes are not).
2. **Base modulefiles live at `/global/u2/j/jackm/modulefiles`** (the overlay is at
   `$SANDBOX/modulefiles`). Full stack in ONE shell (each Bash call is a fresh shell):
   ```
   module use /global/u2/j/jackm/modulefiles; module load lorrax_C 2>/dev/null
   module use $PSCRATCH/lorrax_sandbox/modulefiles; module load lorrax_agent 2>/dev/null
   lxattach >/dev/null 2>&1   # sets SLURM_JOBID; no pipe
   lxrun python3 -u -m gw.gw_jax -i $(pwd)/cohsex.in > gw.out 2>&1
   ```
The `execute_workflow` skill documents only raw `srun --jobid` and doesn't mention the overlay
or these gotchas — consider adding a pointer there.
