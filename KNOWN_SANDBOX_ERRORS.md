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

### 2026-05-18: scalar (`nspinor=1`) Si WFN.h5 trips `centroid.kmeans_cli` — `get_spinor_rotations` returns (n,2,2) even when nspinor=1, so `unfold_psi` blows up the spinor axis from 1 to 2
- **Where**: `sources/lorrax_B/src/common/symmetry_maps.py:1220` (`SymMaps.get_spinor_rotations`) and `sources/lorrax_B/src/common/symmetry_maps.py:726` (`unfold_psi`'s `np.einsum("jk,nkl->njl", U_eff, cnk)`), reached from `centroid.kmeans_cli` via `WFNReader._eager_build` (`sources/lorrax_B/src/file_io/wfn_loader.py:1020`).
- **What happened**: Built a fully scalar Si NSCF with `noncolin=.false. lspinorb=.false.` (planning a scalar non-bispinor k-grid sweep for the planner audit). Resulting `WFN.h5` has `nspin=1, nspinor=1, wfns/coeffs shape (102, 1, 1693, 2)` (correct BGW layout). `kmeans_cli WFN.h5` crashed in `_eager_build` with `ValueError: could not broadcast input array from shape (4,2,537) into shape (4,1,537)` — `unfold_psi`'s spinor rotation is hardcoded `(n_sym, 2, 2)` and the einsum turns input `(nb, 1, ngk)` into `(nb, 2, ngk)`. The path is `kmeans_cli → WFNReader → SymMaps.get_spinor_rotations → unfold_psi`.
- **Expected**: For nspinor=1 the spinor rotation should be identity (the 1×1 trivial U) and `unfold_psi`'s einsum should preserve the input shape. Per the docstring of `unfold_psi`: "For ns=1 (non-SOC), U_eff is the 1×1 identity and this einsum is a no-op (callers can still pass it without special-casing)." — but the construction in `get_spinor_rotations` hardcodes `(nsym, 2, 2)` so this no-op contract isn't honored.
- **Workaround**: For the planner audit, fall back to `noncolin=.true. lspinorb=.false.` (nspinor=2 without SOC). The cohsex.in `bispinor=false` flag still produces a non-bispinor pipeline (no 4-channel transverse μ_L); the `ns=2` factor in the planner's per-rank formulas (`_bytes_c128(nk, ns, ns, mu, r_chunk, shard=p_xy)` for the P-pair carry, etc.) is exercised but it's the production setting for any scalar-non-SOC GW workload, so it's still a legitimate "non-bispinor scalar" case. Truly nspinor=1 testing would require a fix to `get_spinor_rotations` to special-case the trivial spinor.
- **UPDATE 2026-05-18 (Agent A `agent/si-nonbispinor-mu-sweep`)**: FIXED in `lorrax_A` commits `8c18925` (eager `unfold_psi`, `symmetry_maps.py`) and `dc0b254` (phdf5 `WfnLoader._ensure_phdf5_static`, `wfn_loader.py`). When `cnk.shape[1] == 1` (eager) or `self.nspinor == 1` (phdf5), the spinor mixing is now an identity no-op (TRS branch's complex conjugation still applies; iσ_y is undefined for ns=1). Both fixes are independent of the underlying 2×2 `get_spinor_rotations` shape, which is left alone. All 44 `tests/test_{unfold_psi_trs,wfn_loader_eager,wfn_loader_phdf5_clamp,wfn_transforms,v_q_transverse_unfold,trs_unfold_centroid_perm}.py` tests still pass. Truly scalar Si μ-sweep at `nspinor=1` completed end-to-end ζ-fit on the Agent A branch.

### 2026-05-18: cusolverMpPotrf status=7 INTERNAL_ERROR under BFC + MEM_FRACTION=0.95 on 2D mesh
- **Where**: `sources/lorrax_A/src/ffi/cusolvermp/cpp/batched_potrf_ffi.cc:136`, called from `factor_c_q` → `batched_distributed_cholesky` in `src/common/isdf_fitting.py:1104`. Active when `_resolve_solver_kind_charge` returns `'cusolvermp_cholesky'` (= true 2D mesh, `px≥2 AND py≥2`) and the run uses `XLA_PYTHON_CLIENT_ALLOCATOR=default + PREALLOCATE=true + MEM_FRACTION=0.95`.
- **What happened**: Si 4×4×4 25 Ry scalar non-bispinor on 4 GPUs / 2×2 mesh under BFC+0.95 dies in the first Cholesky with `cusolverMpPotrf (q=0) failed: status=7`. NCCL banner prints `[lorrax cusolverMp] library 0.7.2, NCCL 2.26.3, comm path: NCCL, grid: 2x2 (row-major)` immediately before the failure. C_q here is only ~1.5 MB so this is NOT a numerical / PSD issue.
- **Expected**: cusolverMpPotrf should succeed; the bispinor sister sweep `agent_t_si_bispinor_sweep.md` ran cleanly at MEM_FRACTION=0.95 but on a `1×4` mesh (so the `sharded_cholesky` path was selected instead).
- **Root cause hypothesis**: NCCL user-buffer-pool registration tries to allocate inside the (already-95%-BFC-preallocated) HBM and gets refused. The non-2D mesh case avoids cusolverMp entirely, which is why bispinor was immune.
- **Workaround**: For OOM-relevant measurements on a true 2D mesh, set `cusolvermp_charge = off` and `cusolvermp_lu = off` in cohsex.in. This selects the in-tree `sharded_cholesky` / `lu` paths (same code used on 1×N meshes). Alternatively `MEM_FRACTION=0.80` leaves ~8 GB headroom for NCCL/cuSOLVERMp and the FFI succeeds. Both are wired into `runs/Si/MU_nonbispinor_2026-05-18/_run_gw.sh` as the `bfc_pre95` (with cusolvermp_off) and `bfc_pre80` variants.
