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

### 2026-07-08: compare-skill `parse_sigma_hp` silently returns zero blocks on `frequency_dependence 1` (HL-GPP) logs
- **Where**: `skills/compare/SKILL.md` §2a `parse_sigma_hp`, used against `runs/Si/00_si_4x4x4_60band/02*_bgw_hl_*/sigma_hp.log`.
- **What happened**: HL-GPP (`freq_dep=1`) sigma_hp.log has an 11-column band row (`n Emf Eo X SX-X CH Sig KIH Eqp0 Eqp1 Znk` — no primed CH`/Sig`/Eqp0`/Eqp1` columns). The parser's `len(p) >= 15` gate rejects every row, so it returns an empty list with no error. The documented column table in the skill only describes the 15-column (`freq_dep=3` / `exact_static_ch 0`) layout.
- **Expected**: parser handles both layouts (BGW emits primed columns only when the exact-static-CH machinery is active).
- **Workaround**: extended `parse_sigma_hp` to detect the 11-column layout and fall back `CHp=CH`, `Sigp=Sig`, `Corp=Cor`; extension added to `skills/compare/SKILL.md` (parser-only change, per the "add new parsing code to the Skill" rule).

### 2026-06-17: pw2bgw on a DFT+U calculation aborts on the `.hub1` Hubbard-projector write
- **Where**: GW pipeline for VI3/CrI3 with `HUBBARD`/`lda_plus_u=T` in the NSCF (`runs/{VI3,CrI3}/04_gw_6x6_600b_2026-06-17`). The existing CrI3 GW runs were all plain PBE, so this DFT+U pw2bgw path was never exercised.
- **What happened**: `pw2bgw.x` (GPU build, `espresso/7.5-libxc-7.0.0-gpu`) calls `orthoUwfc()` unconditionally when `lda_plus_u=T` (PP/src/pw2bgw.f90:458), which `save_buffer`s the Hubbard wfcs to `./<prefix>.hub1` (PW/src/orthoatwfc.f90:113, `IF nks>1`). The davcio write fails: `Error in routine davcio (13): error writing file "./VI3.hub1"` → MPI_Abort. Reproduced with 1 task, 4 tasks (G-vector parallel: different MPI_Abort), kih_flag on/off, and after freeing 6 TB of quota — so it is **not** disk space and **not** the kih path. Quota was a red herring (15 PB free, 7 TB headroom).
- **Expected**: pw2bgw should export the (already +U-gapped) wavefunctions/eigenvalues to WFN; the orthoUwfc/.hub1 step is only needed for `vhub_flag` Hubbard-potential matrix elements, which the LORRAX kih pipeline does not use.
- **Workaround (works)**: strip the `<dftU>...</dftU>` blocks from the **disposable** `<prefix>.save/data-file-schema.xml` so pw2bgw reads `lda_plus_u=F` and skips orthoUwfc entirely: `sed -i '/<dftU/,/<\/dftU>/d' <prefix>.save/data-file-schema.xml`, then run pw2bgw with ONE task (`-n1`, `MPICH_GPU_SUPPORT_ENABLED=0`). The stored gapped wavefunctions + eigenvalues are untouched → WFN.h5 is correct. Verified: VI3 WFN (18 GB BIN) + WFN.h5 (41 GB) produced. Baked into the run_*_gw.sbatch backups.

### 2026-06-17: `wfn2hdf.x` not on PATH from the `berkeleygw` module (Lmod prepend not applied)
- **Where**: execute_workflow skill (`wfn2hdf.x BIN WFN WFN.h5`). `module load berkeleygw` is ambiguous (two versions: `4.0-gcc-12.3`, `4.0-nvhpc-23.9`); even loading the explicit version via `bash -lc`, `which wfn2hdf.x` stays empty (the modulefile's `prepend_path PATH` didn't take, same class of Lmod-in-non-login-context issue as the espresso entry below).
- **Workaround**: call it by absolute path: `/global/common/software/nersc9/berkeleygw/zen3/gcc-12/mpich/berkeleygw/BerkeleyGW-4.0/bin/wfn2hdf.x BIN WFN WFN.h5`. Runs fine (runtime libs satisfied by the default env); ~91 s for VI3 600-band/36-kpt.

### 2026-06-17: execute_workflow LORRAX module names are stale (`*_chunked` don't exist on main)
- **Where**: `skills/execute_workflow/SKILL.md` LORRAX steps use `gw.kin_ion_io_chunked` and `psp.get_dipole_mtxels_chunked`.
- **What happened**: neither module exists on `lorrax_D@main` (`ModuleNotFoundError`). Caught by a shifter import smoke-test before launching the queued jobs.
- **Workaround**: correct names are `gw.kin_ion_io` and `psp.get_dipole_mtxels` (verified by import + by how the real runs invoke them). Skill should be updated.

### 2026-06-16: `module load espresso` is a no-op in the agent's non-interactive Bash shell — needs `bash -lc`
- **Where**: `skills/execute_workflow/SKILL.md` Perlmutter section ("Load modules once at session start: `module load espresso berkeleygw`"). Works in an interactive `salloc` shell; fails in the Claude Code Bash tool.
- **What happened**: In the agent Bash tool, `module load espresso/7.5-libxc-7.0.0-gpu` only printed a `cray-libsci`/`cray-mpich` "reloaded" line and did **not** apply the modulefile's `depends_on("PrgEnv-nvidia")` swap or its `prepend_path("PATH", .../bin)` — `which pw.x` stayed empty and PrgEnv-gnu stayed loaded, even in a fresh shell with no lorrax modules. `module swap PrgEnv-gnu PrgEnv-nvidia` likewise didn't take. (The NERSC GPU espresso build is `depends_on` PrgEnv-nvidia + cray-fftw + cray-hdf5-parallel; the partial load left pw.x unreachable.)
- **Expected**: `module load espresso/...` should put `pw.x` on PATH and swap PrgEnv, as it does interactively.
- **Workaround**: Wrap the whole QE invocation in a login shell: `bash -lc 'module load espresso/7.5-libxc-7.0.0-gpu; export SLURM_JOBID=<jid>; OMP_NUM_THREADS=16 srun --jobid=$SLURM_JOBID --gres=gpu:4 --overlap -N1 -n4 -c16 pw.x -npools 4 -i scf.in > scf.out 2>&1'`. Verified: under `bash -lc` the PrgEnv-gnu→nvidia swap applies and `pw.x` resolves to `/global/common/software/nersc9/espresso/7.5-libxc-7.0.0-gpu/bin/pw.x`. (Also note the default `espresso` module is now **7.5** with libxc-7.0.0, not 7.4; HUBBARD-card + noncollinear DFT+U syntax is unchanged.)

### 2026-06-16: backgrounded raw `salloc --constraint="gpu&hbm80g" -q interactive` revoked with "Connection timed out" before nodes boot
- **Where**: The recommended raw-salloc recipe in the prod-bispinor task spec / `feedback_lxalloc_gpu_constraint_mixes_hbm` memory: `nohup salloc -N 4 --gpus=16 --constraint="gpu&hbm80g" -q interactive -t 02:00:00 -A m2651 -J lx-alloc-$USER bash -c "sleep 7200" &`.
- **What happened**: Twice in a row (jobs 54615249, 54616243) the scheduler ASSIGNED the hbm80g nodes (`SchedNodeList=nid[008377,008380,008597,008600]`, `Features=gpu&a100&hbm80g`) but the backgrounded salloc died with `salloc: error: Unable to allocate resources: Connection timed out` and the allocation was immediately revoked. The 80 GB nodes are clearly available (resources granted); the failure is the salloc client↔controller connection timing out while the hbm80g nodes boot (slower boot than the default 40 GB nodes), aggravated by `nohup`/background detaching the salloc client.
- **Expected**: salloc holds the allocation once nodes are granted.
- **Workaround**: Add `SALLOC_WAIT_ALL_NODES=1` and increase the client message timeout, or (more robust under `nohup`) submit a real `sbatch` job that holds the nodes + sleeps, then `lxattach` to it. Retrying the raw salloc also sometimes succeeds (transient). Documented here pending human review; falling back to 40 GB `lxalloc` (`memory_per_device_gb=28`) is the spec's sanctioned fallback if 80 GB keeps timing out.

### 2026-06-16: `lxrun` GPU srun missing `--overlap` → step creation fails "Requested nodes are busy" on attached hbm80g allocations
- **Where**: `modulefiles/lorrax_agent/1.0.lua`, the GPU-path `srun` line in the `lxrun` shell function (was lines ~265-273). The `lxshell` override (same file, ~277-300) already uses `--overlap` and documents exactly this footgun, but `lxrun` never got the same flag.
- **What happened**: After `lxattach` to a running `lx-alloc-jackm` allocation (JID 54544991, the raw-salloc `--constraint=gpu&hbm80g` 80 GB type), `lxrun python3 -u -m psp.orbital_magnetization ...` — and even `lxrun hostname` — died immediately with `srun: error: Unable to create step for job 54544991: Requested nodes are busy`, despite `lxstatus` showing 4/4 nodes free and no competing steps on the job (only `.extern`). Without `--overlap`, the step request cannot coexist with the allocation's implicit/extern step, so SLURM reports the node busy (or hangs).
- **Expected**: `lxrun` should create the step like `lxshell` does. Confirmed by direct test on the same node: `srun --jobid=$JID -N1 -n1 --gres=gpu:1 --immediate=10 hostname` → "busy"; adding `--overlap` → succeeds. `--gpus-per-task=1 --overlap` also works.
- **Workaround / fix applied**: Added `--overlap` to `lxrun`'s GPU `srun` line (one line, mirroring the existing `lxshell` precedent and rationale). Safe for all allocation types — `--overlap` only permits step coexistence; the pool `prelaunch` still selects distinct free nodes, so node assignment is unchanged. Flagged for human review in case the CPU-path `srun` (no `--overlap` either) should get the same treatment.

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

## ~~2026-06-16 — IBZ-cascade crashes the bispinor zeta fit (full-BZ Z_q vs IBZ L_q)~~ FIXED

A bispinor `gw.gw_jax` run on `main` (`e85be60`) with a non-orbit-closed centroid set died in
the **charge-channel** zeta fit:
```
ValueError: B.shape[0]=9 != Nq=5
  ffi/cusolvermp/batched.py:200  batched_distributed_potrs
  common/isdf_fitting.py:1301    solve_zeta
```
**Root cause (ordering bug):** `fit_zeta_to_h5` sliced `C_q`/`L_q` to IBZ (`isdf_fitting.py:~2079`)
using the *initial* `write_ibz_only`, but the orbit-closure **auto-fallback** that flips
`write_ibz_only=False` ran *after* `factor_c_q`. On a non-closed set the charge channel fell back
(`write_ibz_only=False` → `Z_q` full-BZ 9) while `L_q` was already IBZ-sliced (5) → the potrs shape
crash. The IBZ unfold math was correct; only the fallback path produced inconsistent shapes.

**FIXED** (`lorrax_C agent/bispinor-ibz-zeta-fallback-fix` `fc9984e`): move the closure check
before the slice so `write_ibz_only` is finalized first; a non-closed set now falls back to full-BZ
cleanly. `LORRAX_FORCE_FULL_BZ=1` is **no longer needed**. Validated bit-identical sym==nosym with
orbit-closed centroids; pytest 21 passed. See `reports/bispinor_ibz_zeta_fallback_fix_2026-06-16/`.
NB the original `centroids_frac_640.txt` was itself not orbit-closed (z-mirror partners absent);
regenerate with `kmeans_cli` (orbit-aware by default for ntran>1) to activate the IBZ speedup.

NB — **the plain `export` DID reach the container here**, empirically contradicting the
2026-05-16 "Shifter env passthrough" entry above: run 1 (no env) crashed IBZ-active at Nq=5;
run 2 (`export LORRAX_FORCE_FULL_BZ=1`, identical centroids) ran full-BZ (9 q). The only changed
input was the export, so on `lorrax_C` @ `e85be60` via `lxrun`, shifter forwards host env and no
`LORRAX_SHIFTER --env=` idiom was needed. (Possibly module/shifter-config-specific — the 2026-05-16
note was on `lorrax_B`. If a future full-BZ guard silently no-ops, fall back to the `--env=` idiom.)

## 2026-06-16 — two multi-GPU JAX jobs on the same node collide (coordination-service abort)

Launching two `lxrun python3 -u -m gw.gw_jax ...` steps **concurrently on the same node** (1-node
allocation) crashed one with EXIT 134:
```
ABORTED: /job:jax_worker/replica:0/task:1 unexpectedly tried to connect with a different
incarnation. It has likely restarted.
... F .../pjrt/distributed/client.h:80] Terminating process because the JAX distributed service
detected fatal errors.
```
The two steps' JAX distributed coordination services collide (same coordinator addr/port on the
shared node). `lxrun`'s pool `prelaunch` picks free nodes, but with only one node both land on it.
**Run multi-GPU JAX jobs sequentially per node, or pin them to distinct nodes** (`LORRAX_NNODES`
/ separate allocations). Cost one wasted run during the bispinor IBZ sym/nosym validation.

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

## 2026-06-16: tests/test_w_bispinor_supermatrix.py fails collection (gw.w_bispinor import)
`pytest tests/test_w_bispinor_supermatrix.py` errors at collection with
`ModuleNotFoundError: No module named 'gw.w_bispinor'`, even though `src/gw/w_bispinor.py`
exists (other `gw.*` imports in the same test dir resolve). Pre-existing on
`agent/bispinor-ibz-lorentz-unfold` (reproduces with local edits stashed) — a test-env
sys.path / packaging quirk, not a code regression. Other bispinor tests
(test_v_q_transverse_unfold, test_v_q_bispinor_helpers, test_compute_V_q_bispinor_g_flat,
test_sigma_x_bispinor) collect + pass fine. Workaround: run those explicitly, exclude
test_w_bispinor_supermatrix until the import path is fixed.

## 2026-06-17: bare `pytest` from lorrax_C/A/D silently tests lorrax_B (editable .pth) — ROOT CAUSE of the gw.w_bispinor mystery above

The venv `.venv/lib/python3.12/site-packages/__editable__.lorrax-0.1.0.pth` pins the `lorrax`
package (`common`, `gw`, `file_io`, …) to **`/global/u2/j/jackm/software/lorrax_B/src`**. So a bare
`pytest` (or `python3 -c "import common"`) from ANY other checkout resolves modules from **lorrax_B**,
not the session's checkout — `module load lorrax_C` does NOT set `PYTHONPATH` to lorrax_C/src
(`PYTHONPATH` is just `/opt/nersc/pymon`). Two consequences:
1. Edits to `lorrax_C/src/common/*`, `gw/*`, `file_io/*` are **invisible to bare pytest** — it tests
   the unfixed lorrax_B copy. (Verified live: a regression test crashed against `lorrax_B/.../symmetry_maps.py:562`,
   then passed once pinned to lorrax_C.) Production `gw.gw_jax` is unaffected — its launcher sets the
   right src path (the prod screened-IBZ crash traceback was `sources/lorrax_C/.../symmetry_maps.py:562`).
2. lorrax_C-only modules absent from lorrax_B (e.g. `gw.w_bispinor`) fail collection with
   `ModuleNotFoundError` — this is the cause of the "test_w_bispinor_supermatrix.py fails collection"
   entry above, NOT a packaging quirk.

**Fix/workaround:** prepend the checkout's src before running pytest:
`export PYTHONPATH=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_C/src:$PYTHONPATH`
(PYTHONPATH wins over the site-packages .pth). Verify with
`python3 -c "import common; print(common.__file__)"` → must show the lorrax_C path. The checkpoint
skill's `uv run python -m pytest` does not pin this; checkpoints that reported "pytest passed" from a
non-B checkout actually exercised lorrax_B.

### 2026-07-01: Base `lorrax_{A..D}` modulefiles' FFI default dirs point to purged $SCRATCH paths
- **Where**: `/global/homes/j/jackm/modulefiles/lorrax_D/0.1.0.lua` (and presumably A/B/C). Defaults `LORRAX_FFI_NVHPC_DIR=/pscratch/sd/j/jackm/lorrax_nvhpc`, `LORRAX_FFI_PHDF5_DIR=/pscratch/sd/j/jackm/lorrax_phdf5_cray/stage`, `LORRAX_FFI_SLATE_DIR=/pscratch/sd/j/jackm/lorrax_slate_cray/stage`.
- **What happened**: those scratch paths no longer exist (purged; deps relocated to `$HOME/software` per home_migration_2026-06-24). `module load lorrax_D` bakes the dead paths into `LORRAX_SHIFTER` `--volume=` mounts, so `lxrun` would fail to mount /lorrax_nvhpc, /lorrax_phdf5, /lorrax_slate.
- **Expected**: defaults should match the relocated `/global/homes/j/jackm/software/lorrax_*` dirs (execute_workflow SKILL.md already uses the home paths).
- **Symptom if hit**: `lxrun` aborts *immediately* (exit 1), NOT a hang: `shifter_realpath: failed to lstat /var/udiMount/pscratch/.../lorrax_nvhpc` → `FAILED to find real path for volume "from"` → `FAILED to setup image`. (Confirmed 2026-07-02.)
- **FIXED 2026-07-02** for `lorrax_D`: `0.1.0.lua` defaults now point to `/global/homes/j/jackm/software/lorrax_{nvhpc,phdf5_cray/stage,slate_cray/stage}` (verified: `module load lorrax_D` with no overrides mounts correctly and exposes cuSOLVERMp 0.7.2). **A/B/C still carry the dead $SCRATCH defaults** — apply the same three-line edit (lines ~79-81) or keep exporting the `LORRAX_FFI_*_DIR` overrides before `module load`.

### 2026-07-02: cusolvermp_cholesky FFI deadlock on 2×2 mesh — NOT REPRODUCIBLE in the fixed env (resolved)
- **Original report**: MoS2 3×3 COHSEX overlay (640 centroids) on 4 GPUs hung **18+ min at 0% GPU util** at `Computing L_q = chol(C_q) [path=cusolvermp_cholesky]` (`common/isdf_fitting.py:1104` → `ffi.cusolvermp.batched_distributed_cholesky`, distributed potrf). Hypothesised stale/mismatched `.so` after the 2026-06-24 `$HOME/software` FFI relocation.
- **Re-investigation 2026-07-02 (env: `LORRAX_FFI_*_DIR=$HOME/software/...` before `module load lorrax_D`, salloc 4×A100, 2×2 mesh) — the deadlock did NOT reproduce anywhere:**
  1. **Isolated FFI** `common.cusolvermp_batched_test --mesh 2x2 --dtype c128`: potrf+potrs collectives complete in ms. Banner: `[lorrax cusolverMp] library 0.7.2, NCCL 2.26.3, comm path: NCCL, grid: 2x2 (row-major)`. (Test then errors on a *harness* bug — `RuntimeError: Array has been deleted` because it `process_allgather`s `A` after `batched_distributed_cholesky` donated it; unrelated to the FFI.)
  2. **Full pipeline**: made variant `runs/MoS2/00_mos2_3x3_cohsex/03_lorrax_cusolvermp_repro` (copy of `02_..._noavg` with `cusolvermp_charge = on`), ran on 4 GPUs (2×2). Completed **EXIT 0** through cusolvermp_cholesky → chi0/W → sigma (~5.5 min wall, all in W/sigma compile; cholesky itself ~0 s). QP energies match the native `off`/1-GPU run's `eqp0.dat` to **max|Δ|=1.3e-5** (reduction-order noise) — distributed potrf is *correct*, not just non-hanging.
- **`ldd` (in-container) is clean**, no missing symbols. Note a benign **version mix**: `.so` RUNPATH is baked to `/lorrax_nvhpc/25.5_cuda12.9/.../lib64` (ships cuSOLVERMp **0.6.0** + CAL 0.4.4 + cublasMp 0.4.0 + nvshmem); `libcusolverMp.so.0` resolves to **0.7.2** only because the `LORRAX_SHIFTER` `LD_LIBRARY_PATH` lists the `0.7.2_cuda12.9` dir first (the 0.7.2 tree ships *only* cusolverMp, so cal/cublasmp/nvshmem fall through to 25.5 — harmless because 0.7.x uses the NCCL-comm grid path, not CAL).
- **Latent fragility (mitigated, not a live bug)**: if the 0.7.2 dir ever drops off `LD_LIBRARY_PATH`, the loader falls through RUNPATH to cuSOLVERMp **0.6.0**, which uses the `comm path: CAL` shim and prints the 2D-grid correctness WARNING. Forcing 0.6.0 (prepend the 25.5 lib dir) was tested: it loads the CAL path but potrf still **completed** (did not deadlock) — so even 0.6.0 is not the hang. **Diagnostic**: the rank-0 banner must read `library 0.7.2 … comm path: NCCL`. If it says `0.6.0 … CAL`, the wrong lib won — fix `LD_LIBRARY_PATH`/`LORRAX_FFI_NVHPC_DIR`, don't disable cusolvermp.
- **Most likely original cause**: a transient mid-migration FFI state (partially-populated `$HOME/software` dirs, or a dead-`$SCRATCH`-mount attempt — see purged-paths KSE above; note that failure now *errors immediately*, it doesn't hang), possibly compounded by NCCL falling back to **TCP Socket** transport (no `libibverbs` in the container → `Failed to open libibverbs.so` → `Using network Socket`) while cuSOLVERMp bootstraps a *second, independent* NCCL communicator alongside XLA's — a known class of occasionally-flaky Socket rendezvous. Not reproducible on demand.
- **Recommendation**: `cusolvermp_charge = auto`/`on` is safe again on 2×2; the `off` workaround is no longer required. Env recipe (verified end-to-end): export `LORRAX_FFI_{NVHPC,SLATE,PHDF5}_DIR=$HOME/software/lorrax_*` **before** `module load lorrax_D` (or use the now-fixed lorrax_D default), confirm the `library 0.7.2 … NCCL` banner. If a hang ever recurs, capture a `py-spy dump`/SIGQUIT stack of the hung rank and check whether it is in `ncclCommInitRank` (bootstrap) vs `cusolverMpPotrf` (collective), and whether the banner printed.

### 2026-07-02: compare SKILL `parse_sigma_freq_debug` column layout is STALE
- **Where**: `skills/compare/SKILL.md` §2c parser and column doc for LORRAX `sigma_freq_debug.dat`.
- **What the skill says**: 13 columns `k n Edft-Ef E_dft kin_ion sex_0 coh_0 x_bare sig_c(0) sig_c+(w) sig_c-(w) sig_c_invld(0) sig_c(Edft) [sig_c_head]`, with the key comparison quantity `sig_c(Edft)` at **col 12**.
- **Actual format written by LORRAX 0.1.0 (verified 2026-07-02, fresh MoS2 3×3 GN-PPM run)**: 14 columns, tab-separated, different meaning:
  `0:k  1:n  2:E_dft  3:Edft-Ef  4:kin_ion  5:V_H  6:x_bare  7:x_head  8:sig_c(Edft).Re  9:sig_c(Edft).Im  10:sig_c_head(Edft).Re  11:sig_c_head(Edft).Im  12:eqp0  13:eqp1`
  Header line in-file: `# k n E_dft Edft-Ef kin_ion V_H x_bare x_head sig_c(Edft).Re sig_c(Edft).Im sig_c_head(Edft).Re sig_c_head(Edft).Im eqp0 eqp1`.
- **Consequence of trusting the stale parser**: it reads **col 12 = eqp0** as if it were `sig_c(Edft)`, giving a nonsense BGW-vs-LORRAX Σc comparison (MAE ~8 eV, wrong sign/shape). The correct comparison quantity is now **col 8 = `sig_c(Edft).Re`** (real part) vs BGW `Corp` (SX-X+CH'). The head is a SEPARATE column (col 10), NOT folded into col 8, so decide explicitly whether to add it (for the fresh MoS2 run, no-head col-8 tracked BGW better: MAE 1.75 eV vs 3.19 eV with head).
- **Fix**: update §2c parser to the 14-col layout above (`sigc_re=p[8]`, `sigc_im=p[9]`, `sigc_head_re=p[10]`, `eqp0=p[12]`), keyed on `len(p)>=14`. Working v2 parser used for the comparison lives at `reports/gnppm_gate_2026-07-02/` notes. Not yet edited into the skill (left for a source-owner pass since the skill is a shared doc).

### 2026-07-03: `tools/profile_gw_xprof.py` has a stale `gw_isdf` import (broken)
- **Where**: `sources/lorrax_D/tools/profile_gw_xprof.py:66` (also present in the other checkouts' copies).
- **What**: the tool does `from gw_isdf import gw_jax`, but the package was renamed to `gw` (the CLI is `python -m gw.gw_jax`). Every invocation dies with `ModuleNotFoundError: No module named 'gw_isdf'`, so the XProf memory-viewer capture path (`profile_gw_xprof.py` → `analyze_xprof_memory.py`) is unusable as-is. `analyze_xprof_memory.py` additionally imports `xprof.convert.raw_to_tool_data`, which may not be installed in the container.
- **Fix**: line 66 → `from gw import gw_jax` (**DONE 2026-07-03** in `lorrax_D`; the capture now runs and writes a valid `*.xplane.pb`).
- **Second gap (open)**: `analyze_xprof_memory.py:17` imports `from xprof.convert.raw_to_tool_data import xspace_to_tool_data`, but **`xprof` is not installed** in the `nvcr.io/nvidia/jax:25.04` container → `ModuleNotFoundError: No module named 'xprof'`. The per-module memory-viewer summary is therefore unavailable in this env. Use the faithful whole-run peak via `LORRAX_MEM_DEBUG=1` under the BFC allocator (`XLA_PYTHON_CLIENT_ALLOCATOR=default`, unset `TF_GPU_ALLOCATOR`), which reports `peak_bytes_in_use` — that is what the memory-model validation in `reports/memory_model_refit_2026-07-03/` used.

### 2026-07-06: streamed Σ_PPM path crashes on `fillvalue=0.0` (h5py>=3.13) — blocked G1
- **Where**: `sources/lorrax_D/src/gw/ppm_sigma.py:1558`, the `kij_stream` accumulator's `create_dataset("sigma_c_kij_ry", dtype=complex128, ..., fillvalue=0.0)`.
- **What**: the container's **h5py 3.15.1** rejects a *float* `fillvalue` on a **complex128** dataset with `ValueError: Unable to synchronously create dataset (no appropriate function for conversion path)`. Any run with `sigma_omega_accumulation = kij_stream` (single-process + a `sigma_kij_h5_file`) dies at dataset creation before writing any Σ_c. Minimal repro: `h5py.File(...).create_dataset('x', shape=(2,2), dtype=np.complex128, chunks=(2,2), fillvalue=0.0)` fails; `fillvalue=0j` (or `complex(0.0)`) succeeds. Older h5py silently coerced the float — which is why the consensus_draft authors never hit it and assumed the stream path merely drops the head.
- **Why it matters**: this precursor crash **masks Bug B** (streamed head-drop). With the float fill, the stream path never runs to the point where the head-less Σ_c is observable, so a strict-xfail G1 keyed on Bug B could never flip to XPASS — the crash would permanently hold the xfail. Fixed here as a one-char env-compat change `fillvalue=0.0 → 0j` (zero numerical effect) so G1 can actually detect Bug B; **WS1 (the Bug-B fix) must keep this** or re-do it.
- **Second Bug-B sibling (not fixed — WS1 scope)**: with the stream path running, `sigma_freq_debug_output = true` feeds the `None` `sigma_c_omega` into the eqp z-factor writer (`gw_jax.py:860` → `eqp_bgw.compute_z_factor_from_omega_grid`) and crashes with `sigma_c_omega shape () mismatched against (…)`. G1 sidesteps it by disabling the debug writer in both run variants; WS1 should make the streamed downstream (`gw_jax.py:628` no-QP-solve fallthrough + the freq_debug writer) handle `sigma_c_omega is None` cleanly.
