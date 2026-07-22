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

### 2026-07-21: `lorrax_C` module requests a nonexistent Shifter source mount
- **Where**: privileged `lxrun` of the full test suite on GPU allocation 56296339 after loading `lorrax_C` and `lorrax_agent`.
- **What happened**: Shifter aborted before pytest with `shifter_realpath: failed to lstat /var/udiMount/pscratch/sd/j/jackm/lorrax_nvhpc` and `FAILED to find real path for volume "from": /pscratch/sd/j/jackm/lorrax_nvhpc`.
- **Expected**: the module's `LORRAX_SHIFTER` mount should point to an existing source/environment path, allowing the active `sources/lorrax_D` checkout to run.
- **Workaround**: inspect the installed LORRAX modules and use the module whose mount configuration targets the active `lorrax_D` environment.

### 2026-07-21: sandboxed `lxrun` cannot query an active allocation
- **Where**: full LORRAX checkpoint test launch on active GPU allocation 56296339.
- **What happened**: the launcher printed repeated `environment: line ...: ERROR:: command not found` messages and `lx_pool: timeout running squeue -j 56296339 --noheader -o '%i|%j|%N|%L'`; pytest never started.
- **Expected**: `lxrun python3 -m pytest -q tests` should select the free node in the active allocation and launch the test process.
- **Workaround**: rerun the identical launcher outside the filesystem sandbox so its Slurm queries and inherited module environment work normally.

### 2026-07-21: sandboxed `lxalloc` cannot read Slurm `udiRoot.conf`
- **Where**: checkpoint GPU allocation for the minimax-solver optimization (`module load lorrax_C ... && lxalloc 1 00:30:00`).
- **What happened**: `lxalloc` reached `salloc`, which printed `udiRoot.conf must be owned by user root!` and `salloc: error: FAILED to read udiRoot configuration file!`, then exited with a segmentation fault (status 139).
- **Expected**: `lxalloc` should start a one-node interactive GPU allocation for the required full LORRAX test suite.
- **Workaround**: rerun the allocation outside the filesystem sandbox so Slurm can inspect its root-owned configuration.

### 2026-07-21: `skills/build_inputs/SKILL.md` points at three directories that do not exist
- **Where**: `skills/build_inputs/SKILL.md` — Step 0 ("Read the pseudopotential `.upf` files … from `assets/pseudos_standard` (or `pseudos_stringent`…)"), Step 1 ("Start from the `scf.in` template in `assets/templates/`"), and Step 1 again ("Pseudopotential files live in `assets/pseudos_standard/`").
- **What happened**: none of `assets/templates/`, `assets/pseudos_standard/`, `assets/pseudos_stringent/` exist. Following the skill literally to build `runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21` fails at the copy step.
- **Expected**: the paths the top-level `AGENTS.md` documents and that actually exist: templates in `templates/` (repo root, not under `assets/`), pseudos in `assets/pseudopotentials/standard/` and `assets/pseudopotentials/stringent/`.
- **Workaround**: used `templates/scf.in` + `templates/nscf.in` + `templates/pw2bgw.in` and `assets/pseudopotentials/standard/{Mo,S}.upf` (md5-verified byte-identical to the pseudos `runs/MoS2/A_bse_figures_2026-07-20` used). No physics impact; the skill's directory names are just stale.

### 2026-07-21: `templates/` has two conflicting MoS2 pseudopotential sets
- **Where**: `templates/Mo_ONCV_PBE_FR-1.0.upf` / `templates/S_ONCV_PBE_FR-1.1.upf` vs `assets/pseudopotentials/standard/{Mo,S}.upf`; `templates/nscf.in` names the former in `ATOMIC_SPECIES` while `templates/scf.in` names the latter (`Mo.upf`/`S.upf`).
- **What happened**: the two sets are **different files** (md5 `703a6da1…`/`fca78818…` vs `4e1c3579…`/`a7319d53…`). A run built by copying `templates/scf.in` and `templates/nscf.in` verbatim would use different pseudopotentials in SCF and NSCF — silently, since QE reads whatever filename is in `ATOMIC_SPECIES` from `pseudo_dir`.
- **Expected**: `templates/scf.in` and `templates/nscf.in` should name the same pseudopotential files.
- **Workaround**: set both to `Mo.upf` / `S.upf` from `assets/pseudopotentials/standard/`, matching every existing MoS2 run in `runs/`.

### 2026-07-09: `module load lorrax_D lorrax_agent` fails in non-login shells (agent Bash contexts)
- **Where**: `CLAUDE.md` compute instructions (`cd $LORRAX_SANDBOX && module use modulefiles && module load lorrax_D lorrax_agent && lxattach`).
- **What happened**: in a non-login agent shell, `modulefiles/` (sandbox) contains only `lorrax_agent`; the `lorrax_D` base module lives in `~/modulefiles`, which is only on MODULEPATH via the login profile. Additionally, sourcing Lmod init manually and evaluating the module output in a plain non-interactive shell glob-mangles the `lxrun` shell function body (`$((nodes * 4))` — the `*` expands against CWD contents), producing a corrupted function.
- **Expected**: the documented two-liner should work from any shell.
- **Workaround**: run everything through `bash -l` (login shell) and `module use ~/modulefiles` explicitly before `module use modulefiles`. Also note: an `lxalloc` held by a harness *background task* dies when the harness reaps the task, killing the allocation mid-run — start detached allocations with `setsid nohup ... lxalloc ... &` instead.


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

- **2026-07-11 update**: container version VERIFIED from the image filesystem:
  `nvcr.io/nvidia/jax:25.04-py3` ships **jax 0.5.3.dev20250415 + jax_cuda12
  plugins** (not 0.7.2, not 0.8 — both stale claims; SKILL.md:136 corrected).
  The pcast landmine below is FIXED on lorrax_C `agent/ffi-host-platform`
  (`1421db1`, version-guarded shim) — sharded_cholesky/1-D meshes are safe
  again on that branch.  NEW second landmine found the same day: the venv's
  jax 0.9.1 rejects `process_allgather(..., tiled=False)` at
  `minimax_screening.py:44` (multi-rank only; container jax accepts it) —
  see the 2026-07-11 §3.5 entry.  Bumping LORRAX_IMAGE to a jax-0.9/cuda13
  NGC build remains the open environment task; note both FFI .so's
  (liblorrax_ffi.so, liblorrax_ffi_host.so) must then be rebuilt against the
  new container's XLA FFI headers (build.sh / build_host.sh handle this).

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
- **Fix**: update §2c parser to the 14-col layout above (`sigc_re=p[8]`, `sigc_im=p[9]`, `sigc_head_re=p[10]`, `eqp0=p[12]`), keyed on `len(p)>=14`. Working v2 parser used for the comparison lives at `reports/gnppm_gate_2026-07-02/` notes. **RESOLVED 2026-07-08**: §2c now carries a header-driven `parse_sigma_freq_debug_v2` (column names read from the in-file `# k n ...` header, robust to the mode-dependent column set — e.g. Si 3D runs have no `x_head`/`sig_c_head` columns) as the current parser; the old fixed-layout parser is retained, marked legacy, for archival pre-refactor files.

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

### 2026-07-08: `docs/docs_gwjax/COHSEX_INPUT.md` `ppm_invalid_mode` section is STALE (pre-refactor)
- **Where**: `docs/docs_gwjax/COHSEX_INPUT.md` §6 (`ppm_invalid_mode`, `ppm_fallback_omega`) and §output-spec (`sigma_c_invalid_static_kij_ev`).
- **What the doc says**: default `"static_limit"`, accepted value `"fixed_2ry"`, and an invalid-mode static-correction output dataset.
- **Actual (lorrax_D, wiring of 2026-07-04; `gw/gw_config.py:314,569`, `gw/ppm_sigma.py:552-565`)**: default is **`"zero"`**; accepted values are **`zero` / `skip` / `2ry`** (`2ry`, not `fixed_2ry` — `fixed_2ry` raises `ValueError`). `static_limit`/`infinity` raise **NotImplementedError** (BGW mode 3 pending); `imaginary` unsupported. The old static-correction path (and the `sig_c_invld(0)` freq-debug column / `sigma_c_invalid_static_kij_ev` dataset) was removed in the refactor — pre-refactor logs (e.g. `runs/Si/00_si_4x4x4_60band/01_lorrax_gn_ppm/gw_gnppm.out`, 2026-04) show `policy=static_limit` prints that no longer exist.
- **Consequence**: an agent writing `ppm_invalid_mode = fixed_2ry` (or relying on the documented `static_limit` default) gets a crash or a NotImplementedError. Use `zero` or `2ry`.
- **Fix**: doc update deferred to a source-owner pass (source checkout is shared/live today). See `reports/gw_refactor_map_2026-07-01/BGW_INVALID_POLE_RESEARCH.md` for the correct mapping.

### 2026-07-08: concurrent `lxrun` steps in a shared allocation collide on the JAX coordinator port
- **Where**: `sources/lorrax_*/src/runtime/__init__.py` (`init_jax_distributed` → no-args `jax.distributed.initialize()` auto-detect) + the `lorrax_agent` `lxrun` wrapper.
- **What**: when two agents run `lxrun` steps concurrently inside the SAME `lx-alloc-jackm` job and their 1-node steps land on (or resolve the coordinator to) the same node, JAX's SLURM auto-detect derives the same coordinator port for both steps. The second step's workers connect to the first step's coordination service and BOTH runs die mid-flight with `ABORTED: /job:jax_worker/replica:0/task:N unexpectedly tried to connect with a different incarnation. It has likely restarted.` Observed 3× on JID 55674298 (2026-07-08, agents sharing the pool).
- **Workaround used**: temporary patch honoring `LORRAX_COORD_PORT` (explicit `jax.distributed.initialize(coordinator_address=f"$SLURMD_NODENAME:$LORRAX_COORD_PORT", ...)`) with a per-invocation unique port — worked first try. Patch was reverted with the session's instrumentation; a permanent fix should give `lxrun` a unique per-step port (e.g. derived from `$RANDOM`/step id) and thread it through `init_jax_distributed`.

### 2026-07-10: login-node shifter volume bind-mount transiently fails (FFI container builds)
- **Where**: `src/ffi/common/cpp/run_shifter.sh` in login-node mode (no `SLURM_JOBID`), documented in `src/ffi/slate/README.md` as "login node — doesn't need an allocation".
- **What**: `shifter --volume=$HOME/software/lorrax_nvhpc:/lorrax_nvhpc ...` failed with `BIND MOUNT FAILED from /var/udiMount//global/homes/... to /var/udiMount/lorrax_nvhpc` / `FAILED to setup image` on login node (2026-07-10 ~02:15). Not a config error — the identical invocation succeeds when routed through a compute node (`SLURM_JOBID` set, srun mode). Likely transient udiRoot state on that login node.
- **Workaround**: run container builds through an allocation: `SLURM_JOBID=<jid> SLURM_OVERLAP=1 LORRAX_NGPU=1 LORRAX_NTASKS=1 bash src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh`.

### 2026-07-11: lorrax_C in-tree sharded cholesky broken under container JAX 0.7.2 (`lax.pcast` missing)
- **FIXED on agent/ffi-host-platform (`1421db1`)**: version-guarded identity shim (jax <= 0.8 has no VMA tracking, identity is exact). Entry kept for other checkouts still on plain main.
- **Where**: `sources/lorrax_C/src/common/cholesky_2d.py:186` (`_chol_2d_local` → `lax.pcast`), hit via `isdf/core.py:factor_c_q` whenever `distributed_cholesky = off` selects `path=sharded_cholesky`.
- **What**: commit `c7e6695` (2026-06-13, "fix(jax-0.9): VMA pcast + tiled=True for multi-process CPU compat") uses `jax.lax.pcast`, which only exists in JAX ≥0.9. The GPU Shifter container pinned by the lorrax_C/lxrun stack (nvcr.io/nvidia/jax:25.04-py3, JAX 0.7.2 per `skills/execute_workflow/SKILL.md`) has no `lax.pcast`, so every in-tree-cholesky GW run dies in fit_zeta with `AttributeError: module jax.lax has no attribute pcast` (observed 2026-07-11, MoS2 3x3 bispinor fixture, 4 GPUs, JID 55791797; `runs/MoS2/C_bispinor_backend_timing_2026-07-11/00_lorrax_gpu_intree/gw.out`).
- **Consequence**: the `distributed_cholesky = off` baseline cannot run on the GPU container until either the container moves to JAX ≥0.9 or the source gains a version-guarded fallback (pre-c7e6695 code used a plain `jnp.zeros` panel init). cusolvermp/slate backends are unaffected (they bypass `cholesky_2d.py`).

### 2026-07-11: documented native CPU MPI recipe (§3.5 + lorrax_agent CPU branch) cannot run multi-rank GW e2e
- **Where**: `sources/lorrax_C/docs/ENVIRONMENT_COMPREHENSIVE.md` §3.5 ("CPU multi-process MPI runs (production-quality)") and the `lorrax_agent` overlay's `LORRAX_PARTITION=cpu` `lxrun` branch (native srun, no container — despite some references describing the CPU path as containerized).
- **What broke (verified 2026-07-11, MoS2 3×3 bispinor GN-PPM, 4 ranks 2×2, Milan)**: three independent staleness issues.
  1. **jax 0.9.1 incompatibility (fatal)**: the lorrax_C venv now has jax 0.9.1; `gw/minimax_screening.py:44` (`_scalar_to_host_float`) calls `multihost_utils.process_allgather(..., tiled=False)` on a committed (non-fully-addressable) scalar, which jax 0.9.1 rejects: `ValueError: Gathering global non-fully-addressable arrays only supports tiled=True`. Every multi-rank GN-PPM/minimax run dies in `fit_ppm` AFTER all ζ fits (~4.4 min wasted). Single-rank unaffected (`process_count()==1` branch). The container jax (0.5.3.dev, nvcr 25.04) accepts it — so GPU runs never see this. Source fix needed (tiled=True or pre-replicated fetch); not applied here (no-source-mod session).
  2. **mpi4py is not installed** in the lorrax_C venv (nor isdf_site, nor the container), contradicting §3.5's "Required dependencies (one-time, inside the venv)". `use_ffi_io=true` on CPU therefore always falls back to `H5PY_ALLGATHER` (rank-0 serial writes) with a `[config]` warning; the documented `PHDF5_HOST` route is unreachable.
  3. **Undefined/mismatched variables in §3.5**: the recipe uses `$LORRAX_VENV` (set by NO module; the venv lives at `$LORRAX_ROOT/.venv`) and `PYTHONPATH=$LORRAX_SRC/src` (the base lorrax_X module sets `LORRAX_SRC=<repo>/src`, so this points at `<repo>/src/src`, which does not exist; correct is `PYTHONPATH=$LORRAX_SRC` or `$LORRAX_SRC:$LORRAX_SITE:<deps>`).
- **Working alternative (validated e2e ×5 this session)**: run the shifter container on the CPU node — raw srun, no lxrun:
  `srun --jobid=$JID --mpi=cray_shasta -N1 -n4 -c32 --cpu-bind=cores shifter --image=nvcr.io/nvidia/jax:25.04-py3 --module=mpich --volume=<nvhpc,phdf5,slate stages> --env=PYTHONPATH=<src:site:deps> --env=LD_LIBRARY_PATH=/lorrax_slate/lib:/lorrax_phdf5/lib:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep --env=MPICH_GPU_SUPPORT_ENABLED=0 --env=JAX_PLATFORMS=cpu python3 -u -m gw.gw_jax -i cohsex.in`
  (drop `--module=gpu` and the `libmpi_gtl_cuda` LD_PRELOAD on CPU nodes). `liblorrax_ffi_host.so` resolves fully in-container: CUDA-free SLATE via its RUNPATH (`$HOME/software/slate_builds/cpu`), libsci from `/lorrax_slate`, MPICH from `/lorrax_phdf5`. See `runs/MoS2/C_bispinor_backend_timing_2026-07-11/manifest.yaml` execution notes.

## 2026-07-15: lorrax_agent module eval glob-expands `*` under non-interactive shells

`module load lorrax_D lorrax_agent` from a non-interactive/harness shell (incl.
`#!/bin/bash -l` scripts under salloc) emits the lorrax_agent shell functions
through Lmod's `sh` init eval WITHOUT noglob: every literal `*` in the function
bodies (`$((nodes * 4))` in lxalloc/lxrun) glob-expands against the cwd, the
eval dies with `syntax error near unexpected token 'then'`, and NO functions or
env vars (incl. LORRAX_SHIFTER) are defined. Interactive login shells are fine.
Workaround: don't load modules in scripted contexts — hardcode the shifter
invocation (copy LORRAX_SHIFTER from an interactive shell / this entry's
sibling script reports/bse_refactor_map_2026-07-15/cleanup_verify/) and call
srun directly, or call lx_pool.py by absolute path for pool coordination.
Possible fix: quote-safe emission in modulefiles/lorrax_agent/1.0.lua
(set -f around the eval, or emit functions via a sourced .sh file instead of
inline eval).

## 2026-07-15: fresh git worktrees lack liblorrax_ffi.so — gnppm/bispinor gates error

The compiled FFI library (`src/ffi/common/cpp/build/liblorrax_ffi.so`) is a
.gitignored build artifact, so ANY fresh worktree/clone fails every
gnppm/bispinor-fixture test (9 errors + si_cohsex_3d failure in
test_gw_jax_regression + test_invariance_gates) with "liblorrax_ffi.so not
found" — identically at origin/main and on feature branches (verified both,
reports/bse_refactor_map_2026-07-15/cleanup_verify/). NOT a code regression.
Fix: copy the .so from the main checkout into the same relative path in the
worktree (or rebuild) before running the full suite.

## 2026-07-17: background-task notification displayed phantom results table (agent-harness rendering, not on disk)

During the primer_response_study C3 prototype (this session, JID 56052603), a
run_in_background watcher's completion notification rendered a full 16-row
results table with numbers ("CLEAN-INTERP ... 4.061e-02", "(194s)" timings)
that exist in NO file: the watcher's own output file on disk, the run log, the
npz, and `sacct` step history (no step of that duration) all disagree with the
notification text and agree with each other. Treat background-task notification
BODIES as untrusted rendering; before quoting any number, grep it from the
on-disk log (`trust production log over mock` applies to notifications too).
The authoritative C3 outputs are
runs/MoS2/A_bse_w0_resolvent_2026-07-16/primer_response_study/proto2_out_c3_3x3.log
(SLURM step 56052603.11) + proto2_c3_3x3.npz; all reported numbers were
re-verified from disk.

## 2026-07-17: two MoS2 fixture traps found by the C2 primer-response prototype (proto1_*)

1. **`zeta_q.h5` `mf_header/kpoints/rk` is UNWRAPPED (0, 1/3, 2/3) but the
   stored zeta spheres/coefficients follow the BGW wrap (2/3 == -1/3).**
   Using rk as-is for |q+G| Coulomb weights or for the e^{iq.r} lab
   reconstruction is wrong at every wrap-affected q (5 of 9 on 3x3):
   |q+G|^2 runs to 53 Ry on a 30 Ry sphere, make_Vq-vs-disk fails at
   0.6-0.8, TRS appears broken at 0.6, and — the big one — interpolation
   studies built on the unwrapped lab continuation scramble 5 of 9 fields:
   measured 155x on the physical exchange-block LOO metric (rankcut-1e-4
   ladder: 4.5e-3 wrapped vs 0.70 unwrapped, same q0/solve/truth).
   `interp_study/*.py` (vq_loo, physical_contract, zeta-direct — the #3.5
   ladder) and the C3 proto2 re-base all use the unwrapped rk; their
   RELATIVE tile conclusions survive but the physical-metric ladder rows
   and the "zeta_R flat" table (1.00/0.82/0.65; wrapped: 1.00/0.39/0.16)
   need re-reading. Fix: q_wrap = rk - round(rk) everywhere the sphere/
   lab phase/Coulomb is touched (proto1_prep.py does this + self-check
   `sphere_max|q+G|^2-cutoff == 0`).
2. **`tmp/isdf_tensors_640.h5:psi_full_y` is NOT in the band span of any
   WFN.h5 on disk at k != 0** (span-projection residual 29-40% vs
   05_lorrax_cohsex_native/WFN.h5 == qe/nscf/WFN.h5 == WFN_qp.h5; k=0
   matches to 2.7e-16; enk_full == WFN el to 7e-14 at ALL k). psi_full_y
   is a processed set, not raw eigenvectors of the stored WFN. Any
   cross-validation of restart-vs-WFN wavefunction CONTENT (band
   transports, non-circular fit-RHS rebuilds, "exact" pair-row
   references) silently fails at k != 0. Provenance unresolved —
   restart-writer investigation needed (out of scope read-only session).

## 2026-07-17: half-boundary (q=1/2) extension of the rk wrap trap — the stored sphere center at BZ-boundary q is per-q IRREGULAR (writer FP fuzz)

Follow-up to the 2026-07-17 rk-unwrap entry. On grids with half-integer q
components (MoS2 6x6, Si 4x4x4) the fix `q_wrap = rk - round(rk)` is NOT
sufficient: at components exactly 1/2 the stored zeta sphere may sit at
+1/2 or -1/2 PER Q, decided by float fuzz in the writer's own wrap (rk
values a few ULP above/below 0.5 round differently). Measured on
`interp_study/mos2_6x6/lorrax/tmp/zeta_q.h5`: 2 of the 11 half-boundary
q's (q=9 (1/6,-1/2)->( 1/6,+1/2), q=33 (-1/6,-1/2)->(-1/6,+1/2)) are
centered opposite to the round() guess; the campaign's own fourtails 6x6
log shows the sphere gate FAILING at +14.45 Ry over cutoff for this
reason (the run proceeded; its 6x6 shell tables at boundary q carry an
e^{iG0.r} contamination on those 2 training fields — 3x3 tables and the
overall four-tails verdict unaffected). Robust fix (implemented in
`primer_response_study/offgrid_prep.py::fix_sphere_wrap`): derive the
center per q FROM the sphere itself — the unique candidate wrap with
max|q_c+G|^2 <= cutoff over the stored G's; assert exact fit. After the
fix makeVq-vs-disk = 2.8e-9 at all 36 q (6x6). Any future consumer of
zeta_q.h5 at boundary q must use the sphere-derived center, not round().

## 2026-07-17: CAMPAIGN_REPORT sec 5a item 2 points the Si negative control at an IBZ-only-zeta fixture; and Si 3D disk tiles use the mini-BZ-averaged head

- **Where**: `runs/MoS2/A_bse_w0_resolvent_2026-07-16/primer_response_study/CAMPAIGN_REPORT.md`
  sec 5a item 2 (and the task spec derived from it): fixture named
  `runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16/work_sym/tmp/isdf_tensors_792.h5`.
- **What happened**: work_sym (and work_demo) zeta_q.h5 store IBZ-ONLY
  zeta (8 spheres; `zeta_q_G {8,792,588}`) while the restart is full-BZ
  (64) — no per-q truth or training zeta at 56 of 64 q. The negative
  control cannot be scored on that fixture.
- **Expected**: a full-BZ-zeta Si fixture.
- **Workaround**: `work_old/tmp/` (isdf_tensors_960.h5 + zeta_q.h5,
  n_mu=960, `zeta_q_G {64,960,588}`) is the only full-BZ-zeta Si restart
  on disk; the control ran there. Note its zeta_q.h5 `mf_header/kpoints/rk`
  is the IBZ list (8 rows) — the 64 spheres follow the row-major kgrid
  enumeration (last index fastest), gate-verified (makeVq-vs-disk all 64 q).
- **Bonus trap for 3D fixtures**: Si disk `V_qmunu` is built with the
  mini-BZ MC-averaged head at G=0 for every q != 0
  (`gw/compute_vcoul.py::build_v_head_miniBZ_avg_3d`, injected when
  `mc_average_vcoul_body=True` (default) and sys_dim=3;
  `v_q_g_flat.py:526`). A bare 8pi/|q+G|^2 rebuild fails makeVq-vs-disk
  at med 9.6e-3 / max 5.7e-2; with the (seed-42, nmc=2^18) table the
  gate is machine-level. 2D fixtures are unaffected (no head injection).

## 2026-07-18: fixed global Miller supersets are NOT contained in every stored zeta sphere (6x6 boundary-q; KeyError trap)

- **Where**: any consumer that maps a fixed Miller G-set across all q of
  `zeta_q.h5` spheres (e.g. a Gaussian LR superset built from
  `min_q |q+G|^2 <= K2max`). First hit in
  `primer_response_study/tile_prep.py::sphere_slot` on MoS2 6x6.
- **What happens**: a G kept because SOME q brings `|q+G|^2` under the
  bound can exceed the 30 Ry sphere cutoff at a FAR q (worst on grids with
  q-components at 1/2) — the slot lookup then has no entry (3x3 is
  accidentally immune; 6x6 fails with a KeyError on e.g. G=(1,3,-6)).
- **Correct handling**: those (q,G) channels are ZERO in the stored
  representation (the fit is band-limited to the sphere); zero-fill them
  and bound their weight — for a Gaussian LR window the worst-case weight
  at any q where the channel is missing is `exp(-cutoff/(4 alpha^2))`
  (measured 5.8e-17 at alpha=0.45; 26 channels affected on 6x6).
  Implemented in `tile_prep.py::{sphere_slot,F_channels}` (sentinel -1 +
  zero column + printed weight bound).

## 2026-07-20: `git -C <symlink> worktree add <RELATIVE path>` lands on the wrong filesystem

`sources/lorrax_A` is a symlink to the home FS. `git -C sources/lorrax_A worktree
add sources/worktrees/lorrax_A_x` (relative dest) silently created the worktree at
`/global/u2/.../lorrax_A/sources/worktrees/...` (home FS) instead of pscratch —
import failures downstream. Not a doc error; the trap is symlink + `-C` + relative
dest. FIX: always use an ABSOLUTE destination path for `git worktree add`.

## 2026-07-20 — Crashed GW/JAX runs leave multi-GB GPU zombies → cascading OOMs

When a `gw_jax`/GN-PPM run aborts (assert, OOM, kill) under `TF_GPU_ALLOCATOR=cuda_malloc_async`, the async allocator does **not** release device memory on abort — the process lingers holding ~10 GB, and the next sequential `srun` on the same node OOMs on startup for no logical reason. Seen during the ppm-sigma bisect/regularization sweeps (many back-to-back GN-PPM reruns). Not a doc error. WORKAROUND: between sequential GPU runs, kill leftovers explicitly — `nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9` (or check `nvidia-smi` and reap stale PIDs) before the next launch; or space runs across nodes.

**2026-07-21 symptom refinement (rank-truncation validation):** the leftover-zombie failure does NOT always present as an OOM. Running a **bispinor** GN-PPM immediately after a scalar GN-PPM on the same GPU (no reap) crashed instead in the bispinor V_q companion-tile reader with `INVALID_ARGUMENT: phdf5 read: negative offset/valid_shape at dim 0 offset=-9223372036854775808` (INT64_MIN) — a garbage SlabIO offset, deep in `v_q_bispinor.py:get_tile → slab_io.read_slab`, with a totally misleading stack that looks like an FFI/HDF5 bug. It is the **same zombie**: a `nvidia-smi ... | xargs -r kill -9` reap before the run makes it disappear (verified — clean reap → rc=0, identical config). Takeaway: ALWAYS reap between sequential GPU `gw_jax` runs; a phdf5 negative-offset crash is a zombie tell, not a code bug.

## 2026-07-21: command sandbox cleanup can panic on the synthetic `.agents` mount

- **Where**: a read-only `exec_command` from the sandbox root after reading the
  required session documents.
- **Symptom**: the command produced its expected output, then the wrapper emitted
  `failed to remove synthetic bubblewrap mount target .../.agents: Resource busy`
  from `linux-sandbox/src/linux_run_main.rs`.
- **Impact/workaround**: no data loss or failed read was observed. Treat the wrapper
  panic as infrastructure noise when the requested command output is complete; avoid
  relying on the resulting exit status alone to decide whether the command ran.

## 2026-07-21: W(omega)-chain manifest names a missing local report

- **Where**: `runs/MoS2/A_bse_w_omega_chain_2026-07-16/manifest.yaml`, in the
  artifact list.
- **What is wrong**: the manifest lists `report.md`, but no such file exists in
  that run directory.
- **Actual documentation**: the convergence, timing, and validation write-up is
  in `reports/bse_refactor_map_2026-07-15/PHASE2_LOG.md`, under
  `W(omega) Lanczos-chain model (2026-07-16, agent/bse-phase2, lorrax_A)`.

## 2026-07-21: PPM regularization commit names a missing report

- **Where**: LORRAX commit `d011a36` (`agent/gw-ppm-sigma-reg`) says its report
  is `reports/gw_ppm_sigma_regularization_2026-07-20/`.
- **What is wrong**: that directory exists but contains only `scratch/` logs and
  analysis scripts; there is no `report.md` or equivalent standalone write-up.
- **Available record**: use the detailed commit message/diff and the 2026-07-21
  rank-truncation entry in `CHANGELOG.md` for the HGL crossing-width fix and its
  validation status.

## 2026-07-21: rank-truncation changelog entry names a missing report

- **Where**: the 2026-07-21 `CHANGELOG.md` entry points to
  `reports/gw_rank_truncation_2026-07-20/` as the feature report.
- **What is wrong**: the directory contains only
  `rank_truncation_recovery.png`; no `report.md` or other text report is present.
- **Available record**: use the detailed changelog entry and LORRAX commit
  `23af6b9` for the implementation and validation claims.

## 2026-07-21: `runs/MoS2/A_bse_figures_2026-07-20/manifest.yaml` was not valid YAML

- **Where**: `runs/MoS2/A_bse_figures_2026-07-20/manifest.yaml`, `steps:` block.
- **What was wrong**: two keys were written `key:{ ... }` with no space after the
  colon (`02_lorrax_gw_d3h_16gpu:{`, `01_lorrax_exciton_bands:{`). YAML requires
  a space before a flow mapping, so `yaml.safe_load` failed with
  `ScannerError: mapping values are not allowed here`. Every run directory is
  required by `AGENTS.md` to carry a manifest; a manifest no tool can parse is a
  silent failure — nothing reads these, so it went unnoticed since 2026-07-20.
- **Fixed**: space inserted after both colons; the file now parses and all six
  steps load. Swept the other `runs/*/*/manifest.yaml` for the same typo — none.
- **Suggested guard**: a `yaml.safe_load` check on every `manifest.yaml` would
  belong in the checkpoint routine (`skills/checkpoint/SKILL.md`); today nothing
  validates them.

## 2026-07-21: `zeta_rcond` sweep claim in the rank-truncation record is too strong

- **Where**: the 2026-07-21 `CHANGELOG.md` rank-truncation entry and the
  `charge_zeta_solve` section of `docs/docs_gwjax/COHSEX_INPUT.md`, which stated
  that well-conditioned CCTs (n_μ ≤ pair-density rank) "are unaffected:
  rank_truncate drops ~0 modes and equals cholesky within ~1e-13".
- **What is wrong**: true for MoS₂ 4×4/640c, false in general. Bulk Si 4×4×4 /
  960 centroids — the `si_cohsex_3d` BGW anchor — has CCT spectrum below
  `1e-6·λ_max`, and the new default shifts its `sigTOT` by 1.02 meV
  (`VH` column by 2.49 meV), i.e. above its 0.48 meV agreement with BerkeleyGW.
  Measured sweep is in `reports/gw_conduction_postfix_2026-07-21/si_rcond_sweep.sh`
  and is now quoted in the docs and in the fixture input.
- **Consequence**: the Si fixture pins `zeta_rcond = 1e-10` so the BGW anchor is
  not silently re-frozen onto a LORRAX self-value.

## 2026-07-21: `psp.get_dipole_mtxels` cannot run on a converged reference (single-process + unchunked FFT box)

- **Where**: `src/psp/get_dipole_mtxels.py:main` (`sources/worktrees/lorrax_gw_converged`,
  branch `agent/gw-converged-campaign`; the code is unchanged from `main`), via
  `common/wfn_transforms.py:read_Gvecs_to_devices`.
- **What is wrong**: two independent limits compound.
  1. The tool **never calls `runtime.init_jax_distributed()`** (unlike
     `centroid.kmeans_cli` and `gw.gw_jax`). Launching it under `srun -n 16`
     therefore starts **16 independent single-process copies**, not one 16-GPU
     job — the mesh it builds, `Mesh(np.array(jax.devices()).reshape(1, -1))`,
     only ever sees the 4 (here 1, after `select_gpu.sh`) devices local to that
     process.
  2. `read_Gvecs_to_devices` materialises the **full FFT-box representation**
     `(nk, nb, nspinor, nx, ny, nz)` complex128 — its own docstring flags this
     ("still materialises the FFT-box representation for caller back-compat …
     the g_flat path is ~6-11 % the size").
  On the converged MoS2 reference (144 k × 326 bands × 174 960 grid points ×
  2 spinors × 16 B) that is **262 GB in one allocation**. Measured:
  `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 262826311680 bytes`
  — **byte-identical on 1 GPU and on 16 GPUs**, which is the fingerprint of (1).
  Logs: `reports/gw_converged_12x12_80ry_2026-07-21/logs/dipole.log`,
  `dipole_16gpu.log`.
- **Why it matters**: `dipole.h5` is not optional. `gw.head_correction`
  defaults to `wcoul0_source = s_tensor`, and with neither `eps0mat.h5` nor
  `dipole.h5` present `resolve_head_sample` raises
  `RuntimeError: Failed to resolve q=0 Coulomb head`. So the whole GW is gated
  on a preprocessing tool that does not scale past roughly
  `nk · nband · n_rtot · nspinor · 16 B ≲ 0.9 · VRAM`.
  It has never been hit before because every prior MoS2 run was 30 Ry / 6×6
  (36 k × 200 b × 46 080 → 21 GB, which fit).
- **Workaround used**: a separate `dipole.in` with a reduced band window
  (`nval 26 / ncond 38 / nband 64` → 51.6 GB, 8 m 33 s on one A100-80GB) while
  the GW itself keeps `nband = 326`. Legitimate because `dipole.h5` feeds ONLY
  the q=0 head, so the band window is a head-convergence parameter, not a Σ
  parameter — and the truncation is measured, not assumed (see
  `reports/gw_converged_12x12_80ry_2026-07-21`, `head_convergence` control).
  Note `nband_eff = min(wfn.nbands, max(nelec + ncond, nband))`, so `ncond`
  must be lowered too; lowering `nband` alone does nothing.
- **Suggested fix (not done here — not this session's scope)**: call
  `init_jax_distributed()` at the top of `get_dipole_mtxels.main`, build the
  mesh the way `gw_jax._build_mesh` does, and take the `g_flat` path
  (`WfnLoader.load` directly) instead of `read_Gvecs_to_devices`, transforming
  per k-chunk. The dipole only needs `<mk|v|nk>` per k, so nothing requires the
  whole (nk, nb) box resident at once.

## 2026-07-21: two silent scale limits hit by the first converged (12x12 / 80 Ry) campaign

Both are LORRAX behaviours, not doc errors, but both are *silent* — nothing in
the log says the intended thing did not happen — so they belong here.

### (1) The charge zeta-solve silently drops BOTH conditioning cures above a 4 GiB cap

- **Where**: `src/isdf/core.py`, `_REPLICATED_CHOL_MAX_STACK_BYTES = 4 * 1024**3`
  + `_replicate_charge_ok(nq, n_rmu)` + `_resolve_solver_kind_charge`.
- **What happens**: only the *replicated* route carries the rank-revealing
  truncation (`23af6b9`) and the mesh-invariant replicated factor (`ca78008`):

      if _replicate_charge_ok(nq, n_rmu):
          return ('replicated_rank_truncate' if charge_zeta_solve == 'rank_truncate'
                  else 'replicated_cholesky')
      return 'cusolvermp_cholesky' if is_2d else 'sharded_cholesky'

  and the cap is `nq * n_rmu**2 * 16 <= 4 GiB`.  The converged MoS2 campaign is
  **above** it — nq = 74 (IBZ), n_rmu = 2416 → 6.44 GiB — so `charge_zeta_solve
  = rank_truncate` in `cohsex.in` was **accepted and then ignored**, and the run
  took `cusolvermp_cholesky`.  The code's own comment documents the fallback
  ("above the replication cap it silently uses the distributed Cholesky"), but
  **no warning is emitted**: the only trace is the route name inside the
  per-q `Computing L_q = ...  [PSD, charge channel, path=...]` line.
- **How it was caught**: the stage-3 sanity gate asserted the route name and
  failed; the physics gates all passed, so nothing else would have flagged it.
  Measured consequence at convergence (rank_truncate vs the silent cholesky
  fallback, everything else identical): K direct gap 2.6356 vs 2.6653 eV
  (29.7 meV), indirect 2.5079 vs 2.5208 eV (12.9 meV), and per-band mean
  |Δ⟨E_QP−E_DFT⟩| 11.5 meV (valence) → 81.4 meV (bands 61–80).  Not a
  catastrophe, but not a no-op either.
- **Suggested fix**: emit a WARNING when `charge_zeta_solve = rank_truncate` is
  requested and `_replicate_charge_ok` is False, naming the cap and the actual
  stack size.  A silently-ignored physics knob is the failure mode
  `feedback_parsed_but_unread_not_dead` warns about, one level down.
- **Workaround used**: `reports/gw_converged_12x12_80ry_2026-07-21/gw_probe.py
  --cap-gib 8` rebinds the constant in-process (no source edit); the replicated
  route then runs and the gate passes.

#### 2026-07-21 (addendum): the cap scales with the q AXIS, so `--cap-gib 8` is
#### not enough the moment you ask for full-BZ zeta — and it fails silently again

- **Where**: same three symbols.  The stack is `nq * n_rmu**2 * 16`, and `nq`
  is the number of q **written to disk** — 74 under the IBZ cascade, 144 with
  `LORRAX_FORCE_FULL_BZ=1`.
- **What happens**: regenerating the converged MoS2 ζ on the full BZ (needed
  because `bse.vq_interp` refuses IBZ-only storage) doubles the stack from
  6.9 GiB to **13.36 GiB**, back above the campaign's raised 8 GiB cap.  The
  route silently reverts to `cusolvermp_cholesky` and `charge_zeta_solve =
  rank_truncate` is ignored a second time — with **no warning**, exactly as
  entry (1) predicts.
- **Measured consequence** (`reports/bse_exciton_smooth_2026-07-21`):
  the ζ that comes out is **~4.5× larger in norm** than the production ζ at the
  same q (thin-slice relF 4.6 against the IBZ file), and rebuilding
  `V(q) = conj(ζ̃√v)(ζ̃√v)^T` from it misses the production `V_qmunu` tiles by
  **relF 15.9 – 31.9 at every one of the 144 q** (median 29.7).  The production
  IBZ ζ rebuilds the same tiles to **1.8e-15 (Γ) / 5.2e-9**, so the rebuild
  machinery and every convention in it are correct — only the solve route
  changed.  A near-singular CCT (κ≈1e13) run through plain Cholesky instead of
  the rank-revealing pseudo-inverse is the whole difference.
- **Workaround used**: the same wrapper with a bigger number,
  `gw_probe.py --cap-gib 16`, plus a hard check for
  `path=replicated_rank_truncate` in the log.
- **Suggested fix (unchanged, now with a second data point)**: warn — loudly —
  when `charge_zeta_solve = rank_truncate` is requested and
  `_replicate_charge_ok` is False, printing the cap and the actual stack size.
  Better still, make the cap a `cohsex.in` key so it is set with the physics
  rather than rebound by a probe wrapper.  A knob whose silent disengagement
  costs a factor 4.5 in ζ should not be discoverable only by asserting on a
  log substring.

#### 2026-07-21 (addendum 2): the arbitrary-Q exciton path has NO off-grid gate, and off-grid `eps_c(k+Q)` collapses

- **Where**: `bse/exciton_bands.py` — `gate_htransform_vs_stored` is called at
  the ONE guaranteed on-grid point (Gamma) only, and `--vq-mode ongrid` forces
  every Q on-grid, so nothing in the pipeline has ever tested the interpolated
  conduction leg at an OFF-grid Q.
- **What happens**: on the converged 12x12 / 80 Ry / n_mu = 2412 reference with
  fH window nb = 28, a 39-Q M-Gamma-K path (`--vq-mode interp`) returns **11 of
  its 37 off-grid Q with the WHOLE eigenvalue multiplet collapsed 166-1066 meV
  below the local trend**, while all 3 on-grid Q are exact. The collapses are
  isolated single points, not dispersion: e.g. Q=(0, 0.1316, 0) gives
  E1 = 1.2454 eV between neighbours at 2.3330 and 2.2895 eV.
- **It is NOT the exchange model.** The companion run (11 Q, all on the mesh,
  `--vq-mode interp` vs run 08's exact stored `V_qmunu[wrap(-Q)]` tiles) agrees
  to **max |dE1| = 0.009 meV, 0.026 meV over all 8 branches**. The b26p V_Q
  interpolation is exonerated; the defect is on the interpolated
  `eps_c(k+Q)` / `psi_c(k+Q)` htransform leg.
- **Likely cause, already documented as a knob**: the driver's own `--a-band`
  help says "a large default `a` from a dispersive top guard band can collapse
  off-grid eps_c(k+Q) by eV" — the f-transform width `a = 4*BW` is taken from
  the top band of the fH window, and the selected conduction caches then land in
  the `f' -> 0` compression zone at particular k+Q. Untested because the window
  sweep that chose nb = 28 was itself entirely on-grid.
- **Suggested fix**: (i) add an OFF-grid gate — the driver already knows which Q
  are on the mesh, so a smoothness/2nd-difference check on `eps_c(k+Q)` across
  the Q list costs nothing and would have caught this; (ii) sweep `--a-band`
  and re-measure; (iii) until then, `--vq-mode interp` results at off-grid Q
  must not be reported as physics.
- **Data**: `reports/bse_exciton_smooth_2026-07-21/offgrid_collapse.json`,
  `exciton_bands.npz`, `plots/mos2_exciton_arbitraryQ_diagnostic.png`.

### (2) The htransform / BSE cannot be built at production n_mu — `fH_R` is replicated

- **Where**: `src/bandstructure/bse_setup.py:156`,
  `fH_R_rep = jax.device_put(fH_R, rep)` with `rep = NamedSharding(mesh_xy, P())`.
- **What happens**: `fH_R` is `(nk_co, rank, rank)` complex128 with
  `rank = nspinor * n_mu`, and it is replicated on **every** device by design
  ("so each q-batch is a local matmul + eigh").  Cost `nk * (ns*n_mu)^2 * 16`:
  at 6x6 / n_mu = 1496 that is 5.2 GB (fine), at 12x12 / n_mu = 2412 it is
  **49.93 GiB per device** and OOMs an A100-80GB
  (`RESOURCE_EXHAUSTED ... 53616328704 bytes`).  It is **quadratic in n_mu and
  independent of the device mesh**, so adding GPUs does not help.
- **Consequence**: `bse.exciton_bands` and every `bandstructure.htransform`
  consumer (band figures, the b_max sweep) are unrunnable on the production
  n_mu = 2412 GW.  Workaround: a second GW at n_mu = 1236 on the same reference
  purely as the BSE/figure producer (13.1 GiB replicated).
- **Suggested fix**: shard `fH_R` on the leading `nk_co` axis (the q-batch loop
  already batches over q; each batch needs the full R-sum, so this wants either
  an all-gather per batch or a sharded Fourier sum), or store `fH_k` and do the
  R-sum inside the batch.  Either way it should not be `P()`.

### (3) (fixed in this session) htransform Galerkin band chunk was a fixed 64

`streaming_galerkin_solve` streamed psi over the whole r-axis with a fixed
`band_chunk_size = 64`, and `to_rchunk` returns ONE replicated
`(nk, bc, nspinor, n_rtot)` array — 0.81 GB **per band** on this reference, so
the default asked for 51.6 GB in a single allocation.  Fixed on
`agent/gw-converged-campaign` (`5e50b8e`): the parameter is now a ceiling,
lowered so a chunk stays under 6 GiB.  Memory-only (band chunking is a pure
accumulation split), and it is what let stages 4(d) and 5 run at all.

## 2026-07-21 (addendum): `bse.exciton_bands` has a RESOLUTION ceiling at 80 Ry, set by the replicated `fH_R`

Extends entry (2) above with the quantitative version, after eight
configurations on the converged reference
(`reports/gw_converged_12x12_80ry_2026-07-21` §5 has the full table).

- The driver's physics gate (`max|Δε_c|` at Γ < 50 meV, subspace min-sval > 0.5)
  is the binding constraint, not any of the structural ones. Best achieved:
  **338.2 meV / 0.3283** — refused, correctly.
- The discriminating variable is **centroid density against the real-space
  grid**, not the interpolation window and not the k-grid. The 30 Ry run that
  passes at 9.5 meV has 1496 centroids over 46 080 grid points; 80 Ry has
  174 960 points, so matching needs **n_μ ≈ 5680**.
- `bse_setup.compute_wfns_fi` replicates `fH_R (nk, ns·n_μ, ns·n_μ)` c128, so
  n_μ = 5680 costs **69.2 GiB/device** at 6x6 and **276.9 GiB/device** at 12x12.
  Accuracy and memory are therefore mutually exclusive at 80 Ry on any GPU.
- **Fix**: shard `fH_R`. 16-way, n_μ = 5680 is 4.3 GiB/device and the run fits.
- Other rules discovered along the way, all undocumented, all load-bearing:
  `nspinor·n_μ > nk·nb` (capacity); `n_μ` divisible by the mesh y-extent and
  `n_q` by `px·py` (two separate `ValueError`s); orbit-closed D3h centroids write
  IBZ-only ζ while `vq_interp` demands full-BZ, joined only by the env var
  `LORRAX_FORCE_FULL_BZ=1`; and the interp window must be the BSE window plus a
  few guards (over-packing shows up in the gate's ENERGY metric, not in `ctilde`
  orthogonality, which stays at 1e-14 while the energies are 361 meV wrong).

## 2026-07-21 (agent/bse-exciton-converged): four htransform defects, and a correction to the entry above

Found while sweeping the htransform fH band window on the SAME converged
12x12 / 80 Ry / n_mu = 2412 data.  Deliverables:
`reports/bse_exciton_converged_2026-07-21/`.

### (A) CORRECTION — "a non-zero `b_start` is separately broken" is not a bug

The parent report (`gw_converged_12x12_80ry_2026-07-21` §4d) recorded that a
gap-centred window (bands 18-34) came back with `rank = 0`, `sigma_max = 0`, and
filed it as a separate breakage.  It is not.  `common/wfn_transforms.
load_centroids_band_chunked` ends with

```python
nb_user_in_range = max(0, meta.b_id_4_user - b_start)
if nb_user_in_range < nb_total:      # zero the user-band-pad rows
```

`b_id_4_user` is the input's `nband`, an **ABSOLUTE** band index, while the
window is `[nelec - nval, nelec + ncond)`.  Setting `nband = nval + ncond` (the
natural reading of "the htransform's own band window", and what
`gwbands.in` does when `nval = nelec`) makes `b_id_4_user < b_start` for any
gap-centred window, so EVERY band is zeroed and the SVD of an all-zero matrix
returns rank 0.  The rule is `nband >= nelec + ncond`.
**Suggested fix**: raise in `Meta.from_system` when `nband < nelec + ncond`
instead of silently zeroing psi downstream.

### (B) The retained SVD rank is not mesh-aligned — hard crash whenever psi_mu is rank-deficient

- **Where**: `src/bandstructure/htransform.py`, `streaming_galerkin_solve`,
  `rank = int((s_host > s_host.max() * rtol).sum())`.
- **What happens**: `G`, its Cholesky factor, `ctilde` and `fH` all live on a
  `(rank, rank)` face sharded `P('x','y')`.  When `nk*nb` reaches the numerical
  rank of the centroid-sampled psi the retained rank is an arbitrary integer and
  the first `device_put` onto that face raises
  `ValueError: ... dimension 0 should be divisible by 4, but it is equal to 4570`.
  Reproduced at nb = 32 (rank 4570) on a 4x4 mesh.
- **Fixed on this branch**: round the retained rank DOWN to
  `lcm(mesh.x, mesh.y)`; the dropped directions sit at the `rtol` threshold
  (sigma/sigma_max ~ 1e-8).  A `[warn] psi-at-centroids is RANK-DEFICIENT` line
  now prints the measured capacity bound.

### (C) `htransform.read_eqp_energies` cannot read LORRAX's own `eqp1.dat`, and fails SILENTLY

- **Where**: `src/bandstructure/htransform.py:539` and its only caller,
  `initialize_wfns(..., eqp_file=...)`.
- **What happens**: the parser expects `k-point N:` blocks with `n=… EQP=…`
  text; LORRAX's GW writes the BGW columnar form (`kx ky kz nbands` then
  `spin band EDFT EQP`).  `initialize_wfns` wraps the call in
  `try/except Exception` and logs `EQP override skipped: …`, so a caller that
  asks for QP energies silently gets **DFT** ones.  Two parsers exist for the
  same file family (`bse.bse_io.read_bgw_eqp` reads it correctly).
- **Worked around** in `bse.exciton_bands --eqp` by routing both legs through
  `bse_io.read_bgw_eqp`.  **Suggested fix**: delete
  `htransform.read_eqp_energies`, call `bse_io.read_bgw_eqp`, and let the
  exception propagate.

### (D) `apply_eqp_corrections(input_file=...)` asserts the eqp file is IBZ-sized

`bse/bse_io.py:1210` does `assert nk_ibz == sym.nk_red`.  LORRAX's GW writes
`eqp1.dat` on the FULL BZ (144 blocks here), so passing `input_file` aborts and
only the `input_file=None` energy-matching branch works.  The branch is O(nk^2 nb)
Python but correct; the k-order is in fact the identity (verified: the file's
own E_DFT column reproduces `enk_full` to 0.0000 meV over (144, 80)).

### (E) Entry (2)/(addendum) above is now FIXED, and its "n_mu ~ 5680" conclusion does not bind

`compute_wfns_fi` no longer replicates `fH_R`; it stays `P(None,'x','y')` and the
q-Fourier sum is device-local (the R axis is unsharded), with one all-to-all onto
the q axis before the eigh.  Two further `with_sharding_constraint` calls were
needed — on the `build_fH_R` einsum output and on the `_q_batch` einsum output —
because XLA otherwise materialises the full `(nk, rank, rank)` / `(bs, rank, rank)`
products on every device (measured 57.8 GiB and 9 x 11.4 GiB respectively).
With those three changes the htransform runs at n_mu = 2412 / nk = 144 at every
window in {16 … 40} on 16 x A100-80GB, peak ~17 GiB.

## 2026-07-21 — a co-tenant agent's `pkill -9` kills other agents' job steps

**Symptom.** Steps launched into the shared `lx-alloc-$USER` allocation die with
`srun: error: nid00XXXX: task N: Killed` (rc=137) 40-60 s in, with no Python
traceback, no `RESOURCE_EXHAUSTED`, and no host OOM. Confirmed NOT an OOM:
`/sys/fs/cgroup/system.slice/slurmstepd.scope/job_<JID>/memory.events` reports
`oom_kill 0` and `memory.current` 41 GB against a `memory.max` of 241 GB, with
190-217 GB free on every node.

**Cause.** Run scripts under `runs/MoS2/09_*` and `runs/MoS2/10_*` open with a
node-wide reaper step. Observed live on 2026-07-21 22:50 (agent A's exciton
production launcher, `runs/MoS2/10_mos2_exciton_anchored_2026-07-21/run_exciton.sh`):

```
srun --jobid=$JID --overlap --immediate=90 -N4 -n4 --gres=gpu:4 \
     bash -c 'pkill -9 -u $USER -f "[p]ython3"; exit 0'
```

`pkill -9 -u $USER -f "[p]ython3"` kills **every** python3 process the user owns
on all four nodes — including a co-tenant agent's live multi-rank run. Sibling
scripts use narrower patterns (`[b]se.exciton_bands`, `[e]ps_window_sweep`), but
the bracketing only stops the reaper from matching *itself*; nothing scopes it
to the launching step. Three separate measurement runs were destroyed by this.

**Rule.** Never `pkill -9 -u $USER` in a shared allocation. Scope the kill to
your own step — `scancel --signal=KILL <JID>.<STEPID>` from `squeue -s`, or
match on a per-agent job name (`lxrun` already tags steps `lx-<AGENT>-*`), or
`pkill -9 -f "$PWD"` so the pattern is anchored to your own run directory.

**Workaround while this persists.** `runs/MoS2/11_mos2_htransform_ffi_eigh_2026-07-21/run_gate.sh`
retries (`TRIES=N`) and distinguishes a collision (bare SIGKILL) from a real
device OOM (which comes back through Python as `RESOURCE_EXHAUSTED` and is
recorded in the gate JSON's `failures` block).


## 2026-07-21 — concurrent LORRAX steps in one allocation collide on the JAX coordinator port

**Symptom.** A second run started while another is live dies with
`ABORTED: /job:jax_worker/replica:0/task:N unexpectedly tried to connect with a
different incarnation. It has likely restarted.` (rc=134), or hangs until srun
SIGKILLs every task (rc=137). Nothing is wrong with either program.

**Cause.** `jax.distributed.initialize()` with no arguments uses JAX's SLURM
cluster detection, which derives the coordinator **port from `SLURM_JOB_ID`**.
Every step of one allocation therefore lands on the same host:port, so two
concurrent runs join one coordinator, disagree about the world size, and abort.
In a shared pool (agents A-D on one `salloc`) this fires constantly.

**Fix (landed on `agent/htransform-distributed-eigh`).** `runtime.init_jax_distributed`
now takes the explicit `(coordinator_address, num_processes, process_id)` path
whenever `JAX_COORDINATOR_ADDRESS` is set, and passes `local_device_ids` on that
path too (without it the explicit form assumes each process owns every local GPU
and dies with `CUDA_ERROR_INVALID_DEVICE: invalid device ordinal` under the
one-GPU-per-process binding `select_gpu.sh` sets up). Launchers should pass a
per-launch address, and must pin `--nodelist` so process 0's host is known:

```
--env=JAX_COORDINATOR_ADDRESS=<first node of --nodelist>:<port unique to the launch>
```

See `runs/MoS2/11_mos2_htransform_ffi_eigh_2026-07-21/run_shifter.sh`.


## 2026-07-21 — `liblorrax_ffi.so` must be dlopened AFTER h5py binds its HDF5

**Symptom.** `ValueError: Not a datatype (not a datatype)` from `h5py/h5t.pyx`
when a test or script loads the FFI library and only then imports a LORRAX
module that pulls `h5py` (e.g. `bandstructure.bse_setup`).

**Cause.** `liblorrax_ffi.so` links the Cray parallel HDF5 staged at
`/lorrax_phdf5`; h5py imported afterwards initialises against those symbols.

**Fix.** `tests/test_ffi_linalg_contract.py` now imports `h5py` at module scope
before the FFI availability probes. Any new script that dlopens the FFI must do
the same.


## 2026-07-21 — the FFI shared object is a per-checkout build artifact

`src/ffi/common/cpp/build/liblorrax_ffi.so` is gitignored, so a fresh worktree
has no FFI at all and every `needs_ffi` test SKIPS — silently, which reads as
"passed". Either run `src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh`
in the new worktree, or symlink the `.so` from a sibling checkout after
confirming `diff -rq` on every `src/ffi/*/cpp` tree is empty.
