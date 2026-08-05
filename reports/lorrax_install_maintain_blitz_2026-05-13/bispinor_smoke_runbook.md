# Bispinor smoke timing runbook — install/maintain blitz integration

**Goal**: confirm that the integration branch (`agent/install-blitz-integration`
on lorrax_D) runs a small bispinor calc in the same wall time as current
main, with identical eqp0 values. If yes, the six rebased blitzes (#0,
#1, #3, #4, #5, #6) are safe to land on main. If no, the diff tells us
which one regressed.

## Test article

`runs/MoS2/D_60Ry_bispinor` — bispinor MoS2 60Ry, 672 centroids, 32 bands,
sys_dim=2, **x_only=true** (bare Σ_X only, no screening — runs in
minutes). `memory_per_device_gb=30` so a 40 GB A100 fits cleanly. Has a
pre-existing `eqp0_first_e2e_smoke.dat` and live `eqp0.dat` for
comparison.

## Step 0 — Confirm the WFN.h5 is current

```bash
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/D_60Ry_bispinor
ls -la WFN.h5 wfn.h5 2>&1 | head -4
# If both eqp0 reference files exist, the WFN was correctly staged for
# the bispinor x_only path.
```

If WFN.h5 is missing, regenerate via the QE→pw2bgw chain (skill
`build_inputs` documents this) — the bispinor path needs the FR-pseudo
WFN from `runs/MoS2/D_60Ry_bispinor/qe/` if present.

## Step 1 — Baseline timing on **current main** (0f355b7)

This run gives us the "memory model work as-is" reference. Use any of
the live checkouts (lorrax_A/B/C/D) — pick one that's currently on main
or check out main temporarily.

**On lorrax_D (which we just used for integration), first save the
integration HEAD and reset to main for the baseline:**

```bash
cd /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
git rev-parse agent/install-blitz-integration  # save this hash; should be 3079a1f
git checkout main                              # should be 0f355b7
module load lorrax_D
lxalloc 1 1                                    # 1 node × 1 hour interactive

# Inside the allocation:
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/D_60Ry_bispinor

# Pre-build any stale jax cache (optional but recommended for clean timing):
# rm -rf $SCRATCH/.jax_cache/lorrax_*

time lxrun python3 -u -m gw.gw_jax -i cohsex.in 2>&1 | tee baseline_main.log
cp eqp0.dat eqp0_baseline.dat
# Watch for the post-init RuntimeWarning from blitz #3's assertion -- on
# baseline (no blitz #3 yet) you won't see it; on integration you might.
```

Pull `real` wall time + the per-stage timings from `baseline_main.log`
(grep for `=== `).

## Step 2 — Rebuild the FFI inside Shifter on the integration branch

Three of the six blitzes touch code that flows into `liblorrax_ffi.so`,
so the .so must be rebuilt before re-timing:

- **Blitz #6**: `batched_potrf_ffi.cc` / `batched_potrs_ffi.cc` /
  `context.cc` / `ctx.h` — direct C++ changes.
- **Blitz #5**: `INSTALL_RPATH` uses `${LORRAX_CONTAINER_PHDF5_PATH}/lib`
  / `${LORRAX_CONTAINER_SLATE_PATH}/lib` — the .so's rpath needs to be
  re-baked. With default env vars the resolved values are still
  `/lorrax_phdf5/lib;/lorrax_slate/lib` (same strings as before), so the
  binary will be functionally identical to the old .so for Perlmutter
  use — but the wrapper is in place for a future Apptainer port.
- **Blitz #3**: `select_gpu.sh` + `run_shifter.sh` + new
  `config/mpi_stacks/cray_mpich.sh` are runtime infrastructure (not
  compiled in), but `run_shifter.sh` is the wrapper around `build.sh` —
  the build will read the new sourced stack file.

**Switch to integration and rebuild:**

```bash
cd /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
git checkout agent/install-blitz-integration   # should land at 3079a1f
module unload lorrax_D
module load lorrax_D                            # picks up the new modulefile

# Inside Shifter (rebuild the FFI):
bash src/ffi/common/cpp/run_shifter.sh src/ffi/common/cpp/build.sh --fresh
# --fresh forces a clean CMake configure so the new INSTALL_RPATH
# template is re-evaluated.  Without --fresh, CMake's cache may hold
# the old literal /lorrax_*/lib values from the previous configure.

# Confirm the .so was rebuilt and rpath is sane:
ldd src/ffi/common/cpp/build/liblorrax_ffi*.so | grep "not found" && echo "FAIL: missing libs" || echo "OK: all libs resolve"
readelf -d src/ffi/common/cpp/build/liblorrax_ffi*.so | grep -E "RUNPATH|RPATH"
```

If `pip install -e .` (Blitz #1) is the preferred build path, that
would also work — but it's a larger workflow change and not necessary
for this smoke test. Stick with `run_shifter.sh build.sh --fresh` for
timing parity with baseline.

## Step 3 — Re-run on integration branch

```bash
# Still inside lxalloc, still in the run directory:
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/D_60Ry_bispinor

# Clear the JAX compile cache to avoid stale PTX from the old FFI symbols:
# (the .so changed identity for batched_potrf/potrs, so cache entries are stale)
rm -rf $SCRATCH/.jax_cache/lorrax_*  # or just the entries from this LORRAX checkout

time lxrun python3 -u -m gw.gw_jax -i cohsex.in 2>&1 | tee integration.log
cp eqp0.dat eqp0_integration.dat
```

## Step 4 — Compare

### Correctness — eqp0 should be bit-identical (modulo header timestamps)

```bash
# Strip header lines (first 2 contain timestamps) and diff the numeric rows:
diff <(tail -n +3 eqp0_baseline.dat) <(tail -n +3 eqp0_integration.dat) | head -30
# Expect: empty diff (or numerically negligible: |Δ| < 1e-9 eV per eigenvalue).
```

A non-empty diff is a real regression — most likely candidate is Blitz #6
(the ScratchAllocator migration in batched_potrf/potrs); a smaller
candidate is Blitz #0 (allocator change might surface a latent
mem-ordering bug). Bisect by reverting the suspected commit (`git
revert` on a throwaway branch) and re-running.

### Wall time — should be within ~5% of baseline

```bash
grep "^real\b" baseline_main.log integration.log
# Or grep "=== " for per-stage breakdowns:
grep "=== " baseline_main.log | head -20
grep "=== " integration.log | head -20
```

What to look for:

- **Total time within ±5%**: green. The blitzes are infrastructure
  changes; they shouldn't move the needle on a small x_only bispinor
  smoke.
- **Total time 5–20% slower**: possibly the allocator change is
  surfacing real allocation overhead at this scale. Worth profiling
  but not a blocker — the previous allocator was the bad one
  (`platform` = eager cudaMalloc per allocation).
- **Total time >20% slower OR substantial per-stage shift**: real
  regression. Most likely suspect: Blitz #6 if cuSOLVERMp/ζ-fit stages
  shifted; Blitz #0 if everything is uniformly slower.
- **Total time faster**: this is what we expect if the previous run
  was on the `platform` allocator. The fix is real.

## Step 5 — If green, merge to main

```bash
cd /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
git checkout main
git merge --ff-only agent/install-blitz-integration  # fast-forward if no
                                                      # other commits landed
git push origin main                                 # only if you're ready
```

If you want to land them as separate PRs for review, the per-blitz
branches still exist:
- `agent/lorrax-c-allocator-fix` in lorrax_C
- `agent/blitz-{1,3,4-distinit,5-redo,6-redo}` in lorrax_D and the
  workspaces under `sources/blitz_workspaces/`

The cherry-pick onto integration didn't rewrite their hashes; you can
push each independently.

## Step 6 — If red

Bisect on the integration branch (one commit at a time). The commits
are deliberately ordered FFI → modulefile → CI so the build-side
changes land first:

```bash
git checkout c5577bc  # blitz #0 alone on top of main
# rebuild FFI, re-run, time
git checkout aeb22e0  # +blitz #1
# (re-time only matters if pip install -e . was used; with build.sh,
#  identical to c5577bc)
git checkout f0bfc5d  # +blitz #3
git checkout e7f828b  # +blitz #6 -- the highest-suspicion commit for
                      #  bispinor timing
# ... etc
```

The most likely regression suspects in order:

1. **Blitz #6** (`e7f828b`): ScratchAllocator migration changes how the
   per-q workspace gets pulled from the JAX pool. If there's a hidden
   serialization or extra `cudaStreamSynchronize` involved in
   `scratch.Allocate()` calls that wasn't there with persistent Ctx-owned
   `cudaMalloc`, that's where you'll see it.
2. **Blitz #0** (`c5577bc`): the `platform` → `cuda_async` allocator
   change. Should be strictly faster, but the async pool has a different
   warm-up behavior; first-call latencies might shift.
3. **Blitz #5** (`bdb934c`): if `INSTALL_RPATH` ends up pointing to a
   different library than before (e.g. resolved env var disagrees with
   the old literal), the .so could be linking to a different HDF5 or
   SLATE build. The `readelf -d` check in step 2 catches this.
4. **Blitz #3** (`f0bfc5d`): the post-init `jax.process_count() ==
   SLURM_NTASKS` assert is a warning, not a hard raise — but if it
   fires, FFI collectives are wrong. Watch the log for the
   `RuntimeWarning`.

## Cleanup

When done, optionally delete the persistent worktrees:

```bash
cd /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D
git worktree remove ../blitz_workspaces/blitz5
git worktree remove ../blitz_workspaces/blitz6
git worktree remove ../blitz_workspaces/blitz4-distinit
# branches survive in .git refs; delete with -D if you don't want them:
# git branch -D agent/blitz-5-redo agent/blitz-6-redo agent/blitz-4-distinit
```

The integration branch stays on lorrax_D until you merge or abandon.
