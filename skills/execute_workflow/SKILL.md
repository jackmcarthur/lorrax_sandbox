# GW Calculation Execution

Execute a QE → BGW and/or QE → GWJAX quasiparticle calculation whose input files have
already been assembled by the `build_input_files` skill. Read the run's `manifest.yaml`
first: its `pipeline` field says which codes to run, its `platform` field says where,
and its `system.prefix` gives the QE prefix for `<prefix>.save/`.

After the full pipeline completes, update `manifest.yaml` step states and record results
in `CHANGELOG.md` (at the repo root). CHANGELOG is the shared memory across sessions —
every meaningful milestone (SCF converged, comparison numbers, failures) should be noted
there so the next agent can continue without re-discovering what happened.

## Pipeline overview

1. **DFT** (QE `pw.x` SCF) → charge density
2. **Wavefunctions** (QE `pw.x` NSCF → `pw2bgw.x` → `wfn2hdf.x`) → `WFN.h5`, and
   `WFNq.h5` for BGW (not needed for 0D molecules)
3. **Screening + self-energy** (BGW epsilon → sigma, and/or GWJAX)
4. **Comparison** (if both codes ran): compare Cor' from `sigma_hp.log` vs sigC_EDFT
   from `eqp0.dat`. See `PARSE_OUTPUTS.md` for column definitions.

**Critical**: each NSCF overwrites `<prefix>.save/`, so `pw2bgw.x` + `wfn2hdf.x` must
run immediately after each NSCF, before the next `pw.x` call.

---

## Perlmutter (NERSC)

### Interactive session

Check for an existing allocation first: `squeue -u $USER`. If none exists:

```bash
salloc --nodes=4 --qos=interactive --time=04:00:00 \
       --constraint=gpu --gpus=16 --account=m2651 \
       bash -c "sleep 14300"
```

Each node has 4× NVIDIA A100 GPUs (40 GB; 80 GB nodes available with
`--constraint="gpu&hbm80g"`). The trailing `sleep` keeps the allocation alive.
Do all work from other terminals via `srun --jobid=$JOBID`.

Load modules once at session start:
```bash
module load espresso berkeleygw
```

### Quantum ESPRESSO

QE parallelizes over k-points with the `-npools` flag. Set `-npools` equal to the
number of GPUs: each pool gets one GPU and solves one k-point independently with no
inter-GPU communication. For small systems (<20 atoms), this is optimal and is the
only parallelization you need. There is no benefit to using more GPUs than k-points
in the calculation.

Always set `OMP_NUM_THREADS=16 -c 16`. The default 128 hardware threads per task
causes severe performance degradation on Perlmutter.

**Step 1: SCF** (1 node, 4 GPUs)

```bash
OMP_NUM_THREADS=16 srun --jobid=$JOBID --gres=gpu:4 -N 1 -n 4 -c 16 \
    pw.x -npools 4 -i scf.in > scf.out 2>&1
```

Verify: `grep "convergence has been achieved" scf.out`

**Step 2a: NSCF (unshifted) → WFN**

Scale GPUs to the number of k-points, up to 4 per node × $NODES nodes.

```bash
ln -sf ../scf/${PREFIX}.save .

OMP_NUM_THREADS=16 srun --jobid=$JOBID --gres=gpu:4 -N $NODES -n $((NODES*4)) -c 16 \
    pw.x -npools $((NODES*4)) -i nscf.in > nscf.out 2>&1
```

**Immediately** convert before any other pw.x call:

```bash
# pw2bgw: CPU workaround required (GPU pw2bgw segfaults with kih)
MPICH_GPU_SUPPORT_ENABLED=0 srun --jobid=$JOBID --gres=gpu:1 -N 1 -n 1 \
    pw2bgw.x -i pw2bgw.in > pw2bgw.out 2>&1

srun --jobid=$JOBID --gres=gpu:1 -N 1 -n 1 wfn2hdf.x BIN WFN WFN.h5
```

Verify: `ls -la WFN.h5 vxc.dat kih.dat`

**Step 2b: NSCF (shifted) → WFNq** (BGW only; skip for 0D)

```bash
OMP_NUM_THREADS=16 srun --jobid=$JOBID --gres=gpu:4 -N $NODES -n $((NODES*4)) -c 16 \
    pw.x -npools $((NODES*4)) -i nscfq.in > nscfq.out 2>&1

MPICH_GPU_SUPPORT_ENABLED=0 srun --jobid=$JOBID --gres=gpu:1 -N 1 -n 1 \
    pw2bgw.x -i pw2bgwq.in > pw2bgwq.out 2>&1

srun --jobid=$JOBID --gres=gpu:1 -N 1 -n 1 wfn2hdf.x BIN WFNq WFNq.h5
```

### BerkeleyGW

BGW uses one MPI rank per GPU. Always set `HDF5_USE_FILE_LOCKING=FALSE` on Lustre.

**Step 3: Epsilon**

```bash
ln -sf WFN.h5 WFN_inner.h5

HDF5_USE_FILE_LOCKING=FALSE \
srun --jobid=$JOBID --gres=gpu:4 -N $NODES -n $((NODES*4)) -c 16 \
    epsilon.cplx.x < epsilon.inp > epsilon.out 2>&1
```

Verify: `grep "Job Done" epsilon.out && ls eps0mat.h5`
(For grids > 1×1×1, also verify `epsmat.h5` exists.)

**Step 4: Sigma**

```bash
HDF5_USE_FILE_LOCKING=FALSE \
srun --jobid=$JOBID --gres=gpu:4 -N $NODES -n $((NODES*4)) -c 16 \
    sigma.cplx.x < sigma.inp > sigma.out 2>&1
```

Verify: `grep "Job Done" sigma.out`
Outputs: `sigma_hp.log` (preferred for comparison), `sigma.out`, `ch_converge.dat`.

If sigma fails silently (no "Job Done"), the most common cause is `WFN.h5` having
too few bands — `nbnd` in NSCF must exceed `number_bands` in sigma.inp by ≥ 2.

### GWJAX

GWJAX runs in the Shifter container (NVIDIA JAX image; jax 0.5.3.dev20250415 +
CUDA-12 plugins — verified from the image filesystem 2026-07-11; note the
lorrax pyproject pins jax>=0.9/cuda13, a build target the pinned image does
NOT satisfy — see KNOWN_SANDBOX_ERRORS 2026-06-15) for
multi-GPU execution. Do not use `uv run` for multi-GPU (known NamedSharding segfault).

Define the Shifter prefix once per session:

```bash
# Shifter does NOT expand $HOME, so paths below are literal. The nvhpc/phdf5/slate
# FFI deps live in $HOME/software (relocated off $SCRATCH 2026-06-24 — scratch is
# purged). This is the Cray-MPICH variant (current production); for the OpenMPI/hpcx
# variant swap the phdf5 stage to lorrax_phdf5_openmpi/stage, use nvhpc 25.5_cuda12.9,
# and drop the mpich module + the gtl LD_PRELOAD.
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
LORRAX_SRC=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src   # active checkout
VOL="--volume=/global/homes/j/jackm/software/lorrax_nvhpc:/lorrax_nvhpc \
     --volume=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage:/lorrax_phdf5 \
     --volume=/global/homes/j/jackm/software/lorrax_slate_cray/stage:/lorrax_slate"
SHIFTER="shifter --module=gpu,mpich --image=nvcr.io/nvidia/jax:25.04-py3 $VOL \
    --env=PYTHONPATH=$LORRAX_SRC:$SITE \
    --env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep \
    --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 \
    --env=MPICH_GPU_SUPPORT_ENABLED=1 \
    --env=JAX_ENABLE_X64=1 \
    --env=HDF5_USE_FILE_LOCKING=FALSE"

# Wrap EVERY srun command as `$SEL $SHIFTER $INC python3 ...` (see Steps 5-6 below):
#   SEL = per-rank GPU selector — sets CUDA_VISIBLE_DEVICES=$SLURM_LOCALID, required by
#         SLATE/cuSolverMp's 1-device-per-process model (use it even single-GPU).
#   INC = re-asserts MPICH_GPU_SUPPORT_ENABLED=1 inside the container (shifter's
#         --module=mpich unsets it), enabling GPU-Direct RDMA for the MPI FFI libs.
SEL=$LORRAX_SRC/ffi/common/cpp/select_gpu.sh
INC=$LORRAX_SRC/ffi/common/cpp/in_container.sh
```

Shifter rules: use literal `/global/homes/j/jackm/...` paths (Shifter does not expand
`$HOME`; verified that `/global/homes` works as a `--volume` source).
Do NOT use `isdf_pyuserbase` overlay (PJRT conflict); use `isdf_site` only. Do NOT use
`--gpu-bind` or hand-set a global `CUDA_VISIBLE_DEVICES`; instead let `select_gpu.sh`
(`$SEL`) bind one GPU per rank via `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`. Without it,
every rank lands on GPU 0 and multi-GPU GW dies with NCCL "Duplicate GPU detected".

**Step 5: Preprocessing** (single GPU each)

Read the centroid count from `centroids_frac_<N>.txt` in cohsex.in. Centroids depend
only on geometry and can be symlinked across k-grid variants of the same system.

```bash
N_CENTROIDS=640   # from centroids_frac_<N>.txt

# a) ISDF centroids (skip if file exists or is symlinked).  Entrypoint is
#    ``centroid.kmeans_cli`` — the ``kmeans_isdf`` module is the algorithm
#    (no __main__).  Add ``--density-mode current`` to additionally produce
#    ``centroids_frac_<N>_current.txt`` for bispinor V_q (required when
#    cfg.bispinor=True).
#
#    WEIGHT: the scalar channel defaults to ``--centroid-weight band_range``
#    = w(r) = Σ_{n∈window} Σ_k w_k|ψ_nk(r)|² over the σ window
#    (0, nval+ncond).  PASS ``--weight-bands 0:<nband>`` to weight by the
#    full band range the run actually uses — on a SLAB the occupied-only
#    ρ(r) puts zero centroids in the vacuum, so vacuum-localized conduction
#    states have no ISDF quadrature support and their ⟨nk|V_H|nk⟩ (hence
#    Vxc) is sign-wrong: MoS2 6×6 |ΔVxc| vs QE kih.dat 27.7 eV mean /
#    275.6 eV max → 0.23 / 2.17 eV when weighted by bands 0:200.
#    ``--centroid-weight charge_density`` restores the historical weight.
#    Orbit closure runs under the charge-density point group either way.
srun --jobid=$JOBID --gres=gpu:1 -N 1 -n 1 $SEL $SHIFTER $INC \
    python3 -u -m centroid.kmeans_cli $N_CENTROIDS --seed 42 \
        --weight-bands 0:$NBAND

# b) Dipole matrix elements
srun --jobid=$JOBID --gres=gpu:1 -N 1 -n 1 $SEL $SHIFTER $INC \
    python3 -u -m psp.get_dipole_mtxels -i cohsex.in

# c) Kinetic + ionic Hamiltonian
srun --jobid=$JOBID --gres=gpu:1 -N 1 -n 1 $SEL $SHIFTER $INC \
    python3 -u -m gw.kin_ion_io -i cohsex.in
```

Verify: `ls centroids_frac_*.txt dipole.h5 kin_ion.h5`

**Step 6: GW calculation** (multi-GPU)

```bash
srun --jobid=$JOBID --gres=gpu:4 -N $NODES -n $((NODES*4)) \
    $SEL $SHIFTER \
    --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    $INC \
    python3 -u -m gw.gw_jax -i $(pwd)/cohsex.in \
    2>&1 | tee gw.out
```

Verify: `ls eqp0.dat sigma_freq_debug.dat sigma_mnk.h5`

If `eqp0.dat` lacks `sigC_EDFT` columns, check that `kin_ion.h5` exists.
(The at-DFT Σ_c interpolation always runs in dynamic modes; the legacy
`sigma_at_dft_energies` key is deprecated — `qp_solver = one_shot_dft` is
the default.)

**Timing reference** (MoS2, 16 GPUs / 4 nodes):

| K-grid | Time |
|--------|------|
| 4×4 | ~109 s |
| 5×5 | ~161 s |
| 6×6 | ~208 s |
| 16×16 | ~27 min |

**Running multiple k-grids concurrently** (1 node each):

```bash
for KG in 4x4 5x5 6x6; do
    srun --jobid=$JOBID --exclusive --gres=gpu:4 -N 1 -n 4 \
        $SEL $SHIFTER \
        --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
        --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
        $INC \
        python3 -u -m gw.gw_jax -i $PSCRATCH/kgrid_cluster/$KG/cohsex.in \
        > $PSCRATCH/kgrid_cluster/$KG/gw.out 2>&1 &
done
wait
```

### Step 7: Comparison

Compare BGW Cor' (SX-X + CH' from sigma_hp.log) against GWJAX sigC_EDFT (eqp0.dat).
The script matches k-points by crystal coordinates mod G and handles the 1-indexed /
0-indexed band offset.

```bash
python compare_bgw_gwjax.py \
    --bgw-hp sigma_hp.log --gw-eqp eqp0.dat --wfn WFN.h5 \
    --out-dat compare_cor.dat --out-png compare_cor.png
```

### After the pipeline

Update `manifest.yaml` step states. Add a CHANGELOG.md entry:

```markdown
## 2026-04-03: MoS2 6×6 complete
- max |ΔCor| = 2.41 eV (constant offset, same as 3×3)
- No k-grid dependence — not a convergence issue
```

---

## Local

QE and BGW binaries in `$PATH`. LORRAX is installed as an editable dependency
of this sandbox via `pyproject.toml`, so `uv run python -m <module>` works
directly from any directory inside the sandbox. Run `uv sync` once after a
fresh clone to create the venv and lock file. All commands run from inside the
run directory (e.g. `runs/co/base/`).

**Steps 1–2: QE** (same pipeline as Perlmutter, `mpirun` instead of `srun`)

```bash
mpirun -np $NPROC pw.x -npools $NPROC -i scf.in > scf.out
```

Verify: `grep "convergence has been achieved" scf.out`

```bash
mpirun -np $NPROC pw.x -npools $NPROC -i nscf.in > nscf.out
mpirun -np 1 pw2bgw.x -i pw2bgw.in > pw2bgw.out
mpirun -np 1 wfn2hdf.x BIN WFN WFN.h5
```

Verify: `ls -la WFN.h5 vxc.dat kih.dat`

For BGW pipeline, repeat with nscfq/pw2bgwq/WFNq (skip for 0D).

**Steps 3–4: BGW**

```bash
ln -sf WFN.h5 WFN_inner.h5
HDF5_USE_FILE_LOCKING=FALSE mpirun -np $NPROC epsilon.cplx.x < epsilon.inp > epsilon.out
```

Verify: `grep "Job Done" epsilon.out && ls eps0mat.h5`

```bash
HDF5_USE_FILE_LOCKING=FALSE mpirun -np $NPROC sigma.cplx.x < sigma.inp > sigma.out
```

Verify: `grep "Job Done" sigma.out && ls sigma_hp.log`

**Steps 5–6: LORRAX** (all modules accept `-i <path-to-input>`)

```bash
uv run python -u -m centroid.kmeans_cli $N_CENTROIDS --seed 42
uv run python -u -m psp.get_dipole_mtxels -i $(pwd)/cohsex.in
uv run python -u -m gw.kin_ion_io -i $(pwd)/cohsex.in
uv run python -u -m gw.gw_jax -i $(pwd)/cohsex.in 2>&1 | tee gw.out
```

Verify: `ls centroids_frac_*.txt dipole.h5 kin_ion.h5` (after preprocessing)
Verify: `ls eqp0.dat sigma_freq_debug.dat` (after GW)

JAX will use a GPU if available, otherwise fall back to CPU (fine for small
test systems like CO at Gamma). No extra environment setup needed.

---

## Common pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| pw2bgw segfault | GPU bug with kih | `MPICH_GPU_SUPPORT_ENABLED=0` |
| `Invalid generic resource` | Wrong flag with `--jobid` | `--gres=gpu:N` not `--gpus=N` |
| HDF5 locking errors | Lustre | `HDF5_USE_FILE_LOCKING=FALSE` |
| `PJRT_Api already exists` | Wrong overlay | `isdf_site` only, not `isdf_pyuserbase` |
| No output / hangs | Buffering | `python3 -u` |
| Slow QE/BGW | 128 OMP threads | `OMP_NUM_THREADS=16 -c 16` |
| Session dies | No foreground process | `salloc ... bash -c "sleep 14300"` |
| cuFFT OOM | XLA scratch | `memory_per_device_gb = 28` |
| sigma silent fail | Too few bands | NSCF nbnd > number_bands + 2 |
| No sigC_EDFT | Missing kin_ion.h5 | Run kin_ion_io (at-DFT Σ_c always on in dynamic modes) |
| Restart fails | Stale tensors | Delete `tmp/isdf_tensors_*.h5`, `restart = false` |
