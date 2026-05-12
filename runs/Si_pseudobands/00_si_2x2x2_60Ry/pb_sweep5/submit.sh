#!/bin/bash
#SBATCH --account=m2651_g
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --job-name=lorrax_A_pb_k_sweep
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Fixed-nbnd~300 k-sweep: 6 configs × 3 seeds = 18 runs of each phase.
# Job allocation: 8 nodes × (4 tasks × 32 CPUs per task + 4 GPUs) = full Perlmutter A100 nodes.
# PB and COHSEX run 4-GPU + 4-task steps → 8 parallel (1 node each, --exclusive).
# Centroids / dipole / kin_ion: 1-GPU steps → pack 4 per node, 32 parallel total (NOT --exclusive).

set -u
module load lorrax_A

ROOT="/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry"
SWEEP="$ROOT/pb_sweep5"
COHSEX="$ROOT/cohsex_sweep5"
cd "$SWEEP"

PYPATH="/global/homes/j/jackm/software/lorrax_A/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
IMG="nvcr.io/nvidia/jax:25.04-py3"
SH="shifter --module=gpu --image=$IMG --env=PYTHONPATH=$PYPATH --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 --env=TF_GPU_ALLOCATOR=cuda_malloc_async"
throttle() { while [ $(jobs -r | wc -l) -ge $1 ]; do wait -n; done; }

# 6 configs × 3 seeds = 18 runs at fixed total-bands ≈ 300 (n_protected ≈ 28 + k×n_w×~0.9 ≈ 300).
CONFIGS=(
  "v1_k2_150w 2 150"
  "v1_k4_75w  4  75"
  "v1_k6_50w  6  50"
  "v1_k8_38w  8  38"
  "v1_k10_30w 10 30"
  "v1_k12_25w 12 25"
)

mkdir -p "$COHSEX"
for cfg in "${CONFIGS[@]}"; do read name pk w <<< "$cfg"
  for s in 0 1 2; do
    d="$SWEEP/${name}_s${s}"
    mkdir -p $d
    ln -sf ../../qe/nscf/silicon.save $d/silicon.save
    ln -sf ../../qe/nscf/Si.upf $d/Si.upf
    cat > $d/nscf.in <<EOF
[nscf]
save_dir = silicon.save
nbnd = 60
kgrid = 2 2 2
nosym = true
output = WFN.h5
tol = 1e-8

pseudobands = true
pb_version = 1
pb_k = ${pk}
pb_M_max = 1500
pb_F = 0.10
pb_n_windows = ${w}
EOF
  done
done

# --------------- STEP 1: PB (4-GPU steps, 8 parallel, --exclusive node) ---------------
echo "=== STEP 1: PB (18 runs, 8-way parallel × 4 GPUs) ==="
t0=$(date +%s)
for cfg in "${CONFIGS[@]}"; do read name pk w <<< "$cfg"
  for s in 0 1 2; do
    d="${name}_s${s}"
    (srun --nodes=1 --ntasks=4 --gpus-per-node=4 --cpus-per-task=32 --exclusive \
          $SH python3 -u -m psp.run_nscf -i $d/nscf.in --pb-seed $s > $d/nscf_pb.out 2>&1) &
    throttle 8
  done
done
wait
echo "  PB done in $(($(date +%s)-t0))s"

# --------------- STEP 1.5: cohsex dir setup ---------------
for cfg in "${CONFIGS[@]}"; do read name pk w <<< "$cfg"
  for s in 0 1 2; do
    d="$COHSEX/${name}_s${s}"
    mkdir -p $d
    ln -sf "$SWEEP/${name}_s${s}/WFN_pseudobands.h5" "$d/WFN.h5"
    ln -sf "$ROOT/17_bgw_explicit_60Ry/vcoul" "$d/bgw_vcoul.txt"
    ln -sf "$ROOT/qe/nscf/WFN.h5" "$d/sym_wfn.h5"
    ln -sf "$ROOT/qe/nscf/Si.upf" "$d/Si.upf"
    ln -sf "$ROOT/qe/nscf/silicon.save" "$d/silicon.save"
    mb=$(python3 -c "import h5py; print(int(h5py.File('$d/WFN.h5','r')['mf_header/kpoints/mnband'][()]))")
    cat > $d/cohsex.in <<EOF
[cohsex]
wfn_file = WFN.h5
nval = 8
ncond = 8
nband = $mb
bispinor = false
sys_dim = 3
bare_coulomb_cutoff = 60.0
epsilon_cutoff = 60.0
memory_per_device_gb = 28
centroids_file = centroids_frac_2000.txt
restart = false
screening = cohsex
vhead = 825.937
whead_0freq = 17.128
use_bgw_vcoul = true
bgw_vcoul_file = bgw_vcoul.txt
bgw_vcoul_sym_wfn = sym_wfn.h5
EOF
  done
done
cd "$COHSEX"

# --------------- STEP 2: centroids (1-GPU; 6 in parallel; NO --exclusive so 4/node) ---------------
echo "=== STEP 2: centroids (6 configs, 1 GPU each) ==="
t0=$(date +%s)
for cfg in "${CONFIGS[@]}"; do read name _ _ <<< "$cfg"
  (cd ${name}_s0 && srun --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=32 \
       $SH python3 -u -m centroid.kmeans_isdf 2000 --no-plot --seed 42 > centroid.out 2>&1) &
  throttle 32
done
wait
for cfg in "${CONFIGS[@]}"; do read name _ _ <<< "$cfg"
  for s in 1 2; do ln -sf ../${name}_s0/centroids_frac_2000.txt ${name}_s${s}/centroids_frac_2000.txt; done
done
echo "  centroids done in $(($(date +%s)-t0))s"

# --------------- STEP 3: dipole + kin_ion (1-GPU; 18 per phase; no --exclusive) ---------------
echo "=== STEP 3: dipole + kin_ion (18 each, 1 GPU; up to 32 concurrent) ==="
t0=$(date +%s)
for cfg in "${CONFIGS[@]}"; do read name _ _ <<< "$cfg"
  for s in 0 1 2; do
    d="${name}_s${s}"
    (cd $d && \
     srun --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=32 $SH python3 -u -m psp.get_dipole_mtxels  -i cohsex.in > dipole.out 2>&1 && \
     srun --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=32 $SH python3 -u -m gw.kin_ion_io_chunked -i cohsex.in > kin.out 2>&1) &
    throttle 32
  done
done
wait
echo "  dipole+kin done in $(($(date +%s)-t0))s"

# --------------- STEP 4: COHSEX (4-GPU, --exclusive, 8 parallel) ---------------
echo "=== STEP 4: COHSEX (18 runs, 8-way parallel × 4 GPUs) ==="
t0=$(date +%s)
for cfg in "${CONFIGS[@]}"; do read name _ _ <<< "$cfg"
  for s in 0 1 2; do
    d="${name}_s${s}"
    (cd $d && srun --nodes=1 --ntasks=4 --gpus-per-node=4 --cpus-per-task=32 --exclusive \
          $SH python3 -u -m gw.gw_jax -i cohsex.in > gw.out 2>&1) &
    throttle 8
  done
done
wait
echo "  COHSEX done in $(($(date +%s)-t0))s"

echo "ALL DONE"
