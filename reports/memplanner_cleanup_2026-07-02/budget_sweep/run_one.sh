#!/bin/bash
# Usage: run_one.sh <budget_int> <alloc: bfc|platform>
# Runs the COHSEX memory-planner validation for one budget in its own subdir.
set -u
B="$1"; ALLOC="${2:-bfc}"
export SLURM_JOBID=55412316
SWEEP=/pscratch/sd/j/jackm/lorrax_sandbox/reports/memplanner_cleanup_2026-07-02/budget_sweep
D=$SWEEP/budget_${B}_${ALLOC}
mkdir -p "$D/tmp"; cd "$D" || exit 1
for f in WFN.h5 centroids_frac_642.txt kin_ion.h5 dipole.h5 kih.dat; do ln -sf ../$f $f; done
cat > cohsex.in <<EOF
[cohsex]
restart = false
centroids_file = centroids_frac_642.txt
nval = 26
ncond = 54
nband = 80
sys_dim = 2
x_only = false
do_screened = true
bispinor = false
screening_method = minimax
no_degen_averaging = true
use_chunked_isdf = true
memory_per_device_gb = ${B}
wfn_file = WFN.h5
output_file = eqp0.dat
sigma_diag_file = sigma_diag.dat
EOF
source /etc/profile.d/zzz-lmod.sh 2>/dev/null
module use /global/homes/j/jackm/modulefiles
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
export JAX_COMPILATION_CACHE_DIR=$SWEEP/.jaxcache
if [ "$ALLOC" = "bfc" ]; then
  export LORRAX_SHIFTER="${LORRAX_SHIFTER/XLA_PYTHON_CLIENT_ALLOCATOR=platform/XLA_PYTHON_CLIENT_ALLOCATOR=default}"
  export LORRAX_SHIFTER="${LORRAX_SHIFTER/TF_GPU_ALLOCATOR=cuda_malloc_async/TF_GPU_ALLOCATOR=default}"
  # surface true peak: peak_bytes_in_use (BFC memory_stats) + running-max nvidia-smi
  export LORRAX_SHIFTER="${LORRAX_SHIFTER/--env=HDF5_USE_FILE_LOCKING=FALSE/--env=HDF5_USE_FILE_LOCKING=FALSE --env=LORRAX_MEM_DEBUG=1 --env=LORRAX_MEM_PROFILE=1}"
fi
echo "START budget=$B alloc=$ALLOC $(date)"
LORRAX_NGPU=1 lxrun python3 -u -m gw.gw_jax -i cohsex.in > run.log 2>&1
echo "EXIT=$? $(date)"
