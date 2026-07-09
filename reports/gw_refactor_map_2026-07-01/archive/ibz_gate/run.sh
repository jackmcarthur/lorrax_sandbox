#!/bin/bash
# Usage: run.sh <ibz|fullbz>
set -u
MODE="$1"; FFB=0; [ "$MODE" = "fullbz" ] && FFB=1
SW=/pscratch/sd/j/jackm/lorrax_sandbox/reports/memplanner_cleanup_2026-07-02/budget_sweep
D=/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_refactor_map_2026-07-01/ibz_gate/$MODE
mkdir -p "$D/tmp"; cd "$D" || exit 1
for f in WFN.h5 centroids_frac_642.txt kin_ion.h5 dipole.h5 kih.dat; do ln -sf $SW/$f $f; done
cat > cohsex.in <<INP
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
memory_per_device_gb = 28
wfn_file = WFN.h5
output_file = eqp0.dat
sigma_diag_file = sigma_diag.dat
INP
source /etc/profile.d/z00_lmod.sh 2>/dev/null
module use /global/homes/j/jackm/modulefiles; module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
export SLURM_JOBID=55413970 JAX_COMPILATION_CACHE_DIR=$SW/.jaxcache LORRAX_FORCE_FULL_BZ=$FFB
LORRAX_NGPU=1 lxrun python3 -u -m gw.gw_jax -i cohsex.in > run.log 2>&1
echo "$MODE (FFB=$FFB) exit=$? ; $(grep -iE 'q-IBZ reduction|full-BZ' run.log | head -1)"
