#!/bin/bash
# LORRAX COHSEX runs on parabands: n_cond ∈ {100, 300, 600, 1000} × N_μ
# ∈ {400, 1200, 3200, 6000} = 16 jobs. val_cond centroids,
# use_bgw_vcoul=true so Σ matches BGW in V(q+G) exactly.
set -u
module load lorrax_B

ROOT=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/cohsex_highncond_2026-04-19
JID=${SLURM_JOBID:?need SLURM_JOBID}
PARA_WFN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/10_bgw_parabands_elpa/WFN_pb.h5
NSCF_WFN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/qe/nscf/WFN.h5
SI_UPF=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/qe/nscf/Si.upf
VHEAD=825.937
WHEAD_0FREQ=17.128

PYPATH="/global/homes/j/jackm/software/lorrax_B/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
IMG="nvcr.io/nvidia/jax:25.04-py3"
SH="shifter --module=gpu --image=$IMG --env=PYTHONPATH=$PYPATH --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 --env=TF_GPU_ALLOCATOR=cuda_malloc_async"

setup_lorrax_dir() {
    local dir=$1 ncond=$2 Nmu=$3
    mkdir -p $dir
    ln -sfn $PARA_WFN $dir/WFN.h5
    ln -sfn $SI_UPF $dir/Si.upf
    ln -sfn $NSCF_WFN $dir/sym_wfn.h5
    ln -sfn $ROOT/bgw/parabands_ncond${ncond}/vcoul $dir/bgw_vcoul.txt
    ln -sfn $ROOT/centroids/centroids_frac_ncond${ncond}_Nmu${Nmu}.txt \
        $dir/centroids_frac.txt
    local nb=$((8 + ncond))
    cat > $dir/cohsex.in <<EOF
[cohsex]
wfn_file = WFN.h5
nval = 8
ncond = $ncond
nband = $nb
bispinor = false
sys_dim = 3

bare_coulomb_cutoff = 60.0
epsilon_cutoff = 60.0
memory_per_device_gb = 28
centroids_file = centroids_frac.txt
restart = false

screening = cohsex

vhead = $VHEAD
whead_0freq = $WHEAD_0FREQ

use_bgw_vcoul = true
bgw_vcoul_file = bgw_vcoul.txt
bgw_vcoul_sym_wfn = sym_wfn.h5
EOF
}

for nc in 100 300 600 1000; do
    for Nm in 400 1200 3200 6000; do
        setup_lorrax_dir $ROOT/lorrax/parabands_ncond${nc}_Nmu${Nm} $nc $Nm
    done
done

throttle() { while [ $(jobs -r | wc -l) -ge $1 ]; do wait -n; done; }

echo "=== STEP 1: dipole + kin_ion (1 GPU per job; 16 parallel) ==="
for d in $ROOT/lorrax/*/; do
    [ -f $d/dipole.h5 ] && [ -f $d/kin_ion.h5 ] && continue
    (cd $d && \
     srun --jobid=$JID --gres=gpu:1 -N 1 -n 1 --cpus-per-task=8 $SH \
         python3 -u -m psp.get_dipole_mtxels -i cohsex.in > dipole.out 2>&1 && \
     srun --jobid=$JID --gres=gpu:1 -N 1 -n 1 --cpus-per-task=8 $SH \
         python3 -u -m gw.kin_ion_io_chunked -i cohsex.in > kin.out 2>&1) &
    throttle 16
done
wait

echo "=== STEP 2: COHSEX (4 GPUs per job; 4 parallel) ==="
for d in $ROOT/lorrax/*/; do
    [ -f $d/eqp0_noqsym.dat ] && continue
    [ -f $d/eqp0.dat ] && continue
    rm -f $d/gw.out
    (cd $d && \
     srun --jobid=$JID --gres=gpu:4 -N 1 -n 4 --cpus-per-task=8 --exclusive $SH \
         python3 -u -m gw.gw_jax -i cohsex.in > gw.out 2>&1) &
    throttle 4
done
wait

echo "=== all LORRAX runs done ==="
for d in $ROOT/lorrax/*/; do
    if [ -f $d/eqp0_noqsym.dat ] || [ -f $d/eqp0.dat ]; then
        echo "  OK: $(basename $d)"
    else
        echo "  FAILED: $(basename $d)"
    fi
done
