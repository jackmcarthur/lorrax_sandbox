#!/bin/bash
# BGW COHSEX reference runs. Sets up 5 new BGW dirs (parabands × 3 n_cond +
# pseudo × 2 n_cond). The pseudo_ncond208 slot reuses the pre-existing
# `03_bgw_pseudobands_100sl` directory (nbands=215 matches).
#
# Each BGW dir: epsilon + sigma with number_bands = 8 + n_cond.
# write_vcoul is present in epsilon.inp + sigma.inp so we get the vcoul text
# file that LORRAX reads via use_bgw_vcoul=true.
#
# Uses module-loaded BerkeleyGW 4.0 binaries (GPU-enabled NVHPC build).
# 4 MPI × 16 OMP per BGW job → one full A100 node. 4 BGW jobs run in
# parallel across the 4-node allocation.
set -u
module load berkeleygw

ROOT=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/cohsex_valcond_2026-04-18
JID=${SLURM_JOBID:?need SLURM_JOBID}
NSCFQ=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/qe/nscfq/WFNq.h5
PARA_WFN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/10_bgw_parabands_elpa/WFN_pb.h5
PSEUDO_WFN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/03_bgw_pseudobands_100sl/WFN_pseudo.h5

# ---- helpers ----
setup_bgw_dir() {
    local dir=$1 wfn=$2 ncond=$3
    mkdir -p $dir
    ln -sfn $wfn $dir/WFN.h5
    ln -sfn $wfn $dir/WFN_inner.h5
    ln -sfn $NSCFQ $dir/WFNq.h5
    local nb=$((8 + ncond))
    cat > $dir/epsilon.inp <<EOF
epsilon_cutoff 60.0
degeneracy_check_override
dont_check_norms
use_wfn_hdf5
frequency_dependence 0
number_bands $nb

write_vcoul

begin qpoints
  0.000500000  0.000000000  0.000000000   1.0 1
  0.000000000  0.000000000  0.500000000   1.0 0
  0.000000000  0.500000000  0.500000000   1.0 0
  0.500000000  0.500000000  0.500000000   1.0 0
end
EOF
    cat > $dir/sigma.inp <<EOF
band_index_min 1
band_index_max 16
number_bands $nb
screened_coulomb_cutoff 60.0

frequency_dependence 0
exact_static_ch 0

degeneracy_check_override
dont_check_norms
use_wfn_hdf5
dont_use_vxcdat
use_kihdat

write_vcoul

begin kpoints
  0.000000000  0.000000000  0.000000000  1.0
  0.000000000  0.000000000  0.500000000  1.0
  0.000000000  0.500000000  0.500000000  1.0
  0.500000000  0.500000000  0.500000000  1.0
end
EOF
    # Copy kih.dat from pseudo_100sl run (deterministic valence eigenvalues,
    # shared across all of our runs since scf is the same).
    ln -sf /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/03_bgw_pseudobands_100sl/kih.dat $dir/kih.dat
}

run_bgw_dir() {
    local dir=$1 log=$2
    (cd $dir && \
     echo "[bgw] epsilon on $dir" && \
     HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=16 \
         srun --jobid=$JID --nodes=1 --ntasks=4 --cpus-per-task=16 \
              --gpus-per-node=4 --exclusive \
              epsilon.cplx.x < epsilon.inp > epsilon.out 2>&1 && \
     echo "[bgw] sigma on $dir" && \
     HDF5_USE_FILE_LOCKING=FALSE OMP_NUM_THREADS=16 \
         srun --jobid=$JID --nodes=1 --ntasks=4 --cpus-per-task=16 \
              --gpus-per-node=4 --exclusive \
              sigma.cplx.x < sigma.inp > sigma.out 2>&1 && \
     echo "[bgw] DONE $dir") > $log 2>&1
}

# ---- Setup: 5 new dirs, 1 reused ----
for nc in 60 140 208; do
    setup_bgw_dir $ROOT/bgw/parabands_ncond${nc} $PARA_WFN $nc
done
for nc in 60 140; do
    setup_bgw_dir $ROOT/bgw/pseudo_ncond${nc} $PSEUDO_WFN $nc
done
# Reuse ncond208 pseudo via symlink (already ran at nbands=215 = 8+207+1
# padding). Match exactly by running our own.
setup_bgw_dir $ROOT/bgw/pseudo_ncond208 $PSEUDO_WFN 208

# ---- Run epsilon+sigma in parallel, 4 nodes = 4 BGW jobs concurrent ----
throttle() { while [ $(jobs -r | wc -l) -ge $1 ]; do wait -n; done; }

for nc in 60 140 208; do
    run_bgw_dir $ROOT/bgw/parabands_ncond${nc} $ROOT/logs/bgw_parabands_ncond${nc}.log &
    throttle 4
done
for nc in 60 140 208; do
    run_bgw_dir $ROOT/bgw/pseudo_ncond${nc} $ROOT/logs/bgw_pseudo_ncond${nc}.log &
    throttle 4
done
wait

echo "=== all BGW runs done ==="
for d in $ROOT/bgw/*/; do
    if grep -q "Job Done" $d/sigma.out 2>/dev/null; then
        echo "  OK: $d"
    else
        echo "  FAILED: $d"
    fi
done
