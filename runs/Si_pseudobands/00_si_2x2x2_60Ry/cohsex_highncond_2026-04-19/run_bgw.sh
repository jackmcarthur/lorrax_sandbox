#!/bin/bash
# BGW COHSEX refs for parabands at n_cond ∈ {100, 300, 600, 1000}.
# Each: number_bands = 8 + n_cond. write_vcoul so LORRAX can consume.
set -u
module load berkeleygw

ROOT=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/cohsex_highncond_2026-04-19
JID=${SLURM_JOBID:?need SLURM_JOBID}
NSCFQ=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/qe/nscfq/WFNq.h5
PARA_WFN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/10_bgw_parabands_elpa/WFN_pb.h5

setup_bgw_dir() {
    local dir=$1 ncond=$2
    mkdir -p $dir
    ln -sfn $PARA_WFN $dir/WFN.h5
    ln -sfn $PARA_WFN $dir/WFN_inner.h5
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
    ln -sfn /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/03_bgw_pseudobands_100sl/kih.dat $dir/kih.dat
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

for nc in 100 300 600 1000; do
    setup_bgw_dir $ROOT/bgw/parabands_ncond${nc} $nc
done

throttle() { while [ $(jobs -r | wc -l) -ge $1 ]; do wait -n; done; }

for nc in 100 300 600 1000; do
    run_bgw_dir $ROOT/bgw/parabands_ncond${nc} $ROOT/logs/bgw_parabands_ncond${nc}.log &
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
