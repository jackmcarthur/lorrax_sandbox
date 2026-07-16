#!/bin/bash -l
# Sym-centroid BSE-degeneracy experiment (Si si_cohsex_debug fixture).
# Module-free srun+shifter pattern (adapted from runs/VI3/.../run_vi3_lorrax.sh);
# modules are broken in scripted shells (KNOWN_SANDBOX_ERRORS 2026-07-15).
# Launch via:  salloc ... -J lx-alloc-$USER bash run_experiment.sh
set -u

RUN=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16
LROOT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A
SEL=$LROOT/src/ffi/common/cpp/select_gpu.sh
INC=$LROOT/src/ffi/common/cpp/in_container.sh
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
VOL="--volume=/global/homes/j/jackm/software/lorrax_nvhpc:/lorrax_nvhpc --volume=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage:/lorrax_phdf5 --volume=/global/homes/j/jackm/software/lorrax_slate_cray/stage:/lorrax_slate"
LDP="--env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 --env=MPICH_GPU_SUPPORT_ENABLED=1"
COMMON="--image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich $VOL --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_PREALLOCATE=false --env=XLA_PYTHON_CLIENT_ALLOCATOR=platform --env=JAX_ENABLE_X64=1 --env=JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache $LDP"
SHIFTER="shifter $COMMON --env=PYTHONPATH=$LROOT/src:$SITE:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
J="--jobid=${SLURM_JOBID} --overlap"
STEP="srun $J -N1 -n1 --gres=gpu:1 $SEL $SHIFTER $INC"

N_C=${N_C:-960}
SEED=${SEED:-42}
OVS=${OVS:-1.0}

echo "=== sym-centroid experiment START $(date) (JOBID=$SLURM_JOBID, N_C=$N_C seed=$SEED oversample=$OVS) ==="

run_arm() {
  local arm="$1"     # old | sym
  local kflag="$2"   # --no-orbit | (empty)
  local wd=$RUN/work_$arm
  cd $wd
  echo "--- [$arm] centroids ($kflag) $(date) ---"
  $STEP python3 -u -m centroid.kmeans_cli $N_C $kflag --seed $SEED --oversample $OVS \
      > cent_$arm.out 2>&1
  local cent
  cent=$(ls -t centroids_frac_*.txt 2>/dev/null | head -1)
  if [ -z "$cent" ]; then echo "[$arm] FAIL: no centroids produced"; tail -20 cent_$arm.out; return 1; fi
  echo "[$arm] centroids file = $cent ($(grep -c -v '^#' $cent) points)"
  sed -i "s|^centroids_file = .*|centroids_file = $cent|" cohsex_si_test.in
  grep '^centroids_file' cohsex_si_test.in

  echo "--- [$arm] gw_jax COHSEX (do_screened) $(date) ---"
  $STEP python3 -u -m gw.gw_jax -i $wd/cohsex_si_test.in > gw_$arm.out 2>&1
  if [ ! -s eqp_si_test.dat ]; then echo "[$arm] WARN: eqp_si_test.dat missing"; tail -25 gw_$arm.out; fi
  ls -la tmp/isdf_tensors_*.h5 2>&1 | tail -1

  echo "--- [$arm] dense-H degeneracy analysis $(date) ---"
  $STEP python3 -u $RUN/analyze_degeneracy.py $wd/cohsex_si_test.in $arm $RUN/results_$arm.json \
      > analyze_$arm.out 2>&1
  cat analyze_$arm.out
}

run_arm old "--no-orbit" || echo "OLD arm had an error"
run_arm sym ""           || echo "SYM arm had an error"

echo "=== sym-centroid experiment DONE $(date) ==="
