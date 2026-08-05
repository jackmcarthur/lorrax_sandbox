#!/bin/bash
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module load berkeleygw
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/01_bgw_cohsex_noavg
echo "=== sigma.cplx.x noavg, jid 55396549, $(date), exe=$(which sigma.cplx.x) ==="
HDF5_USE_FILE_LOCKING=FALSE \
  srun --jobid=55396549 --gres=gpu:4 -N 1 -n 4 -c 16 \
  sigma.cplx.x < sigma.inp > sigma.out 2>&1
echo "=== rc=$? ==="
grep -c "Job Done" sigma.out && echo "JOB DONE" || echo "NO Job Done"
