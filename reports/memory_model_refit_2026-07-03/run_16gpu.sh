#!/bin/bash
source /etc/profile.d/zzz-lmod.sh
module use /global/homes/j/jackm/modulefiles; module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent; export SLURM_JOBID=55447628
BASE=/pscratch/sd/j/jackm/lorrax_sandbox/reports/memory_model_refit_2026-07-03
LOGD=$BASE/logs; RES=$BASE/sweep16_results.txt; : > $RES
for cell in charge_p16_b10 charge_p16_b28 bisp_p16_b10 bisp_p16_b28; do
  cd $BASE/val/$cell
  L=$LOGD/$cell.log
  LORRAX_NGPU=4 LORRAX_NNODES=4 lxrun env -u TF_GPU_ALLOCATOR XLA_PYTHON_CLIENT_ALLOCATOR=default \
     LORRAX_MEM_DEBUG=1 LORRAX_EXIT_AFTER_ZETA=1 \
     python3 -u -m gw.gw_jax -i cohsex.in > $L 2>&1
  echo "===== $cell (exit $?) =====" >> $RES
  grep -iE "Mesh:|r_chunk *=|band_chunk *=|persistent|HWM estimate|binder|GPU high-water|RESOURCE_EXHAUSTED" $L | sed 's/^/  /' >> $RES
done
echo "SWEEP16 DONE" >> $RES
