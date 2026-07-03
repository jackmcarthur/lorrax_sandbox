#!/bin/bash
source /etc/profile.d/zzz-lmod.sh
module use /global/homes/j/jackm/modulefiles
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
export SLURM_JOBID=55447628
VBASE=/pscratch/sd/j/jackm/lorrax_sandbox/reports/memory_model_refit_2026-07-03/val
LOGD=/pscratch/sd/j/jackm/lorrax_sandbox/reports/memory_model_refit_2026-07-03/logs
RES=/pscratch/sd/j/jackm/lorrax_sandbox/reports/memory_model_refit_2026-07-03/sweep_results.txt
: > $RES
# charge_p4_b18 already done; run the other five 4-GPU cells
for cell in charge_p4_b10 charge_p4_b28 bisp_p4_b10 bisp_p4_b18 bisp_p4_b28; do
  cd $VBASE/$cell
  L=$LOGD/$cell.log
  LORRAX_NGPU=4 lxrun env -u TF_GPU_ALLOCATOR XLA_PYTHON_CLIENT_ALLOCATOR=default \
     LORRAX_MEM_DEBUG=1 LORRAX_EXIT_AFTER_ZETA=1 \
     python3 -u -m gw.gw_jax -i cohsex.in > $L 2>&1
  echo "===== $cell (exit $?) =====" >> $RES
  grep -iE "r_chunk *=|band_chunk *=|HWM estimate|binder|GPU high-water" $L | sed 's/^/  /' >> $RES
done
echo "SWEEP DONE" >> $RES
