#!/bin/bash
source /etc/profile.d/zzz-lmod.sh
module use /global/homes/j/jackm/modulefiles; module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent; export SLURM_JOBID=55447628
BASE=/pscratch/sd/j/jackm/lorrax_sandbox/reports/memory_model_refit_2026-07-03
LOGD=$BASE/logs
# 1) bisp_p4_b28 with ns²-aware util
cd $BASE/val/bisp_p4_b28
LORRAX_NGPU=4 lxrun env -u TF_GPU_ALLOCATOR XLA_PYTHON_CLIENT_ALLOCATOR=default LORRAX_MEM_DEBUG=1 LORRAX_EXIT_AFTER_ZETA=1 \
   python3 -u -m gw.gw_jax -i cohsex.in > $LOGD/bisp_p4_b28_utilfix.log 2>&1
echo "bisp_b28 exit=$?"
grep -iE "r_chunk =|HWM estimate|high-water|RESOURCE_EXHAUSTED" $LOGD/bisp_p4_b28_utilfix.log | tail -4
# 2) regenerate gnppm reference (full run, NGPU=1)
GD=$BASE/gnppm_regen; rm -rf $GD; cp -r /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/tests/regression/gnppm_debug $GD; rm -rf $GD/tmp
cd $GD
LORRAX_NGPU=1 lxrun python3 -u -m gw.gw_jax -i gnppm_test.in > $LOGD/gnppm_regen.log 2>&1
echo "gnppm_regen exit=$?"
ls -la $GD/sigma_diag_gnppm_test.dat 2>/dev/null
echo "GNPPM_REGEN_DONE"
