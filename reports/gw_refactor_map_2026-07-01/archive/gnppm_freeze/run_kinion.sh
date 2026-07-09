#!/bin/bash
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module use /global/homes/j/jackm/modulefiles; module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
export SLURM_JOBID=55404949
cd /pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_refactor_map_2026-07-01/gnppm_freeze
rm -f kin_ion.h5
LORRAX_NGPU=1 lxrun python3 -u -m gw.kin_ion_io -i gnppm_test.in 2>&1 | grep -iE "ecutrho|Using|Wrote|kin_ion|Error|Traceback|Done" | head
echo "=== EXIT $? ; kin_ion.h5 written? ==="; ls -la kin_ion.h5 2>&1 | tail -1
