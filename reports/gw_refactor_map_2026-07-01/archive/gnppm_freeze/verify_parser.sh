#!/bin/bash
source /etc/profile.d/z00_lmod.sh 2>/dev/null || true
module use /global/homes/j/jackm/modulefiles; module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_D lorrax_agent
export SLURM_JOBID=55404949
cd /pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_refactor_map_2026-07-01/gnppm_freeze
echo "=== 1. import smoke (circular-import check) ==="
LORRAX_NGPU=1 lxrun python3 -c "import gw.kin_ion_io, psp.get_dipole_mtxels, psp.get_DFT_mtxels, psp.run_sternheimer; from gw.gw_config import read_lorrax_input; print('IMPORTS OK; parser:', read_lorrax_input.__module__)"
echo "=== 2. kin_ion_io run (exercises canonical parser + ecutrho->ecutwfc default) ==="
LORRAX_NGPU=1 lxrun python3 -u -m gw.kin_ion_io -i gnppm_test.in 2>&1 | grep -iE "ecutrho|kin_ion|Using|Wrote|Error|Traceback|sys_dim" | head
echo "=== EXIT $? ==="
