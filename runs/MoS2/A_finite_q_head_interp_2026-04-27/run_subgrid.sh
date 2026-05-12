#!/bin/bash
JID="${1}"; export SLURM_JOBID="$JID"
set -f; module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_A; module load lorrax_agent; set +f; set -e
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_finite_q_head_interp_2026-04-27
LORRAX_NGPU=1 LORRAX_NNODES=1 lxrun python3 -u v_q_subgrid_interp.py
