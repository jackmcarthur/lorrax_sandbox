#!/bin/bash
set -u
WT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_vh_sym
JID=$(squeue -u jackm -h -o "%.12i %j" | awk '/lx-alloc/{print $1; exit}')
echo "[alloc] JID=$JID"
[ -z "$JID" ] && { echo "no allocation; salloc needed"; exit 3; }
SH="shifter --image=nvcr.io/nvidia/jax:25.04-py3 --module=gpu,mpich \
--env=PYTHONPATH=$WT/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site \
--env=HDF5_USE_FILE_LOCKING=FALSE --env=MPLBACKEND=Agg"
srun --jobid=$JID --overlap -N1 -n1 --gres=gpu:1 --cpus-per-task=8 --immediate=90 \
  $SH python3 -u /pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/_sigcheck/sigc_omega.py 2>&1 | grep -vE "^\[.*shifter|WARN|Warning" | tail -60
