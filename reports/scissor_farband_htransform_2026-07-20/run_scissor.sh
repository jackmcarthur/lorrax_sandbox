#!/bin/bash
set -u
JID="${JID:?set JID}"
DIR=/pscratch/sd/j/jackm/lorrax_sandbox/reports/scissor_farband_htransform_2026-07-20
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
SHIFTER="shifter --image=nvcr.io/nvidia/jax:25.04-py3 --env=MPLBACKEND=Agg --env=PYTHONPATH=$SITE"
srun --jobid=$JID -N1 -n1 --cpus-per-task=8 --overlap --immediate=120 \
  --job-name=lx-A-scissor --chdir="$DIR" \
  $SHIFTER python3 -u scissor_analysis.py 2>&1
