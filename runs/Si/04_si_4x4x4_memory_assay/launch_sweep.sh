#!/bin/bash
# Launch memory sweep on Perlmutter (interactive allocation required)
#
# Get allocation first (1 node, 4 GPUs, 4 tasks):
#   salloc --nodes=1 --qos=interactive --time=02:00:00 \
#          --constraint=gpu --gpus=4 --account=m2651 \
#          bash -c "sleep 7100"
#
# For 4-node runs (16 GPUs, 16 tasks):
#   salloc --nodes=4 --qos=interactive --time=04:00:00 \
#          --constraint=gpu --gpus=16 --account=m2651 \
#          bash -c "sleep 14300"
#
# Then from another terminal:
#   bash launch_sweep.sh [--xprof] [--config CONFIG_ID]
#   NODES=4 bash launch_sweep.sh [--xprof] [--config CONFIG_ID]

set -euo pipefail

NODES=${NODES:-1}
GPUS_PER_NODE=4
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

SITE=$HOME/scratchperl/.isdf/isdf_venvs/isdf_site
SHIFTER="shifter --module=gpu --image=nvcr.io/nvidia/jax:25.04-py3 \
    --env=PYTHONPATH=/global/u2/j/jackm/software/lorrax/src:$SITE \
    --env=JAX_ENABLE_X64=1 \
    --env=HDF5_USE_FILE_LOCKING=FALSE \
    --env=LORRAX_MEM_PROFILE=1 \
    --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Memory Sweep ==="
echo "Nodes: $NODES, GPUs: $TOTAL_GPUS ($GPUS_PER_NODE/node)"
echo "Tasks: $TOTAL_GPUS (1 GPU per task)"
echo "Working directory: $SCRIPT_DIR"
echo "Extra args: $@"
echo "===================="

cd "$SCRIPT_DIR"

# 1 task per GPU — this is the PRODUCTION execution mode.
# Use --gres=gpu:4 (all GPUs visible per node) with -n <total_gpus>.
# JAX distributed init handles the 1-GPU-per-process mapping.
# Do NOT use --gpus-per-task=1 — Shifter doesn't set CUDA_VISIBLE_DEVICES
# correctly and JAX distributed init fails.
srun --jobid="$SLURM_JOB_ID" \
    --gres=gpu:$GPUS_PER_NODE \
    -N "$NODES" \
    -n "$TOTAL_GPUS" \
    $SHIFTER \
    python3 -u run_memory_sweep.py "$@" \
    2>&1 | tee "sweep_${NODES}N_$(date +%Y%m%d-%H%M%S).out"
