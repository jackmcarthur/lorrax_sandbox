#!/bin/bash
#SBATCH -J fastloop
#SBATCH -p development
#SBATCH -A PHY25006
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o /scratch2/08271/jackmc/lorrax_sandbox/fastloop/work/fastloop.%j.out
#SBATCH -e /scratch2/08271/jackmc/lorrax_sandbox/fastloop/work/fastloop.%j.out
# ---------------------------------------------------------------------------
# fastloop standard form: the full mini-deck driver chain
# (kmeans -> dipole -> kin-ion -> gw -> htransform) at P=1 plus the
# 4-host-device sharding leg, gated against fastloop/pins_mini.json.
#
# Two ways to run:
#   sbatch fastloop/run_fastloop.sbatch          # 1-node dev job
#   bash   fastloop/run_fastloop.sbatch          # inside ANY existing
#                                                #   allocation (idev, or a
#                                                #   step of a bigger job)
# The login node is refused: jax cannot import there (glibc 2.17 vs wheel
# 2.28 — repo docs/environment/overview.md layer 2, verified 2026-07-31).
#
# Knobs (env): FASTLOOP_MODE=check|pin, FASTLOOP_ARGS extra runner args,
# FASTLOOP_THREADS, LORRAX_SRC (defaults to the LIVE tree — this is a
# pre-commit check of the working tree, deliberately unlike production
# jobs, which read frozen bundles), LORRAX_SIF, LORRAX_FFI_HOST_SO.
# Exit: 0 pass | 1 parity drift | 2 stage failure | 3 refusal.
# ---------------------------------------------------------------------------
set -u
case "$(hostname)" in login*)
  echo "REFUSING: login node (no jax outside the container; sbatch this)"; exit 3;;
esac
set +u; module load tacc-apptainer 2>/dev/null; set -u

SB=/scratch2/08271/jackmc/lorrax_sandbox
SIF=${LORRAX_SIF:-/scratch2/08271/jackmc/lorrax_setup/py312.sif}
VENVPY=/work2/08271/jackmc/frontera/lorrax_env/.venv/bin/python
BIND=/home1,/work2,/scratch1,/scratch2,/opt/intel,/tmp
export LORRAX_SRC=${LORRAX_SRC:-/work2/08271/jackmc/frontera/lorrax/src}
export LORRAX_SRC_COMMIT=$(git -C "$LORRAX_SRC/.." rev-parse --short HEAD 2>/dev/null || echo unknown)
mkdir -p $SB/fastloop/work

echo "=== fastloop $(date) host=$(hostname) job=${SLURM_JOB_ID:-none} src@$LORRAX_SRC_COMMIT mode=${FASTLOOP_MODE:-check} ==="
apptainer exec --bind $BIND "$SIF" \
  "$VENVPY" -u $SB/fastloop/run_fastloop.py \
  --mode "${FASTLOOP_MODE:-check}" ${FASTLOOP_ARGS:-}
rc=$?
echo "[fastloop rc=$rc] $(date)"
exit $rc
