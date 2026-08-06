#!/bin/bash
#SBATCH --job-name=si-sym-mchead
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:25:00
#SBATCH --account=m2651
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module use $HOME/modulefiles
module load lorrax_B
LORRAX_NGPU=2 lxrun python3 -u -m gw.gw_jax -i $PWD/cohsex.in 2>&1 | tee gw.out
