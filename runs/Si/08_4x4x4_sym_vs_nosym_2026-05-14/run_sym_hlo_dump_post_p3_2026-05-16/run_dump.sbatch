#!/bin/bash
#SBATCH --job-name=si-hlo-post-p3
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

# Module env
module use $HOME/modulefiles
module load lorrax_B

# Force fresh compile (bypass shared JAX compilation cache so XLA emits
# every module for dumping) — JAX_COMPILATION_CACHE_DIR is a normal
# KEY=VALUE env so it's fine to pass through shifter's --env=.
mkdir -p "$PWD/xla_dump" "$PWD/.jax_cache_fresh"
export LORRAX_SHIFTER="$LORRAX_SHIFTER --env=JAX_COMPILATION_CACHE_DIR=$PWD/.jax_cache_fresh"

# XLA_FLAGS value starts with `--xla_dump_to=...`; shifter --env= parsing
# gets confused when the value starts with `--`.  Set XLA_FLAGS inside
# the container via a bash -c wrapper instead.
DUMPDIR="$PWD/xla_dump"

echo "=== effective LORRAX_SHIFTER ==="
echo "$LORRAX_SHIFTER"
echo "=== launching with in-container XLA_FLAGS=--xla_dump_to=$DUMPDIR ==="

# Si runs at mesh 1x2 — 2 GPUs
LORRAX_NGPU=2 lxrun bash -c "export XLA_FLAGS=--xla_dump_to=$DUMPDIR; echo \"inside container: XLA_FLAGS=\$XLA_FLAGS\"; exec python3 -u -m gw.gw_jax -i $PWD/cohsex.in" 2>&1 | tee gw.out

echo "=== xla_dump entry count ==="
ls "$PWD/xla_dump" | wc -l
echo "=== sbatch wrapper done ==="
