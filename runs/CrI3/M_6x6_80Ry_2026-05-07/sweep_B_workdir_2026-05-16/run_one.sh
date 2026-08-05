#!/usr/bin/env bash
# Run one config of the sweep. Args: config_name r_chunk band_chunk
# Writes gw_<config>.out and appends a row to /tmp/bispinor_80ry_sweep_B.tsv.
# Uses a hand-written srun (bypassing lxrun shell function) so it works in any bash.

set -u

CFG="$1"
R="$2"
B="$3"
GFLAT=360
JID=53058097
SWEEP_DIR="/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/sweep_B_workdir_2026-05-16"
TSV=/tmp/bispinor_80ry_sweep_B.tsv
PLOG=/tmp/bispinor_80ry_sweep_progress.log

LORRAX_BASE=/global/homes/j/jackm/software/lorrax_B
SELECT_GPU=$LORRAX_BASE/src/ffi/common/cpp/select_gpu.sh
IN_CONTAINER=$LORRAX_BASE/src/ffi/common/cpp/in_container.sh

cd "$SWEEP_DIR" || exit 1

# Mutate config
sed -i "s/^r_chunk_size = .*/r_chunk_size = $R/" cohsex.in
sed -i "s/^band_chunk_size = .*/band_chunk_size = $B/" cohsex.in
sed -i "s/^gflat_chunk_size = .*/gflat_chunk_size = $GFLAT/" cohsex.in

# Clean state
rm -rf tmp/ gw_${CFG}.out
OUT="gw_${CFG}.out"
mkdir -p tmp 2>/dev/null
lfs setstripe -c 16 -S 4M tmp/ >/dev/null 2>&1 || true

echo "=== SWEEP_B CONFIG ${CFG} r=${R} b=${B} START $(date) ===" | tee -a $PLOG
echo "=== r_chunk=${R} band_chunk=${B} gflat_chunk=${GFLAT} ===" > $OUT
cat cohsex.in >> $OUT
echo "===" >> $OUT

# Get nodes from squeue (single field "%N" cleanly).
ALLOC_NODES=$(squeue -j $JID -h -o "%N" 2>/dev/null)
if [ -z "$ALLOC_NODES" ] || [ "$ALLOC_NODES" = "(null)" ]; then
  echo "SWEEP_B FAIL: can't read alloc nodes for JID=$JID (got '$ALLOC_NODES')" | tee -a $PLOG
  exit 1
fi
# Expand the brace-encoded nodelist
NODELIST_EXPANDED=$(scontrol show hostnames "$ALLOC_NODES" | tr '\n' ',' | sed 's/,$//')
echo "Using nodes: $NODELIST_EXPANDED" >> $OUT

# Compose the shifter env that lxrun would produce
SHIFTER_ENV_PARTS=(
  "--env=PYTHONPATH=$LORRAX_BASE/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
  "--env=HDF5_USE_FILE_LOCKING=FALSE"
  "--env=XLA_PYTHON_CLIENT_PREALLOCATE=false"
  "--env=XLA_PYTHON_CLIENT_ALLOCATOR=platform"
  "--env=TF_GPU_ALLOCATOR=cuda_malloc_async"
  "--env=LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64:/opt/udiImage/modules/mpich:/opt/udiImage/modules/mpich/dep:/global/common/software/nersc9/darshan/default/lib"
  "--env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0"
  "--env=MPICH_GPU_SUPPORT_ENABLED=1"
  "--env=JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache"
  "--env=LORRAX_MPI_INCLUDE_DIR=/lorrax_phdf5/include"
  "--env=LORRAX_MPICH_LIB_DIR=/opt/udiImage/modules/mpich"
  "--env=LORRAX_FORCE_FULL_BZ=1"
)

T0=$(date +%s)
WALL_START=$(date +%s)

# Launch: 4 nodes × 4 ranks/node = 16 ranks
srun --jobid=$JID --mpi=cray_shasta \
  --nodelist=$NODELIST_EXPANDED -N 4 -n 16 --gres=gpu:4 \
  --immediate=10 --job-name=lx-B-sw${CFG}-$$ \
  "$SELECT_GPU" \
  shifter \
    --image=nvcr.io/nvidia/jax:25.04-py3 \
    --module=gpu,mpich \
    --volume=/global/homes/j/jackm/software/lorrax_nvhpc:/lorrax_nvhpc \
    --volume=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage:/lorrax_phdf5 \
    --volume=/global/homes/j/jackm/software/lorrax_slate_cray/stage:/lorrax_slate \
    "${SHIFTER_ENV_PARTS[@]}" \
  "$IN_CONTAINER" \
  python3 -u -m gw.gw_jax -i $PWD/cohsex.in >> $OUT 2>&1 &
PID=$!
echo "Launched srun PID=$PID for config $CFG (16 ranks across $NODELIST_EXPANDED)" | tee -a $OUT

# Wait for "Started zeta fitting"
COMPILED=0
while true; do
  if grep -q "Started zeta fitting" $OUT 2>/dev/null; then COMPILED=1; break; fi
  if [ $(($(date +%s) - T0)) -gt 600 ]; then echo "compile_timeout" >> $OUT; break; fi
  if grep -qE "Out of memory|RESOURCE_EXHAUSTED|XlaRuntimeError|Traceback" $OUT 2>/dev/null; then break; fi
  if ! kill -0 $PID 2>/dev/null; then break; fi
  sleep 5
done

STATUS="unknown"
if [ "$COMPILED" -eq 0 ]; then
  if grep -qE "Out of memory|RESOURCE_EXHAUSTED|XlaRuntimeError" $OUT 2>/dev/null; then
    STATUS="OOM"
  elif grep -qE "Traceback|FAILED" $OUT 2>/dev/null; then
    STATUS="crash"
  elif [ $(($(date +%s) - T0)) -gt 600 ]; then
    STATUS="compile_timeout"
  else
    STATUS="early_exit"
  fi
fi

# If compiled, sample steady-state
if [ "$COMPILED" -eq 1 ]; then
  # Need at least 6 chunks past chunk 1 to compute mean reliably
  if [ "$R" -le 16384 ]; then
    # n_rchunks_total ≈ 69 → wait for chunk 12-15
    TARGET_PATTERN="r-chunk +(12|13|14|15|16) / "
  else
    # r=24576, n_rchunks_total ≈ 46 → wait for chunk 8-10
    TARGET_PATTERN="r-chunk +(8|9|10|11|12) / "
  fi
  while true; do
    if grep -qE "$TARGET_PATTERN" $OUT 2>/dev/null; then STATUS="ok"; break; fi
    if [ $(($(date +%s) - T0)) -gt 900 ]; then STATUS="fit_timeout"; break; fi
    if grep -qE "Out of memory|RESOURCE_EXHAUSTED|XlaRuntimeError" $OUT 2>/dev/null; then STATUS="OOM"; break; fi
    if grep -qE "Traceback|FAILED" $OUT 2>/dev/null; then STATUS="crash"; break; fi
    if ! kill -0 $PID 2>/dev/null; then
      if grep -qE "Out of memory|RESOURCE_EXHAUSTED|XlaRuntimeError" $OUT 2>/dev/null; then STATUS="OOM"; else STATUS="early_exit"; fi
      break
    fi
    sleep 10
  done
fi

WALL=$(($(date +%s) - WALL_START))

# Kill the srun step PID
kill -TERM $PID 2>/dev/null
sleep 4
kill -KILL $PID 2>/dev/null
wait $PID 2>/dev/null
# Kill any leftover shifter/python under this step's job-name
sleep 2

# Parse: r-chunk progress bars
MEAN_S=""
N_SAMPLED=0
N_RCHUNKS_TOTAL=""
PARSE_OUT=$(python3 - "$OUT" <<'PY'
import re, sys
path = sys.argv[1]
times = []
with open(path, errors='replace') as f:
    for line in f:
        m = re.search(r'\[\s*(\d{2}):(\d{2}):(\d{2})\s*\|.*?\]\s*r-chunk\s+(\d+)\s*/\s*(\d+)', line)
        if m:
            hh, mm, ss, idx, total = m.groups()
            t = int(hh)*3600 + int(mm)*60 + int(ss)
            times.append((t, int(idx), int(total)))
if len(times) < 3:
    print(f"NSAMPLED 0")
    print(f"MEAN ")
    print(f"NRCHUNKS_TOTAL " + (str(times[0][2]) if times else ""))
    sys.exit(0)
# skip first sample (compile artifact)
samples = times[1:]
t0, i0 = samples[0][0], samples[0][1]
tN, iN = samples[-1][0], samples[-1][1]
if tN < t0: tN += 86400
mean = (tN - t0) / max(iN - i0, 1)
print(f"NSAMPLED {iN - i0}")
print(f"MEAN {mean:.2f}")
print(f"NRCHUNKS_TOTAL {times[0][2]}")
PY
)
MEAN_S=$(echo "$PARSE_OUT" | awk '/^MEAN/{print $2}')
N_SAMPLED=$(echo "$PARSE_OUT" | awk '/^NSAMPLED/{print $2}')
N_RCHUNKS_TOTAL=$(echo "$PARSE_OUT" | awk '/^NRCHUNKS_TOTAL/{print $2}')

# Parse memory lines
FFTBOX=$(grep -oE "per-iter FFT box [0-9.]+ GB/rank" $OUT | tail -1 | grep -oE "[0-9.]+" | head -1)
PSI=$(grep -oE "ψ\(G-flat\) host cache: [0-9.]+ GB/process resident" $OUT | head -1 | grep -oE "[0-9.]+" | head -1)
PAIR=$(grep -oE "pair[ -][^ ]+ [0-9.]+ GB" $OUT | head -1 | grep -oE "[0-9.]+" | head -1)

NOTES=""
if [ "$STATUS" = "OOM" ]; then
  NOTES=$(grep -oE "(Out of memory|RESOURCE_EXHAUSTED|XlaRuntimeError)[^\"]{0,100}" $OUT | head -1 | tr '\t\n' '  ')
fi
if [ "$STATUS" = "crash" ]; then
  NOTES=$(grep -E "Traceback|FAILED" $OUT | head -1 | tr '\t\n' '  ')
fi

# Append TSV row
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$CFG" "$R" "$B" "$GFLAT" "$STATUS" "${MEAN_S:--}" "${N_SAMPLED:--}" "${FFTBOX:--}" "${PSI:--}" "${PAIR:--}" "$NOTES (n_rchunks_total=${N_RCHUNKS_TOTAL:-?})" >> $TSV

echo "=== SWEEP_B CONFIG ${CFG} r=${R} b=${B} DONE wall=${WALL}s status=${STATUS} mean_per_chunk=${MEAN_S:-?}s ===" | tee -a $PLOG
