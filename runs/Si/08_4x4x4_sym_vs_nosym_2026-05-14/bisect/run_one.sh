#!/bin/bash
# run_one.sh <sha> <label>
#
# Re-runs LORRAX cohsex on run_sym/ for one commit on lorrax_B, parses
# Σ_X, compares to run_nosym/ (which stays on HEAD; bug fires only in the
# sym-unfold path so the nosym reference is independent of commit).
#
# Must be invoked as `bash -lc "...; run_one.sh <sha> <label>"` with
# `module load lorrax_B lorrax_agent` and `SLURM_JOBID` already set so
# the `lxrun` function is in scope.
#
# Writes:
#   bisect/<label>/sigma_freq_debug.dat
#   bisect/<label>/gw.out
#   bisect/<label>/compare.log
#   bisect/<label>/summary.txt
set -e

SHA="$1"
LABEL="$2"

if [ -z "$SHA" ] || [ -z "$LABEL" ]; then
    echo "usage: run_one.sh <sha> <label>" >&2
    exit 1
fi

if ! declare -f lxrun > /dev/null; then
    echo "ERROR: lxrun is not defined in this shell. Source lorrax_agent first." >&2
    exit 2
fi

BASE="/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14"
SRC="/global/homes/j/jackm/software/lorrax_B"
RUN_SYM="$BASE/run_sym"
OUTDIR="$BASE/bisect/$LABEL"
mkdir -p "$OUTDIR"

echo "[bisect] === $LABEL  $SHA ==="
echo "[bisect] checking out lorrax_B @ $SHA"
git -C "$SRC" checkout -q "$SHA"
git -C "$SRC" log -1 --oneline

# Clean stale bytecode.
find "$SRC/src" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

cd "$RUN_SYM"
echo "[bisect] launching lxrun gw_jax"
rm -rf tmp 2>/dev/null || true
mkdir -p tmp
lxrun python3 -u -m gw.gw_jax -i "$RUN_SYM/cohsex.in" > "$OUTDIR/gw.out" 2>&1 || true

# Copy outputs.
cp -f "$RUN_SYM/sigma_freq_debug.dat" "$OUTDIR/" 2>/dev/null || true

cd "$BASE"
python3 compare_sigma_x.py > "$OUTDIR/compare.log" 2>&1 || true

MAX_DTOTAL=$(grep "max |ΔΣ_X (total)|" "$OUTDIR/compare.log" 2>/dev/null | awk '{print $5}')
MAX_DXBARE=$(grep "max |Δx_bare|" "$OUTDIR/compare.log" 2>/dev/null | awk '{print $4}')
VERDICT=$(grep "Verdict:" "$OUTDIR/compare.log" 2>/dev/null | awk '{print $2}')
[ -z "$MAX_DTOTAL" ] && MAX_DTOTAL="ERR"
[ -z "$MAX_DXBARE" ] && MAX_DXBARE="ERR"
[ -z "$VERDICT" ] && VERDICT="ERR"
echo "[bisect] $LABEL  $SHA  max|ΔΣ_X|=${MAX_DTOTAL}meV max|Δx_bare|=${MAX_DXBARE}meV  $VERDICT" \
    | tee -a "$BASE/bisect/SUMMARY.txt"
echo "$LABEL $SHA max_dtotal_meV=${MAX_DTOTAL} max_dxbare_meV=${MAX_DXBARE} verdict=${VERDICT}" \
    > "$OUTDIR/summary.txt"
