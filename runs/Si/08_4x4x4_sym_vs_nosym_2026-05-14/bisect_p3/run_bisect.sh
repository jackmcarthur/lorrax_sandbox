#!/bin/bash
# Bisect runner — iterate through a list of commits and run LORRAX-on-sym for each.
# Must be invoked with `module load lorrax_B lorrax_agent` and SLURM_JOBID set.
set -e

BASE="/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14"
SRC="/global/homes/j/jackm/software/lorrax_B"
RUN_SYM="$BASE/run_sym"
OUTROOT="$BASE/bisect_p3"
SUMMARY="$OUTROOT/SUMMARY.txt"
mkdir -p "$OUTROOT"

if ! command -v lxrun > /dev/null && ! declare -f lxrun > /dev/null; then
  echo "ERROR: lxrun not defined" >&2
  exit 2
fi

run_one() {
  local SHA="$1"
  local LABEL="$2"
  local OUTDIR="$OUTROOT/$LABEL"
  mkdir -p "$OUTDIR"

  echo "[bisect] === $LABEL  $SHA ==="
  git -C "$SRC" checkout -q "$SHA"
  echo "[bisect]    $(git -C "$SRC" log -1 --oneline)"

  # Clean stale bytecode and tmp
  find "$SRC/src" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
  rm -rf "$RUN_SYM/tmp" 2>/dev/null || true
  mkdir -p "$RUN_SYM/tmp"

  cd "$RUN_SYM"
  lxrun python3 -u -m gw.gw_jax -i "$RUN_SYM/cohsex.in" > "$OUTDIR/gw.out" 2>&1 || true

  cp -f "$RUN_SYM/sigma_freq_debug.dat" "$OUTDIR/" 2>/dev/null || true
  cp -f "$RUN_SYM/eqp0.dat"             "$OUTDIR/" 2>/dev/null || true

  cd "$BASE"
  python3 compare_sigma_x.py > "$OUTDIR/compare.log" 2>&1 || true

  local MAX_DTOTAL=$(grep "max |ΔΣ_X (total)|" "$OUTDIR/compare.log" | awk '{print $5}')
  local MAX_DXBARE=$(grep "max |Δx_bare|" "$OUTDIR/compare.log" | awk '{print $4}')
  local VERDICT=$(grep "Verdict:" "$OUTDIR/compare.log" | awk '{print $2}')
  [ -z "$MAX_DTOTAL" ] && MAX_DTOTAL="ERR"
  [ -z "$VERDICT" ] && VERDICT="ERR"

  # Also grab the n=0 sym Σ_X from eqp0.dat (or sigma_freq_debug) for sanity
  local SYM_VBM=$(awk '/k-point 0:/{f=1; next} f && /n=0 / {print $3; exit}' "$RUN_SYM/eqp0.dat" 2>/dev/null)
  [ -z "$SYM_VBM" ] && SYM_VBM="-"

  echo "[bisect] $LABEL  $SHA  max|dΣ|=${MAX_DTOTAL}meV  max|dXb|=${MAX_DXBARE}meV  sym_VBM=${SYM_VBM}eV  $VERDICT" | tee -a "$SUMMARY"
}

# Reset header
echo "# bisect_p3 — Si 4x4x4 sym ψ-unfold Σ_X regression" > "$SUMMARY"
echo "# label  sha   max|ΔΣ_X|   max|Δx_bare|   sym_VBM   verdict" >> "$SUMMARY"

# Commit list — earliest first
for spec in \
  "07aa13e c02_orbit_kmeans_apr30" \
  "fe5e3e8 c03_apr29_eod" \
  "9e644e9 c04_pre_phase2"
do
  run_one $spec
done

# Reset to HEAD
git -C "$SRC" checkout -q 69ab42c
echo "[bisect] === all done; reset to HEAD 69ab42c ===" | tee -a "$SUMMARY"
