#!/bin/bash
# Drive BSE-on-top for the four centroid runs.
# Step 1: persist W0 in each restart h5 (chi0+W solve, ~3s/run).
# Step 2: build eqp1.dat with COHSEX-gap scissors.
# Step 3: run bse_jax --bse --tda --lanczos with eqp + n-occ.
#
# Usage:  ./run_bse_sweep.sh
# Outputs (per N): tmp/isdf_tensors_<N>.h5 (W0 added), eqp1.dat, bse.out
set -e
SWEEP_DIR=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/B_centroid_sweep_2026-04-27
cd "$SWEEP_DIR"

# Per-N scissors from COHSEX sig_gap_gamma (sweep_results.json)
declare -A SCISSORS=(
  [336]=2.3033
  [480]=2.2946
  [624]=2.2916
  [768]=2.2930
)

for N in 336 480 624 768; do
  RUN=$SWEEP_DIR/N_$N
  echo
  echo "=========================================================="
  echo "  Centroid count N_mu = $N  (scissors=${SCISSORS[$N]} eV)"
  echo "=========================================================="
  cd "$RUN"

  # 1. Persist W0 if not already done.
  W0_READY=$(python3 -c "import h5py; f=h5py.File('tmp/isdf_tensors_${N}.h5','r'); v=bool(f['W0_qmunu'].attrs.get('W0_ready', False)) if 'W0_qmunu' in f else False; f.close(); print(v)")
  if [ "$W0_READY" = "True" ]; then
    echo "  [w0] already persisted, skipping"
  else
    echo "  [w0] computing W0 from V_qmunu + chi0"
    lxrun python3 -u "$SWEEP_DIR/run_w0_persist.py" -i cohsex.in 2>&1 | tee w0_persist.out
  fi

  # 2. Build eqp1.dat with per-N scissors.
  echo "  [eqp1] building scissors-only eqp1.dat"
  python3 "$SWEEP_DIR/make_eqp1.py" --run-dir "$RUN" --scissors-eV "${SCISSORS[$N]}" \
      --n-occ 4 --n-bands 60

  # 3. Run BSE Lanczos.
  echo "  [bse] running BSE --bse --tda --lanczos"
  /usr/bin/time -f "  [bse] wall=%e s" \
      lxrun python3 -u -m bse.bse_jax -i cohsex.in --bse --tda --lanczos \
      --n-val 4 --n-cond 4 --n-eig 12 --max-lanczos-iter 200 \
      --n-occ 4 --eqp eqp1.dat 2>&1 | tee bse.out

done

echo
echo "All four BSE runs complete."
