#!/bin/bash
# STEP 1 — the CHEAP eps-leg gate.  Runs ONLY the htransform leg (no BSE), so
# it costs ~30-90 s per window instead of ~35 min, and it is the thing that
# decides whether the exciton solve is worth launching at all.
#
# WHAT IT GATES.  For every sweep window it recovers the interpolated
# conduction energies eps_c(k+Q) at the PRODUCTION off-grid Q and reports
#   * on-grid  max |eps_ht - eps_stored|                            (meV)
#   * off-grid max/mean |2nd difference of eps_c along Q|, per leg  (meV)
# A real exciton band needs the 2nd differences at the few-meV level; the
# 2026-07-21 failure had 2905 meV (M-Gamma) / 2391 meV (Gamma-K).
#
# THE TWO REGRESSIONS THIS SWEEP TESTS, one at a time:
#   1. ANCHORING.  htransform needs a band subspace CONTIGUOUS FROM E_min
#      upward.  n_occ = 26 for MoS2 (spinor), so nval = 26 puts the fH window
#      at absolute bands [0, 26+ncond) — semicore included.  The failing run
#      used nval = 14, i.e. bands [12, 40): the deep states are dropped and the
#      lower window boundary cuts through the valence manifold.
#   2. --a-band.  The f-transform width a = 4*bandwidth(a_band).  Left at its
#      default it comes from the TOP of the fH window — a dispersive guard band
#      — and the flag's own help warns that collapses off-grid eps_c by eV.
#      'auto' = top of the BSE conduction selection (nval + n_cond - 1), which
#      is exactly the known-good run's --a-band 33 at nval = 26 / n_cond = 8.
#
#   usage: JID=<jid> [WINDOWS="26,14"] [NCOND=8] [ABAND=auto] \
#          [KSTRIDE=24] [TAG=gate] ./run_eps_gate.sh
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/10_mos2_exciton_anchored_2026-07-21
REP=/pscratch/sd/j/jackm/lorrax_sandbox/reports/bse_exciton_smooth_2026-07-21
RD=$R/00_eps_gate
SH=$R/run_shifter.sh
WINDOWS="${WINDOWS:-26,14}"
NCOND="${NCOND:-8}"
ABAND="${ABAND:-auto}"
KSTRIDE="${KSTRIDE:-24}"
TAG="${TAG:-gate}"

# GPU zombies masquerade as a phdf5 "negative offset" crash -- reap first.
srun --jobid=$JID --overlap --immediate=60 -N4 -n4 --gres=gpu:1 \
     bash -c 'pkill -9 -u $USER -f "[b]se.exciton_bands"; pkill -9 -u $USER -f "[e]ps_window_sweep"; exit 0' >/dev/null 2>&1
sleep 3

echo "=== eps gate start $(date +%T)  windows=${WINDOWS}  n_cond=${NCOND}  a-band=${ABAND}"
JID=$JID NNODES=4 NTASKS=16 GRES=4 \
  EXTRA_ENV="--env=LORRAX_SKIP_VQ_GATES=1 --env=LORRAX_GALERKIN_CHUNK_GIB=${GCHUNK:-6}" "$SH" "$RD" \
  python3 -u $REP/eps_window_sweep.py -i exciton.in --eqp eqp1.dat \
    --px 4 --py 4 --windows "$WINDOWS" \
    --n-cond "$NCOND" --a-band "$ABAND" \
    --k-stride "$KSTRIDE" --out "$RD/${TAG}" > "$RD/run_${TAG}.log" 2>&1
rc=$?
echo "=== eps gate end $(date +%T) rc=$rc"
grep -aE "^\[window|^    (M-Gamma|Gamma-K)|SVD of|Streaming Galerkin|f-transform|^Wrote|Traceback|Error" \
     "$RD/run_${TAG}.log" | tail -40
exit $rc
