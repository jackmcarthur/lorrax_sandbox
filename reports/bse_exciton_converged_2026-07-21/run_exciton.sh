#!/bin/bash
# TASK 2 — 8v8c exciton bandstructure on the CONVERGED 12x12 / 80 Ry G0W0.
#
#   usage: JID=<jid> [NVAL=16] [NCOND=16] [MODE=ongrid|interp] [NQSEG=6,2,4,1]
#          [TAG=main] ./run_exciton.sh
#
# NVAL/NCOND are the HTRANSFORM fH window (nval+ncond bands, centred on the
# gap at [26-NVAL, 26+NCOND)); the BSE itself is always 8v8c.  ``nband`` in
# cohsex.in is Meta.b_id_4_user, an ABSOLUTE band index — it must cover the
# window's top (26+NCOND) or load_centroids_band_chunked zeroes the upper
# bands and the SVD returns rank 0.
#
# MODE=ongrid  every Q lands on the 12x12 BSE grid, so the EXACT stored
#              V_qmunu[wrap(-Q)] tile is used — no exchange interpolation and
#              no full-BZ zeta_q.h5 (the production restart stores zeta on the
#              IBZ, 74 of 144 q, because the D3h centroids close under orbit).
# MODE=interp  dense arbitrary-Q path through bse.vq_interp; needs a full-BZ
#              zeta_q.h5 (see run_zeta_fullbz.sh).
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/08_mos2_exciton_converged_2026-07-21
RD=$R/01_lorrax_exciton_8v8c_gw
REP=/pscratch/sd/j/jackm/lorrax_sandbox/reports/bse_exciton_converged_2026-07-21
SH=$R/run_shifter.sh
NVAL="${NVAL:-16}"; NCOND="${NCOND:-16}"
MODE="${MODE:-ongrid}"; TAG="${TAG:-main}"
NB=$((NVAL + NCOND)); NBAND=$((26 + NCOND))
NT="${NT:-16}"; NN=4; PX=4; PY=4
if [ "$NT" = 8 ]; then PX=2; PY=4; fi
NEIG="${NEIG:-8}"
# a_band: window-relative index whose bandwidth sets the f-transform width.
# EMPTY by default — the flag is omitted, which is exactly what the window sweep
# (00_htransform_window_sweep) validated.  Setting it here would invalidate the
# gate numbers that chose the window.
ABAND="${ABAND:-}"
ABAND_ARG=""; [ -n "$ABAND" ] && ABAND_ARG="--a-band $ABAND"

if [ "$MODE" = ongrid ]; then
  # Gamma-M-K-Gamma sampled ON the 12x12 grid: 6 + 2 + 4 + 1 = 13 Q.
  # Gamma->M   steps (0, 1/12, 0)      6 intervals
  # M->K       steps (2/12, -1/12, 0)  2 intervals
  # K->Gamma   steps (-1/12,-1/12, 0)  4 intervals
  SEG="6 2 4"; VQ=ongrid
else
  SEG="${SEG:-20 12 23}"; VQ=interp
fi
set -- $SEG

cat > "$RD/exciton.in" <<EOF
[cohsex]
# 8v8c exciton bandstructure on the CONVERGED MoS2 G0W0
# (runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21/00b_lorrax_gw_2400c_ranktrunc:
#  80 Ry, 12x12x1 = 144 k, 326 screening bands, 2412 band-range + D3h centroids,
#  GN-PPM, rank-truncated charge CCT, xi-floor; eqp1 direct gap 2.6356 eV @ K).
#
# htransform fH window: ${NB} bands = [$((26-NVAL)), $((26+NCOND))), i.e. ${NVAL}v + ${NCOND}c
# CENTRED ON THE GAP.  Capacity rule nspinor*n_mu > nk*nb: 2*2412 = 4824 >
# 144*${NB} = $((144*NB)).  The BSE window (8v8c, absolute bands 18..34) sits
# strictly interior with $((NVAL-8)) valence and $((NCOND-8)) conduction guards.
# nband = ${NBAND} is Meta.b_id_4_user (ABSOLUTE), covering the window top.
centroids_file = centroids.txt
nval = ${NVAL}
ncond = ${NCOND}
nband = ${NBAND}
sys_dim = 2
bispinor = false
wfn_file = WFN.h5

K_POINTS {crystal_b}
4
  0.0000000000 0.0000000000 0.0000000000 1     # Gamma
  0.0000000000 0.5000000000 0.0000000000 $1    # M
  0.3333333333 0.3333333333 0.0000000000 $2    # K
  0.0000000000 0.0000000000 0.0000000000 $3    # Gamma
EOF
# htransform.generate_kpath_from_qe_segments reads the interval count off the
# DESTINATION node (``segments[i+1]['n']``), so the first line's count is unused.

# GPU zombies masquerade as a phdf5 "negative offset" crash — reap first.
srun --jobid=$JID --overlap --immediate=60 -N4 -n4 --gres=gpu:1 \
     bash -c 'pkill -9 -u $USER -f exciton_bands; pkill -9 -u $USER -f window_sweep; exit 0' >/dev/null 2>&1
sleep 3

echo "=== exciton start $(date +%T)  fH window ${NB} (${NVAL}v+${NCOND}c)  vq=${VQ}  ${NT} GPU (${PX}x${PY})  a_band=${ABAND:-default}"
JID=$JID NNODES=$NN NTASKS=$NT GRES=4 "$SH" "$RD" \
  python3 -u -m bse.exciton_bands -i exciton.in \
    --n-val 8 --n-cond 8 --n-eig $NEIG --block-size 8 --max-iter 40 \
    --vq-mode $VQ --eqp eqp1.dat $ABAND_ARG \
    --px $PX --py $PY --skip-rerun-check \
    --out-prefix mos2_exciton_${TAG} > "$RD/run_exciton_${TAG}.log" 2>&1
rc=$?
echo "=== exciton end $(date +%T) rc=$rc"
grep -aE "\[gate\]|\[eqp\]|Q path|full-band htransform|exchange:|Wrote|Traceback|Error|Assertion" \
     "$RD/run_exciton_${TAG}.log" | tail -20
exit $rc
