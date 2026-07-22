#!/bin/bash
# SMOOTH arbitrary-Q exciton bandstructure on the converged MoS2 G0W0.
#
#   usage: JID=<jid> DIR=<rundir> NM=<intervals M->Gamma> NK=<intervals Gamma->K>
#          [MODE=interp|ongrid] [TAG=main] [NT=16] ./run_exciton.sh
#
# The Q path is M -> Gamma -> K, i.e. the owner's TWO SEGMENTS radiating from
# Gamma (Gamma->M reversed, then Gamma->K), with Gamma appearing exactly once.
# ``generate_kpath_from_qe_segments`` reads the interval count off the
# DESTINATION node (segments[i+1]['n']), so NM belongs to the Gamma line and NK
# to the K line; the first node's count is unused.
#   NM=19 NK=19 -> 20 Q on Gamma->M and 20 on Gamma->K, 39 total  (production)
#   NM=6  NK=4  -> the 11 Q of that path that land ON the 12x12 mesh (the
#                  interp-vs-ongrid cross-check; every one of them also appears
#                  in run 08's ongrid Gamma-M-K-Gamma path)
#
# fH window nb = 28 (nval = ncond = 14, bands [12, 40)) — the measured
# htransform capacity ceiling at n_mu = 2412 (rank(psi_mu)/nk = 31.74, so
# nb <= 31; 28 is the largest passing sweep point and leaves 6 valence + 6
# conduction guard bands around the 8v8c BSE window, absolute bands 18..34).
#
# MODE=interp  arbitrary Q through bse.vq_interp (b26p): needs the FULL-BZ
#              zeta_q.h5 built by run_zeta_fullbz.sh (tmp -> ../tmp_fullbz).
# MODE=ongrid  exact stored V_qmunu[wrap(-Q)] tiles; only valid on-grid.
#
# LORRAX_SKIP_VQ_GATES=1: vq_interp's reference gate/null batteries materialise
# REPLICATED (nq, n_mu, ngkmax) and (nq, n_mu, n_mu) chunks (15.9 GB and 11.2 GB
# per 48-q chunk at n_mu = 2412) and OOM.  The zeta-vs-production consistency
# they would check is done once, properly chunked, by run_zeta_gate.sh, and the
# driver's own on-grid htransform ENERGY gate still runs every time.
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/09_mos2_exciton_smooth_2026-07-21
DIR="${DIR:?set DIR}"
RD=$R/$DIR
SH=$R/run_shifter.sh
NM="${NM:?set NM}"; NK="${NK:?set NK}"
MODE="${MODE:-interp}"; TAG="${TAG:-main}"
NVAL=14; NCOND=14
NB=$((NVAL + NCOND)); NBAND=$((26 + NCOND))
NT="${NT:-16}"; NN=4; PX=4; PY=4
NEIG="${NEIG:-8}"

cat > "$RD/exciton.in" <<EOF
[cohsex]
# SMOOTH arbitrary-Q 8v8c exciton bandstructure on the CONVERGED MoS2 G0W0
# (runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21/00b_lorrax_gw_2400c_ranktrunc:
#  80 Ry, 12x12x1 = 144 k, 326 screening bands, 2412 band-range + D3h centroids,
#  GN-PPM, rank-truncated charge CCT, xi-floor; eqp1 direct gap 2.6356 eV @ K).
#
# htransform fH window: ${NB} bands = [$((26-NVAL)), $((26+NCOND))), i.e. ${NVAL}v + ${NCOND}c,
# CENTRED ON THE GAP.  nband = ${NBAND} is Meta.b_id_4_user (ABSOLUTE) and must
# cover the window top, else load_centroids_band_chunked zeroes the upper bands.
# Q path: M -> Gamma -> K, ${NM} + ${NK} intervals.
centroids_file = centroids.txt
nval = ${NVAL}
ncond = ${NCOND}
nband = ${NBAND}
sys_dim = 2
bispinor = false
wfn_file = WFN.h5

K_POINTS {crystal_b}
3
  0.0000000000 0.5000000000 0.0000000000 1     # M
  0.0000000000 0.0000000000 0.0000000000 ${NM}    # Gamma
  0.3333333333 0.3333333333 0.0000000000 ${NK}    # K
EOF

# GPU zombies masquerade as a phdf5 "negative offset" crash — reap first.
srun --jobid=$JID --overlap --immediate=60 -N4 -n4 --gres=gpu:1 \
     bash -c 'pkill -9 -u $USER -f exciton_bands; pkill -9 -u $USER -f gw_jax; exit 0' >/dev/null 2>&1
sleep 3

echo "=== exciton start $(date +%T)  dir=${DIR}  vq=${MODE}  NM=${NM} NK=${NK}  ${NT} GPU"
JID=$JID NNODES=$NN NTASKS=$NT GRES=4 \
  EXTRA_ENV="--env=LORRAX_SKIP_VQ_GATES=1" "$SH" "$RD" \
  python3 -u -m bse.exciton_bands -i exciton.in \
    --n-val 8 --n-cond 8 --n-eig $NEIG --block-size 8 --max-iter 40 \
    --vq-mode $MODE --eqp eqp1.dat \
    --px $PX --py $PY --skip-rerun-check \
    --out-prefix mos2_exciton_${TAG} > "$RD/run_exciton_${TAG}.log" 2>&1
rc=$?
echo "=== exciton end $(date +%T) rc=$rc"
grep -aE "\[gate\]|\[eqp\]|Q path|exchange:|Wrote|Traceback|Error|OOM|RESOURCE_EXHAUSTED" \
     "$RD/run_exciton_${TAG}.log" | tail -25
exit $rc
