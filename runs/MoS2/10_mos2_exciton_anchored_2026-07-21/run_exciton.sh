#!/bin/bash
# STEP 2 — the arbitrary-Q 8v8c exciton bandstructure on the CONVERGED MoS2
# G0W0, with the htransform basis ANCHORED AT E_min and its α-basis rank
# CAPPED.
#
#   usage: JID=<jid> DIR=<rundir> NM=<intervals Γ->M> NK=<intervals Γ->K> \
#          [TAG=main] [NEIG=8] [EXTRAQ="x,y,z;..."] ./run_exciton.sh
#
# Q path: M -> Gamma -> K, i.e. the owner's TWO SEGMENTS radiating from Gamma
# with Gamma appearing exactly once.  ``generate_kpath_from_qe_segments`` reads
# the interval count off the DESTINATION node, so NM belongs to the Gamma line
# and NK to the K line; the first node's count is unused.
#   NM=19 NK=19 -> 20 Q on Gamma->M and 20 on Gamma->K, 39 total  (production)
#   NM=3  NK=3  -> 7 Q, the cheap probe that reads the driver's OWN on-grid
#                  htransform gate before committing to the full solve
#
# WINDOW.  nval = 26 = n_occ, so the fH window is absolute bands [0, 40):
# CONTIGUOUS FROM E_min, semicore included.  htransform interpolates a band
# SUBSPACE, and a subspace is only smooth in Q if it is closed from the bottom
# — a window that starts at band 12 (the 2026-07-21 failure) cuts the valence
# manifold and the projector is discontinuous wherever the cut crosses a band.
# 8 conduction guards above the 8v8c BSE selection (bands 26..33).
#
# WHY IT WORKS NOW.  Both of the above were already true of the 2026-07-20
# known-good run.  What actually blocked this configuration at the converged
# reference was a REGRESSION in the Galerkin build (streaming_galerkin_solve
# summed G over BAND chunks, dropping the cross-chunk terms); see the report.
# With that fixed the anchored window measures 7.4 meV on-grid and 186/170 meV
# off-grid |2nd diff| of eps_c, versus 311 meV / 4784 meV before.
#
# --a-band 33 = the top of the 8v8c BSE conduction selection.  Left at its
# default, ``a`` comes from the top of the fH window — a dispersive guard band
# — which the flag's own help warns collapses off-grid eps_c by eV.
#
# LORRAX_SKIP_VQ_GATES=1: vq_interp's reference gate batteries materialise
# REPLICATED (nq, n_mu, ngkmax) and (nq, n_mu, n_mu) chunks (15.9 / 11.2 GB per
# 48-q chunk at n_mu = 2412) and OOM.  The zeta-vs-production consistency they
# check was done once, properly chunked, by run 09's run_zeta_gate.sh, and the
# driver's own on-grid htransform ENERGY gate still runs every time.
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/10_mos2_exciton_anchored_2026-07-21
DIR="${DIR:?set DIR}"
RD=$R/$DIR
SH=$R/run_shifter.sh
NM="${NM:?set NM}"; NK="${NK:?set NK}"
TAG="${TAG:-main}"; NEIG="${NEIG:-8}"
NVAL=26; NCOND=14
NB=$((NVAL + NCOND)); NBAND=$((26 + NCOND))
NT=16; NN=4; PX=4; PY=4
EXTRAQ="${EXTRAQ:-}"

mkdir -p "$RD"
for f in WFN.h5 eqp1.dat centroids.txt; do
  [ -e "$RD/$f" ] || ln -sfn "$R/00_eps_gate/$f" "$RD/$f"
done
[ -e "$RD/tmp" ] || ln -sfn ../tmp_fullbz "$RD/tmp"

cat > "$RD/exciton.in" <<EOF
[cohsex]
# ANCHORED arbitrary-Q 8v8c exciton bandstructure on the CONVERGED MoS2 G0W0
# (runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21/00b_lorrax_gw_2400c_ranktrunc:
#  80 Ry, 12x12x1 = 144 k, 326 screening bands, 2412 band-range + D3h centroids,
#  GN-PPM, rank-truncated charge CCT, xi-floor; eqp1 direct gap 2.6356 eV @ K).
#
# htransform fH window: ${NB} bands = absolute [0, ${NB}), i.e. ${NVAL}v + ${NCOND}c.
# nval = ${NVAL} = n_occ, so the window is CONTIGUOUS FROM E_min with the semicore
# states included — the htransform interpolates a band SUBSPACE and that
# subspace is only smooth in Q if it is closed from the bottom.
# nband = ${NBAND} is Meta.b_id_4_user (ABSOLUTE) and must cover the window top,
# else load_centroids_band_chunked zeroes the upper bands.
# BSE window (CLI): --n-val 8 --n-cond 8 -> conduction bands [26, 34), interior
# to the fH window with 6 guards above.
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

# GPU zombies masquerade as a phdf5 "negative offset" crash -- reap first.
srun --jobid=$JID --overlap --immediate=60 -N4 -n4 --gres=gpu:1 \
     bash -c 'pkill -9 -u $USER -f "[b]se.exciton_bands"; pkill -9 -u $USER -f "[e]ps_window_sweep"; exit 0' >/dev/null 2>&1
sleep 3

XQ=""; [ -n "$EXTRAQ" ] && XQ="--extra-q $EXTRAQ"
echo "=== exciton start $(date +%T)  dir=${DIR}  NM=${NM} NK=${NK}  ${NT} GPU"
JID=$JID NNODES=$NN NTASKS=$NT GRES=4 \
  EXTRA_ENV="--env=LORRAX_SKIP_VQ_GATES=1 --env=LORRAX_GALERKIN_CHUNK_GIB=${GCHUNK:-6}" "$SH" "$RD" \
  python3 -u -m bse.exciton_bands -i exciton.in \
    --n-val 8 --n-cond 8 --n-eig $NEIG --block-size 8 --max-iter 40 \
    --vq-mode interp --eqp eqp1.dat --a-band 33 \
    $XQ \
    --px $PX --py $PY --skip-rerun-check \
    --out-prefix mos2_exciton_${TAG} > "$RD/run_exciton_${TAG}.log" 2>&1
rc=$?
echo "=== exciton end $(date +%T) rc=$rc"
grep -aE "SVD of|\[gate\]|\[eqp\]|full-band htransform|f-transform|Q path|exchange:|Wrote|Traceback|Error|OOM|RESOURCE_EXHAUSTED|AssertionError" \
     "$RD/run_exciton_${TAG}.log" | sort -u | tail -25
exit $rc
