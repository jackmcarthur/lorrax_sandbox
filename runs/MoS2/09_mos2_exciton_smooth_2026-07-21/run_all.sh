#!/bin/bash
# Gate the regenerated full-BZ zeta, then run BOTH exciton jobs.
# The gate is a hard precondition: a zeta that does not rebuild the production
# V_qmunu tiles cannot produce a trustworthy interpolated V_Q, so the exciton
# runs do not start unless it passes.
#
#   usage: JID=<jid> ./run_all.sh
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/09_mos2_exciton_smooth_2026-07-21

JID=$JID "$R/run_zeta_gate.sh" || { echo "ZETA GATE FAILED — stopping"; exit 1; }

# (1) the 11 Q of the M->Gamma->K path that land on the 12x12 mesh, run through
#     the INTERP exchange.  Every one of them also appears in run 08's ongrid
#     Gamma-M-K-Gamma path, so this is the apples-to-apples interp-vs-exact table.
JID=$JID DIR=01_lorrax_exciton_interp_ongrid_check NM=6 NK=4 MODE=interp \
  TAG=interp "$R/run_exciton.sh" || { echo "RUN 01 FAILED"; exit 1; }

# (2) the production path: 20 Q on Gamma->M + 20 on Gamma->K, arbitrary Q.
JID=$JID DIR=02_lorrax_exciton_smooth_40q NM=19 NK=19 MODE=interp \
  TAG=smooth "$R/run_exciton.sh" || { echo "RUN 02 FAILED"; exit 1; }

echo "=== all done $(date +%T)"
