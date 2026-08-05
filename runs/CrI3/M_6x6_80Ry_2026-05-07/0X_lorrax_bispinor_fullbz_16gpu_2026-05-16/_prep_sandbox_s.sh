#!/bin/bash -l
# Prep a sandbox subdir for Agent S: symlinks to inputs + a copy of cohsex.in
set -u
SB="${1:?sandbox dir name (e.g. A1_sb)}"
COHSEX_IN="${2:?absolute path to cohsex_*.in template}"

PARENT="/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16"

mkdir -p "$PARENT/$SB"
mkdir -p "$PARENT/$SB/tmp"
cd "$PARENT/$SB"

for f in WFN.h5 kih.dat kin_ion.h5 vxc.dat centroids_frac_1508.txt centroids_frac_1504_current.txt; do
  if [ ! -e "$f" ]; then ln -sf "../$f" "$f"; fi
done

cp "$COHSEX_IN" "cohsex.in"
echo "Sandbox $PARENT/$SB ready (cohsex.in copied from $COHSEX_IN)"
ls -la
