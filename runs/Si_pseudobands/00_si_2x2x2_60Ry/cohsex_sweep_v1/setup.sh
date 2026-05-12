#!/bin/bash
# Set up 18 COHSEX+BGW vcoul dirs, one per PB WFN
set -u
SWEEP_ROOT="/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry"
PB_ROOT="$SWEEP_ROOT/pb_sweep"
COHSEX_ROOT="$SWEEP_ROOT/cohsex_sweep"
BGW_VCOUL="$SWEEP_ROOT/17_bgw_explicit_60Ry/vcoul"
SYM_WFN="$SWEEP_ROOT/qe/nscf/WFN.h5"
mkdir -p "$COHSEX_ROOT"
for v in 1 2; do for w in 50 100 150; do for s in 0 1 2; do
  d="$COHSEX_ROOT/v${v}_${w}win_s${s}"
  mkdir -p "$d"
  ln -sf "$PB_ROOT/v${v}_${w}win_s${s}/WFN_pseudobands.h5" "$d/WFN.h5"
  ln -sf "$BGW_VCOUL" "$d/bgw_vcoul.txt"
  ln -sf "$SYM_WFN" "$d/sym_wfn.h5"
  # Read mnband from PB WFN
  mb=$(python3 -c "import h5py; f=h5py.File('$d/WFN.h5','r'); print(int(f['mf_header/kpoints/mnband'][()]))")
  cat > "$d/cohsex.in" <<EOF
[cohsex]
wfn_file = WFN.h5
nval = 8
ncond = 8
nband = $mb
bispinor = false
sys_dim = 3

bare_coulomb_cutoff = 60.0
epsilon_cutoff = 60.0
memory_per_device_gb = 28
centroids_file = centroids_frac_2000.txt
restart = false

screening = cohsex

vhead = 825.937
whead_0freq = 17.128

use_bgw_vcoul = true
bgw_vcoul_file = bgw_vcoul.txt
bgw_vcoul_sym_wfn = sym_wfn.h5
EOF
done; done; done
echo "18 cohsex dirs prepared"
ls "$COHSEX_ROOT"
