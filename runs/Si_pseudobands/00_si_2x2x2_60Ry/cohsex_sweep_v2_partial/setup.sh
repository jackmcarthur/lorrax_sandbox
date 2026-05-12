#!/bin/bash
# Set up 18 cohsex dirs from pb_sweep
set -u
ROOT="/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry"
BGW_VCOUL="$ROOT/17_bgw_explicit_60Ry/vcoul"
SYM_WFN="$ROOT/qe/nscf/WFN.h5"
mkdir -p "$ROOT/cohsex_sweep"
for v in 1 2; do for w in 50 100 150; do for s in 0 1 2; do
  d="$ROOT/cohsex_sweep/v${v}_${w}win_s${s}"
  mkdir -p "$d"
  ln -sf "$ROOT/pb_sweep/v${v}_${w}win_s${s}/WFN_pseudobands.h5" "$d/WFN.h5"
  ln -sf "$BGW_VCOUL" "$d/bgw_vcoul.txt"
  ln -sf "$SYM_WFN" "$d/sym_wfn.h5"
  ln -sf "$ROOT/qe/nscf/Si.upf" "$d/Si.upf"
  ln -sf "$ROOT/qe/nscf/silicon.save" "$d/silicon.save"
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
