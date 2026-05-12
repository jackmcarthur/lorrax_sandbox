#!/bin/bash
# LORRAX COHSEX runs (18 total): val_cond-selected centroids × {parabands, pseudo}
# × n_cond ∈ {60, 140, 208} × N_μ ∈ {400, 800, 1200}. Uses use_bgw_vcoul=true
# so the BGW V(q+G) values feed directly into LORRAX's Σ, eliminating the
# MiniBZ head convention difference and isolating the *centroid-selection*
# contribution to the BGW-vs-LORRAX gap.
set -u
module load lorrax_B

ROOT=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/cohsex_valcond_2026-04-18
JID=${SLURM_JOBID:?need SLURM_JOBID}
NSCF=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/qe/nscf/WFN.h5
SI_UPF=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/qe/nscf/Si.upf

# Mini-BZ head values taken from the production run 29_cohsex_pb_v2_nosym_bgwv
# (Si 2×2×2 at 60 Ry). With use_bgw_vcoul=true the BGW value should dominate;
# these are kept for parity with the production cohsex template.
VHEAD=825.937
WHEAD_0FREQ=17.128

PYPATH="/global/homes/j/jackm/software/lorrax_B/src:/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site:/pscratch/sd/j/jackm/lorrax_sandbox/sources"
IMG="nvcr.io/nvidia/jax:25.04-py3"
SH="shifter --module=gpu --image=$IMG --env=PYTHONPATH=$PYPATH --env=HDF5_USE_FILE_LOCKING=FALSE --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 --env=TF_GPU_ALLOCATOR=cuda_malloc_async"

setup_lorrax_dir() {
    local dir=$1 wfn_label=$2 wfn_src=$3 ncond=$4 Nmu=$5
    mkdir -p $dir
    ln -sfn $wfn_src $dir/WFN.h5
    ln -sfn $SI_UPF $dir/Si.upf
    # Symmetry WFN for bgw_vcoul_sym_wfn — must be real deterministic NSCF, not pseudoband.
    ln -sfn $NSCF $dir/sym_wfn.h5
    # BGW vcoul text file — from the matching-ncond BGW run.
    ln -sfn $ROOT/bgw/${wfn_label}_ncond${ncond}/vcoul $dir/bgw_vcoul.txt
    # val_cond centroids file — the N_μ selected points, trimmed from our assay.
    ln -sfn $ROOT/centroids/${wfn_label}_centroids_frac_ncond${ncond}_Nmu${Nmu}.txt \
        $dir/centroids_frac.txt
    local nb=$((8 + ncond))
    cat > $dir/cohsex.in <<EOF
[cohsex]
wfn_file = WFN.h5
nval = 8
ncond = $ncond
nband = $nb
bispinor = false
sys_dim = 3

bare_coulomb_cutoff = 60.0
epsilon_cutoff = 60.0
memory_per_device_gb = 28
centroids_file = centroids_frac.txt
restart = false

screening = cohsex

vhead = $VHEAD
whead_0freq = $WHEAD_0FREQ

use_bgw_vcoul = true
bgw_vcoul_file = bgw_vcoul.txt
bgw_vcoul_sym_wfn = sym_wfn.h5
EOF
}

# Build 18 LORRAX dirs.
for wfn in parabands_4200 pseudo_100sl_216; do
    # Short label for dir names.
    [ "$wfn" = "parabands_4200" ] && wfn_short=parabands || wfn_short=pseudo
    wfn_src=""
    if [ "$wfn" = "parabands_4200" ]; then
        wfn_src=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/10_bgw_parabands_elpa/WFN_pb.h5
    else
        wfn_src=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/03_bgw_pseudobands_100sl/WFN_pseudo.h5
    fi
    for nc in 60 140 208; do
        for Nm in 400 800 1200; do
            dir=$ROOT/lorrax/${wfn_short}_ncond${nc}_Nmu${Nm}
            # Set up run dir — label used for BGW vcoul + centroids lookup is the
            # long label (parabands_4200 / pseudo_100sl_216) matching file names.
            setup_lorrax_dir $dir $wfn $wfn_src $nc $Nm
        done
    done
done

throttle() { while [ $(jobs -r | wc -l) -ge $1 ]; do wait -n; done; }

# ---- STEP 1: dipole + kin_ion (1 GPU per job; 16 parallel) ----
echo "=== STEP 1: dipole + kin_ion ==="
for d in $ROOT/lorrax/*/; do
    [ -f $d/dipole.h5 ] && continue
    (cd $d && \
     srun --jobid=$JID --gres=gpu:1 -N 1 -n 1 --cpus-per-task=8 $SH \
         python3 -u -m psp.get_dipole_mtxels -i cohsex.in > dipole.out 2>&1 && \
     srun --jobid=$JID --gres=gpu:1 -N 1 -n 1 --cpus-per-task=8 $SH \
         python3 -u -m gw.kin_ion_io_chunked -i cohsex.in > kin.out 2>&1) &
    throttle 16
done
wait

# ---- STEP 2: COHSEX (4 GPUs per job; 4 parallel on 16-GPU allocation) ----
echo "=== STEP 2: COHSEX ==="
for d in $ROOT/lorrax/*/; do
    [ -f $d/eqp0.dat ] && continue
    [ -f $d/eqp0_noqsym.dat ] && continue
    (cd $d && \
     srun --jobid=$JID --gres=gpu:4 -N 1 -n 4 --cpus-per-task=8 --exclusive $SH \
         python3 -u -m gw.gw_jax -i cohsex.in > gw.out 2>&1) &
    throttle 4
done
wait

echo "=== all LORRAX runs done ==="
for d in $ROOT/lorrax/*/; do
    if [ -f $d/eqp0_noqsym.dat ] || [ -f $d/eqp0.dat ]; then
        echo "  OK: $(basename $d)"
    else
        echo "  FAILED: $(basename $d)"
    fi
done
