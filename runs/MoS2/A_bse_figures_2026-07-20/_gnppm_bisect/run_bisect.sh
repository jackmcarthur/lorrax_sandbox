#!/bin/bash
# Parametrized GN-PPM Sigma_c bisect runner.  Runs ONLY the gw_jax step
# (kin_ion + dipole are static, symlinked pre-built).  Isolates each run in
# work/<LABEL>/ so tmp/ + sigma_freq_debug.dat don't collide.
#   env: JID SRC LABEL CENT [FFB=0] [NNODES=4] [NTASKS=16] [GRES=4]
set -uo pipefail
JID="${JID:?}"; SRC="${SRC:?}"; LABEL="${LABEL:?}"; CENT="${CENT:?}"
FFB="${FFB:-0}"; NNODES="${NNODES:-4}"; NTASKS="${NTASKS:-16}"; GRES="${GRES:-4}"
BR=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/_gnppm_bisect
WD="$BR/work/$LABEL"
rm -rf "$WD"; mkdir -p "$WD/tmp"; cd "$WD"
for f in WFN.h5 kih.dat kin_ion.h5 dipole.h5 Mo.upf S.upf; do ln -sf "$BR/$f" "$f"; done
ln -sf "$BR/$CENT" "$CENT"
# cohsex.in — GN-PPM producer, minimal outputs (no sigma_mnk h5, need only freq_debug)
cat > cohsex.in <<EOF
[cohsex]
compute_mode = gn_ppm
restart = false
centroids_file = $CENT
nval = 26
ncond = 74
nband = 200
sys_dim = 2
x_only = false
do_screened = true
bispinor = false
self_consistent = false
use_ppm_sigma = true
screening_method = minimax
ppm_omega_p = 2.0
ppm_fallback_omega = 2.0
sigma_omega_min_ev = -10.0
sigma_omega_max_ev = 10.0
sigma_omega_step_ev = 0.5
sigma_regularization_ev = 0.25
use_chunked_isdf = true
memory_per_device_gb = 28
sigma_at_dft_energies = true
sigma_freq_debug_output = true
sigma_debug_split_contrib = true
fermi_reference = midgap
wfn_file = WFN.h5
output_file = eqp0.dat
bare_coulomb_cutoff = 30.0
EOF

IMAGE="nvcr.io/nvidia/jax:25.04-py3"
SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site
DEPS=/pscratch/sd/j/jackm/lorrax_sandbox/sources
NVHPC=/global/homes/j/jackm/software/lorrax_nvhpc
PHDF5=/global/homes/j/jackm/software/lorrax_phdf5_cray/stage
SLATE=/global/homes/j/jackm/software/lorrax_slate_cray/stage
SLATE_INSTALL=/global/homes/j/jackm/software/slate/install
MPICH=/opt/udiImage/modules/mpich
DARSHAN=/global/common/software/nersc9/darshan/default/lib
PYPATH="$SRC:$SITE:$DEPS"
LDLIB="$SLATE_INSTALL/lib64:/lorrax_slate/lib:/lorrax_phdf5/lib:$NVHPC/0.7.2_cuda12.9/math_libs/12.9/lib64:$MPICH:$MPICH/dep:$DARSHAN"
SEL=$SRC/ffi/common/cpp/select_gpu.sh
INC=$SRC/ffi/common/cpp/in_container.sh
JAXCACHE="${JAXCACHE:-${SCRATCH:-$HOME}/.jax_cache}"

echo "=== gw_jax start $(date +%s) label=$LABEL src=$SRC cent=$CENT ffb=$FFB nodes=$NNODES ntasks=$NTASKS"
srun --jobid=$JID --overlap --immediate=120 -N "$NNODES" -n "$NTASKS" \
  --gres=gpu:"$GRES" --cpus-per-task=16 --cpu-bind=cores --chdir="$WD" $SEL \
  shifter --image="$IMAGE" --module=gpu,mpich \
    --volume="$NVHPC:/lorrax_nvhpc" \
    --volume="$PHDF5:/lorrax_phdf5" \
    --volume="$SLATE:/lorrax_slate" \
    --env=PYTHONPATH="$PYPATH" \
    --env=HDF5_USE_FILE_LOCKING=FALSE \
    --env=LD_LIBRARY_PATH="$LDLIB" \
    --env=LD_PRELOAD=/lorrax_slate/lib/libmpi_gtl_cuda.so.0 \
    --env=MPICH_GPU_SUPPORT_ENABLED=1 \
    --env=JAX_ENABLE_X64=1 \
    --env=JAX_COMPILATION_CACHE_DIR="$JAXCACHE" \
    --env=XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    --env=MPLBACKEND=Agg \
    --env=OMP_NUM_THREADS=16 \
    --env=LORRAX_FORCE_FULL_BZ="$FFB" \
    $INC \
    python3 -u -m gw.gw_jax -i "$WD/cohsex.in" > gw.out 2>&1
GWRC=$?
echo "=== gw_jax end $(date +%s) rc=$GWRC label=$LABEL"
# metric extraction: Gamma(k=0) VBM(n=25) Re Sigma_c
if [ -f sigma_freq_debug.dat ]; then
  awk 'NF>=14 && $1==0 && $2==25 {printf "METRIC label='"$LABEL"' k=%s n=%s E_dft=%s V_H=%s x_bare=%s ReSigC=%s eqp1=%s\n",$1,$2,$3,$6,$7,$9,$14}' sigma_freq_debug.dat
else
  echo "METRIC label=$LABEL NO_SIGMA_FREQ_DEBUG rc=$GWRC"
fi
grep -i "q-IBZ reduction\|q axis on disk\|full BZ\|write_ibz" gw.out 2>/dev/null | head -2
exit $GWRC
