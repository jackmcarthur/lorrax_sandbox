# ===========================================================================
# env_common.sh -- OUTER (batch-shell) env shared by every b600/P=64 leg.
# Source it from the sbatch body before building the inner script.
#
# Every path here is frozen: LORRAX_ROOT is the copied config tree and
# LORRAX_SRC_FALLBACK the copied src tree, both taken from
# /work2/08271/jackmc/frontera/lorrax @ 1fc2759.  No job reads the live
# checkout.
# ===========================================================================
export R=/scratch2/08271/jackmc/b600_p64
export DECK=/scratch2/08271/jackmc/mos2_4x4_test          # read-only deck
export SRCREV=1fc2759

export LORRAX_ROOT=$R/lorrax_frozen                        # frozen config/ + src/
export LORRAX_SRC_FALLBACK=$R/lorrax_frozen/src
export LORRAX_BUNDLE=$R/bundle_1fc2759/lorrax_cpu_bundle.tar
export LORRAX_SIF=/scratch2/08271/jackmc/lorrax_setup/py312.sif
export LORRAX_FFI_HOST_SO=/work2/08271/jackmc/frontera/lorrax_ffi_unified/build_host_ONE/liblorrax_ffi_host.so
export LORRAX_MPIWRAPPER_SO=/work2/08271/jackmc/frontera/lorrax_mpiwrapper/install/lib64/libmpiwrapper.so
export LORRAX_SLATE_HOST_LIB=/work2/08271/jackmc/frontera/slate_builds/cpu/install/lib64
export LORRAX_IMPI_ROOT=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64
export LORRAX_HDF5_ROOT=/home1/apps/intel19/impi19_0/phdf5/1.14.6
export LORRAX_MKL_LIB=/opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin
export LORRAX_ICC_RUNTIME=/opt/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin
export LORRAX_PMI2_LIB=/work2/08271/jackmc/frontera/host_pmi/libpmi2.so.0

# W Dyson residual telemetry (AQ campaign used it; cheap, and it is the only
# in-run correctness observable for the screened interaction).
export LORRAX_W_RESIDUAL_CHECK=1

# RDMA/UCX userspace staging binds.  NEVER bind anything under /dev.
export HL=/usr/lib64:/hostlibs:ro,/usr/lib64/libibverbs,/etc/libibverbs.d
export BIND=/home1,/work2,/scratch1,/scratch2,/opt/intel,/tmp,$HL

[ -f "$LORRAX_MPIWRAPPER_SO" ] || { echo "REFUSING: no MPIwrapper at $LORRAX_MPIWRAPPER_SO"; exit 2; }
[ -f "$LORRAX_FFI_HOST_SO" ]   || { echo "REFUSING: no FFI host .so at $LORRAX_FFI_HOST_SO"; exit 2; }
[ -f "$LORRAX_SIF" ]           || { echo "REFUSING: no container at $LORRAX_SIF"; exit 2; }
[ -f "$LORRAX_BUNDLE" ]        || { echo "REFUSING: no frozen bundle at $LORRAX_BUNDLE"; exit 2; }

# per-LAUNCH coordinator port (AP.6)
_coord_host=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
export JAX_COORDINATOR_ADDRESS="$_coord_host:$((13000 + SLURM_JOB_ID % 2000))"

echo "=== leg env: src@$SRCREV  P=$SLURM_NTASKS  N=$SLURM_NNODES  coord=$JAX_COORDINATOR_ADDRESS"
echo "    bundle  $LORRAX_BUNDLE"
echo "    ffi so  $(sha256sum $LORRAX_FFI_HOST_SO | cut -c1-16)"
echo "    mpiw    $(sha256sum $LORRAX_MPIWRAPPER_SO | cut -c1-16)"
