# ===========================================================================
# env_common.sh -- OUTER (batch-shell) env for the bispinor ζ-reuse gate.
#
# Transcribed from /scratch2/08271/jackmc/b600_bispinor_p64/harness/env_common.sh
# with four changes, each stated:
#   * R points at this run directory;
#   * SRCREV is the fix/bispinor-zeta-reuse worktree HEAD, frozen into
#     $R/lorrax_frozen (see SRC_PROVENANCE.txt).  NO job reads the live
#     checkout;
#   * LORRAX_FFI_HOST_SO points at build_host_PADFIX, the CANONICAL host
#     .so (lorrax_ffi_unified/CANONICAL.md) -- NOT build_host_ONE, which
#     the archived b600 harnesses use and which lacks d935ce7;
#   * the staged bundle supplies the VENV only.  PYTHONPATH is pointed at
#     $R/lorrax_frozen/src instead of the bundle's src (inner_common.sh),
#     because the bundle predates this change set.  The venv, overlay and
#     every dependency are byte-identical to the certified runs.
# ===========================================================================
export R=/scratch2/08271/jackmc/bispinor_zeta_reuse
export DECK=/scratch2/08271/jackmc/mos2_4x4_test          # read-only deck
export REF=/scratch2/08271/jackmc/b600_p64                # bundle source

export LORRAX_ROOT=$R/lorrax_frozen                        # frozen config/
export LORRAX_FROZEN_SRC=$R/lorrax_frozen/src              # frozen src/
export LORRAX_SRC_FALLBACK=$R/lorrax_frozen/src
export LORRAX_BUNDLE=$REF/bundle_1fc2759/lorrax_cpu_bundle.tar
export LORRAX_SIF=/scratch2/08271/jackmc/lorrax_setup/py312.sif
export LORRAX_FFI_HOST_SO=/work2/08271/jackmc/frontera/lorrax_ffi_unified/build_host_PADFIX/liblorrax_ffi_host.so
export LORRAX_MPIWRAPPER_SO=/work2/08271/jackmc/frontera/lorrax_mpiwrapper/install/lib64/libmpiwrapper.so
export LORRAX_SLATE_HOST_LIB=/work2/08271/jackmc/frontera/slate_builds/cpu/install/lib64
export LORRAX_IMPI_ROOT=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64
export LORRAX_HDF5_ROOT=/home1/apps/intel19/impi19_0/phdf5/1.14.6
export LORRAX_MKL_LIB=/opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin
export LORRAX_ICC_RUNTIME=/opt/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin
export LORRAX_PMI2_LIB=/work2/08271/jackmc/frontera/host_pmi/libpmi2.so.0

export LORRAX_W_RESIDUAL_CHECK=1

# RDMA/UCX userspace staging binds.  NEVER bind anything under /dev.
export HL=/usr/lib64:/hostlibs:ro,/usr/lib64/libibverbs,/etc/libibverbs.d
export BIND=/home1,/work2,/scratch1,/scratch2,/opt/intel,/tmp,$HL

[ -f "$LORRAX_MPIWRAPPER_SO" ] || { echo "REFUSING: no MPIwrapper at $LORRAX_MPIWRAPPER_SO"; exit 2; }
[ -f "$LORRAX_FFI_HOST_SO" ]   || { echo "REFUSING: no FFI host .so at $LORRAX_FFI_HOST_SO"; exit 2; }
[ -f "$LORRAX_SIF" ]           || { echo "REFUSING: no container at $LORRAX_SIF"; exit 2; }
[ -f "$LORRAX_BUNDLE" ]        || { echo "REFUSING: no frozen bundle at $LORRAX_BUNDLE"; exit 2; }
[ -f "$LORRAX_FROZEN_SRC/gw/gw_init.py" ] || { echo "REFUSING: no frozen src"; exit 2; }

# per-LAUNCH coordinator port
_coord_host=$(scontrol show hostnames "$SLURM_NODELIST" | head -1)
export JAX_COORDINATOR_ADDRESS="$_coord_host:$((13000 + SLURM_JOB_ID % 2000))"

echo "=== leg env: P=$SLURM_NTASKS  N=$SLURM_NNODES  coord=$JAX_COORDINATOR_ADDRESS"
echo "    frozen src  $LORRAX_FROZEN_SRC"
echo "    gw_init.py  $(sha256sum $LORRAX_FROZEN_SRC/gw/gw_init.py | cut -c1-16)"
echo "    bundle      $LORRAX_BUNDLE  (venv only)"
echo "    ffi so      $(sha256sum $LORRAX_FFI_HOST_SO | cut -c1-16)  [build_host_PADFIX]"
echo "    mpiw        $(sha256sum $LORRAX_MPIWRAPPER_SO | cut -c1-16)"
