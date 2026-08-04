set -eu
export LORRAX_ROOT=/scratch2/08271/jackmc/slabio_padding/src_fix
export LORRAX_VENV=/work2/08271/jackmc/frontera/lorrax_env/.venv
export LORRAX_FFI_HOST_STAGE=/work2/08271/jackmc/frontera/lorrax_ffi_unified/build_host_IMPLICITPAD
export LORRAX_SLATE_HOST_INSTALL_DIR=/work2/08271/jackmc/frontera/slate_builds/cpu/install
export LORRAX_HDF5_ROOT=/home1/apps/intel19/impi19_0/phdf5/1.14.6
export LORRAX_IMPI_ROOT=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64
export LD_LIBRARY_PATH=/work2/08271/jackmc/frontera/slate_builds/cpu/install/lib64:/opt/intel/compilers_and_libraries_2020.1.217/linux/mkl/lib/intel64_lin:/home1/apps/intel19/impi19_0/phdf5/1.14.6/lib:/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/lib/release:/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/lib:/opt/intel/compilers_and_libraries_2020.1.217/linux/compiler/lib/intel64_lin:${LD_LIBRARY_PATH:-}
bash /scratch2/08271/jackmc/slabio_padding/src_fix/config/frontera/build_ffi_host.sh --fresh
