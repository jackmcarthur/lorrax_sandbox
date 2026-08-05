export LORRAX_FFI_PHDF5=1
export LORRAX_FFI_STAGE=/work2/08271/jackmc/frontera/lorrax_ffi_p12_cuda
export LORRAX_FFI_SO=/work2/08271/jackmc/frontera/lorrax_ffi_p12_cuda/build_phdf5/liblorrax_ffi.so
export LORRAX_FFI_SO_PHDF5=/work2/08271/jackmc/frontera/lorrax_ffi_p12_cuda/build_phdf5/liblorrax_ffi.so
source /scratch2/08271/jackmc/lorrax_sandbox/runs/p12_gpu_cert/srcsnap_p12fix/config/frontera/ffi_env.sh
export LORRAX_FFI_SO=/work2/08271/jackmc/frontera/lorrax_ffi_p12_cuda/build_phdf5/liblorrax_ffi.so
export PYTHONPATH=/scratch2/08271/jackmc/lorrax_sandbox/runs/p12_gpu_cert/srcsnap_p12fix/src
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
cd /scratch2/08271/jackmc/lorrax_sandbox/runs/p12_gpu_cert/run_g108_defaults_fix
exec /work2/08271/jackmc/frontera/lorrax_env/.venv/bin/python -u -m gw.gw_jax -i gw.in > /scratch2/08271/jackmc/lorrax_sandbox/runs/p12_gpu_cert/run_g108_defaults_fix/rank$SLURM_PROCID.log 2>&1
