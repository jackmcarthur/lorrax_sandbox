export PYTHONPATH=/scratch2/08271/jackmc/lorrax_sandbox/runs/p12_gpu_cert/srcsnap_p12fix/src
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
cd /scratch2/08271/jackmc/lorrax_sandbox/runs/p12_gpu_cert/srcsnap_p12fix
exec /work2/08271/jackmc/frontera/lorrax_env/.venv/bin/python -m pytest tests/test_runtime_startup_report.py -q
