. /scratch2/08271/jackmc/b600_p64/harness/inner_common.sh
export PYTHONPATH=/scratch2/08271/jackmc/slabio_padding/src_fix/src:$PYTHONPATH
export JAX_PLATFORMS=cpu
cd /scratch2/08271/jackmc/slabio_padding/src_fix
"$LORRAX_PY" -m pytest -q \
  tests/test_file_io.py \
  tests/test_slab_io_routing.py \
  tests/test_restart_pad_roundtrip.py \
  tests/test_compute_all_V_q_g_flat.py \
  tests/test_compute_V_q_bispinor_g_flat.py \
  tests/test_qp_solver_config.py \
  tests/test_sanity_gates_jax.py \
  > /scratch2/08271/jackmc/slabio_padding/pytest.7888657/pytest.out 2> /scratch2/08271/jackmc/slabio_padding/pytest.7888657/pytest.err
exit $?
