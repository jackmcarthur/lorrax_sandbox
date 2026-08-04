. /scratch2/08271/jackmc/b600_p64/harness/inner_common.sh
export PYTHONPATH=/scratch2/08271/jackmc/slabio_padding/src_fix/src:$PYTHONPATH
export JAX_PLATFORMS=cpu
cd /scratch2/08271/jackmc/slabio_padding/src_fix
"$LORRAX_PY" -m pytest -q tests/test_file_io.py > /scratch2/08271/jackmc/slabio_padding/gate16.7888651/pytest/pytest.out 2> /scratch2/08271/jackmc/slabio_padding/gate16.7888651/pytest/pytest.err
exit $?
