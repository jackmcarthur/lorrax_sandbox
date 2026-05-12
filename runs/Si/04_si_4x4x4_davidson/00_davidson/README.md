# Si 4x4x4 DFT Hamiltonian Validation

This directory is the sandbox-side entrypoint for the standalone `psp`
DFT-Hamiltonian validation flow.

## Canonical path

Use the maintained source test:

```bash
python3 -u -m psp.tests.test_dft_hamiltonian
```

Do not treat the older local helper scripts here as the canonical path.
`run_davidson.py` and `run_direct_diag.py` predate the current `psp` API and
still reference deleted setup helpers. `run_direct_diag_v2.py` is useful as
debugging history, but the maintained reproducer lives upstream in
`sources/lorrax/src/psp/tests/test_dft_hamiltonian.py`.

## Exact command in this sandbox

With an active Perlmutter interactive allocation:

```bash
export JOBID=51470500   # replace with your live interactive job id
cd /global/homes/j/jackm/scratchperl/lorrax_sandbox/runs/Si/04_si_4x4x4_davidson/00_davidson
./launch_test_dft_hamiltonian.sh
```

The launcher uses:

- Code: `/global/homes/j/jackm/scratchperl/lorrax_sandbox/sources/lorrax/src`
- Extra local deps: `/global/homes/j/jackm/scratchperl/lorrax_sandbox/sources`
- Reference data: `/global/homes/j/jackm/scratchperl/lorrax_sandbox/runs/Si/00_si_4x4x4_60band`
- Runtime: `shifter --module=gpu --image=nvcr.io/nvidia/jax:25.04-py3`

## Expected result

Success ends with:

```text
PASS: all k-points match QE to < 0.01 mRy
```

and reports `Max MAE: 0.0000 mRy`.

## Important detail

The mixed-k failure mode came from IBZ vs unfolded indexing. The canonical
test avoids that trap by loading full-BZ wavefunctions via
`read_Gvecs_to_devices` and then mapping each IBZ k-point to the matching
unfolded index before building `H_k`.
