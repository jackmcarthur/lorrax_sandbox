# Public-release staging package

## Summary

LORRAX's distribution metadata now carries the native FFI CMake, C++/CUDA, header, and
staging-script sources. Source distributions additionally carry the cluster configuration
templates and installation documentation. The install docs now identify JAX >=0.9 / CUDA
13 as the public pure-JAX default while retaining CUDA 12.9 as the only validated native
FFI reference environment.

Source branch `agent/release-staging-package`, commit `b924917`, was pushed to
`origin/agent/release-staging-package`.

## Code changes

| File | Change |
|---|---|
| `pyproject.toml` | Added an explicit setuptools build backend and an allowlist for native FFI package data. |
| `MANIFEST.in` | Added FFI sources, cluster templates, and install docs to sdists; pruned the generated CMake build tree. |
| `src/ffi/common/ffi_loader.py` | Reports the build script at its actual installed path. |
| `docs/installation/*.md` | Separated the CUDA-13 pure-JAX default from the validated CUDA-12.9 native stack and documented wheel/source build locations. |
| `docs/quickstart.md` | Aligned the quickstart with the CUDA-13 dependency default and documented CPU backend selection. |

No license, checksum policy, prebuilt shared library, or vendor binary was added.

## Verification

| Check | Result |
|---|---|
| `git diff --check` | Pass |
| Setuptools `build_py` payload | 67 intended files under `ffi/`; representative CMake, CUDA, header, staging, and porting files present |
| Generated build-tree exclusion | Pass; no `ffi/common/cpp/build/` in wheel layout or sdist manifest |
| Actual sdist archive | Pass; `lorrax-0.1.0.tar.gz`, 380 entries, config/install/FFI assertions satisfied |
| Loader syntax and installed-path error hint | Pass |
| `JAX_PLATFORMS=cpu ... pytest -q tests/test_ffi_linalg_contract.py` | 1 passed, 22 skipped in 5.24 s |

The base Python environment did not contain the `wheel` module, so an actual `.whl`
archive was not emitted locally. The setuptools wheel payload stage was built and checked
directly. The full GPU regression suite was intentionally not run because the user waived
multi-minute suite runs during the cluster shutdown window.

## Status

- [x] Package all in-tree native build and staging sources.
- [x] Exclude ignored CMake output from public artifacts.
- [x] Include cluster configuration and install documentation in the sdist.
- [x] Document CUDA 13 without implying CUDA-13 native FFI validation.
- [x] Push the isolated feature branch.
- [ ] Validate a full CUDA-13 native FFI build when an ABI-matched vendor stack is selected.

## Open questions

The public Python dependency and the validated HPC native environment intentionally remain
different support rows. A future CUDA-13 native release needs one matched cuSolverMp,
cuBLASMp, CAL/NCCL, compiler, and CUDA runtime stack plus the multi-rank FFI contract tests;
the Perlmutter CUDA-12 staging tree cannot be reused for that validation.
