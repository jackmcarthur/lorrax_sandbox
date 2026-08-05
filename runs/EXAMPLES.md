# Heavy artifact register

Run trees are curate-COPIED into `runs/<name>/`: harnesses, sbatch scripts,
per-rank stdout/stderr, `rc.txt`, results notes. Anything large — HDF5 output,
built `.so`, frozen source bundles — stays where it was produced and is listed
here by absolute path. Nothing is moved; CLAIMS.md provenance cites the
original paths, and those paths are the ones below.

These live on `/scratch2`, which is **purged**. A row whose path has gone is a
dead citation, not a lost result: the curated harness plus the `.so` sha256 is
enough to rebuild it. Copy anything you intend to keep to `/work2`.

Recorded 2026-08-04.

---

## 1. vh2d — `from_rbox` + n-side reshard, P=4

Curated: `runs/vh2d_rbox_reshard_p4/` (53 files)

| what | path |
|---|---|
| frozen source bundle | `/scratch2/08271/jackmc/vh2d/lorrax_frozen` |

Jobs 7888475, 7888481, 7888526, 7888533, 7888534 (all `development`, P=4).
7888534 is the bit-identity case: corner-sentinel padding, unmasked, `0.000e+00`
against the masked run. 7888526 is the `UnexpectedTracerError` capture that
established "do not wrap the body in an outer jit".

Per-rank `VmHWM` is in the curated `rank_*.out` — 0.383 GiB local / 0.402 GiB
distributed at rank 2, which is the memory claim's primary evidence.

## 2. slabio_audit — collective-strand defect, P=4 and P=16

Curated: `runs/slabio_audit_p4_p16/` (338 files), including `RESULTS.md` and
the gate `phdf5_padded_rank_write.py`.

| what | path |
|---|---|
| base source tree | `/scratch2/08271/jackmc/slabio_audit/src_base` |
| fixed source tree | `/scratch2/08271/jackmc/slabio_audit/src_fix` |
| raw readback h5, P=4 | `/scratch2/08271/jackmc/slabio_audit/run.7888525/p4_raw/raw.h5` (34 MB) |
| base-arm raw h5 | `/scratch2/08271/jackmc/slabio_audit/run.7888528/p4_raw/raw.h5` (34 MB) |

Jobs 7888470 (base arm: `p4_oobw` **hangs 306 s**, `p4_oobr` **silent timeout
420 s**, `p16_oobw` hangs), 7888525 (fix arm: all six rc=0), 7888528
(read-after-write control), 7888471 (build), 7888537 (gate).

The hang artifacts are the point of this tree — they are what a stranded
communicator looks like with no HDF5 error and no traceback.

`.so`: `build_host_SLABIO`, sha256 `f0ecf821bcb005ae…`, 905736 bytes, built
2026-08-04T16:13:25Z from `slabio_audit/src_fix`. Symbol diff vs `PADFIX`
EMPTY (475 both), which is what makes the A/B a one-`.so` diff.

## 3. slabio_padding — implicit padding, byte-identity, P=4 and P=16

Curated: `runs/slabio_implicit_padding_p4_p16/` (921 files)

| what | path |
|---|---|
| base source tree | `/scratch2/08271/jackmc/slabio_padding/src_base` |
| fixed source tree | `/scratch2/08271/jackmc/slabio_padding/src_fix` |
| regression decks (WFN/dipole/sigma h5) | `/scratch2/08271/jackmc/slabio_padding/src_fix/tests/regression/` |

Jobs 7888641 (build), 7888644/7888647 (gate iterations), 7888650 (P=4),
7888651 (P=16), 7888657/7888660 (pytest and its A/B).

The claim is a sha256 match between the pre-ruling spelling
(`global_shape=`+`valid_shape=`) and bare `write_slab(name, A)`:
**`c12e54ab4a217c1a`** at P=4 (mu=5, mu_pad=8, 1 wholly-padded rank) and
**`33af697ff8fc5d6c`** at P=16 (mu=17, mu_pad=32, 7 wholly-padded ranks).
Those hashes are in the curated gate logs.

`.so`: `build_host_IMPLICITPAD`, sha256 `51c69680d078bf85…`, 914336 bytes,
built 2026-08-04T17:14:01Z from `slabio_padding/src_fix`. 475 exported
symbols; symbol diff vs `build_host_SLABIO` EMPTY.

## 4. bispinor_zeta_reuse — ζ reuse for bispinor, bi4/P=4

Curated: `runs/bispinor_zeta_reuse_bi4/` (126 files), including
`SRC_PROVENANCE.txt` and the three-leg harness.

| what | path |
|---|---|
| frozen source bundle | `/scratch2/08271/jackmc/bispinor_zeta_reuse/lorrax_frozen` |
| leg A (fit) Σ | `/scratch2/08271/jackmc/bispinor_zeta_reuse/run_bi4/out_A/sigma_mnk.h5` (340 MB) |
| leg B (reuse) Σ | `…/run_bi4/out_B/sigma_mnk.h5` (340 MB) |
| leg C (refit) Σ | `…/run_bi4/out_C/sigma_mnk.h5` (340 MB) |
| the four ζ | `…/run_bi4/tmp/zeta_q*.h5`, `v_q_bispinor.h5` (93 MB) |

Jobs 7888568 (authoritative 3-leg gate + 46-check reader matrix), 7888569
(23 unit tests + fastloop, both legs), 7888477/7888535 (earlier revisions),
7888522 (caught the legacy-key defect).

The three `sigma_mnk.h5` are what the EXACT-0 claim rests on — B-vs-A and
C-vs-A are `max|Δ| = 0.000e+00` on 23660 parsed values, the only differing
text line being the timestamp header. **Keep these three if you keep
anything**: they are the byte-level evidence that Σ^B is not silently dropped
on the reuse path, and that is the failure 3d89885 fixed once already.

Frozen source: repo `57df1c71` + 3 modified + 1 untracked file (listed in
`SRC_PROVENANCE.txt`); `gw_init.py` sha256 `0854105 5f8d871b0…`.

## 5. io_harden — C++ IO FFI hardening

**IN FLIGHT at the time of writing** (jobs 7888701, 7888703). Not curated;
the tree is still being written. Curate after it lands.

| what | path |
|---|---|
| working tree | `/scratch2/08271/jackmc/io_harden` |
| fixed source | `/scratch2/08271/jackmc/io_harden/src_fix` |

---

## Canonical `.so` builds

`/work2/08271/jackmc/frontera/lorrax_ffi_unified/` — see its `CANONICAL.md`.

| build | sha256 (head) | note |
|---|---|---|
| `build_host_PADFIX` | `13a0b667261e071a…` | d935ce7 padded-rank write fix; the same-source control |
| `build_host_SLABIO` | `f0ecf821bcb005ae…` | + asymmetric-bounds fix, both directions |
| `build_host_IMPLICITPAD` | `51c69680d078bf85…` | + implicit padding |

`build_host_ONE` is missing the d935ce7 fix and its revision is
unrecoverable (453 exported symbols vs 475). `b600_p64`,
`b600_bispinor_p64` and `zeta_T_prodkappa` still point at it **on purpose**
so their logs stay truthful — those are records, not live configs.
