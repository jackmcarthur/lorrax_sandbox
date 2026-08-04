# SlabIO read/write audit — 2026-08-04

Audit of all four SlabIO backends and both directions for siblings of d935ce7
(the empty-selection bounds test that stranded two 32-node legs on 2026-08-02).
Branch `fix/slab-io-audit`, commit **8050e34**, worktree
`/work2/08271/jackmc/frontera/wt-slabio`.  Artifacts under
`/scratch2/08271/jackmc/slabio_audit/`.

Jobs: **7888471** (build), **7888470** (arm=base), **7888525** (arm=fix),
**7888528** (arm=base, read-after-write control), **7888537** (regression gate).

## 0. Method

Instrument copied verbatim from the certified `phdf5_padrank` harness: every
rank's stdout AND stderr to its own file, written by the shell, with
`LORRAX_PHDF5_WRITE_DEBUG=1` so the writer announces its per-call decision on
every rank.  stderr written just before process death is lost under
srun+apptainer, so the merged log is not the instrument.  One case per process
launch; every case under an external `timeout`, because the failure mode under
test is a hang.

A/B control.  Both arms run the same probe.  `arm=base` is src @ 57df1c7 with
`build_host_PADFIX` (built from d935ce7); `arm=fix` is this branch with
`build_host_SLABIO`.  No commit between d935ce7 and 57df1c7 touches
`src/ffi/cpp`, so PADFIX's C++ is byte-identical to this branch's base and the
two `.so` differ by exactly the diff below.  Exported-symbol diff between them
is EMPTY (475 both), so nothing else moved.

Geometry is the production one in kind: dataset `(n_q, mu, nG)` LOGICAL, buffer
`(n_q, mu_padded, nG)` PHYSICAL, sharded `P(None, ('x','y'), None)`, with
`mu = P + 1` so ranks with `r*loc >= mu` own a wholly-padded block.  At P=4 that
is rank 3; at P=16, ranks 9-15.

## 1. CONFIRMED — a refusal that fires on a proper subset of ranks

The write path's bounds test was fixed in d935ce7 so that an EMPTY selection
vetoes it.  It was still taken on **this rank's advanced offset**, so it reached
a verdict only on ranks whose selection is non-empty.  When the caller's logical
slab overruns the dataset AND some ranks are wholly padding, the two populations
disagree and the ones that do not refuse enter the collective alone.

Captured verbatim, arm=base, P=4 (job 7888470, `run.7888470/p4_oobw/rank_*.err`):

```
rank=0 extent=[2,5,8] offset=[0,1,0] count=[2,2,8] collective=1 oob=0 empty=0
rank=1 extent=[2,5,8] offset=[0,3,0] count=[2,2,8] collective=1 oob=0 empty=0
rank=2 extent=[2,5,8] offset=[0,5,0] count=[2,1,8] collective=1 oob=1 empty=0
rank=3 extent=[2,5,8] offset=[0,7,0] count=[2,0,8] collective=1 oob=0 empty=1
```

One rank of four refuses.  The other three enter `H5Dwrite`.  The job then hangs
for 306 s and dies at the JAX shutdown-barrier deadline
(`DEADLINE_EXCEEDED: Barrier timed out ... Shutdown::`), rc=1 on every rank,
with no HDF5 error and no traceback anywhere.  At P=16 the same shape gives
rank 8 alone `oob=1`, ranks 9-15 `empty=1`, ranks 0-7 `oob=0` — 1 of 16 refusing.

The READ path is worse: it had **no bounds test at all**, which made
`H5Sselect_hyperslab` the de-facto one, and that also fails only on ranks with a
non-empty selection.  arm=base P=4 `p4_oobr` produced **no message on any rank**
and was killed by the external timeout at 420 s.  Rank 0's last line is the
probe's own "issuing the overrunning read"; ranks 1-3 printed nothing past
startup.  This is the exact signature that cost a day on 2026-08-02.

### Fix and certification

The bounds test now runs on the LOGICAL slab, `offset_base + valid_shape`, in
both `write_ffi.cc` and `read_ffi.cc`.  Both are replicated control vectors —
identical on every rank by construction — and max over ranks of
`offset[d] + file_count[d]` is exactly `offset_base[d] + valid_shape[d]`, so the
test accepts and refuses exactly the same calls.  What changes is who refuses.
A globally empty request (`valid_shape[d] == 0` anywhere) is still never
bounds-tested, preserving d935ce7's doctrine.

| case | arm=base (7888470) | arm=fix (7888525) |
|---|---|---|
| `p4_oobw` | **HANG**, 306 s, 1 of 4 ranks refused, killed at the JAX shutdown barrier | **7 s**, 4 of 4 refuse: `phdf5 async write: logical slab out of bounds ds=/zeta_like extent=[2,5,8] offset_base=[0,1,0] valid_shape=[2,5,8]` |
| `p16_oobw` | **HANG**, 1 of 16 refused (rank 8; ranks 9-15 `empty=1`) | **6 s**, 16 of 16 refuse |
| `p4_oobr` | **HANG**, timed out at 420 s, **zero output on every rank** | **6 s**, 4 of 4 refuse: `phdf5 read: logical slab out of bounds ... refused identically on every rank` |

## 2. CONFIRMED — the read path IS correct for padding

The question the audit was asked: is `read_ffi.cc:140-149` right, or accidentally
right?  It is **right for the padding contract** and was measured to be so, not
inferred.

`rt` reads back every written file at the PADDED physical shape with
`valid_shape` = the logical extent, mu-sharded, so wholly-padded ranks take the
empty-selection branch of the READ collective; each rank checks its OWN block.
23 checks PASS at P=4 (1 wholly-padded rank) and at P=16 (7 wholly-padded
ranks), on BOTH arms, `max|delta| over ranks = 0.000e+00` everywhere.  The
padded blocks come back at exactly the memset zeros.  This is the shape
`zeta_loader.read_zeta_G_slab` uses in production.

## 3. CONFIRMED — the three backends are substitutable

Full writer x reader matrix, each in two request shapes (logical/replicated and
padded/mu-sharded), plus a serial-h5py readback of each written file.  P=4 and
P=16, both arms, all `0.000e+00`:

| writer \ reader | ffi | phdf5_host | allgather |
|---|---|---|---|
| ffi | 0 | 0 | 0 |
| phdf5_host | 0 | 0 | 0 |
| allgather | 0 | 0 | 0 |

`replica_dup` (spec `P(None,'x',None)` on a 2-D mesh, so the `y` axis is a
replica axis and every rank with `coord_y != 0` must drop to a null selection
and still join the collective) round-trips exact.

`H5D_FILL_TIME_NEVER` + `H5D_ALLOC_TIME_EARLY`: a dataset created and never
written reads back all zeros (0 nonzero of 80).  This is what
`tagged_arrays.write_restart_state_to_h5(init_W0=True)` calls an "all-zeros
placeholder", and the measurement says it holds — but it holds because a fresh
Lustre inode reads back zero, not because HDF5 guarantees it.  `mode='w'`
unlinks the target, so production always gets a fresh inode.

## 4. PLAUSIBLE — read on a handle with writes still in flight

`write_slab` on the FFI backend only ENQUEUES.  `read_slab` drained only when it
first saw a dataset name (`_ds_id`), and `_introspect_dataset` (serial h5py on
the same path) ran before even that.  Three hazards: reading pre-write bytes;
`ctx->pinned_buf` memset under an in-flight `H5Dwrite` that is reading from it
(CUDA build); and two threads inside HDF5/MPI-IO on one file handle — the hazard
`create_dataset` already drains for, plus a collective-order mismatch if rank A
reads-then-writes while rank B writes-then-reads.

MEASURED: the window is reachable.  With 12 q-slabs of 1 MB/rank queued,
`dispatcher.pending = 1` at `read_slab` entry on every rank, on BOTH arms
(7888525 and the base control 7888528).  On the base arm that pending write
completed **concurrently with the read** — `[SlabIO.close] draining 0 pending
writes` afterwards — so two threads were inside HDF5/MPI-IO on one handle, as
predicted.  It produced no wrong data at this scale.  Verdict: **PLAUSIBLE,
precondition confirmed, corruption not observed.**  At production scale the
window is one `H5Dwrite` wide (~11 s), not one microsecond.

Fixed anyway: one unconditional `_drain_pending()` at the top of
`_FfiBackend.read_slab`.  Cost is zero when the queue is empty.  Round-trip
correctness unchanged (`rt`, `raw` both PASS on the fix arm).

## 5. PLAUSIBLE — collective-call-count asymmetries (code reading)

Not measured; exercising them means deliberately writing a non-SPMD caller.

* `_FfiBackend.close()` gated `_barrier("slab_io_ffi_close_attrs")` on the
  per-rank `_deferred_attrs` list.  Every `write_attr` call site in the tree is
  SPMD today (`tagged_arrays`, `v_q_bispinor`, `sigma_output` — all conditions
  rank-independent), so the barrier count matches; but that is a property of the
  callers, not of the method.  The barrier is now unconditional.
* `_MpiHostBackend.close()` cannot do that: its loop body is TWO collectives per
  attr (`comm.bcast` + a collective `create_dataset`).  It now agrees the
  deferred-attr name list across ranks with one unconditional `allgather` and
  refuses BY NAME on a divergence, instead of hanging with no message.
* `ffi.io.open_file` caches handles by path and ignored the mode.  A `mode='w'`
  caller on an already-open path gets the cached context back — and
  `_replace_inode_for_write` has already unlinked the target on rank 0, so every
  subsequent write lands in an orphaned inode and the run finishes rc=0 with
  nothing on disk.  Now refuses by name.  Deliberately NOT measured: exercising
  it on the base arm means two SlabIO objects sharing one `PhdfCtx*`, and the
  first `close_file` frees it under the second — a double-free, not an
  experiment.
* `read_ffi.cc` had no dataset-rank check (`write_ffi.cc` has always had one).
  With the new extent read that would overrun the `extent` vector on an ndim
  mismatch; the check is added, rank-invariant.
* `_MpiHostBackend.read_slab(as_numpy=True)` returned this rank's BLOCK for a
  sharded request, where the allgather backend returns the full slab.
  Unreachable through `SlabIO.read_slab` (which does not forward `as_numpy` on
  the phdf5 paths), reachable by using the backend directly.  Now refuses.

## 6. Left alone — flagged for the owner

1. **`ensure_dataset` does not verify an existing dataset's shape or dtype**
   (`context.cc:506`).  `H5Dopen` succeeds, the hid_t is returned, and the write
   clips against whatever extent that dataset happens to have.  In `mode='a'` a
   rerun at a different `mu` silently writes into the old geometry.  Adding the
   check would newly refuse reruns that today "work" (by writing a prefix), so
   it is a behaviour change, not a bug fix.
2. **`create_dataset` on an existing dataset diverges between backends.**
   `_AllgatherBackend` does `del self._file[name]` then recreates (contents
   reset); `_MpiHostBackend` returns early ("respect the existing dataset") and
   the C++ `ensure_dataset` reuses it.  A caller that creates then writes only
   part of the extent gets different bytes in the untouched region depending on
   the backend.  Which one is correct is a contract call.
3. **`AsyncDispatcher._raise_if_error` clears the stashed error as it raises**,
   and `_FfiBackend.close()` calls `_drain_pending()` before `_close_file`.  A
   worker exception on one rank therefore aborts that rank's `close()` before
   the collective `H5Fclose`, leaving the file handle open and `_closed` unset.
   The upstream strand has usually already happened by then, but a `try/finally`
   here would at least keep the close symmetric.  Changing it changes error
   semantics.
4. **`ctx->pinned_buf` is shared** between the SYNCHRONOUS `ReadImpl` (runs on
   the XLA thread) and the writer thread (async write on CUDA;
   `read_kchunk_union` on both platforms).  §4's drain closes the SlabIO
   exposure; a direct `ffi.io` user mixing the sync reader with the kchunk-union
   reader on one handle still races.  The fix is a C++ redesign (route the sync
   read through the writer queue), out of scope here.
5. **`_MpiHostBackend.write_slab` does not call `_validate_block_divisible`**
   while the FFI backend does.  This is a capability difference, not an
   oversight: the host backend derives offsets from real shard indices and
   handles uneven sharding that the FFI cannot.  Adding the check would remove
   working functionality.
6. **`ReadKchunkImpl` / `ReadKchunkUnion` take no logical bounds test.**  Their
   offsets and counts are caller-supplied per k, so there is no single logical
   slab to test; the union path does check that the filespace and memspace
   selections have the same point count.
7. **`H5D_FILL_TIME_NEVER` makes "created but not written = zeros" a filesystem
   property.**  Measured true on a fresh Lustre inode (§3).  It would not hold
   for a dataset allocated inside a reused region.

## 7. Regression gate

`tests/multi_device/phdf5_padded_rank_write.py` gains three cases, keeping the
one-case-per-launch discipline and the "refuse rather than gate nothing" rule:

* `PADRANK_CASE=read_pad` — the read-side empty rendezvous at `mu = P+1`; every
  rank checks its own block; wholly-padded ranks must come back exactly zero.
* `PADRANK_CASE=oob_write` — the overrunning write; rc=0 iff THIS rank refused.
* `PADRANK_CASE=oob_read` — the same for the read path.

The oob cases fail by HANGING, so the docstring says to run them under a
wall-clock bound and read each rank's own rc: the gate is "every rank refused",
and a rank that neither refused nor finished is the failure.

CERTIFIED, job 7888537 at P=4: all six cases rc=0 on every rank, including the
three pre-existing ones unchanged (`repro` / `control` / `exact`, round-trip
0 mismatched).  `read_pad` reports rank 3 as WHOLLY PADDED with 0 mismatched;
`oob_write` and `oob_read` each show 4 of 4 ranks refusing by name in 6-7 s.

## 8. What did NOT change

The accept path.  Every round-trip in §2 and §3 is `0.000e+00` on both arms, at
P=4 and P=16, in 23 checks per launch.  The new bounds test is provably the same
predicate as the old one taken at its maximum over ranks, so no call that
previously succeeded now refuses.

## 9. Commits

* **8050e34** `slab_io: a refusal inside a collective must fire on every rank or none`
* **dccb02e** `gate: the padded-rank gate now covers the READ path and both oob refusals`

Branch `fix/slab-io-audit`, worktree `/work2/08271/jackmc/frontera/wt-slabio`.
NOT merged, NOT pushed.  `.so`:
`/work2/08271/jackmc/frontera/lorrax_ffi_unified/build_host_SLABIO/liblorrax_ffi_host.so`
(sha256 `f0ecf821bcb005ae`, 475 exported symbols, PROVENANCE stamped with the
source commit and the A/B control).
