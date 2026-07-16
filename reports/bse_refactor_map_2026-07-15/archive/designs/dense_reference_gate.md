# Design — DENSE-REFERENCE KERNEL GATE (BSE Phase 2, step 1)

Branch `agent/bse-phase2` off `main` (lorrax_A @ 6bd4dc9). New test module
`tests/test_bse_dense_reference.py` + one derived fixture in `tests/conftest.py`.
No source edits in this deliverable — the gate lands FIRST (briefly xfail),
then the B1 fix (next commit) flips it green. Evidence base:
`archive/adjudication/VERDICT.md`, `archive/adjudication/empirical_bsemat.md`,
`archive/files/kernel_dataflow_trace.md` (§Purpose formulas are load-bearing).

## 1. Fixture strategy — piggyback the existing GW e2e state

Reuse `gnppm_session` (conftest.py:96) verbatim — no second GW run. Verified it
is the right producer: `tests/regression/gnppm_debug/gnppm_test.in` has
`do_screened = true`, so `gw_jax.py:284 persist_w0_and_head` writes into
`tmp/isdf_tensors_*.h5`: `V_qmunu (nq,μ,ν)` flat-q, `W0_qmunu (nq,μ,ν)` with
`W0_ready=True`, `G0_mu_nu (μ,)`, `psi_full_y (nk,nb,ns,μ)`, `enk_full (nk,nb)`
Ry, `vhead`/`whead`, `kgrid` (int64[3]) — exactly the static-W bundle BSE reads
(tagged_arrays.py:82–120). MoS2 3×3×1 ⇒ **nk=9**, nspinor=2, nband=46.

New fixture `bse_dense_state` (session scope): copies `gnppm_session.run_dir`
(incl. `tmp/`) once — mandatory, since any driver re-run mutates the restart
(harness.copy_fixture `tmp_from=`), but here we do NOT re-run the driver at all.
The BSE side is a **library call, no subprocess**:

```python
from bse import bse_io
restart = bse_io._find_restart_file(str(run_dir / "gnppm_test.in"))
data = bse_io._load_ring_subset(restart, n_val=2, n_cond=2, px=1, py=1,
                                input_file=str(run_dir / "gnppm_test.in"))
```

Small BSE problem: **2v2c × 9k ⇒ N = nc·nv·nk = 36** (dense eigh trivial).
`_load_ring_subset` returns the padded, head-injected `psi_c, psi_v, eps_c,
eps_v, V_q0 (μ,μ), W_q (μ,μ,nkx,nky,nkz)` the matvecs consume — build the dense
reference from THOSE arrays so the check is apples-to-apples (no independent
Coulomb recompute; px=py=1 ⇒ band pad multiple 1, μ pad shared by both sides).
`WFN.h5` is present in the fixture dir (kgrid resolution for flat-q).

## 2. Dense H builder — explicit (k,k′) quadrature in numpy

Flat pair index `I=(c,v,k)` matching the code's `X[b,c,v,k]` layout. `Nk=nk`,
`q(k,k') = (k−k') mod (nkx,nky,nkz)` (C-order flatten). All from the SETTLED
formulas (kernel_dataflow_trace.md §Purpose; VERDICT.md exchange line):

```python
M = np.einsum('kcsm,kvsm->kcvm', np.conj(psi_c), psi_v)     # pair amplitude
D[c,v,k]                = eps_c[k,c] - eps_v[k,v]           # diagonal, correct

# EXCHANGE — DENSE in (k,k'), the settled fix (no delta_kk'):
Kx[c,v,k, c',v',k'] = (1/Nk) * sum_{mu,nu}
                        conj(M[k,c,v,mu]) * V_q0[mu,nu] * M[k',c',v',nu]

# DIRECT — screened W, W_{mu,nu}(k-k'):
Kd[c,v,k, c',v',k'] = (1/Nk) * sum_{mu,nu}
    ( sum_t conj(psi_c[k,c,t,mu]) * psi_c[k',c',t,mu] )      # conduction line
    * W_q[mu, nu, *q(k,k')]                                  # index at k-k'
    * ( sum_s psi_v[k,v,s,nu] * conj(psi_v[k',v',s,nu]) )    # valence line

H = np.diag(D.ravel()) + Kx.reshape(N,N) - Kd.reshape(N,N)
```

Written index-by-index (nested loops over k,k',c,c',v,v' or the einsums above);
N≤144 so cost is negligible. The `1/Nk` = the two `1/√Nk` factors the matvecs
carry (bse_serial.py:47,49,63). **W index sign** (`q=k−k'` vs `k'−k`) is pinned
empirically by the W-only positive control (§3b) — if it fails, flip the sign;
it must NOT be left ambiguous. Head: V_q0 here is post-injection (matches the
matvec); for the BGW test (§4) rebuild exchange from the **bare** q=0 tile
(`f['V_qmunu'][0]`, G=0 excluded) since BGW `/mats/exchange` excludes the head.

## 3. What it asserts (primary gate, plain suite)

Random complex `X` shape `(nb_block, nc, nv, nk)` (independent re/im — do NOT
reuse one PRNGKey, cf. bse_io.py:914 weirdness — a correlated probe masks
phase errors). For each matvec kind run on the SAME `data`:

- **serial** `bse_serial.apply_bse_hamiltonian_single_device`
- **simple** `bse_simple.build_bse_simple_matvec`
- **ring** / **gather** `bse_ring_comm.build_bse_ring_matvec(matvec_kind=…)`
- **stack (future)** non-TDA `build_bse_ring_matvec_full` — see §5.

(a) **Full H:** `matvec(X) ≈ (H @ X.ravel())` to `atol=1e-9` (x64), reshaped.
    *This is the xfail gate*: the k-diagonal code FAILS it on exchange
    (off-diag k-blocks missing), PASSES after B1.
(b) **W-only positive control** (must pass EVEN pre-fix — proves only exchange
    is wrong and pins the W sign): `(matvec_{include_W=True} −
    matvec_{include_W=False})(X) ≈ −(Kd @ X)`.
(c) **D+V isolation:** `matvec_{include_W=False}(X) ≈ ((diag(D)+Kx) @ X)` —
    the exchange-only xfail, cleanest failure locus (D diagonal is exact).
(d) **Spectrum:** `np.linalg.eigvalsh(H)` lowest `n_eig=4` ≈ the iterative
    solver (`bse_lanczos.solve_bse`, 1-device, on the serial matvec) to
    `atol=1e-6` Ry. Pre-fix the solver rides the buggy matvec ⇒ differs ⇒
    xfail; post-fix both resolve H's spectrum.

Assertions (a),(c),(d) carry `@pytest.mark.xfail(reason="B1 k-diagonal exchange
— dense fix pending", strict=True)`; **strict** so they XPASS-error the moment
B1 lands, forcing removal of the marker in the same commit. (b) is a plain
assert (positive control, no xfail).

## 4. Second test — LORRAX K^x vs BGW `/mats/exchange` (marked `extra`)

`@pytest.mark.extra` (opt-in; needs `runs/` data — skip if absent):
`runs/Si/04_si_4x4x4_bse/00_bgw_bse/bsemat.h5` + a Si 4×4×4 LORRAX restart.
Guard: `pytest.importorskip`-style `skipif` on both files existing.

Build LORRAX dense exchange `Kx` (bare tile, §2, head excluded) for the Si
4v4c/64k window, reshape to `(k,k',c,c',v,v')`. Read BGW with the mapping from
empirical_bsemat.md §1: `/mats/exchange` C-order axes
`{ikp, ik, icp, ic, ivp, iv, flavor}` (shape `{64,64,4,4,4,4,2}`), flavor→complex.
Apply the six BGW conventions (valence-axis flip, Ry, iv=1=highest valence).
ISDF ≠ exact pair densities, so assert the **dense signature**, not tight
numerics: per-k-block Frobenius norms — off-diagonal blocks (e.g. (ik,ikp)=(0,1),
(0,63)) are NONZERO and within ~20% of the ratio-to-diagonal BGW shows
(diag max 0.387 / off (0,1) 0.269 / (0,63) 0.151, empirical_bsemat.md §2).
This is the end-to-end proof that the fix reproduces BGW's dense k-coupling; it
is a `extra` diagnostic, NOT part of the <30 s plain-suite budget.

## 5. Expected failure mode & the xfail choreography

**Order is deliberate:** this gate is committed FIRST, on `agent/bse-phase2`,
while the code is still k-diagonal. At that commit:
- (b) W-positive-control: **PASS** (W untouched by B1).
- (a),(c),(d): **xfail** (strict) — the coded exchange is the (k,k) diagonal of
  H scaled 1/Nk; dense H has off-diagonal k-blocks of equal magnitude
  (VERDICT.md, empirical §2), so `matvec(X) ≠ H@X` by O(1).
- `extra` BGW test: xfail/skip likewise (LORRAX exchange still k-diagonal).

The **B1 fix commit** (next in the series) changes the exchange encode/decode in
all matvecs to the k-summed form (`'kcvN,bcvk->bN'` encode, broadcast decode —
VERDICT.md §Architecture; ONE unified path, no parallel Q=0/finite-Q code per
the no-redundancy mandate). At that commit the strict xfails XPASS → flip to
plain asserts (remove markers) in the SAME commit. Golden gates
(`test_gw_jax_regression` cohsex/si_cohsex_3d/gnppm/bispinor +
`test_ibz_equals_full_bz`) stay green — B1 does not touch GW Σ paths. Expect
BSE singlet/bright eigenvalue shifts (re-anchor deliberately); Si triplet
manifold ~unchanged.

## 6. Cost budget & runtime notes

Plain-suite portion adds **< 30 s**: no new GW subprocess (reuses
`gnppm_session`), one `_load_ring_subset` (~0.5 s), a 36×36 numpy build, ~5
small jit matvec compiles (shared XLA cache), one 36×36 `eigvalsh`, one short
1-device Lanczos. Runs on **1 GPU**, MoS2-scale, pytest-collected. The stack
(non-TDA `_full`) assert is `xfail(reason="B2 malformed B-encode einsum",
strict=False)` today — it crashes at trace with `include_W=True` (B2,
kernel_dataflow_trace.md §B2); it becomes a real assert after Phase-1 B2 lands.

GPU verification of this gate before commit (module system is BROKEN in
non-interactive shells — KNOWN_SANDBOX_ERRORS 2026-07-15; NO lxrun/module load):
```
salloc --nodes=1 --qos=interactive --time=00:40:00 --constraint=gpu --gpus=1 \
       --account=m2651 -J lx-alloc-$USER <script>
```
`<script>` = the validated module-free `srun … shifter --image=nvcr.io/nvidia/
jax:25.04-py3 --module=gpu,mpich --volume=<nvhpc,phdf5,slate stages> …` pattern
(copy from KNOWN_SANDBOX_ERRORS 2026-07-11 "Working alternative" +
`reports/bse_refactor_map_2026-07-15/cleanup_verify/`), with
`--env=PYTHONPATH=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A/src:…`
(lorrax_A src FIRST — else bare pytest resolves lorrax_B, KSE 2026-06-17), and
stage `liblorrax_ffi.so` into the worktree first (gitignored artifact, KSE
2026-07-15). Run the full plain 1-GPU suite before the final commit of each
series. Never `python3` on a login node.
