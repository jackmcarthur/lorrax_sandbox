# LORRAX `gflat_memory_model.py` — synthesis

Branch: `agent/bispinor-ibz` (lorrax_B), HEAD = `0f355b7`.
Scope: the planner that picks `(band_chunk, r_chunk, gflat_chunk_size)` for `fit_zeta_to_h5` + `compute_V_q`, plus the runtime instrumentation that validates it.

## 1. Branch state — commits and what they fix

| commit | what landed | source of truth |
|---|---|---|
| `c21d855` | initial per-rank HBM model + chunker | n/a (origin) |
| `381e010` | per-peak `fft_box_factor`, util→0.94, r-first picker | agent_c_design.md §3 |
| `21f2ed6` | centroids + L_q persistent in Peak D | live_arrays Peak D probe |
| `409be4f` | full persistent-array refit; centroids ×4, sphere-idx leak term, Peak E, cs cap=100 | agent_g_census, agent_h_full_lifecycle, agent_i_v_q_memory |
| `d1fcd20`, `94542c2` | wfn_loader/wfn_transforms canonical g_index caches (plug the 3→8 sphere-idx leak across channels) | agent_g census; agent_l live verify |
| `48ee189` | thread cohsex.in `gflat_chunk_size` through planner; warn when override exceeds cap=100 | agent_j cs=200 silent miss |
| `81817e2`, `da7b41f`, `9afa11e`, `685f11b` | canonical `WfnLoader.box_index_dev` accessor; collapses 3 content-distinct sphere-idx device buffers to 1; planner constant restored to 1 | agent_l (sphere=3 found), agent_m (refactor + restored) |
| `6ba1fad` | `_mem_probe` falls back to `nvidia-smi` per-rank when `jax.devices()[0].memory_stats()` returns None | sandbox bug under platform allocator |
| `e2de150` | docs/MEMORY_MODEL.md "Predicted-vs-realized faithfulness" section | agent_n |
| `0f355b7` | `gflat_acc` charged in Peak C persistent base (was `=0.0` "to avoid double-count with D") | agent_q live_arrays census |

## 2. Planner architecture

Five per-rank HBM peaks; HWM = `max(A, B, C, D, E)`. Peaks C and D have isolated transient slots (separate jits, `block_until_ready` between them — `isdf_fitting.py:2442–2496`), so `gflat_acc` correctly appears in both persistent bases. Peak E is the V_q tile compute.

| peak | jit | dominant transient |
|---|---|---|
| A | `gflat_to_rmu` (centroid load) | `fft_box = b_chunk · n_rtot · 16 · 4.0` |
| B | CCT/Cholesky | open-spin pair-accumulator |
| C | `fit_one_rchunk` | `pair_density_slots × c128(nk, ns², mu) × r_chunk` |
| D | `accumulate_rchunk_to_gflat` | `cs × n_rtot · 16 · 2.0` (factor_D=2 HLO-verified) |
| E | `compute_V_q` per-tile | `2 × c128(n_q_ibz, mu, ngkmax) / p_xy` (ζ_L+ζ_R slabs) |

### 2.1 Constants and their pedigree (HLO-calibrated)

- `pair_density_slots = 3` — `agent_d_hlo_calibration.md` M1 (CrI3 80Ry bispinor + Si 4×4×4 charge), bit-exact match.
- `fft_box_factor_D = 2.0` — `agent_d` M2 at cs=360.
- `fft_box_factor_A = 4.0` — empirical legacy from initial commit; HLO-revisited not needed since A is rarely binding.
- `N_SPHERE_IDX_BUFFERS_BISPINOR = 1` — post-canonical-accessor (commit `9afa11e`). Was 3 pre-refactor (3 content-distinct numpy sources hitting `wfn_transforms._cached_gindex_dev`).
- `GFLAT_CHUNK_SIZE_CAP = 100` — empirical cliff at cs=1414 OOM. Smooth scaling holds up to cs=1000 (agent_n X6); the mechanism for the cs=1000→1414 cliff is **not yet diagnosed** (Open improvement #4 below).

## 3. Persistent-array catalog (all `live_arrays`-verified at production CrI3 80Ry)

Per `docs/MEMORY_MODEL.md` appendix and agent_g/agent_h census:

| array | live_arrays signature | formula (per-rank bytes) | lifetime |
|---|---|---|---|
| centroids ×4 | `c128(nk, ns, mu, nb_l)` and `c128(nk, ns, mu, nb_r)` — 4 buffers per channel (rmuT_X + Y for L+R) | `4 · _bytes_c128(nk, ns, mu, nb_total, shard=p_xy)` | `persistent_throughout_zeta_fit` |
| L_q | `c128(nq, mu, mu)` sharded p_xy | `_bytes_c128(nq, mu, mu, shard=p_xy)` | `persistent_throughout_zeta_fit` |
| gflat_acc | `c128(nq_disk, mu, ngkmax)` sharded p_xy | `_bytes_c128(nq_disk, mu, ngkmax, shard=p_xy)` | `persistent_throughout_zeta_fit` (Peak C + D) |
| sphere_idx | `int32(nk, nx, ny, nz)` REPLICATED — every rank | `n_buffers · nk · prod(fft_grid) · 4` (n_buffers=1 post-canonical-accessor) | `persistent_throughout_zeta_fit` |
| V_qmunu_CC | `c128(nq, mu, mu)` sharded p_xy | `_bytes_c128(nq, mu, mu, shard=p_xy)` | `persistent_post_compute_V_q` (carried into Σ_X) |

### 3.1 Pitfalls when verifying

- `_bytes_c128(*dims, shard=p)` returns the per-rank byte count for replicated-or-sharded buffers. **Replicated buffers (sphere_idx) cost N×, not N÷rank.** Easy to misclassify.
- Production `nb_total = (val_l + cond_l) + (val_r + cond_r)`. For bispinor with 4 channels, the planner does NOT multiply by 4 — channel state is freed by `gc.collect()` between channels (validated at agent_l W4 = 3 sphere buffers, not 8).
- Cross-channel state actually carrying over: ~0.26 GB/dev (small relative to gflat_acc). Worth tracking if a future bug surfaces.

## 4. Calibration — predicted vs realized HBM

### 4.1 CrI3 6×6 80 Ry SOC bispinor, 16 GPUs, 4×4 mesh

| config | r_chunk | planner HWM_pred | nvsmi peak (`platform` alloc) | mem_stats peak (BFC + 95%) | %-err pred vs truth |
|---|---|---|---|---|---|
| natural | 19312 | 55.99 GB/dev | 8.37 | n/a | — |
| sweet-spot (X3/A2) | 24576 | **70.11 GB/dev** | 8.67 | **76.05 GB/dev** | **−8.5% under** |
| cliff (Z1) | 28672 | ~77.4 GB/dev | n/a | OOM (75.3 GB single-block request) | correctly refused |

### 4.2 Si 4×4×4 SOC bispinor, 4 GPUs, 1×4 mesh — μ sweep

| μ | r_chunk × n_chunks | HWM_pred | mem_stats peak (BFC+95) | %-err | chunk uniformity |
|---|---|---|---|---|---|
| 384  | 10268 × 2 | 56.00 | 56.30 | **−0.5%** | uniform |
| 768  | 5832 × 3  | 55.99 | 62.78 | **−10.8%** | **stub last chunk = 2160 (37% of nominal)** |
| 1200 | 3552 × 4  | 55.97 | 59.03 | **−5.2%** | uniform |
| 1800 | 2280 × 7  | 56.00 | 57.28 | **−2.2%** | uniform |

### 4.3 Pattern

- CrI3 −8.5% is **not** a system-independent constant. Si spans −0.5% to −10.8%.
- The %-err correlates with **last-chunk size uniformity**, not with μ. Worst case (μ=768) has a 37%-sized stub last chunk; best case (μ=384, μ=1800) has uniform chunks.
- Smooth scaling up to cs=1000 (agent_n X6). cs=1414 OOM is discontinuous and uncharacterized.

## 5. Methodology — how to measure faithfully

### 5.1 The OOM-relevant metric

`jax.devices()[0].memory_stats()['peak_bytes_in_use']` is the **ground truth** XLA-arena peak — only available under `XLA_PYTHON_CLIENT_ALLOCATOR=default` (BFC). Returns `None` under `platform` (cudaMallocAsync).

Standard sandbox env (`platform` + `PREALLOCATE=false`) is correct for FFI compatibility (NCCL/cuSOLVERMp/phdf5 starvation if you preallocate XLA's BFC). **For HBM measurement only**, override the agent's run command:
```bash
LORRAX_SHIFTER="$LORRAX_SHIFTER \
  --env=XLA_PYTHON_CLIENT_ALLOCATOR=default \
  --env=XLA_PYTHON_CLIENT_PREALLOCATE=true \
  --env=XLA_PYTHON_CLIENT_MEM_FRACTION=0.95"
```
Verified to expose `peak_bytes_in_use` across both CrI3 and Si runs.

### 5.2 What NOT to trust

- **`nvidia-smi memory.used` under sandbox-default allocator**: cudaMallocAsync returns pages between operations at sub-second granularity. Sampling at seconds misses in-jit peaks. Round 7 (`agent_n_faithfulness_audit.md`) wrongly concluded 7-8× over-prediction based on this metric; corrected by Round 8 (`agent_o_allocator_audit.md`).
- **`live_arrays` byte-sum divided by world-size**: mis-counts sharded vs replicated. Replicated arrays count on every rank; sharded arrays count once total. Rounds 3-6 used naïve sums as proxies; Round 7+ requires explicit shard accounting.
- **SimpleNamespace mocks of `meta` for offline planner replay**: easy to miss bispinor factors. Mine missed a 4× factor on centroids and pair-density. Always grep `gw.out` for "G-flat memory model — chunk plan + HWM estimate" — that's the production planner output, logged verbatim per run.
- **`device.memory_stats()` under `platform` allocator**: returns `None`. Logged to `KNOWN_SANDBOX_ERRORS.md`.

### 5.3 Probe instrumentation

`_mem_probe` in `src/common/isdf_fitting.py` (commit `6ba1fad` + earlier). Active when `LORRAX_MEM_DEBUG=1`. Captures, per rank, at named probe points (`pre_rchunk_loop`, `after_fit_one_rchunk chunk=N`, `zeta_fit_end`, `pre_v_q`, `post_v_q`):
- `live_arrays` byte sum (global)
- `device.memory_stats()['peak_bytes_in_use']` if available (BFC); else `nvidia-smi memory.used` per rank

Per-r-chunk timing: `LORRAX_RCHUNK_DEBUG=1` enables `[rchunk_dbg]` lines per channel × chunk reporting z_q_build / solve / write / total.

Bound runtime for probing: `LORRAX_MAX_RCHUNKS=3 LORRAX_EXIT_AFTER_ZETA=1` exits after 3 chunks per channel — ~60-90s per Si config, ~5-6 min per CrI3 config.

### 5.4 HLO when planner-vs-runtime disagrees

`memory-usage-report.txt` per jit (under `xla_dump/jit_<name>/`). Categories: `preallocated-temp`, `intermediate`, `constant`, `argument`, `output`. The planner models `intermediate` correctly (per-jit HLO bit-exact at `fit_one_rchunk`: 66.32 GB HLO vs 66.41 GB planner at CrI3 X3, agent_q). What HLO doesn't show: NCCL collective buffers, cross-jit lifetimes, CUDA context overhead.

Workflow in `skills/profiling_stack/SKILL.md`. With ~3000 helper jit dumps in a typical run, the search pattern is `find xla_dump -name "memory-usage-report.txt" -path "*fit_one_rchunk*"`.

## 6. Identified gaps with clear paths to closure

### 6.1 Last-chunk stub aliasing — **highest leverage, smallest fix** [Si μ=768 → -10.8%]

When `r_chunk` doesn't evenly divide `n_rtot`, the last chunk gets a different (smaller) shape. XLA re-traces or pads with extra scratch the planner doesn't model. Si μ=768 (chunks 5832, 5832, 2160 — stub = 37% of nominal) hit -10.8% vs nearby uniform configurations at -0.5% to -2.2%.

**Proposed fix:** in the `r_chunk` picker (`gflat_memory_model.py:720`-ish), after computing the natural `r_chunk = int(headroom_C / α_C)`, snap to the largest divisor of `n_rtot` ≤ that value, OR to `ceil(n_rtot / n_chunks)` where `n_chunks = ceil(n_rtot / r_chunk)`. Should be a 5-10 LOC change. Validate with a Si μ=768 re-run; expect %-err to drop from -10.8% to within Si's uniform-chunk range (−0.5% to −5.2%).

### 6.2 NCCL collective buffers — **deliberately unmodeled**

`device.memory_stats()` peak minus HLO `peak_heap_bytes` ≈ 2-3 GB/dev consistently at CrI3 80Ry — agent_q candidate (2). NCCL allocates outside the XLA graph; no HLO visibility. Would need `NCCL_DEBUG=INFO` logs (not currently captured in any run) to size exactly.

**Decision:** user has chosen NOT to model this for CPU portability (LORRAX may run on CPU; no NCCL there). The −8% on CrI3 / variable on Si is the cost of that choice. If a target system has tighter memory and CrI3 size grows, this becomes a 1-line constant addition.

### 6.3 cs=1000 → 1414 OOM cliff — **mechanism undiagnosed**

`agent_n` X6 (cs=1000) ran cleanly at 8.67 GB nvsmi peak. cs=1414 OOMs (pre-cap-100 era empirics, mentioned in agent_f). Smooth `factor_D=2.0` model has no discontinuity at 1000-1414. Likely cuFFT plan algorithm flip (e.g., radix change between specific batch sizes), but **not verified**.

**Path to closure:** dump HLO at cs=1100, 1200, 1300, 1400 and compare `accumulate_rchunk_to_gflat` `memory-usage-report.txt` per. Look for `preallocated-temp` jumps. ~30 min of compute on one alloc. Currently the cap=100 saves us, so this is low priority — but worth scoping if production needs ever push cs higher.

### 6.4 cross-channel residue — **small, but real**

agent_l W4 measured ~0.26 GB/dev of charge-channel centroid buffers persisting into the first transverse channel's `fit_one_rchunk`. The planner doesn't model this. At μ=1520 it's tiny; at hypothetically larger μ it scales linearly.

**Path to closure:** if Si μ-sweep robustness becomes worse at larger μ, add a `cross_channel_residue` term. Currently below noise.

## 7. Things that have been resolved (don't re-investigate)

- **Centroids ×4 vs ×2**: resolved in `409be4f` — physical buffers are L+R × (rmuT_X + Y) transpose pairs.
- **Sphere-idx 3→8 growth across channels**: resolved in `d1fcd20`+`94542c2` (canonical caches), then `9afa11e` (canonical `WfnLoader.box_index_dev`). Stable at 1 buffer post-fix.
- **gflat_acc accounted only in Peak D**: resolved in `0f355b7`. Charged in both C and D persistent bases (no double-count — separate jits).
- **cs override invisible to planner**: resolved in `48ee189`. Override threaded + warn-on-cap-violation.
- **Pre-Round-7 "planner over-predicts by 7-8×"**: was a measurement artifact (cudaMallocAsync hiding the peak from nvidia-smi). Corrected in agent_o.
- **HLO factor_D=2.0 and pair_density_slots=3**: verified bit-exact against `fit_one_rchunk` HLO `peak_heap_bytes` at production scale (agent_q). Do NOT retune these.

## 8. Operational guidance

- For OOM-relevant capacity decisions, **trust the planner's HWM at default sandbox env, with a 15% safety margin** until §6.1 is fixed (the μ=768 worst case sits inside that margin).
- If considering pushing past the planner's `r_chunk` pick, measure first with `BFC + preallocate=true + MEM_FRACTION=0.95` to get true `peak_bytes_in_use`. The planner is conservatively correct, not loose.
- The empirical CrI3 80Ry cliff is `r_chunk ∈ (24576, 28672)` — model correctly refuses `r=28672` at 77.4 GB > 70 GB budget. No more `r_chunk` headroom available at production scale without reducing some other dimension (mu, ngkmax).
- BFC allocator is ~10% faster than `platform` on pure-JAX workloads (agent_s timing data). Use `platform` when FFI is hot (Sigma stage, cuSOLVERMp, phdf5); use BFC for ζ-fit-only benchmarks.

## 9. Files of record

- Planner: `src/gw/gflat_memory_model.py` (lorrax_B)
- Probe: `src/common/isdf_fitting.py` `_mem_probe`
- Canonical sphere accessor: `src/common/load_wfns.py` `WfnLoader.box_index_dev`
- Docs: `docs/MEMORY_MODEL.md` (includes appendix with live_arrays signatures and faithfulness section)
- Sandbox bugs: `/pscratch/sd/j/jackm/lorrax_sandbox/KNOWN_SANDBOX_ERRORS.md` (`device.memory_stats()` is None under platform; nvsmi misleading under cudaMallocAsync)

## 10. Open work — priority-ordered

1. **§6.1 even-chunk picker** — 5-10 LOC; expected to drop worst-case Si %-err from -10.8% to roughly its uniform-chunk range. Validate on Si μ=768 + CrI3 sweet-spot.
2. **§6.3 cs cliff diagnosis** — only matters if we want to raise the cap. Not currently binding.
3. **§6.2 NCCL constant** — user-deferred for CPU portability. Quick win if portability is later dropped.
4. **§6.4 cross-channel residue** — defer until/unless larger-μ runs surface it.

Items 2-4 are not blocking; item 1 is the only one with both clear leverage and a clear, small implementation path.
