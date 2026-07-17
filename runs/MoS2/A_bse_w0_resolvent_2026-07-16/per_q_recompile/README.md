# Per-q recompile elimination — `bse_w_exact --compare-wq` (MoS2 gnppm, 1 GPU)

Diagnosis + fix for the per-q XLA recompile in the W-resolvent path
(agent/bse-phase2, lorrax_A, 2026-07-17). Full writeup:
`reports/bse_refactor_map_2026-07-15/PHASE2_LOG.md` §"Per-q recompile elimination".

## Files

- `lxrun_census.sh` — module-free srun+shifter runner (1 GPU) = fixture_run's
  `lxrun_free.sh` + `--env=JAX_LOG_COMPILES=1` and a private empty JAX cache dir,
  so every XLA compile prints and nothing is masked by a warm on-disk cache.
- `baseline_compare_wq.log` — BEFORE (f19136e + probe): `scan` compiles 5× (1/q,
  ~4.8 s ea), `_map` 10×, `_roll_static` 10×; `resolve_q` 30.77 s; run ~45 s.
- `after_compare_wq.log` — AFTER: `_block` compiles 1×, `_map` 2×, `_roll_static`
  0×; `resolve_q` 4.60 s; run ~15 s. Per-q rel_err bit-identical to baseline.
- `gates_targeted.log` — `test_bse_w0_resolvent` + `w_omega_chain` + `stack` +
  `dense_reference` = 18 passed / 1 deselected (1 GPU).
- `full_suite_1gpu.log` — full plain 1-GPU `pytest -q`.

## Reproduce

```bash
export JID=<a free lx-alloc-jackm pool node>      # squeue -j <jid> -s : only .extern = free
FIX=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_w0_resolvent_2026-07-16/fixture_run
rm -rf .jax_cache_census                           # cold cache = honest census
./lxrun_census.sh "$FIX" python3 -u -m bse.bse_w_exact -i gnppm_test.in --compare-wq
```

Census: `grep -oE "Finished XLA compilation of [^ ]+ " <log> | sort | uniq -c`.
