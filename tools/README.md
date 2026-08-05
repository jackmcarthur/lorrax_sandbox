# tools/

Working analysis tools. Anything here must run today; broken tools get
fixed or deleted, not kept.

| Tool | Purpose | Runs on |
|---|---|---|
| `hlo/analyze_hlo_dump.py` | Summarize an `--xla_dump_to` dir: memory, collectives, layout boundaries, remat, retraces; `--forbid` gate for gather-class ops. See `docs/HLO_HOWTO.md`. | login python3 (stdlib only) |
| `compare_bgw_gwjax.py` | Multi-k BGW `sigma_hp.log` vs LORRAX `eqp0.dat` comparison, matching k-points via `WFN.h5`. See `skills/compare/SKILL.md`. | in-container (numpy/h5py/matplotlib) |
