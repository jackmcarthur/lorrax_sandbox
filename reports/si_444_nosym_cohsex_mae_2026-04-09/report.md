**Si 4x4x4 No-Sym COHSEX**
Run: `runs/Si/02_si_4x4x4_nosym/15_lorrax_cohsex_rerun_4gpu/`

Fresh Perlmutter rerun of the Si `4x4x4`, `ntran=1` COHSEX case on 1 node / 4 GPUs, using the existing no-symmetry QE/BGW reference data from `runs/Si/02_si_4x4x4_nosym/`.

**Inputs**
- LORRAX COHSEX: `runs/Si/02_si_4x4x4_nosym/15_lorrax_cohsex_rerun_4gpu/cohsex.in`
- BGW COHSEX reference: `runs/Si/02_si_4x4x4_nosym/00_bgw_cohsex/sigma_hp.log`
- LORRAX GN reference for `sigX`: `runs/Si/02_si_4x4x4_nosym/01_lorrax_gnppm/eqp0.dat`

**Runtime**
- Total recorded: `28.114 s`
- `gw_jax.zeta_fit_chunked`: `17.951 s`
- `gw_jax.V_q_compute`: `4.210 s`
- `gw_jax.chi0_W`: `3.149 s`
- `gw_jax.pipeline`: `1.794 s`

**Comparison Method**
- Used the documented parsers in `skills/compare/SKILL.md` and `PARSE_OUTPUTS.md`.
- Matched BGW symmetry-reduced k-points to the full GWJAX grid using `WFN.h5` crystal k-coordinates.
- Compared `970` matched `(k, band)` pairs spanning `64` k-points and `16` BGW sigma bands.

**MAE Results**
- `Sigma_X`: compare LORRAX GN `sigX` to BGW `X`
  - MAE: `0.047014931 eV` (`47.0 meV`)
  - max `|Δ|`: `0.084819240 eV`
- COHSEX correlation: compare `(sigSX - sigX) + sigCOH` to BGW `SX-X + CH'`
  - MAE: `0.006945646 eV` (`6.95 meV`)
  - max `|Δ|`: `0.033804942 eV`

**Additional Static Checks**
- `sigSX` vs BGW `SX = X + (SX-X)`: `0.054159191 eV` MAE
- `sigTOT` vs BGW `SX + CH'`: `0.051297969 eV` MAE

The main requested quantity for COHSEX agreement is the correlation-style comparison above, because `PARSE_OUTPUTS.md` defines the BGW-aligned static COHSEX target as `(sigSX - sigX) + sigCOH` against `SX-X + CH'`.

## 2026-04-12 output-format rerun

Run: `runs/Si/02_si_4x4x4_nosym/16_lorrax_cohsex_rerun_4gpu_repeat/`

Fresh rerun of the same 1-node / 4-GPU Si `4x4x4`, `ntran=1` COHSEX input, launched specifically to capture the current `gw.out` formatting after recent output-writing changes without overwriting variant `15`.

**Observed `gw.out` differences vs `15_lorrax_cohsex_rerun_4gpu`**
- The file now omits the leading `srun: Step created for StepId=...` line.
- Early boot/timing lines were simplified; the chunked-ISDF setup now prints a denser memory-summary block.
- Zeta/V_q progress now appears as progress-bar style status lines directly in `gw.out`.
- A new `STATIC HEAD TERMS (exact COHSEX / BGW-style)` block is present.
- Inline XLA SPMD rematerialization warnings are captured in the file.
- Output/restart paths in the footer are shown under `/pscratch/.../16_lorrax_cohsex_rerun_4gpu_repeat/`.

**Runtime**
- Total recorded: `26.661 s`
- `gw_jax.zeta_fit_chunked`: `18.437 s`
- `gw_jax.V_q_compute`: `4.179 s`
- `gw_jax.chi0_W`: `1.745 s`
- `gw_jax.sigma`: `1.349 s`

**Artifacts**
- `gw.out`: `140` lines, `13840` bytes
- `eqp0.dat`: present
- `qp_wfn_rotations.h5`: present
- `tmp/isdf_tensors_480.h5`: present

**Regression note**
- `eqp0.dat` from variant `16` is **not** byte-identical to variant `15` (`cmp` exit code `1`), so this rerun should be treated as a changed-output-format / changed-results checkpoint rather than a pure logging-only no-op.
