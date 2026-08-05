# qe_sym_clean — sanity probe (not used in main comparison)

Built to test whether removing `noinv=.true.` from `nscf.in` would
produce a smaller (more standard) k-point set than the original sym
variant (`runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5`,
which has `no_t_rev=true, noinv=true` and nrk=48).

**Finding**: even with only `no_t_rev=true` (no `noinv`, no `nosym`),
QE/pw2bgw still writes a 48-k-point WFN for this CrI3 system, with
ntran=6 spatial ops including inversion. This is a property of the
BGW WFN format for inversion-containing systems — `nrk` reflects the
expanded k-list, NOT the standard 36-point full-BZ grid.

Conclusion: the sym WFN's nrk=48 vs LORRAX's 36-point unfolded
k-list mismatch is not avoidable by tweaking QE flags. It needs to
be fixed on the LORRAX side (use `meta.nkpts_unfolded` not
`wfn.nkpts` in `qp_wfn.write_qp_wfn_h5` and `gw_output.write_results`).

This directory was not used in the run_sym / run_nosym comparison.
Kept as evidence of the probe.
