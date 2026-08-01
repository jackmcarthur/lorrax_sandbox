# Known LORRAX issues — live register

One row per broken/deferred item in the main drivers. Append; mark FIXED with
the commit rather than deleting. Evidence = scorecard section or job id.
SUP = needs owner decision (physics convention or release call); everything
else is agent-actionable. Replaces the archived KNOWN_SANDBOX_ERRORS.

## centroids (kmeans_cli)
| issue | evidence | smallest fix | status |
|---|---|---|---|
| psi-at-centroids sharded on ONE mesh axis (sqrt(P) scaling) | BD.4; wfn_transforms.py:1933,2002 | both-axes mu-shard | open |
| weight field built replicated on full r-grid (7 s unscaled) | BD.4 | grid-shard like Lloyd | open |
| pair tensors col-blocked only on single-device path | BD.4; pivoted_cholesky.py:800 | extend col-blocking to multi-device | open |
| P>1 thread-main refusal | BD.3, job 7884867 | prepare_mesh routing | FIXED 24e4dc3 |

## dipole / kin-ion
| issue | evidence | smallest fix | status |
|---|---|---|---|
| replicated (nk,3,nb,nb)/(nk,nb,nb) gathers; only consumer is rank-0 write | BD.4; get_dipole_mtxels.py:912, kin_ion_io.py:453 | gather_k_blocks owner-write mode | open |
| full WFN.h5 h5py read PER RANK at P>1 | BD.2 banner | let auto pick phdf5 collective read | open |
| dipole anti-scales at small decks (0.8x at P=16) | BD.2 | startup amortization only; k-sweep itself scales | wontfix-small-decks |

## gw
| issue | evidence | smallest fix | status |
|---|---|---|---|
| zeta-fit factor: nq x mu^3 eigh on EVERY rank, zero P-scaling; 64% of GW wall at 2979c | BC, job 7884656 (cholesky 105.1 s) | q-parallel factor plan, auto above size threshold | open |
| chi0_W_probe re-runs chi+W nearly in full (~9.6 s) | BC | reuse the real pass | open |
| sigma.exec 88% d2h_wait — host accumulator not overlapped | BC | double-buffer the tau d2h | open |
| Sigma_c is PPM-only; no full-frequency cross-check | gw_config.py:306 | validation mode on bse/w_omega_chain.py | open |
| slab_io=auto -> PHDF5_FFI aborts in MPI_Init on a PMI-less bare launch (no srun) instead of demoting | CLAIMS 18, job 7884926; fastloop catch | auto probes MPI bootstrapability, announced demotion (GPU router precedent) | open |
| gw_jax hangs indefinitely in interpreter teardown after main() at bare P=1 (only FFT/GEMM-FFI chain driver) | CLAIMS 19, job 7884928; fastloop catch | root-cause FFI/XLA:CPU teardown; interim: fastloop GW_WRAPPER os._exit | open |

## htransform
| issue | evidence | smallest fix | status |
|---|---|---|---|
| replicated SVD family: A=(nk*nb, ns*N_mu), Vh, B_at_mu — last N_mu^2-replicated core in the chain | BD.4; htransform.py:262,356,626 | Gram-eigh of A A^H via ffi.linalg plan + mu-shard | open |
| bandstructure.dat written by EVERY rank (shared-FS race) | BD.4; htransform.py:1437,1570 | rank-0 writer gate | open |
| Gamma-label comparison right only by coincidence (norm-0 tie) | BE; htransform.py:1422 | compare canonical label | open |
| ~12 stray eager 1-op modules remain | BE, job 7884871 | jit when touched | low |
| P>1 thread-main refusal | BD.3 | prepare_mesh routing | FIXED 24e4dc3 |

## bse / exciton bands
| issue | evidence | smallest fix | status |
|---|---|---|---|
| ring/preview path h5py-reads FULL V_q/W0/psi per rank | BD.4; bse_io.py:1453-1470 | use the sharded loader already imported | open |
| psi stacks single-axis mu-sharded (sqrt(P)) | BD.4; bse_ring_comm.py:156, exciton_bands.py:241 | both-axes mu-shard | open |
| eager-FFT debt: bse_feast.ensure_W_R, bse_kpm.run_kpm_dos, bse_pseudopoles x2 (same P0-4 class, ratcheted so it cannot spread) | test_fft_shardmap_context.py allowlist | route through make_w_densifier | open |

## bispinor
| issue | evidence | smallest fix | status |
|---|---|---|---|
| Sigma^B = bare transverse exchange only (screened = phase 2) | BISPINOR_DHFB_DESIGN.md:168 | chi^ij + 4x4 Dyson + screened Sigma^B | open (headliner) |
| NO transverse rank gate / basis-adequacy measurement; LU+ridge with no truncation knob | isdf/core.py:1588; priorities-physics #2 | channel-aware prune gate + measured rcond policy | open; SUP for the policy choice |

## cross-cutting
| issue | evidence | smallest fix | status |
|---|---|---|---|
| psp.valence_density_from_kpoint: 17 eager modules, one jit away | BE | jit it (3 call paths — test all) | open |
| mu-convergence non-monotone at 512b (c4951 -> c6947: -240/-183 meV) | OWNER_DECISIONS | convergence-sweep driver + adequacy re-measure | open; SUP for the verdict |
| six mos2_4x4_test sbatch harnesses still carry the compile-cache opt-out | BE | delete ISDF_JAX_CACHE_DIR="" lines | open |
| QE->pw2bgw leg uncertified on Frontera | ASSERTIONS.md | one certified NSCF example (7884642 is the candidate) | open |
| head-wing-fix rescue patches exist ONLY on Perlmutter checkout | _archive/rescue_2026-07-22 | recover after outage | SUP (data risk) |
