"""Algebraic test 1 — ψ unfold check for MoS2 3x3 SOC.

Compares for each (k_full, sym_idx):
  (a) LORRAX's WfnLoader.load(k='full_bz') unfold
  (b) hand-rolled reference from per-element math
  (c) nosym ψ at the same physical k_full (up to unitary in degenerate subspaces)

Hand-rolled reference, per pr3_design.md "correct rule" section:

  G_rot = sym_mats_k[s] @ G_kbar    (no umklapp subtracted here; we want
                                     the same axis basis as cnk_full
                                     returned by unfold_psi: cnk_full[b, σ, g]
                                     corresponds to G-vector sym_mats_k[s] @ g_kbar[g])
  phase = exp(-i (G_rot)·τ_{s_spatial})    (τ=0 for MoS2 so this is 1)
  is_trs = sym_idx >= ntran

  if is_trs:
      spinor_op = iσ_y · conj(U_spinor_spatial[s_spatial])
      inner = (cnk_kbar)* · phase_TRS    where phase_TRS = exp(+i (S·G_kbar)·τ)
                                          but sym_mats_k[TRS row] = -S so
                                          rotated_TRS = -S·G_kbar, then
                                          exp(-i rotated_TRS · τ) = exp(+i S·G_kbar·τ)
                                          which matches conj(spatial_phase).
  else:
      spinor_op = U_spinor_spatial[s_spatial]
      inner = cnk_kbar · phase

  cnk_full = einsum("ij,njg->nig", spinor_op, inner)

For τ=0 (MoS2), phases are 1 and the formula simplifies to:
  spatial: ψ_full(G_rot = S G) = U_s · ψ_kbar(G)
  TRS:    ψ_full(G_rot = -S G) = (iσ_y · conj(U_s)) · conj(ψ_kbar(G))
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
import time
import numpy as np
import h5py
import jax
jax.config.update("jax_enable_x64", True)

# Setup imports
SRC = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src"
sys.path.insert(0, SRC)

from file_io.wfn_loader import WfnLoader  # noqa: E402

# Material paths
RUN_DIR = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/06_sym_vs_nosym_pr3_2026-05-14"
SYM_WFN = f"{RUN_DIR}/run_sym/WFN.h5"
NOSYM_WFN = f"{RUN_DIR}/run_nosym/WFN.h5"

# i·σ_y in our convention (T = iσ_y K)
I_SIGMA_Y = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)


def hand_rolled_unfold_one_k(
    cnk_kbar,     # (nb, ns, ngk) — IBZ ψ at k=kbar
    g_kbar,       # (ngk, 3)      — IBZ G-list
    sym_idx,      # int
    n_sym_spatial,
    sym_mats_k,   # (2 ntran, 3, 3)
    translations, # (ntran, 3)
    U_spinor_spat,  # (ntran, 2, 2)
):
    """Return (cnk_full, G_rot) where cnk_full[b, σ, g] is ψ_full at G_rot[g].

    G_rot is on the same axis order as the IBZ G-list (rotated, no umklapp
    subtraction — caller aligns to physical full-BZ G-set).
    """
    s = int(sym_idx)
    n_t = int(n_sym_spatial)
    is_trs = s >= n_t
    s_spat = s - n_t if is_trs else s

    S_full = np.asarray(sym_mats_k[s], dtype=np.int64)
    g_bar = np.asarray(g_kbar, dtype=np.int64)
    # G axis of cnk_full: G_rot = sym_mats_k[s] @ g_bar  (per unfold_psi docstring).
    G_rot = (S_full @ g_bar.T).T.astype(np.int32)

    tau = np.asarray(translations[s_spat], dtype=np.float64)
    if np.any(np.abs(tau) > 1e-12):
        # exp(-i (rotated)·τ); rotated already has TRS sign baked in via sym_mats_k.
        phase = np.exp(-1j * (G_rot.astype(np.float64) @ tau))  # (ngk,)
    else:
        phase = None

    cnk = np.asarray(cnk_kbar, dtype=np.complex128)
    if is_trs:
        cnk = np.conj(cnk)
        if phase is not None:
            cnk = cnk * phase[None, None, :]
        U_eff = I_SIGMA_Y @ np.conj(np.asarray(U_spinor_spat[s_spat]))
    else:
        if phase is not None:
            cnk = cnk * phase[None, None, :]
        U_eff = np.asarray(U_spinor_spat[s_spat])

    cnk_full = np.einsum("ij,nje->nie", U_eff, cnk)
    return cnk_full, G_rot


def get_umklapp_kg0(k_full_frac, sym_krep, kbar_frac):
    """BGW kg0: k_full = S kbar + kg0 (integer)."""
    skbar = sym_krep @ np.asarray(kbar_frac, dtype=np.float64)
    kg0 = np.rint(np.asarray(k_full_frac, dtype=np.float64) - skbar).astype(np.int32)
    if not np.allclose(skbar + kg0, k_full_frac, atol=1e-6):
        raise ValueError(f"k_full={k_full_frac}, S·kbar={skbar}, kg0={kg0}")
    return kg0


def main():
    print("=" * 78, flush=True)
    print("Test 1 — ψ unfold algebraic check (MoS2 3x3 SOC)", flush=True)
    print("=" * 78, flush=True)

    # --- Load both WFN files via WfnLoader (eager backend) ----------------
    loader_sym = WfnLoader(SYM_WFN, backend="eager")
    loader_nosym = WfnLoader(NOSYM_WFN, backend="eager")
    sym = loader_sym._ensure_sym()
    sym_nosym = loader_nosym._ensure_sym()
    ntran = int(sym.sym_matrices.shape[0])
    nbands = int(loader_sym.nbands)
    nspinor = int(loader_sym.nspinor)
    ngkmax = int(loader_sym.ngkmax)
    n_k_full = int(sym.unfolded_kpts.shape[0])
    assert nbands == int(loader_nosym.nbands), f"nbands mismatch: sym={nbands} nosym={loader_nosym.nbands}"
    assert nspinor == int(loader_nosym.nspinor)

    print(f"ntran={ntran}, nbands={nbands}, nspinor={nspinor}, ngkmax={ngkmax}, n_k_full={n_k_full}", flush=True)
    print(f"sym sym_idx_k = {sym.sym_idx_k}", flush=True)
    print(f"sym irr_idx_k = {sym.irr_idx_k}", flush=True)
    print(f"nosym sym_idx_k (should be all 0) = {sym_nosym.sym_idx_k}", flush=True)

    # --- Test 3 (G-vector consistency check) — interleaved here -----------
    print("\n--- Test 3: G-vector full-BZ k-list consistency ---", flush=True)
    g_lorrax = loader_sym.gvecs(k="full_bz")           # (n_k_full, ngkmax, 3)
    g_nosym  = loader_nosym.gvecs(k="full_bz")         # (n_k_full, ngkmax, 3)
    ngk_lorrax = loader_sym.ngk_valid(k="full_bz")
    ngk_nosym  = loader_nosym.ngk_valid(k="full_bz")
    gvec_pass = True
    gvec_diffs = []
    for k_full in range(n_k_full):
        if int(ngk_lorrax[k_full]) != int(ngk_nosym[k_full]):
            gvec_pass = False
            gvec_diffs.append((k_full, "ngk", int(ngk_lorrax[k_full]), int(ngk_nosym[k_full])))
            continue
        n = int(ngk_lorrax[k_full])
        gs = g_lorrax[k_full, :n].copy()
        gn = g_nosym[k_full, :n].copy()
        # Sort lex order to compare sets
        gs_set = set(map(tuple, gs.tolist()))
        gn_set = set(map(tuple, gn.tolist()))
        if gs_set != gn_set:
            gvec_pass = False
            missing = (gs_set - gn_set) | (gn_set - gs_set)
            gvec_diffs.append((k_full, "set", len(missing)))
    print(f"  Test 3 verdict: {'PASS' if gvec_pass else 'FAIL'}", flush=True)
    if not gvec_pass:
        for d in gvec_diffs[:5]:
            print(f"    failure: k_full={d[0]}, kind={d[1]}, {d[2:]}", flush=True)

    # --- Now Test 1: ψ unfold -------------------------------------------
    print("\n--- Test 1: ψ unfold (per-k_full) ---", flush=True)
    # Load full set: LORRAX's full-BZ unfold, all bands, both spinor
    print("Loading ψ via LORRAX (all bands, full_bz)...", flush=True)
    t0 = time.time()
    psi_lorrax = np.asarray(loader_sym.load(bands=(0, nbands), k="full_bz"))
    # Shape: (n_k_full, nb_padded, nspinor, ngkmax). For eager bn_padded = nbands.
    print(f"  shape={psi_lorrax.shape}, dtype={psi_lorrax.dtype}, took {time.time()-t0:.2f}s", flush=True)

    # IBZ-only read of the same WFN — raw ψ_kbar without unfold
    print("Loading ψ via LORRAX (ibz)...", flush=True)
    psi_ibz = np.asarray(loader_sym.load(bands=(0, nbands), k="ibz"))
    print(f"  ibz shape={psi_ibz.shape}", flush=True)

    # nosym ψ at full BZ — for nosym ntran=1, full_bz = ibz; ψ at k_full is direct.
    psi_nosym_full = np.asarray(loader_nosym.load(bands=(0, nbands), k="full_bz"))
    print(f"  nosym full_bz shape={psi_nosym_full.shape}", flush=True)

    # nosym G-vectors at each k_full (= raw IBZ list since nosym IBZ = full BZ)
    g_nosym_full = loader_nosym.gvecs(k="full_bz")    # (n_k_full, ngkmax, 3)
    ngk_nosym_full = loader_nosym.ngk_valid(k="full_bz")

    # DFT eigenvalues (sym vs nosym) — for degenerate-subspace grouping
    el_sym = np.asarray(loader_sym.energies)        # (1, nrk, mnband)
    el_nosym = np.asarray(loader_nosym.energies)

    # We assume the band ordering matches between sym and nosym files at
    # the same physical k (same DFT pseudopotential & inputs).  Verify
    # eigenvalue alignment.

    # --- Loop over all full-BZ k -----------------------------------------
    worst_ab = []   # (a) LORRAX vs (b) hand-rolled
    worst_bc = []   # (b) hand-rolled vs (c) nosym (degenerate-subspace unitary)
    worst_unitary = []  # worst-case unitary error
    for k_full in range(n_k_full):
        s = int(sym.sym_idx_k[k_full])
        kbar = int(sym.irr_idx_k[k_full])
        ngk_k = int(loader_sym.ngk[kbar])
        kbar_frac = sym.unfolded_kpts[k_full] if False else np.asarray(loader_sym.kpoints[kbar])
        k_full_frac = sym.unfolded_kpts[k_full]
        is_trs = s >= ntran
        s_spat = s - ntran if is_trs else s

        # ψ_kbar from raw IBZ read: shape (nbands, nspinor, ngk_k)
        cnk_kbar = psi_ibz[kbar, :, :, :ngk_k]
        g_kbar = loader_sym._gvecs_raw[
            int(loader_sym._kpt_starts[kbar]) : int(loader_sym._kpt_starts[kbar]) + ngk_k]

        # (b) hand-rolled
        cnk_handroll, G_rot = hand_rolled_unfold_one_k(
            cnk_kbar, g_kbar, s, ntran,
            sym.sym_mats_k, sym.translations, sym.U_spinor)
        # Apply umklapp: G axis of LORRAX's unfold has G_lorrax = G_rot - kg0
        sym_krep_k = sym.sym_mats_k[s]
        kg0 = get_umklapp_kg0(k_full_frac, sym_krep_k, kbar_frac)
        G_handroll_final = G_rot - kg0[None, :]

        # (a) LORRAX's unfold output for this k
        cnk_lorrax_k = psi_lorrax[k_full, :, :, :ngk_k]   # (nbands, nspinor, ngk_k)
        # The G-axis of LORRAX is identical to G_handroll_final because
        # LORRAX's gvecs(k='full_bz') uses g_rot = S·g_bar - kg0.
        g_lorrax_k = g_lorrax[k_full, :ngk_k]

        # Sanity: G axes match position-by-position?
        if not np.array_equal(g_lorrax_k, G_handroll_final):
            # They might just be permuted — but for this implementation
            # they should match elementwise since both follow the same
            # iteration over the IBZ G-list.
            n_match = int(np.sum(np.all(g_lorrax_k == G_handroll_final, axis=-1)))
            raise RuntimeError(
                f"k_full={k_full} G-axis mismatch: only {n_match}/{ngk_k} entries match")

        # (a) vs (b) — bit-equal?
        diff_ab = np.abs(cnk_lorrax_k - cnk_handroll)
        max_ab = float(diff_ab.max())
        worst_ab.append((k_full, s, is_trs, max_ab))
        if max_ab > 1e-12:
            # find worst element
            arg = np.unravel_index(diff_ab.argmax(), diff_ab.shape)
            print(f"  k_full={k_full} (sym={s},trs={is_trs}): "
                  f"(a)LORRAX vs (b)handroll max diff = {max_ab:.3e} at "
                  f"band={arg[0]}, spinor={arg[1]}, g={arg[2]}", flush=True)

        # (c) nosym ψ at same physical k_full
        ngk_no = int(ngk_nosym_full[k_full])
        cnk_no = psi_nosym_full[k_full, :, :, :ngk_no]   # (nbands, nspinor, ngk_no)
        g_no = g_nosym_full[k_full, :ngk_no]

        # The G-axes may have different ORDER between sym-unfolded and nosym.
        # Build a permutation from g_lorrax_k (== G_handroll_final) → g_no.
        # If the SETS are equal (per Test 3), this permutation exists.
        if ngk_k != ngk_no:
            print(f"  k_full={k_full}: ngk_sym_unfold={ngk_k} vs ngk_nosym={ngk_no} — skipping")
            continue
        # Build mapping: idx_in_handroll[g] -> position in nosym g-list
        # i.e. nosym_to_handroll[i_no] = j_h such that g_no[i_no] == g_handroll_final[j_h]
        # Easier: re-sort both onto a common canonical lex order.
        # Use a dict keyed by tuple-of-3 ints.
        no_set = {tuple(g_no[i].tolist()): i for i in range(ngk_no)}
        h_to_n = np.empty(ngk_k, dtype=np.int32)
        ok = True
        for j in range(ngk_k):
            key = tuple(G_handroll_final[j].tolist())
            if key not in no_set:
                ok = False
                break
            h_to_n[j] = no_set[key]
        if not ok:
            print(f"  k_full={k_full}: G-set mismatch sym vs nosym")
            continue

        # Reorder nosym ψ onto the handroll G-axis
        cnk_no_aligned = cnk_no[:, :, np.argsort(h_to_n)]
        # NOTE: argsort gives nosym indices in order of handroll's g list
        # so cnk_no_aligned[b, σ, j] = cnk_no[b, σ, h_to_n[j]] ... but
        # argsort(h_to_n) returns indices that sort h_to_n. Let me redo:
        # we want cnk_no_aligned[j] = cnk_no[h_to_n[j]] for j in 0..ngk_k-1.
        cnk_no_aligned = cnk_no[:, :, h_to_n]

        # Per-degenerate-group overlap check
        # Group by eigenvalue tolerance — use sym ψ_kbar's energies? Same
        # physical k means el_sym[0, kbar, b] = el_nosym[0, k_full, b].
        # Use nosym eigenvalues for grouping.
        E = el_nosym[0, k_full, :nbands]
        # Find groups of degenerate / near-degenerate bands. Use 0.005 Ry =
        # 68 meV: independent SCFs differ by ~0.1 meV per eigenvalue, so
        # bands within 68 meV of each other can mix gauge-wise to a level
        # comparable to the independent-SCF disagreement. A correct unfold
        # must match the nosym ψ within this subspace.
        tol_E = 5e-3  # Ry units (el is in Ry per BGW convention)
        groups = []
        b_start = 0
        for b in range(1, nbands):
            if abs(E[b] - E[b-1]) > tol_E:
                groups.append((b_start, b))
                b_start = b
        groups.append((b_start, nbands))

        worst_group_err = 0.0
        worst_group_info = None
        for (g_lo, g_hi) in groups:
            n_deg = g_hi - g_lo
            if n_deg == 0:
                continue
            # Build U = <handroll | nosym>_G  for this band group.
            # shape: (n_deg, n_deg)
            ket_hand = cnk_handroll[g_lo:g_hi, :, :]      # (n_deg, nspinor, ngk)
            ket_nos = cnk_no_aligned[g_lo:g_hi, :, :]
            ket_hand_flat = ket_hand.reshape(n_deg, -1)
            ket_nos_flat = ket_nos.reshape(n_deg, -1)
            U = ket_hand_flat.conj() @ ket_nos_flat.T   # (n_deg, n_deg)
            # Unitary check
            I = U @ U.conj().T
            err = float(np.max(np.abs(I - np.eye(n_deg))))
            # |U|² row sums = 1
            row_sum = np.abs(U) ** 2
            row_err = float(np.max(np.abs(row_sum.sum(axis=1) - 1.0)))
            tot_err = max(err, row_err)
            if tot_err > worst_group_err:
                worst_group_err = tot_err
                worst_group_info = (g_lo, g_hi, err, row_err)
        worst_bc.append((k_full, s, is_trs, worst_group_err, worst_group_info))

    # --- Summarize -------------------------------------------------------
    print("\n=== Test 1 summary ===", flush=True)
    print("k_full sym_idx is_trs   |a-b|_max         (a)LORRAX vs (b)handroll", flush=True)
    for k_full, s, is_trs, m in worst_ab:
        print(f"  {k_full:3d}    {s:3d}    {int(is_trs)}     {m:.3e}", flush=True)
    print("\nk_full sym_idx is_trs   max|UU†-I|       (b)handroll vs (c)nosym (per degen group)", flush=True)
    for k_full, s, is_trs, err, info in worst_bc:
        info_str = f"group=[{info[0]},{info[1]})" if info else ""
        print(f"  {k_full:3d}    {s:3d}    {int(is_trs)}     {err:.3e}      {info_str}", flush=True)

    max_ab_overall = max(m for _, _, _, m in worst_ab)
    max_bc_overall = max(err for _, _, _, err, _ in worst_bc)
    verdict_ab = "PASS" if max_ab_overall < 1e-10 else "FAIL"
    verdict_bc = "PASS" if max_bc_overall < 1e-10 else "FAIL"
    print(f"\n  (a) vs (b) verdict: {verdict_ab}, max = {max_ab_overall:.3e}", flush=True)
    print(f"  (b) vs (c) verdict: {verdict_bc}, max = {max_bc_overall:.3e}", flush=True)

    # Worst-case info
    worst_bc_sorted = sorted(worst_bc, key=lambda x: -x[3])
    print("\n  Worst (b) vs (c):", flush=True)
    for k_full, s, is_trs, err, info in worst_bc_sorted[:3]:
        info_str = f"group=[{info[0]},{info[1]})" if info else ""
        print(f"    k_full={k_full} sym={s} trs={int(is_trs)} err={err:.3e} {info_str}", flush=True)


if __name__ == "__main__":
    main()
