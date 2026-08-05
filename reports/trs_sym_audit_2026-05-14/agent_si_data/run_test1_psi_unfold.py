"""Si Fd-3m algebraic ψ unfold audit — v2 (kg0-aware).

Three independent ψ at full-BZ k:
  (a) LORRAX WfnLoader.load(k='full_bz') (calls unfold_psi internally).
  (b) Hand-rolled reference: independent reconstruction from WFN.h5 header
      (G-rotation = mtrx[s].T · G_irr - kg0; τ-phase = exp(-i (mtrx[s].T·G_irr)·τ);
      spinor rotation = U_spinor_spatial[s] (TRS-augmented: iσ_y·conj on top)).
  (c) Nosym WFN at the matching full-BZ k.

We scatter all three onto a common FFT-box (g mod fft_grid → linear index)
and compare on the box grid. For (a) vs (b): expect bit-equality (LORRAX
matches hand-rolled). For (b) vs (c): expect agreement up to unitary gauge
within each degenerate band group (independent SCF runs).

Categories tested: identity, proper-symmorphic, proper-non-symmorphic,
improper-symmorphic, improper-non-symmorphic, TRS (if any).

Output JSON + summary stdout.
"""

from __future__ import annotations
import sys, os, json
import numpy as np
import h5py

LORRAX_SRC = "/global/u2/j/jackm/software/lorrax_B/src"
sys.path.insert(0, LORRAX_SRC)

import jax
jax.config.update("jax_enable_x64", True)

from common.symmetry_maps import SymMaps, unfold_psi
from file_io.wfn_loader import WfnLoader


SYM_WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/05_si_4x4x4_sym/qe/nscf/WFN.h5"
NOSYM_WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5"


def build_match_table(kpts_a, kpts_b, tol=1e-6):
    """For each kpts_a[i], return j such that kpts_b[j] matches modulo 1; -1 if none."""
    a = np.asarray(kpts_a, dtype=np.float64)
    b = np.asarray(kpts_b, dtype=np.float64)
    out = np.full(a.shape[0], -1, dtype=np.int64)
    for i in range(a.shape[0]):
        diff = b - a[i][None, :]
        diff_w = diff - np.round(diff)
        norms = np.max(np.abs(diff_w), axis=1)
        j = int(np.argmin(norms))
        if norms[j] < tol:
            out[i] = j
    return out


def g_to_box_flat(gvecs, fft_grid):
    """G (..., 3) int → flat box index (Nx*Ny*Nz wrap)."""
    Nx, Ny, Nz = [int(x) for x in fft_grid]
    g = np.asarray(gvecs, dtype=np.int64)
    return ((g[..., 0] % Nx) * Ny + (g[..., 1] % Ny)) * Nz + (g[..., 2] % Nz)


def scatter_to_box(psi, gvecs_for_psi, ngk_valid, fft_grid):
    """psi (nb, ns, ngkmax) → psi_box (nb, ns, NxNyNz) c128."""
    nb, ns, _ = psi.shape
    Nx, Ny, Nz = [int(x) for x in fft_grid]
    Nbox = Nx * Ny * Nz
    box = np.zeros((nb, ns, Nbox), dtype=np.complex128)
    bidx = g_to_box_flat(gvecs_for_psi[:ngk_valid], fft_grid)
    box[:, :, bidx] = psi[:, :, :ngk_valid]
    return box


def hand_unfold_psi(psi_kbar, gvecs_kbar, sym_idx, sym, translations, ntran, kg0):
    """Hand-rolled reference for ψ at full-BZ k.

    Returns
    -------
    psi_full : (nb, ns, ngk_kbar) — coefficients
    g_full : (ngk_kbar, 3) int — G_full labels (i.e. mtrx_k @ G_irr - kg0)
    """
    sym_idx = int(sym_idx)
    is_trs = sym_idx >= ntran
    s_spatial = sym_idx - ntran if is_trs else sym_idx

    # G rotation (LORRAX convention: sym_mats_k acts row-wise on G).
    S_full = np.asarray(sym.sym_mats_k[sym_idx], dtype=np.int64)
    g_rot_raw = np.einsum('ij,kj->ki', S_full, np.asarray(gvecs_kbar, dtype=np.int64))
    g_full = g_rot_raw - np.asarray(kg0, dtype=np.int64)[None, :]

    # τ-phase: uses S G_kbar (LORRAX convention; equals g_rot_raw).
    tau = np.asarray(translations[s_spatial], dtype=np.float64)
    phase = np.exp(-1j * (g_rot_raw.astype(np.float64) @ tau))   # (ngk,)

    # Spinor rotation
    U_s = np.asarray(sym.U_spinor[s_spatial], dtype=np.complex128)
    psi_kbar_c = np.asarray(psi_kbar, dtype=np.complex128)
    if is_trs:
        I_SIGMA_Y = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)
        out_spatial = np.einsum('ab,nbk->nak', U_s, psi_kbar_c) * phase[None, None, :]
        psi_full = np.einsum('ab,nbk->nak', I_SIGMA_Y, np.conj(out_spatial))
    else:
        psi_full = np.einsum('ab,nbk->nak', U_s, psi_kbar_c) * phase[None, None, :]

    return psi_full, g_full


def compare_subspace(box_x, box_y, e_kirr, tol=5e-6):
    """For each degenerate band group: compute unitarity-deviation of
    U = <x|y> (= box_x @ conj(box_y).T) and gauge-fixed residual."""
    nb = box_x.shape[0]
    # Group bands by energy.
    groups = []
    i = 0
    while i < nb:
        j = i + 1
        while j < nb and abs(e_kirr[j] - e_kirr[i]) < tol:
            j += 1
        groups.append((i, j))
        i = j

    max_unitarity_dev = 0.0
    max_gauge_resid = 0.0
    grp_results = []
    for (lo, hi) in groups:
        ng = hi - lo
        X = box_x[lo:hi].reshape(ng, -1)
        Y = box_y[lo:hi].reshape(ng, -1)
        Nx = np.linalg.norm(X, axis=1)
        Ny = np.linalg.norm(Y, axis=1)
        # Normalize (avoid renormalising zero vectors)
        Xn = np.where(Nx[:, None] > 0, X / np.where(Nx[:, None] == 0, 1, Nx[:, None]), X)
        Yn = np.where(Ny[:, None] > 0, Y / np.where(Ny[:, None] == 0, 1, Ny[:, None]), Y)
        U = Xn @ np.conj(Yn).T   # (ng, ng)
        unitarity_dev = float(np.linalg.norm(U @ np.conj(U.T) - np.eye(ng)))
        # gauge-fixed residual: X projected via U onto Y subspace
        # If U is the right gauge fix, X ≈ U @ Yn  for normalised X,Y. Take
        # residual = || Xn - U @ Yn ||_F (over all bands in group).
        Yproj = U @ Yn
        gauge_resid = float(np.linalg.norm(Xn - Yproj))
        max_unitarity_dev = max(max_unitarity_dev, unitarity_dev)
        max_gauge_resid = max(max_gauge_resid, gauge_resid)
        grp_results.append({
            "lo": int(lo), "hi": int(hi),
            "unitarity_dev": unitarity_dev,
            "gauge_resid": gauge_resid,
            "Nx": Nx.tolist(),
            "Ny": Ny.tolist(),
        })
    return {"max_unitarity_dev": max_unitarity_dev,
            "max_gauge_resid": max_gauge_resid,
            "groups": grp_results}


def main():
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    out_dir = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/trs_sym_audit_2026-05-14/agent_si_data"
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 0 — non-symmorphic structure
    # ------------------------------------------------------------------
    with h5py.File(SYM_WFN, "r") as f:
        sym_ntran = int(f["mf_header/symmetry/ntran"][()])
        sym_mtrx = f["mf_header/symmetry/mtrx"][:]
        sym_tnp = f["mf_header/symmetry/tnp"][:]      # 2π·τ_frac

    tau_frac_max = np.array([
        np.max(np.abs(((sym_tnp[s] / (2 * np.pi) + 0.5) % 1.0) - 0.5))
        for s in range(sym_ntran)
    ])
    is_nonsymm = tau_frac_max > 1e-6
    n_nonsymm = int(np.sum(is_nonsymm))
    dets = np.linalg.det(sym_mtrx.astype(float)).round().astype(int)

    print(f"=== Step 0: non-symmorphic ===")
    print(f"  ntran = {sym_ntran}")
    print(f"  non-symmorphic ops (|τ_frac| > 1e-6): {n_nonsymm}/{sym_ntran}")
    print(f"  det distribution: {np.unique(dets, return_counts=True)}")
    assert n_nonsymm == 36, f"expected 36 non-symmorphic ops, got {n_nonsymm}"
    print(f"  Step 0 PASS")
    print()

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------
    loader_sym = WfnLoader(SYM_WFN)
    loader_nos = WfnLoader(NOSYM_WFN)
    sym = loader_sym._ensure_sym()
    fft_grid = np.asarray(loader_sym.fft_grid, dtype=np.int64)
    ntran = sym_ntran
    unfolded_kpts = np.asarray(sym.unfolded_kpts, dtype=np.float64)
    irr_idx_k = np.asarray(sym.irr_idx_k, dtype=np.int64)
    sym_idx_k = np.asarray(sym.sym_idx_k, dtype=np.int64)
    nk_full = unfolded_kpts.shape[0]

    nosym_kpts = np.asarray(loader_nos.kpoints, dtype=np.float64)
    nos_match = build_match_table(unfolded_kpts, nosym_kpts)
    assert (nos_match >= 0).all(), f"unmatched k: {np.where(nos_match < 0)[0]}"

    used_syms = np.unique(sym_idx_k)
    cats: dict[str, list[int]] = {"identity": [], "proper_symm": [], "proper_nonsymm": [],
                                   "improper_nonsymm": [], "improper_symm": [], "trs": []}
    for s in used_syms:
        s_int = int(s)
        if s_int >= ntran:
            cats["trs"].append(s_int)
            continue
        d = int(dets[s_int]); nonsymm = bool(is_nonsymm[s_int])
        if s_int == 0:
            cats["identity"].append(s_int)
        elif d > 0 and not nonsymm:
            cats["proper_symm"].append(s_int)
        elif d > 0 and nonsymm:
            cats["proper_nonsymm"].append(s_int)
        elif d < 0 and not nonsymm:
            cats["improper_symm"].append(s_int)
        elif d < 0 and nonsymm:
            cats["improper_nonsymm"].append(s_int)
    print("=== Categories of exercised sym_idx ===")
    for cat, lst in cats.items():
        print(f"  {cat:20s}: {len(lst)} ops, sample: {lst[:5]}")
    print()

    # Build test pairs: pick FIRST 2 sym_idx per category, FIRST 2 k_full per sym_idx
    test_records = []
    for cat, lst in cats.items():
        for s_pick in lst[:2]:
            ks_pick = np.where(sym_idx_k == s_pick)[0]
            for kf in ks_pick[:2]:
                test_records.append({"category": cat, "k_full": int(kf),
                                     "sym_idx": int(s_pick),
                                     "k_irr": int(irr_idx_k[kf])})
    # Add a few more from improper_symm/non-symm if we have them
    print(f"  built {len(test_records)} (k_full, sym_idx) test pairs")
    for tr in test_records:
        print(f"    {tr}")
    print()

    # ------------------------------------------------------------------
    # Bands window
    # ------------------------------------------------------------------
    b_lo, b_hi = 0, 16
    nb_window = b_hi - b_lo
    ns = int(loader_sym.nspinor)

    # IBZ raw data
    ngk_irr = np.asarray(loader_sym.ngk, dtype=np.int64)
    kpt_starts_irr = np.zeros(loader_sym.nkpts, dtype=np.int64)
    for ik in range(1, loader_sym.nkpts):
        kpt_starts_irr[ik] = kpt_starts_irr[ik - 1] + int(ngk_irr[ik - 1])
    coeffs_raw_irr = loader_sym._coeffs_raw
    gvecs_raw_irr = loader_sym._gvecs_raw

    # Nosym raw data
    ngk_nos = np.asarray(loader_nos.ngk, dtype=np.int64)
    kpt_starts_nos = np.zeros(loader_nos.nkpts, dtype=np.int64)
    for ik in range(1, loader_nos.nkpts):
        kpt_starts_nos[ik] = kpt_starts_nos[ik - 1] + int(ngk_nos[ik - 1])
    coeffs_raw_nos = loader_nos._coeffs_raw
    gvecs_raw_nos = loader_nos._gvecs_raw

    # LORRAX unfold path (entire full-BZ)
    print("=== LORRAX full-BZ load ===")
    psi_lorrax_full = np.asarray(loader_sym.load(bands=(b_lo, b_hi), k='full_bz'))
    g_lorrax_full = np.asarray(loader_sym.gvecs(k='full_bz'))
    ngk_lorrax_full = np.asarray(loader_sym.ngk_valid(k='full_bz'))
    print(f"  shape ψ_lorrax_full: {psi_lorrax_full.shape}")

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    results = []
    for tr in test_records:
        kf = tr["k_full"]
        k_irr = tr["k_irr"]
        s_idx = tr["sym_idx"]
        cat = tr["category"]
        s_spatial = s_idx - ntran if s_idx >= ntran else s_idx

        sym_krep = np.asarray(sym.sym_mats_k[s_idx], dtype=np.int64)
        k_full = unfolded_kpts[kf]
        skbar = sym_krep @ np.asarray(loader_sym.kpoints[k_irr])
        kg0 = np.rint(k_full - skbar).astype(np.int64)

        # (a) LORRAX
        ngk_a = int(ngk_lorrax_full[kf])
        psi_a = psi_lorrax_full[kf, :nb_window, :, :ngk_a]
        gv_a = g_lorrax_full[kf, :ngk_a]

        # (b) Hand-rolled
        start = int(kpt_starts_irr[k_irr])
        end = start + int(ngk_irr[k_irr])
        raw = coeffs_raw_irr[b_lo:b_hi, :, start:end, :]
        psi_kbar = raw[..., 0] + 1j * raw[..., 1]
        gvecs_kbar = gvecs_raw_irr[start:end]
        psi_b, gv_b = hand_unfold_psi(psi_kbar, gvecs_kbar, s_idx, sym,
                                       loader_sym.translations, ntran, kg0)
        ngk_b = psi_b.shape[-1]

        # (c) Nosym
        kn = int(nos_match[kf])
        start = int(kpt_starts_nos[kn])
        end = start + int(ngk_nos[kn])
        raw = coeffs_raw_nos[b_lo:b_hi, :, start:end, :]
        psi_c = raw[..., 0] + 1j * raw[..., 1]
        gv_c = gvecs_raw_nos[start:end]
        ngk_c = gv_c.shape[0]

        # Verify (a) and (b) live on the same G-set
        ga_set = set(map(tuple, gv_a.tolist()))
        gb_set = set(map(tuple, gv_b.tolist()))
        gc_set = set(map(tuple, gv_c.tolist()))
        a_vs_b_g = (len(ga_set & gb_set), len(ga_set ^ gb_set))
        a_vs_c_g = (len(ga_set & gc_set), len(ga_set ^ gc_set))

        # Box scatter for fair comparison
        box_a = scatter_to_box(psi_a, gv_a, ngk_a, fft_grid)
        box_b = scatter_to_box(psi_b, gv_b, ngk_b, fft_grid)
        box_c = scatter_to_box(psi_c, gv_c, ngk_c, fft_grid)

        # Sanity norms
        n_a = float(np.max(np.linalg.norm(box_a.reshape(nb_window, -1), axis=1)))
        n_b = float(np.max(np.linalg.norm(box_b.reshape(nb_window, -1), axis=1)))
        n_c = float(np.max(np.linalg.norm(box_c.reshape(nb_window, -1), axis=1)))

        # Raw differences
        ab_max = float(np.max(np.abs(box_a - box_b)))
        bc_max = float(np.max(np.abs(box_b - box_c)))
        ac_max = float(np.max(np.abs(box_a - box_c)))

        # Degenerate-subspace comparison vs nosym
        e_kirr = np.asarray(loader_sym.energies[0, k_irr, b_lo:b_hi])
        cmp_ab = compare_subspace(box_a, box_b, e_kirr)
        cmp_ac = compare_subspace(box_a, box_c, e_kirr)
        cmp_bc = compare_subspace(box_b, box_c, e_kirr)

        rec = {
            "category": cat, "k_full": kf, "k_irr": k_irr, "sym_idx": s_idx,
            "kg0": kg0.tolist(),
            "is_nonsymm": bool(is_nonsymm[s_spatial]),
            "det": int(dets[s_spatial]),
            "is_trs": bool(s_idx >= ntran),
            "n_ngk_a": ngk_a, "n_ngk_b": ngk_b, "n_ngk_c": ngk_c,
            "g_intersect_ab": a_vs_b_g[0], "g_symdiff_ab": a_vs_b_g[1],
            "g_intersect_ac": a_vs_c_g[0], "g_symdiff_ac": a_vs_c_g[1],
            "max_norm_a": n_a, "max_norm_b": n_b, "max_norm_c": n_c,
            "raw_max_ab": ab_max,
            "raw_max_bc": bc_max,
            "raw_max_ac": ac_max,
            "ab_unitarity_dev": cmp_ab["max_unitarity_dev"],
            "ab_gauge_resid": cmp_ab["max_gauge_resid"],
            "ac_unitarity_dev": cmp_ac["max_unitarity_dev"],
            "ac_gauge_resid": cmp_ac["max_gauge_resid"],
            "bc_unitarity_dev": cmp_bc["max_unitarity_dev"],
            "bc_gauge_resid": cmp_bc["max_gauge_resid"],
        }
        results.append(rec)

        flag_ab = "PASS" if rec["raw_max_ab"] < 1e-10 else ("MARG" if rec["raw_max_ab"] < 1e-6 else "FAIL")
        flag_ac = "PASS" if rec["ac_unitarity_dev"] < 1e-10 and rec["ac_gauge_resid"] < 1e-6 else (
            "MARG" if rec["ac_unitarity_dev"] < 1e-6 else "FAIL")
        flag_bc = "PASS" if rec["bc_unitarity_dev"] < 1e-10 and rec["bc_gauge_resid"] < 1e-6 else (
            "MARG" if rec["bc_unitarity_dev"] < 1e-6 else "FAIL")
        print(f"  ({cat:20s}) k_f={kf:2d}  k_irr={k_irr} s={s_idx:2d}  "
              f"det={rec['det']:+d}  τ≠0={rec['is_nonsymm']!s:5}  TRS={rec['is_trs']!s:5} kg0={kg0}")
        print(f"    G-intersect a∩b: {a_vs_b_g[0]}/{ngk_a} (symdiff {a_vs_b_g[1]});  a∩c: {a_vs_c_g[0]}/{ngk_a} (symdiff {a_vs_c_g[1]})")
        print(f"    raw   max|Δ| : a-b={rec['raw_max_ab']:.3e}  b-c={rec['raw_max_bc']:.3e}  a-c={rec['raw_max_ac']:.3e}")
        print(f"    unit dev    : a-b={rec['ab_unitarity_dev']:.3e}  a-c={rec['ac_unitarity_dev']:.3e}  b-c={rec['bc_unitarity_dev']:.3e}")
        print(f"    gauge resid : a-b={rec['ab_gauge_resid']:.3e}  a-c={rec['ac_gauge_resid']:.3e}  b-c={rec['bc_gauge_resid']:.3e}")
        print(f"    norms max   : a={n_a:.4f}  b={n_b:.4f}  c={n_c:.4f}  [{flag_ab}/{flag_ac}/{flag_bc}]")

    with open(os.path.join(out_dir, "test1_psi_unfold.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {len(results)} records.")

    # Aggregate summary
    print("\n=== Aggregate summary ===")
    by_cat = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, lst in by_cat.items():
        max_ab = max(r["raw_max_ab"] for r in lst)
        max_ac = max(r["ac_unitarity_dev"] for r in lst)
        max_bc = max(r["bc_unitarity_dev"] for r in lst)
        max_ac_g = max(r["ac_gauge_resid"] for r in lst)
        max_bc_g = max(r["bc_gauge_resid"] for r in lst)
        print(f"  {cat:20s}: a-b raw max = {max_ab:.3e}   |   "
              f"a-c unit_dev = {max_ac:.3e} gauge_resid = {max_ac_g:.3e}   |   "
              f"b-c unit_dev = {max_bc:.3e} gauge_resid = {max_bc_g:.3e}")


if __name__ == "__main__":
    main()
