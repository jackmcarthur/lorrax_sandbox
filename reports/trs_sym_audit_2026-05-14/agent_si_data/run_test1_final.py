"""Si Fd-3m ψ unfold audit — final.

Runs ψ-unfold three ways for every used (k_full, sym_idx):
  (a) LORRAX WfnLoader.load(k='full_bz')         (current SymMaps convention)
  (b1) Hand-rolled using LORRAX's U_spinor        (= 'A' convention: bvecinv·sym_mats_k·bvec)
  (b2) Hand-rolled using CORRECTED U_spinor       (= 'C' convention: bvecinv·mtrx·bvec)
  (c)  Nosym WFN at the matching full-BZ k

We expect:
  (a) == (b1): bit-equal (b1 is the from-scratch reference for LORRAX's choice)
  (b1) vs (c): gauge_resid SHOULD be sub-microvolt; current LORRAX shows large
              residual on proper_nonsymm and proper_symm operations.
  (b2) vs (c): gauge_resid should be sub-microvolt across the board (this is
              the BGW convention — verify it's the fix).

All in box-scatter on the WFN.h5 fft_grid.
"""
import sys, os, json
sys.path.insert(0, "/global/u2/j/jackm/software/lorrax_B/src")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import h5py
from file_io.wfn_loader import WfnLoader
from common.symmetry_maps import unfold_psi


SYM_WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/05_si_4x4x4_sym/qe/nscf/WFN.h5"
NOSYM_WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5"
OUT_DIR = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/trs_sym_audit_2026-05-14/agent_si_data"


def get_su2_from_R_cart(R_cart):
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    R = R_cart.copy()
    if np.linalg.det(R) < 0:
        R = -R
    Q = np.zeros((4, 4))
    Q[0, 0] = R[0, 0] + R[1, 1] + R[2, 2]
    Q[0, 1] = Q[1, 0] = R[1, 2] - R[2, 1]
    Q[0, 2] = Q[2, 0] = R[2, 0] - R[0, 2]
    Q[0, 3] = Q[3, 0] = R[0, 1] - R[1, 0]
    Q[1, 1] = R[0, 0] - R[1, 1] - R[2, 2]
    Q[1, 2] = Q[2, 1] = R[0, 1] + R[1, 0]
    Q[1, 3] = Q[3, 1] = R[0, 2] + R[2, 0]
    Q[2, 2] = -R[0, 0] + R[1, 1] - R[2, 2]
    Q[2, 3] = Q[3, 2] = R[1, 2] + R[2, 1]
    Q[3, 3] = -R[0, 0] - R[1, 1] + R[2, 2]
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    q = eigenvectors[:, np.argmax(eigenvalues)]
    q = q / np.linalg.norm(q)
    q0, q1, q2, q3 = q
    theta = 2 * np.arccos(np.clip(q0, -1.0, 1.0))
    sin_half = np.sqrt(max(0.0, 1.0 - q0**2))
    if sin_half < 1e-8:
        theta = 0.0
        n = np.array([1.0, 0.0, 0.0])
    elif np.isclose(theta, np.pi):
        axis = np.array([q1, q2, q3])
        n = axis / np.linalg.norm(axis)
    else:
        n = np.array([q1, q2, q3]) / sin_half
        n = n / np.linalg.norm(n)
    spinor = np.cos(theta / 2) * np.eye(2, dtype=complex)
    spinor -= 1j * np.sin(theta / 2) * (n[0] * sigma_x + n[1] * sigma_y + n[2] * sigma_z)
    return spinor


def g_to_box(g, fft):
    Nx, Ny, Nz = fft
    g = np.asarray(g, dtype=np.int64)
    return ((g[..., 0] % Nx) * Ny + (g[..., 1] % Ny)) * Nz + (g[..., 2] % Nz)


def scatter(psi, gv, ngk_v, fft):
    nb, ns, _ = psi.shape
    Nx, Ny, Nz = fft
    box = np.zeros((nb, ns, Nx * Ny * Nz), dtype=np.complex128)
    bidx = g_to_box(gv[:ngk_v], (Nx, Ny, Nz))
    box[:, :, bidx] = psi[:, :, :ngk_v]
    return box


def compare(box_x, box_y, e_kirr, tol=5e-6):
    nb = box_x.shape[0]
    groups = []
    i = 0
    while i < nb:
        j = i + 1
        while j < nb and abs(e_kirr[j] - e_kirr[i]) < tol:
            j += 1
        groups.append((i, j)); i = j
    max_unit = 0.0; max_gauge = 0.0
    for (lo, hi) in groups:
        ng = hi - lo
        X = box_x[lo:hi].reshape(ng, -1)
        Y = box_y[lo:hi].reshape(ng, -1)
        Nx_ = np.linalg.norm(X, axis=1); Ny_ = np.linalg.norm(Y, axis=1)
        Xn = X / np.where(Nx_[:, None] == 0, 1, Nx_[:, None])
        Yn = Y / np.where(Ny_[:, None] == 0, 1, Ny_[:, None])
        U = Xn @ np.conj(Yn).T
        max_unit = max(max_unit, float(np.linalg.norm(U @ np.conj(U.T) - np.eye(ng))))
        max_gauge = max(max_gauge, float(np.linalg.norm(Xn - U @ Yn)))
    return max_unit, max_gauge


def build_match_table(kpts_a, kpts_b, tol=1e-6):
    a = np.asarray(kpts_a, dtype=np.float64); b = np.asarray(kpts_b, dtype=np.float64)
    out = np.full(a.shape[0], -1, dtype=np.int64)
    for i in range(a.shape[0]):
        diff = b - a[i][None, :]; diff_w = diff - np.round(diff)
        norms = np.max(np.abs(diff_w), axis=1)
        j = int(np.argmin(norms))
        if norms[j] < tol: out[i] = j
    return out


def hand_unfold(psi_kbar, g_kbar, s_idx, sym_mats_k, translations, U_spinor_spatial,
                ntran, kg0):
    """Pure hand-rolled, given an explicit U_spinor_spatial table.

    NOT a wrapper of unfold_psi. Reproduces the math in pr3_design.md from first principles.
    """
    is_trs = s_idx >= ntran
    s_sp = s_idx - ntran if is_trs else s_idx
    S = np.asarray(sym_mats_k[s_idx], dtype=np.int64)
    g_rot_raw = np.einsum('ij,kj->ki', S, g_kbar.astype(np.int64))
    g_full = g_rot_raw - kg0[None, :]
    tau = np.asarray(translations[s_sp], dtype=np.float64)
    phase = np.exp(-1j * (g_rot_raw.astype(np.float64) @ tau))
    U_s = np.asarray(U_spinor_spatial[s_sp], dtype=np.complex128)
    if is_trs:
        I_SIGMA_Y = np.array([[0., 1.], [-1., 0.]], dtype=np.complex128)
        out_sp = np.einsum('ab,nbk->nak', U_s, psi_kbar.astype(np.complex128)) * phase[None, None, :]
        psi_full = np.einsum('ab,nbk->nak', I_SIGMA_Y, np.conj(out_sp))
    else:
        psi_full = np.einsum('ab,nbk->nak', U_s, psi_kbar.astype(np.complex128)) * phase[None, None, :]
    return psi_full, g_full


def main():
    loader_sym = WfnLoader(SYM_WFN)
    loader_nos = WfnLoader(NOSYM_WFN)
    sym = loader_sym._ensure_sym()
    fft_grid = tuple(int(x) for x in loader_sym.fft_grid)
    ntran = int(loader_sym.ntran)
    bvec = np.asarray(loader_sym.bvec)
    bvecinv = np.linalg.inv(bvec)
    mtrx = np.asarray(loader_sym.sym_matrices)        # (ntran,3,3)
    sym_mats_k_spatial = np.asarray(sym.sym_mats_k[:ntran])   # = mtrx.T

    # Step 0 confirmation
    with h5py.File(SYM_WFN, "r") as f:
        tnp = f["mf_header/symmetry/tnp"][:]
    tau_frac_max = np.array([
        np.max(np.abs(((tnp[s] / (2*np.pi) + 0.5) % 1.0) - 0.5)) for s in range(ntran)])
    is_nonsymm = tau_frac_max > 1e-6
    dets = np.linalg.det(mtrx.astype(float)).round().astype(int)
    n_nonsymm = int(np.sum(is_nonsymm))
    print(f"Step 0: ntran={ntran}, non-symmorphic ops {n_nonsymm}/{ntran}; det dist {np.unique(dets, return_counts=True)}")
    assert n_nonsymm == 36

    # ---- Two U_spinor tables: LORRAX-A and corrected-C ----
    R_A = np.einsum('ij,njk,kl->nil', bvecinv, sym_mats_k_spatial, bvec)
    R_C = np.einsum('ij,njk,kl->nil', bvecinv, mtrx, bvec)
    U_A = np.array([get_su2_from_R_cart(R_A[i]) for i in range(ntran)])
    U_C = np.array([get_su2_from_R_cart(R_C[i]) for i in range(ntran)])
    # Sanity: U_A should match sym.U_spinor (= LORRAX's current)
    print(f"  |sym.U_spinor - U_A|_max = {np.max(np.abs(np.asarray(sym.U_spinor) - U_A)):.3e}")
    print()

    # Build full table of test cases
    unfolded_kpts = np.asarray(sym.unfolded_kpts)
    irr_idx_k = np.asarray(sym.irr_idx_k); sym_idx_k = np.asarray(sym.sym_idx_k)
    used = np.unique(sym_idx_k)
    nosym_kpts = np.asarray(loader_nos.kpoints)
    nos_match = build_match_table(unfolded_kpts, nosym_kpts)
    assert (nos_match >= 0).all()

    # Pick a comprehensive set: every distinct sym_idx, FIRST 1 k_full per
    cats = {"identity": [], "proper_symm": [], "proper_nonsymm": [],
            "improper_symm": [], "improper_nonsymm": [], "trs": []}
    for s_int in used:
        s_int = int(s_int)
        if s_int >= ntran:
            cats["trs"].append(s_int); continue
        d = int(dets[s_int]); nsm = bool(is_nonsymm[s_int])
        if s_int == 0: cats["identity"].append(s_int)
        elif d > 0 and not nsm: cats["proper_symm"].append(s_int)
        elif d > 0 and nsm: cats["proper_nonsymm"].append(s_int)
        elif d < 0 and not nsm: cats["improper_symm"].append(s_int)
        else: cats["improper_nonsymm"].append(s_int)

    print("Categories:", {k: len(v) for k, v in cats.items()})

    # Bands window
    b_lo, b_hi = 0, 16
    nb_win = b_hi - b_lo

    # LORRAX full-BZ load (single shot)
    psi_lor = np.asarray(loader_sym.load(bands=(b_lo, b_hi), k='full_bz'))
    g_lor = np.asarray(loader_sym.gvecs(k='full_bz'))
    ngk_lor = np.asarray(loader_sym.ngk_valid(k='full_bz'))

    # Raw IBZ
    ngk_irr = np.asarray(loader_sym.ngk, dtype=np.int64)
    starts_irr = np.cumsum(np.concatenate([[0], ngk_irr[:-1]])).astype(np.int64)
    coeffs_irr = loader_sym._coeffs_raw
    gv_irr = loader_sym._gvecs_raw

    # Raw nosym
    ngk_nos = np.asarray(loader_nos.ngk, dtype=np.int64)
    starts_nos = np.cumsum(np.concatenate([[0], ngk_nos[:-1]])).astype(np.int64)
    coeffs_nos = loader_nos._coeffs_raw
    gv_nos = loader_nos._gvecs_raw

    # Iterate test cases
    results = []
    print()
    print(f"{'cat':16s} {'k_f':>3s} {'k_irr':>5s} {'s':>3s} {'det':>4s} {'τ≠0':>4s} "
          f"{'a-b1 raw':>10s} {'b1-c unit':>10s} {'b1-c gauge':>11s} {'b2-c unit':>10s} {'b2-c gauge':>11s}")
    for cat, lst in cats.items():
        for s_idx in lst:
            ks = np.where(sym_idx_k == s_idx)[0]
            kf = int(ks[0]) if len(ks) > 0 else None
            if kf is None: continue
            kirr = int(irr_idx_k[kf])
            s_sp = s_idx - ntran if s_idx >= ntran else s_idx

            sym_krep = np.asarray(sym.sym_mats_k[s_idx], dtype=np.int64)
            kg0 = np.rint(unfolded_kpts[kf] - sym_krep @ loader_sym.kpoints[kirr]).astype(np.int64)

            # (a) LORRAX
            ngk_a = int(ngk_lor[kf])
            psi_a = psi_lor[kf, :nb_win, :, :ngk_a]
            gv_a = g_lor[kf, :ngk_a]

            # (b1) hand-rolled with LORRAX U_spinor
            ngk_kbar = int(ngk_irr[kirr])
            start = int(starts_irr[kirr])
            raw = coeffs_irr[b_lo:b_hi, :, start:start+ngk_kbar, :]
            psi_kbar = raw[..., 0] + 1j*raw[..., 1]
            g_kbar = gv_irr[start:start+ngk_kbar]
            psi_b1, gv_b1 = hand_unfold(
                psi_kbar, g_kbar, s_idx, sym.sym_mats_k,
                loader_sym.translations, U_A, ntran, kg0)
            # (b2) corrected U_spinor
            psi_b2, gv_b2 = hand_unfold(
                psi_kbar, g_kbar, s_idx, sym.sym_mats_k,
                loader_sym.translations, U_C, ntran, kg0)

            # (c) nosym
            kn = int(nos_match[kf])
            ngk_c = int(ngk_nos[kn])
            start = int(starts_nos[kn])
            raw = coeffs_nos[b_lo:b_hi, :, start:start+ngk_c, :]
            psi_c = raw[..., 0] + 1j*raw[..., 1]
            gv_c = gv_nos[start:start+ngk_c]

            # box scatter
            box_a = scatter(psi_a, gv_a, ngk_a, fft_grid)
            box_b1 = scatter(psi_b1, gv_b1, ngk_kbar, fft_grid)
            box_b2 = scatter(psi_b2, gv_b2, ngk_kbar, fft_grid)
            box_c = scatter(psi_c, gv_c, ngk_c, fft_grid)

            ab_raw = float(np.max(np.abs(box_a - box_b1)))
            e_kirr = np.asarray(loader_sym.energies[0, kirr, b_lo:b_hi])
            u_b1c, g_b1c = compare(box_b1, box_c, e_kirr)
            u_b2c, g_b2c = compare(box_b2, box_c, e_kirr)

            rec = dict(category=cat, k_full=kf, k_irr=kirr, sym_idx=int(s_idx),
                       det=int(dets[s_sp]), is_nonsymm=bool(is_nonsymm[s_sp]),
                       is_trs=bool(s_idx >= ntran), kg0=kg0.tolist(),
                       ab_raw_max=ab_raw,
                       b1c_unit=u_b1c, b1c_gauge=g_b1c,
                       b2c_unit=u_b2c, b2c_gauge=g_b2c)
            results.append(rec)
            print(f"{cat:16s} {kf:3d} {kirr:5d} {s_idx:3d} {rec['det']:+4d} "
                  f"{'T' if rec['is_nonsymm'] else 'F':>4s} "
                  f"{ab_raw:10.2e} {u_b1c:10.2e} {g_b1c:11.2e} {u_b2c:10.2e} {g_b2c:11.2e}")

    with open(os.path.join(OUT_DIR, "test1_final.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Aggregate
    print()
    print("=== Aggregate (max within category) ===")
    by_cat = {}
    for r in results: by_cat.setdefault(r["category"], []).append(r)
    for cat, lst in by_cat.items():
        max_ab = max(r["ab_raw_max"] for r in lst)
        b1c_g = max(r["b1c_gauge"] for r in lst)
        b2c_g = max(r["b2c_gauge"] for r in lst)
        print(f"  {cat:18s} n={len(lst):3d}  a-b1 raw max={max_ab:.3e}  "
              f"b1-c gauge max={b1c_g:.3e}   b2-c gauge max={b2c_g:.3e}")


if __name__ == "__main__":
    main()
