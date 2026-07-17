#!/usr/bin/env python3
"""RUNG 9 — the Γ on-site block (the size-1 star the star-comparison tests are
blind to; the 518 μeV doublet is 81% Γ).

Build the point-group rep D_R at Γ directly from ψ (symmorphic ops only, τ=0 ⇒
no umklapp phase; spinor rotation U_spinor[s] from SymMaps).  For the real-space
op, (Rψ)(r_μ) = U_spinor[s] · ψ(r_{α_s(μ)}).  Then:
  (1) ψ_Γ covariance residual: ‖Ψ_R − Ψ D_R‖/‖Ψ_R‖ with D_R = Ψ†Ψ_R (projection
      onto the degenerate subspace) + unitarity ‖D_R†D_R−I‖.  Small ⇒ ψ_Γ (hence
      the transition density M_Γ) is point-group covariant.
  (2) Γ-block covariance: ‖[U(R), H_Γ]‖ with the pair-basis rep U(R)=D_R^c⊗D_R^{v*}
      for H_Γ = D_Γ + Kx_Γ − Kd_Γ (and each term) — which term fails to commute.
"""
import sys, os, json
import numpy as np, h5py
LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16"
sys.path.insert(0, os.path.join(LROOT, "src"))
import jax; jax.config.update("jax_enable_x64", True)
from file_io.wfn_loader import WfnLoader
from common import symmetry_maps
from centroid.orbit_syms import compute_centroid_sym_perm
from bse import bse_io
RY = 13.6056980659
WFN = os.path.join(LROOT, "tests/regression/si_cohsex_debug/WFN.h5")
FFT = np.array([24, 24, 24])


def subspace_rep(Psi, PsiR):
    """Psi,(d,n): centroid-flattened ψ over a subspace (NOT orthonormal on the
    subsampled centroid grid).  Gram-corrected representation D=(Ψ†Ψ)⁻¹Ψ†Ψ_R;
    residual = out-of-subspace leakage; 'unitarity' in the Gram metric: D†G D = G."""
    G = Psi.conj().T @ Psi                        # (n,n) Gram
    D = np.linalg.solve(G, Psi.conj().T @ PsiR)   # (n,n) representation
    resid = np.linalg.norm(PsiR - Psi @ D) / (np.linalg.norm(PsiR) + 1e-30)
    unit = np.linalg.norm(D.conj().T @ G @ D - G) / (np.linalg.norm(G) + 1e-30)
    return D, float(resid), float(unit)


def main():
    arm = "sym"
    inp = f"{RUN}/work_{arm}/cohsex_si_test.in"
    restart = bse_io._find_restart_file(inp)
    wfn = WfnLoader(WFN, backend="eager"); sym = symmetry_maps.SymMaps(wfn)
    ntran = int(wfn.ntran)
    tnp = np.asarray(wfn.translations[:ntran])/(2*np.pi)
    Uspin = np.asarray(sym.U_spinor[:ntran])       # (48,2,2)
    cfrac = np.loadtxt(f"{RUN}/work_{arm}/centroids_frac_792.txt")
    ridx = np.rint(cfrac*FFT[None]).astype(np.int64) % FFT[None]
    alpha, Lwrap = compute_centroid_sym_perm(
        ridx, wfn.sym_matrices[:ntran], wfn.translations[:ntran], FFT, validate=True)

    with h5py.File(restart, "r") as f:
        psi = np.asarray(f["psi_full_y"][:])       # (nk,nb,s,μ)
        V0 = np.asarray(f["V_qmunu"][0]); W0 = np.asarray(f["W0_qmunu"][0])
    kG = 0                                          # Γ = full-BZ index 0
    val = [4, 5, 6, 7]; cond = [8, 9, 10, 11]
    nmu = psi.shape[3]
    symm_ops = [s for s in range(ntran) if np.linalg.norm(tnp[s]) < 1e-6]
    print(f"[Γ sym] nmu={nmu} #symmorphic ops(τ=0)={len(symm_ops)}", flush=True)

    out = {"arm": arm, "n_symmorphic": len(symm_ops)}

    # ψ_Γ over a subspace as (2μ, n): stack spin.
    def PsiG(bands):
        p = psi[kG][bands]           # (n,s,μ)
        return np.transpose(p, (1, 2, 0)).reshape(-1, len(bands))  # (s*μ, n)

    def rotate(bands, s):
        # (Rψ)(r_μ)=U_spinor[s]·ψ(r_{α_s(μ)}) — spin-mix + centroid perm.
        p = psi[kG][bands]                          # (n,s,μ)
        a = alpha[s]
        p_perm = p[:, :, a]                          # ψ(r_{α_s(μ)})
        p_rot = np.einsum("ab,nbm->nam", Uspin[s], p_perm, optimize=True)
        return np.transpose(p_rot, (1, 2, 0)).reshape(-1, len(bands))

    # (1) ψ_Γ covariance residual (try α and α^{-1}, keep the unitary one)
    res = {}
    for nm, bands in (("val", val), ("cond", cond)):
        Psi = PsiG(bands)
        worst_r = 0.0; worst_u = 0.0; worst_s = None
        Dstore = {}
        for s in symm_ops:
            _, r, u = subspace_rep(Psi, rotate(bands, s))
            Dstore[s] = (r, u)
            if r > worst_r:
                worst_r, worst_u, worst_s = r, u, s
        res[nm] = {"worst_resid": worst_r, "unitarity_at_worst": worst_u, "worst_op": worst_s}
        print(f"  (1) ψ_Γ[{nm}] covariance: worst proj-resid={worst_r:.3e} "
              f"(op {worst_s}, ‖D†D−I‖={worst_u:.3e})", flush=True)
    out["psiG_covariance"] = res

    # (2) Γ-block commutator.  Build D_R^c, D_R^v then U(R)=D^c ⊗ conj(D^v) on
    # the pair basis |c v⟩ (row index a=(c,v)); test [U(R),T_Γ] per term.
    data = bse_io._load_ring_subset(restart, n_val=4, n_cond=4, px=1, py=1, input_file=inp)
    pc = np.asarray(data["psi_c"])[kG]; pv = np.asarray(data["psi_v"])[kG]  # (n,s,μ)
    ec = np.asarray(data["eps_c"])[kG]; ev = np.asarray(data["eps_v"])[kG]
    V0h = np.asarray(data["V_q0"]); Wf = np.asarray(data["W_q"]).reshape(nmu, nmu, 64)
    W0h = Wf[:, :, 0]
    nc = len(cond); nv = len(val)
    # transition density M_cv(μ)=Σ_s conj(ψ_c)ψ_v ; pair dens for Kd
    M = np.einsum("csm,vsm->cvm", np.conj(pc), pv, optimize=True)          # (c,v,μ)
    Dcv = (ec[:, None] - ev[None, :])                                      # (c,v)
    Kx = np.einsum("cvm,mn,CVn->cvCV", np.conj(M), V0h, M, optimize=True)  # /Nk drops (per-k const)
    Pc = np.einsum("csm,Csm->cCm", np.conj(pc), pc, optimize=True)
    Pv = np.einsum("vsn,Vsn->vVn", pv, np.conj(pv), optimize=True)
    tmp = np.einsum("cCm,mn->cCn", Pc, W0h, optimize=True)
    Kd = np.einsum("cCn,vVn->cvCV", tmp, Pv, optimize=True)
    Dmat = np.zeros((nc, nv, nc, nv), complex)
    for c in range(nc):
        for v in range(nv):
            Dmat[c, v, c, v] = Dcv[c, v]
    HG = Dmat + Kx - Kd
    # flatten pair (c,v)->a
    def flat(T):
        return T.reshape(nc*nv, nc*nv)
    comm = {}
    Pcv = PsiG(cond); Pvv = PsiG(val)
    for s in symm_ops:
        Dc, rc, uc = subspace_rep(Pcv, rotate(cond, s))
        Dv, rv, uv = subspace_rep(Pvv, rotate(val, s))
        # pair-basis rep: |cv> -> Σ Dc[C,c] conj(Dv[V,v]) |CV>
        U = np.einsum("Cc,Vv->CVcv", Dc, np.conj(Dv)).reshape(nc*nv, nc*nv)
        for nm, T in (("D", flat(Dmat)), ("Kx", flat(Kx)), ("Kd", flat(Kd)), ("H", flat(HG))):
            c1 = np.linalg.norm(U @ T - T @ U) / (np.linalg.norm(T) + 1e-30)
            comm.setdefault(nm, []).append(c1)
    out["gamma_block_commutator"] = {nm: {"max_rel": float(np.max(v)),
                                          "mean_rel": float(np.mean(v))}
                                     for nm, v in comm.items()}
    for nm in ("D", "Kx", "Kd", "H"):
        cc = comm[nm]
        print(f"  (2) Γ-block ‖[U(R),{nm:>2}]‖/‖{nm}‖: max={np.max(cc):.3e} "
              f"mean={np.mean(cc):.3e}", flush=True)

    # scale context for the split
    print(f"  Γ D diag (e_c-e_v) eV: {np.round(np.unique(np.round(Dcv.ravel()*RY,4)),4)}",
          flush=True)
    json.dump(out, open(f"{RUN}/diag/rung9_gamma.json", "w"), indent=2)
    print("  wrote rung9_gamma.json", flush=True)


if __name__ == "__main__":
    main()
