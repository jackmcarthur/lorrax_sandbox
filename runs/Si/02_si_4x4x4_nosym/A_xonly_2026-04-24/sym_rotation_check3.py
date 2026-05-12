"""Try many candidate rotation variants to find which makes S block-diagonal."""
import sys
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_A/src")

import numpy as np
from file_io import WFNReader
from common import symmetry_maps

SYM = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/qe/nscf/WFN.h5"
NOS = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5"
NBAND = 8

wfn_s = WFNReader(SYM); wfn_n = WFNReader(NOS)
sym = symmetry_maps.SymMaps(wfn_s)
ntran = wfn_s.ntran


def kidx(kfrac, kps, tol=2e-3):
    for i, k in enumerate(kps):
        d = kfrac - k; d = d - np.round(d)
        if np.max(np.abs(d)) < tol: return i
    return None


def overlap(cA, gA, cB, gB, fft):
    nx, ny, nz = [int(x) for x in fft]
    def to(cc, gg):
        n = cc.shape[0]
        G = np.zeros((n, 2, nx, ny, nz), dtype=np.complex128)
        gx = np.mod(gg[:, 0], nx); gy = np.mod(gg[:, 1], ny); gz = np.mod(gg[:, 2], nz)
        for i in range(n):
            G[i, 0, gx, gy, gz] = cc[i, 0]
            G[i, 1, gx, gy, gz] = cc[i, 1]
        return G.reshape(n, -1)
    return to(cA, gA).conj() @ to(cB, gB).T


def variant(tau_mode, sgn_tau, use_kg0, spinor_mode, nk_full):
    """
    tau_mode: 'raw' (tau from wfn as-is), 'over_2pi' (tau/2π), 'times_2pi' (tau*2π)
    sgn_tau: +1 or -1 (sign of phase argument)
    use_kg0: include +kg0 or +(-kg0) in phase
    spinor_mode: 'U', 'U.T', 'U†', 'U*'
    """
    sym_idx, kbar_idx, sym_krep = sym._get_symmetry_context(nk_full)
    cnk = wfn_s.get_cnk_batch(kbar_idx, np.arange(NBAND))
    # (Skip time-reversal branch for now; sym_idx=6,7 are spatial)
    tau = np.asarray(wfn_s.translations[sym_idx], dtype=np.float64)
    if tau_mode == 'over_2pi': tau = tau / (2 * np.pi)
    elif tau_mode == 'times_2pi': tau = tau * (2 * np.pi)
    if np.any(np.abs(tau) > 1e-12):
        k_gvecs = wfn_s.get_gvec_nk(kbar_idx).astype(np.float64)
        rotated = np.einsum('ij,gj->gi', sym_krep.astype(np.int32), k_gvecs.astype(np.int32)).astype(np.float64)
        if use_kg0 != 0:
            kg0 = sym._get_umklapp_vector(wfn_s, nk_full, sym_idx, kbar_idx, sym_krep).astype(np.float64)
            rotated = rotated + use_kg0 * kg0
        phase_arg = rotated @ tau
        phase = np.exp(sgn_tau * 1j * phase_arg)
        cnk = cnk * phase[None, None, :]
    U = sym.U_spinor[sym_idx]
    if spinor_mode == 'U.T': U = U.T
    elif spinor_mode == 'U†': U = U.conj().T
    elif spinor_mode == 'U*': U = U.conj()
    cnk = np.einsum('jk,nkl->njl', U, cnk)
    return cnk, sym.get_gvecs_kfull(wfn_s, nk_full)


for nk_full in [21, 48, 30, 15]:
    sym_idx, kbar_idx, _ = sym._get_symmetry_context(nk_full)
    k_full = np.asarray(sym.unfolded_kpts[nk_full])
    nk_nos = kidx(k_full, wfn_n.kpoints)
    print(f"\n=== nk_full={nk_full} k={tuple(k_full.round(4))}, sym_idx={sym_idx}, kbar={kbar_idx} ===")
    cnos = wfn_n.get_cnk_batch(nk_nos, np.arange(NBAND))
    gnos = wfn_n.get_gvec_nk(nk_nos)
    best = None
    for tau_mode in ['raw', 'over_2pi']:
        for sgn in [-1, +1]:
            for use_kg0 in [0, +1, -1]:
                for sp in ['U', 'U.T', 'U†', 'U*']:
                    crot, grot = variant(tau_mode, sgn, use_kg0, sp, nk_full)
                    S = overlap(crot, grot, cnos, gnos, wfn_s.fft_grid)
                    diag = float(np.sum(np.abs(np.diag(S))**2))
                    # Block-diagonal metric: after truncating tiny elements, is S block-unitary?
                    # Use energy grouping (nosym energies)
                    energies = wfn_n.energies[0, nk_nos, :NBAND] * 13.6057
                    blocks = []
                    i = 0
                    while i < NBAND:
                        j = i + 1
                        while j < NBAND and abs(energies[j] - energies[i]) < 5e-3:
                            j += 1
                        blocks.append((i, j))
                        i = j
                    off_block = 0.0
                    for (i0, i1) in blocks:
                        for (j0, j1) in blocks:
                            if i0 != j0:
                                off_block += float(np.sum(np.abs(S[i0:i1, j0:j1])**2))
                    key = (round(diag, 3), round(off_block, 3))
                    if best is None or off_block < best[0]:
                        best = (off_block, diag, tau_mode, sgn, use_kg0, sp)
    ob, dg, tm, sg, kg, sp = best
    print(f"  best variant: tau={tm}, sgn={sg}, kg0={kg}, spinor={sp}  →  diag²={dg:.3f}  off_block²={ob:.4f}")
