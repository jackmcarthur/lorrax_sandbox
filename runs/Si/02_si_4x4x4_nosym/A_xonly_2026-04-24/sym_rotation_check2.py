"""Localize WHICH sym ops fail. Test sym_idx 6 and 7 on a single k-point with
several candidate fixes: (a) add iσ_y for time-reversed; (b) try phase sign
flips on the non-symmorphic factor; (c) inspect the Umklapp vector."""
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


def standard_rot(sym, wfn, bands, nk_full):
    """What LORRAX currently does."""
    return sym.get_cnk_fullzone_batch(wfn, bands, nk_full), sym.get_gvecs_kfull(wfn, nk_full)


def iσy_tr_rot(sym, wfn, bands, nk_full):
    """LORRAX but with T = iσ_y · K for time-reversed sym ops."""
    sym_idx, kbar_idx, sym_krep = sym._get_symmetry_context(nk_full)
    cnk = wfn.get_cnk_batch(kbar_idx, bands)
    if sym_idx >= ntran:
        cnk = np.conj(cnk)
        # apply iσ_y: new[0]=i*c[1], new[1]=-i*c[0]
        cnk = np.stack([1j * cnk[:, 1], -1j * cnk[:, 0]], axis=1)
    else:
        phase = sym._get_fractional_translation_phase(wfn, nk_full, sym_idx, kbar_idx, sym_krep)
        if phase is not None:
            cnk = cnk * phase[None, None, :]
    cnk = np.einsum('jk,nkl->njl', sym.U_spinor[sym_idx], cnk)
    return cnk, sym.get_gvecs_kfull(wfn, nk_full)


def phase_flipped_rot(sym, wfn, bands, nk_full):
    """Try flipping the non-symmorphic phase sign."""
    sym_idx, kbar_idx, sym_krep = sym._get_symmetry_context(nk_full)
    cnk = wfn.get_cnk_batch(kbar_idx, bands)
    if sym_idx >= ntran:
        cnk = np.conj(cnk)
        cnk = np.stack([1j * cnk[:, 1], -1j * cnk[:, 0]], axis=1)
    else:
        phase = sym._get_fractional_translation_phase(wfn, nk_full, sym_idx, kbar_idx, sym_krep)
        if phase is not None:
            cnk = cnk * np.conj(phase)[None, None, :]  # flipped
    cnk = np.einsum('jk,nkl->njl', sym.U_spinor[sym_idx], cnk)
    return cnk, sym.get_gvecs_kfull(wfn, nk_full)


for nk_full, label in [(21, "(.25,.25,.25)"), (48, "(.75,0,0)")]:
    sym_idx, kbar_idx, _ = sym._get_symmetry_context(nk_full)
    k_full = np.asarray(sym.unfolded_kpts[nk_full])
    nk_nos = kidx(k_full, wfn_n.kpoints)
    print(f"\n=== nk_full={nk_full} {label}, sym_idx={sym_idx} (ntran={ntran}), kbar={kbar_idx} ===")
    print(f"  sym_mats_k[{sym_idx}] =\n{sym.sym_mats_k[sym_idx]}")
    # print tau
    if sym_idx < ntran:
        print(f"  tau (frac) = {wfn_s.translations[sym_idx]}")
    else:
        print(f"  (time-reversed of spatial idx {sym_idx - ntran})")
    cnos = wfn_n.get_cnk_batch(nk_nos, np.arange(NBAND))
    gnos = wfn_n.get_gvec_nk(nk_nos)
    for name, fn in [("standard", standard_rot),
                      ("+iσy TR", iσy_tr_rot),
                      ("phase*", phase_flipped_rot)]:
        crot, grot = fn(sym, wfn_s, np.arange(NBAND), nk_full)
        S = overlap(crot, grot, cnos, gnos, wfn_s.fft_grid)
        diag = float(np.sum(np.abs(np.diag(S)) ** 2))
        off  = float(np.sum(np.abs(S) ** 2) - diag)
        total = float(np.sum(np.abs(S) ** 2))
        # unitarity
        SUU = S @ S.conj().T
        eye = np.eye(NBAND)
        uerr = float(np.linalg.norm(SUU - eye))
        print(f"  {name:12s}: Σ|S|²={total:.4f}  |diag|²={diag:.4f}  off={off:.4f}  ‖SS†-I‖={uerr:.3f}")
