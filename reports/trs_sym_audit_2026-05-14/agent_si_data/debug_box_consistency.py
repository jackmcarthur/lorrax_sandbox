"""Verify whether psi_a (LORRAX) and psi_b (hand-roll) end up at the same box position when properly aligned."""
import sys
sys.path.insert(0, "/global/u2/j/jackm/software/lorrax_B/src")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from file_io.wfn_loader import WfnLoader
from common.symmetry_maps import unfold_psi

sym = WfnLoader("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/05_si_4x4x4_sym/qe/nscf/WFN.h5")
nos = WfnLoader("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5")
s = sym._ensure_sym()
ntran = sym.ntran

for kf in [0, 1, 11, 27]:
    kirr = int(s.irr_idx_k[kf])
    sidx = int(s.sym_idx_k[kf])
    sym_krep = np.asarray(s.sym_mats_k[sidx], dtype=np.int64)
    ngk = int(sym.ngk[kirr])
    start = int(sym._kpt_starts[kirr])
    raw = sym._coeffs_raw[0:4, :, start:start + ngk, :]
    psi_kbar = raw[..., 0] + 1j * raw[..., 1]                     # (4, 2, ngk)
    g_kbar = sym._gvecs_raw[start:start + ngk]
    # unfold_psi (LORRAX path with the proper API)
    psi_uf = unfold_psi(
        psi_kbar, sym_idx=sidx, n_sym_spatial=ntran,
        g_kbar=g_kbar, sym_mats_k=s.sym_mats_k,
        translations=sym.translations, U_spinor_spatial=s.U_spinor)

    # LORRAX load path
    psi_load = np.asarray(sym.load(bands=(0, 4), k='full_bz'))[kf, :4, :, :ngk]

    # Compare values directly (same slot indices, since g-axis labels differ only by kg0)
    print(f"k_f={kf:2d}  k_irr={kirr} s={sidx:2d}  ngk={ngk}  "
          f"max|unfold_psi - load|={np.max(np.abs(psi_uf - psi_load)):.3e}")
    k_full = s.unfolded_kpts[kf]
    skbar = sym_krep @ sym.kpoints[kirr]
    kg0 = np.rint(k_full - skbar).astype(np.int64)
    print(f"   kg0={kg0}")
