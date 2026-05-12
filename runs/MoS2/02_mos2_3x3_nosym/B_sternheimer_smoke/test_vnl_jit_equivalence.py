"""Equivalence test: jit'd build_vnl_kdata vs current path.

Builds Z two ways for several k-points and verifies bitwise (or near-
bitwise) equality on the physical-G slice:

  (a) current:  build_vnl_kdata_from_kvec(kvec, Gk_natural, setup)
  (b) padded:   build_vnl_kdata_from_kvec(kvec, Gk_padded_to_ngkmax, setup)
                then mask: Z_padded[:, :nG_actual] should equal (a)

This catches any regression in the new jit'd assembly's tail-G handling
before we wire it into setup_H_k_from_kvec.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')

from runtime import set_default_env; set_default_env()
import numpy as np
import jax.numpy as jnp

from common import Meta, symmetry_maps
from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import build_dft_potentials, build_rho_val_from_wfn
from psp.dft_operators import build_T_diag_from_kvec
from psp.vnl_ops import build_vnl_kdata_from_kvec
from psp.gvec_utils import compute_ngkmax

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(wfn, pseudos, rho_val, truncation_2d=True, verbose=False)

kpts = np.asarray(sym.unfolded_kpts, dtype=np.float64)
ngkmax = int(compute_ngkmax(kpts, np.asarray(wfn.bdot),
                             float(wfn.ecutwfc), tuple(wfn.fft_grid)))
print(f"ngkmax = {ngkmax}, nk = {len(kpts)}")

max_abs = 0.0
max_rel = 0.0
nGs = []
for ik in range(len(kpts)):
    kv = kpts[ik]
    T_diag, Gx, Gy, Gz = build_T_diag_from_kvec(kv, wfn)
    Gk_natural = np.stack([np.asarray(Gx), np.asarray(Gy), np.asarray(Gz)], axis=-1)
    nG = Gk_natural.shape[0]
    nGs.append(nG)

    # (a) current: natural-nG path
    kdata_a = build_vnl_kdata_from_kvec(kv, Gk_natural, vnl_setup)
    Z_a = np.asarray(kdata_a.Z)   # (total_R, nG)

    # (b) padded: ngkmax-padded G-sphere (zero-pad Gk_int)
    pad = ngkmax - nG
    Gk_padded = np.concatenate([Gk_natural,
                                np.zeros((pad, 3), dtype=Gk_natural.dtype)], axis=0)
    kdata_b = build_vnl_kdata_from_kvec(kv, Gk_padded, vnl_setup)
    Z_b = np.asarray(kdata_b.Z)   # (total_R, ngkmax)

    # Compare physical-G slice
    diff = Z_a - Z_b[:, :nG]
    a_abs = float(np.max(np.abs(diff)))
    a_norm = float(np.max(np.abs(Z_a)))
    a_rel = a_abs / a_norm if a_norm > 0 else 0.0
    max_abs = max(max_abs, a_abs)
    max_rel = max(max_rel, a_rel)

    # Tail check: padded entries must NOT be zero (they're computed at q=|kvec|);
    # the contract is the *caller* must mask them.
    tail_norm = float(np.max(np.abs(Z_b[:, nG:]))) if pad > 0 else 0.0
    print(f"  ik={ik}: nG={nG}  pad={pad}  "
          f"|ΔZ_phys|_∞={a_abs:.3e} (rel {a_rel:.2e})  "
          f"|Z_pad|_∞={tail_norm:.3e}")

print(f"\n[summary] nG range: {min(nGs)}–{max(nGs)};  ngkmax={ngkmax}")
print(f"[summary] max |Z_a - Z_b[:, :nG]|_∞ = {max_abs:.3e}")
print(f"[summary] max relative              = {max_rel:.3e}")
assert max_rel < 1e-12, f"Z mismatch across natural-nG vs padded path: {max_rel:.3e}"
print("[PASS] padded-Gk_int path produces bitwise-equal Z on physical-G slice.")
