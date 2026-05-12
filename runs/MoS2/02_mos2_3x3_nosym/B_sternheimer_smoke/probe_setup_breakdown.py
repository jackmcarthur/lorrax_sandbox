"""Granular timing of run_sternheimer setup phase.

Splits the 17.9-s 'Setup complete' wall into its constituent calls so
we can see which block to attack next.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling')

from runtime import set_default_env; set_default_env()
from common.jax_compile_cache import ensure_jax_compile_cache
ensure_jax_compile_cache()

import numpy as np
import jax
import jax.numpy as jnp

from common import Meta, symmetry_maps
from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import (
    build_dft_potentials, build_rho_val_from_wfn,
    build_ionic_and_core, build_V_scf,
)
import psp.vnl_ops as vnl_ops
from psp.dft_operators import compute_V_H_and_V_xc, build_G_cart, setup_H_k_from_kvec
from psp.gvec_utils import compute_ngkmax

def t(label, fn):
    t0 = time.perf_counter()
    out = fn()
    # block on any jax arrays in the output
    if isinstance(out, jax.Array):
        out.block_until_ready()
    elif isinstance(out, tuple):
        for x in out:
            if isinstance(x, jax.Array):
                x.block_until_ready()
    print(f"  [{time.perf_counter()-t0:6.2f}s]  {label}", flush=True)
    return out

T = time.perf_counter()
print("── pre-init ──", flush=True)
wfn = t("WFNReader", lambda: WFNReader('WFN.h5'))
sym = t("SymMaps",   lambda: symmetry_maps.SymMaps(wfn))
n_occ = int(wfn.nelec)
meta = t("Meta", lambda: Meta.from_system(wfn, sym, nval=n_occ, ncond=0,
                                          nband=n_occ, n_rmu=0, bispinor=False))
pseudos = t("load_pseudopotentials", lambda: load_pseudopotentials('.'))

print("\n── ρ_val ──", flush=True)
rho_val = t("build_rho_val_from_wfn", lambda: build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False))

print("\n── build_dft_potentials components ──", flush=True)
fft_grid = wfn.fft_grid
nspinor = int(wfn.nspinor)
truncation_2d = True

V_loc, rho_core, rho_core_G = t(
    "build_ionic_and_core",
    lambda: build_ionic_and_core(wfn, pseudos, fft_grid, truncation_2d=truncation_2d),
)
G_cart = t(
    "build_G_cart",
    lambda: build_G_cart(int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2]),
                         float(wfn.blat) * np.asarray(wfn.bvec, dtype=float)),
)
V_H, V_xc = t(
    "compute_V_H_and_V_xc",
    lambda: compute_V_H_and_V_xc(
        jnp.asarray(rho_val, dtype=jnp.float64), rho_core, rho_core_G,
        G_cart, jnp.asarray(wfn.bdot, dtype=jnp.float64),
        jnp.asarray(wfn.bvec, dtype=jnp.float64), wfn.blat,
        truncation_2d=truncation_2d),
)
V_scf = t("build_V_scf", lambda: build_V_scf(V_loc, V_H, V_xc))
vnl_setup = t(
    "vnl_ops.build_vnl_setup",
    lambda: vnl_ops.build_vnl_setup(
        wfn, pseudos=pseudos, nspinor=nspinor,
        q_max=float(np.sqrt(float(wfn.ecutwfc))) * 1.01),
)

print("\n── ψ unfold ──", flush=True)
from common.load_wfns import load_kpoint_fftbox
nb_load = n_occ + 20
nk_full = int(sym.nk_tot)
def _load_psi():
    psi_list = []
    for ik in range(nk_full):
        psi_list.append(load_kpoint_fftbox(wfn, sym, meta, ik, nb_load))
    return jnp.stack(psi_list, axis=0)
psi_box_full = t(f"unfold ψ (nk={nk_full}, nb={nb_load})", _load_psi)

print("\n── H_cache build ──", flush=True)
kpts_all = np.asarray(sym.unfolded_kpts, dtype=np.float64)
ngkmax = int(compute_ngkmax(kpts_all, np.asarray(wfn.bdot),
                             float(wfn.ecutwfc), tuple(wfn.fft_grid)))
def _build_H_cache():
    H_cache = []
    for ik in range(nk_full):
        kv = np.asarray(sym.unfolded_kpts[ik], dtype=np.float64)
        H_k = setup_H_k_from_kvec(kv, V_scf, vnl_setup, wfn, meta,
                                   V_loc_r=V_loc, ngkmax=ngkmax)
        Gk_int = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
        H_cache.append((H_k, Gk_int))
    # Block on the last one
    H_cache[-1][0].T_diag.block_until_ready()
    return H_cache
H_cache = t(f"H_cache (nk={nk_full})", _build_H_cache)

print(f"\n total wall = {time.perf_counter()-T:.2f}s")
