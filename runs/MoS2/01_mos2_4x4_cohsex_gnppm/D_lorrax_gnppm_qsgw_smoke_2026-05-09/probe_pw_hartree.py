#!/usr/bin/env python3
"""Independent plane-wave Hartree reference for MoS2 4x4.

Builds V_H two ways using the get_DFT_mtxels building blocks:
  - V_H_3D : standard 3D Coulomb (no slab truncation)
  - V_H_2D : 2D-truncated Coulomb (slab geometry)

Then for k=0 prints diagonal V_H matrix elements per band, alongside:
  - LORRAX hartree_k diagonal (from kin_ion + (eqp run's V_H reconstruction))
  - QE_kih (from qe/nscf/kih.dat, which is K + V_ion + V_H_3D)
  - LORRAX kin_ion diagonal (K + V_loc + V_NL, from kin_ion.h5)

Tells us whether the 6 eV per-band discrepancy lives in V_H or in kin_ion.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")

from file_io import WFNReader
from common import symmetry_maps, Meta
from common.load_wfns import read_Gvecs_to_devices

# Match cohsex.in for nval/ncond/nband
NVAL = 26
NCOND = 54
NBAND = 80
from psp.get_DFT_mtxels import (
    compute_valence_density,
    compute_hartree_potential_real,
    compute_local_V_k,
)
from psp.dft_operators import generate_gvectors_k

WFN = "WFN.h5"
KIH_REF = "../qe/nscf/kih.dat"  # if exists
KIN_ION_LORRAX = "kin_ion.h5"

HA2EV = 27.211386245988
RY2EV = 13.605693122994


def main():
    print(f"JAX devices: {jax.devices()}", flush=True)
    print(f"Reading WFN: {WFN}", flush=True)
    wfn = WFNReader(WFN)
    sym = symmetry_maps.SymMaps(wfn)
    meta = Meta.from_system(wfn, sym, NVAL, NCOND, NBAND, 0, False)
    print(f"  nspinor={wfn.nspinor}  nelec={wfn.nelec}  cell_vol={wfn.cell_volume:.4f} bohr^3")
    print(f"  nk_irr={wfn.nkpts}  nk_tot={sym.nk_tot}  nb_total={wfn.nbands}")
    print(f"  fft_grid={meta.fft_grid}  rho_grid={getattr(wfn, 'grid_rho', None)}")
    print(f"  bvec=\n{wfn.bvec}\n  blat={wfn.blat}", flush=True)

    devs = jax.devices()
    mesh_xy = Mesh(np.asarray(devs).reshape(len(devs), 1), axis_names=("x", "y"))

    # Load wavefunctions to device(s) — full mesh, sharded over x
    print("Loading wavefunctions to devices...", flush=True)
    bispinor = False  # 2-component spinor, not 4-component Dirac bispinor
    global_psi_G, _ = read_Gvecs_to_devices(
        wfn, sym, (0, NBAND), meta, bispinor, mesh_xy)
    print(f"  global_psi_G.shape = {global_psi_G.shape}", flush=True)

    # Compute valence density on rho grid (uses k-point weights)
    print("\nComputing valence density rho_v(r)...", flush=True)
    rho_v = compute_valence_density(global_psi_G, sym, wfn)
    rho_v_np = np.asarray(rho_v)
    integrated_charge = float(np.sum(rho_v_np)) * float(wfn.cell_volume) / float(rho_v_np.size)
    print(f"  rho_v shape = {rho_v_np.shape}")
    print(f"  integrated charge = {integrated_charge:.4f} (expect {2*int(wfn.nelec)} for spin-degen, or nelec={int(wfn.nelec)})")

    bdot_j = jnp.asarray(wfn.bdot, dtype=jnp.float64)
    bvec_j = jnp.asarray(wfn.bvec, dtype=jnp.float64)
    blat = float(wfn.blat)

    # V_H two ways
    print("\nComputing V_H_3D (no truncation)...", flush=True)
    V_H_3D = compute_hartree_potential_real(rho_v, bdot_j, bvec=bvec_j, blat=blat, truncation_2d=False)
    V_H_3D = np.asarray(V_H_3D)
    print(f"  V_H_3D real range = [{V_H_3D.real.min():.4f}, {V_H_3D.real.max():.4f}] Ry, mean={V_H_3D.real.mean():.4f}")

    print("\nComputing V_H_2D (slab truncation)...", flush=True)
    V_H_2D = compute_hartree_potential_real(rho_v, bdot_j, bvec=bvec_j, blat=blat, truncation_2d=True)
    V_H_2D = np.asarray(V_H_2D)
    print(f"  V_H_2D real range = [{V_H_2D.real.min():.4f}, {V_H_2D.real.max():.4f}] Ry, mean={V_H_2D.real.mean():.4f}")

    # Per-k-point V_H diagonal matrix elements
    nk = int(sym.nk_tot)
    nb = int(global_psi_G.shape[1])
    nb_show = min(nb, 80)

    print(f"\nComputing per-k V_H matrix elements (nk={nk}, nb={nb_show})...", flush=True)
    V_H_3D_diag = np.zeros((nk, nb_show), dtype=np.float64)
    V_H_2D_diag = np.zeros((nk, nb_show), dtype=np.float64)

    for ik in range(nk):
        Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)
        wfn_k = global_psi_G[ik, :nb_show]
        VH3 = compute_local_V_k(wfn_k, Gk_crys, V_H_3D, wfn.cell_volume)
        VH2 = compute_local_V_k(wfn_k, Gk_crys, V_H_2D, wfn.cell_volume)
        VH3 = np.asarray(VH3)
        VH2 = np.asarray(VH2)
        V_H_3D_diag[ik] = np.real(np.diagonal(VH3))
        V_H_2D_diag[ik] = np.real(np.diagonal(VH2))
        if ik == 0:
            # off-diagonal sanity
            offmag3 = np.max(np.abs(VH3 - np.diag(np.diagonal(VH3))))
            offmag2 = np.max(np.abs(VH2 - np.diag(np.diagonal(VH2))))
            print(f"  k=0: |off-diag| max  3D={offmag3:.4e}  2D={offmag2:.4e} (Ry)")

    # Convert to eV
    V_H_3D_diag_eV = V_H_3D_diag * RY2EV
    V_H_2D_diag_eV = V_H_2D_diag * RY2EV

    # Load LORRAX kin_ion (eV already? check).  kin_ion.h5 stores complex (nk, nb, nb).
    print(f"\nLoading LORRAX kin_ion: {KIN_ION_LORRAX}", flush=True)
    kin_ion = None
    kin_ion_unit = None
    if os.path.exists(KIN_ION_LORRAX):
        with h5py.File(KIN_ION_LORRAX, "r") as h5:
            print(f"  keys: {list(h5.keys())}")
            if "kin_ion" in h5:
                kin_ion = np.asarray(h5["kin_ion"])  # (nk, nb, nb), likely Ry or eV
                kin_ion_unit = h5["kin_ion"].attrs.get("units", "unknown")
        print(f"  kin_ion.shape={kin_ion.shape}, dtype={kin_ion.dtype}, unit attr='{kin_ion_unit}'")
        kin_ion_diag = np.real(np.diagonal(kin_ion[:, :nb_show, :nb_show], axis1=1, axis2=2))
        # Heuristic: convert to eV if values look like Ry
        scale = 1.0
        if abs(kin_ion_diag).max() < 200.0:
            scale = RY2EV
            print(f"  values look like Ry, multiplying by {RY2EV}")
        kin_ion_diag_eV = kin_ion_diag * scale

    # Try to load QE kih.dat (BGW reference, eV)
    print(f"\nLoading QE kih reference: {KIH_REF}", flush=True)
    qe_kih = None
    qe_kih_path_tried = []
    for cand in [KIH_REF, "../qe/nscf/kih.dat", "../../qe/nscf/kih.dat",
                 "../00_lorrax_cohsex/kih.dat", "../00_bgw_cohsex/kih.dat"]:
        qe_kih_path_tried.append(cand)
        if os.path.exists(cand):
            print(f"  found {cand}")
            kpts_ref = np.asarray(sym.unfolded_kpts, dtype=np.float64)
            qe_kih = parse_kih_dat(cand, nk, nb_show, kpts_ref=kpts_ref)
            break
    if qe_kih is None:
        print(f"  no kih.dat found (tried {qe_kih_path_tried})")

    # Parse LORRAX sigma_diag.dat for hartree_k V_H per band
    sd_path = "sigma_diag.dat"
    lor_VH = None
    if os.path.exists(sd_path):
        lor_VH = parse_sigma_diag_VH(sd_path, nk, nb_show)
        print(f"  LORRAX hartree_k V_H from sigma_diag.dat: shape {lor_VH.shape}")

    # Print table for k=0
    print(f"\n=== k=0 diagonal comparison (eV) ===")
    print(f"{'ib':>3}  {'V_H_2D':>11}  {'V_H_LOR':>11}  {'LOR-2D':>11}  ", end="")
    if qe_kih is not None and kin_ion is not None:
        print(f"{'QE-kin_ion':>11}", end="")
    print()
    print("-" * 80)
    for ib in range(nb_show):
        row = f"{ib:>3}  {V_H_2D_diag_eV[0, ib]:>11.4f}  "
        if lor_VH is not None and ib < lor_VH.shape[1] and not np.isnan(lor_VH[0, ib]):
            d = lor_VH[0, ib] - V_H_2D_diag_eV[0, ib]
            row += f"{lor_VH[0, ib]:>11.4f}  {d:>+11.4f}  "
        else:
            row += f"{'-':>11}  {'-':>11}  "
        if qe_kih is not None and ib < qe_kih.shape[1] and kin_ion is not None:
            impl_VH = qe_kih[0, ib] - kin_ion_diag_eV[0, ib]
            row += f"{impl_VH:>+11.4f}"
        print(row)
    if lor_VH is not None:
        d = lor_VH[0, :nb_show] - V_H_2D_diag_eV[0, :nb_show]
        d = d[~np.isnan(d)]
        print(f"\n  LOR_VH - V_H_PW_2D, k=0:  mean={d.mean():+.4f}  std={d.std():.4f}  max|.|={np.abs(d).max():.4f} eV")
        # All-k summary
        if lor_VH.shape[0] >= nk:
            d_all = lor_VH[:nk, :nb_show] - V_H_2D_diag_eV[:nk, :nb_show]
            d_all = d_all[~np.isnan(d_all)]
            print(f"  LOR_VH - V_H_PW_2D, all k: mean={d_all.mean():+.4f}  std={d_all.std():.4f}  max|.|={np.abs(d_all).max():.4f} eV")

    # Summary stats for V_H_3D vs implied V_H from QE - LORRAX kin_ion
    if qe_kih is not None and kin_ion is not None:
        impl_VH = qe_kih[0, :nb_show] - kin_ion_diag_eV[0, :nb_show]
        d3 = V_H_3D_diag_eV[0] - impl_VH
        d2 = V_H_2D_diag_eV[0] - impl_VH
        print()
        print(f"Per-band residual V_H_PW(3D) - (QE_kih - LORRAX_kin_ion), k=0:")
        print(f"  mean = {d3.mean():+.4f}  std = {d3.std():.4f}  max|.| = {np.abs(d3).max():.4f} eV")
        print(f"Per-band residual V_H_PW(2D) - (QE_kih - LORRAX_kin_ion), k=0:")
        print(f"  mean = {d2.mean():+.4f}  std = {d2.std():.4f}  max|.| = {np.abs(d2).max():.4f} eV")

    # Save artifact
    out = "probe_pw_hartree.npz"
    save = {
        "V_H_3D_diag_eV": V_H_3D_diag_eV,
        "V_H_2D_diag_eV": V_H_2D_diag_eV,
        "rho_v_int_charge": float(integrated_charge),
    }
    if kin_ion is not None:
        save["kin_ion_diag_eV"] = kin_ion_diag_eV
    if qe_kih is not None:
        save["qe_kih_eV"] = qe_kih
    np.savez(out, **save)
    print(f"\nSaved {out}")


def parse_sigma_diag_VH(path: str, nk_expect: int, nb_expect: int) -> np.ndarray:
    """Parse 'VH=...' field from sigma_diag.dat per (k, n)."""
    out = np.full((nk_expect, nb_expect), np.nan)
    cur_k = -1
    import re as _re
    re_k = _re.compile(r"^k-point\s+(\d+):")
    re_n = _re.compile(r"^n=\s*(\d+).*VH=\s*([-+0-9.eE]+)")
    with open(path) as f:
        for line in f:
            mk = re_k.match(line.strip())
            if mk:
                cur_k = int(mk.group(1))
                continue
            mn = re_n.match(line.strip())
            if mn and 0 <= cur_k < nk_expect:
                ib = int(mn.group(1))
                if 0 <= ib < nb_expect:
                    out[cur_k, ib] = float(mn.group(2))
    return out


def parse_kih_dat(path: str, nk_expect: int, nb_expect: int,
                  kpts_ref: np.ndarray | None = None) -> np.ndarray | None:
    """Parse BGW kih.dat written by pw2bgw.

    Format:
        kx ky kz nbnd ispin    <- header per k-block
        spin_idx band_idx val_re val_im   <- nbnd*nspin rows
    Returns array (nk_expect, nb_expect) in eV, NaN where unread.
    Reorders to match kpts_ref if provided (frac coords (nk_expect, 3)).
    """
    blocks = []  # list of (kpt, table[band] = val)
    cur_k = None
    cur_tbl = {}
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) == 5:
                # header line
                if cur_k is not None:
                    blocks.append((cur_k, cur_tbl))
                cur_k = tuple(float(x) for x in parts[:3])
                cur_tbl = {}
            elif len(parts) == 4:
                ispin = int(parts[0])
                ib = int(parts[1]) - 1
                vr = float(parts[2])
                if ispin == 1:  # take spin-up only (collinear ↑ = ↓ here)
                    cur_tbl[ib] = vr
    if cur_k is not None:
        blocks.append((cur_k, cur_tbl))

    print(f"  parsed {len(blocks)} k-blocks from {path}")
    if not blocks:
        return None

    out = np.full((nk_expect, nb_expect), np.nan)
    if kpts_ref is not None:
        # reorder: for each LORRAX k-index, find matching BGW k
        for ik_lor, k_lor in enumerate(kpts_ref):
            best = None
            best_d = 1e9
            for ib_idx, (kbgw, tbl) in enumerate(blocks):
                d = sum(min((float(k_lor[a]) - kbgw[a]) % 1.0,
                           (kbgw[a] - float(k_lor[a])) % 1.0) for a in range(3))
                if d < best_d:
                    best_d, best = d, ib_idx
            if best is not None and best_d < 1e-3:
                tbl = blocks[best][1]
                for ib in range(nb_expect):
                    if ib in tbl:
                        out[ik_lor, ib] = tbl[ib]
    else:
        # naive order
        for ik in range(min(nk_expect, len(blocks))):
            tbl = blocks[ik][1]
            for ib in range(nb_expect):
                if ib in tbl:
                    out[ik, ib] = tbl[ib]
    return out


if __name__ == "__main__":
    main()
