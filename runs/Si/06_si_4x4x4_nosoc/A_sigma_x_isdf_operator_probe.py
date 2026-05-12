"""Read-only probe for the Si 4x4x4 no-SOC LORRAX bare-X error.

This script compares three Gamma-point exchange evaluations:

1. BGW's sigma_hp.log X column.
2. The persisted LORRAX ISDF operator: psi_full_y + V_qmunu from the
   completed x-only run.
3. The same persisted V_qmunu contracted with centroid wavefunctions built
   without the Bloch e^{ikr} phase, as a convention toggle.

It also verifies that psi_full_y matches the WFN transform used by
common.load_wfns.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A/src")

from runtime import set_default_env

set_default_env()

from common.symmetry_maps import SymMaps
from file_io import WFNReader


ROOT = Path("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc")
WFN_PATH = ROOT / "qe/nscf/WFN.h5"
BGW_SIGMA_HP = ROOT / "D_bgw_cohsex/sigma_hp.log"
LOR_RUN = ROOT / "D_lorrax_xonly_overlay"
ISDF_H5 = LOR_RUN / "tmp/isdf_tensors_480.h5"
CENTROIDS = LOR_RUN / "centroids_frac_480.txt"
GW_OUT = LOR_RUN / "gw.out"

N_VAL = 8
N_SIGMA = 16
N_BANDS = 60
RY2EV = 13.605693122994


def as_complex(arr):
    """Convert h5py compound complex datasets to numpy complex arrays."""
    arr = np.asarray(arr)
    if arr.dtype.fields and "r" in arr.dtype.fields and "i" in arr.dtype.fields:
        return arr["r"] + 1j * arr["i"]
    return arr


def parse_bgw_x_gamma(path: Path, n_bands: int) -> np.ndarray:
    vals = []
    in_gamma = False
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            m = re.match(r"k\s*=\s*([-\d.Ee+ ]+?)\s+ik\s*=\s*(\d+)", s)
            if m:
                k = np.array([float(x) for x in m.group(1).split()])
                in_gamma = np.linalg.norm(k) < 1.0e-10
                continue
            if not in_gamma:
                continue
            p = s.split()
            if len(p) >= 5 and p[0].isdigit():
                vals.append(float(p[3]))
                if len(vals) == n_bands:
                    return np.asarray(vals, dtype=np.float64)
    raise RuntimeError(f"Could not parse {n_bands} Gamma X values from {path}")


def parse_lorrax_printed_x(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if "Bare Σ_X diagonal (eV), k=0:" in line:
                return np.asarray([float(x) for x in line.split(":", 1)[1].split()])
    raise RuntimeError(f"Could not parse printed LORRAX X from {path}")


def parse_lorrax_x_head_ry(path: Path) -> float:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if "Σ^X head (occ)" in line:
                return float(line.split("=")[1].split()[0])
    raise RuntimeError(f"Could not parse Σ^X head from {path}")


def load_centroid_indices(path: Path, fft_grid) -> np.ndarray:
    vals = np.loadtxt(path, dtype=np.float64)
    if vals.ndim == 1:
        vals = vals[None, :]
    # Centroid files in this sandbox are either flat integer indices or
    # integer xyz grid rows. Accept both.
    if vals.shape[1] == 1:
        flat = vals[:, 0].astype(np.int64)
        return flat
    if vals.shape[1] >= 3:
        xyz = vals[:, :3]
        if np.max(np.abs(xyz - np.rint(xyz))) < 1.0e-8:
            return xyz.astype(np.int64)
        return np.mod(np.rint(xyz * np.asarray(fft_grid)[None, :]), np.asarray(fft_grid)[None, :]).astype(np.int64)
    raise ValueError(f"Unrecognized centroid file shape {vals.shape}")


def centroid_flat_indices(centroids, fft_grid):
    nx, ny, nz = fft_grid
    centroids = np.asarray(centroids)
    if centroids.ndim == 1:
        return centroids.astype(np.int64)
    return (centroids[:, 0] * (ny * nz) + centroids[:, 1] * nz + centroids[:, 2]).astype(np.int64)


def build_centroid_wfns(wfn, sym, centroids_xyz_or_flat, nbands: int):
    nx, ny, nz = (int(x) for x in wfn.fft_grid)
    n_grid = nx * ny * nz
    n_k = int(sym.nk_tot)
    nspinor = int(wfn.nspinor)
    flat_idx = centroid_flat_indices(centroids_xyz_or_flat, (nx, ny, nz))
    n_mu = len(flat_idx)

    kgrid = np.asarray(wfn.kgrid, dtype=np.float64)
    kvecs_frac = np.asarray(sym.kvecs_asints, dtype=np.float64) / kgrid[None, :]
    fx = np.arange(nx, dtype=np.float64)[None, :, None, None] / nx
    fy = np.arange(ny, dtype=np.float64)[None, None, :, None] / ny
    fz = np.arange(nz, dtype=np.float64)[None, None, None, :] / nz

    periodic = np.zeros((n_k, nbands, nspinor, n_mu), dtype=np.complex128)
    full = np.zeros_like(periodic)
    for ik in range(n_k):
        gv = np.asarray(sym.get_gvecs_kfull(wfn, ik))
        cnk = sym.get_cnk_fullzone_batch(wfn, np.arange(nbands), ik)
        box = np.zeros((nbands, nspinor, nx, ny, nz), dtype=np.complex128)
        box[:, :, gv[:, 0], gv[:, 1], gv[:, 2]] = cnk
        # LORRAX convention: normalized periodic part u(r).
        u_r = np.fft.ifftn(box, axes=(2, 3, 4)) * np.sqrt(n_grid)
        phase = np.exp(
            2j
            * np.pi
            * (
                kvecs_frac[ik, 0] * fx
                + kvecs_frac[ik, 1] * fy
                + kvecs_frac[ik, 2] * fz
            )
        )
        periodic[ik] = u_r.reshape(nbands, nspinor, n_grid)[:, :, flat_idx]
        full[ik] = (u_r * phase).reshape(nbands, nspinor, n_grid)[:, :, flat_idx]
    return periodic, full


def sigma_x_from_isdf(psi, V_qmunu, sym, x_head_ry: float, n_sigma: int) -> np.ndarray:
    """Compute Gamma diagonal X using persisted ISDF V and centroid psi.

    psi is (nk, nb, ns, n_mu), V_qmunu is (4,4,4,n_mu,n_mu).
    """
    kgrid = np.asarray(sym.kvecs_asints.max(axis=0) + 1, dtype=np.int64)
    # Prefer the WFN kgrid shape over max+1 for robustness.
    kgrid = np.asarray((4, 4, 4), dtype=np.int64)
    n_k = int(sym.nk_tot)
    gamma = int(np.argmin(np.linalg.norm(np.asarray(sym.unfolded_kpts), axis=1)))
    out = np.zeros(n_sigma, dtype=np.float64)

    lookup = {tuple(k): i for i, k in enumerate(np.asarray(sym.kvecs_asints, dtype=np.int64))}
    for q_int in np.asarray(sym.kvecs_asints, dtype=np.int64):
        qx, qy, qz = (int(x) for x in q_int)
        inner_key = tuple(np.mod(-q_int, kgrid))
        ik_inner = lookup[inner_key]
        V = V_qmunu[qx, qy, qz]
        psi_n = psi[gamma, :n_sigma]
        psi_v = psi[ik_inner, :N_VAL]
        # pair[v,n,mu] = sum_s conj(psi_v) * psi_n
        pair = np.einsum("vsu,nsu->vnu", np.conj(psi_v), psi_n, optimize=True)
        out += -np.einsum("vnu,uw,vnw->n", np.conj(pair), V, pair, optimize=True).real / n_k

    # LORRAX handles the true q=0/G=0 exchange head analytically, and only
    # occupied target bands have <v|n> overlap for this term.
    out[:N_VAL] += x_head_ry
    return out * RY2EV


def main() -> int:
    wfn = WFNReader(str(WFN_PATH))
    sym = SymMaps(wfn)
    centroids = load_centroid_indices(CENTROIDS, wfn.fft_grid)
    bgw = parse_bgw_x_gamma(BGW_SIGMA_HP, N_SIGMA)
    lor_printed = parse_lorrax_printed_x(GW_OUT)
    x_head_ry = parse_lorrax_x_head_ry(GW_OUT)

    with h5py.File(ISDF_H5, "r") as h5:
        psi_h5 = as_complex(h5["psi_full_y"][:])
        V_qmunu = as_complex(h5["V_qmunu"][0, 0, 0])

    psi_periodic, psi_full = build_centroid_wfns(wfn, sym, centroids, N_BANDS)
    psi_h5_err = np.max(np.abs(psi_h5 - psi_full))
    psi_periodic_norm = np.max(np.abs(np.sum(np.abs(psi_periodic[:, :N_SIGMA]) ** 2, axis=(2, 3)) - 1.0))
    psi_full_norm = np.max(np.abs(np.sum(np.abs(psi_full[:, :N_SIGMA]) ** 2, axis=(2, 3)) - 1.0))

    sig_h5 = sigma_x_from_isdf(psi_h5, V_qmunu, sym, x_head_ry, N_SIGMA)
    sig_full = sigma_x_from_isdf(psi_full, V_qmunu, sym, x_head_ry, N_SIGMA)
    sig_periodic = sigma_x_from_isdf(psi_periodic, V_qmunu, sym, x_head_ry, N_SIGMA)

    print(f"Using LORRAX source: {sys.path[0]}")
    print(f"ntran={wfn.ntran} nk_tot={sym.nk_tot} nelec={wfn.nelec} nspinor={wfn.nspinor}")
    print(f"psi_full_y vs rebuilt full-Bloch centroid psi: max |Δ| = {psi_h5_err:.3e}")
    print(f"centroid-only norm diagnostic (not expected to be 1): periodic max |sum_mu|ψ|²-1|={psi_periodic_norm:.3e}, full={psi_full_norm:.3e}")
    print(f"parsed LORRAX X head = {x_head_ry:.9f} Ry = {x_head_ry * RY2EV:.6f} eV")
    print()
    print("Reproduce printed LORRAX Γ X from persisted ISDF tensors:")
    print(" n   printed      h5_contract   diff_meV")
    for i, (a, b) in enumerate(zip(lor_printed, sig_h5[: len(lor_printed)]), start=1):
        print(f"{i:2d} {a:11.6f} {b:12.6f} {(b-a)*1000.0:10.3f}")
    print()
    print("Γ X comparison vs BGW (meV):")
    for label, vals in (
        ("h5_full_phase", sig_h5),
        ("rebuilt_full_phase", sig_full),
        ("periodic_no_phase", sig_periodic),
    ):
        diff = (vals - bgw) * 1000.0
        print(
            f"{label:20s} "
            f"val_MAE={np.mean(np.abs(diff[:N_VAL])):9.3f} "
            f"val_max={np.max(np.abs(diff[:N_VAL])):9.3f} "
            f"cond_MAE={np.mean(np.abs(diff[N_VAL:])):9.3f} "
            f"cond_max={np.max(np.abs(diff[N_VAL:])):9.3f}"
        )
    print()
    print(" n      BGW_X       h5_ISDF      periodic     h5-BGW meV")
    for i in range(N_SIGMA):
        print(
            f"{i+1:2d} {bgw[i]:11.6f} {sig_h5[i]:11.6f} "
            f"{sig_periodic[i]:11.6f} {(sig_h5[i]-bgw[i])*1000.0:11.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
