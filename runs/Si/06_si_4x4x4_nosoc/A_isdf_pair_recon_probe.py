"""Probe ISDF reconstruction of exact exchange pair products.

For selected Gamma exchange q-points, compare exact periodic pair Fourier
components

    M_vn(G) = FFT[conj(u_v,k-q) u_n,k](G)

against the reconstruction implied by the persisted LORRAX zeta:

    M_vn^ISDF(G) = sum_mu rho_vn(r_mu) FFT[exp(-iqr) zeta_q,mu](G)

where rho_vn(r_mu) uses full Bloch phases.  The script also toggles q -> -q
and conjugation of zeta to catch sign/convention mistakes.
"""

from __future__ import annotations

import os
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
from file_io.read_bgw_vcoul import fill_v_grid_for_q, read_bgw_vcoul


ROOT = Path("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc")
WFN_PATH = ROOT / "qe/nscf/WFN.h5"
VCOUL_PATH = ROOT / "D_bgw_cohsex/vcoul"
LOR_RUN = ROOT / "D_lorrax_xonly_overlay"
ZETA_H5 = LOR_RUN / "tmp/zeta_q.h5"
ISDF_H5 = LOR_RUN / "tmp/isdf_tensors_480.h5"
CENTROIDS = LOR_RUN / "centroids_frac_480.txt"
N_VAL = 8
N_SIGMA = 16
N_BANDS = 60
RY2EV = 13.605693122994


def as_complex(arr):
    arr = np.asarray(arr)
    if arr.dtype.fields and "r" in arr.dtype.fields and "i" in arr.dtype.fields:
        return arr["r"] + 1j * arr["i"]
    return arr


def load_centroids(path: Path, fft_grid) -> tuple[np.ndarray, np.ndarray]:
    vals = np.loadtxt(path, dtype=np.float64)
    xyz = vals[:, :3]
    grid = np.asarray(fft_grid, dtype=np.int64)
    if np.max(np.abs(xyz - np.rint(xyz))) < 1.0e-8:
        idx = xyz.astype(np.int64)
    else:
        idx = np.mod(np.rint(xyz * grid[None, :]), grid[None, :]).astype(np.int64)
    flat = idx[:, 0] * (grid[1] * grid[2]) + idx[:, 1] * grid[2] + idx[:, 2]
    return idx, flat.astype(np.int64)


def build_u_grid(wfn, sym, ik: int, nbands: int) -> np.ndarray:
    nx, ny, nz = (int(x) for x in wfn.fft_grid)
    n_grid = nx * ny * nz
    gv = np.asarray(sym.get_gvecs_kfull(wfn, ik))
    cnk = sym.get_cnk_fullzone_batch(wfn, np.arange(nbands), ik)
    box = np.zeros((nbands, int(wfn.nspinor), nx, ny, nz), dtype=np.complex128)
    box[:, :, gv[:, 0], gv[:, 1], gv[:, 2]] = cnk
    return np.fft.ifftn(box, axes=(2, 3, 4)) * np.sqrt(n_grid)


def q_phase(q_int, kgrid, fft_grid, sign: int) -> np.ndarray:
    nx, ny, nz = (int(x) for x in fft_grid)
    fx = np.arange(nx, dtype=np.float64)[None, :, None, None] / nx
    fy = np.arange(ny, dtype=np.float64)[None, None, :, None] / ny
    fz = np.arange(nz, dtype=np.float64)[None, None, None, :] / nz
    qfrac = np.asarray(q_int, dtype=np.float64) / np.asarray(kgrid, dtype=np.float64)
    return np.exp(sign * 2j * np.pi * (qfrac[0] * fx + qfrac[1] * fy + qfrac[2] * fz))


def zeta_fft_for_q(zeta_dset, q_int, kgrid, fft_grid, variant: str) -> np.ndarray:
    nkx, nky, nkz = (int(x) for x in kgrid)
    q_mod = np.mod(q_int, np.asarray(kgrid, dtype=np.int64))
    if "minusq" in variant:
        q_read = np.mod(-q_mod, np.asarray(kgrid, dtype=np.int64))
    else:
        q_read = q_mod
    q_flat = int(q_read[0] * (nky * nkz) + q_read[1] * nkz + q_read[2])
    zeta = as_complex(zeta_dset[q_flat, :, :]).T
    if "conj" in variant:
        zeta = np.conj(zeta)
    phase = q_phase(q_mod, kgrid, fft_grid, sign=-1)
    nmu = zeta.shape[0]
    nx, ny, nz = (int(x) for x in fft_grid)
    return np.fft.fftn(zeta.reshape(nmu, nx, ny, nz) * phase, axes=(1, 2, 3)).reshape(nmu, nx, ny, nz)


def contribution_from_M(M, v_box):
    return -np.einsum("vnxyz,vnxyz,xyz->n", M.conj(), M, v_box, optimize=True).real


def main() -> int:
    wfn = WFNReader(str(WFN_PATH))
    sym = SymMaps(wfn)
    table = read_bgw_vcoul(str(VCOUL_PATH))
    table_gamma = type(table)(
        q_fracs=table.q_fracs[:8].copy(),
        G_miller_per_q=table.G_miller_per_q[:8],
        vcoul_per_q=table.vcoul_per_q[:8],
    )
    fft_grid = tuple(int(x) for x in wfn.fft_grid)
    kgrid = tuple(int(x) for x in wfn.kgrid)
    nx, ny, nz = fft_grid
    n_grid = nx * ny * nz
    centroid_xyz, centroid_flat = load_centroids(CENTROIDS, fft_grid)

    kints = np.asarray(sym.kvecs_asints, dtype=np.int64)
    lookup = {tuple(k): i for i, k in enumerate(kints)}
    gamma = lookup[(0, 0, 0)]
    u_gamma = build_u_grid(wfn, sym, gamma, N_SIGMA)
    u_gamma_mu = u_gamma.reshape(N_SIGMA, int(wfn.nspinor), n_grid)[:, :, centroid_flat]

    variants = ("q", "minusq", "q_conj", "minusq_conj")
    q_tests = [
        np.array([0, 0, 0], dtype=np.int64),
        np.array([0, 0, 1], dtype=np.int64),
        np.array([0, 1, 1], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int64),
    ]
    print(f"Using LORRAX source: {sys.path[0]}")
    print(f"FFT={fft_grid} kgrid={kgrid} nmu={len(centroid_flat)}")
    print("Per-q exact-vs-zeta reconstruction errors for Gamma X.")
    print("Errors are contribution differences in meV before the outer 1/Nk average.")
    print()

    with h5py.File(ZETA_H5, "r") as h5:
        with h5py.File(ISDF_H5, "r") as vh5:
            V_persist = as_complex(vh5["V_qmunu"][0, 0, 0])
        zeta_dset = h5["zeta_q"]
        for q_int in q_tests:
            q_mod = np.mod(q_int, np.asarray(kgrid, dtype=np.int64))
            inner = lookup[tuple(np.mod(-q_mod, np.asarray(kgrid, dtype=np.int64)))]
            u_v = build_u_grid(wfn, sym, inner, N_VAL)
            pair_exact_r = np.einsum("vsxyz,nsxyz->vnxyz", np.conj(u_v), u_gamma, optimize=True)
            M_exact = np.fft.fftn(pair_exact_r, axes=(2, 3, 4))

            phase_plus_mu = q_phase(q_mod, kgrid, fft_grid, sign=+1).reshape(1, nx, ny, nz)
            u_v_mu = u_v.reshape(N_VAL, int(wfn.nspinor), n_grid)[:, :, centroid_flat]
            # Full-Bloch pair at centroids: exp(+iqr) conj(u_v) u_n.
            phase_mu = phase_plus_mu.reshape(n_grid)[centroid_flat]
            pair_mu = (
                np.einsum("vsu,nsu->vnu", np.conj(u_v_mu), u_gamma_mu, optimize=True)
                * phase_mu[None, None, :]
            )

            q_frac = q_mod / np.asarray(kgrid, dtype=np.float64)
            v_box = fill_v_grid_for_q(
                table_gamma,
                q_frac,
                fft_grid=fft_grid,
                cell_volume=float(wfn.cell_volume),
                sym_mats_k=sym.sym_mats_k,
            )
            exact_ev = contribution_from_M(M_exact, v_box) * RY2EV
            print(f"q_int={tuple(q_mod)} inner_k={inner} exact val_sum={exact_ev[:N_VAL].sum():+.6f} eV")
            for variant in variants:
                ZG = zeta_fft_for_q(zeta_dset, q_mod, kgrid, fft_grid, variant)
                M_isdf = np.einsum("vnu,uxyz->vnxyz", pair_mu, ZG, optimize=True)
                isdf_ev = contribution_from_M(M_isdf, v_box) * RY2EV
                if variant == "q":
                    qx, qy, qz = (int(x) for x in q_mod)
                    Vq = V_persist[qx, qy, qz]
                    persist_ev = (
                        -np.einsum("vnu,uw,vnw->n", np.conj(pair_mu), Vq, pair_mu, optimize=True).real
                        * RY2EV
                    )
                    pv = (persist_ev - isdf_ev) * 1000.0
                    print(
                        f"  persisted V_qmunu vs zeta+BGW_v: "
                        f"val_MAE={np.mean(np.abs(pv[:N_VAL])):10.3f} "
                        f"cond_MAE={np.mean(np.abs(pv[N_VAL:])):10.3f} "
                        f"band1_d={pv[0]:+10.3f} meV"
                    )
                d = (isdf_ev - exact_ev) * 1000.0
                rel_M = (
                    np.linalg.norm((M_isdf - M_exact).reshape(-1))
                    / max(np.linalg.norm(M_exact.reshape(-1)), 1.0e-30)
                )
                print(
                    f"  {variant:12s} rel|M|={rel_M:9.3e} "
                    f"val_MAE={np.mean(np.abs(d[:N_VAL])):10.3f} "
                    f"cond_MAE={np.mean(np.abs(d[N_VAL:])):10.3f} "
                    f"band1_d={d[0]:+10.3f}"
                )
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
