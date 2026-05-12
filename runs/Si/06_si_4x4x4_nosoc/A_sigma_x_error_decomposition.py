"""Decompose Gamma Σ_X error into pair-fit and persisted-V pieces.

Accumulated over all 64 q-points:

  exact_BGW_v     exact pair products + BGW write_vcoul body
  isdf_BGW_v      persisted zeta pair reconstruction + BGW write_vcoul body
  isdf_persist_V  persisted zeta pair reconstruction + persisted V_qmunu

The last line should reproduce LORRAX's printed x-only Sigma_X.  The
differences isolate whether the error is in zeta/pair reconstruction or in
the Coulomb matrix that was actually persisted.
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
from file_io.read_bgw_vcoul import fill_v_grid_for_q, read_bgw_vcoul


ROOT = Path("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc")
WFN_PATH = ROOT / "qe/nscf/WFN.h5"
VCOUL_PATH = ROOT / "D_bgw_cohsex/vcoul"
SIGMA_HP = ROOT / "D_bgw_cohsex/sigma_hp.log"
LOR_RUN = ROOT / "D_lorrax_xonly_overlay"
ZETA_H5 = LOR_RUN / "tmp/zeta_q.h5"
ISDF_H5 = LOR_RUN / "tmp/isdf_tensors_480.h5"
CENTROIDS = LOR_RUN / "centroids_frac_480.txt"
GW_OUT = LOR_RUN / "gw.out"
N_VAL = 8
N_SIGMA = 16
RY2EV = 13.605693122994


def as_complex(arr):
    arr = np.asarray(arr)
    if arr.dtype.fields and "r" in arr.dtype.fields and "i" in arr.dtype.fields:
        return arr["r"] + 1j * arr["i"]
    return arr


def parse_bgw_x_gamma(path: Path) -> np.ndarray:
    vals = []
    in_gamma = False
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        m = re.match(r"k\s*=\s*([-\d.Ee+ ]+?)\s+ik\s*=\s*(\d+)", s)
        if m:
            in_gamma = np.linalg.norm([float(x) for x in m.group(1).split()]) < 1.0e-10
            continue
        if in_gamma:
            p = s.split()
            if len(p) >= 5 and p[0].isdigit():
                vals.append(float(p[3]))
                if len(vals) == N_SIGMA:
                    return np.asarray(vals)
    raise RuntimeError("Could not parse BGW X")


def parse_lorrax_printed_x(path: Path) -> np.ndarray:
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Bare Σ_X diagonal (eV), k=0:" in line:
            return np.asarray([float(x) for x in line.split(":", 1)[1].split()])
    raise RuntimeError("Could not parse LORRAX printed X")


def parse_x_head_ry(path: Path) -> float:
    for line in path.read_text(encoding="utf-8").splitlines():
        if "Σ^X head (occ)" in line:
            return float(line.split("=")[1].split()[0])
    raise RuntimeError("Could not parse X head")


def load_centroids(path: Path, fft_grid):
    vals = np.loadtxt(path, dtype=np.float64)[:, :3]
    grid = np.asarray(fft_grid, dtype=np.int64)
    if np.max(np.abs(vals - np.rint(vals))) < 1.0e-8:
        xyz = vals.astype(np.int64)
    else:
        xyz = np.mod(np.rint(vals * grid[None, :]), grid[None, :]).astype(np.int64)
    flat = xyz[:, 0] * (grid[1] * grid[2]) + xyz[:, 1] * grid[2] + xyz[:, 2]
    return xyz, flat.astype(np.int64)


def q_phase(q_int, kgrid, fft_grid, sign):
    nx, ny, nz = (int(x) for x in fft_grid)
    fx = np.arange(nx)[None, :, None, None] / nx
    fy = np.arange(ny)[None, None, :, None] / ny
    fz = np.arange(nz)[None, None, None, :] / nz
    qf = np.asarray(q_int, dtype=np.float64) / np.asarray(kgrid, dtype=np.float64)
    return np.exp(sign * 2j * np.pi * (qf[0] * fx + qf[1] * fy + qf[2] * fz))


def build_u_grid(wfn, sym, ik, nbands):
    nx, ny, nz = (int(x) for x in wfn.fft_grid)
    n_grid = nx * ny * nz
    gv = np.asarray(sym.get_gvecs_kfull(wfn, ik))
    cnk = sym.get_cnk_fullzone_batch(wfn, np.arange(nbands), ik)
    box = np.zeros((nbands, int(wfn.nspinor), nx, ny, nz), dtype=np.complex128)
    box[:, :, gv[:, 0], gv[:, 1], gv[:, 2]] = cnk
    return np.fft.ifftn(box, axes=(2, 3, 4)) * np.sqrt(n_grid)


def contrib_from_M(M, v_box):
    return -np.einsum("vnxyz,vnxyz,xyz->n", M.conj(), M, v_box, optimize=True).real


def main() -> int:
    wfn = WFNReader(str(WFN_PATH))
    sym = SymMaps(wfn)
    fft_grid = tuple(int(x) for x in wfn.fft_grid)
    kgrid = tuple(int(x) for x in wfn.kgrid)
    nx, ny, nz = fft_grid
    n_grid = nx * ny * nz
    _, cflat = load_centroids(CENTROIDS, fft_grid)
    table = read_bgw_vcoul(str(VCOUL_PATH))
    table_gamma = type(table)(
        q_fracs=table.q_fracs[:8].copy(),
        G_miller_per_q=table.G_miller_per_q[:8],
        vcoul_per_q=table.vcoul_per_q[:8],
    )

    kints = np.asarray(sym.kvecs_asints, dtype=np.int64)
    lookup = {tuple(k): i for i, k in enumerate(kints)}
    gamma = lookup[(0, 0, 0)]
    u_gamma = build_u_grid(wfn, sym, gamma, N_SIGMA)

    exact_bgw = np.zeros(N_SIGMA)
    isdf_bgw = np.zeros(N_SIGMA)
    persist = np.zeros(N_SIGMA)
    x_head_ry = parse_x_head_ry(GW_OUT)

    with h5py.File(ZETA_H5, "r") as zh5, h5py.File(ISDF_H5, "r") as vh5:
        zd = zh5["zeta_q"]
        Vp = as_complex(vh5["V_qmunu"][0, 0, 0])
        psi_h5 = as_complex(vh5["psi_full_y"][:])
        for iq, q_int in enumerate(kints):
            q_mod = np.mod(q_int, np.asarray(kgrid, dtype=np.int64))
            qx, qy, qz = (int(x) for x in q_mod)
            inner = lookup[tuple(np.mod(-q_mod, np.asarray(kgrid, dtype=np.int64)))]
            u_v = build_u_grid(wfn, sym, inner, N_VAL)
            pair_exact = np.einsum("vsxyz,nsxyz->vnxyz", np.conj(u_v), u_gamma, optimize=True)
            M_exact = np.fft.fftn(pair_exact, axes=(2, 3, 4))

            pair_mu = np.einsum(
                "vsu,nsu->vnu",
                np.conj(psi_h5[inner, :N_VAL]),
                psi_h5[gamma, :N_SIGMA],
                optimize=True,
            )
            zeta = as_complex(zd[qx * (kgrid[1] * kgrid[2]) + qy * kgrid[2] + qz, :, :]).T
            zeta_G = np.fft.fftn(
                zeta.reshape(zeta.shape[0], nx, ny, nz) * q_phase(q_mod, kgrid, fft_grid, -1),
                axes=(1, 2, 3),
            ).reshape(zeta.shape[0], nx, ny, nz)

            q_frac = q_mod / np.asarray(kgrid, dtype=np.float64)
            v_box = fill_v_grid_for_q(
                table_gamma,
                q_frac,
                fft_grid=fft_grid,
                cell_volume=float(wfn.cell_volume),
                sym_mats_k=sym.sym_mats_k,
            )
            inner_frac = kints[inner] / np.asarray(kgrid, dtype=np.float64)
            g_unwrap = np.rint(inner_frac - (-q_frac)).astype(int)
            v_box_direct = np.roll(v_box, shift=tuple(g_unwrap), axis=(0, 1, 2))
            exact_bgw += contrib_from_M(M_exact, v_box_direct) / int(sym.nk_tot)

            zeta_weighted = zeta_G * np.sqrt(np.maximum(v_box, 0.0))[None, :, :, :]
            V_bgw = np.einsum("uxyz,vxyz->uv", np.conj(zeta_weighted), zeta_weighted, optimize=True)
            isdf_bgw += (
                -np.einsum("vnu,uw,vnw->n", np.conj(pair_mu), V_bgw, pair_mu, optimize=True).real
                / int(sym.nk_tot)
            )
            persist += (
                -np.einsum("vnu,uw,vnw->n", np.conj(pair_mu), Vp[qx, qy, qz], pair_mu, optimize=True).real
                / int(sym.nk_tot)
            )
            if (iq + 1) % 16 == 0:
                print(f"  processed {iq + 1}/64 q-points", flush=True)

    for arr in (exact_bgw, isdf_bgw, persist):
        arr[:N_VAL] += x_head_ry
    exact_ev = exact_bgw * RY2EV
    isdf_bgw_ev = isdf_bgw * RY2EV
    persist_ev = persist * RY2EV
    bgw_ref = parse_bgw_x_gamma(SIGMA_HP)
    lor_print = parse_lorrax_printed_x(GW_OUT)

    print()
    print(f"Using LORRAX source: {sys.path[0]}")
    print("Stage errors vs BGW Gamma X (meV):")
    for label, vals in (
        ("exact_BGW_v", exact_ev),
        ("isdf_BGW_v", isdf_bgw_ev),
        ("isdf_persist_V", persist_ev),
    ):
        d = (vals - bgw_ref) * 1000.0
        print(
            f"{label:16s} val_MAE={np.mean(np.abs(d[:N_VAL])):9.3f} "
            f"val_max={np.max(np.abs(d[:N_VAL])):9.3f} "
            f"cond_MAE={np.mean(np.abs(d[N_VAL:])):9.3f} "
            f"cond_max={np.max(np.abs(d[N_VAL:])):9.3f}"
        )
    print()
    print("Component deltas (meV):")
    zeta_d = (isdf_bgw_ev - exact_ev) * 1000.0
    v_d = (persist_ev - isdf_bgw_ev) * 1000.0
    print(f"  zeta/pair only: val_MAE={np.mean(np.abs(zeta_d[:N_VAL])):.3f}, cond_MAE={np.mean(np.abs(zeta_d[N_VAL:])):.3f}")
    print(f"  persisted V:    val_MAE={np.mean(np.abs(v_d[:N_VAL])):.3f}, cond_MAE={np.mean(np.abs(v_d[N_VAL:])):.3f}")
    print(f"  printed-vs-persist max diff: {np.max(np.abs(lor_print - persist_ev[:len(lor_print)]))*1000.0:.3f} meV")
    print()
    print(" n  BGW       exact_BGW  isdf_BGW   persist    zeta_d  V_d  total_d")
    for i in range(N_SIGMA):
        print(
            f"{i+1:2d} {bgw_ref[i]:10.6f} {exact_ev[i]:10.6f} {isdf_bgw_ev[i]:10.6f} "
            f"{persist_ev[i]:10.6f} {zeta_d[i]:8.3f} {v_d[i]:8.3f} "
            f"{(persist_ev[i]-bgw_ref[i])*1000.0:8.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
