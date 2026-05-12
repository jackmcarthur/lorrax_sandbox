"""Direct Gamma Σ_X with BGW vcoul vs LORRAX native vcoul.

This isolates the Coulomb-grid part of the larger LORRAX x-only error.
It performs the exact G-space exchange contraction, no ISDF, using either
BGW's write_vcoul body values or the native sqrt_v grid produced by
gw.compute_vcoul.make_v_munu_chunked_kernel.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A/src")

from runtime import set_default_env

set_default_env()

import jax.numpy as jnp

from common.symmetry_maps import SymMaps
from file_io import WFNReader
from file_io.read_bgw_vcoul import fill_v_grid_for_q, read_bgw_vcoul
from gw.compute_vcoul import make_v_munu_chunked_kernel


ROOT = Path("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc")
WFN_PATH = ROOT / "qe/nscf/WFN.h5"
VCOUL_PATH = ROOT / "D_bgw_cohsex/vcoul"
SIGMA_HP = ROOT / "D_bgw_cohsex/sigma_hp.log"
GW_OUT = ROOT / "D_lorrax_xonly_overlay/gw.out"
N_VAL = 8
N_SIGMA = 16
RY2EV = 13.605693122994


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
    raise RuntimeError("Could not parse BGW Gamma X")


def parse_x_head_ry(path: Path) -> float:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if "Σ^X head (occ)" in line:
                return float(line.split("=")[1].split()[0])
    raise RuntimeError("Could not parse LORRAX X head")


def native_v_grid(kernels, q_int, kgrid, fft_grid):
    q_wrapped = np.where(
        np.asarray(q_int, dtype=np.float64) > np.asarray(kgrid, dtype=np.float64) / 2.0,
        np.asarray(q_int, dtype=np.float64) - np.asarray(kgrid, dtype=np.float64),
        np.asarray(q_int, dtype=np.float64),
    )
    sqrt_v, _ = kernels.get_sqrt_v_and_phase(jnp.asarray(q_wrapped))
    sqrt_v = np.asarray(sqrt_v).reshape(-1)
    vals = (sqrt_v.real * sqrt_v.real).astype(np.float64)
    out = np.zeros(int(np.prod(fft_grid)), dtype=np.float64)
    if kernels.sphere_idx is None:
        out[:] = vals
    else:
        out[np.asarray(kernels.sphere_idx)] = vals
    return out.reshape(tuple(int(x) for x in fft_grid))


def main() -> int:
    wfn = WFNReader(str(WFN_PATH))
    sym = SymMaps(wfn)
    nx, ny, nz = (int(v) for v in wfn.fft_grid)
    n_grid = nx * ny * nz
    n_k = int(sym.nk_tot)
    nb = N_VAL + N_SIGMA
    bvec = np.asarray(float(wfn.blat) * wfn.bvec, dtype=np.float64)
    kgrid = tuple(int(x) for x in wfn.kgrid)
    fft_grid = (nx, ny, nz)
    x_head_ry = parse_x_head_ry(GW_OUT)

    psi = np.zeros((n_k, nb, int(wfn.nspinor), nx, ny, nz), dtype=np.complex128)
    for ik in range(n_k):
        gv = np.asarray(sym.get_gvecs_kfull(wfn, ik))
        cnk = sym.get_cnk_fullzone_batch(wfn, np.arange(nb), ik)
        box = np.zeros((nb, int(wfn.nspinor), nx, ny, nz), dtype=np.complex128)
        box[:, :, gv[:, 0], gv[:, 1], gv[:, 2]] = cnk
        # Unnormalized periodic part, matching the original direct probe.
        psi[ik] = np.fft.ifftn(box, axes=(2, 3, 4)) * n_grid

    table = read_bgw_vcoul(str(VCOUL_PATH))
    table_gamma = type(table)(
        q_fracs=table.q_fracs[:8].copy(),
        G_miller_per_q=table.G_miller_per_q[:8],
        vcoul_per_q=table.vcoul_per_q[:8],
    )
    kernels = make_v_munu_chunked_kernel(
        nx,
        ny,
        nz,
        *kgrid,
        bvec,
        float(wfn.cell_volume),
        sys_dim=3,
        bdot=np.asarray(wfn.bdot, dtype=np.float64),
        mc_average_vcoul_body=True,
        vcoul_cutoff_ry=25.0,
    )

    kints = np.asarray(sym.kvecs_asints, dtype=np.int64)
    lookup = {tuple(k): i for i, k in enumerate(kints)}
    gamma = lookup[(0, 0, 0)]
    sig_bgw = np.zeros(N_SIGMA, dtype=np.float64)
    sig_native = np.zeros(N_SIGMA, dtype=np.float64)

    for q_int in kints:
        q_mod = np.mod(q_int, np.asarray(kgrid, dtype=np.int64))
        inner = lookup[tuple(np.mod(-q_mod, np.asarray(kgrid, dtype=np.int64)))]
        pair = np.einsum(
            "vsxyz,nsxyz->vnxyz",
            np.conj(psi[inner, :N_VAL]),
            psi[gamma, :N_SIGMA],
            optimize=True,
        )
        M = np.fft.fftn(pair, axes=(2, 3, 4)) / n_grid

        q_frac = q_mod / np.asarray(kgrid, dtype=np.float64)
        inner_frac = kints[inner] / np.asarray(kgrid, dtype=np.float64)
        g_unwrap = np.rint(inner_frac - (-q_frac)).astype(int)
        v_bgw = fill_v_grid_for_q(
            table_gamma,
            q_frac,
            fft_grid=fft_grid,
            cell_volume=float(wfn.cell_volume),
            sym_mats_k=sym.sym_mats_k,
        )
        v_native = native_v_grid(kernels, q_mod, kgrid, fft_grid)
        v_bgw = np.roll(v_bgw, shift=tuple(g_unwrap), axis=(0, 1, 2))
        v_native = np.roll(v_native, shift=tuple(g_unwrap), axis=(0, 1, 2))

        sig_bgw += -np.einsum("vnxyz,vnxyz,xyz->n", M.conj(), M, v_bgw, optimize=True).real / n_k
        sig_native += -np.einsum("vnxyz,vnxyz,xyz->n", M.conj(), M, v_native, optimize=True).real / n_k

    sig_bgw[:N_VAL] += x_head_ry
    sig_native[:N_VAL] += x_head_ry
    bgw_ref = parse_bgw_x_gamma(SIGMA_HP, N_SIGMA)
    sig_bgw_ev = sig_bgw * RY2EV
    sig_native_ev = sig_native * RY2EV

    print(f"Using LORRAX source: {sys.path[0]}")
    print(f"native sqrt_v sphere size: {kernels.n_sph} of {n_grid}")
    for label, vals in (("direct_BGW_v", sig_bgw_ev), ("direct_native_v", sig_native_ev)):
        diff = (vals - bgw_ref) * 1000.0
        print(
            f"{label:16s} "
            f"val_MAE={np.mean(np.abs(diff[:N_VAL])):9.3f} "
            f"val_max={np.max(np.abs(diff[:N_VAL])):9.3f} "
            f"cond_MAE={np.mean(np.abs(diff[N_VAL:])):9.3f} "
            f"cond_max={np.max(np.abs(diff[N_VAL:])):9.3f}"
        )
    print()
    print(" n      BGW_X       BGW_v_exact native_v_exact native-BGW meV")
    for i in range(N_SIGMA):
        print(
            f"{i+1:2d} {bgw_ref[i]:11.6f} {sig_bgw_ev[i]:12.6f} "
            f"{sig_native_ev[i]:14.6f} {(sig_native_ev[i]-bgw_ref[i])*1000.0:14.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
