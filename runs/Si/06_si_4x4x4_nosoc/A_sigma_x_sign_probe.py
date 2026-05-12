"""Agent A probe for direct G-space Sigma_X sign conventions.

This is a read-only diagnostic for the Si 4x4x4 no-SOC run.  It compares
direct G-space exchange at Gamma against BGW's X column while toggling the
G-index convention used to contract pair densities with BGW's MC-averaged
``vcoul`` table.
"""

from __future__ import annotations

import os
import re
import sys

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src")

from runtime import set_default_env

set_default_env()

from common.symmetry_maps import SymMaps
from file_io import WFNReader
from file_io.read_bgw_vcoul import fill_v_grid_for_q, read_bgw_vcoul


ROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc"
WFN_PATH = f"{ROOT}/qe/nscf/WFN.h5"
VCOUL_PATH = f"{ROOT}/D_bgw_cohsex/vcoul"
SIGMA_HP = f"{ROOT}/D_bgw_cohsex/sigma_hp.log"
N_VAL = 8
N_SIGMA = 16
RY2EV = 13.605693122994


def parse_bgw_x_gamma(path: str, n_bands: int) -> np.ndarray:
    vals: list[float] = []
    in_gamma = False
    with open(path, encoding="utf-8") as fh:
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


def main() -> int:
    wfn = WFNReader(WFN_PATH)
    sym = SymMaps(wfn)
    nx, ny, nz = (int(v) for v in wfn.fft_grid)
    n_grid = nx * ny * nz
    n_k = int(sym.nk_tot)
    nspinor = int(wfn.nspinor)
    v_cell = float(wfn.cell_volume)
    nb = N_VAL + N_SIGMA
    bvec_cart = float(wfn.blat) * np.asarray(wfn.bvec, dtype=np.float64)
    ecutwfc = 25.0

    print(f"FFT grid=({nx},{ny},{nz}) Nk={n_k} Vcell={v_cell:.8f} nspinor={nspinor}")
    print(f"Using LORRAX source: {sys.path[0]}")

    psi = np.zeros((n_k, nb, nspinor, nx, ny, nz), dtype=np.complex128)
    psi_clip = np.zeros_like(psi)
    coeff_norm_err = []
    real_norm_err = []
    clip_norm_loss = []
    ix = np.concatenate([np.arange(0, nx // 2 + 1), np.arange(-(nx // 2 - 1), 0)])
    iy = np.concatenate([np.arange(0, ny // 2 + 1), np.arange(-(ny // 2 - 1), 0)])
    iz = np.concatenate([np.arange(0, nz // 2 + 1), np.arange(-(nz // 2 - 1), 0)])
    gx_all, gy_all, gz_all = np.meshgrid(ix, iy, iz, indexing="ij")
    g_all = np.stack([gx_all.ravel(), gy_all.ravel(), gz_all.ravel()], axis=-1).astype(int)
    for ik in range(n_k):
        k_full = np.asarray(sym.unfolded_kpts[ik], dtype=np.float64)
        qg_cart = (k_full[None, :] + g_all) @ bvec_cart
        target_set = set(map(tuple, g_all[np.sum(qg_cart ** 2, axis=1) <= ecutwfc + 1.0e-10].tolist()))
        gv = np.asarray(sym.get_gvecs_kfull(wfn, ik))
        cnk = sym.get_cnk_fullzone_batch(wfn, np.arange(nb), ik)
        coeff_norm_err.extend((np.sum(np.abs(cnk) ** 2, axis=(1, 2)) - 1.0).tolist())
        box = np.zeros((nb, nspinor, nx, ny, nz), dtype=np.complex128)
        box[:, :, gv[:, 0], gv[:, 1], gv[:, 2]] = cnk
        psi[ik] = np.fft.ifftn(box, axes=(2, 3, 4)) * n_grid
        real_norm_err.extend((np.sum(np.abs(psi[ik]) ** 2, axis=(1, 2, 3, 4)) / n_grid - 1.0).tolist())

        keep = np.array([tuple(g) in target_set for g in gv], dtype=bool)
        box_clip = np.zeros_like(box)
        box_clip[:, :, gv[keep, 0], gv[keep, 1], gv[keep, 2]] = cnk[:, :, keep]
        psi_clip[ik] = np.fft.ifftn(box_clip, axes=(2, 3, 4)) * n_grid
        clip_norm_loss.extend((1.0 - np.sum(np.abs(cnk[:, :, keep]) ** 2, axis=(1, 2))).tolist())

    print(
        "WFN norm error: "
        f"coeff max={np.max(np.abs(coeff_norm_err)):.3e}, "
        f"real max={np.max(np.abs(real_norm_err)):.3e}"
    )
    print(
        "Target-sphere clip norm loss: "
        f"max={np.max(clip_norm_loss):.3e}, "
        f"mean={np.mean(clip_norm_loss):.3e}"
    )

    kpts_full = np.asarray(sym.unfolded_kpts)
    ik_gamma = int(np.argmin(np.linalg.norm(kpts_full, axis=1)))
    if np.linalg.norm(kpts_full[ik_gamma]) >= 1.0e-8:
        raise RuntimeError("Could not identify Gamma in unfolded k-points")

    table = read_bgw_vcoul(VCOUL_PATH)
    table_gamma = type(table)(
        q_fracs=table.q_fracs[:8].copy(),
        G_miller_per_q=table.G_miller_per_q[:8],
        vcoul_per_q=table.vcoul_per_q[:8],
    )
    print(f"BGW vcoul unique q blocks: all={len(table.q_fracs)}, gamma-loop={len(table_gamma.q_fracs)}")
    bgw = parse_bgw_x_gamma(SIGMA_HP, N_SIGMA)

    variant_names = []
    for table_name in ("all_vcoul_blocks", "gamma_vcoul_blocks"):
        for psi_name in ("raw", "target_clip"):
            for q_name in ("q=k-kp", "q=kp-k"):
                for pair_name in ("old_pair", "standard_pair"):
                    for v_name in ("vG", "vMinusG"):
                        variant_names.append(f"{table_name}/{psi_name}/{q_name}/{pair_name}/{v_name}")
    variants = {name: np.zeros(N_SIGMA, dtype=np.float64) for name in variant_names}

    gx_neg = (-np.arange(nx)) % nx
    gy_neg = (-np.arange(ny)) % ny
    gz_neg = (-np.arange(nz)) % nz

    for iq, qfrac in enumerate(kpts_full):
        target = (-qfrac) % 1.0
        diffs = np.linalg.norm(((kpts_full - target[None, :] + 0.5) % 1.0) - 0.5, axis=1)
        ik_kmq = int(np.argmin(diffs))
        if diffs[ik_kmq] >= 1.0e-6:
            raise RuntimeError(f"Could not map k-q for iq={iq}, q={qfrac}")
        g_unwrap = np.rint(kpts_full[ik_kmq] - (-qfrac)).astype(int)

        v_by_q = {}
        for table_name, table_src in (("all_vcoul_blocks", table), ("gamma_vcoul_blocks", table_gamma)):
            v_by_q[table_name] = {}
            for q_name, q_lookup in (("q=k-kp", qfrac), ("q=kp-k", -qfrac)):
                v_box = fill_v_grid_for_q(
                    table_src,
                    q_lookup,
                    fft_grid=(nx, ny, nz),
                    cell_volume=v_cell,
                    sym_mats_k=sym.sym_mats_k,
                )
                if np.linalg.norm(q_lookup) < 1.0e-8:
                    iq_table, _, _ = table_src.find_q_index(q_lookup, sym_mats_k=sym.sym_mats_k)
                    g_miller = table_src.G_miller_per_q[iq_table]
                    vcoul_q = table_src.vcoul_per_q[iq_table]
                    zero_idx = np.where((g_miller == 0).all(axis=1))[0]
                    if len(zero_idx):
                        v_box[0, 0, 0] = vcoul_q[zero_idx[0]] / v_cell

                v_plus = np.roll(v_box, shift=tuple(g_unwrap), axis=(0, 1, 2))
                v_neg = v_box[np.ix_(gx_neg, gy_neg, gz_neg)]
                v_minus = np.roll(v_neg, shift=tuple(-g_unwrap), axis=(0, 1, 2))
                v_by_q[table_name][q_name] = {"vG": v_plus, "vMinusG": v_minus}

        for psi_name, psi_src in (("raw", psi), ("target_clip", psi_clip)):
            pair_old = np.einsum(
                "vsxyz,nsxyz->vnxyz",
                np.conj(psi_src[ik_kmq, :N_VAL]),
                psi_src[ik_gamma, :N_SIGMA],
                optimize=True,
            )
            m_old = np.fft.fftn(pair_old, axes=(2, 3, 4)) / n_grid

            pair_std = np.einsum(
                "nsxyz,vsxyz->vnxyz",
                np.conj(psi_src[ik_gamma, :N_SIGMA]),
                psi_src[ik_kmq, :N_VAL],
                optimize=True,
            )
            m_std = np.fft.fftn(pair_std, axes=(2, 3, 4)) / n_grid

            for table_name in ("all_vcoul_blocks", "gamma_vcoul_blocks"):
                for q_name in ("q=k-kp", "q=kp-k"):
                    for v_name in ("vG", "vMinusG"):
                        variants[f"{table_name}/{psi_name}/{q_name}/old_pair/{v_name}"] += -np.einsum(
                            "vnxyz,vnxyz,xyz->n",
                            m_old.conj(),
                            m_old,
                            v_by_q[table_name][q_name][v_name],
                            optimize=True,
                        ).real / n_k
                        variants[f"{table_name}/{psi_name}/{q_name}/standard_pair/{v_name}"] += -np.einsum(
                            "vnxyz,vnxyz,xyz->n",
                            m_std.conj(),
                            m_std,
                            v_by_q[table_name][q_name][v_name],
                            optimize=True,
                        ).real / n_k

    print("")
    print("Variant errors vs BGW Gamma X (meV):")
    for name, sx_ry in variants.items():
        sx_ev = sx_ry * RY2EV
        diff = (sx_ev - bgw) * 1000.0
        val = diff[:N_VAL]
        cond = diff[N_VAL:]
        print(
            f"{name:24s} "
            f"val_MAE={np.mean(np.abs(val)):8.3f} val_max={np.max(np.abs(val)):8.3f} "
            f"cond_MAE={np.mean(np.abs(cond)):8.3f} cond_max={np.max(np.abs(cond)):8.3f}"
        )

    best_name = min(
        variants,
        key=lambda name: np.mean(np.abs((variants[name] * RY2EV - bgw) * 1000.0)),
    )
    best = variants[best_name] * RY2EV
    print(f"\nBest variant: {best_name}")
    print(" n      BGW_X       direct_X    diff_meV")
    for i, (x_bgw, x_dir) in enumerate(zip(bgw, best), start=1):
        print(f"{i:2d}  {x_bgw:11.6f} {x_dir:11.6f} {(x_dir - x_bgw) * 1000.0:10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
