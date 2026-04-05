from __future__ import annotations

"""
G-space -> ISDF W projection test for q=Gamma (2D slab truncation case).

Purpose
-------
This debug script builds W(G,G',omega) directly from eps0mat.h5 at two
frequencies (omega=0 and omega=i*omega_p), projects W to ISDF space using
zeta_q, and evaluates diagonal static/dynamic self-energy terms in the ISDF
basis.

Core projection equation
------------------------
    W_munu(omega) = sum_{G,G'} zeta_mu^*(G) W_GG'(omega) zeta_nu(G')

where zeta_mu(G) is obtained by FFT of zeta_q(mu, r) on the WFN FFT grid and
sampling at epsilon-matrix G-vectors (eps-order).

Current behavior
----------------
1) Read eps^{-1}(q=0,omega) at ifreq0 (static) and ifreqp (imag freq).
2) Build W(G,G',omega) from eps^{-1} with selectable Coulomb-side convention:
     right (default): W = eps^{-1} @ diag(v)
     left:            W = diag(v) @ eps^{-1}
   with BGW-like q=0 head/wing patching to enforce W_00 and zero q=0 wings.
3) Project W to ISDF basis via the equation above.
4) Compute static ISDF terms for requested bands:
     X, SEX-X(static), COH(static)
5) Optional GN fit in ISDF basis from W^c(0), W^c(i*omega_p), then evaluate
   Sigma_c^(+) and Sigma_c^(-) at omega=E_n^DFT-E_F with broadening eta.
   Main GN columns skip invalid modes (invalid_gpp_mode=0 style).
   Optional debug columns include invalid-mode static and 2Ry corrections.

MoS2/debug focus
----------------
- Uses 2D Coulomb truncation for v(G) on eps-order G-vectors.
- By default, reads q=0 head overrides from the GW input file:
    vhead, whead_0freq, whead_imfreq
  which matches the current MoS2 debug setup.
- If overrides are not present, falls back to compute_q0_averages.

This is a debugging utility, not a production workflow.
"""

import argparse
import glob
import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from gw.gw_init import read_cohsex_input
from gw.minimax_screening import extract_gn_ppm_parameters_from_Wc
from gw.vcoul import compute_V_qfullG_for_q, compute_q0_averages
from file_io import EPSReader, WFNReader, resolve_input_paths

RYD2EV = 13.6056980659

# Keep numerical behavior aligned with the double-precision BGW references.
jax.config.update("jax_enable_x64", True)


def _find_latest(patterns: list[str]) -> str:
    cands: list[str] = []
    for p in patterns:
        cands.extend(sorted(glob.glob(p)))
    if not cands:
        raise FileNotFoundError(f"No files matched: {patterns}")
    return cands[-1]


def _get_gvecs_in_eps_order(eps: EPSReader, iq: int) -> np.ndarray:
    nmtx = int(eps.nmtx[iq])
    gind = np.asarray(eps.gind_eps2rho[iq, :nmtx], dtype=np.int64)
    comps = np.asarray(eps.comps, dtype=np.int32)
    if np.max(gind) >= comps.shape[0]:
        raise ValueError(
            f"eps g-index exceeds components: max(gind)={int(np.max(gind))}, "
            f"ncomps={comps.shape[0]}"
        )
    return np.asarray(comps[gind, :], dtype=np.int32)


def _g0_index(gvecs: np.ndarray) -> int:
    hit = np.where(np.all(gvecs == 0, axis=1))[0]
    if hit.size == 0:
        raise ValueError("Could not locate G=(0,0,0) in eps basis.")
    return int(hit[0])


def _map_g_to_fft_indices(gvecs: np.ndarray, nfft: tuple[int, int, int]) -> np.ndarray:
    out = np.asarray(gvecs, dtype=np.int64).copy()
    nx, ny, nz = (int(nfft[0]), int(nfft[1]), int(nfft[2]))
    negx = out[:, 0] < 0
    negy = out[:, 1] < 0
    negz = out[:, 2] < 0
    out[negx, 0] += nx
    out[negy, 1] += ny
    out[negz, 2] += nz
    if (
        np.any(out[:, 0] < 0)
        or np.any(out[:, 0] >= nx)
        or np.any(out[:, 1] < 0)
        or np.any(out[:, 1] >= ny)
        or np.any(out[:, 2] < 0)
        or np.any(out[:, 2] >= nz)
    ):
        raise ValueError("Mapped FFT indices out of range.")
    return out


def _build_vg_for_q(
    *,
    wfn: WFNReader,
    gvecs_eps: np.ndarray,
    qvec_wrapped: np.ndarray,
    sys_dim: int,
) -> np.ndarray:
    v_qg = compute_V_qfullG_for_q(
        wfn=wfn,
        qvec_wrapped=np.asarray(qvec_wrapped, dtype=np.float64),
        comps_qG=np.asarray(gvecs_eps, dtype=np.int32),
        vc0_mean=0.0,
        do_Dmunu=False,
        sys_dim=int(sys_dim),
    )
    return np.array(np.asarray(v_qg, dtype=np.complex128), copy=True)


def _resolve_head_values(
    *,
    wfn: WFNReader,
    eps: EPSReader,
    sys_dim: int,
    v_head_override: float | None,
    w_head_override: float | None,
) -> tuple[complex, complex]:
    if v_head_override is not None and w_head_override is not None:
        return complex(float(v_head_override)), complex(float(w_head_override))

    class _Meta:
        pass

    meta = _Meta()
    meta.sys_dim = int(sys_dim)
    kgrid = np.asarray(getattr(wfn, "kgrid", np.array([1, 1, 1], dtype=np.int32)), dtype=np.int32).reshape(-1)
    if kgrid.size < 3:
        kgrid = np.array([1, 1, 1], dtype=np.int32)
    meta.nkx = int(kgrid[0])
    meta.nky = int(kgrid[1])
    meta.nkz = int(kgrid[2])
    vc0, wc0 = compute_q0_averages(
        wfn=wfn,
        epshead=np.asarray(eps.epshead, dtype=np.complex128),
        meta=meta,
    )
    v_head = complex(vc0) if v_head_override is None else complex(float(v_head_override))
    w_head = complex(wc0) if w_head_override is None else complex(float(w_head_override))
    return v_head, w_head


def _to_bgw_eps_orientation(eps_mat: np.ndarray, use_transpose: bool) -> np.ndarray:
    if use_transpose:
        return np.asarray(eps_mat.T, dtype=np.complex128)
    return np.asarray(eps_mat, dtype=np.complex128)


def _head_patch_epsinv_q0(
    epsinv: np.ndarray,
    g0_idx: int,
    v_g0: complex,
    w_head: complex,
    *,
    zero_wings: bool,
) -> np.ndarray:
    out = np.array(epsinv, copy=True)
    if abs(v_g0) <= 1.0e-20:
        raise ValueError("Cannot patch epsinv head: v(G=0) is ~0")
    out[:, g0_idx] = 0.0 + 0.0j
    if bool(zero_wings):
        out[g0_idx, :] = 0.0 + 0.0j
    out[g0_idx, g0_idx] = w_head / v_g0
    return out


def _add_head_in_munu(
    nohead_munu: np.ndarray,
    *,
    g0_mu: np.ndarray,
    head_au: complex,
    cell_volume: float,
) -> np.ndarray:
    """Add q=0 head in μν basis with the same form used in gw_jax."""
    outer_u = np.conj(g0_mu)[:, None] * g0_mu[None, :]
    return np.asarray(nohead_munu, dtype=np.complex128) + (complex(head_au) / float(cell_volume)) * outer_u


def _load_zeta_eps_order(
    *,
    zeta_h5: str,
    fft_shape: tuple[int, int, int],
    eps_fft_idx: np.ndarray,
    n_mu_use: int | None,
    mu_batch: int,
) -> np.ndarray:
    nx, ny, nz = (int(fft_shape[0]), int(fft_shape[1]), int(fft_shape[2]))
    nrtot = nx * ny * nz
    with h5py.File(zeta_h5, "r") as f:
        z = f["zeta_q"]
        if z.ndim != 5:
            raise ValueError(f"Unexpected zeta_q shape: {z.shape}")
        nmu = int(z.shape[3])
        if int(z.shape[4]) != nrtot:
            raise ValueError(
                f"zeta_q n_rtot mismatch: zeta={int(z.shape[4])}, fft={nrtot}"
            )
        if n_mu_use is not None and int(n_mu_use) > 0:
            nmu = min(nmu, int(n_mu_use))

        ng = int(eps_fft_idx.shape[0])
        out = np.zeros((nmu, ng), dtype=np.complex128)
        ix = eps_fft_idx[:, 0]
        iy = eps_fft_idx[:, 1]
        iz = eps_fft_idx[:, 2]

        for i0 in range(0, nmu, int(mu_batch)):
            i1 = min(i0 + int(mu_batch), nmu)
            zeta_r = np.asarray(z[0, 0, 0, i0:i1, :], dtype=np.complex128)
            zeta_box = zeta_r.reshape((i1 - i0, nx, ny, nz))
            zeta_g = np.fft.fftn(zeta_box, axes=(1, 2, 3))
            out[i0:i1, :] = zeta_g[:, ix, iy, iz]
    return out


@jax.jit
def _project_w_to_isdf_jit(w_gg: jax.Array, zeta_mu_g: jax.Array) -> jax.Array:
    # W_munu = Z^H W Z
    tmp = jnp.matmul(w_gg, jnp.transpose(zeta_mu_g))
    return jnp.matmul(jnp.conj(zeta_mu_g), tmp)


def _sigma_like_diag(
    *,
    psi_all: np.ndarray,   # (nb, nspinor, nmu)
    op_munu: np.ndarray,   # (nmu, nmu)
    n_occ: int,
    n_solve: int,
) -> np.ndarray:
    out = np.zeros((int(n_solve),), dtype=np.complex128)
    psi_occ = psi_all[: int(n_occ)]
    for n in range(int(n_solve)):
        psi_n = psi_all[n]
        m = np.einsum("vsm,sm->vm", np.conj(psi_occ), psi_n, optimize=True)
        out[n] = -np.einsum("vm,mn,vn->", np.conj(m), op_munu, m, optimize=True)
    return out


def _sigma_c_branches_diag(
    *,
    psi_all: np.ndarray,    # (nb_sum, ns, nmu)
    enk_all: np.ndarray,    # (nb_sum,)
    n_valence: int,
    n_solve: int,
    efermi: float,
    b_munu: np.ndarray,
    omega_munu: np.ndarray,
    valid_munu: np.ndarray,
    eta_ry: float,
) -> tuple[np.ndarray, np.ndarray]:
    nv = int(n_valence)
    nsolve = int(n_solve)
    tiny = 1.0e-30
    sig_p = np.zeros((nsolve,), dtype=np.complex128)
    sig_m = np.zeros((nsolve,), dtype=np.complex128)
    mask = np.asarray(valid_munu, dtype=bool)
    eta = 1j * float(eta_ry)

    psi_occ = psi_all[:nv]
    psi_unocc = psi_all[nv:]
    h_v = efermi - enk_all[:nv]       # >=0
    e_c = enk_all[nv:] - efermi       # >=0

    for n in range(nsolve):
        omega_rel = float(enk_all[n] - efermi)
        psi_n = psi_all[n]

        m_v = np.einsum("vsm,sm->vm", np.conj(psi_occ), psi_n, optimize=True)
        d_p = omega_rel + h_v[:, None, None] + omega_munu[None, :, :] + eta
        safe_p = mask[None, :, :] & (np.abs(d_p) > tiny)
        term_p = np.where(
            safe_p,
            np.conj(m_v)[:, :, None] * b_munu[None, :, :] * m_v[:, None, :] / d_p,
            0.0 + 0.0j,
        )
        sig_p[n] = -np.sum(term_p)

        m_c = np.einsum("csm,sm->cm", np.conj(psi_unocc), psi_n, optimize=True)
        d_m = omega_rel - e_c[:, None, None] - omega_munu[None, :, :] + eta
        safe_m = mask[None, :, :] & (np.abs(d_m) > tiny)
        term_m = np.where(
            safe_m,
            np.conj(m_c)[:, :, None] * b_munu[None, :, :] * m_c[:, None, :] / d_m,
            0.0 + 0.0j,
        )
        sig_m[n] = -np.sum(term_m)

    return sig_p, sig_m


def _sigma_sx_gpp_bgw_diag(
    *,
    psi_all: np.ndarray,    # (nb_sum, ns, nmu)
    enk_all: np.ndarray,    # (nb_sum,)
    n_valence: int,
    n_solve: int,
    efermi: float,
    b_munu: np.ndarray,
    omega_munu: np.ndarray,
    valid_munu: np.ndarray,
    gamma_ev: float,
    sexcutoff: float,
    eta_denom_ry: float = 0.0,
) -> np.ndarray:
    """
    BGW-like dynamic SX-X channel for GN-PPM.

    This mirrors the denominator handling used in bgw_src/Sigma/mtxel_cor.f90:
    - near-pole handling controlled by gpp_broadening (gamma)
    - SX amplitude truncation controlled by gpp_sexcutoff

    Notes:
    - Uses the ISDF-fitted GN parameters (B, Omega) where 2*B*Omega plays the
      role of Omega2*vcoul in BGW's G-space formulas.
    - Returns only the SX-X-like occupied contribution.
    """
    nv = int(n_valence)
    nsolve = int(n_solve)
    sig_sx = np.zeros((nsolve,), dtype=np.complex128)

    mask = np.asarray(valid_munu, dtype=bool)
    om = np.asarray(omega_munu, dtype=np.float64)
    b = np.asarray(b_munu, dtype=np.complex128)

    # BGW defaults: gpp_broadening=0.5 eV, gpp_sexcutoff=4.0
    gamma_ry = float(gamma_ev) / RYD2EV
    limittwo = gamma_ry * gamma_ry
    # Match BGW Common/nrtype.f90 defaults:
    #   TOL_Small = 1.0d-6, TOL_Zero = 1.0d-12
    tol_small = 1.0e-6
    tol_zero = 1.0e-12
    limitone = 1.0 / (4.0 * tol_small)

    # Map ISDF GN quantities to BGW-like kernels.
    # Omega2*vcoul <-> 2*B*Omega
    omega2v = 2.0 * b * om
    # (delta - eps^{-1})*vcoul <-> -Wc(0) = 2*B/Omega
    om_safe = np.where(np.abs(om) > tol_zero, om, 1.0)
    iepsv = np.where(mask, -(2.0 * b / om_safe), 0.0 + 0.0j)
    ssxcut = float(sexcutoff) * np.abs(iepsv)

    psi_occ = psi_all[:nv]
    h_v = efermi - enk_all[:nv]

    for n in range(nsolve):
        omega_rel = float(enk_all[n] - efermi)
        psi_n = psi_all[n]
        m_v = np.einsum("vsm,sm->vm", np.conj(psi_occ), psi_n, optimize=True)

        sx_acc = 0.0 + 0.0j
        for iv in range(nv):
            wx = omega_rel + float(h_v[iv])

            wdiff = wx - om
            wdiffr = wdiff * wdiff

            delw = np.zeros_like(b, dtype=np.complex128)
            good_wd = wdiffr > tol_zero
            if float(eta_denom_ry) != 0.0:
                delw[good_wd] = om[good_wd] / (wdiff[good_wd] + 1j * float(eta_denom_ry))
            else:
                delw[good_wd] = om[good_wd] / wdiff[good_wd]
            delwr = np.real(delw * np.conj(delw))

            cond1 = (wdiffr > limittwo) & (delwr < limitone)
            cond2 = (~cond1) & (delwr > tol_zero)

            ssx = np.zeros_like(b, dtype=np.complex128)

            # Regular branch: ssx = Omega2 / (wx^2 - wtilde^2)
            cden1 = (wx * wx) - (om * om)
            good1 = cond1 & (np.abs(cden1) > tol_zero)
            if float(eta_denom_ry) != 0.0:
                ssx[good1] = omega2v[good1] / (cden1[good1] + 1j * float(eta_denom_ry))
            else:
                ssx[good1] = omega2v[good1] / cden1[good1]

            # Near-pole combined branch assigned to SX (CH=0 locally)
            cden2 = 4.0 * (om * om) * (delw + 0.5)
            good2 = cond2 & (np.abs(cden2) > tol_zero)
            if float(eta_denom_ry) != 0.0:
                ssx[good2] = -omega2v[good2] * delw[good2] / (cden2[good2] + 1j * float(eta_denom_ry))
            else:
                ssx[good2] = -omega2v[good2] * delw[good2] / cden2[good2]

            # BGW gpp_sexcutoff pole filtering (only for wx < 0)
            if wx < 0.0:
                ssx[np.abs(ssx) > ssxcut] = 0.0 + 0.0j

            ssx = np.where(mask, ssx, 0.0 + 0.0j)

            mv = m_v[iv]
            sx_acc += np.einsum("m,mn,n->", np.conj(mv), ssx, mv, optimize=True)

        sig_sx[n] = -sx_acc

    return sig_sx


def _sigma_ch_gpp_bgw_diag(
    *,
    psi_all: np.ndarray,    # (nb_sum, ns, nmu)
    enk_all: np.ndarray,    # (nb_sum,)
    n_solve: int,
    b_munu: np.ndarray,
    omega_munu: np.ndarray,
    valid_munu: np.ndarray,
    gamma_ev: float,
    eta_denom_ry: float = 0.0,
) -> np.ndarray:
    """
    BGW-like dynamic CH' channel for GN-PPM.

    This matches the CH' gating used in the CO reference comparator:
    cond1 = (|wx-wtilde|^2 > gamma^2) and (|wtilde/(wx-wtilde)|^2 < 1/(4*tol_small))
    and accumulates only B/(wx-wtilde) on cond1 (invalid modes skipped).
    """
    nsum = int(psi_all.shape[0])
    nsolve = int(n_solve)
    out = np.zeros((nsolve,), dtype=np.complex128)

    mask = np.asarray(valid_munu, dtype=bool)
    om = np.asarray(omega_munu, dtype=np.float64)
    b = np.asarray(b_munu, dtype=np.complex128)

    gamma_ry = float(gamma_ev) / RYD2EV
    limittwo = gamma_ry * gamma_ry
    tol_small = 1.0e-6
    tol_zero = 1.0e-12
    limitone = 1.0 / (4.0 * tol_small)

    for n in range(nsolve):
        psi_n = psi_all[n]
        e_n = float(enk_all[n])
        acc = 0.0 + 0.0j

        for n1 in range(nsum):
            psi_n1 = psi_all[n1]
            wx = e_n - float(enk_all[n1])
            if abs(wx) < tol_zero:
                wx = tol_zero

            m = np.einsum("sm,sm->m", np.conj(psi_n1), psi_n, optimize=True)

            wdiff = wx - om
            wdiffr = wdiff * wdiff
            delw = np.zeros_like(om, dtype=np.complex128)
            good = wdiffr > tol_zero
            delw[good] = om[good] / wdiff[good]
            delwr = np.real(delw * np.conj(delw))

            cond1 = (wdiffr > limittwo) & (delwr < limitone) & mask

            kernel = np.zeros_like(b, dtype=np.complex128)
            goodk = cond1 & (np.abs(wdiff) > tol_zero)
            if float(eta_denom_ry) != 0.0:
                kernel[goodk] = b[goodk] / (wdiff[goodk] + 1j * float(eta_denom_ry))
            else:
                kernel[goodk] = b[goodk] / wdiff[goodk]

            acc += np.einsum("m,mn,n->", np.conj(m), kernel, m, optimize=True)

        out[n] = acc

    return out


def run(args: argparse.Namespace) -> int:
    gw_inp = os.path.abspath(args.gw_input)
    gw_dir = os.path.dirname(gw_inp)
    params = read_cohsex_input(gw_inp)
    resolve_input_paths(params, gw_dir)

    if args.restart_file:
        restart_file = os.path.abspath(args.restart_file)
    else:
        restart_file = _find_latest(
            [os.path.join(gw_dir, "tmp", "isdf_tensors_*.h5"), os.path.join(gw_dir, "isdf_tensors_*.h5")]
        )
    if args.zeta_h5:
        zeta_h5 = os.path.abspath(args.zeta_h5)
    else:
        zeta_h5 = _find_latest([os.path.join(gw_dir, "tmp", "zeta_q.h5"), os.path.join(gw_dir, "zeta_q.h5")])
    eps0mat = os.path.abspath(args.eps0mat) if args.eps0mat else os.path.join(gw_dir, "eps0mat.h5")

    eps = EPSReader(eps0mat)
    wfn = WFNReader(str(params["wfn_file"]))
    sys_dim = int(params.get("sys_dim", 2))
    iq = int(args.iq)
    nmtx = int(eps.nmtx[iq])
    if int(args.nmtx_max) > 0:
        nmtx = min(nmtx, int(args.nmtx_max))
    gvecs_eps = _get_gvecs_in_eps_order(eps, iq)[:nmtx]
    g0_idx = _g0_index(gvecs_eps)

    qvec_wrapped = np.asarray(eps.qpts[iq], dtype=np.float64)
    v_g = _build_vg_for_q(
        wfn=wfn,
        gvecs_eps=gvecs_eps,
        qvec_wrapped=qvec_wrapped,
        sys_dim=sys_dim,
    )

    # Default to gw input-file head overrides when present.
    v_head_override = args.v_head_au if args.v_head_au is not None else params.get("vhead")
    w0_head_override = args.w0_head_au if args.w0_head_au is not None else params.get("whead_0freq")
    wi_head_override = args.wi_head_au if args.wi_head_au is not None else params.get("whead_imfreq")
    v_head, w0_head = _resolve_head_values(
        wfn=wfn,
        eps=eps,
        sys_dim=sys_dim,
        v_head_override=v_head_override,
        w_head_override=w0_head_override,
    )
    _, wi_head = _resolve_head_values(
        wfn=wfn,
        eps=eps,
        sys_dim=sys_dim,
        v_head_override=v_head_override,
        w_head_override=wi_head_override,
    )
    v_g[g0_idx] = v_head

    epsinv0_raw = np.asarray(
        eps.get_eps_matrix(iq=iq, ifreq=int(args.ifreq0), imatrix=int(args.imatrix)),
        dtype=np.complex128,
    )[:nmtx, :nmtx]
    epsinvi_raw = np.asarray(
        eps.get_eps_matrix(iq=iq, ifreq=int(args.ifreqp), imatrix=int(args.imatrix)),
        dtype=np.complex128,
    )[:nmtx, :nmtx]
    epsinv0 = _to_bgw_eps_orientation(epsinv0_raw, bool(args.transpose_eps))
    epsinvi = _to_bgw_eps_orientation(epsinvi_raw, bool(args.transpose_eps))

    if bool(args.patch_head):
        epsinv0 = _head_patch_epsinv_q0(
            epsinv0,
            g0_idx,
            v_g[g0_idx],
            w0_head,
            zero_wings=bool(args.zero_q0_wings),
        )
        epsinvi = _head_patch_epsinv_q0(
            epsinvi,
            g0_idx,
            v_g[g0_idx],
            wi_head,
            zero_wings=bool(args.zero_q0_wings),
        )

    # Build W with selectable side convention.
    # right: W = epsinv @ diag(v) (Hermitian static/imag if epsinv is consistent)
    # left:  W = diag(v) @ epsinv (legacy BGW matrix-element debugging path)
    if str(args.w_side).lower() == "left":
        w0_gg = v_g[:, None] * epsinv0
        wi_gg = v_g[:, None] * epsinvi
    else:
        w0_gg = epsinv0 * v_g[None, :]
        wi_gg = epsinvi * v_g[None, :]
    v_gg = np.diag(v_g)
    wc0_gg = w0_gg - v_gg
    wci_gg = wi_gg - v_gg

    # Project only body (G!=0, G'!=0), then inject q=0 heads in μν basis.
    # This avoids FFT-normalization ambiguity in the isolated q=0 head channel.
    w0_gg_body = np.array(w0_gg, copy=True)
    wi_gg_body = np.array(wi_gg, copy=True)
    v_gg_body = np.array(v_gg, copy=True)
    for mat in (w0_gg_body, wi_gg_body, v_gg_body):
        mat[g0_idx, :] = 0.0 + 0.0j
        mat[:, g0_idx] = 0.0 + 0.0j

    fft_shape = tuple(int(x) for x in np.asarray(wfn.fft_grid, dtype=np.int64))
    eps_fft_idx = _map_g_to_fft_indices(gvecs_eps, fft_shape)
    zeta_mu_g = _load_zeta_eps_order(
        zeta_h5=zeta_h5,
        fft_shape=fft_shape,
        eps_fft_idx=eps_fft_idx,
        n_mu_use=(None if int(args.n_mu_use) <= 0 else int(args.n_mu_use)),
        mu_batch=int(args.mu_batch),
    )
    nmu = int(zeta_mu_g.shape[0])

    # Project body W/V to ISDF basis.
    w0_munu_body = np.asarray(
        _project_w_to_isdf_jit(
            jnp.asarray(w0_gg_body, dtype=jnp.complex128),
            jnp.asarray(zeta_mu_g, dtype=jnp.complex128),
        )
    )
    wi_munu_body = np.asarray(
        _project_w_to_isdf_jit(
            jnp.asarray(wi_gg_body, dtype=jnp.complex128),
            jnp.asarray(zeta_mu_g, dtype=jnp.complex128),
        )
    )
    v_munu_body_proj = np.asarray(
        _project_w_to_isdf_jit(
            jnp.asarray(v_gg_body, dtype=jnp.complex128),
            jnp.asarray(zeta_mu_g, dtype=jnp.complex128),
        )
    )

    # Load ISDF-basis wavefunctions from restart.
    with h5py.File(restart_file, "r") as f:
        psi_full = np.asarray(f["psi_full_y"], dtype=np.complex128)
        enk_full = np.asarray(f["enk_full"], dtype=np.float64)
        g0_mu = np.asarray(f["G0_mu_nu"], dtype=np.complex128) if "G0_mu_nu" in f else None
        v0_nohead_restart = np.asarray(f["V0_noG0_munu"], dtype=np.complex128) if "V0_noG0_munu" in f else None
    if psi_full.shape[0] != 1:
        raise ValueError("This test currently supports nk=1 only.")
    if int(psi_full.shape[-1]) < nmu:
        raise ValueError(
            f"restart n_mu={int(psi_full.shape[-1])} < projected n_mu={nmu}. "
            "Reduce --n-mu-use or regenerate consistent tensors."
        )
    psi_full = psi_full[:, :, :, :nmu]
    if g0_mu is None:
        raise ValueError("restart file does not contain G0_mu_nu; cannot inject q=0 head in μν basis.")
    g0_mu = np.asarray(g0_mu[:nmu], dtype=np.complex128)
    if v0_nohead_restart is not None:
        v_munu_body = np.asarray(v0_nohead_restart[:nmu, :nmu], dtype=np.complex128)
    else:
        v_munu_body = np.asarray(v_munu_body_proj, dtype=np.complex128)

    v_munu = _add_head_in_munu(v_munu_body, g0_mu=g0_mu, head_au=v_head, cell_volume=float(wfn.cell_volume))
    w0_munu = _add_head_in_munu(w0_munu_body, g0_mu=g0_mu, head_au=w0_head, cell_volume=float(wfn.cell_volume))
    wi_munu = _add_head_in_munu(wi_munu_body, g0_mu=g0_mu, head_au=wi_head, cell_volume=float(wfn.cell_volume))
    wc0_munu = w0_munu - v_munu
    wci_munu = wi_munu - v_munu

    print("")
    print("ISDF projection summary")
    print(f"  gw_input:      {gw_inp}")
    print(f"  restart_file:  {restart_file}")
    print(f"  zeta_h5:       {zeta_h5}")
    print(f"  eps0mat:       {eps0mat}")
    print(f"  sys_dim:       {sys_dim}")
    print(f"  nmtx used:     {nmtx}")
    print(f"  n_mu used:     {nmu}")
    print(f"  W side:        {str(args.w_side).lower()}")
    print(f"  q0 head(v):    {float(np.real(v_head)):.6f} a.u.")
    print(f"  q0 head(W0):   {float(np.real(w0_head)):.6f} a.u.")
    print(f"  q0 head(Wiwp): {float(np.real(wi_head)):.6f} a.u.")
    print(f"  ||W_G(0)||_F:  {np.linalg.norm(w0_gg, ord='fro'):.10e}")
    print(f"  ||W_G(iwp)||_F:{np.linalg.norm(wi_gg, ord='fro'):.10e}")
    print(f"  ||W_mu_body(0)||_F:  {np.linalg.norm(w0_munu_body, ord='fro'):.10e}")
    print(f"  ||W_mu_body(iwp)||_F:{np.linalg.norm(wi_munu_body, ord='fro'):.10e}")
    print(f"  ||W_mu(0)||_F:       {np.linalg.norm(w0_munu, ord='fro'):.10e}")
    print(f"  ||W_mu(iwp)||_F:     {np.linalg.norm(wi_munu, ord='fro'):.10e}")
    h0 = np.linalg.norm(w0_munu - np.conj(w0_munu.T), ord="fro") / max(np.linalg.norm(w0_munu, ord="fro"), 1.0e-30)
    hi = np.linalg.norm(wi_munu - np.conj(wi_munu.T), ord="fro") / max(np.linalg.norm(wi_munu, ord="fro"), 1.0e-30)
    print(f"  herm_rel(W_mu,0):   {h0:.6e}")
    print(f"  herm_rel(W_mu,iwp): {hi:.6e}")

    if args.output_h5:
        out_h5 = os.path.abspath(args.output_h5)
        os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)
        with h5py.File(out_h5, "w") as h5:
            h5.create_dataset("W0_proj_q000_munu", data=np.asarray(w0_munu, dtype=np.complex128))
            h5.create_dataset("Wiwp_proj_q000_munu", data=np.asarray(wi_munu, dtype=np.complex128))
            h5.create_dataset("V_proj_q000_munu", data=np.asarray(v_munu, dtype=np.complex128))
            h5.create_dataset("Wc0_proj_q000_munu", data=np.asarray(wc0_munu, dtype=np.complex128))
            h5.create_dataset("Wci_proj_q000_munu", data=np.asarray(wci_munu, dtype=np.complex128))
            h5.create_dataset("W0_body_proj_q000_munu", data=np.asarray(w0_munu_body, dtype=np.complex128))
            h5.create_dataset("Wiwp_body_proj_q000_munu", data=np.asarray(wi_munu_body, dtype=np.complex128))
            h5.create_dataset("V_body_proj_q000_munu", data=np.asarray(v_munu_body, dtype=np.complex128))
            h5.attrs["fro_W0_proj_q000"] = float(np.linalg.norm(w0_munu, ord="fro"))
            h5.attrs["fro_Wiwp_proj_q000"] = float(np.linalg.norm(wi_munu, ord="fro"))
            h5.attrs["nmtx_used"] = int(nmtx)
            h5.attrs["nmu_used"] = int(nmu)
            h5.attrs["omega_p_ry"] = float(args.omega_p_ry)
        print(f"  W-projection h5: {out_h5}")

    b0 = int(args.band_offset)
    nsolve = int(args.n_valence + args.n_cond_solve)
    bsolve1 = b0 + nsolve
    bsum1 = b0 + int(args.n_sum_states)
    nb = int(psi_full.shape[1])
    if bsum1 > nb:
        raise ValueError(f"Requested bands exceed restart: end={bsum1}, nb={nb}")
    if bsolve1 > bsum1:
        raise ValueError("n_valence+n_cond_solve must be <= n_sum_states")
    if int(args.n_valence) <= 0 or int(args.n_valence) >= int(args.n_sum_states):
        raise ValueError("n_valence must be in (0, n_sum_states)")

    psi_sum = psi_full[0, b0:bsum1]     # (nsum, ns, mu)
    enk_sum = enk_full[0, b0:bsum1]
    psi_solve = psi_full[0, b0:bsolve1]
    enk_solve = enk_full[0, b0:bsolve1]
    e_vbm = float(np.max(enk_sum[: int(args.n_valence)]))
    e_cbm = float(np.min(enk_sum[int(args.n_valence):]))
    efermi = 0.5 * (e_vbm + e_cbm)

    # Static terms.
    sigma_x = _sigma_like_diag(
        psi_all=psi_solve,
        op_munu=v_munu,
        n_occ=int(args.n_valence),
        n_solve=nsolve,
    )
    sigma_sex_static = _sigma_like_diag(
        psi_all=psi_solve,
        op_munu=w0_munu,
        n_occ=int(args.n_valence),
        n_solve=nsolve,
    )
    sigma_sex_minus_x_static = sigma_sex_static - sigma_x
    sigma_like_ri_wc0 = _sigma_like_diag(
        psi_all=psi_sum,
        op_munu=wc0_munu,
        n_occ=int(args.n_sum_states),
        n_solve=nsolve,
    )
    sigma_coh_static = -0.5 * sigma_like_ri_wc0

    print("")
    print("Static ISDF sigma table (eV)")
    print("  n\tE_dft\tX\tSEX-X_static\tCOH_static\tCor_static\tSig_static")
    for i in range(nsolve):
        n = b0 + i
        x = sigma_x[i] * RYD2EV
        sxm = sigma_sex_minus_x_static[i] * RYD2EV
        coh = sigma_coh_static[i] * RYD2EV
        cor = sxm + coh
        sig = x + cor
        print(
            f"  {n:d}\t{float(enk_solve[i] * RYD2EV):.6f}\t"
            f"{x.real:.6f}\t{sxm.real:.6f}\t{coh.real:.6f}\t{cor.real:.6f}\t{sig.real:.6f}"
        )

    if bool(args.do_gn):
        wc0_q = wc0_munu[None, None, None, None, :, None, :]
        wci_q = wci_munu[None, None, None, None, :, None, :]
        ppm = extract_gn_ppm_parameters_from_Wc(
            jnp.asarray(wc0_q, dtype=jnp.complex128),
            jnp.asarray(wci_q, dtype=jnp.complex128),
            omega_p=float(args.omega_p_ry),
            fallback_omega=float(args.fallback_omega_ry),
        )
        b_munu = np.asarray(ppm.b_qmunu[0, 0, 0], dtype=np.complex128)
        omega_munu = np.asarray(ppm.omega_qmunu[0, 0, 0], dtype=np.float64)
        valid = np.asarray(ppm.valid_qmunu[0, 0, 0], dtype=bool)
        print("")
        print(
            "GN fit in ISDF basis: "
            f"omega_p={float(args.omega_p_ry):.6f} Ry, invalid={100.0 * float(np.mean(~valid)):.2f}%"
        )

        # Main columns: skip invalid.
        eta_ry = float(args.eta_ev) / RYD2EV
        eta_denom_ry = float(args.eta_denom_ev) / RYD2EV
        sig_p_legacy, sig_m = _sigma_c_branches_diag(
            psi_all=psi_sum,
            enk_all=enk_sum,
            n_valence=int(args.n_valence),
            n_solve=nsolve,
            efermi=efermi,
            b_munu=b_munu,
            omega_munu=omega_munu,
            valid_munu=valid,
            eta_ry=eta_ry,
        )
        if str(args.gn_sx_form).lower() == "bgw":
            sig_p = _sigma_sx_gpp_bgw_diag(
                psi_all=psi_sum,
                enk_all=enk_sum,
                n_valence=int(args.n_valence),
                n_solve=nsolve,
                efermi=efermi,
                b_munu=b_munu,
                omega_munu=omega_munu,
                valid_munu=valid,
                gamma_ev=float(args.gpp_broadening_ev),
                sexcutoff=float(args.gpp_sexcutoff),
                eta_denom_ry=eta_denom_ry,
            )
            sig_m = _sigma_ch_gpp_bgw_diag(
                psi_all=psi_sum,
                enk_all=enk_sum,
                n_solve=nsolve,
                b_munu=b_munu,
                omega_munu=omega_munu,
                valid_munu=valid,
                gamma_ev=float(args.gpp_broadening_ev),
                eta_denom_ry=eta_denom_ry,
            )
        else:
            sig_p = sig_p_legacy

        # Precompute eV series for optional machine-readable output.
        sxm_gn_ev = np.asarray(sig_p * RYD2EV, dtype=np.complex128)
        coh_gn_ev = np.asarray(sig_m * RYD2EV, dtype=np.complex128)
        cor_gn_ev = np.asarray(sxm_gn_ev + coh_gn_ev, dtype=np.complex128)
        sig_gn_ev = np.asarray((sigma_x * RYD2EV) + cor_gn_ev, dtype=np.complex128)
        edft_ev = np.asarray(enk_solve * RYD2EV, dtype=np.float64)
        band_1idx = np.asarray([b0 + i + 1 for i in range(nsolve)], dtype=np.int32)

        if str(args.out_cor_dat).strip():
            out_cor_dat = os.path.abspath(str(args.out_cor_dat))
            os.makedirs(os.path.dirname(out_cor_dat) or ".", exist_ok=True)
            with open(out_cor_dat, "w", encoding="utf-8") as f:
                f.write("# n1idx  E_dft_eV  SEXmX_GN_eV  COH_GN_eV  Cor_GN_eV  Sig_GN_eV\n")
                for i in range(nsolve):
                    f.write(
                        f"{int(band_1idx[i]):4d}  "
                        f"{float(edft_ev[i]): .9f}  "
                        f"{float(np.real(sxm_gn_ev[i])): .9f}  "
                        f"{float(np.real(coh_gn_ev[i])): .9f}  "
                        f"{float(np.real(cor_gn_ev[i])): .9f}  "
                        f"{float(np.real(sig_gn_ev[i])): .9f}\n"
                    )
            print(f"  Wrote Cor table: {out_cor_dat}")

        print("")
        if bool(args.debug_invalid_mode):
            valid_f = valid.astype(np.float64)
            invalid_f = (~valid).astype(np.float64)

            # Static pieces reconstructed from the GN residue on valid nodes:
            # Wc_valid_static(mu,nu) = -2 B(mu,nu) / Omega(mu,nu), valid only.
            omega_safe = np.where(np.abs(omega_munu) > 1.0e-14, omega_munu, 1.0)
            wc0_valid_gn = np.where(valid, -2.0 * b_munu / omega_safe, 0.0 + 0.0j)
            sxm_x_static_gn_valid = _sigma_like_diag(
                psi_all=psi_solve,
                op_munu=wc0_valid_gn,
                n_occ=int(args.n_valence),
                n_solve=nsolve,
            )
            coh_static_gn_valid = -0.5 * _sigma_like_diag(
                psi_all=psi_sum,
                op_munu=wc0_valid_gn,
                n_occ=int(args.n_sum_states),
                n_solve=nsolve,
            )

            wc0_invalid = wc0_munu * invalid_f
            dsex_inv_static = _sigma_like_diag(
                psi_all=psi_solve,
                op_munu=wc0_invalid,
                n_occ=int(args.n_valence),
                n_solve=nsolve,
            )
            dcoh_inv_static = -0.5 * _sigma_like_diag(
                psi_all=psi_sum,
                op_munu=wc0_invalid,
                n_occ=int(args.n_sum_states),
                n_solve=nsolve,
            )
            sig_inv_static = dsex_inv_static + dcoh_inv_static

            omega_2ry = np.where(valid, omega_munu, float(args.fallback_omega_ry))
            sig_p_2, sig_m_2 = _sigma_c_branches_diag(
                psi_all=psi_sum,
                enk_all=enk_sum,
                n_valence=int(args.n_valence),
                n_solve=nsolve,
                efermi=efermi,
                b_munu=b_munu * invalid_f,
                omega_munu=omega_2ry,
                valid_munu=np.ones_like(valid, dtype=bool),
                eta_ry=eta_ry,
            )
            print(
                "Dynamic GN ISDF table (eV) [main: skip invalid, debug: invalid corrections]"
            )
            print(
                "  n\tE_dft\tSEX-X_GN\tCOH_GN\tSEX-X_static_valid(R)\tCOH_static_valid(R)\t"
                "d(SEX-X)_inv_static\td(COH)_inv_static\tdCor_inv_static\tdCor_inv_2Ry\tCor_GN\tSig_GN"
            )
            for i in range(nsolve):
                n = b0 + i
                sxm_gn = sxm_gn_ev[i]
                coh_gn = coh_gn_ev[i]
                sxm_stat_valid = sxm_x_static_gn_valid[i] * RYD2EV
                coh_stat_valid = coh_static_gn_valid[i] * RYD2EV
                dsx_inv = dsex_inv_static[i] * RYD2EV
                dcoh_inv = dcoh_inv_static[i] * RYD2EV
                dcor_static = sig_inv_static[i] * RYD2EV
                dcor_2ry = (sig_p_2[i] + sig_m_2[i]) * RYD2EV
                cor = cor_gn_ev[i]
                sig = sig_gn_ev[i]
                print(
                    f"  {n:d}\t{float(enk_solve[i] * RYD2EV):.6f}\t"
                    f"{sxm_gn.real:.6f}\t{coh_gn.real:.6f}\t"
                    f"{sxm_stat_valid.real:.6f}\t{coh_stat_valid.real:.6f}\t"
                    f"{dsx_inv.real:.6f}\t{dcoh_inv.real:.6f}\t"
                    f"{dcor_static.real:.6f}\t{dcor_2ry.real:.6f}\t"
                    f"{cor.real:.6f}\t{sig.real:.6f}"
                )
        else:
            print("Dynamic GN ISDF table (eV) [skip invalid in main columns]")
            print("  n\tE_dft\tSEX-X_GN\tCOH_GN\tCor_GN\tSig_GN")
            for i in range(nsolve):
                n = b0 + i
                sxm_gn = sxm_gn_ev[i]
                coh_gn = coh_gn_ev[i]
                cor = cor_gn_ev[i]
                sig = sig_gn_ev[i]
                print(
                    f"  {n:d}\t{float(enk_solve[i] * RYD2EV):.6f}\t"
                    f"{sxm_gn.real:.6f}\t{coh_gn.real:.6f}\t{cor.real:.6f}\t{sig.real:.6f}"
                )

    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Project W(G,G',omega) from eps0mat to ISDF basis (2D slab debug) and evaluate static/GN sigma terms."
    )
    p.add_argument("-g", "--gw-input", default="cohsex.in", help="GW input file (default: cohsex.in)")
    p.add_argument("--restart-file", default="", help="Path to isdf_tensors_*.h5")
    p.add_argument("--zeta-h5", default="", help="Path to zeta_q.h5")
    p.add_argument("--eps0mat", default="eps0mat.h5", help="Path to eps0mat.h5")
    p.add_argument("--iq", type=int, default=0, help="q-index in eps file (default: 0)")
    p.add_argument("--imatrix", type=int, default=0, help="eps matrix index (default: 0)")
    p.add_argument("--ifreq0", type=int, default=0, help="static frequency index (default: 0)")
    p.add_argument("--ifreqp", type=int, default=1, help="imaginary-frequency index (default: 1)")
    p.add_argument("--omega-p-ry", type=float, default=2.0, help="GN fit omega_p in Ry (default: 2.0)")
    p.add_argument("--fallback-omega-ry", type=float, default=2.0, help="fallback omega for invalid GN in Ry")
    p.add_argument("--eta-ev", type=float, default=0.5, help="broadening eta in eV for GN denominators")
    p.add_argument("--eta-denom-ev", type=float, default=0.0, help="optional +i*eta in BGW-like CH/SX denominators")
    p.add_argument("--v-head-au", type=float, default=None, help="override V head (a.u.)")
    p.add_argument("--w0-head-au", type=float, default=None, help="override W(0) head (a.u.)")
    p.add_argument("--wi-head-au", type=float, default=None, help="override W(i*omega_p) head (a.u.)")
    p.add_argument(
        "--patch-head",
        type=int,
        default=1,
        help="apply q=0 head patch: set epsinv[G0,G0]=W00/V00 and clear G0 column (and row if zero-q0-wings=1)",
    )
    p.add_argument(
        "--zero-q0-wings",
        type=int,
        default=1,
        help="when patching head, zero both G0 row/column wings (BGW fixwing-style)",
    )
    p.add_argument("--transpose-eps", type=int, default=1, help="use BGW-matching transposed eps orientation")
    p.add_argument(
        "--w-side",
        default="right",
        choices=("right", "left"),
        help=(
            "How to apply v(G) to eps^{-1}: "
            "right -> W=epsinv@diag(v) (Hermitian, GWJAX-compatible), "
            "left -> W=diag(v)@epsinv (legacy/BGW matrix-element debugging)."
        ),
    )
    p.add_argument("--nmtx-max", type=int, default=0, help="optional truncate nmtx (0=full)")
    p.add_argument("--n-mu-use", type=int, default=0, help="optional truncate number of mu (0=all)")
    p.add_argument("--mu-batch", type=int, default=32, help="mu batch size for FFT loading from zeta")
    p.add_argument("--band-offset", type=int, default=0, help="starting band index")
    p.add_argument("--n-valence", type=int, default=10, help="number of valence bands in solve/sums")
    p.add_argument("--n-cond-solve", type=int, default=10, help="number of conduction bands to report")
    p.add_argument("--n-sum-states", type=int, default=40, help="total states in explicit sums")
    p.add_argument("--do-gn", type=int, default=1, help="compute GN dynamic sigma columns")
    p.add_argument(
        "--gn-sx-form",
        default="bgw",
        choices=("bgw", "legacy"),
        help="dynamic SX-X expression: 'bgw' (BGW-like gpp_broadening/sexcutoff) or legacy branch formula",
    )
    p.add_argument("--gpp-broadening-ev", type=float, default=0.5, help="BGW-style gpp_broadening in eV")
    p.add_argument("--gpp-sexcutoff", type=float, default=4.0, help="BGW-style gpp_sexcutoff (dimensionless)")
    p.add_argument("--debug-invalid-mode", type=int, default=0, help="print invalid-mode static and 2Ry correction columns")
    p.add_argument("--output-h5", default="", help="Optional output H5 path for projected W/V matrices")
    p.add_argument("--out-cor-dat", default="", help="Optional output text table for dynamic Cor_GN vs band")
    args = p.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
