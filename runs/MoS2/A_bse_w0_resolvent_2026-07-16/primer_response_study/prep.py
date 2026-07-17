"""prep.py — shared module for the primer-response prototypes (proto0_*).

Fixture loaders + verbatim-reused validated routines from
../interp_study/{falloff_cq.py, vq_loo.py, physical_contract.py} (provenance
cited inline), plus the C1 construction pieces: pair-row builders, eigh frames,
half-whitened Phi, SR/LR v-kernels (production convention from
sources/lorrax_A/src/gw/compute_vcoul.py), single-particle transport links,
pair cross-Grams, gap-window physical metrics, TDA exciton assembly
(conventions from sources/lorrax_A/src/bse/bse_serial.py).

Production mapping (owner sharding note): every eigh/SVD/polar and every
N_mu^2 matrix here is single-GPU jax.numpy at fixture size by owner ruling;
the production form shards the N_mu^2 objects P('x','y') and routes
factorizations through the cusolvermp/slate FFI (src/ffi) — deferred until
results warrant.

Conjugation conventions (the single place where they live — resolved
numerically by resolve_conventions() and then frozen):
  code pair rows  x_p(mu) = sum_s conj(psi_{m,k-q,s}(mu)) psi_{n,k,s}(mu)
                  (spin-traced; matches bse_serial.compute_pair_amplitude)
  Y rows          Y[p,mu] = conj(x_p(mu))   (the "bra side")
  C convention    C_code[nu,mu] per isdf.core.c_q_from_psi_sm — equals either
                  Gram(x-rows) or Gram(Y-rows) = its transpose; resolved by
                  assert against the verbatim falloff rebuild.
  make_Vq         V[mu,nu] = sum_G conj(zt_mu) v zt_nu (verbatim vq_loo.py)
  Phi             chosen so  Phi diag(v) Phi^H == S R^H V R S  at 1e-12
  a rows          chosen so  a Vc a^H == conj(M) V M^T  and ||a_p|| <= 1
"""
import json
import numpy as np
import h5py

import jax
import jax.numpy as jnp

BASE = "/pscratch/sd/j/jackm/lorrax_sandbox"
STUDY = f"{BASE}/runs/MoS2/A_bse_w0_resolvent_2026-07-16/primer_response_study"

FIX = {
    "mos2_3x3": {
        "isdf": f"{BASE}/runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native/tmp/isdf_tensors_640.h5",
        "zeta": f"{BASE}/runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native/tmp/zeta_q.h5",
        # exact input of the 05 run (05_lorrax_cohsex_native/WFN.h5 -> qe/nscf)
        "wfn":  f"{BASE}/runs/MoS2/00_mos2_3x3_cohsex/qe/nscf/WFN.h5",
        "sys_dim": 2,
    },
    "mos2_6x6": {
        "isdf": f"{BASE}/runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/mos2_6x6/lorrax/tmp/isdf_tensors_640.h5",
        "zeta": f"{BASE}/runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/mos2_6x6/lorrax/tmp/zeta_q.h5",
        "wfn":  f"{BASE}/runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/mos2_6x6/qe/nscf/WFN.h5",
        "sys_dim": 2,
    },
    "mos2_4x4": {
        "isdf": f"{BASE}/runs/MoS2/01_mos2_4x4_cohsex_gnppm/C_lorrax_gnppm_replicated_postproc/tmp/isdf_tensors_640.h5",
        "zeta": None,
        "wfn":  None,   # centroid-quadrature overlaps only
        "sys_dim": 2,
        "kgrid_fallback": (4, 4, 1),
        "nocc": 26,     # same material/window as mos2_3x3 (ifmax=26)
    },
    "si_4x4x4": {
        "isdf": f"{BASE}/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16/work_sym/tmp/isdf_tensors_792.h5",
        "zeta": None,
        "wfn":  None,
        "sys_dim": 3,
        "nocc": 8,      # Si 8 valence e, bispinor bands
    },
}

RY_TO_MEV = 13605.693122994

# ---------------------------------------------------------------------------
# fixture loading
# ---------------------------------------------------------------------------

class Fixture:
    pass


def load_fixture(name):
    fx = Fixture()
    fx.name = name
    spec = FIX[name]
    fx.sys_dim = spec["sys_dim"]
    with h5py.File(spec["isdf"], "r") as f:
        fx.psi = f["psi_full_y"][()]                 # (nk, nb, ns, n_mu)
        fx.enk = f["enk_full"][()]                   # (nk, nb) Ry
        V = f["V_qmunu"][()]
        W = f["W0_qmunu"][()]
        if V.ndim == 8:      # sharded-mesh layout (1,1,1,kx,ky,kz,mu,mu)
            nmu = V.shape[-1]
            V = V[0, 0, 0].reshape(-1, nmu, nmu)
            W = W[0, 0, 0].reshape(-1, nmu, nmu)
        fx.Vdisk = V
        fx.W0 = W
        if "kgrid" in f:
            fx.kgrid = tuple(int(x) for x in f["kgrid"][()])
        else:
            fx.kgrid = tuple(spec["kgrid_fallback"])
        fx.vhead = complex(f["vhead"][()]) if "vhead" in f else None
    fx.nk, fx.nb, fx.ns, fx.nmu = fx.psi.shape
    fx.nq = int(np.prod(fx.kgrid))
    assert fx.nq == fx.nk == fx.Vdisk.shape[0], (fx.nq, fx.nk, fx.Vdisk.shape)

    fx.has_zeta = spec["zeta"] is not None
    if fx.has_zeta:
        with h5py.File(spec["zeta"], "r") as f:
            fx.ZG = f["zeta_q_G"][()]                        # (nq, nmu, ngkmax)
            fx.gvec = f["isdf_header/gvec_components"][()]   # (nq, 3, ngkmax)
            fx.ngk = f["isdf_header/ngk"][()].astype(int)
            fx.g0_mu = f["g0_mu"][()]
            fx.r_mu_fft_idx = f["isdf_header/centroids/r_mu_fft_idx"][()]
            fx.fg = tuple(int(x) for x in f["mf_header/gspace/FFTgrid"][()])
            fx.qfr = f["mf_header/kpoints/rk"][()]           # (nq, 3)
            fx.bdot = f["mf_header/crystal/bdot"][()]
            fx.bvec = f["mf_header/crystal/bvec"][()]        # units of blat=2pi/alat
            fx.blat = float(f["mf_header/crystal/blat"][()])
            fx.celvol = float(f["mf_header/crystal/celvol"][()])
            fx.ecutwfc = float(f["mf_header/kpoints/ecutwfc"][()])
            fx.el = f["mf_header/kpoints/el"][()]            # (1, nk, mnband) Ry
            fx.ifmax = f["mf_header/kpoints/ifmax"][()][0]   # (nk,)
        # production Cartesian reciprocal vectors in TRUE bohr^-1
        # (gw_init.py:292  bvec = blat * wfn.bvec)
        fx.bvec_bohr = fx.blat * fx.bvec
        fx.ngkmax = fx.ZG.shape[2]
    else:
        # q fractions from C-order enumeration (asserted against rk when zeta
        # exists; here it IS the definition).
        kg = fx.kgrid
        fx.qfr = np.array([[ix / kg[0], iy / kg[1], iz / kg[2]]
                           for ix in range(kg[0]) for iy in range(kg[1])
                           for iz in range(kg[2])])
        fx.bdot = fx.bvec = None
        fx.el = None
        fx.ifmax = None

    # integer 3-vector per q + flat C-order map; assert rk ordering is C-order
    kg = np.array(fx.kgrid)
    iq = np.rint(np.asarray(fx.qfr) * kg[None, :]).astype(int) % kg[None, :]
    fx.iq3 = iq
    # BGW wrap convention (v_q_g_flat.py:232): q_int > kg/2 -> q_int - kg.
    # ALL physical K evaluations and Bloch phases use fx.qwrap, never fx.qfr.
    iq_w = iq - kg[None, :] * ((2 * iq) > kg[None, :])
    fx.qwrap = iq_w / kg[None, :].astype(np.float64)
    flat = (iq[:, 0] * kg[1] + iq[:, 1]) * kg[2] + iq[:, 2]
    assert np.array_equal(flat, np.arange(fx.nq)), \
        f"{name}: rk not flat C-order; build explicit maps"
    # wrapped k-q index table: kmq_idx[q, k]
    d3 = (iq[None, :, :] - iq[:, None, :]) % kg[None, None, :]   # [q,k,3]
    fx.kmq_idx = (d3[..., 0] * kg[1] + d3[..., 1]) * kg[2] + d3[..., 2]
    # occupied count (bispinor band counting) for gap-window rows
    if fx.ifmax is not None:
        assert np.all(fx.ifmax == fx.ifmax[0])
        fx.nocc = int(fx.ifmax[0])
    else:
        fx.nocc = int(spec["nocc"])
        # sanity: global gap between nocc-1 and nocc bands
        gap = np.min(fx.enk[:, fx.nocc]) - np.max(fx.enk[:, fx.nocc - 1])
        assert gap > 0, f"{name}: nocc={fx.nocc} not gapped ({gap:.3e} Ry)"
    return fx


def load_wfn(fx, path):
    """WFN.h5 per-k G-space coefficients (window bands only) + box scatter.

    Returns dict with per-k (gvecs slice (ngk_k,3), coeffs (nb, ns, ngk_k) c128)
    and el (nk, mnband).
    """
    out = {}
    with h5py.File(path, "r") as f:
        ngk = f["mf_header/kpoints/ngk"][()].astype(int)
        el = f["mf_header/kpoints/el"][()][0]
        gv = f["wfns/gvecs"][()]
        co = f["wfns/coeffs"][()]          # (mnband, ns, ngktot, 2)
        rk = f["mf_header/kpoints/rk"][()]
    coeffs = co[..., 0] + 1j * co[..., 1]
    offs = np.concatenate([[0], np.cumsum(ngk)])
    per_k = []
    for k in range(len(ngk)):
        sl = slice(offs[k], offs[k + 1])
        per_k.append((gv[sl], coeffs[:fx.nb, :, sl]))
    out["per_k"] = per_k
    out["el"] = el
    out["rk"] = rk
    out["ngk"] = ngk
    return out


def wfn_u_on_grid(fx, wfn, k, bands):
    """u_{n,k}(r) on the FFT grid for selected bands: u = sum_G c(G) e^{iGr}.

    Returns (len(bands), ns, n_rtot) complex128, C-order flat grid.
    Normalization asserted against psi_full_y in the gates.
    """
    nx, ny, nz = fx.fg
    gvk, ck = wfn["per_k"][k]
    fi = ((gvk[:, 0] % nx) * ny + (gvk[:, 1] % ny)) * nz + (gvk[:, 2] % nz)
    box = np.zeros((len(bands), fx.ns, nx * ny * nz), dtype=np.complex128)
    box[:, :, fi] = ck[bands][:, :, :]
    # LORRAX normalization: psi_full_y == u/sqrt(N_r) with u = sum_G c e^{iGr}
    # (measured global scalar 1/sqrt(46080)=4.6585e-3, zero per-band phase).
    u = np.fft.ifftn(box.reshape(len(bands), fx.ns, nx, ny, nz),
                     axes=(2, 3, 4), norm="backward") * np.sqrt(nx * ny * nz)
    return u.reshape(len(bands), fx.ns, nx * ny * nz)


# ---------------------------------------------------------------------------
# verbatim-reused validated routines (provenance: interp_study/vq_loo.py,
# physical_contract.py) — flat_idx, recon, to_sphere, relF, C rebuild
# ---------------------------------------------------------------------------

def grid_geometry(fx):
    nx, ny, nz = fx.fg
    rx = np.arange(nx) / nx
    ry = np.arange(ny) / ny
    rz = np.arange(nz) / nz
    RX, RY, RZ = np.meshgrid(rx, ry, rz, indexing="ij")
    fx.rfrac = np.stack([RX.ravel(), RY.ravel(), RZ.ravel()], 1)
    fx.n_rtot = nx * ny * nz
    # centroid fractional coords from fft idx (C-order flat grid)
    idx = fx.r_mu_fft_idx
    fx.rmu_frac = idx.astype(np.float64) / np.array([nx, ny, nz])[None, :]
    fx.rmu_flat = ((idx[:, 0] % nx) * ny + (idx[:, 1] % ny)) * nz + (idx[:, 2] % nz)


def flat_idx(fx, gv):
    nx, ny, nz = fx.fg
    gx = gv[0] % nx; gy = gv[1] % ny; gz = gv[2] % nz
    return ((gx * ny) + gy) * nz + gz


def recon(fx, q):
    """zeta_q(mu,r) on the full FFT grid, band-limited to sphere(q). Verbatim
    vq_loo.py structure (validated round-trip), but Bloch phase at the BGW
    WRAPPED q (the writer's convention) — vq_loo used the unwrapped rk, which
    is self-consistent for round-trips but relabels the cell-periodic object
    by e^{2pi i G0.r} at wrap-affected q's."""
    nmu, n_rtot = fx.nmu, fx.n_rtot
    nx, ny, nz = fx.fg
    box = np.zeros((nmu, n_rtot), dtype=np.complex128)
    fi = flat_idx(fx, fx.gvec[q]); n = int(fx.ngk[q])
    box[:, fi[:n]] = fx.ZG[q][:, :n]
    R = np.fft.ifftn(box.reshape(nmu, nx, ny, nz), axes=(1, 2, 3),
                     norm="backward").reshape(nmu, n_rtot)
    return R * np.exp(2j * np.pi * (fx.rfrac @ fx.qwrap[q]))[None, :]


def to_sphere(fx, zeta_r, q, qfrac=None, gvec=None, ngkq=None):
    """forward: zeta(mu,r) -> zeta~(mu,G) on sphere(q). Verbatim vq_loo.py,
    wrapped-q Bloch phase; explicit (qfrac, gvec) for the seam checks."""
    nx, ny, nz = fx.fg
    qf = fx.qwrap[q] if qfrac is None else qfrac
    gv = fx.gvec[q] if gvec is None else gvec
    n = int(fx.ngk[q]) if ngkq is None else int(ngkq)
    ph = np.exp(-2j * np.pi * (fx.rfrac @ qf))
    box = np.fft.fftn((zeta_r * ph[None, :]).reshape(fx.nmu, nx, ny, nz),
                      axes=(1, 2, 3), norm="backward").reshape(fx.nmu, fx.n_rtot)
    fi = flat_idx(fx, gv)
    out = np.zeros((fx.nmu, fx.ngkmax), dtype=np.complex128)
    out[:, :n] = box[:, fi[:n]]
    return out


def relF(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def rebuild_C_falloff(fx):
    """C_q rebuilt exactly a la interp_study/falloff_cq.py == the production
    isdf.core.c_q_from_psi_sm formula (einsum 'kmna,knbr->karmb', |P_R|^2 spin
    contraction, forward-norm FFTs, final (0,2,1) transpose). Verbatim."""
    psi = fx.psi
    nkx, nky, nkz = fx.kgrid
    nq, nmu, ns = fx.nq, fx.nmu, fx.ns
    psiY = psi
    psiX = np.conj(psi).transpose(0, 3, 1, 2)
    P = np.einsum("kmna,knbr->karmb", psiX, psiY, optimize=True).reshape(
        nkx, nky, nkz, ns, nmu, nmu, ns)
    P_R = np.fft.ifftn(P, axes=(0, 1, 2), norm="forward")
    C_R = np.einsum("xyzavmb,xyzavmb->xyzvm", np.conj(P_R), P_R, optimize=True)
    C_q = np.transpose(
        np.fft.fftn(C_R, axes=(0, 1, 2), norm="forward").reshape(nq, nmu, nmu),
        (0, 2, 1))
    return C_q


# ---------------------------------------------------------------------------
# pair rows (spin-traced) + Grams on GPU
# ---------------------------------------------------------------------------

def pair_rows_k(fx, q, k, conj_left=True, psi=None):
    """Spin-traced pair rows at fixed k: rows (m,n) -> (nb*nb, nmu).

    conj_left=True :  x_p(mu) = sum_s conj(psi_{m,kmq,s}) psi_{n,k,s}   (code x)
    conj_left=False:  Y[p,mu] = sum_s psi_{m,kmq,s} conj(psi_{n,k,s})   (= conj x)
    """
    if psi is None:
        psi = fx.psi
    A = psi[fx.kmq_idx[q, k]]          # (nb, ns, nmu)  left leg (band m)
    Bm = psi[k]                        # (nb, ns, nmu)  right leg (band n)
    if conj_left:
        return jnp.einsum("asm,bsm->abm", jnp.conj(A), Bm).reshape(-1, fx.nmu)
    return jnp.einsum("asm,bsm->abm", A, jnp.conj(Bm)).reshape(-1, fx.nmu)


def gram_C_from_rows(fx, q, conj_left, psi=None):
    """C = sum_k X_k^H X_k on device."""
    C = jnp.zeros((fx.nmu, fx.nmu), dtype=jnp.complex128)
    for k in range(fx.nk):
        Xk = pair_rows_k(fx, q, k, conj_left, psi)
        C = C + Xk.conj().T @ Xk
    return np.asarray(C)


def cross_gram_H(fx, q, qp, t_by_k, conj_left, t_conj, psi=None):
    """H = sum_k X_{q,k}^H  [rotate-left-band(t_k) X_{qp,k}].

    t_by_k[k]: (nb, nb) unitary band link  t_{k-q <- k-qp}.
    Transported rows: X'[(m,n),nu] = sum_{m'} tt[m,m'] X_qp[(m',n),nu]
    with tt = conj(t) if t_conj else t (resolved by the gauge gate).
    """
    H = jnp.zeros((fx.nmu, fx.nmu), dtype=jnp.complex128)
    for k in range(fx.nk):
        Xq = pair_rows_k(fx, q, k, conj_left, psi).reshape(fx.nb, fx.nb, fx.nmu)
        Xp = pair_rows_k(fx, qp, k, conj_left, psi).reshape(fx.nb, fx.nb, fx.nmu)
        t = jnp.asarray(t_by_k[k])
        tt = jnp.conj(t) if t_conj else t
        Xrot = jnp.einsum("mM,Mnv->mnv", tt, Xp)
        H = H + jnp.einsum("abm,abn->mn",
                           jnp.conj(Xq), Xrot)
    return np.asarray(H)


# ---------------------------------------------------------------------------
# production Coulomb kernel + SR/LR split
# (formula verbatim from gw/compute_vcoul.py: v = 8*pi/denom * f2d / celvol,
#  denom = |bvec^T (q+G)|^2 in (2pi/alat)^2 units; zc = pi / bvec[2,2];
#  denom<1e-12 -> v=0 (== exact K=0 slot / production head exclusion))
# ---------------------------------------------------------------------------

K2_ZERO = 1e-12
K2_SMALL = 1e-8    # small-K series threshold (unexercised on these grids)


def vdim_at(fx, qG_frac):
    """Production bare kernel v_dim(K) at fractional K=(q+G) columns (3,nG).

    Verbatim compute_vcoul.compute_v_q_per_G with bvec = blat*bvec_hdr
    (true bohr^-1, gw_init.py:292) and vcoul_cutoff_ry = ecutwfc.
    Returns (v, denom_Ry). v has the exact-K=0 slot set to 0."""
    qG_cart = fx.bvec_bohr.T @ qG_frac
    denom = np.sum(qG_cart * qG_cart, axis=0)          # |K|^2 in true Ry
    zero = denom < K2_ZERO
    dsafe = np.where(zero, 1.0, denom)
    fact = 1.0 / fx.celvol
    if fx.sys_dim == 2:
        zc = float(np.pi / fx.bvec_bohr[2, 2])
        kxy = np.sqrt(qG_cart[0] ** 2 + qG_cart[1] ** 2)
        kz = qG_cart[2]
        f2d = 1.0 - np.exp(-zc * kxy) * np.cos(kz * zc)
        v = np.where(zero, 0.0, (8.0 * np.pi / dsafe) * f2d * fact)
    else:
        v = np.where(zero, 0.0, (8.0 * np.pi / dsafe) * fact)
    cut = getattr(fx, "ecutwfc", None)
    if cut is not None:
        v = np.where(denom > cut, 0.0, v)
    return v, denom


def v_split_at(fx, qG_frac, alpha_int):
    """(v_sr, v_lr) at fractional K columns; alpha_int in (2pi/alat)^-1-conjugate
    units (i.e. K^2/(4 alpha_int^2) with K^2 = denom).

    v_sr = v_dim * (-expm1(-K^2/4a^2))   [stable; response sec 5]
    v_lr = v_dim * exp(-K^2/4a^2)
    Small-K series branches (K2 < K2_SMALL, exact-zero slot excluded):
      3D:   v_sr -> (2pi/a^2) * (1 - K^2/(8a^2)) / celvol
      slab, Gz any: v_sr -> 8pi*f2d/K^2 * (K^2/4a^2)(1 - K^2/8a^2) which for
            Gz=0 vanishes linearly ~ 8pi*zc*kxy/(4a^2)/celvol.
    These branches are asserted-unreachable on the fixture grids (min finite
    K^2 ~ 0.04) but implemented per the response's mandate."""
    v, denom = vdim_at(fx, qG_frac)
    x = denom / (4.0 * alpha_int ** 2)
    sr_fac = -np.expm1(-x)
    lr_fac = np.exp(-x)
    v_sr = v * sr_fac
    v_lr = v * lr_fac
    small = (denom >= K2_ZERO) & (denom < K2_SMALL)
    if np.any(small):
        fact = 1.0 / fx.celvol
        if fx.sys_dim == 3:
            series = (2.0 * np.pi / alpha_int ** 2) * (
                1.0 - denom[small] / (8.0 * alpha_int ** 2)) * fact
            v_sr = v_sr.copy(); v_sr[small] = series
        else:
            qG_cart = fx.bvec_bohr.T @ qG_frac
            zc = float(np.pi / fx.bvec_bohr[2, 2])
            kxy = np.sqrt(qG_cart[0] ** 2 + qG_cart[1] ** 2)[small]
            kz = qG_cart[2][small]
            f2d = 1.0 - np.exp(-zc * kxy) * np.cos(kz * zc)
            series = 8.0 * np.pi * f2d / (4.0 * alpha_int ** 2) * (
                1.0 - denom[small] / (8.0 * alpha_int ** 2)) * fact
            v_sr = v_sr.copy(); v_sr[small] = series
        v_lr = v_lr.copy(); v_lr[small] = (v[small] - v_sr[small])
    return v_sr, v_lr


def sphere_K(fx, q):
    """fractional K=(qwrap+G) columns (3, ngk_q) for the stored sphere at q."""
    n = int(fx.ngk[q])
    return fx.qwrap[q][:, None] + fx.gvec[q][:, :n].astype(np.float64), n


def alpha_ladder(fx, c_alphas=(0.5, 1.0, 2.0, 4.0), L_R_bohr=10.0):
    """alpha in true bohr^-1 (denom is true Ry): alpha = c*(2pi/L_R)."""
    return {c: c * (2.0 * np.pi / L_R_bohr) for c in c_alphas}


def make_Vq_generic(zt, v, n):
    """V[mu,nu] = sum_G conj(zt_mu) v zt_nu  (verbatim vq_loo.py make_Vq)."""
    A = zt[:, :n] * np.sqrt(np.maximum(v[:n], 0.0))[None, :]
    return np.conj(A) @ A.T


# ---------------------------------------------------------------------------
# frames: eigh(C) -> R, S ; Phi ; whitened tiles Vc
# ---------------------------------------------------------------------------

def eigh_frame(C):
    """Descending eigh of Hermitian C on GPU. Returns (R, lam) with lam desc."""
    Ch = 0.5 * (C + C.conj().T)
    lam, R = jnp.linalg.eigh(jnp.asarray(Ch))
    lam = np.asarray(lam)[::-1].copy()
    R = np.asarray(R)[:, ::-1].copy()
    return R, lam


def phi_from_zeta(fx, q, R, S, phi_conj):
    """Phi(G) = S R^H (conj?)(zeta~) on sphere(q); rows ordered desc-sigma."""
    n = int(fx.ngk[q])
    Z = fx.ZG[q][:, :n]
    Zc = np.conj(Z) if phi_conj else Z
    return (S[:, None] * (R.conj().T @ Zc)), n


def vc_tile(Phi, v):
    """Vc = Phi diag(v) Phi^H, Hermitian PSD (v >= 0)."""
    A = Phi * np.sqrt(np.maximum(v, 0.0))[None, :]
    Vc = A @ A.conj().T
    return 0.5 * (Vc + Vc.conj().T)


# ---------------------------------------------------------------------------
# single-particle transport links
# ---------------------------------------------------------------------------

def regauge_u_to_fixture(fx, u_fields, psi_fix_k, bands):
    """Rotate WFN-derived u fields (nb_sel, ns, n_rtot) into the fixture's
    band gauge at one k.  The code's loader re-gauges degenerate multiplets
    (measured: WFN vs psi_full_y match at k=0 but rows anti-align at other k),
    so Omega = polar(<u_cen|psi_fix>) over the selected bands maps WFN gauge ->
    fixture gauge; exact up to the ISDF centroid quadrature (validated by the
    returned residual).  bands: list of band indices (into the window) that
    u_fields covers; psi_fix_k: fixture psi at this k, full window."""
    ucen = u_fields[:, :, fx.rmu_flat]                    # (nb_sel, ns, nmu)
    A = ucen.reshape(len(bands), -1)
    B = psi_fix_k[bands].reshape(len(bands), -1)
    O = np.conj(A) @ B.T                                  # <u|psi_fix>
    U, s, Vh = np.linalg.svd(O)
    Om = U @ Vh
    u_new = np.einsum("msr,mn->nsr", u_fields, Om)
    resid = np.linalg.norm(u_new[:, :, fx.rmu_flat] - psi_fix_k[bands]) \
        / np.linalg.norm(psi_fix_k[bands])
    return u_new, resid


def build_regauged_fields(fx, wfn):
    """All-window WFN u fields per k, rotated into the FIXTURE band gauge
    (the code's loader re-gauges degenerate multiplets: WFN matches
    psi_full_y at k=0 but not at all k).  Returns (UF dict k->(nb,ns,n_rtot),
    residuals per k)."""
    UF, res = {}, []
    allb = list(range(fx.nb))
    for k in range(fx.nk):
        u = wfn_u_on_grid(fx, wfn, k, allb)
        ug, r = regauge_u_to_fixture(fx, u, fx.psi[k], allb)
        UF[k] = ug
        res.append(r)
    return UF, np.array(res)


def band_overlaps_fields(fx, UF, pairs):
    """Exact band overlaps O[m,m'] = <u_{m,kap}|u_{m',kapp}> from regauged
    grid fields (Parseval == G-space sum; u normalized so <u|u> = 1).
    Returns dict[(kap,kapp)] -> (nb,nb) complex."""
    flat = {}

    def get(k):
        if k not in flat:
            flat[k] = jnp.asarray(UF[k].reshape(fx.nb, -1))
        return flat[k]

    out = {}
    for (ka, kb) in pairs:
        out[(ka, kb)] = np.asarray(get(ka).conj() @ get(kb).T)
    return out


def band_overlaps_centroid(fx, pairs):
    """Centroid-quadrature fallback: O ~ sum_s sum_mu conj(psi_kap) psi_kapp."""
    out = {}
    P = fx.psi.reshape(fx.nk, fx.nb, fx.ns * fx.nmu)
    for (ka, kb) in pairs:
        out[(ka, kb)] = np.asarray(np.conj(P[ka]) @ P[kb].T)
    return out


def polar_unitary(O):
    """Closest unitary to O (SVD polar factor)."""
    U, s, Vh = np.linalg.svd(O)
    return U @ Vh, s


def edge_links_T(fx, q, qp, overlaps, R_by_q, S_by_q, rank, conj_left, t_conj,
                 psi=None, H_cache=None):
    """T_{q<-qp} at given rank + principal cosines. H cached per (q,qp)."""
    key = (q, qp)
    if H_cache is not None and key in H_cache:
        H = H_cache[key]
    else:
        t_by_k = []
        for k in range(fx.nk):
            ka, kb = fx.kmq_idx[q, k], fx.kmq_idx[qp, k]
            t, _ = polar_unitary(overlaps[(ka, kb)])
            t_by_k.append(t)
        H = cross_gram_H(fx, q, qp, t_by_k, conj_left, t_conj, psi)
        if H_cache is not None:
            H_cache[key] = H
    Rq, Sq = R_by_q[q][:, :rank], S_by_q[q][:rank]
    Rp, Sp = R_by_q[qp][:, :rank], S_by_q[qp][:rank]
    M = (Rq.conj().T @ H @ Rp) / Sq[:, None] / Sp[None, :]
    U, cos, Vh = np.linalg.svd(M)
    return U @ Vh, cos


# ---------------------------------------------------------------------------
# gap-window physical metrics
# ---------------------------------------------------------------------------

def gap_window_rows(fx, q, nv=3, nc=3, psi=None):
    """Spin-traced M rows (code x convention, = bse_serial.compute_pair_amplitude
    with the conduction leg at wrapped k-q):
        M[(k,c,v), mu] = sum_s conj(psi_{c,k-q,s}(mu)) psi_{v,k,s}(mu)
    v in top-nv valence, c in bottom-nc conduction (from ifmax/nocc)."""
    if psi is None:
        psi = fx.psi
    occ = fx.nocc
    vb = list(range(occ - nv, occ))
    cb = list(range(occ, occ + nc))
    rows = []
    for k in range(fx.nk):
        A = psi[fx.kmq_idx[q, k]][cb]     # (nc, ns, nmu)
        Bv = psi[k][vb]                   # (nv, ns, nmu)
        rows.append(np.einsum("csm,vsm->cvm", np.conj(A), Bv).reshape(-1, fx.nmu))
    return np.concatenate(rows, 0)        # (nk*nc*nv, nmu)


def B_from_tile(Mrows, V):
    """B[p,p'] = sum_{mu,nu} conj(M_p(mu)) V[mu,nu] M_p'(nu)."""
    return np.conj(Mrows) @ V @ Mrows.T


def B_metrics(Bp, Bt):
    """(relF, med-elem, p90-elem) with element rel err on |Bt| (floored)."""
    floor = 1e-14 * np.max(np.abs(Bt))
    rel = np.abs(Bp - Bt) / np.maximum(np.abs(Bt), floor)
    return relF(Bp, Bt), float(np.median(rel)), float(np.percentile(rel, 90))


# ---------------------------------------------------------------------------
# TDA exciton assembly (conventions verbatim from bse_serial.py:
#  K^x = (1/Nk) conj(M) V M^T ; W-direct ortho-FFT convolution == dense
#  (1/Nk) [sum_t conj(psi_c[k]) psi_c[k']] W_{k-k'} [sum_s psi_v[k] conj(psi_v[k'])];
#  H = D + K^x - W ; no singlet factor (bispinor). Finite q0: conduction leg
#  at wrapped k-q0 everywhere; W tile index k-k' unchanged.)
# ---------------------------------------------------------------------------

def assemble_H(fx, q0, B_block, nv=3, nc=3):
    """Dense TDA H at exciton momentum q0 from stored ingredients, with the
    exchange block passed in (B_block = conj(M) V M^T, any provenance).
    Basis order matches gap_window_rows: p = (k, c, v) C-order."""
    occ = fx.nocc
    vb = list(range(occ - nv, occ))
    cb = list(range(occ, occ + nc))
    nk, nmu = fx.nk, fx.nmu
    dim = nk * nc * nv
    # D
    D = np.zeros(dim)
    for k in range(nk):
        kc = fx.kmq_idx[q0, k]
        d = fx.enk[kc][cb][:, None] - fx.enk[k][vb][None, :]
        D[k * nc * nv:(k + 1) * nc * nv] = d.ravel()
    H = np.diag(D).astype(np.complex128)
    # exchange
    H += B_block / nk
    # direct W
    W = fx.W0    # (nq, mu, nu)
    psi = fx.psi
    for k in range(nk):
        kc = fx.kmq_idx[q0, k]
        for kp in range(nk):
            kcp = fx.kmq_idx[q0, kp]
            # W tile at wrapped (k - kp): flat index of wrap(ik - ikp)
            d3 = (fx.iq3[k] - fx.iq3[kp]) % np.array(fx.kgrid)
            iw = int((d3[0] * fx.kgrid[1] + d3[1]) * fx.kgrid[2] + d3[2])
            Cp = np.einsum("csm,dsm->cdm", np.conj(psi[kc][cb]), psi[kcp][cb])
            Vp = np.einsum("vsn,wsn->vwn", psi[k][vb], np.conj(psi[kp][vb]))
            blk = np.einsum("cdm,mn,vwn->cvdw", Cp, W[iw], Vp) / nk
            H[k * nc * nv:(k + 1) * nc * nv, kp * nc * nv:(kp + 1) * nc * nv] -= \
                blk.reshape(nc * nv, nc * nv)
    return 0.5 * (H + H.conj().T)


def lowest_eigs(H, n=4):
    w = np.linalg.eigvalsh(H)
    return w[:n]


# ---------------------------------------------------------------------------
# interpolation weights
# ---------------------------------------------------------------------------

def wrap_int(n, N):
    return n - N * ((2 * n) > N)


def r_sorted(fx):
    """R vectors of the coarse grid sorted by real-space metric length
    (verbatim vq_loo.py: adot approx from bdot inverse; ordering only)."""
    kg = fx.kgrid
    Rvecs = np.array([[wrap_int(ix, kg[0]), wrap_int(iy, kg[1]), wrap_int(iz, kg[2])]
                      for ix in range(kg[0]) for iy in range(kg[1])
                      for iz in range(kg[2])])
    if fx.bdot is not None:
        adot = np.linalg.inv(fx.bdot) * (2 * np.pi) ** 2
    else:
        adot = np.eye(3)
    Rdist = np.sqrt(np.abs(np.einsum("ri,ij,rj->r", Rvecs, adot, Rvecs)))
    return Rvecs[np.argsort(Rdist, kind="stable")]


def w_fourier(fx, q0_frac, train_fracs, nR):
    """Truncated-R Fourier predictor weights (verbatim vq_loo.py
    interp_ingredient: w = f0 @ pinv(F))."""
    Rset = r_sorted(fx)[:nR]
    F = np.exp(-2j * np.pi * (np.asarray(train_fracs) @ Rset.T))
    f0 = np.exp(-2j * np.pi * (np.asarray(q0_frac) @ Rset.T))
    return f0 @ np.linalg.pinv(F)


def w_nn_avg(fx, q0):
    """On-grid LOO nonneg stencil: average of the +-e_x, +-e_y (,+-e_z)
    neighbors on the wrapped torus (exact for q-linear functions)."""
    kg = np.array(fx.kgrid)
    dirs = []
    for ax in range(3):
        if kg[ax] > 1:
            for sgn in (+1, -1):
                e = np.zeros(3, int); e[ax] = sgn
                dirs.append(e)
    idx = []
    for e in dirs:
        j3 = (fx.iq3[q0] + e) % kg
        idx.append(int((j3[0] * kg[1] + j3[1]) * kg[2] + j3[2]))
    w = {}
    for j in idx:
        w[j] = w.get(j, 0.0) + 1.0 / len(idx)
    return w


def w_multilinear(fx, Q_frac, coarse_iq3, coarse_kg):
    """Nonneg multilinear weights of Q_frac in the coarse cell (wrapped torus).
    coarse_iq3: integer coords of the coarse points on the COARSE grid
    (ckg = coarse_kg); returns dict coarse_index -> weight."""
    ckg = np.array(coarse_kg)
    t = np.asarray(Q_frac) * ckg           # position in coarse-grid units
    out = {}
    lo = np.floor(t).astype(int)
    fr = t - lo
    axes = [ax for ax in range(3) if ckg[ax] > 1]
    corners = [(0, 0, 0)]
    for ax in axes:
        corners = [c[:ax] + (b,) + c[ax + 1:] for c in corners for b in (0, 1)]
    lut = {tuple(v): i for i, v in enumerate(coarse_iq3)}
    for c in corners:
        j3 = tuple((lo + np.array(c)) % ckg)
        wgt = 1.0
        for ax in range(3):
            wgt *= (1.0 - fr[ax]) if c[ax] == 0 else fr[ax]
        if wgt > 1e-15:
            j = lut[j3]
            out[j] = out.get(j, 0.0) + wgt
    return out


def save_flags(flags, path=f"{STUDY}/proto0_conventions.json"):
    with open(path, "w") as f:
        json.dump(flags, f, indent=1)


def load_flags(path=f"{STUDY}/proto0_conventions.json"):
    with open(path) as f:
        return json.load(f)
