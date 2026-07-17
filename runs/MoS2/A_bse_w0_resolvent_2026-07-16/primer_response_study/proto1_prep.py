"""proto1_prep — shared prep for the C2 prototype (primer-response study).

Scope: loaders + validated routines + conventions + self-check battery for the
C2 construction (global BZ-periodic frame / four-tails / spectral-Phi interp).
Plain numpy + h5py; jax.numpy optionally for eigh/SVD hot spots (owner ruling).
READ-ONLY on all fixtures and on sources/lorrax_A.

Production mapping (comments only, per owner sharding note): every eigh/SVD/
polar on (n_mu, n_mu) here becomes the N_mu^2 P('x','y')-sharded distributed
form via the cusolvermp/slate FFI (src/ffi) in production; the per-edge H
Grams become sharded zgemms. Deferred until results warrant.

CONVENTIONS (the single place where they live; derived from source read-only):

1. Torus (wrapped-label) convention. LORRAX builds C_q by lattice-FFT
   convolution over the coarse k-torus (isdf/core.py::c_q_from_psi_sm,
   einsum 'kmna,knbr->karmb' -> ifftn_k -> spin-contract -> fftn_k).
   Expanding that convolution exactly (see report):
       C_q[mu,nu] = sum_{k,n,m} conj(rho_knm(r_mu)) rho_knm(r_nu),
       rho_knm(r) = sum_s conj(u_{n, wrap(k-q), s}(r)) u_{m, k, s}(r),
   with u = stored cell-periodic spinors at WRAPPED k labels and NO umklapp
   phases anywhere (the two spin sums factorize per leg; the "4-channel"
   sum_ab |P_ab|^2 collapses to the Gram of SPIN-TRACED pair rows).
   Therefore X_q[p=(k,n,m), mu] = rho_knm(r_mu) and X^H X == C_q exactly.
   Consequence for the seam: within torus labels q+G0 IS q_wrap and
   X(q+G0) == X(q_wrap) with D == I; the spec's D_{G0}=diag(e^{-iG0.r_mu})
   seam identity is a statement about the UNWRAPPED/glued (phys) convention
   X_phys(q)[p,mu] = e^{-i G0(k,q).r_mu} X_torus(q)[p,mu] (G0(k,q) = wrap
   vector of k-q), which we verify separately (selfcheck_seam_phys).

2. V-tile orientation (make_Vq, == gw/v_q_g_flat.py `conj(L)*v @ R.T`):
       V[mu,nu] = sum_G conj(zt_mu(q+G)) v(q+G) zt_nu(q+G)      (Hermitian PSD)
   Whitened frame: eigh(C) = R diag(S^2) R^H (S real >0, fixed global rank r):
       Phi   = S R^H zeta      (both reps; Phi_G = S R^H zt)     [response eq]
       K     := contract(Phi_G) with the SAME conj-on-left rule:
                K[i,j] = sum_G conj(Phi_G[i,G]) v Phi_G[j,G]
   MATH CORRECTION to the task spec: with these definitions the identity is
       K == (S R^H V R S)^T        (equivalently  K^T == S R^H V R S),
   NOT K == S R^H V R S. The transpose arises because Phi_G = S R^H zt
   contracts as [S R^T V conj(R) S] under the conj-on-left rule. One
   consistent orientation is all that matters; we assert the ^T identity to
   1e-12 and carry K (conj-on-left convention) everywhere.
   Physical exchange block (standard BSE orientation, conj on left):
       B[p,p'] = sum_G conj(Mg[p,G]) v Mg[p',G],  Mg = pair rows on sphere.
   ISDF/whitened rep: Mg ~= a @ Phi_G with a = M_centroid_rows @ R S^{-1};
   then B = conj(a) K a^T.

3. Coulomb: slab (sys_dim=2) production kernel (gw/compute_vcoul.py):
       v(K) = 8 pi / K^2 * f2d * (1/V_cell),   f2d = 1 - exp(-zc kxy) cos(kz zc),
       zc = pi / bvec[2,2]  (bvec units: |bvec^T g|^2 in Ry).
   Head (K^2 < 1e-12) -> 0 (G=0 excluded everywhere here).
   SR/LR split: v_LR = v * exp(-K^2/(4 alpha^2)), v_SR = v * (-expm1(-K^2/(4 alpha^2)));
   for the slab at G_z=0, K_par->0: v_SR -> 0 linearly (no 3D 2pi/alpha^2
   constant) — the full dimension-specific v_dim is used, per response sec 5.

Provenance of verbatim-reused routines (interp_study, validated there):
   flat_idx / recon / to_sphere : vq_loo.py lines 61-83
   order-robust C_q rebuild     : physical_contract.py lines 30-38
   truncated-R Fourier weights  : vq_loo.py interp_ingredient (pinv form)
   relF                         : vq_loo.py line 126
"""
import numpy as np
import h5py

BASE = "/pscratch/sd/j/jackm/lorrax_sandbox"
FIX = {
    "MoS2_3x3": {
        "restart": f"{BASE}/runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native/tmp/isdf_tensors_640.h5",
        "zeta":    f"{BASE}/runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native/tmp/zeta_q.h5",
        "wfn":     f"{BASE}/runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native/WFN.h5",
    },
    "MoS2_6x6": {
        "restart": f"{BASE}/runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/mos2_6x6/lorrax/tmp/isdf_tensors_640.h5",
        "zeta":    f"{BASE}/runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/mos2_6x6/lorrax/tmp/zeta_q.h5",
        "wfn":     f"{BASE}/runs/MoS2/A_bse_w0_resolvent_2026-07-16/interp_study/mos2_6x6/qe/nscf/WFN.h5",
    },
}


def relF(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


class Fixture:
    """All arrays + grid machinery for one fixture. Everything read-only."""

    def __init__(self, name):
        p = FIX[name]
        self.name = name
        with h5py.File(p["restart"], "r") as f:
            self.psi = f["psi_full_y"][()]           # (nk, nb, ns, n_mu) u at centroids
            self.kgrid = f["kgrid"][()].astype(int)
            self.Vqmunu = f["V_qmunu"][()]           # (nq, mu, nu) head-zeroed bare tile
            self.W0 = f["W0_qmunu"][()]
            self.enk = f["enk_full"][()]             # (nk, nb) Ry
            self.vhead = float(np.real(f["vhead"][()])) if "vhead" in f else None
            self.g0vec = f["G0_mu_nu"][()] if "G0_mu_nu" in f else None
        with h5py.File(p["zeta"], "r") as f:
            self.ZG = f["zeta_q_G"][()]              # (nq, n_mu, ngkmax) c128
            self.gvec = f["isdf_header/gvec_components"][()].astype(np.int64)
            self.ngk = f["isdf_header/ngk"][()].astype(int)
            self.fg = f["mf_header/gspace/FFTgrid"][()].astype(int)
            # mf_header rk stores the UNWRAPPED QE list (0..(N-1)/N); the
            # stored zeta spheres are built around the BGW-WRAPPED q
            # (q > 1/2 -> q-1). Measured: at the 5 wrap-affected q's the
            # unwrapped |q+G|^2 runs to 53 Ry on a 30 Ry sphere and
            # make_Vq-vs-disk fails at 0.6-0.8 (TRS partners!), while the
            # wrapped form matches at ~1e-9 everywhere. Wrap here; the
            # R-DFT phases e^{-2pi i q.R} are identical mod 1.
            qraw = f["mf_header/kpoints/rk"][()]
            self.qfr_raw = qraw
            self.qfr = qraw - np.round(qraw)          # (nq,3) BGW-wrapped
            self.bdot = f["mf_header/crystal/bdot"][()]
            self.adot = f["mf_header/crystal/adot"][()]
            # BGW stores bvec in units of blat=2pi/alat; physical Bohr^-1
            # (i.e. sqrt(Ry)) needs the blat factor — measured as the 10.4%
            # makeVq-vs-disk residual (= blat^2 - 1) before this fix.
            self.blat = float(np.real(f["mf_header/crystal/blat"][()]))
            self.bvec = f["mf_header/crystal/bvec"][()] * self.blat
            self.celvol = float(np.real(f["mf_header/crystal/celvol"][()]))
            self.r_mu_fft_idx = f["isdf_header/centroids/r_mu_fft_idx"][()].astype(int)
            self.zeta_cutoff = float(f["isdf_header/zeta_cutoff_ry"][()])
            self.ifmax = f["mf_header/kpoints/ifmax"][()]
        self.wfn_path = p["wfn"]
        self.nk, self.nb, self.ns, self.n_mu = self.psi.shape
        self.nq = self.ZG.shape[0]
        self.ngkmax = self.ZG.shape[2]
        self.nx, self.ny, self.nz = [int(x) for x in self.fg]
        self.n_rtot = self.nx * self.ny * self.nz
        assert self.nk == self.nq
        kg = self.kgrid
        # integer grid coords of stored k/q points + lookup (wrap-safe)
        self.k_int = np.rint(self.qfr_raw * kg[None, :]).astype(int) % kg[None, :]
        self.k_lookup = {tuple(v): i for i, v in enumerate(self.k_int)}
        assert len(self.k_lookup) == self.nq
        # fractional r grid (C-order flat), and centroid fractional coords
        rx = np.arange(self.nx) / self.nx
        ry = np.arange(self.ny) / self.ny
        rz = np.arange(self.nz) / self.nz
        RX, RY, RZ = np.meshgrid(rx, ry, rz, indexing="ij")
        self.rfrac = np.stack([RX.ravel(), RY.ravel(), RZ.ravel()], 1)
        self.rmu_frac = self.r_mu_fft_idx / np.array([self.nx, self.ny, self.nz])[None, :]
        self.rmu_flat = ((self.r_mu_fft_idx[:, 0] * self.ny) + self.r_mu_fft_idx[:, 1]) * self.nz \
            + self.r_mu_fft_idx[:, 2]
        # valence count (SOC bispinor: ifmax bands occupied)
        self.nv = int(self.ifmax.ravel()[0])
        assert np.all(self.ifmax == self.nv)
        self._wfn_cache = None

    # ---------------- index helpers -----------------------------------------
    def kq_index(self, ki, qi):
        """index of wrap(k - q) on the stored grid + the wrap vector G0 (int)."""
        d = self.k_int[ki] - self.k_int[qi]
        w = d % self.kgrid
        G0 = (d - w) // self.kgrid          # k - q = w + G0*kgrid  (frac: w/N + G0)
        return self.k_lookup[tuple(w)], G0

    def flat_idx(self, gv):
        """vq_loo.py flat_idx (verbatim): (3, n) int Miller -> flat C-order FFT idx."""
        gx = gv[0] % self.nx
        gy = gv[1] % self.ny
        gz = gv[2] % self.nz
        return ((gx * self.ny) + gy) * self.nz + gz

    # ---------------- zeta recon / sphere (vq_loo verbatim) ------------------
    def recon(self, q, rows=None):
        """zeta_q(mu, r) lab frame on the full grid (band-limited to sphere(q))."""
        ZGq = self.ZG[q] if rows is None else self.ZG[q][rows]
        nmu = ZGq.shape[0]
        box = np.zeros((nmu, self.n_rtot), dtype=np.complex128)
        fi = self.flat_idx(self.gvec[q])
        n = int(self.ngk[q])
        box[:, fi[:n]] = ZGq[:, :n]
        R = np.fft.ifftn(box.reshape(nmu, self.nx, self.ny, self.nz),
                         axes=(1, 2, 3), norm="backward").reshape(nmu, self.n_rtot)
        return R * np.exp(2j * np.pi * (self.rfrac @ self.qfr[q]))[None, :]

    def to_sphere(self, zr, q):
        """forward: rows(r) -> rows(G) on sphere(q). Linear in the row axis."""
        ph = np.exp(-2j * np.pi * (self.rfrac @ self.qfr[q]))
        box = np.fft.fftn((zr * ph[None, :]).reshape(-1, self.nx, self.ny, self.nz),
                          axes=(1, 2, 3), norm="backward").reshape(zr.shape[0], self.n_rtot)
        fi = self.flat_idx(self.gvec[q])
        n = int(self.ngk[q])
        out = np.zeros((zr.shape[0], self.ngkmax), dtype=np.complex128)
        out[:, :n] = box[:, fi[:n]]
        return out

    # ---------------- Coulomb kernels ---------------------------------------
    def Kvecs(self, q):
        """|q+G| Cartesian (3, ngk) in sqrt(Ry) units + K^2, on sphere(q)."""
        n = int(self.ngk[q])
        qG = self.qfr[q][:, None] + self.gvec[q][:, :n].astype(np.float64)
        Kc = self.bvec.T @ qG
        return Kc, np.sum(Kc * Kc, axis=0), n

    def vq(self, q, kind="slab", alpha=None, taper=None):
        """v(q+G) on sphere(q).  kind: slab|slab_sr|slab_lr|bare3d.
        Head handling matches production compute_v_q_per_G: only the true
        divergence K^2 < 1e-12 (the q=0, G=0 slot) is zeroed; at q != 0 the
        finite G=0 term is part of the body (measured: zeroing G=0 at all q
        moves makeVq-vs-disk from ~1e-3 to 0.33 — disk includes it).
        taper=(K0_ry, K1_ry): multiply by w_cut^2 (cos^2 rolloff, two legs)."""
        Kc, K2, n = self.Kvecs(q)
        zero = K2 < 1e-12
        K2s = np.where(zero, 1.0, K2)
        zc = np.pi / self.bvec[2, 2]
        f2d = 1.0 - np.exp(-zc * np.sqrt(Kc[0] ** 2 + Kc[1] ** 2)) * np.cos(Kc[2] * zc)
        v = 8.0 * np.pi / K2s * f2d / self.celvol
        if kind == "bare3d":
            v = 8.0 * np.pi / K2s / self.celvol
        if kind in ("slab_sr", "slab_lr"):
            assert alpha is not None
            g = np.exp(-K2 / (4.0 * alpha ** 2))
            v = v * g if kind == "slab_lr" else v * (-np.expm1(-K2 / (4.0 * alpha ** 2)))
        v = np.where(zero, 0.0, v)
        if taper is not None:
            K0, K1 = taper
            x = np.clip((K2 - K0) / max(K1 - K0, 1e-30), 0.0, 1.0)
            w = np.cos(0.5 * np.pi * x) ** 2
            v = v * w * w      # squared window: two auxiliary-function legs
        return v, n

    def make_Vq(self, zt, q, **kw):
        """V[mu,nu] = sum_G conj(zt_mu) v zt_nu  (vq_loo convention, slab v)."""
        v, n = self.vq(q, **kw)
        A = zt[:, :n] * np.sqrt(v[:n])[None, :]
        return np.conj(A) @ A.T

    def contractK(self, PhiG, q, **kw):
        """Same conj-on-left contraction for whitened rows: K[i,j]=sum conj(Phi_i) v Phi_j."""
        return self.make_Vq(PhiG, q, **kw)

    # ---------------- C_q rebuild (order-robust; physical_contract.py) -------
    def build_Cq(self):
        psi = self.psi
        nq, nb, ns, n_mu = self.nq, self.nb, self.ns, self.n_mu
        kg = self.kgrid
        psiX = np.conj(psi).transpose(0, 3, 1, 2)
        P = np.einsum("kmna,knbr->karmb", psiX, psi, optimize=True)
        Rall = np.array([[rx, ry, rz] for rx in range(kg[0])
                         for ry in range(kg[1]) for rz in range(kg[2])])
        Rw = ((Rall + kg // 2) % kg) - (kg // 2)
        EqR = np.exp(2j * np.pi * (self.qfr @ Rw.T))
        P_R = (EqR.T @ P.reshape(nq, -1)).reshape(len(Rw), ns, n_mu, n_mu, ns)
        C_R = np.einsum("ravmb,ravmb->rvm", np.conj(P_R), P_R, optimize=True)
        C_q = np.transpose(((np.exp(-2j * np.pi * (self.qfr @ Rw.T)) / nq)
                            @ C_R.reshape(len(Rw), -1)).reshape(nq, n_mu, n_mu), (0, 2, 1))
        self.Rw = Rw
        self.C_R_full = C_R          # (nR, nu, mu) lattice image (for shell reference)
        return C_q

    # ---------------- pair rows at centroids --------------------------------
    def build_X(self, q, kset=None, phys=False):
        """X[p=(k,n,m), mu] = sum_s conj(u_{n, wrap(k-q), s}(mu)) u_{m, k, s}(mu).
        phys=True: multiply rows by the glued-continuation umklapp phase
        e^{-i G0(k,q) . r_mu} (unwrapped convention; for seam checks only)."""
        ks = range(self.nk) if kset is None else kset
        out = np.empty((len(list(ks)), self.nb, self.nb, self.n_mu), dtype=np.complex128)
        ks = range(self.nk) if kset is None else kset
        for i, k in enumerate(ks):
            kq, G0 = self.kq_index(k, q)
            row = np.einsum("nsm,Msm->nMm", np.conj(self.psi[kq]), self.psi[k])
            if phys and np.any(G0):
                ph = np.exp(-2j * np.pi * (self.rmu_frac @ G0.astype(float)))
                row = row * ph[None, None, :]
            out[i] = row
        return out.reshape(-1, self.n_mu)

    def gap_window_pairs(self, q, nvw=3, ncw=3):
        """Spin-traced BSE exchange rows M_cvk(mu) = sum_s conj(u_{c,k-q,s}) u_{v,k,s},
        c in bottom-ncw conduction, v in top-nvw valence, all k. (npair_w, n_mu)."""
        nv = self.nv
        cs = range(nv, nv + ncw)
        vs = range(nv - nvw, nv)
        rows = np.empty((self.nk, ncw, nvw, self.n_mu), dtype=np.complex128)
        for k in range(self.nk):
            kq, _ = self.kq_index(k, q)
            rows[k] = np.einsum("csm,vsm->cvm",
                                np.conj(self.psi[kq][list(cs)]), self.psi[k][list(vs)])
        return rows.reshape(-1, self.n_mu)

    # ---------------- WFN loader + band overlaps ----------------------------
    def load_wfn(self):
        if self._wfn_cache is not None:
            return self._wfn_cache
        with h5py.File(self.wfn_path, "r") as f:
            co = f["wfns/coeffs"][()]              # (nb_w, ns, ngktot, 2)
            gv = f["wfns/gvecs"][()]               # (ngktot, 3)
            ngk_k = f["mf_header/kpoints/ngk"][()].astype(int)
            el = f["mf_header/kpoints/el"][()]     # (1, nk, nb_w) Ry
            rk = f["mf_header/kpoints/rk"][()]
        c = co[..., 0] + 1j * co[..., 1]
        off = np.concatenate([[0], np.cumsum(ngk_k)])
        per_k = []
        for k in range(self.nk):
            sl = slice(off[k], off[k + 1])
            per_k.append({"c": np.ascontiguousarray(c[:, :, sl]),
                          "g": np.ascontiguousarray(gv[sl]),
                          "lut": {tuple(v): i for i, v in enumerate(gv[sl])}})
        self._wfn_cache = {"per_k": per_k, "el": el[0], "rk": rk}
        return self._wfn_cache

    def u_grid(self, k, nbmax=None):
        """u_{n,k,s}(r) on the full FFT grid from WFN coeffs: sum_G c e^{2pi i G.r}."""
        w = self.load_wfn()["per_k"][k]
        nbw = w["c"].shape[0] if nbmax is None else nbmax
        box = np.zeros((nbw, self.ns, self.n_rtot), dtype=np.complex128)
        fi = self.flat_idx(w["g"].T)
        box[:, :, fi] = w["c"][:nbw]
        return np.fft.ifftn(box.reshape(nbw, self.ns, self.nx, self.ny, self.nz),
                            axes=(2, 3, 4), norm="forward").reshape(nbw, self.ns, self.n_rtot)

    def band_overlap_G(self, ka, kb, Gshift=(0, 0, 0), nbmax=None):
        """O[n,n'] = sum_{s,G} conj(c_{n,ka,s}(G)) c_{n',kb,s}(G + Gshift).
        Gshift: relabel for phys-convention seam edges (u_{k-G0}(G) = u_k(G-G0):
        overlap with the glued continuation at kb-G0 uses c_kb(G + G0))."""
        w = self.load_wfn()["per_k"]
        A, B = w[ka], w[kb]
        nbw = A["c"].shape[0] if nbmax is None else nbmax
        Gs = np.asarray(Gshift)
        idxA, idxB = [], []
        for i, g in enumerate(A["g"]):
            j = B["lut"].get(tuple(g + Gs))
            if j is not None:
                idxA.append(i)
                idxB.append(j)
        ca = A["c"][:nbw][:, :, idxA].reshape(nbw, -1)
        cb = B["c"][:nbw][:, :, idxB].reshape(nbw, -1)
        return np.conj(ca) @ cb.T

    def band_overlap_centroid(self, ka, kb, nbmax=None):
        nbw = self.nb if nbmax is None else nbmax
        return np.einsum("nsm,Nsm->nN", np.conj(self.psi[ka][:nbw]), self.psi[kb][:nbw])


def polar(M):
    """Unitary polar factor via SVD (Procrustes). Production: sharded FFI SVD."""
    U, s, Vh = np.linalg.svd(M)
    return U @ Vh, s


def frames_from_C(Cq, rank):
    """eigh -> (R, S) top-`rank` slice, descending. Production: cusolvermp eigh."""
    lam, R = np.linalg.eigh(0.5 * (Cq + Cq.conj().T))
    lam = lam[::-1]
    R = R[:, ::-1]
    lam_r = lam[:rank]
    assert lam_r[-1] > 0, f"rank {rank} hits nonpositive eigenvalue {lam_r[-1]:.3e}"
    return R[:, :rank], np.sqrt(lam_r), lam


def matlog_u(W):
    """Principal log of a unitary matrix: W = U e^{i theta} U^H -> U (i theta) U^H.
    Returns (L, theta) with theta in (-pi, pi]. Branch fragility if |theta|~pi."""
    ev, U = np.linalg.eig(W)
    th = np.angle(ev)
    L = (U * (1j * th)[None, :]) @ np.linalg.inv(U)
    return L, th


def truncR_weights(q_train, q0, Rset):
    """vq_loo interp_ingredient weights: w = f0 @ pinv(F). Rows of q in frac."""
    F = np.exp(-2j * np.pi * (q_train @ Rset.T))
    f0 = np.exp(-2j * np.pi * (q0 @ Rset.T))
    return f0 @ np.linalg.pinv(F)


# ===========================================================================
# Self-check battery (gate: run before any prototype result is reported)
# ===========================================================================
def run_selfchecks(fx, C_q, verbose=True):
    rep = {}

    def log(k, v, tol=None):
        rep[k] = v
        flag = "" if tol is None else ("  OK" if v <= tol else "  ** FAIL **")
        if verbose:
            print(f"  [check] {k:<46s} {v:.3e}{flag}")

    # 0a. recon/forward sphere round trip
    n0 = int(fx.ngk[0])
    zt = fx.to_sphere(fx.recon(0), 0)
    log("recon_roundtrip_sphere_Gamma", relF(zt[:, :n0], fx.ZG[0][:, :n0]), 1e-13)

    # 0a'. sphere consistency: with the WRAPPED q, every stored G satisfies
    # |q+G|^2 <= cutoff (catches the rk-unwrapped trap at wrap-affected q's)
    k2max = max(fx.Kvecs(q)[1].max() for q in range(fx.nq))
    log("sphere_max|q+G|^2_minus_cutoff", max(0.0, k2max - fx.zeta_cutoff), 1e-9)

    # 0b. WFN <-> psi_full_y consistency (u at centroids; pins normalization+order)
    ug = fx.u_grid(0, nbmax=fx.nb)[:, :, fx.rmu_flat]      # (nb, ns, n_mu)
    num = np.vdot(ug, fx.psi[0])
    sc = num / np.vdot(ug, ug)
    log("wfn_vs_psi_full_y_scale_resid_k0",
        np.linalg.norm(sc * ug - fx.psi[0]) / np.linalg.norm(fx.psi[0]), 1e-8)
    rep["wfn_scale"] = sc

    # 0c. X^H X == C_q (torus convention), Gamma + a finite q.
    # tol 2e-9: the FFT-convolution path and the direct Gram accumulate in
    # different orders; with cond(C)~1e7 the relative Frobenius agreement
    # floor is ~1e-11 (measured 6e-11/9e-11) — convention-pinning, not exact.
    for q in (0, 1):
        X = fx.build_X(q)
        log(f"XHX_vs_Cq_torus_q{q}", relF(np.conj(X.T) @ X, C_q[q]), 2e-9)

    # 0d. stored-V cross-check: make_Vq(slab) vs on-disk V_qmunu (report-only)
    Vd = fx.make_Vq(fx.ZG[1], 1)
    log("makeVq_slab_vs_disk_Vqmunu_q1", relF(Vd, fx.Vqmunu[1]))

    # (i) seam identity, PHYS convention: X_phys(q+G0) == X_phys(q_wrap) D_{G0}
    #     Torus statement (D==I) is exact by construction; verify the phys one
    #     by building X at a formal q+G0 through unwrapped labels.
    q = 1
    G0 = np.array([0, 1, 0])       # one reciprocal vector; q+G0 wraps to q itself
    Xp = fx.build_X(q, phys=True)
    # X_phys at q+G0: left leg at k-(q+G0): wrap vector gains +G0 for every k
    ph_extra = np.exp(-2j * np.pi * (fx.rmu_frac @ G0.astype(float)))
    Xp_sewn = Xp * ph_extra[None, :]
    # direct build with q+G0 folded in (same wrapped kq, wrap vector shifted):
    Xp2 = np.empty_like(Xp)
    i = 0
    for k in range(fx.nk):
        kq, G0k = fx.kq_index(k, q)
        row = np.einsum("nsm,Msm->nMm", np.conj(fx.psi[kq]), fx.psi[k])
        ph = np.exp(-2j * np.pi * (fx.rmu_frac @ (G0k + G0).astype(float)))
        Xp2[i:i + fx.nb * fx.nb] = (row * ph[None, None, :]).reshape(-1, fx.n_mu)
        i += fx.nb * fx.nb
    log("seam_phys_XqG0_vs_Xwrap_DG0", relF(Xp2, Xp_sewn), 1e-12)

    # (ii) G-relabel: lab-frame recon invariant under (q,zbar) -> (q+G0, shifted zbar)
    n1 = int(fx.ngk[q])
    lab1 = fx.recon(q)
    box = np.zeros((fx.n_mu, fx.n_rtot), dtype=np.complex128)
    gshift = fx.gvec[q][:, :n1] - G0[:, None]          # zbar_{q+G0}(G) = zbar_q(G+G0)
    box[:, fx.flat_idx(gshift)] = fx.ZG[q][:, :n1]
    lab2 = np.fft.ifftn(box.reshape(fx.n_mu, fx.nx, fx.ny, fx.nz),
                        axes=(1, 2, 3), norm="backward").reshape(fx.n_mu, fx.n_rtot)
    lab2 *= np.exp(2j * np.pi * (fx.rfrac @ (fx.qfr[q] + G0)))[None, :]
    log("seam_relabel_lab_recon_qG0", relF(lab2, lab1), 1e-12)

    # (v) per-G split exactness + small-K series guard
    for q in range(min(3, fx.nq)):
        v, n = fx.vq(q)
        vs, _ = fx.vq(q, kind="slab_sr", alpha=0.63)
        vl, _ = fx.vq(q, kind="slab_lr", alpha=0.63)
        log(f"vSR+vLR==v_q{q}", float(np.max(np.abs(vs[:n] + vl[:n] - v[:n]))
                                      / max(np.max(np.abs(v[:n])), 1e-300)), 1e-13)
    return rep
