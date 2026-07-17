"""proto0 script B — MoS2 6x6: OFF-GRID with truth (3x3 subgrid -> 27
complement q's) + on-grid 6x6 LOO.  C1 transported target-frame V^SR.

Coarse set = {0,2,4}x{0,2,4} subgrid of the 6x6 fixture (9 q, same centroids
and same fit window as the targets by construction); targets = the 27
complement q's, truth = stored 6x6 zeta~/V.  Also re-runs the slim Sec-3.5
baseline ladder off-grid under the primary metrics.
"""
import numpy as np
import jax
import jax.numpy as jnp

import prep
from prep import (load_fixture, load_wfn, grid_geometry, recon, to_sphere,
                  relF, rebuild_C_falloff, vdim_at, v_split_at, sphere_K,
                  alpha_ladder, make_Vq_generic, eigh_frame, phi_from_zeta,
                  vc_tile, polar_unitary, gap_window_rows, B_from_tile,
                  B_metrics, lowest_eigs, RY_TO_MEV, wrap_int, wfn_u_on_grid)

OUT = {}
flags = prep.load_flags()
CONJ_LEFT, T_CONJ = flags["CONJ_LEFT"], flags["T_CONJ"]
assert flags.get("FLAVOR") == "response"
PHI_CONJ = False    # response flavor: Phi_R = S R^H zeta~ (= L^H A)
print(f"[6x6] conventions: {flags}")

fx = load_fixture("mos2_6x6")
grid_geometry(fx)
wfn = load_wfn(fx, prep.FIX["mos2_6x6"]["wfn"])
assert np.max(np.abs(wfn["rk"] - fx.qfr)) < 1e-12
print(f"[6x6] nk={fx.nk} nb={fx.nb} nmu={fx.nmu} nocc={fx.nocc}")

# gates: round-trip, disk-V match, psi vs WFN
zt0 = to_sphere(fx, recon(fx, 0), 0); n0 = int(fx.ngk[0])
rt = relF(zt0[:, :n0], fx.ZG[0][:, :n0])
Vfull_true = []
for q in range(fx.nq):
    Kf, n = sphere_K(fx, q)
    v, _ = vdim_at(fx, Kf)
    Vfull_true.append(make_Vq_generic(fx.ZG[q], v, n))
Vfull_true = np.array(Vfull_true)
vq_match = [relF(Vfull_true[q], fx.Vdisk[q]) for q in range(fx.nq)]
u0 = wfn_u_on_grid(fx, wfn, 0, list(range(fx.nb)))[:, :, fx.rmu_flat]
psidev = relF(u0, fx.psi[0])
print(f"[gate] round-trip {rt:.1e} | disk-V med {np.median(vq_match):.1e} "
      f"max {np.max(vq_match):.1e} | psi-vs-WFN {psidev:.1e}")
assert rt < 1e-14 and psidev < 1e-8

# hygiene
el = fx.el[0]
gap_top = np.min(el[:, fx.nb] - el[:, fx.nb - 1])
print(f"[gate] window-top gap min_k = {gap_top:.3e} Ry "
      f"({'CLEAN' if gap_top > 1e-6 else 'SPLIT MULTIPLET'})")

kg = np.array(fx.kgrid)
is_coarse = np.all(fx.iq3 % 2 == 0, axis=1)
coarse = np.where(is_coarse)[0]
targets = np.where(~is_coarse)[0]
print(f"[6x6] coarse subgrid: {len(coarse)} q; off-grid targets: {len(targets)}")

# C (falloff verbatim) + row-Gram assert at one q
C_all = rebuild_C_falloff(fx)
Cx = prep.gram_C_from_rows(fx, 3, conj_left=CONJ_LEFT)
cchk = relF(Cx, C_all[3])
print(f"[gate] rows-Gram vs falloff C (q=3): {cchk:.2e}")
assert cchk < 1e-11

R_by_q, S_by_q = {}, {}
conds = []
for q in range(fx.nq):
    R, lam = eigh_frame(C_all[q])
    R_by_q[q] = R
    S_by_q[q] = np.sqrt(np.maximum(lam, 1e-300))
    conds.append(lam[0] / lam[-1])
print(f"[6x6] cond(C): med={np.median(conds):.2e} max={np.max(conds):.2e}")

alphas = alpha_ladder(fx)
C_ALPHAS = list(alphas.keys())
RANKS = [640, 480, 320, 160]
vfull_by_q, vsr_by_qa, vlr_by_qa = {}, {}, {}
Vcsr_by_qa = {}
for q in range(fx.nq):
    Kf, n = sphere_K(fx, q)
    vfull_by_q[q], _ = vdim_at(fx, Kf)
    for ca, a_int in alphas.items():
        vs, vl = v_split_at(fx, Kf, a_int)
        vsr_by_qa[(q, ca)] = vs
        vlr_by_qa[(q, ca)] = vl
for q in coarse:
    Phi, _ = phi_from_zeta(fx, q, R_by_q[q], S_by_q[q], PHI_CONJ)
    for ca in C_ALPHAS:
        Vcsr_by_qa[(q, ca)] = vc_tile(Phi, vsr_by_qa[(q, ca)])
# on-grid LOO also transports 6x6-neighbor tiles: build Vcsr for ALL q at ca=1
for q in range(fx.nq):
    if (q, 1.0) not in Vcsr_by_qa:
        Phi, _ = phi_from_zeta(fx, q, R_by_q[q], S_by_q[q], PHI_CONJ)
        Vcsr_by_qa[(q, 1.0)] = vc_tile(Phi, vsr_by_qa[(q, 1.0)])

Mrows_by_q = {q: gap_window_rows(fx, q) for q in range(fx.nq)}
B_true = {q: B_from_tile(Mrows_by_q[q], Vfull_true[q]) for q in range(fx.nq)}
B_sr_true = {}
for q in range(fx.nq):
    for ca in C_ALPHAS:
        B_sr_true[(q, ca)] = B_from_tile(
            Mrows_by_q[q],
            make_Vq_generic(fx.ZG[q], vsr_by_qa[(q, ca)], int(fx.ngk[q])))

def l_rows(Mr, R, S, r):
    """rows of L (norm <= 1): the response-flavor dressing.
    Physical block B = conj( l Vc_R l^H ) + B_lr."""
    return (Mr @ R[:, :r]) / S[:r][None, :]

# exact LR channel from regauged WFN fields (fixture band gauge)
print("[prep] regauging WFN fields ...")
UF, rg_res = prep.build_regauged_fields(fx, wfn)
print(f"[prep] regauge residual: med={np.median(rg_res):.2e} "
      f"max={np.max(rg_res):.2e}")
assert np.max(rg_res) < 1e-5
print("[prep] exact LR form factors ...")
occ = fx.nocc
vb = list(range(occ - 3, occ)); cb = list(range(occ, occ + 3))
u_cache = {k: UF[k][vb + cb] for k in range(fx.nk)}
B_lr_exact = {}
B_lr_fit1 = {}
for q0 in list(targets) + list(coarse):
    n = int(fx.ngk[q0])
    fi = prep.flat_idx(fx, fx.gvec[q0])[:n]
    ph = np.exp(-2j * np.pi * (fx.rfrac @ fx.qwrap[q0]))
    F = np.zeros((fx.nk * 9, n), dtype=np.complex128)
    r = 0
    for k in range(fx.nk):
        uc = u_cache[fx.kmq_idx[q0, k]][3:]
        uv = u_cache[k][:3]
        rho = np.einsum("csm,vsm->cvm", np.conj(uc), uv).reshape(9, fx.n_rtot)
        # stored zeta~ is the UNNORMALIZED FFT — F matches (no 1/N_r).
        box = np.fft.fftn((rho * ph[None, :]).reshape(9, *fx.fg),
                          axes=(1, 2, 3), norm="backward")
        F[r:r + 9] = box.reshape(9, fx.n_rtot)[:, fi]
        r += 9
    for ca in C_ALPHAS:
        B_lr_exact[(q0, ca)] = np.conj(F * vlr_by_qa[(q0, ca)][None, :n]) @ F.T
    B_lr_fit1[q0] = B_from_tile(
        Mrows_by_q[q0], make_Vq_generic(fx.ZG[q0], vlr_by_qa[(q0, 1.0)], n))
lr_cons = np.median([relF(B_lr_fit1[q], B_lr_exact[(q, 1.0)]) for q in targets])
print(f"[prep] LR fit-vs-exact consistency (c_a=1) med: {lr_cons:.2e}")

# ---------------- transport machinery (jit-friendly loops) ----------------
# overlaps from the regauged fields (fixture gauge; Parseval-exact); host
# flatten cache to bound device memory at 6x6.
_flat_cache = {}
def _flat(k):
    if k not in _flat_cache:
        _flat_cache[k] = UF[k].reshape(fx.nb, -1)
    return _flat_cache[k]

_O_cache = {}
def overlap(ka, kb):
    key = (ka, kb)
    if key not in _O_cache:
        _O_cache[key] = np.asarray(
            jnp.asarray(np.conj(_flat(ka))) @ jnp.asarray(_flat(kb)).T)
    return _O_cache[key]

@jax.jit
def _edge_k(psiA, psiB, psiP, t):
    """one k contribution to H: X_q^H [rotate(t) X_qp].
    psiA = psi[kmq(q,k)], psiB = psi[k], psiP = psi[kmq(qp,k)]."""
    if CONJ_LEFT:
        Xq = jnp.einsum("asm,bsm->abm", jnp.conj(psiA), psiB)
        Xp = jnp.einsum("asm,bsm->abm", jnp.conj(psiP), psiB)
    else:
        Xq = jnp.einsum("asm,bsm->abm", psiA, jnp.conj(psiB))
        Xp = jnp.einsum("asm,bsm->abm", psiP, jnp.conj(psiB))
    tt = jnp.conj(t) if T_CONJ else t
    Xr = jnp.einsum("mM,Mnv->mnv", tt, Xp)
    return jnp.einsum("abm,abn->mn", jnp.conj(Xq), Xr)

H_cache = {}
def edge_H(q, qp):
    key = (q, qp)
    if key not in H_cache:
        H = jnp.zeros((fx.nmu, fx.nmu), dtype=jnp.complex128)
        psi_j = jnp.asarray(fx.psi)
        for k in range(fx.nk):
            ka, kb = int(fx.kmq_idx[q, k]), int(fx.kmq_idx[qp, k])
            t, _ = polar_unitary(overlap(ka, kb))
            H = H + _edge_k(psi_j[ka], psi_j[k], psi_j[kb], jnp.asarray(t))
        H_cache[key] = np.asarray(H)
    return H_cache[key]

T_cache = {}
def edge_T(q, qp, r):
    key = (q, qp, r)
    if key not in T_cache:
        H = edge_H(q, qp)
        M = (R_by_q[q][:, :r].conj().T @ H @ R_by_q[qp][:, :r]) \
            / S_by_q[q][:r][:, None] / S_by_q[qp][:r][None, :]
        U, cos, Vh = np.linalg.svd(M)
        T_cache[key] = (U @ Vh, cos)
    return T_cache[key]

# exciton base for off-grid targets (GPU W assembly)
print("[prep] exciton H_base for 27 targets ...")
W0_j = jnp.asarray(fx.W0)
psi_j = jnp.asarray(fx.psi)

def h_base(q0):
    nk = fx.nk
    dim = nk * 9
    D = np.zeros(dim)
    for k in range(nk):
        kc = fx.kmq_idx[q0, k]
        D[k * 9:(k + 1) * 9] = (fx.enk[kc][cb][:, None]
                                - fx.enk[k][vb][None, :]).ravel()
    H = np.diag(D).astype(np.complex128)
    kgl = np.array(fx.kgrid)
    for k in range(nk):
        kc = int(fx.kmq_idx[q0, k])
        kcp = np.array([int(fx.kmq_idx[q0, kp]) for kp in range(nk)])
        d3 = (fx.iq3[k][None, :] - fx.iq3) % kgl[None, :]
        iw = (d3[:, 0] * kgl[1] + d3[:, 1]) * kgl[2] + d3[:, 2]
        Cp = jnp.einsum("csm,pdsm->pcdm", jnp.conj(psi_j[kc, cb]),
                        psi_j[kcp][:, cb])
        Vp = jnp.einsum("vsn,pwsn->pvwn", psi_j[k, vb],
                        jnp.conj(psi_j[:, vb]))
        blk = jnp.einsum("pcdm,pmn,pvwn->pcvdw", Cp, W0_j[iw], Vp) / nk
        blk = np.asarray(blk).reshape(nk, 9, 9)
        for kp in range(nk):
            H[k * 9:(k + 1) * 9, kp * 9:(kp + 1) * 9] -= blk[kp]
    return 0.5 * (H + H.conj().T)

H_base = {}
eig_true = {}
for q0 in targets:
    Hb = h_base(int(q0))
    H_base[int(q0)] = Hb
    eig_true[int(q0)] = lowest_eigs(Hb + B_true[int(q0)] / fx.nk)

def exciton_shift(q0, B_pred):
    e = lowest_eigs(H_base[q0] + B_pred / fx.nk)
    return np.abs(e - eig_true[q0]) * RY_TO_MEV

# ---------------- weights ----------------
coarse_kg = (3, 3, 1)
coarse_iq3 = (fx.iq3[coarse] // 2)
coarse_fr = fx.qfr[coarse]

def wts_multilin(q0):
    w = prep.w_multilinear(fx, fx.qfr[q0], [tuple(v) for v in coarse_iq3],
                           coarse_kg)
    return {int(coarse[j]): wt for j, wt in w.items()}

# Fourier weights on the COARSE (3x3) lattice: R = integer unit-cell vectors
# with the coarse-grid wrap (|R| <= 1 shells), phases e^{-2pi i q.R} — exactly
# the nR=7 truncated-R scheme of Sec 3.5 evaluated at the 9 coarse points.
def wts_fourier9_v2(q0, nR=7):
    Rvecs = np.array([[wrap_int(ix, 3), wrap_int(iy, 3), 0]
                      for ix in range(3) for iy in range(3)])
    adot = np.linalg.inv(fx.bdot) * (2 * np.pi) ** 2
    Rd = np.sqrt(np.abs(np.einsum("ri,ij,rj->r", Rvecs, adot, Rvecs)))
    Rset = Rvecs[np.argsort(Rd, kind="stable")][:nR]
    F = np.exp(-2j * np.pi * (coarse_fr @ Rset.T))
    f0 = np.exp(-2j * np.pi * (np.asarray(fx.qfr[q0]) @ Rset.T))
    w = f0 @ np.linalg.pinv(F)
    return {int(coarse[j]): w[j] for j in range(len(coarse))}

# ---------------- OFF-GRID runs ----------------
def run_target(q0, wts, rank, ca):
    r = rank
    Vc = np.zeros((r, r), dtype=np.complex128)
    for qi, w in wts.items():
        T, _ = edge_T(q0, qi, r)
        Vc = Vc + w * (T @ Vcsr_by_qa[(qi, ca)][:r, :r] @ T.conj().T)
    Vc = 0.5 * (Vc + Vc.conj().T)
    lM = l_rows(Mrows_by_q[q0], R_by_q[q0], S_by_q[q0], r)
    B_sr = np.conj(lM @ Vc @ lM.conj().T)
    return B_sr, B_sr + B_lr_exact[(q0, ca)]

print("\n[C1 OFF-GRID 6x6 from 3x3 subgrid] building 243 edges ...")
import time
t0 = time.time()
for j, q0 in enumerate(targets):
    for qi in coarse:
        edge_H(int(q0), int(qi))
    if j % 9 == 0:
        print(f"   target {j}/27 edges done ({time.time()-t0:.0f}s)")
print(f"   all edges in {time.time()-t0:.0f}s")

cos_off = np.array([edge_T(int(q0), int(qi), 640)[1]
                    for q0 in targets for qi in coarse])
print(f"[diagB off-grid] cosines(r=640): min={cos_off.min():.4f} "
      f"p1={np.percentile(cos_off,1):.4f} med={np.median(cos_off):.4f} "
      f"frac<0.9={np.mean(cos_off < 0.9):.3f}")
OUT["cos_off"] = cos_off

results = {}
print(f"\n[C1 OFF-GRID] PRIMARY metrics, 27 targets")
print(f"{'w':>6} {'r':>4} {'c_a':>4} | {'SR med':>9} {'SR max':>9} | "
      f"{'TOT med':>9} {'TOT max':>9} {'p90':>9} | {'exc med':>8} {'exc max':>8}")
for wname, wfun in [("mlin", wts_multilin), ("f7", wts_fourier9_v2)]:
    for rank in RANKS:
        for ca in C_ALPHAS:
            if rank != 640 and ca != 1.0:
                continue
            srs, tots, p90s, excs = [], [], [], []
            for q0 in targets:
                q0 = int(q0)
                B_sr, B_tot = run_target(q0, wfun(q0), rank, ca)
                srs.append(relF(B_sr, B_sr_true[(q0, ca)]))
                rB, rmed, rp90 = B_metrics(B_tot, B_true[q0])
                tots.append(rB); p90s.append(rp90)
                excs.append(np.max(exciton_shift(q0, B_tot)))
            results[(wname, rank, ca)] = dict(sr=srs, tot=tots, p90=p90s,
                                              exc=excs)
            print(f"{wname:>6} {rank:>4} {ca:>4} | {np.median(srs):>9.3e} "
                  f"{np.max(srs):>9.2e} | {np.median(tots):>9.3e} "
                  f"{np.max(tots):>9.2e} {np.median(p90s):>9.2e} | "
                  f"{np.median(excs):>8.3f} {np.max(excs):>8.3f}")
OUT["offgrid"] = {str(k): v for k, v in results.items()}

# baseline ladder off-grid (slim): masterzeta, interpCZ rankcut 1e-2, zetadirect
print("\n[baseline OFF-GRID 6x6] slim ladder, primary metrics")
zeta_r_c = {int(q): recon(fx, int(q)) for q in coarse}
Z_r_c = {q: C_all[q] @ zeta_r_c[q] for q in zeta_r_c}
def solve_zeta(Cm, Zm, mode, lam):
    Ch = 0.5 * (Cm + Cm.conj().T)
    U, s, Vh = np.linalg.svd(Ch, hermitian=True)
    if mode == "raw":
        sinv = 1.0 / s
    else:
        sinv = np.where(s > lam * s[0], 1.0 / np.where(s > lam * s[0], s, 1), 0)
    return (Vh.conj().T * sinv) @ (U.conj().T @ Zm)
bl = {m: dict(B=[], exc=[]) for m in ["masterzeta", "cz_rc2", "cz_raw", "zdir"]}
for q0 in targets:
    q0 = int(q0)
    wts = wts_fourier9_v2(q0)
    n = int(fx.ngk[q0])
    for meth in bl:
        if meth == "masterzeta":
            zt = to_sphere(fx, zeta_r_c[int(coarse[0])], q0)
        elif meth == "zdir":
            zr = sum(w * zeta_r_c[qi] for qi, w in wts.items())
            zt = to_sphere(fx, zr, q0)
        else:
            C0 = sum(w * C_all[qi] for qi, w in wts.items())
            Z0 = sum(w * Z_r_c[qi] for qi, w in wts.items())
            zt = to_sphere(fx, solve_zeta(C0, Z0,
                                          "raw" if meth == "cz_raw" else "rankcut",
                                          0.0 if meth == "cz_raw" else 1e-2), q0)
        Bp = B_from_tile(Mrows_by_q[q0], make_Vq_generic(zt, vfull_by_q[q0], n))
        bl[meth]["B"].append(relF(Bp, B_true[q0]))
        bl[meth]["exc"].append(np.max(exciton_shift(q0, Bp)))
for meth, d in bl.items():
    print(f"   {meth:>10}: B relF med={np.median(d['B']):.3e} "
          f"max={np.max(d['B']):.3e} | exc med={np.median(d['exc']):.3f} "
          f"max={np.max(d['exc']):.3f} meV")
OUT["baseline_off"] = bl

# ---------------- ON-GRID 6x6 LOO (nn4, ca=1, rank ladder) ----------------
print("\n[C1 ON-GRID 6x6 LOO] nn4, c_a=1")
loo = {r: dict(sr=[], tot=[]) for r in RANKS}
for q0 in range(fx.nq):
    wts = prep.w_nn_avg(fx, q0)
    for qi in wts:
        edge_H(q0, qi)
    for r in RANKS:
        B_sr, B_tot = run_target(q0, wts, r, 1.0)
        loo[r]["sr"].append(relF(B_sr, B_sr_true[(q0, 1.0)]))
        loo[r]["tot"].append(B_metrics(B_tot, B_true[q0])[0])
for r in RANKS:
    print(f"   r={r}: SR med={np.median(loo[r]['sr']):.3e} "
          f"max={np.max(loo[r]['sr']):.2e} | TOT med={np.median(loo[r]['tot']):.3e} "
          f"max={np.max(loo[r]['tot']):.2e}")
OUT["ongrid_loo"] = {str(r): v for r, v in loo.items()}

cos_on = np.array([edge_T(q0, qi, 640)[1] for q0 in range(fx.nq)
                   for qi in prep.w_nn_avg(fx, q0)])
print(f"[diagB on-grid-6x6] cosines: min={cos_on.min():.4f} "
      f"med={np.median(cos_on):.4f} frac<0.9={np.mean(cos_on < 0.9):.3f}")
OUT["cos_on"] = cos_on

import json
with open(f"{prep.STUDY}/proto0_b_results.json", "w") as f:
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(x) for x in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o
    json.dump(clean(OUT), f)
print("\n[proto0_b] DONE")
