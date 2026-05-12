"""End-to-end Sternheimer + derivatives driver with timing.

Emits three things:
  (1) χ_{00}(q, 0) at every reduced-BZ q   (primal static response)
  (2) ∂χ_{00}/∂q_i  at every q              (first directional derivatives)
  (3) S_ij = ∂²χ_{00}/∂q_i∂q_j|_0             (S-tensor at q=0)

Structured timing (via pf.region bars + stderr output):

    setup                         — one-off infrastructure
    chi_primal:compile            — the first q's jit trace + XLA compile
    chi_primal:exec               — the remaining q's steady-state solves
    first_deriv:compile           — first jvp of chi_col_contrib traces
    first_deriv:exec              — remaining (q, direction) tangent solves
    s_tensor:kp_solves            — 3 k·p tangent solves per k for δu̇_i
    s_tensor:assemble             — overlaps + FFT + prefactor
    total                         — wall clock of the above

Profile with:

    cd <this dir>   # (has WFN.h5, Mo.upf, S.upf)
    LORRAX_NGPU=1 lxrun python3 -u \\
        /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/run_profiled.py \\
        --out profile_stern  -m ...
    # or just run directly with  PF=1 python3 -u run_full.py  for stderr timings.
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling')

# Opt-in profiling via `pf`.  When PF_OUT is set, we wire up the launcher
# infrastructure; otherwise pf.region is a no-op context.
try:
    import pf
    _HAS_PF = True
    if 'PF_OUT' in os.environ:
        pf.setup_env(os.environ['PF_OUT'])
        pf.attach_compile_log(os.path.join(os.environ['PF_OUT'], 'compile.log'))
except Exception:
    _HAS_PF = False
    class _pf_stub:
        class region:
            def __init__(self, *a, **kw): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass
        @staticmethod
        def trace_profile(*a, **kw):
            class C:
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return C()
        @staticmethod
        def start_memory_sampler(*a, **kw): pass
        @staticmethod
        def stop_memory_sampler(*a, **kw): pass
        @staticmethod
        def write_memory_timeline(*a, **kw): pass
        @staticmethod
        def snapshot_memory(*a, **kw): pass
    pf = _pf_stub()

from runtime import set_default_env
set_default_env()
# Enable persistent compile cache with min_compile_time_secs=0 so even small
# (sub-1s) compiles persist across runs.  Without this, lxrun's
# JAX_COMPILATION_CACHE_DIR is set but JAX's default 1s threshold means most
# of our compiles never land on disk.
from common.jax_compile_cache import ensure_jax_compile_cache
ensure_jax_compile_cache()

import numpy as np
import jax
import jax.numpy as jnp

from common import Meta, symmetry_maps
from common.load_wfns import load_kpoint_fftbox
from file_io import WFNReader
from psp.dft_operators import setup_H_k_from_kvec
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import build_dft_potentials, build_rho_val_from_wfn
from psp.run_sternheimer import (
    _psi_box_to_G_sphere,
    accumulate_chi_density, project_density_to_Gsphere,
    build_sternheimer_source, build_sternheimer_source_preQ,
    build_sternheimer_op_at_kvec_traced,
    chi_col_contrib_at_kvec_traced,
    compute_kp_tangent_at_kvec,
    compute_s_tensor_contrib_at_q0,
    make_density_perturbation, make_umklapp_phase,
)
from solvers.projectors import make_Q_kminq
from solvers.sternheimer_precond import compute_per_band_kinetic, tpa_preconditioner_diag
from solvers.sternheimer_solve import SternheimerOp, sternheimer_solve


def log(name, dt):
    print(f"  [time] {name:<30s}  {dt:8.2f} s", flush=True)


def now(): return time.perf_counter()


# ═══════════════════════════════════════════════════════════════════════
#  SETUP
# ═══════════════════════════════════════════════════════════════════════
from psp.dft_operators import compute_ngkmax

t_setup0 = now()
with pf.region("setup"):
    s0 = now()
    wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
    n_occ = int(wfn.nelec)
    nspinor = int(wfn.nspinor)
    meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ,
                            n_rmu=0, bispinor=False)
    pseudos = load_pseudopotentials('.')
    log("  setup.wfn+meta+pseudos", now() - s0)

    s0 = now()
    rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
    log("  setup.rho_val", now() - s0)

    s0 = now()
    V_scf, V_loc, vnl_setup = build_dft_potentials(
        wfn, pseudos, rho_val, truncation_2d=True, verbose=False)
    log("  setup.V_scf+vnl_setup", now() - s0)

    nk_full = int(sym.nk_tot)
    N_grid = int(np.prod(wfn.fft_grid))
    V_cell = float(wfn.cell_volume)
    bdot = jnp.asarray(wfn.bdot, dtype=jnp.float64)
    spin_factor = 2 if nspinor == 1 else 1
    prefactor_chi = (2.0 * spin_factor * np.sqrt(N_grid)) / (V_cell * nk_full)
    prefactor_st = (2.0 * spin_factor) / (V_cell * nk_full)    # no √N for overlap-based S

    # Unfolded ψ buffer
    s0 = now()
    psi_box_full = jnp.stack(
        [load_kpoint_fftbox(wfn, sym, meta, ik, n_occ) for ik in range(nk_full)],
        axis=0)
    psi_box_full.block_until_ready()
    log("  setup.psi_box_full", now() - s0)

    irk_to_k = np.asarray(sym.irk_to_k_map)
    en_full = jnp.asarray(wfn.energies[0, irk_to_k, :n_occ], dtype=jnp.float64)
    alpha_pv_j = jnp.asarray(
        2.0 * (np.asarray(en_full).max() - np.asarray(en_full).min()),
        dtype=jnp.float64)

    # ── Pad all H_k to common ngkmax — kills shape-polymorphism retraces
    #    (1947 vs 1963 G's on MoS2 3×3 → one compile across all k's).
    s0 = now()
    kpts_all = np.asarray(sym.unfolded_kpts, dtype=np.float64)
    ngkmax = int(compute_ngkmax(kpts_all, np.asarray(wfn.bdot),
                                 float(wfn.ecutwfc), tuple(wfn.fft_grid)))

    H_cache = []
    for ik in range(nk_full):
        kv = np.asarray(sym.unfolded_kpts[ik], dtype=np.float64)
        H_k = setup_H_k_from_kvec(kv, V_scf, vnl_setup, wfn, meta,
                                   V_loc_r=V_loc, ngkmax=ngkmax)
        Gk_int = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
        H_cache.append((H_k, Gk_int, kv))
    H_cache[-1][0].T_diag.block_until_ready()
    log(f"  setup.H_cache (ngkmax={ngkmax}, 9× setup_H_k_from_kvec)", now() - s0)
log(f"setup TOTAL", now() - t_setup0)


# ═══════════════════════════════════════════════════════════════════════
#  (1) χ(q)  primal sweep — all 9 reduced-BZ q's
# ═══════════════════════════════════════════════════════════════════════

# ─── Pre-stack ALL per-kminq data ONCE (independent of q) ──────────────
# Per-kminq fields:  T_diag, mask, Gx/Gy/Gz, vnl_Z (skip — built per-(k,q) via
# Sternheimer op), U_val on kminq sphere, kvec_kminq.
# For each q, we look up kminq_idx = kq_map[:, iq] and gather these arrays.
T_diag_kmq_full    = jnp.stack([H_cache[ik][0].T_diag for ik in range(nk_full)], axis=0)  # (nk, nG)
mask_kmq_full      = jnp.stack([H_cache[ik][0].mask   for ik in range(nk_full)], axis=0)  # (nk, nG)
Gx_kmq_full        = jnp.stack([H_cache[ik][0].Gx     for ik in range(nk_full)], axis=0)  # (nk, nG)
Gy_kmq_full        = jnp.stack([H_cache[ik][0].Gy     for ik in range(nk_full)], axis=0)
Gz_kmq_full        = jnp.stack([H_cache[ik][0].Gz     for ik in range(nk_full)], axis=0)
Gk_int_kmq_full    = jnp.stack([H_cache[ik][1]        for ik in range(nk_full)], axis=0)  # (nk, nG, 3)
kvec_kmq_full      = jnp.asarray(np.asarray(sym.unfolded_kpts), dtype=jnp.float64)        # (nk, 3)
U_kmq_box_full     = psi_box_full[:, :n_occ]                                              # (nk, nv, ns, nx, ny, nz)
# pre-gather U_kmq_G on each k's own G-sphere (per-k different), then mask:
U_kmq_G_full = jnp.stack([
    _psi_box_to_G_sphere(psi_box_full[ik, :n_occ], H_cache[ik][1]) *
    H_cache[ik][0].mask[None, None, :].astype(psi_box_full.dtype)
    for ik in range(nk_full)
], axis=0)                                                                                # (nk, nv, ns, nG)
# K_bar_sq per source k (depends on H_k.T_diag and U_k_G).  Source k = the
# WFN's ik index, NOT shifted.  This is independent of q.
K_bar_sq_full = jnp.stack([
    compute_per_band_kinetic(
        _psi_box_to_G_sphere(psi_box_full[ik, :n_occ], H_cache[ik][1]) *
        H_cache[ik][0].mask[None, None, :].astype(psi_box_full.dtype),
        H_cache[ik][0].T_diag,
    )
    for ik in range(nk_full)
], axis=0)                                                                                 # (nk, nv)
en_occ_full        = en_full[:, :n_occ]                                                    # (nk, nv)
V_scf_shared_chi   = H_cache[0][0].V_scf
vnl_E_shared_chi   = H_cache[0][0].vnl_E

# Pre-compute kq_map as a jax array for in-jit gathers.
kq_map_full = jnp.asarray(np.asarray(sym.kq_map, dtype=np.int32))                          # (nk, n_iq_red)


@jax.jit
def _build_stk_at_q(qvec, kminq_idx, V_pert_base):
    """JIT-compiled per-q stack builder.  Inputs:
        qvec        : (3,) float64        (signed q)
        kminq_idx   : (nk,) int           (kq_map[:, iq] for the source k's)
        V_pert_base : (nx, ny, nz) complex (= 1 for density response)

    Compiles ONCE (signature is static across all q's) and is reused for every
    q-call.  Internally vmaps over the source-k index to build phases,
    masked U-gathers, Vu_G, b, precond_diag.
    """
    # Gather per-kminq stacks via fancy indexing.
    T_diag_kmq    = T_diag_kmq_full[kminq_idx]          # (nk, nG)
    mask_kmq      = mask_kmq_full[kminq_idx]
    Gx_kmq        = Gx_kmq_full[kminq_idx]
    Gy_kmq        = Gy_kmq_full[kminq_idx]
    Gz_kmq        = Gz_kmq_full[kminq_idx]
    Gk_int_kmq    = Gk_int_kmq_full[kminq_idx]          # (nk, nG, 3)
    U_kmq_G       = U_kmq_G_full[kminq_idx]             # (nk, nv, ns, nG)
    kvec_kmq      = kvec_kmq_full[kminq_idx]            # (nk, 3)

    # Per-source-k umklapp G_wrap, phases.  Vectorise over k.
    kvec_k_arr  = kvec_kmq_full                          # (nk, 3) — source k
    G_wrap_int  = jnp.round((kvec_k_arr - qvec[None, :]) - kvec_kmq).astype(jnp.int32)  # (nk, 3)

    nx, ny, nz = wfn.fft_grid
    fx = jnp.arange(nx, dtype=jnp.float64) / nx
    fy = jnp.arange(ny, dtype=jnp.float64) / ny
    fz = jnp.arange(nz, dtype=jnp.float64) / nz
    def _phase_for_k(G_wrap, sign):
        arg = (2.0 * jnp.pi * sign) * (
            G_wrap[0].astype(jnp.float64) * fx[:, None, None]
            + G_wrap[1].astype(jnp.float64) * fy[None, :, None]
            + G_wrap[2].astype(jnp.float64) * fz[None, None, :])
        return jnp.exp(1j * arg).astype(jnp.complex128)

    phase_wrap_stack   = jax.vmap(_phase_for_k, in_axes=(0, None))(G_wrap_int, +1)  # (nk, nx, ny, nz)
    phase_unwrap_stack = jax.vmap(_phase_for_k, in_axes=(0, None))(G_wrap_int, -1)
    V_pert_real_stack  = V_pert_base[None, :, :, :] * phase_wrap_stack              # (nk, nx, ny, nz)

    U_k_box_stack = psi_box_full[:, :n_occ]                                          # (nk, nv, ns, nx, ny, nz)

    def _vu_for_k(U_k_box_one, Gk_int_kmq_one, V_pert_real_one, mask_kmq_one):
        Vu = build_sternheimer_source_preQ(U_k_box_one, Gk_int_kmq_one, V_pert_real_one)
        return Vu * mask_kmq_one[None, None, :].astype(Vu.dtype)
    Vu_G_stack = jax.vmap(_vu_for_k)(U_k_box_stack, Gk_int_kmq, V_pert_real_stack, mask_kmq)

    # b = Q_{kmq}(Vu).  Q_kmq depends on per-k U_kmq_G.
    def _q_apply(U_kmq_G_one, Vu_one):
        coefs = jnp.einsum('msG,vsG->vm', jnp.conj(U_kmq_G_one), Vu_one, optimize=True)
        return Vu_one - jnp.einsum('vm,msG->vsG', coefs, U_kmq_G_one, optimize=True)
    b_stack = jax.vmap(_q_apply)(U_kmq_G, Vu_G_stack)

    # precond per source k = TPA on H_kmq.T_diag with K_bar_sq from source k.
    precond_stack = jax.vmap(tpa_preconditioner_diag)(T_diag_kmq, K_bar_sq_full)

    return dict(
        kvec_p_stack       = kvec_kmq,
        Gkmq_int_stack     = Gk_int_kmq,
        mask_stack         = mask_kmq,
        Gx_stack           = Gx_kmq, Gy_stack=Gy_kmq, Gz_stack=Gz_kmq,
        vnl_E_super        = vnl_E_shared_chi,
        V_scf              = V_scf_shared_chi,
        U_kmq_G_stack      = U_kmq_G,
        eps_vk_stack       = en_occ_full,                    # (nk, nv)
        precond_diag_stack = precond_stack,
        U_k_box_stack      = U_k_box_stack,
        b_stack            = b_stack,
        phase_unwrap_stack = phase_unwrap_stack,
    )


def _per_k_chi(kvec_p, Gkm_int, mask, Gx, Gy, Gz,
                V_scf, vnl_E_super, U_kmq_G, eps_v, precond_diag,
                U_k_box, b, phase_unwrap,
                Gprime_int, prefactor_j, alpha_pv_j_, bdot_,
                tol=1e-8, max_iter=100):
    """Pure per-k chi-column contribution, hoisted to module scope so its
    jit cache is shared across q-values (no closure capture of qvec).
    """
    return chi_col_contrib_at_kvec_traced(
        kvec_p_traced=kvec_p, Gkminq_int_np=Gkm_int, vnl_setup=vnl_setup,
        V_scf=V_scf, mask=mask, Gx=Gx, Gy=Gy, Gz=Gz,
        fft_grid=wfn.fft_grid, bdot=bdot_, vnl_E_super=vnl_E_super,
        U_val_kminq_G=U_kmq_G, eps_v=eps_v,
        alpha_pv_sc=alpha_pv_j_, precond_diag=precond_diag,
        U_val_k_box=U_k_box, b=b,
        Gprime_int=Gprime_int, phase_unwrap=phase_unwrap,
        prefactor=prefactor_j,
        tol=tol, max_iter=max_iter,
    )


# vmap over the 14 per-k leading-axis args; the rest (V_scf, vnl_E_super,
# Gprime_int, prefactor, alpha_pv, bdot, tol, max_iter) are shared across k.
_chi_vmap_over_k = jax.vmap(
    _per_k_chi,
    in_axes=(0, 0, 0, 0, 0, 0,      # kvec_p, Gkm_int, mask, Gx, Gy, Gz
             None, None,             # V_scf, vnl_E_super
             0, 0, 0,                # U_kmq_G, eps_v, precond_diag
             0, 0, 0,                # U_k_box, b, phase_unwrap
             None, None, None, None,  # Gprime_int, prefactor, alpha_pv, bdot
             None, None),             # tol, max_iter
)


@jax.jit
def _chi_sum_over_k(stk, dq, Gprime_int, prefactor_j, alpha_pv, bdot_):
    """Apply k-vmap and sum.  JIT'd at module scope: one trace shared
    across all q-values (only ``dq`` differs between calls; the per-k
    stacks have constant shape thanks to ``ngkmax`` padding)."""
    kvec_p_stack = stk['kvec_p_stack'] - dq[None, :]
    # Tolerance choice: 1e-6 is plenty for χ_{G'=0} head/wing — the
    # diagnostic STERN_DEBUG=1 shows that without Schur warm-start, CG on
    # this MoS2 3×3 case takes ~100 iters to reach 1e-8 residual but
    # converges to 1e-6 in roughly half that.  At 1e-6 the chi values
    # match SoS extrapolated to the same precision the SoS itself
    # achieves (~0.5%).  ``while_loop`` now actually terminates early
    # for ~half the bands.
    chi_per_k = _chi_vmap_over_k(
        kvec_p_stack, stk['Gkmq_int_stack'], stk['mask_stack'],
        stk['Gx_stack'], stk['Gy_stack'], stk['Gz_stack'],
        stk['V_scf'], stk['vnl_E_super'],
        stk['U_kmq_G_stack'], stk['eps_vk_stack'], stk['precond_diag_stack'],
        stk['U_k_box_stack'], stk['b_stack'], stk['phase_unwrap_stack'],
        Gprime_int, prefactor_j, alpha_pv, bdot_,
        1e-6, 80,
    )
    return jnp.sum(chi_per_k, axis=0)


@jax.jit
def _chi_value_and_jacfwd(stk, q_eval, q_base, Gprime_int, prefactor_j, alpha_pv, bdot_):
    """Returns (chi(q), ∂chi/∂q_i) at q_eval — both from a single shared primal
    trace via ``jax.linearize``.  The primal CG solve happens once; the 3
    tangent solves are batched via vmap over the cartesian unit vectors and
    share the same A operator.  Single XLA compile reused across all q's."""
    f = lambda q: _chi_sum_over_k(stk, q - q_base, Gprime_int, prefactor_j,
                                    alpha_pv, bdot_)
    primal, jvp_fn = jax.linearize(f, q_eval)
    eye = jnp.eye(3, dtype=jnp.float64)
    jac_per_dir = jax.vmap(jvp_fn)(eye)              # (3, ng_out)
    return primal, jnp.moveaxis(jac_per_dir, 0, -1)  # primal (ng_out,), jac (ng_out, 3)


def chi_col_at_q(iq_red, qvec=None, stk=None, q_traced=None):
    """Public driver — takes per-q pre-stacked ``stk``."""
    if stk is None:
        qvec, per_k = build_per_k_at_q(iq_red)
        stk = stack_per_k_data(per_k)
    q_base = jnp.asarray(qvec, dtype=jnp.float64)
    if q_traced is None:
        q_traced = q_base
    Gprime_int = jnp.zeros((1, 3), dtype=jnp.int32)
    prefactor_j = jnp.asarray(prefactor_chi, dtype=jnp.float64)
    return _chi_sum_over_k(stk, q_traced - q_base, Gprime_int, prefactor_j,
                            alpha_pv_j, bdot)


def chi_value_and_jacfwd_at_q(iq_red, qvec, stk):
    """Returns (chi(q), Jacobian ∂χ/∂q (1,3)) — both computed in one
    shared-primal trace.  Skip the separate chi_col_at_q + jax.jacfwd
    duplicate primal solve."""
    q_base = jnp.asarray(qvec, dtype=jnp.float64)
    Gprime_int = jnp.zeros((1, 3), dtype=jnp.int32)
    prefactor_j = jnp.asarray(prefactor_chi, dtype=jnp.float64)
    return _chi_value_and_jacfwd(stk, q_base, q_base, Gprime_int, prefactor_j,
                                   alpha_pv_j, bdot)


iq_list = list(range(9))
print(f"\n── (1) χ(q) primal sweep over {len(iq_list)} reduced q's ──")
chi_all = {}

# V_pert_base is q-independent (= 1 for density response).  Build once.
_V_pert_base = make_density_perturbation(wfn.fft_grid)


def build_stk_at_q(iq):
    qvec_pos = np.asarray(wfn.kpoints[iq], dtype=np.float64)
    qvec = qvec_pos - np.round(qvec_pos)
    kminq_idx = np.asarray(sym.kq_map[:, iq], dtype=np.int32)
    stk = _build_stk_at_q(jnp.asarray(qvec, dtype=jnp.float64),
                            jnp.asarray(kminq_idx),
                            _V_pert_base)
    return qvec, stk


# ── Combined phase: χ(q) primal + ∂χ/∂q_i  in ONE shared-primal trace ──
print(f"\n── (1+2) χ(q) + ∂χ/∂q via jax.linearize (shared primal trace) ──")

dchi_all = {}
with pf.region("chi+deriv:compile"):
    t0 = now()
    qvec0, stk0 = build_stk_at_q(iq_list[0])
    chi0, dchi0 = chi_value_and_jacfwd_at_q(iq_list[0], qvec0, stk0)
    chi0.block_until_ready(); dchi0.block_until_ready()
    chi_all[iq_list[0]] = complex(chi0[0])
    for i, name in enumerate(('qx', 'qy', 'qz')):
        dchi_all[(iq_list[0], name)] = complex(dchi0[0, i])
log("chi+deriv:compile (iq=0)", now() - t0)

with pf.region("chi+deriv:exec"):
    t0 = now()
    for iq in iq_list[1:]:
        qvec, stk = build_stk_at_q(iq)
        chi, dchi = chi_value_and_jacfwd_at_q(iq, qvec, stk)
        chi.block_until_ready(); dchi.block_until_ready()
        chi_all[iq] = complex(chi[0])
        for i, name in enumerate(('qx', 'qy', 'qz')):
            dchi_all[(iq, name)] = complex(dchi[0, i])
log(f"chi+deriv:exec ({len(iq_list)-1} q's, primal + 3-dir Jacobian fused)",
    now() - t0)

print(f"  χ(q) values:")
for iq in iq_list:
    qv = np.asarray(wfn.kpoints[iq]) - np.round(np.asarray(wfn.kpoints[iq]))
    print(f"    iq={iq}  q=({qv[0]:+.3f},{qv[1]:+.3f},{qv[2]:+.3f})  "
          f"χ = {chi_all[iq].real:+.4e} + {chi_all[iq].imag:+.2e}j")

print(f"  Selected ∂χ/∂q_i  (iq, dir → value):")
for iq in [0, 1, 4]:
    row = [f"iq={iq}:"]
    for name in ('qx', 'qy', 'qz'):
        v = dchi_all[(iq, name)]
        row.append(f"  {name}={v.real:+.3e}+{v.imag:+.2e}j")
    print("   " + " ".join(row))


# ═══════════════════════════════════════════════════════════════════════
#  (3) S-tensor at q=0 via the explicit P_val-rotation formula
# ═══════════════════════════════════════════════════════════════════════

print(f"\n── (3) S-tensor at q=0 — explicit q=0 solve over all k ──")

# Precompute per-k S-tensor inputs (all stacked along leading k-axis).
kvec_k_stack = jnp.stack([jnp.asarray(H_cache[ik][2]) for ik in range(nk_full)], axis=0)
Gk_int_stack_S = jnp.stack([H_cache[ik][1] for ik in range(nk_full)], axis=0)
mask_stack_S = jnp.stack([H_cache[ik][0].mask for ik in range(nk_full)], axis=0)
Gx_stack_S = jnp.stack([H_cache[ik][0].Gx for ik in range(nk_full)], axis=0)
Gy_stack_S = jnp.stack([H_cache[ik][0].Gy for ik in range(nk_full)], axis=0)
Gz_stack_S = jnp.stack([H_cache[ik][0].Gz for ik in range(nk_full)], axis=0)
V_scf_shared = H_cache[0][0].V_scf
vnl_E_shared = H_cache[0][0].vnl_E

U_k_G_stack_S = []
precond_stack_S = []
eps_v_stack_S = []
for ik in range(nk_full):
    H_k, Gk_int, _ = H_cache[ik]
    U_k_box = psi_box_full[ik, :n_occ]
    mask_k = H_k.mask[None, None, :].astype(psi_box_full.dtype)
    U_k_G = _psi_box_to_G_sphere(U_k_box, Gk_int) * mask_k
    eps_vk = en_full[ik, :n_occ]
    K_bar_sq = compute_per_band_kinetic(U_k_G, H_k.T_diag)
    pd = tpa_preconditioner_diag(H_k.T_diag, K_bar_sq)
    U_k_G_stack_S.append(U_k_G); precond_stack_S.append(pd); eps_v_stack_S.append(eps_vk)
U_k_G_stack_S = jnp.stack(U_k_G_stack_S, axis=0)
precond_stack_S = jnp.stack(precond_stack_S, axis=0)
eps_v_stack_S = jnp.stack(eps_v_stack_S, axis=0)


def _per_k_full_S(kvec_k, Gk_int, mask, Gx, Gy, Gz,
                   U_val_G, eps_vk, precond_diag,
                   V_scf, vnl_E_super):
    """End-to-end per-k S-tensor: 3 first-tangent solves (k·p) → 3 first-tangent
    solves with grad as RHS → (3,3) overlap.  All inside one jitted function so
    the kp + assemble compiles fuse into a single XLA module."""
    grad_kp = compute_kp_tangent_at_kvec(
        kvec_p_base_np=kvec_k, Gkminq_int_np=Gk_int, vnl_setup=vnl_setup,
        V_scf=V_scf, mask=mask, Gx=Gx, Gy=Gy, Gz=Gz,
        fft_grid=wfn.fft_grid, bdot=bdot, vnl_E_super=vnl_E_super,
        U_val_G=U_val_G, eps_v_at_kvec_p=eps_vk,
        alpha_pv_sc=alpha_pv_j, precond_diag=precond_diag,
        tol=1e-7, max_iter=200,
    )
    return compute_s_tensor_contrib_at_q0(
        kvec_p_base_np=kvec_k, Gkminq_int_np=Gk_int, vnl_setup=vnl_setup,
        V_scf=V_scf, mask=mask, Gx=Gx, Gy=Gy, Gz=Gz,
        fft_grid=wfn.fft_grid, bdot=bdot, vnl_E_super=vnl_E_super,
        U_val_G=U_val_G, U_val_grad_kp=grad_kp,
        eps_v=eps_vk,
        alpha_pv_sc=alpha_pv_j, precond_diag=precond_diag,
        tol=1e-7, max_iter=200,
    )


_s_full_vmap_over_k = jax.jit(jax.vmap(_per_k_full_S,
    in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, None, None)))


with pf.region("s_tensor:full"):
    t0 = now()
    S_per_k = _s_full_vmap_over_k(
        kvec_k_stack, Gk_int_stack_S, mask_stack_S,
        Gx_stack_S, Gy_stack_S, Gz_stack_S,
        U_k_G_stack_S, eps_v_stack_S, precond_stack_S,
        V_scf_shared, vnl_E_shared,
    )
    S_per_k.block_until_ready()
    S_total = prefactor_st * np.asarray(jnp.sum(S_per_k, axis=0))
log("s_tensor:full (kp + assemble fused, vmap k)", now() - t0)

print(f"  S-tensor (crystal coords, real):")
for row in S_total.real:
    print(f"    {row[0]:+.4e}  {row[1]:+.4e}  {row[2]:+.4e}")
print(f"  imag max: {float(np.max(np.abs(S_total.imag))):.2e}")

log("TOTAL", now() - t_setup0)

if _HAS_PF and 'PF_OUT' in os.environ:
    pf.write_memory_timeline(os.path.join(os.environ['PF_OUT'], 'memory_timeline.txt'))
    pf.snapshot_memory(os.path.join(os.environ['PF_OUT'], 'memprof', 'end.prof'))
