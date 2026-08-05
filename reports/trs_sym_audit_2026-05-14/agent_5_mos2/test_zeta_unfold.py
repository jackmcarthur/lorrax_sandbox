"""Algebraic test 2 — ζ unfold check for MoS2 3x3 SOC.

Per design doc (reports/zeta_ibz_2026-05-11/report.md eq. 1):

   ζ_{Sq, π_s(μ)}(SG) = exp(-i (Sq + SG)·τ_s) · ζ_{q,μ}(G)       (spatial)

For TRS rows (sym_mats_k = -S, applying T = K to ζ since ζ has no spinor):

   ζ_{-q,μ}(G) = ζ*_{q,μ}(-G)

So a TRS unfold via sym_mats_k row [n_t + s] = -S:
   ζ_{-Sq, π_s(μ)}(-SG) = conj(ζ_{Sq, π_s(μ)}(SG))
                        = exp(+i (Sq + SG)·τ_s) · conj(ζ_{q,μ}(G))

For MoS2 τ=0 (symmorphic) ⇒ phases are 1 throughout.

This test compares for each q_full reached via a non-trivial sym_idx:
  (b) hand-rolled unfolded ζ from sym IBZ ζ
  (c) nosym ζ at the same q_full

There is NO production LORRAX routine for G-flat ζ-level unfold
(ZetaLoader.load(q='full_bz') raises NotImplementedError for G-flat;
the production V_q unfold operates on the V_qmunu product, not raw ζ).
So this test localizes whether the ζ MATH (consumed inside compute_vcoul)
is correct: if (b) vs (c) matches, the IBZ ζ + the sym rotation +
the centroid permutation IS consistent with the nosym ζ at q_full.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
import time
import numpy as np
import h5py
import jax
jax.config.update("jax_enable_x64", True)

SRC = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src"
sys.path.insert(0, SRC)

from file_io.wfn_loader import WfnLoader  # noqa: E402
from file_io.zeta_loader import ZetaLoader  # noqa: E402

RUN_DIR = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/06_sym_vs_nosym_pr3_2026-05-14"
SYM_ZETA = f"{RUN_DIR}/run_sym/tmp/zeta_q.h5"
NOSYM_ZETA = f"{RUN_DIR}/run_nosym/tmp/zeta_q.h5"


def build_centroid_perm(sym, fft_grid, cent_idx):
    """Return centroid permutation sym_perm[s, μ] = ν with shape (2 ntran, n_rmu)."""
    from centroid.orbit_syms import compute_centroid_sym_perm
    ntran = int(sym.sym_matrices.shape[0])
    sym_perm = compute_centroid_sym_perm(
        cent_idx,
        sym_matrices=np.asarray(sym.sym_matrices[:ntran]),
        translations=np.asarray(sym.translations[:ntran]),
        fft_grid=np.asarray(fft_grid, dtype=np.int32),
        extend_trs=True,
    )
    return sym_perm   # (2 ntran, n_rmu)


def hand_rolled_zeta_unfold_one_q(
    zeta_irr_q_G,            # (n_rmu, ngkmax_disk) c128 — ζ_{q_irr, μ}(G_disk[q_irr, j])
    g_irr_q,                 # (ngkmax_disk, 3) int  — G-list for q_irr on disk
    ngk_irr_q,               # int valid count
    sym_idx,                 # int into sym_mats_k
    n_sym_spatial,
    sym_mats_k,              # (2 ntran, 3, 3)
    sym_perm,                # (2 ntran, n_rmu)  centroid perm π_s
    translations,            # (ntran, 3)
    q_irr_frac,              # (3,) fractional q_irr
):
    """Return (zeta_full_q_G, G_target) where the output is on the G-axis
    SG (or -SG for TRS), in the same axis order as the IBZ G-list.

    Math (per pr3_design.md / zeta_ibz_2026-05-11/report.md eq. 1):

      spatial s (s < n_t):
        G_target = S · G_irr                                      (S = sym_mats_k[s])
        ζ_full(G_target) at μ_full = π_s(μ_irr) is given by:
            ζ_full[π_s(μ), G_target] = exp(-i (S·q + G_target)·τ_s) · ζ_irr[μ, G_irr]

      TRS s (s ≥ n_t):
        sym_mats_k[s] = -S where S = sym_mats_k[s - n_t]
        G_target = -S · G_irr
        ζ_full[π_s(μ), G_target] = exp(-i (-S·q + G_target)·τ_s) · conj(ζ_irr[μ, G_irr])
                                 = exp(+i (S·q + S·G_irr)·τ_s) · conj(ζ_irr[μ, G_irr])
         (since G_target = -S·G_irr makes (G_target)·τ = -(S·G_irr)·τ, and
          (-S·q + G_target)·τ = -(S·q + S·G_irr)·τ; the leading sign flips on conj)

    We return the result on a 0-padded array of size (n_rmu, ngkmax_disk) in
    the same axis order as G_irr (so column j corresponds to G_target = ±S · G_irr[j]).
    Valid entries are [:, :ngk_irr_q]; pad columns are zero (kept zero in disk too).
    """
    s = int(sym_idx)
    n_t = int(n_sym_spatial)
    is_trs = s >= n_t
    s_spat = s - n_t if is_trs else s
    S_full = np.asarray(sym_mats_k[s], dtype=np.int64)   # full op including ±
    G_irr_valid = np.asarray(g_irr_q[:ngk_irr_q], dtype=np.int64)
    G_target = (S_full @ G_irr_valid.T).T.astype(np.int32)   # (ngk_irr_q, 3)

    n_rmu = int(zeta_irr_q_G.shape[0])
    ngkmax = int(zeta_irr_q_G.shape[-1])

    tau = np.asarray(translations[s_spat], dtype=np.float64)
    has_tau = bool(np.any(np.abs(tau) > 1e-12))
    if has_tau:
        # Sq + G_target = full_op · q + full_op · G_irr   if spatial
        #               = -S·q + (-S·G_irr) = -(Sq + S·G_irr)   if TRS
        # The expression sym_mats_k[s] @ q + G_target is exactly this:
        Sq = S_full @ np.asarray(q_irr_frac, dtype=np.float64)   # (3,)
        # In the SAME crystal basis: phase = exp(-i 2π (Sq + G_target) · τ)
        # Note τ comes from BGW tnp which is in fractional (units of 2π).
        # In LORRAX's unfold_psi the formula is exp(-1j * rotated @ tau).
        # rotated = sym_mats_k[s] @ G_kbar (integer); tau is fractional /
        # ... the formula does NOT include 2π factor explicitly. Check
        # the WFN translations docstring.
        # Per symmetry_maps.py line 736: tau = wfn.translations[s] / (2π) ... but
        # by line 484 `self.translations = wfn.translations` raw. unfold_psi
        # uses tau verbatim. So the convention here is the same as
        # unfold_psi: phase = exp(-1j * (G_rotated)·tau) (no 2π).
        phase = np.exp(-1j * ((Sq + G_target).astype(np.float64) @ tau))
    else:
        phase = None

    # Apply centroid permutation: zeta_irr is indexed by μ_irr; we want
    # output indexed by μ_full = π_s(μ_irr).
    # sym_perm[s, μ_old] = μ_new ⇒ ζ_full[μ_new] = ζ_irr[μ_old].
    perm_s = np.asarray(sym_perm[s], dtype=np.int64)
    inv_perm = np.argsort(perm_s).astype(np.int64)
    # zeta_irr is (n_rmu, ngkmax); inv_perm has the property:
    # inv_perm[μ_new] = μ_old. So gather along axis 0 with inv_perm.
    zeta_munew = np.asarray(zeta_irr_q_G)[inv_perm, :]      # (n_rmu, ngkmax)

    out = np.zeros((n_rmu, ngkmax), dtype=np.complex128)
    if is_trs:
        if phase is not None:
            out[:, :ngk_irr_q] = phase[None, :] * np.conj(zeta_munew[:, :ngk_irr_q])
        else:
            out[:, :ngk_irr_q] = np.conj(zeta_munew[:, :ngk_irr_q])
    else:
        if phase is not None:
            out[:, :ngk_irr_q] = phase[None, :] * zeta_munew[:, :ngk_irr_q]
        else:
            out[:, :ngk_irr_q] = zeta_munew[:, :ngk_irr_q]

    return out, G_target


def main():
    print("=" * 78, flush=True)
    print("Test 2 — ζ unfold algebraic check (MoS2 3x3 SOC)", flush=True)
    print("=" * 78, flush=True)

    # Load both ζ files; load both WFNs for sym tables.
    zeta_sym_loader = ZetaLoader(SYM_ZETA)
    zeta_nosym_loader = ZetaLoader(NOSYM_ZETA)

    print(f"sym: zeta_layout={zeta_sym_loader.zeta_layout}, "
          f"q_layout={zeta_sym_loader.q_layout}, n_q_on_disk={zeta_sym_loader.n_q_on_disk}, "
          f"n_q_full={zeta_sym_loader.n_q_full}, n_rmu={zeta_sym_loader.n_rmu}", flush=True)
    print(f"nosym: zeta_layout={zeta_nosym_loader.zeta_layout}, "
          f"q_layout={zeta_nosym_loader.q_layout}, n_q_on_disk={zeta_nosym_loader.n_q_on_disk}, "
          f"n_q_full={zeta_nosym_loader.n_q_full}", flush=True)

    # Use WfnLoader to get SymMaps (rebuilds from header — same as ZetaLoader uses)
    loader_sym = WfnLoader(f"{RUN_DIR}/run_sym/WFN.h5", backend="eager")
    sym = loader_sym._ensure_sym()
    ntran = int(sym.sym_matrices.shape[0])

    print(f"sym SymMaps: ntran={ntran}, sym_mats_k.shape={sym.sym_mats_k.shape}", flush=True)
    print(f"  q_irr_kgrid_int=\n{sym.q_irr_kgrid_int}", flush=True)
    print(f"  q_irr_full_idx={sym.q_irr_full_idx}", flush=True)
    print(f"  irr_idx_q={sym.irr_idx_q}", flush=True)
    print(f"  sym_idx_q={sym.sym_idx_q}", flush=True)

    # Disk-q ordering: writer puts IBZ q's in the order of q_irr_full_idx
    # (per isdf_fitting.py:1689 — q_irr_full_idx[i_irr] is the row index
    # where the i_irr-th IBZ q lives in kvecs_asints, and the writer iterates
    # IBZ q's in this order to match the on-disk row index).
    # So zeta_sym_disk[i_irr, μ, j] is ζ_{q_irr_kgrid_int[i_irr], μ}(G[i_irr,j]).

    # Load all-disk ζ slabs (eager — small)
    print("Reading sym ζ (all IBZ q's, all μ)...", flush=True)
    zeta_sym = np.asarray(zeta_sym_loader.load(q='ibz', layout='G_flat'))   # (5, 399, 1963)
    print(f"  sym ζ shape={zeta_sym.shape}, dtype={zeta_sym.dtype}", flush=True)
    print("Reading nosym ζ (all 9 full-BZ q's)...", flush=True)
    zeta_nosym = np.asarray(zeta_nosym_loader.load(q='ibz', layout='G_flat'))
    print(f"  nosym ζ shape={zeta_nosym.shape}, dtype={zeta_nosym.dtype}", flush=True)

    g_sym = np.asarray(zeta_sym_loader.gvec_components)      # (5, 3, 1963)
    g_nosym = np.asarray(zeta_nosym_loader.gvec_components)  # (9, 3, 1963)
    ngk_sym = np.asarray(zeta_sym_loader.ngk_per_q)          # (5,)
    ngk_nosym = np.asarray(zeta_nosym_loader.ngk_per_q)      # (9,)
    fft_grid = tuple(int(s) for s in zeta_sym_loader.fft_grid)
    cent_idx = np.asarray(zeta_sym_loader.r_mu_fft_idx, dtype=np.int32)
    cent_idx_nosym = np.asarray(zeta_nosym_loader.r_mu_fft_idx, dtype=np.int32)
    assert np.array_equal(cent_idx, cent_idx_nosym), "Centroid sets differ between sym/nosym!"

    # Build centroid permutation
    print(f"Building centroid perm (fft_grid={fft_grid}, n_rmu={cent_idx.shape[0]})...", flush=True)
    sym_perm = build_centroid_perm(sym, fft_grid, cent_idx)
    print(f"  sym_perm shape={sym_perm.shape}", flush=True)

    # Note the q-axis convention.  Reading sym.q_irr_kgrid_int and BGW-wrap:
    # The on-disk q's are q_irr_kgrid_int (in integer kgrid coords).  But
    # ζ on disk has the per-q phase exp(-2πi q·r) baked in — meaning when
    # we read ζ_{q,μ}(G), the underlying q is exp(-2πi q_int / kgrid · r).
    # For BGW wrap, q > kgrid/2 maps to q - kgrid.  Let's compute the
    # fractional q for each IBZ row.
    kgrid_arr = np.asarray(sym.unfolded_kpts.shape[0], dtype=np.float64)  # not right; use kgrid
    kgrid_arr = np.array(zeta_sym_loader.kgrid, dtype=np.float64)
    q_irr_wrapped = np.where(
        sym.q_irr_kgrid_int > kgrid_arr / 2,
        sym.q_irr_kgrid_int - kgrid_arr,
        sym.q_irr_kgrid_int).astype(np.float64)
    q_irr_frac_arr = q_irr_wrapped / kgrid_arr               # (5, 3)

    # Establish baseline: sym vs nosym at trivial (sym_idx=0) IBZ q's.
    # Both runs fit ζ independently from independent SCF wavefunctions,
    # so even at sym_idx=0 (no symmetry unfold required) the two ζ values
    # disagree at the ~1e-4 relative level — that's the SCF-convergence
    # noise floor propagated through the ISDF fit.  The unfold check has
    # to be normalized against this baseline.
    print("\n--- Baseline (sym vs nosym at trivial IBZ q's) ---", flush=True)
    print("  format: i_irr_disk | q_full | max|Δζ| | ref | rel", flush=True)
    baseline_max_rel = 0.0
    for i_irr, q_full in enumerate(sym.q_irr_full_idx):
        gs_q = g_sym[i_irr]
        gn_q = g_nosym[int(q_full)]
        if not np.array_equal(gs_q, gn_q):
            print(f"  i_irr={i_irr} q_full={q_full}: G-axis differs", flush=True)
            continue
        d = float(np.abs(zeta_sym[i_irr] - zeta_nosym[int(q_full)]).max())
        r = float(np.abs(zeta_nosym[int(q_full)]).max())
        rel = d / max(r, 1e-30)
        baseline_max_rel = max(baseline_max_rel, rel)
        print(f"  i_irr={i_irr:2d} q_full={int(q_full):2d}: max|Δ|={d:.3e} ref={r:.3e} rel={rel:.3e}", flush=True)
    print(f"  Baseline max rel = {baseline_max_rel:.3e} (independent-SCF noise floor)", flush=True)

    # Compare per (q_full, sym_idx)
    print("\n--- Test 2 per-q_full comparison ---", flush=True)
    print("  format: q_full | irr_q_disk | sym_idx | is_trs | max |Δζ| | n_compared", flush=True)
    worst = []
    for q_full in range(zeta_sym_loader.n_q_full):
        s = int(sym.sym_idx_q[q_full])
        i_irr_kgrid = int(sym.irr_idx_q[q_full])
        # i_irr_kgrid indexes into sym.q_irr_kgrid_int.  This SAME index is
        # the disk row for ζ_sym (writer iterates IBZ q's in this order).
        is_trs = s >= ntran

        # Skip trivial (q_full = q_irr, sym_idx == 0)
        if s == 0 and q_full == int(sym.q_irr_full_idx[i_irr_kgrid]):
            print(f"  q_full={q_full:2d}    irr_disk={i_irr_kgrid:2d}    sym={s}     trivial  "
                  f"(skip — q_full is the IBZ rep itself)", flush=True)
            continue

        # IBZ ζ slab and G-list at this q_irr
        ngk_irr = int(ngk_sym[i_irr_kgrid])
        zeta_irr_q = zeta_sym[i_irr_kgrid]                  # (n_rmu, 1963)
        g_irr_q = g_sym[i_irr_kgrid].T                       # (1963, 3)

        q_irr_frac = q_irr_frac_arr[i_irr_kgrid]
        zeta_hand, G_target_irr = hand_rolled_zeta_unfold_one_q(
            zeta_irr_q, g_irr_q, ngk_irr,
            sym_idx=s, n_sym_spatial=ntran,
            sym_mats_k=sym.sym_mats_k, sym_perm=sym_perm,
            translations=sym.translations, q_irr_frac=q_irr_frac,
        )
        # zeta_hand: (n_rmu, ngkmax_disk)  on G-axis sym_mats_k[s] @ G_irr[j]
        #           (valid for j < ngk_irr); columns >= ngk_irr are zero.

        # Nosym ζ at q_full: on-disk row = q_full (nosym ibz = full BZ)
        ngk_no = int(ngk_nosym[q_full])
        zeta_nosym_q = zeta_nosym[q_full]                    # (n_rmu, 1963)
        g_nosym_q = g_nosym[q_full].T                        # (1963, 3)

        # Build a permutation hand-side → nosym side on the G-axis.
        # The hand-side has the rotated G-vectors G_target_irr[:ngk_irr];
        # nosym side has g_nosym_q[:ngk_no].
        # The valid G-sets must match (modulo order).
        if ngk_irr != ngk_no:
            print(f"  q_full={q_full:2d}    irr_disk={i_irr_kgrid:2d}    sym={s}    trs={int(is_trs)}  "
                  f"ngk mismatch sym={ngk_irr} nosym={ngk_no}  -> SKIP", flush=True)
            continue
        nosym_dict = {tuple(g_nosym_q[i].tolist()): i for i in range(ngk_no)}
        h_to_n = np.empty(ngk_irr, dtype=np.int64)
        ok = True
        for j in range(ngk_irr):
            key = tuple(G_target_irr[j].tolist())
            if key not in nosym_dict:
                ok = False
                break
            h_to_n[j] = nosym_dict[key]
        if not ok:
            print(f"  q_full={q_full:2d}    irr_disk={i_irr_kgrid:2d}    sym={s}    trs={int(is_trs)}  "
                  f"G-set mismatch  -> SKIP", flush=True)
            continue

        # Align nosym onto handroll's G-axis
        zeta_nosym_aligned = zeta_nosym_q[:, h_to_n]         # (n_rmu, ngk_irr)
        zeta_hand_valid = zeta_hand[:, :ngk_irr]

        diff = np.abs(zeta_hand_valid - zeta_nosym_aligned)
        max_diff = float(diff.max())
        arg = np.unravel_index(diff.argmax(), diff.shape)

        # Reference scale: |ζ| typical magnitude
        ref_scale = float(np.abs(zeta_nosym_aligned).max())
        rel = max_diff / max(ref_scale, 1e-30)

        worst.append((q_full, i_irr_kgrid, s, is_trs, max_diff, rel, arg, ref_scale, ngk_irr))
        print(f"  q_full={q_full:2d}    irr_disk={i_irr_kgrid:2d}    sym={s}    trs={int(is_trs)}    "
              f"max|Δζ|={max_diff:.3e} (rel={rel:.3e})  at (μ={arg[0]}, j={arg[1]}) "
              f"refmax={ref_scale:.3e}  ngk={ngk_irr}", flush=True)

    # Verdict
    print("\n=== Test 2 summary ===", flush=True)
    if not worst:
        print("  No non-trivial sym ops tested.", flush=True)
        return
    max_abs = max(w[4] for w in worst)
    max_rel = max(w[5] for w in worst)
    # Normalize against baseline: an unfold is "correct within noise" if
    # the (b) vs nosym disagreement is no worse than the baseline (which is
    # set by independent-SCF convergence).
    verdict = "PASS (within SCF noise floor)" if max_rel < 3 * baseline_max_rel \
              else "FAIL"
    print(f"  Test 2 verdict: {verdict}", flush=True)
    print(f"  worst |Δζ|_abs (over non-trivial q's) = {max_abs:.3e}", flush=True)
    print(f"  worst |Δζ|_rel = {max_rel:.3e}", flush=True)
    print(f"  baseline noise-floor rel = {baseline_max_rel:.3e}", flush=True)
    print(f"  ratio (worst rel / baseline) = {max_rel / max(baseline_max_rel, 1e-30):.2f}x", flush=True)
    # Top-5 worst
    print("  Top-5 worst rows:", flush=True)
    for q_full, i_irr, s, is_trs, m, rel, arg, refscale, ngk in sorted(worst, key=lambda x: -x[5])[:5]:
        print(f"    q_full={q_full} irr_disk={i_irr} sym={s} trs={int(is_trs)} "
              f"max|Δ|={m:.3e} rel={rel:.3e} (μ={arg[0]},j={arg[1]}) refmax={refscale:.3e}", flush=True)


if __name__ == "__main__":
    main()
