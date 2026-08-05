"""Deep-dive on −I unfold for ζ.  Test multiple possibilities:
  - With/without complex conjugation
  - Per (μ, G) localization of the discrepancy
  - Compare ratios: is it a global phase / sign issue?
"""
import os, sys, json
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')
import numpy as np
import h5py
sys.path.insert(0, '/global/u2/j/jackm/software/lorrax_B/src')
from common.symmetry_maps import SymMaps
from file_io.wfn_loader import WfnLoader
from centroid.orbit_syms import compute_centroid_sym_perm

SYM_WFN = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/WFN.h5'
NOSYM_WFN = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_nosym/WFN.h5'
SYM_ZETA = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/tmp/zeta_q.h5'
NOSYM_ZETA = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_nosym/tmp/zeta_q.h5'

w_sym = WfnLoader(SYM_WFN, backend='eager')
w_nos = WfnLoader(NOSYM_WFN, backend='eager')
sym = SymMaps(w_sym._sym_wfn_stub())
sym_n = SymMaps(w_nos._sym_wfn_stub())
ntran = int(w_sym.ntran)
kg = np.asarray(w_sym.kgrid)
ki_sym_q = sym.kvecs_asints
ki_nos_q = sym_n.kvecs_asints
hash_sq = ki_sym_q[:, 0]*kg[1]*kg[2] + ki_sym_q[:, 1]*kg[2] + ki_sym_q[:, 2]
hash_nq = ki_nos_q[:, 0]*kg[1]*kg[2] + ki_nos_q[:, 1]*kg[2] + ki_nos_q[:, 2]
nos_pos_q = {int(h): i for i, h in enumerate(hash_nq.tolist())}
symq2nosq = np.array([nos_pos_q[int(h)] for h in hash_sq.tolist()], dtype=np.int64)

f_sym = h5py.File(SYM_ZETA,'r')
f_nos = h5py.File(NOSYM_ZETA,'r')
z_sym = f_sym['zeta_q_G'][:]
z_nos = f_nos['zeta_q_G'][:]
gvc_sym = f_sym['isdf_header/gvec_components'][()]
ngk_sym = f_sym['isdf_header/ngk'][()]
gvc_nos = f_nos['isdf_header/gvec_components'][()]
ngk_nos = f_nos['isdf_header/ngk'][()]
r_mu_idx = f_sym['isdf_header/centroids/r_mu_fft_idx'][()]
fft_grid = w_sym.fft_grid
mu_perm = compute_centroid_sym_perm(
    r_mu_idx, sym.sym_matrices[:ntran], sym.translations[:ntran], fft_grid)

# Pick qf=4: sym=3, i_irr=2.  q_full and q_irr are (0, 2, 0) and (0, 4, 0).
# Under -I: -q_irr = -(0,4,0) = (0,-4,0) ≡ (0,2,0) mod kgrid=(6,6,1). YES match.
qf, i_irr, s_idx = 4, 2, 3
qn = int(symq2nosq[qf])
print(f"qf={qf}, i_irr={i_irr}, sym={s_idx}, qn={qn}")
print(f"q_irr_kgrid = {sym.q_irr_kgrid_int[i_irr]}, q_full = {sym.kvecs_asints[qf]}, q_nos = {sym_n.kvecs_asints[qn]}")

# Build G_full → G_ibz lookup (i_irr's G-list).
ngk_n = int(ngk_nos[qn])
ngk_i = int(ngk_sym[i_irr])
gvc_n = gvc_nos[qn, :, :ngk_n].T   # (ngk_n, 3)
gvc_i = gvc_sym[i_irr, :, :ngk_i].T  # (ngk_i, 3)
print(f"ngk_n={ngk_n}, ngk_i={ngk_i}")

# Under -I: G_ibz = -G_full
g_pre = -gvc_n
lk = {tuple(int(x) for x in row): i for i, row in enumerate(gvc_i)}
nos_to_ibz = np.array([lk.get(tuple(int(x) for x in g), -1) for g in g_pre])
matched = nos_to_ibz >= 0
print(f"matched G under -I: {int(matched.sum())}/{ngk_n}")

# For each G_n in the nosym list, what is the predicted ζ_pred = ζ_ibz[μ_for_S_perm, -G]?
# Apply forward mu_perm: pred[mu_perm[s, μ], G_n] = ζ_ibz[μ, -G_n] for all μ.
n_rmu = 300
pred = np.zeros((n_rmu, ngk_n), dtype=np.complex128)
for gi in range(ngk_n):
    j = nos_to_ibz[gi]
    if j < 0: continue
    pred[mu_perm[s_idx], gi] = z_sym[i_irr, :, j]
actual = z_nos[qn, :, :ngk_n]

# Where is the disagreement?  Plot ratio for matched entries.
# Take a random sample of (μ, G) cells.
rng = np.random.default_rng(123)
for ii in range(8):
    mu = int(rng.integers(0, n_rmu))
    gi = int(rng.integers(0, int(matched.sum())))
    gi = int(np.where(matched)[0][gi])
    p = pred[mu, gi]; a = actual[mu, gi]
    print(f"  μ={mu:3d}, G_n={gvc_n[gi].tolist()}, G_ibz={(-gvc_n[gi]).tolist()}: "
          f"pred={p:+.4e}  actual={a:+.4e}  |Δ|={abs(p-a):.3e}  "
          f"ratio={p/a if abs(a)>1e-20 else '—'}")

# Try: pred = conj(actual)?
conj_diff = np.max(np.abs(np.conj(pred[:, matched]) - actual[:, matched]))
print(f"\n|conj(pred) − actual|_inf = {conj_diff:.3e}")

# Try: pred at -G uses ζ_ibz at +G (i.e., no G-rotation needed for -I)?
g_pre2 = gvc_n.copy()  # same G, not flipped
lk2 = lk
nos_to_ibz2 = np.array([lk2.get(tuple(int(x) for x in g), -1) for g in g_pre2])
matched2 = nos_to_ibz2 >= 0
print(f"\nWITHOUT -G rotation: matched={int(matched2.sum())}/{ngk_n}")
pred2 = np.zeros((n_rmu, ngk_n), dtype=np.complex128)
for gi in range(ngk_n):
    j = nos_to_ibz2[gi]
    if j < 0: continue
    pred2[mu_perm[s_idx], gi] = z_sym[i_irr, :, j]
delta2 = np.max(np.abs(pred2[:, matched2] - actual[:, matched2]))
print(f"  max|Δ| WITHOUT -G rotation = {delta2:.3e}")

# Quick: is the issue that q_irr and q_full have OVERLAPPING G-lists but with
# different sign conventions on individual G's at the wrap boundary?
# Check: for the FIRST matched G, both nosym and ibz should have the SAME G in
# the rotated frame. Let's verify the rotated G is on the IBZ q's sphere.
gi = int(np.where(matched)[0][0])
print(f"\nFirst matched G: G_full={gvc_n[gi].tolist()}, G_ibz_predicted={(-gvc_n[gi]).tolist()}, j_ibz={nos_to_ibz[gi]}")
print(f"  G_ibz[j]={gvc_i[nos_to_ibz[gi]].tolist()}")

# COULD the discrepancy be the FFT-box phase exp(-2πi q·r)?
# When ζ is stored at q with this phase baked in (writer convention), the
# unfold may also need a phase for the rotated q.  But for -I, q_irr=(0,2,0)/6
# and q_full=-q_irr mod = (0,4,0)/6 (in BGW wrap, both are < 0.5? No, (0,4,0)/6 = 0.667; the BGW wrap subtracts 1 so q_full_wrapped = (0, -2, 0)/6).
# Let's compute q in the BGW convention (BGW wrap: if q > kg/2, subtract kg).
def bgw_wrap(q_int, kg):
    q = np.asarray(q_int, dtype=np.float64)
    q[q > kg/2] = q[q > kg/2] - kg[q > kg/2]
    return q

q_irr_w = bgw_wrap(sym.q_irr_kgrid_int[i_irr], kg)
q_full_w = bgw_wrap(sym.kvecs_asints[qf], kg)
q_nos_w = bgw_wrap(sym_n.kvecs_asints[qn], kg)
print(f"\nBGW-wrapped q's (kgrid units): q_irr={q_irr_w.tolist()}, q_full={q_full_w.tolist()}, q_nos={q_nos_w.tolist()}")
print(f"In fractional: q_irr={q_irr_w/kg}, q_full={q_full_w/kg}")

# Now: under S = -I, the rule should be q_full = S @ q_irr = -q_irr.
# Both q_irr_w = (0,2,0)/6 and q_full_w = (0,-2,0)/6 = -q_irr_w  →  rule satisfied.

f_sym.close(); f_nos.close()
