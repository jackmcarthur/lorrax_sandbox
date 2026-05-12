"""Why is <U_kmq, b> ~ 0.34 when U_kmq is orthonormal and b = Q_kmq(Vu)?"""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')

from runtime import set_default_env
set_default_env()
import jax, jax.numpy as jnp, numpy as np
from common import Meta, symmetry_maps
from common.load_wfns import load_kpoint_fftbox
from file_io import WFNReader
from solvers.projectors import make_Q_kminq

wfn = WFNReader('WFN.h5')
sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)

iq = 1
ik_full = 0
ik_kminq = int(sym.kq_map[ik_full, iq])
print(f'iq={iq}, ik_full={ik_full}, ik_kminq={ik_kminq}')
print(f'  k      = {sym.unfolded_kpts[ik_full]}')
print(f'  k-q    = {sym.unfolded_kpts[ik_kminq]}')

# Load boxes
psi_k = load_kpoint_fftbox(wfn, sym, meta, ik_full,  n_occ)     # (nv, 2, nx, ny, nz)
psi_p = load_kpoint_fftbox(wfn, sym, meta, ik_kminq, n_occ)
Gk = np.asarray(sym.get_gvecs_kfull(wfn, ik_full),  dtype=np.int32)
Gp = np.asarray(sym.get_gvecs_kfull(wfn, ik_kminq), dtype=np.int32)
nx, ny, nz = psi_k.shape[-3:]

def gather(box, G):
    ix = jnp.mod(G[:,0], nx); iy = jnp.mod(G[:,1], ny); iz = jnp.mod(G[:,2], nz)
    return box[..., ix, iy, iz]

# U_kmq on p-sphere
U_p = gather(psi_p, Gp)   # (nv, 2, ngk_p)
Q_kminq = make_Q_kminq(U_p)

# Round-trip FFT on psi_k, then gather at p-sphere
u_r = jnp.fft.ifftn(psi_k, axes=(-3,-2,-1), norm='ortho')
print(f'\nReal-space u_{{v,k}}: shape {u_r.shape}')
u_box_roundtrip = jnp.fft.fftn(u_r, axes=(-3,-2,-1), norm='ortho')
diff_rt = float(jnp.max(jnp.abs(u_box_roundtrip - psi_k)))
print(f'IFFT-FFT roundtrip max diff: {diff_rt:.3e}  (should be ~1e-15)')

# 1) gather roundtrip at k-sphere vs original psi_k on k-sphere
Uk_from_rt = gather(u_box_roundtrip, Gk)   # (nv, 2, ngk_k)
Uk_direct  = gather(psi_k,              Gk)
print(f'Gather round-trip vs direct on k-sphere: max diff {float(jnp.max(jnp.abs(Uk_from_rt - Uk_direct))):.3e}')

# 2) Vu_G = gather(u_box_roundtrip, Gp) — "source on p-sphere before Q"
Vu_G = gather(u_box_roundtrip, Gp)   # (nv, 2, ngk_p)
print(f'\n||Vu_G before Q|| per v (pick v=0,1,5):')
vs = [0, 1, 5]
for v in vs:
    n = float(jnp.sqrt(jnp.sum(jnp.abs(Vu_G[v])**2)))
    print(f'  v={v}: ||Vu_G||={n:.4f}')

# Projector leak BEFORE Q
leak_pre = jnp.einsum('nsG,vsG->vn', jnp.conj(U_p), Vu_G)
print(f'\n<U_p, Vu_G>  max|abs| = {float(jnp.max(jnp.abs(leak_pre))):.3e}  (this is the "raw overlap")')

# Apply Q
b = Q_kminq(Vu_G)
leak_post = jnp.einsum('nsG,vsG->vn', jnp.conj(U_p), b)
print(f'<U_p, Q(Vu_G)>  max|abs| = {float(jnp.max(jnp.abs(leak_post))):.3e}  (should be tiny)')

# Let's look at a band pair:
print(f'\n leak matrix (first 4x4 real part):\n{np.real(np.asarray(leak_pre))[:4, :4]}')
