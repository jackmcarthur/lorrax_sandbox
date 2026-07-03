import sys; sys.path.insert(0, "src")
from types import SimpleNamespace
# Stub jax.numpy-free? planner imports jax.numpy. Provide minimal.
from gw.gflat_memory_model import plan_gflat_chunks

def mesh(px, py): return SimpleNamespace(shape={'x':px,'y':py},
                                          devices=SimpleNamespace(size=px*py))

metas = {
 'MoS2_charge': dict(meta=SimpleNamespace(nk_tot=9,nspinor=2,n_rmu=642,n_rmu_padded=644,
                     n_rtot=108000,ngkmax=5545,fft_grid=(30,30,120)), nb_total=120, ngk=5545, is_b=False),
 'MoS2_bispinor': dict(meta=SimpleNamespace(nk_tot=9,nspinor=2,n_rmu=640,n_rmu_padded=640,
                     n_rtot=108000,ngkmax=5545,fft_grid=(30,30,120)), nb_total=48, ngk=5545, is_b=True),
 'CrI3_80Ry': dict(meta=SimpleNamespace(nk_tot=36,nspinor=2,n_rmu=1508,n_rmu_padded=1520,
                     n_rtot=1125000,ngkmax=59990,fft_grid=(75,75,200)), nb_total=150, ngk=59990, is_b=True),
}
for name,d in metas.items():
    for P in (4,16):
        px = {4:2,16:4}[P]; py=P//px
        for bud in (10.0,18.0,28.0):
            pl = plan_gflat_chunks(meta=d['meta'], mesh_xy=mesh(px,py),
                nb_total=d['nb_total'], ngkmax=d['ngk'], n_q_disk=d['meta'].nk_tot,
                budget_gb=bud, is_bispinor=d['is_b'], max_chunks=64)
            print(f"{name:14s} P={P:2d} bud={bud:4.0f}  cr={pl.r_chunk:7d} bc={pl.band_chunk:3d} "
                  f"q={pl.q_chunk:2d} cs={pl.gflat_chunk_size:3d} Pmin={pl.p_min:3d} "
                  f"persist={pl.persistent_bytes/1e9:5.2f} hwm={pl.hwm_bytes/1e9:6.2f} bind={pl.bottleneck}")
    print()
