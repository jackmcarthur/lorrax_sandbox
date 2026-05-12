import sys
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")
from gw.aot_memory_model.presets import points_fit_one_rchunk
pts = points_fit_one_rchunk("mos2_ortho")
print(f"n_points = {len(pts)}")
for i, (s, k, m) in enumerate(pts):
    cr = k.get("chunk_r")
    bc = k.get("band_chunk")
    print(f"[{i:2d}] kg={s.kgrid} fft={s.fft_grid} mu={s.n_rmu} b={s.n_b} s={s.n_s} | cr={cr} bc={bc} | mesh={m.p_x}x{m.p_y}")
