import sys
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_A/src")
import numpy as np
from file_io import read_bgw_vcoul, fill_v_grid_for_q, WFNReader
from common import symmetry_maps

wfn = WFNReader("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/qe/nscf/WFN.h5")
sym = symmetry_maps.SymMaps(wfn)
tbl = read_bgw_vcoul("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/01_bgw_eps_vcoul/vcoul")
fft_grid = tuple(int(x) for x in wfn.fft_grid)
vol = float(wfn.cell_volume)
sym_mats_k = np.asarray(sym.sym_mats_k, dtype=np.int32)

v_q0 = fill_v_grid_for_q(tbl, (0.0, 0.0, 0.0), fft_grid, vol, sym_mats_k=sym_mats_k)
print(f"q=(0,0,0)   : v[0,0,0]={v_q0[0,0,0]:.3e}  max={np.max(v_q0):.3e}  nonzero={int(np.sum(v_q0!=0))}")

v_qz = fill_v_grid_for_q(tbl, (0.0, 0.0, 0.25), fft_grid, vol, sym_mats_k=sym_mats_k)
print(f"q=(0,0,.25) : v[0,0,0]={v_qz[0,0,0]:.3e}  max={np.max(v_qz):.3e}  nonzero={int(np.sum(v_qz!=0))}")

bvec = np.array(wfn.bvec) * float(wfn.blat)
print("\nOverlay-vs-native at q=(0,0,0), LORRAX units (Ry * 1/Omega):")
for G in [(-1,-1,-1), (-1,0,0), (1,1,1), (0,0,1), (-2,0,0), (2,1,0)]:
    Gc = np.array(G) @ bvec
    g2 = float(np.dot(Gc, Gc))
    native = 8*np.pi/g2/vol
    bgw_val = v_q0[G[0]%fft_grid[0], G[1]%fft_grid[1], G[2]%fft_grid[2]]
    print(f"  G={G}: native={native:.5e}  overlay={bgw_val:.5e}  ratio={bgw_val/native:.5f}")
