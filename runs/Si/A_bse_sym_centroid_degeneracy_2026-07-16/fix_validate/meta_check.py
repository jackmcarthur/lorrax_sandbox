#!/usr/bin/env python3
"""Exercise the screening-window degeneracy fix end-to-end through
Meta.from_system on the real Si WFN, for several (nval,ncond,nband) configs.
"""
import os, sys
LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
sys.path.insert(0, os.path.join(LROOT, "src"))
import jax
jax.config.update("jax_enable_x64", True)
from file_io.wfn_loader import WfnLoader
from common import symmetry_maps
from common.meta import Meta

WFN = os.path.join(LROOT, "tests/regression/si_cohsex_debug/WFN.h5")
wfn = WfnLoader(WFN, backend="eager")
sym = symmetry_maps.SymMaps(wfn)
print(f"world_size(device_count)={jax.device_count()} nelec={wfn.nelec} "
      f"nbands_file={wfn.energies.shape[-1]}\n", flush=True)

for (nval, ncond, nband, tag) in [
    (8, 52, 60, "work_sym as-is (b3=b4=60; expect CLAMP at 60, no drop)"),
    (8, 32, 60, "demo ncond=32 (b3=40; expect b4 60->40, drop 20)"),
    (8, 8, 60, "demo ncond=8 (b3=16; expect b4 60->40, drop 20)"),
    (8, 32, 40, "nband=40 already closed (b3=40; expect no change)"),
]:
    print(f"### {tag}")
    m = Meta.from_system(wfn, sym, nval=nval, ncond=ncond, nband=nband, n_rmu=0)
    print(f"    -> b_id_2(nelec)={m.b_id_2} b_id_3(sigma_top)={m.b_id_3} "
          f"b_id_4_user(screen_top)={m.b_id_4_user} b_id_4(padded)={m.b_id_4}\n", flush=True)
