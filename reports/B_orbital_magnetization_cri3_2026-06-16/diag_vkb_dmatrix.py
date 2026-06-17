"""Direct trustworthy check of the SOC nonlocal D-matrix: QE's deeq_nc (from the
pw2bgw VKB file) vs LORRAX's E_super. Compares the gauge-invariant eigenvalue
spectrum of the per-atom 2*nh x 2*nh spin-projector D-matrix. If the spectra
differ, the SOC nonlocal (V_NL) is the bug; if they match, V_NL is exonerated."""
import os, sys, numpy as np
from scipy.io import FortranFile
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
RY2EV = 13.605693122994
VKB = sys.argv[1]; WFN = sys.argv[2]

# ---- parse VKB header up to deeq_nc ----
f = FortranFile(VKB, 'r')
f.read_record(np.uint8)                      # 1: stitle/sdate/stime
r2 = f.read_record(np.uint8)                 # 2: mixed header
ints = np.frombuffer(r2[:20], np.int32)
nsf, ngm_g, ntran, cell_sym, nat = [int(x) for x in ints]
ints2 = np.frombuffer(r2[28:48], np.int32)
nkstot_ns, nsp, nkb, nhm, npwx_g = [int(x) for x in ints2]
print(f"nsf={nsf} nat={nat} nsp={nsp} nkb={nkb} nhm={nhm}")
for _ in range(10):  f.read_record(np.uint8) # 3..12 skip
nh = f.read_record(np.int32)                 # 13: nh(nsp)  (note: ityp was rec12)
# wait: order is ...ityp(12), nh(13). We skipped 3..12 (10 recs: 3,4,5,6,7,8,9,10,11,12). good.
deeq = f.read_record(np.complex128)          # 14: deeq_nc(nhm,nhm,nat,4)
f.close()
deeq = deeq.reshape((nhm, nhm, nat, 4), order='F')   # jh,ih,iat,(ijs)
print(f"nh per species: {nh}")

# Build per-atom 2nh x 2nh Hermitian D-matrix, collect eigenvalues.
# QE deeq_nc(:,:,iat,ijs): ijs=1..4 -> spin (s,t): 1=(1,1),2=(1,2),3=(2,1),4=(2,2)
from file_io import WfnLoader
wfn = WfnLoader(WFN)
# ityp: need atom->species; re-read quickly (rec 12). Simpler: infer nh per atom from species sizes.
# We only need eigenvalues across all atoms; build per atom using its species nh.
# Get ityp from WFN crystal.
ityp = np.asarray(wfn.atom_types) if hasattr(wfn, 'atom_types') else None
qe_eigs = []
for iat in range(nat):
    nha = nhm  # deeq is padded to nhm; trailing rows/cols are zero for smaller species
    D = np.zeros((2*nhm, 2*nhm), complex)
    for ijs,(s,t) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        D[s*nhm:(s+1)*nhm, t*nhm:(t+1)*nhm] = deeq[:, :, iat, ijs]
    ev = np.linalg.eigvalsh((D+D.conj().T)/2)
    qe_eigs.extend([e for e in ev if abs(e) > 1e-8])
qe_eigs = np.sort(np.array(qe_eigs))

# ---- LORRAX E_super ----
from common import symmetry_maps, Meta
from psp.dft_operators import setup_H_k_from_kvec, compute_ngkmax, build_G_cart
from psp.ionic_gspace import build_ionic_and_core
from psp.pseudos import load_pseudopotentials
import psp.vnl_ops as vnl_ops
sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
vnl_setup = vnl_ops.build_vnl_setup(wfn, pseudos=pseudos, nspinor=int(wfn.nspinor),
                                    q_max=float(np.sqrt(float(wfn.ecutwfc)))*1.01)
fg = wfn.fft_grid
V_loc,_,_ = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=True)
ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts,float),np.asarray(wfn.bdot),
                            float(wfn.ecutwfc),tuple(int(x) for x in fg)))
H_k = setup_H_k_from_kvec(np.asarray(sym.unfolded_kpts[0],float), V_loc, vnl_setup, wfn, meta,
                          V_loc_r=V_loc, ngkmax=ngkmax)
E = np.asarray(H_k.vnl_E)   # (2,2,R,R)
R = E.shape[-1]
M = np.transpose(E, (0,2,1,3)).reshape(2*R, 2*R)
lx_eigs = np.linalg.eigvalsh((M+M.conj().T)/2)
lx_eigs = np.sort(lx_eigs[np.abs(lx_eigs) > 1e-8])

print(f"\nQE deeq_nc nonzero eigs: {len(qe_eigs)};  LORRAX E_super nonzero eigs: {len(lx_eigs)}")
print(f"QE  eig range [{qe_eigs.min():.4f},{qe_eigs.max():.4f}] Ry")
print(f"LX  eig range [{lx_eigs.min():.4f},{lx_eigs.max():.4f}] Ry")
# compare the sorted spectra (match counts permitting)
n = min(len(qe_eigs), len(lx_eigs))
if len(qe_eigs)==len(lx_eigs):
    d = np.abs(qe_eigs - lx_eigs)
    print(f"sorted-eig max|QE-LX| = {d.max()*RY2EV*1000:.2f} meV  mean {d.mean()*RY2EV*1000:.2f} meV")
else:
    print("EIGENVALUE COUNT MISMATCH -> D-matrix structure differs!")
    print("QE  (Ry):", np.array2string(qe_eigs, precision=4, max_line_width=120))
    print("LX  (Ry):", np.array2string(lx_eigs, precision=4, max_line_width=120))
