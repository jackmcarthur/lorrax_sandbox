"""Pure-numpy prototype: validate the W(omega) Lanczos-chain math.

Checks, on small random matrices modeling the screening operator:
  (1) symplectic -> symmetric reduction:
      W(z)-v  =  C^dag (zI - H)^-1 J C   [full 2N symplectic solve]
             ==  2 * Phi [z^2 I - S]^-1 Phi^dag   [N-dim symmetric form]
      H = [[A,B],[-B,-A]], A=D+V, B=V, S=D^{1/2}(D+2V)D^{1/2},
      Phi = v M D^{1/2},  Phi^dag e_nu = D^{1/2} M^dag v e_nu.
  (2) block-Lanczos on S seeded with Phi^dag[:,cols] reproduces
      2 Phi [z^2 I - S]^-1 Phi^dag[:,cols] and converges to it as the
      chain grows (machine-exact at full chain length).
"""
import numpy as np

rng = np.random.default_rng(1)
Nt, Nmu, p = 24, 10, 4          # transitions, density(mu), probe block width
# density->transition pair amplitude M (Nmu x Nt); Coulomb metric v (Nmu x Nmu, HPD)
M = rng.normal(size=(Nmu, Nt)) + 1j*rng.normal(size=(Nmu, Nt))
vh = rng.normal(size=(Nmu, Nmu)) + 1j*rng.normal(size=(Nmu, Nmu))
v = vh @ vh.conj().T + Nmu*np.eye(Nmu)          # HPD
D = np.diag(rng.uniform(0.5, 3.0, size=Nt))     # positive energies (A-B)
Vop = M.conj().T @ v @ M                          # transition-space V = M^dag v M (Hermitian)
Vop = 0.5*(Vop + Vop.conj().T)
A = D + Vop
B = Vop
H = np.block([[A, B], [-B, -A]])
J = np.block([[np.eye(Nt), np.zeros((Nt, Nt))], [np.zeros((Nt, Nt)), -np.eye(Nt)]])
Dhalf = np.diag(np.sqrt(np.clip(np.diag(D), 0, None)))
S = Dhalf @ (D + 2*Vop) @ Dhalf
S = 0.5*(S + S.conj().T)

cols = np.array([0, 3, 5, 7])                    # probe density columns (nu)
E_nu = np.eye(Nmu)[:, cols]                      # (Nmu x p)
f = M.conj().T @ v @ E_nu                         # (Nt x p) seed f_nu = M^dag v e_nu
rhs = np.concatenate([f, -f], axis=0)             # [f; -f]  (2Nt x p)
PhiT = Dhalf @ f                                  # Phi^dag e_nu = D^{1/2} f   (Nt x p)
Phi = v @ M @ Dhalf                               # (Nmu x Nt)

def W_full(z):
    x = np.linalg.solve(z*np.eye(2*Nt) - H, rhs)  # (2Nt x p)
    s = x[:Nt] + x[Nt:]
    return v @ M @ s                              # (Nmu x p)

def W_sym(z):
    y = np.linalg.solve(z*z*np.eye(Nt) - S, PhiT)  # (Nt x p)
    return 2 * Phi @ y

# ---- block Lanczos on S seeded with PhiT ----
def block_qr(Wblk):
    Q, R = np.linalg.qr(Wblk)
    # fix sign so diag(R) real-positive-ish (not required, cosmetic)
    return Q, R

def build_chain(m):
    Q0, R0 = block_qr(PhiT)
    Qs = [Q0]
    alphas, betas = [], []
    Qprev = np.zeros_like(Q0)
    beta_prev = np.zeros((p, p), dtype=complex)
    for j in range(m):
        Wb = S @ Qs[j]
        al = Qs[j].conj().T @ Wb
        Wb = Wb - Qs[j] @ al - Qprev @ beta_prev.conj().T
        # full reorth (2 passes)
        for _ in range(2):
            for Qi in Qs:
                Wb = Wb - Qi @ (Qi.conj().T @ Wb)
        Qn, be = block_qr(Wb)
        alphas.append(al); betas.append(be)
        Qprev = Qs[j]; beta_prev = be
        Qs.append(Qn)
    return Qs, alphas, betas, R0

def chain_T(alphas, betas):
    m = len(alphas)
    T = np.zeros((m*p, m*p), dtype=complex)
    for j in range(m):
        T[j*p:(j+1)*p, j*p:(j+1)*p] = alphas[j]
        if j+1 < m:
            T[(j+1)*p:(j+2)*p, j*p:(j+1)*p] = betas[j]
            T[j*p:(j+1)*p, (j+1)*p:(j+2)*p] = betas[j].conj().T
    return T

def W_chain(Qs, alphas, betas, R0, z):
    m = len(alphas)
    T = chain_T(alphas, betas)
    E = np.zeros((m*p, p), dtype=complex)
    E[:p] = R0
    C = np.linalg.solve(z*z*np.eye(m*p) - T, E)     # (mp x p)
    Vmat = np.concatenate(Qs[:m], axis=1)           # (Nt x mp)
    xz = Vmat @ C                                   # (Nt x p)
    return 2 * Phi @ xz

# ---- (1) full vs symmetric ----
print("== (1) symplectic full  vs  symmetric-reduced ==")
for z in [0.0+0j, 2.0j, 1.3+0.1j, 5.0+0.3j]:
    a, b = W_full(z), W_sym(z)
    rel = np.linalg.norm(a-b)/np.linalg.norm(a)
    print(f"  z={z}:  rel|W_full - W_sym| = {rel:.3e}")

# ---- (2) block Lanczos convergence to symmetric form ----
print("== (2) block-Lanczos chain vs symmetric-reduced (oracle) ==")
zs = {"z=0": 0.0+0j, "z=2i": 2.0j, "z=1.3+0.1i": 1.3+0.1j}
mmax = Nt // p  # full chain reconstructs exactly
for m in [2, 4, 6, mmax]:
    Qs, alphas, betas, R0 = build_chain(m)
    line = f"  m={m:2d} (dim {m*p:3d}): "
    for lbl, z in zs.items():
        wc = W_chain(Qs, alphas, betas, R0, z)
        wo = W_sym(z)
        rel = np.linalg.norm(wc-wo)/np.linalg.norm(wo)
        line += f"{lbl} {rel:.2e}   "
    print(line)
print("done")
