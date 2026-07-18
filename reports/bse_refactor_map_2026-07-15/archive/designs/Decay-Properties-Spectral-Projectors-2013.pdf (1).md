\title{
Decay Properties of Spectral Projectors with Applications to Electronic Structure*
}

\author{
Michele Benzi ${ }^{\dagger}$ \\ Paola Boito ${ }^{\ddagger}$ \\ Nader Razouk ${ }^{§}$
}

Abstract. Motivated by applications in quantum chemistry and solid state physics, we apply general results from approximation theory and matrix analysis to the study of the decay properties of spectral projectors associated with large and sparse Hermitian matrices. Our theory leads to a rigorous proof of the exponential off-diagonal decay ("nearsightedness") for the density matrix of gapped systems at zero electronic temperature in both orthogonal and nonorthogonal representations, thus providing a firm theoretical basis for the possibility of linear scaling methods in electronic structure calculations for nonmetallic systems. We further discuss the case of density matrices for metallic systems at positive electronic temperature. A few other possible applications are also discussed.

Key words. electronic structure, localization, density functional theory, density matrix, spectral gap, matrix function, orthogonal projector

AMS subject classifications. Primary, 65F60, 65F50, 65N22; Secondary, 81Q05, 81Q10
DOI. 10.1137/100814019
I. Introduction. The physical and chemical properties of materials are largely determined by the electronic structure of the atoms and molecules found within them. In all but the simplest cases, the electronic structure can only be determined approximately, and since the late 1920s a huge amount of work has been devoted to finding suitable approximations and numerical methods for solving this fundamental problem. Traditional methods for electronic structure computations are based on the solution of generalized eigenvalue problems ("diagonalization") for a sequence of large Hermitian matrices, known as one-particle Hamiltonians. The computational cost of this approach scales cubically in the size $n$ of the problem, which is in turn determined by the number of electrons in the system. For large systems, the costs become prohibitive; this is often referred to as "the $O\left(n^{3}\right)$ bottleneck" in the literature.

\footnotetext{
*Received by the editors November 8, 2010; accepted for publication (in revised form) April 27, 2012; published electronically February 7, 2013. This work was supported by National Science Foundation grants DMS-0810862 and DMS-1115692 and by a grant of the University Research Committee of Emory University.
http://www.siam.org/journals/sirev/55-1/81401.html
${ }^{\dagger}$ Department of Mathematics and Computer Science, Emory University, Atlanta, GA 30322 (benzi@mathcs.emory.edu).
${ }^{\ddagger}$ Department of Mathematics and Computer Science, Emory University, Atlanta, Georgia 30322. Current address: DMI-XLIM, UMR 7252, Université de Limoges-CNRS, 123 avenue Albert Thomas, 87060 Limoges Cedex, France (paola.boito@unilim.fr).
${ }^{§}$ Department of Mathematics and Computer Science, Emory University, Atlanta, GA 30322. Current address: Ernst \& Young GmbH Wirtschaftsprüfungsgesellschaft, Arnulfstraße 59, 80636 München, Germany (nrazouk@gmail.com).
}

In the last two decades, a number of researchers have developed approaches that are capable in many cases of achieving "optimal" computational complexity: the computational effort scales linearly in the number of electrons, leading to better performance for sufficiently large systems and making the electronic structure problem tractable for large-scale systems. These methods, often referred to as " $O(n)$ methods," apply mostly to insulators. They avoid diagonalization by computing instead the density matrix, a matrix which encodes all the important physical properties of the system. For insulators at zero temperature, this is the spectral projector onto the invariant subspace associated with the eigenvalues of the Hamiltonian falling below a certain value. For systems at positive temperatures, the density matrix can be expressed as a smooth function of the Hamiltonian.

The possibility of developing such methods rests on a deep property of electronic matter, called "nearsightedness" by W. Kohn [75]. Kohn's "Nearsightedness Principle" expresses the fact that for a large class of systems the effects of disturbances, or perturbations, remain localized and thus do not propagate beyond a certain (finite) range; in other words, far away parts of the system do not "see" each other. Mathematically, this property translates into rapid off-diagonal decay in the density matrix. This fast fall-off in the density matrix entries has been often assumed without proof, or proved only in special cases. Moreover, the precise dependence of the rate of decay on properties of the system (such as the band gap in insulators or the temperature in metallic systems) has been the subject of much discussion.

The main goal of this paper is to provide a rigorous mathematical foundation for linear scaling methods in electronic structure computations. We do this by deriving estimates, in the form of decay bounds, for the entries of general density matrices for insulators and for metallic systems at positive electronic temperatures. We also address the question of the dependence of the rate of decay on the band gap and on the temperature. Although immediately susceptible to physical interpretation, our treatment is purely mathematical. By stripping the problem down to its essential features and working at the discrete level, we are able to develop an abstract theory covering nearly all types of systems and discretizations encountered in actual electronic structure problems.

Our results are based on a general theory of decay for the entries in analytic functions of sparse matrices, initially proposed in [12, 14, 106] and further developed here. The theory is based on classical approximation theory and matrix analysis. A bit of functional analysis is used when considering a simple model of "metallic behavior," for which the decay in the density matrix is very slow.

The approach described in this paper has a number of potential applications beyond electronic structure computations, and can be applied to any problem involving functions of large matrices where "locality of interaction" plays a role. Toward the end of the paper we briefly review the possible use of decay bounds in the study of correlations in quantum statistical mechanics and information theory, in the analysis of complex networks, and in some classical problems in numerical linear algebra, like the computation of invariant subspaces of symmetric tridiagonal matrices. The discussion of these topics will be necessarily brief, but we hope it will stimulate further work in these areas.

In this paper we are mostly concerned with the theory behind $O(n)$ methods rather than with specific algorithms. Readers who are interested in the computational aspects should consult any of the many recent surveys on algorithms for electronic structure computations; among them, [20, 97, 113, 116] are especially recommended.

The remainder of the paper is organized as follows. Section 2 provides some background on electronic structure theory. The formulation of the electronic struc- ture problem in terms of spectral projectors is reviewed in section 3. A survey of previous, related work on decay estimates for density matrices is given in section 4. In section 5 we formulate our basic assumptions on the matrices (discrete Hamiltonians) considered in this paper, particularly their normalization and asymptotic behavior for increasing system size ( $n \rightarrow \infty$ ). The approximation (truncation) of matrices with decay properties is discussed in section 6. A few general properties of orthogonal projectors are established in section 7. The core of the paper is represented by section 8, where various types of decay bounds for spectral projectors are stated and proved. In section 9 we discuss the transformation to an orthonormal basis set. The case of vanishing gap is discussed in section 10. Other applications of our results and methods are mentioned in section 11. Finally, concluding remarks and some open problems are given in section 12.
2. Background on Electronic Structure Theories. In this section we briefly discuss the basic principles underlying electronic structure theory. For additional details the reader is referred to, e.g., [22, 77, 88, 89, 116, 125].

Consider a physical system formed by a number of nuclei and $n_{e}$ electrons in three-dimensional (3D) space. The time-independent Schrödinger equation for the system is the eigenvalue problem
$$
\begin{equation*}
\mathcal{H}_{\mathrm{tot}} \Psi_{\mathrm{tot}}=E_{\mathrm{tot}} \Psi_{\mathrm{tot}}, \tag{2.1}
\end{equation*}
$$
where $\mathcal{H}_{\text {tot }}$ is the many-body Hamiltonian operator, $E_{\text {tot }}$ is the total energy, and the functions $\Psi_{\text {tot }}$ are the eigenstates of the system.

The Born-Oppenheimer approximation allows us to separate the nuclear and electronic coordinates. As a consequence, we only seek to solve the quantum mechanical problem for the electrons, considering the nuclei as sources of external potential. Then the electronic part of (2.1) can be written as
$$
\begin{equation*}
\mathcal{H} \Psi=E \Psi \tag{2.2}
\end{equation*}
$$
where $E$ is the electronic energy and the eigenstates $\Psi$ are functions of $3 n_{e}$ spatial coordinates and $n_{e}$ (discrete) spin coordinates.

We denote spatial coordinates as $\mathbf{r}$ and the spin coordinate as $\sigma$; each electron is then defined by $3+1$ coordinates $\mathbf{x}_{i}=\binom{\mathbf{r}_{i}}{\sigma_{i}}$, and wavefunctions are denoted as $\Psi\left(\mathbf{x}_{1}, \ldots, \mathbf{x}_{n_{e}}\right)$. Then the electronic Hamiltonian operator in (2.2) can be written as
$$
\mathcal{H}=T+V_{\mathrm{ext}}+V_{\mathrm{ee}},
$$
where $T=-\frac{1}{2} \nabla^{2}$ is the kinetic energy, $V_{\text {ext }}$ is the external potential (i.e., the potential due to the nuclei), and $V_{\mathrm{ee}}=\frac{1}{2} \sum_{i \neq j}^{n_{e}} \frac{1}{\left|\mathbf{r}_{i}-\mathbf{r}_{j}\right|}$ is the potential due to the electronelectron repulsion. ${ }^{1}$ Moreover, the ground-state energy is given by
$$
E_{0}=\min _{\Psi}\langle\mathcal{H} \Psi, \Psi\rangle,
$$
where the minimum is taken over all the normalized antisymmetric wavefunctions (electrons being Fermions, their wavefunction is antisymmetric). The electronic den-

\footnotetext{
${ }^{1}$ As is customary in physics, we use here atomic units, that is, $e^{2}=\hbar=m=1$, with $e=$ electronic charge, $\hbar=$ reduced Planck's constant, and $m=$ electronic mass.
}
sity is defined as
$$
\rho(\mathbf{r})=n_{e} \sum_{\sigma} \int d \mathbf{x}_{2} \cdots \int d \mathbf{x}_{n_{e}}\left|\Psi\left(\mathbf{r}, \sigma, \mathbf{x}_{2}, \ldots, \mathbf{x}_{n_{e}}\right)\right|^{2}
$$

In this expression, the sum over $\sigma$ is the sum over the spin values of the first electron, while integration with respect to $\mathbf{x}_{i}$, with $2 \leq i \leq n_{e}$, denotes the integral over $\mathbb{R}^{3}$ and sums over both possible spin values for the $i$ th electron.

Observe that (2.2) is a many-particle equation that cannot be separated into several one-particle equations because of the term $V_{\text {ee }}$. Of course, being able to turn (2.2) into a separable equation would simplify the problem considerably, since the number of unknowns per equation would drop from $3 n_{e}+n_{e}$ to $3+1$. This is the motivation for one-electron methods.

For noninteracting particles, the many-body eigenstates $\Psi\left(\mathbf{x}_{1}, \ldots, \mathbf{x}_{n_{e}}\right)$ can be written as Slater determinants of occupied orbitals $\phi_{1}\left(\mathbf{x}_{1}\right), \ldots, \phi_{n_{e}}\left(\mathbf{x}_{n_{e}}\right)$,
$$
\Psi\left(\mathbf{x}_{1}, \ldots, \mathbf{x}_{n_{e}}\right)=\frac{1}{\sqrt{n_{e}!}}\left|\begin{array}{ccc}
\phi_{1}\left(\mathbf{x}_{1}\right) & \ldots & \phi_{n_{e}}\left(\mathbf{x}_{1}\right) \\
\vdots & \ddots & \vdots \\
\phi_{1}\left(\mathbf{x}_{n_{e}}\right) & \ldots & \phi_{n_{e}}\left(\mathbf{x}_{n_{e}}\right)
\end{array}\right|
$$
where each orbital satisfies a single-particle eigenstate equation $\mathcal{H}_{i} \phi_{i}=E_{i} \phi_{i}$. In general, the name "one-particle method" is used also when self-consistent terms (e.g., involving the density) are present in $\mathcal{H}_{i}$; in this case, the equations are solved iteratively, computing at each step the solution to a single-particle problem and then filling the lowest eigenstates with one electron each, to form a Slater determinant. However, some of the properties of a true noninteracting system (such as the fact that the energy is the sum of the eigenvalues of occupied states) are lost.

A fundamental example of the one-particle method is density functional theory (DFT). The main idea behind DFT consists in rewriting the ground-state energy as a density functional rather than a wavefunction functional. Indeed, the first HohenbergKohn theorem [66] states that the potential is uniquely (up to a constant) determined by the ground-state density $\rho(\mathbf{r})$. In other words, the system can be seen as characterized by the density rather than by the potential. Moreover, the ground-state density of a system with given external potential can be computed by minimizing a suitable energy functional of $\rho$ (second Hohenberg-Kohn theorem).

While of crucial theoretical importance, though, these results do not give a recipe for computing electronic structures. The next important step comes with the KohnSham construction [76]: roughly speaking, one replaces the original, nonseparable system with a fictitious system of noninteracting electrons that have exactly the same density as the original system. The single-particle equations for the Kohn-Sham system are (neglecting spin)
$$
\left(-\frac{1}{2} \nabla^{2}+V(\mathbf{r})\right) \psi_{i}(\mathbf{r})=\varepsilon_{i} \psi_{i}(\mathbf{r})
$$
where the $\psi_{i}$ 's are the Kohn-Sham orbitals and $V(\mathbf{r})$ is the single-electron potential. The associated density is
$$
\rho(\mathbf{r})=\sum_{i=1}^{n_{e}}\left|\psi_{i}(\mathbf{r})\right|^{2}
$$

The single-particle potential $V(\mathbf{r})$ can be written as
$$
V(\mathbf{r})=V_{\mathrm{ext}}(\mathbf{r})+\int_{\mathbb{R}^{3}} \frac{\rho(\mathbf{r})}{\left|\mathbf{r}-\mathbf{r}^{\prime}\right|} d \mathbf{r}^{\prime}+V_{x c}[\rho](\mathbf{r})
$$
where the term $V_{x c}[\rho](\mathbf{r})$ is called the exchange-correlation potential and depends on the density. It is important to point out that the Kohn-Sham construction is not an approximation, in that the Kohn-Sham equations are exact and yield the exact density.

On the other hand, the exchange-correlation energy is not known in practice and needs to be approximated. In the local density approximation (LDA) framework, for instance, the exchange energy is based on the energy of a uniform electron gas. Introducing spin allows for a more refined approximation (LSDA, or local spin-density approximation). One may also include gradient corrections, thus obtaining the socalled generalized gradient approximation (GGA).

The solution of the Kohn-Sham equations is usually computed via self-consistent iterations. The iterative process begins with an approximation of the density; the associated approximate exchange-correlation potential is injected into the Kohn-Sham equations. The output density is then used to form a new approximation of the potential. The process continues until the update term for the density or the potential becomes negligible. Observe that the basic building block of this computational technique is the solution of an eigenvalue problem for noninteracting particles.

Electrons at the lowest atomic-like levels ("core" electrons) do not change their state much within chemical processes. For this reason, many computational techniques do not consider them explicitly, and instead replace the Coulomb attraction of the nucleus with a potential (called pseudopotential) that includes the effect of the core electrons on the valence electrons. This approach is always employed when using plane waves as a basis for wavefunctions, since the number of plane waves required to represent core electrons is prohibitive.
3. Density Matrices. As mentioned earlier, conventional methods for electronic structure calculations require the repeated solution of linear eigenvalue problems for a one-electron Hamiltonian operator of the form $\mathcal{H}=-\frac{1}{2} \nabla^{2}+V(\mathbf{r})$. In practice, operators are discretized by grid methods or via Galerkin projection onto the finitedimensional subspace spanned by a set of basis functions $\left\{\phi_{i}\right\}_{i=1}^{n}$. When linear combinations of atom-centered Slater- or Gaussian-type functions (see below) are employed, the total number of basis functions is $n \approx n_{b} \cdot n_{e}$, where $n_{e}$ is the number of (valence) electrons in the system and $n_{b}$ is a small or moderate integer related to the number of basis functions per atom. Traditional electronic structure algorithms diagonalize the discrete Hamiltonian, resulting in algorithms with $O\left(n_{e}^{3}\right)$ (equivalently, $O\left(n^{3}\right)$ ) operation count [77, 89, 116]. In these approaches, a sequence of generalized eigenproblems of the form
$$
\begin{equation*}
H \psi_{i}=\varepsilon_{i} S \psi_{i}, \quad 1 \leq i \leq n_{e}, \tag{3.1}
\end{equation*}
$$
is solved, where $H$ and $S$ are, respectively, the discrete Hamiltonian and the overlap matrix relative to the basis set $\left\{\phi_{i}\right\}_{i=1}^{n}$. The eigenvectors $\psi_{i}$ in (3.1) are known as the occupied states and correspond to the $n_{e}$ lowest generalized eigenvalues $\varepsilon_{1} \leq \cdots \leq \varepsilon_{n_{e}}$, the occupied levels. The overlap matrix $S$ is just the Gram matrix associated with the basis set, $S_{i j}=\left\langle\phi_{j}, \phi_{i}\right\rangle$ for all $i, j$, where $\langle\cdot, \cdot\rangle$ denotes the standard $L^{2}$-inner product. In Dirac's bra-ket notation, which is the preferred one in the physics and chemistry
literature, one writes $S_{i j}=\left\langle\phi_{i} \mid \phi_{j}\right\rangle$. For an orthonormal basis set, $S=I_{n}$ (the $n \times n$ identity matrix) and the eigenvalue problem (3.1) is a standard one.

Instead of explicitly diagonalizing the discretized Hamiltonian $H$, one may reformulate the problem in terms of the density operator $P$, which is the $S$-orthogonal projector ${ }^{2}$ onto the $H$-invariant subspace corresponding to the occupied states, that is, the subspace spanned by the $n_{e}$ eigenvectors $\psi_{i}$ in (3.1). Virtually all quantities of interest in electronic structure theory can be computed as functionals of the density matrix $P$; see, e.g., $[24,95,97]$. It is this reformulation of the problem that allows for the development of potentially more efficient algorithms for electronic structure, including algorithms that asymptotically require only $O\left(n_{e}\right)$ (equivalently, $O(n)$ ) arithmetic operations and storage. Most current methodologies, including Hartree-Fock, DFT (e.g., Kohn-Sham), and hybrid schemes (like BLYP), involve self-consistent field (SCF) iterations, in which the density matrix $P$ must be computed at each SCF step, typically with increasing accuracy as the outer iteration converges; see, e.g., [77, 137].

As stated in section 1, in this paper we use some classical results from polynomial approximation theory and matrix analysis to provide a mathematical foundation for linear scaling electronic structure calculations for a very broad class of systems. We assume that the basis functions $\phi_{i}$ are localized, i.e., decay rapidly outside of a small region. Many of the most popular basis sets used in quantum chemistry, such as Gaussian-type orbitals, which are functions of the form
$$
\phi(x, y, x)=C x^{n_{x}} y^{n_{y}} z^{n_{z}} \mathrm{e}^{-\alpha r^{2}}
$$
where $C$ is a normalization constant, satisfy this requirement [77]. For systems with sufficient separation between atoms, this property implies a fast off-diagonal decay of the entries of the Hamiltonian matrix; moreover, a larger distance between atoms corresponds to a faster decay of matrix entries [77, p. 381]. If the entries that fall below a given (small) truncation tolerance are set to zero, the Hamiltonian turns out to be a sparse matrix.

Decay results are especially easy to state in the banded case, ${ }^{3}$ but more general sparsity patterns will be taken into account as well.

We can also assume from the outset that the basis functions form an orthonormal set. If this is not the case, we perform a congruence transformation to an orthogonal basis and replace the original Hamiltonian $H$ with $\tilde{H}=Z^{T} H Z$, where $S^{-1}=Z Z^{T}$ is either the Löwdin $\left(Z=S^{-1 / 2},[85]\right)$ or the inverse Cholesky $\left(Z=L^{-T}\right.$, with $S=L L^{T}$ ) factor of the overlap matrix $S$; see, e.g., [24]. Here $Z^{T}$ denotes the transpose of $Z$; for the Löwdin factorization, $Z$ is symmetric $\left(Z=Z^{T}\right)$. Up to truncation, the transformed matrix $\tilde{H}$ is still a banded (or sparse) matrix, albeit not as sparse as $H$. Hence, in our decay results we can replace $H$ with $\tilde{H}$. The entries in $S^{-1}$, and therefore those in $Z$, decay at a rate which depends on the conditioning of $S$. This, in turn, will depend on the particular basis set used, on the total number of basis functions, and on the interatomic distances, with larger separations leading to faster decay. This is discussed further in section 9. We note that the case of tight-binding Hamiltonians is covered by our theory. Indeed, the tight-binding method consists in expanding the states of the physical system (e.g., a crystal) in linear combinations of atomic orbitals of the composing atoms; such an approximation is successful if the

\footnotetext{
${ }^{2}$ That is, orthogonal with respect to the inner product associated with $S$.
${ }^{3}$ A square matrix $A=\left(A_{i j}\right)$ is said to be $m$-banded if $A_{i j}=0$ whenever $|i-j|>m$; for instance, a tridiagonal matrix is 1-banded according to this definition.
}
atomic orbitals have little overlap, which translates to a sparse Hamiltonian. The same applies to "real space" finite difference (or finite element) approximations [116].

For a given sparse discrete Hamiltonian $H$ in an orthonormal basis, we consider the problem of approximating the zero-temperature density matrix associated with $H$, that is, the spectral projector $P$ onto the occupied subspace spanned by the eigenvectors corresponding to the smallest $n_{e}$ eigenvalues of $H$ :
$$
P=\psi_{1} \otimes \psi_{1}+\cdots+\psi_{n_{e}} \otimes \psi_{n_{e}} \equiv\left|\psi_{1}\right\rangle\left\langle\psi_{1}\right|+\cdots+\left|\psi_{n_{e}}\right\rangle\left\langle\psi_{n_{e}}\right|
$$
where $H \psi_{i}=\varepsilon_{i} \psi_{i}$ for $i=1, \ldots, n_{e}$. Clearly, $P$ is Hermitian and idempotent: $P= P^{*}=P^{2}$. Consider now the Heaviside (step) function
$$
h(x)=\left\{\begin{array}{lll}
1 & \text { if } & x<\mu \\
\frac{1}{2} & \text { if } & x=\mu \\
0 & \text { if } & x>\mu
\end{array}\right.
$$
where the number $\mu$ (sometimes called the Fermi level or chemical potential [53]) is such that $\varepsilon_{n_{e}}<\mu<\varepsilon_{n_{e}+1}$. If the spectral gap $\gamma=\varepsilon_{n_{e}+1}-\varepsilon_{n_{e}}$, also known as the HOMO-LUMO gap, ${ }^{4}$ is not too small, the step function $h$ is well approximated by the Fermi-Dirac function ${ }^{5} f_{F D}(x)=1 /\left(1+\mathrm{e}^{\beta(x-\mu)}\right)$ for suitable values of $\beta>0$ :
$$
P=h(H) \approx f_{F D}(H)=\left[I_{n}+\exp \left(\beta\left(H-\mu I_{n}\right)\right)\right]^{-1}
$$

The smaller $\gamma$, the larger $\beta$ must be in order to have a good approximation: see Figure 8.8. The parameter $\beta$ can be interpreted as an (artificial) inverse temperature; the zero-temperature limit is quickly approached as $\beta \rightarrow \infty$. A major advantage of the Fermi-Dirac function is that it is analytic; hence, we can replace $h$ with $f_{F D}$ and apply to it a wealth of results from approximation theory for analytic functions.

We emphasize that the study of the zero-temperature limit-that is, the ground state of the system - is of fundamental importance in electronic structure theory. In the words of [89, Chapter 2, pp. 11-12],
> ... the lowest energy ground state of the electrons determines the structure and low-energy motions of the nuclei. The vast array of forms of matterfrom the hardest material known, diamond carbon, to the soft lubricant, graphite carbon, to the many complex crystals and molecules formed by the elements of the periodic table-are largely manifestations of the ground state of the electrons.

The Fermi-Dirac distribution is also used when dealing with systems at positive electronic temperatures ( $T>0$ ) with a small or null gap (e.g., metallic systems); in this case $\beta=\left(k_{B} T\right)^{-1}$, where $k_{B}$ is Boltzmann's constant. In particular, use of the Fermi-Dirac function allows one to compute thermodynamical properties (such as the specific heat) and the $T$-dependence of quantities from first principles. In this case,

\footnotetext{
${ }^{4}$ HOMO = highest occupied molecular orbital; LUMO = lowest unoccupied molecular orbital.
${ }^{5}$ Several other analytic approximations to the step function are known, some of which are preferable to the Fermi-Dirac function from the computational point of view; see, e.g., [80] for a comparative study. For theoretical analysis, however, we find it convenient to work with the Fermi-Dirac function.
}
of course, the matrix $P=f_{F D}(H)$ is no longer an orthogonal projector, not even approximately.

We mention in passing that it is sometimes advantageous to impose the normalization condition $\operatorname{Tr}(P)=1$ on the density matrix; indeed, such a condition is standard and part of the definition of a density matrix in the quantum mechanics literature, beginning with von Neumann [131, 133]. At zero temperature we have $\operatorname{Tr}(P)=\operatorname{rank}(P)=n_{e}$, and $P$ is replaced by $\frac{1}{n_{e}} P$. With this normalization $P$ is no longer idempotent, except when $n_{e}=1$. In this paper we do not make use of such normalization.

The localization ("pseudosparsity") of the density matrix for insulators has been long known to physicists and chemists; see the literature review in the following section. A number of authors have exploited this property to develop a host of linear scaling algorithms for electronic structure computations; see, e.g., [4, 5, 20, 23, 24, 53, $54,75,79,80,89,97,98,99,113,123,135$ ]. In this paper we derive explicitly computable decay bounds which can be used, at least in principle, to determine a priori the bandwidth or sparsity pattern of the truncation of the density matrix corresponding to a prescribed error. As we shall see, however, our decay estimates tend to be conservative and may be pessimistic in practice. Hence, we regard our results primarily as a theoretical contribution, providing a rigorous (yet elementary) mathematical justification for some important localization phenomena observed by physicists. An important aspect of our work is that our bounds are universal, in the sense that they only depend on the bandwidth (or sparsity pattern) of the discrete Hamiltonian $H$, on the smallest and largest eigenvalues of $H$, on the gap $\gamma$, and, when relevant, on the temperature $T$. In particular, our results are valid for a wide range of basis sets and indeed for different discretizations and representations of the Hamiltonian.
4. Related Work. The localization properties of spectral projectors (more generally, density matrices) associated with electronic structure computations in quantum chemistry and solid state physics have been the subject of a large number of papers. Roughly speaking, the results found in the literature fall into three broad categories:
1. Fully rigorous mathematical results for model systems (some quite general).
2. "Semirigorous" results for specific systems; these results are often characterized as "exact" or "analytical" by the authors (usually physicists), but would not be recognized as mathematically rigorous by mathematicians.
3. Nonrigorous results based on a mixture of heuristics, physical reasoning, and numerics.
Contributions in the first group are typically due to researchers working in solid state and mathematical physics. These include the pioneering works of Kohn [74] and des Cloizeaux [36], and the more recent papers by Nenciu [96], Brouder et al. [21], and a group of papers by Prodan, Kohn, and collaborators [103, 104, 105].

Before summarizing the content of these contributions, we should mention that nearly all the results found in the literature are expressed at the continuous level, that is, in terms of decay in functions rather than decay in matrices. The functions are typically functions of (real) space; results are often formulated in terms of the density kernel, but sometimes in terms of the Wannier functions. The latter form an orthonormal basis set associated with a broad class of Hamiltonians and are widely used in solid state physics. Since the Wannier functions span the occupied subspace, localization results for the Wannier functions immediately imply similar localization results for the corresponding spectral projector. Note, however, that the spectral projector may be exponentially localized even when the Wannier functions are not.

At the continuous level, the density matrix $\rho: \mathbb{R}^{d} \times \mathbb{R}^{d} \longrightarrow \mathbb{C}$ is the kernel of the density operator $\mathcal{P}$ defined by
$$
(\mathcal{P} \psi)(\mathbf{r})=\int_{\mathbb{R}^{d}} \rho\left(\mathbf{r}, \mathbf{r}^{\prime}\right) \psi\left(\mathbf{r}^{\prime}\right) d \mathbf{r}^{\prime}
$$
regarded as an integral operator on $L^{2}\left(\mathbb{R}^{d}\right)$; here $d=1,2,3$. The vectors $\mathbf{r}$ and $\mathbf{r}^{\prime}$ represent any two points in $\mathbb{R}^{d}$, and $\left|\mathbf{r}-\mathbf{r}^{\prime}\right|$ is their (Euclidean) distance. The density kernel can be expressed as
$$
\rho\left(\mathbf{r}, \mathbf{r}^{\prime}\right)=\sum_{i=1}^{n_{e}} \psi_{i}(\mathbf{r}) \psi_{i}\left(\mathbf{r}^{\prime}\right)^{*}
$$
where now $\psi_{i}$ is the (normalized) eigenfunction of the Hamiltonian operator $\mathcal{H}$ corresponding to the $i$ th lowest eigenvalue, $i=1, \ldots, n_{e}$, and the asterisk denotes complex conjugation; see, e.g., [88]. The density operator $\mathcal{P}$ admits the Dunford integral representation
$$
\begin{equation*}
\mathcal{P}=\frac{1}{2 \pi \mathrm{i}} \int_{\Gamma}(z I-\mathcal{H})^{-1} d z \tag{4.1}
\end{equation*}
$$
where $\Gamma$ is a simple closed contour in $\mathbb{C}$ surrounding the eigenvalues of $\mathcal{H}$ corresponding to the occupied states, with the remaining eigenvalues on the outside.

In [74], Kohn proved the rapid decay of the Wannier functions for one-dimensional (1D), one-particle Schrödinger operators with periodic and symmetric potentials with nonintersecting energy bands. This type of Hamiltonian describes 1D, centrosymmetric crystals. Kohn's main result takes the form
$$
\begin{equation*}
\lim _{x \rightarrow \infty} w(x) \mathrm{e}^{q x}=0 \tag{4.2}
\end{equation*}
$$
where $w(x)$ denotes a Wannier function (here $x$ is the distance from the center of symmetry) and $q$ is a suitable positive constant. In the same paper (p. 820) Kohn also points out that for free electrons (not covered by his theory, which deals only with insulators) the decay is very slow, like $x^{-1}$.

A few observations are in order: First, the decay result (4.2) is asymptotic, that is, it implies fast decay at sufficiently large distances $|x|$ only. Second, (4.2) is consistent not only with strict exponential decay, but also with decay of the form $x^{p} \mathrm{e}^{-q^{\prime} x}$, where $p$ is arbitrary (positive or negative) and $q^{\prime}>q$. Hence, the actual decay could be faster, but also slower, than exponential. Since the result in (4.2) provides only an estimate (rather than an upper bound) for the density matrix in real space, it is not easy to use in actual calculations. To be fair, such practical aspects were not discussed by Kohn until much later (see, e.g., [75]). Also, later work showed that the asymptotic regime is already achieved for distances of the order of $1-2$ lattice constants, and it helped clarify the form of the power-law prefactor, as discussed below.

The techniques used by Kohn, mostly the theory of analytic functions in one complex variable and some classical asymptotics for linear second-order differential operators with variable coefficients, did not lend themselves naturally to the treatment of higher-dimensional cases or more complicated potentials. The problem of the validity of Kohn's results in two and three dimensions has remained open for a very long time, and has been long regarded as one of the last outstanding problems of one-particle condensed-matter physics. Partial results were obtained by des Cloizeaux
[36] and much later by Nenciu [96]. Des Cloizeaux, who studied both the decay of the Wannier functions and that of the associated spectral projectors, extended Kohn's localization results to 3D insulators with a center of inversion (a specific symmetry requirement) in the special case of simple, isolated (i.e., nondegenerate) energy bands; he also treated the tight-binding limit for arbitrary crystals. Nenciu further generalized Kohn's results to arbitrary $d$-dimensional insulators, again limited to the case of simple bands.

The next breakthrough came much more recently, when Brouder et al. [21] managed to prove localization of the Wannier functions for a broad class of insulators in arbitrary dimensions. The potentials considered by these authors are sufficiently general for the results to be directly applicable to DFT, within both the LDA and the GGA frameworks. The results in [21], however, also prove that for Chern insulators (i.e., insulators for which the Chern invariants, which characterize the band structure, are nonvanishing) the Wannier functions do not decay exponentially, therefore leaving open the question of proving the decay of the density matrix in this case [129]. It should be mentioned that the mathematics in [21] is fairly sophisticated and requires some knowledge of modern differential geometry and topology.

Further papers of interest include the work by Prodan, Kohn, and collaborators [103, 104, 105]. From the mathematical standpoint, the most satisfactory results are perhaps those presented in [104]. In this paper, the authors use norm estimates for complex symmetric operators in Hilbert space to obtain sharp exponential decay estimates for the resolvents of rather general Hamiltonians with spectral gap. Using the contour integral representation formula (4.1), these estimates yield (for sufficiently large separations) exponential spatial decay bounds of the form
$$
\begin{equation*}
\left|\rho\left(\mathbf{r}, \mathbf{r}^{\prime}\right)\right| \leq C \mathrm{e}^{-\alpha\left|\mathbf{r}-\mathbf{r}^{\prime}\right|} \quad(C>0, \alpha>0, \text { const. }) \tag{4.3}
\end{equation*}
$$
for a broad class of insulators. A lower bound on the decay rate $\alpha$ (also known as the decay length or inverse correlation length) is derived, and the behavior of $\alpha$ as a function of the spectral gap $\gamma$ is examined.

Among the papers in the second group, we mention [52, 64, 70, 73, 90, 127, 128]. These papers provide quantitative decay estimates for the density matrix, based on either fairly rigorous analyses of special cases or not fully rigorous discussions of general situations. Significant use is made of approximations, asymptotics, heuristics, and physically motivated assumptions, and the results are often validated by numerical calculations. Also, it is occasionally stated that while the results were derived in the case of simplified models, the conclusions should be valid in general. Several of these authors emphasize the difficulty of obtaining rigorous results for general systems in arbitrary dimension. Despite not being fully rigorous from a mathematical point of view, these contributions are certainly very valuable and seem to have been broadly accepted by physicists and chemists. We note, however, that the results in these papers usually take the form of order-of-magnitude estimates for the density matrix $\rho\left(\mathbf{r}, \mathbf{r}^{\prime}\right)$ in real space, valid for sufficiently large separations $\left|\mathbf{r}-\mathbf{r}^{\prime}\right|$, rather than strict upper bounds. As said before of Kohn's results, this type of estimate may be difficult to use for computational purposes.

In the case of insulators, the asymptotic decay estimates in these papers take the form
$$
\begin{equation*}
\rho\left(\mathbf{r}, \mathbf{r}^{\prime}\right)=C \frac{\mathrm{e}^{-\alpha\left|\mathbf{r}-\mathbf{r}^{\prime}\right|}}{\left|\mathbf{r}-\mathbf{r}^{\prime}\right|^{\sigma}}, \quad\left|\mathbf{r}-\mathbf{r}^{\prime}\right| \rightarrow \infty \quad(\alpha>0, \sigma>0, \text { const. }), \tag{4.4}
\end{equation*}
$$
where higher-order terms have been neglected. Many of these papers concern the precise form of the power-law factor (i.e., the value of $\sigma$ ) in both insulators and metallic systems. The actual functional dependence of $\alpha$ on the gap and of $\sigma$ on the dimensionality of the problem have been the subject of intense discussion, with some authors claiming that $\alpha$ is proportional to $\gamma$, and others finding it to be proportional to $\sqrt{\gamma}$; see, e.g., $[53,70,73,90,127,128]$ and section 8.6. It appears that both types of behavior can occur in practice. For instance, in [73] the authors provide a tight-binding model of an insulator for which the density falls off exponentially with decay length $\alpha=O(\gamma)$ in the diagonal direction of the lattice, and $\alpha=O(\sqrt{\gamma})$ in nondiagonal directions, as $\gamma \rightarrow 0+$. We also note that in [73], the decay behavior of the density matrix for an insulator is found to be given (up to higher-order terms) by
$$
\rho\left(\mathbf{r}, \mathbf{r}^{\prime}\right)=C \frac{\mathrm{e}^{-\alpha\left|\mathbf{r}-\mathbf{r}^{\prime}\right|}}{\left|\mathbf{r}-\mathbf{r}^{\prime}\right|^{d / 2}}, \quad\left|\mathbf{r}-\mathbf{r}^{\prime}\right| \rightarrow \infty,
$$
where $d$ is the dimensionality of the problem. In practice, the power-law factor in the denominator is often ignored, since the exponential decay dominates.

In [52], Goedecker argued that the density matrix for $d$-dimensional ( $d=1,2,3$ ) metallic systems at electronic temperature $T>0$ behaves to leading order like
$$
\begin{equation*}
\rho\left(\mathbf{r}, \mathbf{r}^{\prime}\right)=C \frac{\cos \left(\left|\mathbf{r}-\mathbf{r}^{\prime}\right|\right)}{\left|\mathbf{r}-\mathbf{r}^{\prime}\right|^{(d+1) / 2}} \mathrm{e}^{-k_{B} T\left|\mathbf{r}-\mathbf{r}^{\prime}\right|}, \quad\left|\mathbf{r}-\mathbf{r}^{\prime}\right| \rightarrow \infty \tag{4.5}
\end{equation*}
$$

Note that in the zero-temperature limit, a power-law decay (with oscillations) is observed. An analogous result was also obtained in [70]. Note that the decay length in the exponential goes to zero like the temperature $T$ rather than like $\sqrt{T}$, as claimed, for instance, in [3]. We will return on this topic in section 8.7.

Finally, as representatives of the third group of papers we select [3] and [140]. The authors of [3] use the Fermi-Dirac approximation of the density matrix and consider its expansion in the Chebyshev basis. From an estimate of the rate of decay of the coefficients of the Chebyshev expansion of $f_{F D}(x)$, they obtain estimates for the number of terms needed to satisfy a prescribed error in the approximation of the density matrix. In turn, this yields estimates for the rate of decay as a function of the extreme eigenvalues and spectral gap of the discrete Hamiltonian. Because of some ad hoc assumptions and the many approximations used, the arguments in this paper cannot be considered mathematically rigorous, and the estimates thus obtained are not always accurate. Nevertheless, the idea of using a polynomial approximation for the Fermi-Dirac function and the observation that exponential decay of the expansion coefficients implies exponential decay in the (approximate) density matrix is quite valuable and, as we show in this paper, can be made fully rigorous.

Finally, in [140] the authors present the results of numerical calculations for various insulators in order to gain some insight into the dependence of the decay length on the gap. Their experiments confirm that the decay behavior of $\rho\left(\mathbf{r}, \mathbf{r}^{\prime}\right)$ can be strongly anisotropic and that different rates of decay may occur in different directions; this is consistent with the analytical results in [73].

Despite this considerable body of work, the localization question for density matrices cannot be regarded as completely settled from the mathematical standpoint. We are not aware of any completely general and rigorous mathematical treatment of the decay properties in density matrices associated with general (localized) Hamiltonians, covering all systems with gap as well as metallic systems at positive temperature.

Moreover, rather than order-of-magnitude estimates, actual upper bounds would be more satisfactory.

Almost all the abovementioned results concern the continuous, infinite-dimensional case. In practice, of course, calculations are performed on discrete, $n$-dimensional approximations $H$ and $P$ to the operators $\mathcal{H}$ and $\mathcal{P}$. The replacement of density operators with finite density matrices can be obtained via the introduction of a system of $n$ basis functions $\left\{\phi_{i}\right\}_{i=1}^{n}$, leading to the density matrix $P=\left(P_{i j}\right)$ with
$$
\begin{equation*}
P_{i j}=\left\langle\phi_{j}, \mathcal{P} \phi_{i}\right\rangle=\left\langle\phi_{i}\right| \mathcal{P}\left|\phi_{j}\right\rangle=\int_{\mathbb{R}^{d}} \int_{\mathbb{R}^{d}} \rho\left(\mathbf{r}, \mathbf{r}^{\prime}\right) \phi_{i}(\mathbf{r})^{*} \phi_{j}\left(\mathbf{r}^{\prime}\right) d \mathbf{r} d \mathbf{r}^{\prime} \tag{4.6}
\end{equation*}
$$

As long as the basis functions are localized in space, the decay behavior of the density function $\rho\left(\mathbf{r}, \mathbf{r}^{\prime}\right)$ for increasing spatial separation $\left|\mathbf{r}-\mathbf{r}^{\prime}\right|$ is reflected in the decay behavior of the matrix elements $P_{i j}$ away from the main diagonal (i.e., for $|i-j|$ increasing) or, more generally, for increasing distance $d(i, j)$ in the graph associated with the discrete Hamiltonian; see section 6 for details.

In developing and analyzing $O(n)$ methods for electronic structure computations, it is important to rigorously establish decay bounds for the entries of the density matrices that take into account properties of the discrete Hamiltonians. It is in principle possible to obtain decay estimates for finite-dimensional approximations using localized basis functions from the spatial decay estimates for the density kernel. Note, however, that any estimates obtained inserting (4.3) or (4.5) into (4.6) would depend on the particular set of basis functions used.

In this paper we take a different approach. Instead of starting with the continuous problem and discretizing it, we establish our estimates directly for sequences of matrices of finite, but increasing order. We believe that this approach is closer to the practice of electronic structure calculations, where matrices are the primary computational objects.

We impose a minimal set of assumptions on our matrix sequences so as to reproduce the main features of problems encountered in actual electronic structure computations, while at the same time ensuring a high degree of generality. Since our aim is to provide a rigorous and general mathematical justification to the possibility of $O(n)$ methods, this approach seems to be quite natural. ${ }^{6}$

To put our work further into perspective, we quote from two prominent researchers in the field of electronic structure, one a mathematician, the other a physicist. In his excellent survey [77], Claude Le Bris, discussing the basis for linear scaling algorithms, i.e., the assumed sparsity of the density matrix, wrote (pp. 402 and 404):

> The latter assumption is in some sense an a posteriori assumption, and not easy to analyse .... It is to be emphasized that the numerical analysis of the linear scaling methods overviewed above that would account for cut-off rules and locality assumptions, is not yet available.

It is interesting to compare these statements with two earlier ones made by Stefan Goedecker. In [51] he wrote (p. 261):

To obtain a linear scaling, the extended orbitals [i.e., the eigenfunctions of the one-particle Hamiltonian corresponding to occupied states] have to be

\footnotetext{
${ }^{6}$ We refer the historically-minded reader to the interesting discussion given by John von Neumann in [132] on the benefits that can be expected from a study of the asymptotic properties of large matrices, in contrast to the study of the infinite-dimensional (Hilbert space) case.
}

\begin{abstract}
replaced by the density matrix, whose physical behavior can be exploited to obtain a fast algorithm. This last point is essential. Mathematical and numerical analyses alone are not sufficient to construct a linear algorithm. They have to be combined with physical intuition.
\end{abstract}

A similar statement can be found in [53, p. 1086]:

> Even though $O(N)$ algorithms contain many aspects of mathematics and computer science they have, nevertheless, deep roots in physics. Linear scaling is not obtainable by purely mathematical tricks, but it is based on an understanding of the concept of locality in quantum mechanics.

In the following we provide a general treatment of the question of decay in spectral projectors that is as a priori as possible, in the sense that it relies on a minimal set of assumptions on the discrete Hamiltonians; furthermore, our theory is purely mathematical and therefore completely independent of any physical interpretation. Nevertheless, our theory allows us to shed light on questions like the dependence of the decay length on the temperature in the density matrix for metals at $T>0$; see section 8.7. We do this using for the most part fairly simple mathematical tools from classical approximation theory and linear algebra.

Of course, in the development of practical linear scaling algorithms a deep knowledge of the physics involved is extremely important; we think, however, that locality is as much a mathematical phenomenon as a physical one.

We hope that the increased level of generality attained in this paper (relative to previous treatments in the physics literature) will also help in the development of $O(n)$ methods for other types of problems where spectral projectors and related matrix functions play a central role. A few examples are discussed in section 11.
5. Normalizations and Scalings. We will be dealing with sequences of matrices $\left\{H_{n}\right\}$ of increasing size. We assume that each matrix $H_{n}$ is an Hermitian $n \times n$ matrix, where $n=n_{b} \cdot n_{e}$; here $n_{b}$ is fixed, while $n_{e}$ is increasing. As explained in section 3, the motivation for this assumption is that in most electronic structure codes, once a basis set has been selected the number $n_{b}$ of basis functions per particle is fixed, and one is interested in the scaling as $n_{e}$, the number of particles, increases. Hence, the parameter that controls the system size is $n_{e}$. We also assume that the system is contained in a $d$-dimensional box of volume $V=L^{d}$ and that $L \rightarrow \infty$ as $n_{e} \rightarrow \infty$ in such a way that the average density $n_{e} / L^{d}$ remains constant (thermodynamic limit). This is very different from the case of finite element or finite difference approximations to partial differential equations (PDEs), where the system (or domain) size is considered fixed while the number of basis functions increases or, equivalently, the mesh size $h$ goes to zero.

Our scaling assumption has very important consequences on the structural and spectral properties of the matrix sequence $\left\{H_{n}\right\}$; namely, the following properties hold:
1. The bandwidth of $H_{n}$, which reflects the interaction range of the discrete Hamiltonians, remains bounded as the system size increases [89, p. 454]. More generally, the entries of $H_{n}$ decay away from the main diagonal at a rate independent of $n_{e}$ (hence, of $n$ ). See section 6 for precise definitions and generalizations.
2. The eigenvalue spectra $\sigma\left(H_{n}\right)$ are also uniformly bounded as $n_{e} \rightarrow \infty$. In view of the previous property, this is equivalent to saying that the entries in $H_{n}$ are uniformly bounded in magnitude; this is just a consequence of Geršgorin's theorem (see, e.g., [67, p. 344]).
3. For the case of Hamiltonians modeling insulators or semiconductors, the spectral (HOMO-LUMO) gap does not vanish as $n_{e} \rightarrow \infty$. More precisely, if $\varepsilon_{i}^{(n)}$ denotes the $i$ th eigenvalue of $H_{n}$ and $\gamma_{n}:=\varepsilon_{n_{e}+1}^{(n)}-\varepsilon_{n_{e}}^{(n)}$, then $\inf _{n} \gamma_{n}>0$. This assumption does not hold for Hamiltonians modeling metallic systems; in this case, $\inf _{n} \gamma_{n}=0$, i.e., the spectral gap goes to zero as $n_{e} \rightarrow \infty$.
We emphasize that these properties hold for very general classes of physical systems and discretization methods for electronic structure, with few exceptions (i.e., nonlocalized basis functions, such as plane waves). It is instructive to contrast these properties with those of matrix sequences arising in finite element or finite difference approximations of PDEs, where the matrix size increases as $h \rightarrow 0$, with $h$ a discretization parameter. Considering the case of a scalar, second-order elliptic PDE, we see that the first property only holds in the 1D case, or in higher-dimensional cases when the discretization is refined in only one dimension. (As we will see, this condition is rather restrictive and can be relaxed.) Furthermore, it is generally impossible to satisfy the second assumption and that on the nonvanishing gap ( $\inf _{n} \gamma_{n}>0$ ) simultaneously. Indeed, normalizing the matrices so that their spectra remain uniformly bounded will generally cause the eigenvalues to completely fill the spectral interval as $n \rightarrow \infty$. That is, in general, given any two points inside this interval and for $n$ large enough, at least one eigenvalue of the corresponding $n \times n$ matrix falls between these two points.

Our assumptions allow us to refer to the spectral gap of the matrix sequence $\left\{H_{n}\right\}$ without having to specify whether we are talking about an absolute or a relative gap. As we shall see, it is convenient to assume that all the matrices in the sequence $\left\{H_{n}\right\}$ have spectrum contained in the interval $[-1,1]$; therefore, the absolute gap and the relative gap of any matrix $H_{n}$ are the same, up to the factor 2 . The spectral gap (more precisely, its reciprocal) is a natural measure of the conditioning of the problem of computing the spectral projector onto the occupied subspace, i.e., the subspace spanned by the eigenvectors of $H_{n}$ corresponding to eigenvalues $\varepsilon_{i}^{(n)}<\mu$; see, e.g., [109, p. B4] for a recent discussion. The assumption $\inf _{n} \gamma_{n}>0$ then simply means that the electronic structure problem is uniformly well-conditioned; note that this assumption is also very important for the convergence of the outer SCF iteration [77, 137]. This hypothesis is satisfied for insulators and semiconductors, but not in the case of metals.
6. Approximation of Matrices by Numerical Truncation. Discretization of $\mathcal{H}$, the Hamiltonian operator, by means of basis sets consisting of linear combinations of Slater- or Gaussian-type orbitals leads to matrix representations that are, strictly speaking, full. Indeed, since these basis functions are globally supported, almost all matrix elements $H_{i j}=\left\langle\phi_{j}, \mathcal{H} \phi_{i}\right\rangle \equiv\left\langle\phi_{i}\right| \mathcal{H}\left|\phi_{j}\right\rangle$ are nonzero. The same is true for the entries of the overlap matrix $S_{i j}=\left\langle\phi_{j}, \phi_{i}\right\rangle$. However, owing to the rapid decay of the basis functions outside of a localized region, and due to the local nature of the interactions encoded by the Hamiltonian operator, the entries of $H$ decay exponentially fast with the spatial separation of the basis functions. (For the overlap matrix corresponding to Gaussian-type orbitals, the decay is actually even faster than exponential.)

More formally, we say that a sequence of $n \times n$ matrices $A_{n}=\left(\left[A_{n}\right]_{i j}\right)$ has the exponential off-diagonal decay property if there are constants $c>0$ and $\alpha>0$ independent of $n$ such that
$$
\begin{equation*}
\left|\left[A_{n}\right]_{i j}\right| \leq c \mathrm{e}^{-\alpha|i-j|} \quad \text { for all } \quad i, j=1, \ldots, n \tag{6.1}
\end{equation*}
$$

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/1bc7e2d7-f39d-466c-801c-6b7da6d033c3-15.jpg?height=369&width=909&top_left_y=330&top_left_x=436}
\captionsetup{labelformat=empty}
\caption{Fig. 6.1 Logarithmic plot of the first row of a density matrix and an exponential bound.}
\end{figure}

Corresponding to each matrix $A_{n}$ we then define for a nonnegative integer $m$ the matrix $A_{n}^{(m)}=\left(\left[A_{n}^{(m)}\right]_{i j}\right)$ as follows:
$$
\left[A_{n}^{(m)}\right]_{i j}=\left\{\begin{array}{cc}
{\left[A_{n}\right]_{i j}} & \text { if } \quad|i-j| \leq m, \\
0 & \text { otherwise. }
\end{array}\right.
$$

Clearly, each matrix $A_{n}^{(m)}$ is $m$-banded and can be thought of as an approximation, or truncation, of $A_{n}$. Note that the set of $m$-banded matrices forms a vector subspace $\mathcal{V}_{m} \subseteq \mathbb{C}^{n \times n}$ and that $A_{n}^{(m)}$ is just the orthogonal projection of $A_{n}$ onto $\mathcal{V}_{m}$ with respect to the Frobenius inner product $\langle A, B\rangle_{F}:=\operatorname{Tr}\left(B^{*} A\right)$. Hence, $A_{n}^{(m)}$ is the best approximation of $A_{n}$ in $\mathcal{V}_{m}$ with respect to the Frobenius norm.

Note that we do not require the matrices to be Hermitian or symmetric here; we only assume (for simplicity) that the same pattern of nonzero off-diagonals is present on either side of the main diagonal. The following simple result from [14] provides an estimate of the rate at which the truncation error decreases as the bandwidth $m$ of the approximation increases. In addition, it establishes $n$-independence of the truncation error for $n \rightarrow \infty$ for matrix sequences satisfying (6.1).

Proposition 6.1 (see [14]). Let $A$ be a matrix with entries $A_{i j}$ satisfying (6.1) and let $A^{(m)}$ be the corresponding $m$-banded approximation. Then for any $\epsilon>0$ there is an $\bar{m}$ such that $\left\|A-A^{(m)}\right\|_{1} \leq \epsilon$ for $m \geq \bar{m}$.

The integer $\bar{m}$ in the foregoing proposition is easily found to be given by
$$
\bar{m}=\left\lfloor\frac{1}{\alpha} \ln \left(\frac{2 c}{1-\mathrm{e}^{-\alpha}} \epsilon^{-1}\right)\right\rfloor .
$$

Clearly, this result is of interest only for $\bar{m}<n$ (in fact, for $\bar{m} \ll n$ ).
Example 6.2. Let us consider a tridiagonal matrix $H$ of size $200 \times 200$, with eigenvalues randomly chosen in $[-1,-0.5] \cup[0.5,1]$, and let $P$ be the associated density matrix with $\mu=0$. Numerical computation shows that $P$ satisfies the bound (6.1) with $\alpha=0.6$ and $c=10$ (as long as its entries are larger than the machine precision). Figure 6.1 depicts the absolute value of the entries in the first row of $P$ and the bound (6.1), in a logarithmic scale. Choose, for instance, a tolerance $\epsilon=10^{-6}$; then it follows from the previous formula that the truncated matrix $P^{(m)}$ satisfies $\left\|P-P^{(m)}\right\|_{1} \leq \epsilon$ for any bandwidth $m \geq 29$.

What is important about this simple result is that when applied to a sequence $\left\{A_{n}\right\}=\left(\left[A_{n}\right]_{i j}\right)$ of $n \times n$ matrices having the off-diagonal decay property (6.1) with $c$ and $\alpha$ independent of $n$, the bandwidth $\bar{m}$ is itself independent of $n$. For convenience,
we have stated Proposition 6.1 in the 1 -norm; when $A=A^{*}$ the same conclusion holds for the 2 -norm, owing to the inequality
$$
\begin{equation*}
\|A\|_{2} \leq \sqrt{\|A\|_{1}\|A\|_{\infty}} \tag{6.2}
\end{equation*}
$$
(see [57, Corollary 2.3.2]). Moreover, a similar result also applies to other types of decay, such as algebraic (power-law) decay of the form
$$
\left|\left[A_{n}\right]_{i j}\right| \leq \frac{c}{|i-j|^{p}+1} \quad \text { for all } \quad i, j=1, \ldots, n
$$
with $c$ and $p$ independent of $n$, as long as $p>1$.
Remark 6.3. It is worth emphasizing that the above considerations do not require that the matrix entries $\left[A_{n}\right]_{i j}$ themselves actually decay exponentially away from the main diagonal, but only that they are bounded above in an exponentially decaying manner. In particular, the decay behavior of the matrix entries need not be monotonic.

Although we have limited ourselves to absolute approximation errors in various norms, it is easy to accommodate relative errors by normalizing the matrices. Indeed, upon normalization all the Hamiltonians satisfy $\left\|H_{n}\right\|_{2}=1$; furthermore, for density matrices this property is automatically satisfied, since they are orthogonal projectors. In the next section we also consider using the Frobenius norm for projectors.

The foregoing considerations can be extended to matrices with more general decay patterns, i.e., with exponential decay away from a subset of selected positions $(i, j)$ in the matrix; see, e.g., [14] as well as [31]. In order to formalize this notion, we first recall the definition of geodetic distance $d(i, j)$ in a graph [37]: it is the number of edges in the shortest path connecting two nodes $i$ and $j$, possibly infinite if there is no such path. Next, given a (sparse) matrix sequence $\left\{A_{n}\right\}$, we associate with each matrix $A_{n}$ a graph $G_{n}$ with $n$ nodes and $m=O(n)$ edges. In order to obtain meaningful results, however, we need to impose some restrictions on the types of sparsity allowed. Recall that the degree of node $i$ in a graph is just the number of neighbors of $i$, i.e., the number of nodes at distance 1 from $i$. We denote by $\operatorname{deg}_{n}(i)$ the degree of node $i$ in the graph $G_{n}$. We shall assume that the maximum degree of any node in $G_{n}$ remains bounded as $n \rightarrow \infty$; that is, there exists a positive integer $D$ independent of $n$ such that $\max _{1 \leq i \leq n} \operatorname{deg}_{n}(i) \leq D$ for all $n$. Note that when $A_{n}=H_{n}$ (discretized Hamiltonian), this property is a mathematical restatement of the physical notion of locality, or finite range, of interactions.

Now let us assume that we have a sequence of $n \times n$ matrices $A_{n}=\left(\left[A_{n}\right]_{i j}\right)$ with associated graphs $G_{n}$ and graph distances $d_{n}(i, j)$. We will say that $A_{n}$ has the exponential decay property relative to the graph $G_{n}$ if there are constants $c>0$ and $\alpha>0$ independent of $n$ such that
$$
\begin{equation*}
\left|\left[A_{n}\right]_{i j}\right| \leq c \mathrm{e}^{-\alpha d_{n}(i, j)} \quad \text { for all } \quad i, j=1, \ldots, n \tag{6.3}
\end{equation*}
$$

Proposition 6.4. Let $\left\{A_{n}\right\}$ be a sequence of $n \times n$ matrices satisfying the exponential decay property (6.3) relative to a sequence of graphs $\left\{G_{n}\right\}$ having uniformly bounded maximal degree. Then, for any given $0<\epsilon<c$, each $A_{n}$ contains at most $O(n)$ entries greater than $\epsilon$ in magnitude.

Proof. For a fixed node $i$, the condition $\left|\left[A_{n}\right]_{i j}\right|>\epsilon$ together with (6.3) immediately implies
$$
\begin{equation*}
d_{n}(i, j)<\frac{1}{\alpha} \ln \left(\frac{c}{\epsilon}\right) . \tag{6.4}
\end{equation*}
$$

Since $c$ and $\alpha$ are independent of $n$, inequality (6.4), together with the assumption that the graphs $G_{n}$ have bounded maximal degree, implies that for any row of the matrix (indexed by $i$ ), there is at most a constant number of entries that have magnitude greater than $\epsilon$. Hence, only $O(n)$ entries in $A_{n}$ can satisfy $\left|\left[A_{n}\right]_{i j}\right|>\epsilon$.

Remark 6.5. Note that the hypothesis of uniformly bounded maximal degrees is certainly satisfied if the graphs $G_{n}$ have uniformly bounded bandwidths (recall that the bandwidth of a graph is just the bandwidth of the corresponding adjacency matrix). This special case corresponds to the matrix sequence $\left\{A_{n}\right\}$ having the off-diagonal exponential decay property.

Under the same assumptions as Proposition 6.4, we can show that it is possible to approximate each $A_{n}$ to within an arbitrarily small error $\epsilon>0$ in norm with a sparse matrix $A_{n}^{(m)}$ (i.e., a matrix containing only $O(n)$ nonzero entries).

Proposition 6.6. Assume the hypotheses of Proposition 6.4 are satisfied. Define the matrix $A_{n}^{(m)}=\left(\left[A_{n}^{(m)}\right]_{i j}\right)$, where
$$
\left[A_{n}^{(m)}\right]_{i j}=\left\{\begin{array}{cl}
{\left[A_{n}\right]_{i j}} & \text { if } \quad d_{n}(i, j) \leq m \\
0 & \text { otherwise }
\end{array}\right.
$$

Then, for any given $\epsilon>0$, there exists $\bar{m}$ independent of $n$ such that $\left\|A_{n}-A_{n}^{(m)}\right\|_{1}<\epsilon$ for all $m \geq \bar{m}$. Moreover, if $A=A^{*}$, then also $\left\|A_{n}-A_{n}^{(m)}\right\|_{2}<\epsilon$ for all $m \geq \bar{m}$. Furthermore, each $A_{n}^{(m)}$ contains only $O(n)$ nonzeros.

Proof. For each $n$ and $m$ and for $1 \leq j \leq n$, let
$$
K_{n}^{m}(j):=\left\{i \mid 1 \leq i \leq n \text { and } d_{n}(i, j)>m\right\} .
$$

We have
$$
\left\|A_{n}-A_{n}^{(m)}\right\|_{1}=\max _{1 \leq j \leq n} \sum_{i \in K_{n}^{m}(j)}\left|\left[A_{n}\right]_{i j}\right| \leq c \max _{1 \leq j \leq n} \sum_{i \in K_{n}^{m}(j)} \mathrm{e}^{-\alpha d_{n}(i, j)} .
$$

Letting $\lambda=\mathrm{e}^{-\alpha}$, we obtain
$$
\left\|A_{n}-A_{n}^{(m)}\right\|_{1} \leq c \max _{1 \leq j \leq n} \sum_{i \in K_{n}^{m}(j)} \lambda^{d_{n}(i, j)} \leq c \sum_{k=m+1}^{n} \lambda^{k}<c \sum_{k=m+1}^{\infty} \lambda^{k}=c \frac{\lambda^{m+1}}{1-\lambda} .
$$

Since $0<\lambda<1$, for any given $\epsilon>0$ we can always find $\bar{m}$ such that
$$
c \frac{\lambda^{m+1}}{1-\lambda} \leq \epsilon \quad \text { for all } \quad m \geq \bar{m}
$$

If $A_{n}=A_{n}^{*}$, then $\left\|A_{n}-A_{n}^{(m)}\right\|_{2} \leq\left\|A_{n}-A_{n}^{(m)}\right\|_{1}<\epsilon$ for all $m \geq \bar{m}$. The last assertion follows from the bounded maximal degree assumption.

Hence, when forming the overlap matrices and discrete Hamiltonians, only matrix elements corresponding to "sufficiently nearby" basis functions (i.e., basis functions having sufficient overlap) need to be computed, the others being negligibly small. The resulting matrices are therefore sparse, and indeed banded for 1D problems, with a number of nonzeros that grows linearly in the matrix dimension. The actual bandwidth, or sparsity pattern, may depend on the choice and numbering (ordering) of basis functions and (for the discrete Hamiltonians) on the strength of the interactions, i.e., on the form of the potential function $V$ in the Hamiltonian operator.

It should be kept in mind that while the number of nonzeros in the Hamiltonians discretized using (say) Gaussian-type orbitals is $O(n)$, the actual number of nonzeros per row can be quite high, indeed much higher than when finite differences or finite elements are used to discretize the same operators. It is not unusual to have hundreds or even thousands of nonzeros per row. On the other hand, the matrices are very often not huge in size. As already mentioned, the size $n$ of the matrix is the total number of basis functions, which is a small or moderate multiple (between 2 and 25 , say) of the number $n_{e}$ of electrons. For example, if $n_{b} \approx 10$ and $n_{e} \approx 2000$, the size of $H$ will be $n \approx 20,000$ and $H$ could easily contain several millions of nonzeros. This should be compared with "real space" discretizations based on finite elements or high-order finite difference schemes [116]. The resulting Hamiltonians are usually very sparse, with a number of nonzero entries per row averaging a few tens at most [7]. However, these matrices are of much larger dimension than the matrices obtained using basis sets consisting of atom-centered orbitals. In this case, methodologies based on approximating the density matrix are currently not feasible, except for 1D problems. The same remark applies to discretizations based on plane waves, which tend to produce matrices of an intermediate size between those obtained using localized basis sets and those resulting from the use of real space discretizations. These matrices are actually dense and are never formed explicitly. Instead, they are only used in the form of matrix-vector products, which can be implemented efficiently by means of FFTs; see, e.g., [116].

The possibility of developing linear scaling methods for electronic structure largely depends on the localization properties of the density matrix $P$. It is therefore critical to understand the decay behavior of the density matrix. Since at zero temperature the density matrix is just a particular spectral projector, we consider next some general properties of such projectors.
7. General Properties of Orthogonal Projectors. While our main goal in this paper is to study decay properties in orthogonal projectors associated with certain sequences of sparse matrices of increasing size, it is useful to first establish some a priori estimates for the entries of general projectors. Indeed, the intrinsic properties of a projector, like idempotency, positive semidefiniteness, and the relations between their trace, rank, and Frobenius norm, tend to impose rather severe constraints on the magnitude of its entries, particularly for increasing dimension and rank.

We begin by observing that in an orthogonal projector $P$, all entries $P_{i j}$ satisfy $\left|P_{i j}\right| \leq 1$ and, since $P$ is positive semidefinite, its largest entry is on the main diagonal. Also, the trace and rank coincide: $\operatorname{Tr}(P)=\operatorname{rank}(P)$. Moreover, $\|P\|_{2}=1$ and $\|P\|_{F}=\sqrt{\operatorname{Tr}(P)}$.

In the context of electronic structure computations, we deal with a sequence of $n \times n$ orthogonal projectors $\left\{P_{n}\right\}$ of rank $n_{e}$, where $n=n_{b} \cdot n_{e}$ with $n_{e}$ increasing and $n_{b}$ fixed. Hence,
$$
\begin{equation*}
\operatorname{Tr}\left(P_{n}\right)=\operatorname{rank}\left(P_{n}\right)=n_{e} \quad \text { and } \quad\left\|P_{n}\right\|_{F}=\sqrt{n_{e}} . \tag{7.1}
\end{equation*}
$$

For convenience, we will call a sequence of orthogonal projectors $\left\{P_{n}\right\}$ satisfying (7.1) a density matrix sequence; the entries of $P_{n}$ will be denoted by $\left[P_{n}\right]_{i j}$. We have the following lemma.

Lemma 7.1. Let $\left\{P_{n}\right\}$ be a density matrix sequence. Then
$$
\frac{\sum_{i \neq j}\left|\left[P_{n}\right]_{i j}\right|^{2}}{\left\|P_{n}\right\|_{F}^{2}} \leq 1-\frac{1}{n_{b}}
$$

Proof. Just observe that $\operatorname{Tr}\left(P_{n}\right)=\sum_{i=1}^{n}\left[P_{n}\right]_{i i}=n_{e}$ together with $\left|\left[P_{n}\right]_{i i}\right| \leq 1$ for all $i$ imply that the minimum of the sum $\sum_{i=1}^{n}\left|\left[P_{n}\right]_{i i}\right|^{2}$ is achieved when $\left[P_{n}\right]_{i i}=$ $\frac{n_{e}}{n}=\frac{1}{n_{b}}$ for all $i$. Hence, $\sum_{i=1}^{n}\left|\left[P_{n}\right]_{i i}\right|^{2} \geq \frac{n}{n_{b}^{2}}=\frac{n_{e}}{n_{b}}$. Therefore,
$$
\begin{equation*}
\sum_{i \neq j}\left|\left[P_{n}\right]_{i j}\right|^{2}=\left\|P_{n}\right\|_{F}^{2}-\sum_{i=1}^{n}\left|\left[P_{n}\right]_{i i}\right|^{2} \leq\left(1-\frac{1}{n_{b}}\right) n_{e} \tag{7.2}
\end{equation*}
$$
and the result follows dividing through by $\left\|P_{n}\right\|_{F}^{2}=n_{e}$.
Remark 7.2. From the proof one can trivially see that the bound (7.2) is sharp. In section 10 we give a nontrivial example where the bound is attained.

Theorem 7.3. Let $\left\{P_{n}\right\}$ be a density matrix sequence. Then, for any $\epsilon>0$, the number of entries of $P_{n}$ greater than or equal to $\epsilon$ in magnitude grows at most linearly with $n$.

Proof. Clearly, it suffices to show that the number of off-diagonal entries $\left[P_{n}\right]_{i j}$ with $\left|\left[P_{n}\right]_{i j}\right| \geq \epsilon$ can grow at most linearly with $n$. Let
$$
\mathcal{I}=\{(i, j) \mid 1 \leq i, j \leq n \text { and } i \neq j\} \quad \text { and } \quad \mathcal{I}_{\epsilon}=\left\{(i, j) \in \mathcal{I}| |\left[P_{n}\right]_{i j} \mid \geq \epsilon\right\}
$$

Then obviously
$$
\sum_{i \neq j}\left|\left[P_{n}\right]_{i j}\right|^{2}=\sum_{(i, j) \in \mathcal{I}_{\epsilon}}\left|\left[P_{n}\right]_{i j}\right|^{2}+\sum_{(i, j) \in \mathcal{I} \backslash \mathcal{I}_{\epsilon}}\left|\left[P_{n}\right]_{i j}\right|^{2}
$$
and, if $\left|\mathcal{I}_{\epsilon}\right|=K$, then
$$
\sum_{i \neq j}\left|\left[P_{n}\right]_{i j}\right|^{2} \geq K \epsilon^{2} \quad \Rightarrow \quad \frac{\sum_{i \neq j}\left|\left[P_{n}\right]_{i j}\right|^{2}}{\left\|P_{n}\right\|_{F}^{2}} \geq \frac{K \epsilon^{2}}{n_{e}}=\frac{K \epsilon^{2} n_{b}}{n} .
$$

Hence, by Lemma 7.1,
$$
\frac{K \epsilon^{2} n_{b}}{n} \leq \frac{\sum_{i \neq j}\left|\left[P_{n}\right]_{i j}\right|^{2}}{\left\|P_{n}\right\|_{F}^{2}} \leq 1-\frac{1}{n_{b}},
$$
from which we obtain the bound
$$
\begin{equation*}
K \leq \frac{n}{\epsilon^{2} n_{b}}\left(1-\frac{1}{n_{b}}\right), \tag{7.3}
\end{equation*}
$$
which shows that the number $K$ of entries of $P_{n}$ with $\left|\left[P_{n}\right]_{i j}\right| \geq \epsilon$ can grow at most as $O(n)$ for $n \rightarrow \infty$.

Remark 7.4. Due to the presence of the factor $\epsilon^{2}$ in the denominator of the bound (7.3), for small $\epsilon$ the proportion of entries of $P_{n}$ that are not smaller than $\epsilon$ can actually be quite large unless $n$ is huge. Nevertheless, the result is interesting because it shows that in any density matrix sequence, the proportion of entries larger than a prescribed threshold must vanish as $n \rightarrow \infty$. In practice, for density matrices corresponding to sparse Hamiltonians with gap, localization occurs already for moderate values of $n$.

We pointed out in the previous section that if the entries in a matrix sequence $\left\{A_{n}\right\}$ decay at least algebraically with exponent $p>1$ away from the main diagonal, with rates independent of $n$, then for any prescribed $\epsilon>0$ it is possible to find a
sequence of approximants $\left\{A_{n}^{(m)}\right\}$ with a fixed bandwidth $m$ (or sparsity pattern) such that $\left\|A_{n}-A_{n}^{(m)}\right\|<\epsilon$. This applies in particular to density matrix sequences. The next result shows that, in principle, a linear rate of decay is enough to allow for banded (or sparse) approximation to within any prescribed relative error in the Frobenius norm.

Theorem 7.5. Let $\left\{P_{n}\right\}$ be a density matrix sequence and assume that there exists $c>0$ independent of $n$ such that $\left|\left[P_{n}\right]_{i j}\right| \leq c /(|i-j|+1)$ for all $i, j=1, \ldots, n$. Then, for all $\epsilon>0$, there exists a positive integer $\bar{m}$ independent of $n$ such that
$$
\frac{\left\|P_{n}-P_{n}^{(m)}\right\|_{F}}{\left\|P_{n}\right\|_{F}} \leq \epsilon \quad \text { for all } m \geq \bar{m}
$$
where $P_{n}^{(m)}$ is the $m$-banded approximation obtained by setting to zero all the entries of $P_{n}$ outside the band.

Proof. We subtract $P_{n}^{(m)}$ from $P_{n}$ and compute $\left\|P_{n}-P_{n}^{(m)}\right\|_{F}^{2}$ by adding the squares of the nonzero entries in the upper triangular part of $P_{n}-P_{n}^{(m)}$ diagonal by diagonal and multiplying the result by 2 (since the matrices are Hermitian). Using the decay assumption we obtain
$$
\left\|P_{n}-P_{n}^{(m)}\right\|_{F}^{2} \leq 2 c^{2} \sum_{k=1}^{n-m-1} \frac{k}{(n-k+1)^{2}}=2 c^{2} \sum_{k=1}^{n-m-1} \frac{k}{[k-(n+1)]^{2}}
$$

To obtain an upper bound for the right-hand side, we observe that the function
$$
f(x)=\frac{x}{(x-a)^{2}}, \quad a=n+1
$$
is strictly increasing and convex on the interval $[1, n-m]$. Hence, the sum can be bounded above by the integral of $f(x)$ taken over the same interval:
$$
\sum_{k=1}^{n-m-1} \frac{k}{(n-k+1)^{2}}<\int_{1}^{n-m} \frac{x}{(x-a)^{2}} d x, \quad a=n+1
$$

Evaluating the integral and substituting $a=n+1$ in the result, we obtain
$$
\left\|P_{n}-P_{n}^{(m)}\right\|_{F}^{2}<2 c^{2}\left[\ln \left(\frac{m+1}{n}\right)+(n+1)\left(\frac{1}{m+1}-\frac{1}{n}\right)\right] .
$$

Dividing by $\left\|P_{n}\right\|_{F}^{2}=n_{e}$, we find
$$
\frac{\left\|P_{n}-P_{n}^{(m)}\right\|_{F}^{2}}{\left\|P_{n}\right\|_{F}^{2}}<\frac{2 c^{2}}{n_{e}}\left[\ln \left(\frac{m+1}{n}\right)+(n+1)\left(\frac{1}{m+1}-\frac{1}{n}\right)\right]<\frac{2 c^{2}}{n_{e}} \frac{n+1}{m+1} .
$$

Recalling that $n=n_{b} \cdot n_{e}$, we can rewrite the last inequality as
$$
\frac{\left\|P_{n}-P_{n}^{(m)}\right\|_{F}^{2}}{\left\|P_{n}\right\|_{F}^{2}}<\frac{2 c^{2}}{m+1} \frac{n+1}{n_{e}}=\frac{2 c^{2}}{m+1}\left(n_{b}+\frac{1}{n_{e}}\right) \leq \frac{2 c^{2}}{m+1}\left(n_{b}+1\right),
$$
a quantity which can be made arbitrarily small by taking $m$ sufficiently large.
Remark 7.6. In practice, linear decay (or even algebraic decay with a small exponent $p \geq 1$ ) is too slow to be useful in the development of practical $O(n)$ algorithms.

For example, from the above estimates we obtain $\bar{m}=O\left(\epsilon^{-2}\right)$, which is clearly not a very encouraging result, even allowing for the fact that the above bound may be pes- simistic in general. To date, practical linear scaling algorithms have been developed only for density matrix sequences exhibiting exponential off-diagonal decay.

In the case of exponential decay, one can prove the following result.
Theorem 7.7. Let $\left\{P_{n}\right\}$ be a density matrix sequence with $\left|\left[P_{n}\right]_{i j}\right| \leq c \mathrm{e}^{-\alpha|i-j|}$, where $c>0$ and $\alpha>0$ are independent of $n$. Let $\left\{P_{n}^{(m)}\right\}$ be the corresponding sequence of $m$-banded approximations. Then there exists $k_{0}>0$ independent of $n$ and $m$ such that
$$
\frac{\left\|P_{n}-P_{n}^{(m)}\right\|_{F}^{2}}{\left\|P_{n}\right\|_{F}^{2}} \leq k_{0} \mathrm{e}^{-2 \alpha m} .
$$

Proof. The proof is similar to that of Theorem 7.5, except that it is now easy to evaluate the upper bound and the constants exactly. We omit the details.

Remark 7.8. It is immediate to see that the foregoing bound implies the much more favorable estimate $\bar{m}=O\left(\ln \epsilon^{-1}\right)$.

Again, similar results hold for arbitrary sparsity patterns, replacing $|i-j|$ with the graph distance. More precisely, the following result holds.

Theorem 7.9. Let $\left\{P_{n}\right\}$ be a density matrix sequence with the exponential decay property with respect to a sequence of graphs $\left\{G_{n}\right\}$ having uniformly bounded maximal degree. Then, for all $\epsilon>0$, there exists a positive integer $\bar{m}$ independent of $n$ such that
$$
\frac{\left\|P_{n}-P_{n}^{(m)}\right\|_{F}}{\left\|P_{n}\right\|_{F}} \leq \epsilon \quad \text { for all } \quad m \geq \bar{m}
$$
where $P_{n}^{(m)}$ is sparse, i.e., it contains only $O(n)$ nonzeros.
We consider now some of the consequences of approximating full, but localized matrices with sparse ones. The following quantity plays an important role in many electronic structure codes:
$$
\langle E\rangle=\operatorname{Tr}(P H)=\varepsilon_{1}+\varepsilon_{2}+\cdots+\varepsilon_{n_{e}},
$$
where $\varepsilon_{i}$ denotes the $i$ th eigenvalue of the discrete Hamiltonian $H$. Minimization of $\operatorname{Tr}(P H)$, subject to the constraints $P=P^{*}=P^{2}$ and $\operatorname{Tr}(P)=n_{e}$, is the basis of several linear scaling algorithms; see, e.g., [24, 53, 77, 79, 93, 95, 97]. Note that in the tight-binding model, and also within the independent electron approximation, the quantity $\langle E\rangle$ represents the single-particle energy [6, 53, 97, 128]. Now, assume that $\hat{H} \approx H$ and $\hat{P} \approx P$ and define the corresponding approximation of $\langle E\rangle$ as $\langle\hat{E}\rangle=\operatorname{Tr}(\hat{P} \hat{H})$. (We note in passing that in order to compute $\langle\hat{E}\rangle=\operatorname{Tr}(\hat{P} \hat{H})$, only the entries of $\hat{P}$ corresponding to nonzero entries in $\hat{H}$ need to be computed.) Let $\Delta_{P}=\hat{P}-P$ and $\Delta_{H}=\hat{H}-H$. We have
$$
\langle\hat{E}\rangle=\operatorname{Tr}\left[\left(P+\Delta_{P}\right)\left(H+\Delta_{H}\right)\right]=\operatorname{Tr}(P H)+\operatorname{Tr}\left(P \Delta_{H}\right)+\operatorname{Tr}\left(\Delta_{P} H\right)+\operatorname{Tr}\left(\Delta_{P} \Delta_{H}\right) .
$$

Neglecting the last term, we obtain for $\delta_{E}=|\langle E\rangle-\langle\hat{E}\rangle|$ the bound
$$
\delta_{E} \leq\left|\operatorname{Tr}\left(P \Delta_{H}\right)\right|+\left|\operatorname{Tr}\left(\Delta_{P} H\right)\right| .
$$

Recalling that the Frobenius norm is the matrix norm induced by the inner product $\langle A, B\rangle=\operatorname{Tr}\left(B^{*} A\right)$, using the Cauchy-Schwarz inequality and $\|P\|_{F}=\sqrt{n_{e}}$ we find
$$
\delta_{E} \leq \sqrt{n_{e}}\left\|\Delta_{H}\right\|_{F}+\left\|\Delta_{P}\right\|_{F}\|H\|_{F}
$$

Now, since the orthogonal projector $P$ is invariant with respect to scalings of the Hamiltonian, we can assume $\|H\|_{F}=1$, so that $\delta_{E} \leq \sqrt{n_{e}}\left\|\Delta_{H}\right\|_{F}+\left\|\Delta_{P}\right\|_{F}$ holds. In practice, a bound on the relative error would be more meaningful. Unfortunately, it is not easy to obtain a rigorous bound in terms of the relative error in the approximate projector $\hat{P}$. If, however, we replace the relative error in $\langle\hat{E}\rangle$ with the normalized error obtained by dividing the absolute error by the number $n_{e}$ of electrons, we obtain
$$
\frac{\delta_{E}}{n_{e}} \leq \frac{\left\|\Delta_{H}\right\|_{F}}{\sqrt{n_{e}}}+\frac{\left\|\Delta_{P}\right\|_{F}}{n_{e}} .
$$

A similar bound for $\delta_{E} / n_{e}$ that involves matrix 2 -norms can be obtained as follows. Recall that $n=n_{b} \cdot n_{e}$, and that $\|A\|_{F} \leq \sqrt{n}\left\|_{A}\right\|_{2}$ for any $n \times n$ matrix $A$. Observing that the von Neumann trace inequality [68, pp. 182-183] implies $\left|\operatorname{Tr}\left(P \Delta_{H}\right)\right| \leq \operatorname{Tr}(P)\left\|\Delta_{H}\right\|_{2}=n_{e}\left\|\Delta_{H}\right\|_{2}$, we obtain
$$
\begin{equation*}
\frac{\delta_{E}}{n_{e}} \leq\left\|\Delta_{H}\right\|_{2}+\sqrt{\frac{n_{b}}{n_{e}}}\left\|\Delta_{P}\right\|_{2} \tag{7.4}
\end{equation*}
$$

Since $n_{b}$ is constant, an interesting consequence of (7.4) is that for large system sizes (i.e., in the limit as $n_{e} \rightarrow \infty$ ), the bound on the normalized error in $\langle\hat{E}\rangle$ is essentially determined by the truncation error in the Hamiltonian $H$ rather than by the error in the density matrix $P$.

On the other hand, scaling $H$ so that $\|H\|_{F}=1$ may not be advisable in practice. Indeed, since the Frobenius norm of the Hamiltonian grows unboundedly for $n_{e} \rightarrow \infty$, rescaling $H$ so that $\|H\|_{F}=1$ would lead to a loss of significant information when truncation is applied in the case of large systems. A more sensible scaling, which is often used in algorithms for electronic structure computations, is to divide $\|H\|$ by its largest eigenvalue in magnitude, so that $\|H\|_{2}=1$. This is consistent with the assumption, usually satisfied in practice, that the spectra of the Hamiltonians remain bounded as $n_{e} \rightarrow \infty$. (Note that this is the same normalization used to establish the decay bounds in section 8.) With this scaling we readily obtain, to first order, the bound
$$
\begin{equation*}
\frac{\delta_{E}}{n_{e}} \leq\left\|\Delta_{H}\right\|_{2}+n_{b}\left\|\Delta_{P}\right\|_{2} \tag{7.5}
\end{equation*}
$$
showing that errors in $\Delta_{H}$ and $\Delta_{P}$ enter the estimate for the normalized error in the objective function $\operatorname{Tr}(P H)$ with approximately the same weight, since $n_{b}$ is a moderate constant. We also note that since both error matrices $\Delta_{H}$ and $\Delta_{P}$ are Hermitian, (6.2) implies that the bounds (7.4) and (7.5) remain true if the 2 -norm is replaced by the 1 -norm. We mention that the problem of the choice of norm in the measurement of truncation errors has been discussed in [111, 114]. These authors emphasize the use of the 2 -norm, which is related to the distance between the exact and inexact (perturbed) occupied subspaces $\mathcal{X}:=\operatorname{Range}(P)$ and $\hat{\mathcal{X}}:=\operatorname{Range}(\hat{P})$ as measured by the sine of the principal angle between $\mathcal{X}$ and $\hat{\mathcal{X}}$; see [111].

One important practical aspect, which we do not address here, is that in many quantum chemistry codes the matrices have a natural block structure (where each block corresponds, for instance, to the basis functions centered at a given atom); hence, dropping is usually applied to submatrices rather than to individual entries. Exploitation of the block structure is also desirable in order to achieve high performance in matrix-matrix products and other operations; see, e.g., [24, 25, 112].

We conclude this section with a few remarks on the infinite-dimensional case. Recall that any separable, complex Hilbert space $\mathscr{H}$ is isometrically isomorphic to
the sequence space
$$
\ell^{2}:=\left\{\left(\xi_{n}\right) \mid \xi_{n} \in \mathbb{C} \forall n \in \mathbb{N} \text { and } \sum_{n=1}^{\infty}\left|\xi_{n}\right|^{2}<\infty\right\} .
$$

Moreover, if $\left\{e_{n}\right\}$ is an orthonormal basis in $\mathscr{H}$, to any bounded linear operator $\mathcal{A}$ on $\mathscr{H}$ there corresponds the infinite matrix $A=\left(A_{i j}\right)$ acting on $\ell^{2}$, uniquely defined by $A_{i j}=\left\langle e_{j}, \mathcal{A} e_{i}\right\rangle$. Note that each column of $A$ must be in $\ell^{2}$, hence the entries $A_{i j}$ in each column of $A$ must go to zero for $i \rightarrow \infty$. The same is true for the entries in each row (for $j \rightarrow \infty$ ) since $A^{*}=\left(A_{j i}^{*}\right)$, the adjoint of $A$, is also a (bounded) operator defined everywhere on $\ell^{2}$. More precisely, for any bounded linear operator $A=\left(A_{i j}\right)$ on $\ell^{2}$ the following bounds hold:
$$
\begin{equation*}
\sum_{j=1}^{\infty}\left|A_{i j}\right|^{2} \leq\|A\|_{2}^{2} \quad \text { for all } i \quad \text { and } \quad \sum_{i=1}^{\infty}\left|A_{i j}\right|^{2} \leq\|A\|_{2}^{2} \quad \text { for all } j, \tag{7.6}
\end{equation*}
$$
since $\|A\|_{2}=\left\|A^{*}\right\|_{2}$.
An orthogonal projector $\mathcal{P}$ on $\mathscr{H}$ is a self-adjoint $\left(\mathcal{P}=\mathcal{P}^{*}\right)$, idempotent $\left(\mathcal{P}=\mathcal{P}^{2}\right)$ linear operator. Such an operator is necessarily bounded, with norm $\|\mathcal{P}\|=1$. Hence, (7.6) implies
$$
\begin{equation*}
\sum_{j=1}^{\infty}\left|P_{i j}\right|^{2} \leq 1 \tag{7.7}
\end{equation*}
$$
where $P=\left(P_{i j}\right)$ denotes the matrix representation of $\mathcal{P}$. The idempotency condition implies
$$
P_{i j}=\sum_{k=1}^{\infty} P_{i k} P_{k j} \quad \text { for all } \quad i, j=1,2, \ldots
$$

In particular, for $i=j$ we get, using the hermiticity property $P_{i j}=P_{j i}^{*}$,
$$
\begin{equation*}
P_{i i}=\sum_{k=1}^{\infty} P_{i k} P_{k i}=\sum_{k=1}^{\infty}\left|P_{i k}\right|^{2} \quad \text { for all } \quad i=1,2, \ldots \tag{7.8}
\end{equation*}
$$

Now, since $P$ is a projector its entries satisfy $\left|P_{i j}\right| \leq 1$; therefore, (7.8) is a strengthening of inequality (7.7). Note in particular that the off-diagonal entries in the first row (or column) of $P$ must satisfy
$$
\sum_{j>1}\left|P_{1 j}\right|^{2} \leq 1-\left|P_{11}\right|^{2}
$$
those in the second row (or column) must satisfy
$$
\sum_{j>2}\left|P_{2 j}\right|^{2} \leq 1-\left|P_{22}\right|^{2}-\left|P_{12}\right|^{2}
$$
and in general the entries $P_{i j}$ with $j>i$ must satisfy
$$
\begin{equation*}
\sum_{j>i}\left|P_{i j}\right|^{2} \leq 1-\sum_{k=1}^{i}\left|P_{k i}\right|^{2} \quad \text { for all } i=1,2, \ldots . \tag{7.9}
\end{equation*}
$$

Hence, decay in the off-diagonal entries in the $i$ th row of $P$ must be fast enough for the bounds (7.9) to hold. In general, however, it is not easy to quantify the asymptotic rate of decay to zero of the off-diagonal entries in an arbitrary orthogonal projector on $\ell^{2}$. In general, the rate of decay can be rather slow. In section 10 we will see an example of a spectral projector associated with a very simple tridiagonal Hamiltonian for which the off-diagonal entries decay linearly to zero.
8. Decay Results. In this section we present and discuss some results on the decay of entries for the Fermi-Dirac function applied to Hamiltonians and for the density matrix (spectral projector corresponding to occupied states). We consider both the banded case and the case of more general sparsity patterns. The proofs, which require some basic tools from polynomial approximation theory, will be given in section 8.3.
8.1. Bounds for the Fermi-Dirac Function. We begin with the following result for the banded case. As usual in this paper, in the following one should think of the positive integer $n$ as being of the form $n=n_{b} \cdot n_{e}$, with $n_{b}$ constant and $n_{e} \rightarrow \infty$.

Theorem 8.1. Let $m$ be a fixed positive integer and consider a sequence of matrices $\left\{H_{n}\right\}$ such that
(i) $H_{n}$ is an $n \times n$ Hermitian, $m$-banded matrix for all $n$;
(ii) for every $n$, all the eigenvalues of $H_{n}$ lie in the interval $[-1,1]$.

For a given Fermi level $\mu$ and inverse temperature $\beta$, define for each $n$ the $n \times n$ Hermitian matrix $F_{n}:=f_{F D}\left(H_{n}\right)=\left[I_{n}+\mathrm{e}^{\beta\left(H_{n}-\mu I_{n}\right)}\right]^{-1}$. Then there exist constants $c>0$ and $\alpha>0$, independent of $n$, such that the following decay bound holds:
$$
\begin{equation*}
\left|\left[F_{n}\right]_{i j}\right| \leq c \mathrm{e}^{-\alpha|i-j|}, \quad i \neq j \tag{8.1}
\end{equation*}
$$

The constants $c$ and $\alpha$ can be chosen as
$$
\begin{align*}
& c=\frac{2 \chi M(\chi)}{\chi-1}, \quad M(\chi)=\max _{z \in \mathcal{E}_{\chi}}\left|f_{F D}(z)\right|  \tag{8.2}\\
& \alpha=\frac{1}{m} \ln \chi \tag{8.3}
\end{align*}
$$
for any $1<\chi<\bar{\chi}$, where
$$
\begin{align*}
\bar{\chi} & =\frac{\sqrt{\sqrt{\left(\beta^{2}\left(1-\mu^{2}\right)-\pi^{2}\right)^{2}+4 \pi^{2} \beta^{2}}-\beta^{2}\left(1-\mu^{2}\right)+\pi^{2}}}{\sqrt{2} \beta} \\
& +\frac{\sqrt{\sqrt{\left(\beta^{2}\left(1-\mu^{2}\right)-\pi^{2}\right)^{2}+4 \pi^{2} \beta^{2}}+\beta^{2}\left(1+\mu^{2}\right)+\pi^{2}}}{\sqrt{2} \beta} \tag{8.4}
\end{align*}
$$
and $\mathcal{E}_{\chi}$ is the unique ellipse with foci in -1 and 1 , with semiaxes $\kappa_{1}>1$ and $\kappa_{2}>0$, and $\chi=\kappa_{1}+\kappa_{2}$.

Remark 8.2. The ellipse $\mathcal{E}_{\chi}$ in the previous theorem is unique because the identity $\sqrt{\kappa_{1}^{2}-\kappa_{2}^{2}}=1$, valid for any ellipse with foci in 1 and -1 , implies $\kappa_{1}-\kappa_{2}=1 /\left(\kappa_{1}+\right. \kappa_{2}$ ), hence the parameter $\chi=\kappa_{1}+\kappa_{2}$ alone completely characterizes the ellipse.

Remark 8.3. Theorem 8.1 can be immediately generalized to the case where the spectra of the sequence $\left\{H_{n}\right\}$ are contained in an interval $[a, b]$ for any $a<b \in \mathbb{R}$. It suffices to shift and scale each Hamiltonian,
$$
\widehat{H}_{n}=\frac{2}{b-a} H_{n}-\frac{a+b}{b-a} I_{n}
$$

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/1bc7e2d7-f39d-466c-801c-6b7da6d033c3-25.jpg?height=702&width=775&top_left_y=373&top_left_x=502}
\captionsetup{labelformat=empty}
\caption{Fig. 8.1 Bounds (8.1) with $\mu=0$ and $\beta=10$ for three different values of $\chi$.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/1bc7e2d7-f39d-466c-801c-6b7da6d033c3-25.jpg?height=440&width=785&top_left_y=1226&top_left_x=485}
\captionsetup{labelformat=empty}
\caption{Fig. 8.2 Logarithmic plot of the bounds (8.1) with $\mu=0$ and $\beta=10$.}
\end{figure}
so that $\widehat{H}_{n}$ has spectrum in $[-1,1]$. For the decay bounds to be independent of $n$, however, $a$ and $b$ must be independent of $n$.

It is important to note that there is a certain amount of arbitrariness in the choice of $\chi$, and therefore of $c$ and $\alpha$. If one is mainly interested in a fast asymptotic decay behavior (i.e., for sufficiently large $|i-j|$ ), it is desirable to choose $\chi$ as large as possible. On the other hand, if $\chi$ is very close to $\bar{\chi}$, then the constant $c$ is likely to be quite large and the bounds might be too pessimistic. Let us look at an example. Take $\mu=0$; in this case we have
$$
\bar{\chi}=\left(\pi+\sqrt{\beta^{2}+\pi^{2}}\right) / \beta \quad \text { and } \quad M(\chi)=\left|1 /\left(1+\mathrm{e}^{\beta \zeta}\right)\right|, \quad \text { where } \quad \zeta=\mathrm{i} \frac{\chi^{2}-1}{2 \chi} .
$$

Note that, in agreement with experience, decay is faster for smaller $\beta$ (i.e., higher electronic temperatures); see sections 8.3 and 8.7 for additional details and discussion. Figures 8.1 and 8.2 show the behavior of the bound given by (8.1) on the first row

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/1bc7e2d7-f39d-466c-801c-6b7da6d033c3-26.jpg?height=635&width=820&top_left_y=365&top_left_x=457}
\captionsetup{labelformat=empty}
\caption{Fig. 8.3 Plot of $c$ as a function of $\chi$ with $\mu=0$ and $\beta=10$.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/1bc7e2d7-f39d-466c-801c-6b7da6d033c3-26.jpg?height=443&width=792&top_left_y=1134&top_left_x=485}
\captionsetup{labelformat=empty}
\caption{Fig. 8.4 Logarithmic plot of the bounds (8.1) with $\mu=0$ and $\beta=10$ for several values of $\chi$.}
\end{figure}
of a $200 \times 200$ tridiagonal matrix ( $m=1$ ) for $\beta=10$ and for three values of $\chi$. It is easy to see from the plots that the asymptotic behavior of the bounds improves as $\chi$ increases; however, the bound given by $\chi=1.362346$ is less useful than the bound given by $\chi=1.3$. Figure 8.3 is a plot of $c$ as a function of $\chi$ and it shows that $c$ grows very large when $\chi$ is close to $\bar{\chi}$. This is expected, since $f_{F D}(z)$ has two poles given by $z= \pm \mathrm{i} \pi / \beta$ on the regularity ellipse $\mathcal{E}_{\bar{\chi}}$. It is clear from Figures 8.1 and 8.2 that $\chi=1.3$ is the best choice among the three proposed values if one is interested in determining a bandwidth outside of which the entries of $F_{n}$ can be safely neglected. As already observed in [14, 106], improved bounds can be obtained by adaptively choosing different (typically increasing) values of $\chi$ as $|i-j|$ grows, and by using as a bound the (lower) envelope of the curves plotted in Figure 8.4, which shows the behavior of the decay bounds for several values of $\chi \in(1.1, \bar{\chi})$, with $\bar{\chi} \approx 1.3623463$.

The results of Theorem 8.1 can be generalized to the case of Hamiltonians with rather general sparsity patterns; see $[14,31,106]$. To this end, we make use of the notion of geodetic distance in a graph already used in section 6. The following result holds.

\section*{Theorem 8.4. Consider a sequence of matrices $\left\{H_{n}\right\}$ such that}
(i) $H_{n}$ is an $n \times n$ Hermitian matrix for all $n$;
(ii) the spectra $\sigma\left(H_{n}\right)$ are uniformly bounded and contained in $[-1,1]$ for all $n$. Let $d_{n}(i, j)$ be the graph distance associated with $H_{n}$. Then the following decay bound holds:
$$
\begin{equation*}
\left|\left[F_{n}\right]_{i j}\right| \leq c \mathrm{e}^{-\theta d_{n}(i, j)}, \quad i \neq j, \tag{8.5}
\end{equation*}
$$
where $\theta=\ln \chi$ and the remaining notation and choice of constants are as in Theorem 8.1.

We remark that in order for the bound (8.5) to be meaningful from the point of view of linear scaling, we need to impose some restrictions on the asymptotic sparsity of the graph sequence $\left\{G_{n}\right\}$. As discussed in section 6, $O(n)$ approximations of $F_{n}$ are possible if the graphs $G_{n}$ have maximum degree uniformly bounded with respect to $n$. This guarantees that the distance $d_{n}(i, j)$ grows unboundedly with $|i-j|$, at a rate independent of $n$ for $n \rightarrow \infty$.
8.2 Density Matrix Decay for Systems with Gap. The previous results establish exponential decay bounds for the Fermi-Dirac function of general localized Hamiltonians and thus for density matrices of arbitrary systems at positive electronic temperature. In this subsection we consider the case of gapped systems (like insulators) at zero temperature. In this case, as we know, the density matrix is the spectral projector onto the occupied subspace. As an example, we consider the density matrix corresponding to the linear alkane n -Dopentacontane $\mathrm{C}_{52} \mathrm{H}_{106}$ composed of 52 carbon and 106 hydrogen atoms, discretized in a Gaussian-type orbital basis. The number of occupied states is 209 , or half the total number of electrons in the system. ${ }^{7}$ The corresponding Hamiltonian in the original nonorthogonal basis is displayed in Figure 9.1 (top) and the "orthogonalized" Hamiltonian $\tilde{H}$ is shown in Figure 9.1 (bottom). Figure 8.5 displays the zero-temperature density matrix, which is seen to decay exponentially away from the main diagonal. Comparing Figure 8.5 and Figure 9.1, we can see that for a truncation level of $10^{-8}$, the bandwidth of the density matrix is only slightly larger than that of the Hamiltonian. The eigenvalue spectrum of the Hamiltonian, scaled and shifted so that its spectrum is contained in the interval $[-1,1]$, is shown in Figure 8.6. One can clearly see a large gap $(\approx 1.4)$ between the 52 low-lying eigenvalues corresponding to the core electrons in the system, as well as the smaller HOMO-LUMO gap ( $\approx 0.1$ ) separating the 209 occupied states from the virtual (unoccupied) ones. It is worth emphasizing that the exponential decay of the density matrix is independent of the size of the system; that is, if the alkane chain was made arbitrarily long by adding C and H atoms to it, the density matrix would be of course much larger in size, but its bandwidth would remain virtually unchanged for the same truncation level, due to the fact that the bandwidth and the HOMO-LUMO gap of the Hamiltonian do not appreciably change as the number of particles increases. It is precisely this independence of the rate of decay (hence, of the bandwidth) to system size that makes $O(n)$ approximations possible (and competitive) for large $n$.

Let us now see how Theorem 8.1 can be used to prove decay bounds on the entries of density matrices. Let $H$ be the discrete Hamiltonian associated with a certain physical system and let $\mu$ be the Fermi level of interest for this system. We assume that the spectrum of $H$ has a gap $\gamma$ around $\mu$, that is, we have $\gamma=\varepsilon^{+}-\varepsilon^{-}>0$, where

\footnotetext{
${ }^{7}$ Here spin is being taken into account, so that the density kernel is given by $\rho\left(\mathbf{r}, \mathbf{r}^{\prime}\right)= 2 \sum_{i=1}^{n_{e} / 2} \psi_{i}(\mathbf{r}) \psi_{i}\left(\mathbf{r}^{\prime}\right)^{*}$; see, e.g., [88, p. 10].
}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/1bc7e2d7-f39d-466c-801c-6b7da6d033c3-28.jpg?height=598&width=586&top_left_y=339&top_left_x=573}
\captionsetup{labelformat=empty}
\caption{Fig. 8.5 Magnitude of the entries in the density matrix for the linear alkane $\mathrm{C}_{52} \mathrm{H}_{106}$ chain, with 209 occupied states. White: $<10^{-8}$; yellow: $10^{-8}-10^{-6}$; green: $10^{-6}-10^{-4}$; blue: $10^{-4}-10^{-2}$; black: $>10^{-2}$. Note: nz refers to the number of "black" entries.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/1bc7e2d7-f39d-466c-801c-6b7da6d033c3-28.jpg?height=725&width=949&top_left_y=1121&top_left_x=418}
\captionsetup{labelformat=empty}
\caption{Fig. 8.6 Spectrum of the Hamiltonian for $\mathrm{C}_{52} \mathrm{H}_{106}$.}
\end{figure}
$\varepsilon^{+}$is the smallest eigenvalue of $H$ to the right of $\mu$ and $\varepsilon^{-}$is the largest eigenvalue of $H$ to the left of $\mu$. In the particular case of the HOMO-LUMO gap, we have $\varepsilon^{-}=\varepsilon_{n_{e}}$ and $\varepsilon^{+}=\varepsilon_{n_{e}+1}$.

The Fermi-Dirac function can be used to approximate the Heaviside function; the larger $\beta$, the better the approximation. More precisely, the following result is easy to prove (see [106]).

Proposition 8.5. Let $\delta>0$ be given. If $\beta$ is such that
$$
\begin{equation*}
\beta \geq \frac{2}{\gamma} \ln \left(\frac{1-\delta}{\delta}\right), \tag{8.6}
\end{equation*}
$$
then $1-f_{F D}\left(\varepsilon^{-}\right) \leq \delta$ and $f_{F D}\left(\varepsilon^{+}\right) \leq \delta$.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/1bc7e2d7-f39d-466c-801c-6b7da6d033c3-29.jpg?height=686&width=975&top_left_y=322&top_left_x=403}
\captionsetup{labelformat=empty}
\caption{Fig. 8.7 Approximations of the Heaviside function by the Fermi-Dirac function $(\mu=0)$ for different values of $\gamma$ and $\delta=10^{-6}$.}
\end{figure}

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/1bc7e2d7-f39d-466c-801c-6b7da6d033c3-29.jpg?height=607&width=1030&top_left_y=1211&top_left_x=365}
\captionsetup{labelformat=empty}
\caption{Fig. 8.8 Behavior of the minimum acceptable value of $\beta$ as a function of $\gamma$ for different values of $\delta$.}
\end{figure}

In Figure 8.7 we show Fermi-Dirac approximations to the Heaviside function (with a jump at $\mu=0$ ) for different values of $\gamma$ between 0.1 and 1 , where $\beta$ has been chosen so as to reduce the error in Proposition 8.5 above the value $\delta=10^{-6}$. The behavior of $\beta$ as a function of $\gamma$ according to (8.6) is plotted in Figure 8.8.

As a consequence of Theorem 8.1 and Proposition 8.5 we have the following corollary.

Corollary 8.6. Let $n_{b}$ be a fixed positive integer and $n=n_{b} \cdot n_{e}$, where the integers $n_{e}$ form a monotonically increasing sequence. Let $\left\{H_{n}\right\}$ be a sequence of Hermitian $n \times n$ matrices with the following properties:
1. Each $H_{n}$ has bandwidth $m$ independent of $n$.
2. There exist two fixed intervals $I_{1}=[-1, a], I_{2}=[b, 1] \subset \mathbb{R}$, with $\gamma=b-a>$ 0 , such that for all $n=n_{b} \cdot n_{e}, I_{1}$ contains the smallest $n_{e}$ eigenvalues of $H_{n}$ (counted with their multiplicities) and $I_{2}$ contains the remaining $n-n_{e}$ eigenvalues.
Let $P_{n}$ denote the $n \times n$ spectral projector onto the subspace spanned by the eigenvectors associated with the $n_{e}$ smallest eigenvalues of $H_{n}$ for each $n$. Let $\delta>0$ be arbitrary. Then there exist constants $c>0, \alpha>0$ independent of $n$ such that
$$
\begin{equation*}
\left|\left[P_{n}\right]_{i j}\right| \leq \min \left\{1, c \mathrm{e}^{-\alpha|i-j|}\right\}+\delta \quad \text { for all } \quad i \neq j \tag{8.7}
\end{equation*}
$$

The constants $c$ and $\alpha$ can be computed from (8.2) and (8.3), where $\chi$ is chosen in the interval $(1, \bar{\chi})$, with $\bar{\chi}$ given by (8.4) and $\beta$ such that (8.6) holds.

Corollary 8.6 allows us to determine a priori a bandwidth $\bar{m}$ independent of $n$ outside of which the entries of $P_{n}$ are smaller than a prescribed tolerance $\tau>0$. Observe that it is not possible to incorporate $\delta$ in the exponential bound, but, at least in principle, one may always choose $\delta$ smaller than a certain threshold. For instance, one may take $\delta<\tau / 2$ and define $\bar{m}$ as the smallest integer value of $m$ such that the relation $c \mathrm{e}^{-\alpha m} \leq \tau / 2$ holds.

In the case of Hamiltonians with a general sparsity pattern one may apply Theorem 8.4 to obtain a more general version of Corollary 8.6. If the fixed bandwidth hypothesis is removed, the following bound holds:
$$
\begin{equation*}
\left|\left[P_{n}\right]_{i j}\right| \leq \min \left\{1, c \mathrm{e}^{-\theta d_{n}(i, j)}\right\}+\delta \quad \text { for all } \quad i \neq j \tag{8.8}
\end{equation*}
$$
with $\theta=\ln \chi$. Once again, for the result to be meaningful some restriction on the sparsity patterns, like the uniformly bounded maximum degree assumption already discussed, must be imposed.
8.3. Proof of Decay Bounds. Theorem 8.1 is a consequence of results proved in [12, Theorem 2.2] and [106, Theorem 2.2]; its proof relies on a fundamental result in polynomial approximation theory known as Bernstein's theorem [92]. Given a function $f$ continuous on $[-1,1]$ and a positive integer $k$, the $k$ th best approximation error for $f$ is the quantity
$$
E_{k}(f)=\inf \left\{\max _{-1 \leq x \leq 1}|f(x)-p(x)|: p \in P_{k}\right\}
$$
where $P_{k}$ is the set of all polynomials with real coefficients and degree less than or equal to $k$. Bernstein's theorem describes the asymptotic behavior of the best approximation error for a function $f$ analytic on a domain containing the interval $[-1,1]$.

Consider the family of ellipses in the complex plane with foci in -1 and 1 . As already mentioned, an ellipse in this family is completely determined by the sum $\chi>1$ of its half-axes and will be denoted as $\mathcal{E}_{\chi}$.

Theorem 8.7 (Bernstein). Let the function $f$ be analytic in the interior of the ellipse $\mathcal{E}_{\chi}$ and continuous on $\mathcal{E}_{\chi}$. Moreover, assume that $f(z)$ is real for real $z$. Then
$$
E_{k}(f) \leq \frac{2 M(\chi)}{\chi^{k}(\chi-1)}
$$
where $M(\chi)=\max _{z \in \mathcal{E}_{\chi}}|f(z)|$.

Let us now consider the special case where $f(z):=f_{F D}(z)=1 /\left(1+\mathrm{e}^{\beta(z-\mu)}\right)$ is the Fermi-Dirac function of parameters $\beta$ and $\mu$. Observe that $f_{F D}(z)$ has poles in $\mu \pm \mathrm{i} \frac{\pi}{\beta}$, so the admissible values for $\chi$ with respect to $f_{F D}(z)$ are given by $1<\chi<\bar{\chi}$, where the parameter $\bar{\chi}$ is such that $\mu \pm \mathrm{i} \frac{\pi}{\beta} \in \mathcal{E}_{\bar{\chi}}$ (the regularity ellipse for $f=f_{F D}$ ). Also observe that smaller values of $\beta$ correspond to a greater distance between the poles of $f_{F D}(z)$ and the real axis, which in turn yields a larger value of $\bar{\chi}$. In other words, the smaller $\beta$, the faster the decay in Theorem 8.1. Explicit computation of $\bar{\chi}$ yields (8.4).

Now, let $H_{n}$ be as in Theorem 8.1. We have
$$
\left\|f_{F D}\left(H_{n}\right)-p_{k}\left(H_{n}\right)\right\|_{2}=\max _{x \in \sigma\left(H_{n}\right)}\left|f_{F D}(x)-p_{k}(x)\right| \leq E_{k}\left(f_{F D}\right) \leq c q^{k+1}
$$
where $c=2 \chi M(\chi) /(\chi-1)$ and $q=1 / \chi$. The Bernstein approximation of degree $k$ gives a bound on $\left|\left[f_{F D}\left(H_{n}\right)\right]_{i j}\right|$ when $\left[p_{k}\left(H_{n}\right)\right]_{i j}=0$, that is, when $|i-j|>m k$. We may also assume $|i-j| \leq m(k+1)$. Therefore, we have
$$
\left|\left[f_{F D}\left(H_{n}\right)\right]_{i j}\right| \leq c \mathrm{e}^{m(k+1) \ln \left(q^{1 / m}\right)}=c \mathrm{e}^{-\alpha m(k+1)} \leq c \mathrm{e}^{-\alpha|i-j|}
$$

As for Theorem 8.4, note that for a general sparsity pattern we have $\left[\left(H_{n}\right)^{k}\right]_{i j}=0$, and therefore $\left[p_{k}\left(H_{n}\right)\right]_{i j}=0$, whenever $d_{n}(i, j)>k$. Writing $d_{n}(i, j)=k+1$ we obtain
$$
\left|\left[f_{F D}\left(H_{n}\right)\right]_{i j}\right| \leq c(1 / \chi)^{k+1}=c \mathrm{e}^{-\theta d_{n}(i j)}
$$

Let us now prove Corollary 8.6. Assume that $\beta$ satisfies the inequality (8.6) for given values of $\delta$ and $\gamma$. If we approximate the Heaviside function with step at $\mu$ by means of the Fermi-Dirac function $f_{F D}(x)=1 /\left(1+\mathrm{e}^{\beta(x-\mu)}\right)$, the pointwise approximation error is given by $g(x)=\mathrm{e}^{\beta(x-\mu)} /\left(1+\mathrm{e}^{\beta(x-\mu)}\right)$ for $x<\mu$ and by $f_{F D}(x)$ for $x>\mu$. It is easily seen that $g(x)$ is a monotonically increasing function, whereas $f_{F D}$ is monotonically decreasing. As a consequence, for each Hamiltonian $H_{n}$ we have that $1-f_{F D}(\lambda) \leq \delta$ for all eigenvalues $\lambda \in I_{1}$ and $f_{F D}(\lambda) \leq \delta$ for all $\lambda \in I_{2}$. In other words, the pointwise approximation error on the spectrum of $H_{n}$ is always bounded by $\delta$. Therefore, we have
$$
\left|\left[P_{n}-f_{F D}\left(H_{n}\right)\right]_{i j}\right| \leq\left\|P_{n}-f_{F D}\left(H_{n}\right)\right\|_{2} \leq \delta
$$

We may then conclude using Theorem 8.1 that
$$
\left|\left[P_{n}\right]_{i j}\right| \leq\left|\left[f_{F D}\left(H_{n}\right)\right]_{i j}\right|+\delta \leq c \mathrm{e}^{-\alpha|i-j|}+\delta
$$

Finally, recall that in an orthogonal projector no entry can exceed unity in absolute value. With this in mind, (8.7) and (8.8) readily follow.
8.4. Additional Bounds. Theorems 8.1 and 8.4 rely on Bernstein's result on best polynomial approximation. Following the same argument, one may derive decay bounds for the density matrix from any other estimate on the best polynomial approximation error for classes of functions that include the Fermi-Dirac function. For instance, consider the following result of Achieser (see [92, Theorem 78] and [1]).

Theorem 8.8. Let the function $f$ be analytic in the interior of the ellipse $\mathcal{E}_{\chi}$. Suppose that $|\operatorname{Re} f(z)|<1$ holds in $\mathcal{E}_{\chi}$ and that $f(z)$ is real for real $z$. Then the following bound holds:
$$
\begin{equation*}
E_{k}(f) \leq \frac{4}{\pi} \sum_{\nu=0}^{\infty} \frac{(-1)^{\nu}}{(2 \nu+1) \cosh ((2 \nu+1)(k+1) \ln \chi)} \tag{8.9}
\end{equation*}
$$

The series in (8.9) converges quite fast; therefore, it suffices to compute a few terms explicitly to obtain a good approximation of the bound. A rough estimate shows that, in order to approximate the right-hand side of (8.9) within a tolerance $\tau$, one may truncate the series after $\nu_{0}$ terms, where $r^{\nu_{0}}<\tau(1-r)$ and $r=\chi^{-\frac{k+1}{2}}$.

Observe that, as in Bernstein's results, there is again a degree of arbitrariness in the choice of $\chi$. However, the admissible range for $\chi$ is smaller here because of the hypothesis $|\operatorname{Re} f(z)|<1$.

The resulting matrix decay bounds have the form
$$
\begin{equation*}
\left|\left[f_{F D}\left(H_{n}\right)\right]_{i j}\right| \leq \frac{4}{\pi} \sum_{\nu=0}^{\infty} \frac{(-1)^{\nu}}{(2 \nu+1) \cosh ((2 \nu+1)(d(i, j)+1) \ln \chi)} \tag{8.10}
\end{equation*}
$$
for the case of general sparsity patterns. While these bounds are less transparent than those derived from Bernstein's theorem, they are computable. We have found that the bounds (8.10) improve on (8.1) for entries close to the main diagonal, but do not seem to have a better asymptotic behavior. A possibility would be to combine the two bounds by taking the smaller of the two values.

So far we have only considered bounds based on best approximation of analytic functions defined on a single interval. In [61], Hasson obtained an interesting result on polynomial approximation of a step function defined on the union of two symmetric intervals. Let $a, b \in \mathbb{R}$ with $0<a<b$ and let $\operatorname{sgn}(x)$ be the sign function defined on $[-b,-a] \cup[a, b]$, i.e., $\operatorname{sgn}(x)=-1$ on $[-b,-a]$ and $\operatorname{sgn}(x)=1$ on $[a, b]$. Notice that the sign function is closely related to the Heaviside function $h(x)$, since we have $h(x)=\frac{1}{2}(1+\operatorname{sgn}(x))$.

Proposition 8.9. There exists a positive constant $K$ such that
$$
\begin{equation*}
E_{k}(\operatorname{sgn} ;[-b,-a] \cup[a, b]) \leq K \frac{\left(\sqrt{\frac{b-a}{b+a}}\right)^{k}}{\sqrt{k}} \tag{8.11}
\end{equation*}
$$

Given a sequence of Hamiltonians $\left\{H_{n}\right\}$ with gapped spectra, one may choose $a$ and $b$ and shift $H_{n}$, if necessary, so that the spectrum of each $H_{n}$ is contained in $[-b,-a] \cup[a, b]$ and the eigenvalues corresponding to occupied states belong to $[-b,-a]$. Then we obtain the following decay bound for the density matrix:
$$
\begin{equation*}
\left|\left[P_{n}\right]_{i j}\right| \leq K \frac{\mathrm{e}^{-\xi d(i, j)}}{2 \sqrt{d(i, j)}}, \quad \text { where } \quad \xi=\frac{1}{2} \ln \frac{b+a}{b-a} \tag{8.12}
\end{equation*}
$$

Under the bounded maximal degree condition, the rate of decay is independent of $n$.
A few remarks on (8.12) are in order:
- Since (8.12) relies directly on a polynomial approximation of the step function, we do not need here the extra term $\delta$ found in (8.8).
- Unfortunately, it is not possible to assess whether (8.12) may be useful in practice without an explicit formula-or at least an estimate-for the constant $K$. The asymptotic decay rate, however, is faster than exponential and indeed faster than for other bounds; a comparison is shown in Figure 8.9 (top). Notice that this logarithmic plot is only meant to show the slope of the bound (which is computed for $K=1$ ).
- A disadvantage of (8.12) is the requirement that the intervals containing the spectra $\sigma\left(H_{n}\right)$ should be symmetric with respect to 0 . Of course one may always choose $a$ and $b$ so that this hypothesis is satisfied, but the quality of

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/1bc7e2d7-f39d-466c-801c-6b7da6d033c3-33.jpg?height=1249&width=774&top_left_y=339&top_left_x=507}
\captionsetup{labelformat=empty}
\caption{Fig. 8.9 Top: Logarithmic plot of Hasson (dashed line) and Bernstein-type (dotted line) decay bounds for a $100 \times 100$ tridiagonal matrix with spectrum in $[-1,-0.25] \cup[0.25,1]$. The solid line plots the first row of the "exact" density matrix. Bottom: Logarithmic plot of Hasson decay bounds (dashed line) and first rows of density matrices associated with matrices with different eigenvalue distributions (solid and dotted lines).}
\end{figure}
the decay bound deteriorates if $b$ (or $-b$ ) is not close to the maximum (resp., minimum) eigenvalue; see Figure 8.9 (bottom). The dashed line shows the slope of the decay bound for $a=0.25$ and $b=1$, in a logarithmic scale. The dotted line displays the behavior of the first row of the density matrix associated with a tridiagonal $100 \times 100$ matrix with spectrum in $[-1,-0.25] \cup [0.25,1]$. The solid line refers to the first row of the density matrix associated with a matrix with spectrum in $[-0.4375,-0.25] \cup[0.25,1]$. The first matrix is clearly better approximated by the decay bound than the second one.
As one can see from the two plots in Figure 8.9, even for $c=K=1$ both types of decay bounds are rather conservative, and estimating the truncation bandwidth $\bar{m}$ needed to achieve a prescribed error from these bounds would lead to an overly large band. Hence, the bounds may not be very useful in practice. For further discussion of these issues, see section 8.9.
8.5. Further Results. Let us assume again that we have a sequence $\left\{H_{n}\right\}$ of Hermitian $n \times n$ Hamiltonians (with $n=n_{b} \cdot n_{e}, n_{b}$ fixed, $n_{e} \rightarrow \infty$ ) such that
- the matrices $H_{n}$ are banded with uniformly bounded bandwidth, or sparse with graphs having uniformly bounded maximum degree;
- the spectra $\sigma\left(H_{n}\right)$ are uniformly bounded;
- the sequence $\left\{H_{n}\right\}$ has a "stable" spectral gap; i.e., there exist real numbers $g_{1}<g_{2}$ such that $\left[g_{1}, g_{2}\right] \cap \sigma\left(H_{n}\right)=\emptyset$ for sufficiently large $n$.
In this subsection we let
- $\mu:=\left(g_{2}+g_{1}\right) / 2$ (Fermi level);
- $\gamma:=g_{2}-\mu=\mu-g_{1}$ (absolute spectral gap).

Note that because of the uniformly bounded spectra assumption, the absolute spectral gap is within a constant of the relative gap previously defined.

Chui and Hasson study in [29] the asymptotic behavior of the error of best polynomial approximation for a sufficiently smooth function $f$ defined on the set $I=[-b,-a] \cup[a, b]$, with $0<a<b$. Denote as $\mathcal{C}(I)$ the space of real-valued continuous functions on $I$, with the uniform convergence norm. Then we have (see [29, Theorem 1] and [84]) the following theorem.

Theorem 8.10. Let $f \in \mathcal{C}(I)$ be such that $\left.f\right|_{[-b,-a]}$ is the restriction of a function $f_{1}$ analytic on the left half-plane $\operatorname{Re} z<0$ and $\left.f\right|_{[a, b]}$ is the restriction of a function $f_{2}$ analytic on the right half-plane $\operatorname{Re} z>0$. Then
$$
\limsup _{k \rightarrow \infty}\left[E_{k}(f, I)\right]^{1 / k} \leq \sqrt{\frac{b-a}{b+a}}
$$
where $E_{k}(f, I)$ is the error of best polynomial approximation for $f$ on $I$.
The authors of [29] observe that the above result cannot be obtained by extending $f(x)$ to a continuous function on $[-b, b]$ and applying known bounds for polynomial approximation over a single interval. Theorem 8.10 looks potentially useful for our purposes, except that it provides an asymptotic result, rather than an explicit bound for each value of $k$. Therefore, we need to reformulate the argument in [29]. To this end, we prove a variant of Bernstein's theorem (cf. Theorem 8.7) adapted to our goals. Instead of working on the interval $[-1,1]$, we want to bound the approximation error on the interval $\left[a^{2}, b^{2}\right]$.

Theorem 8.11. Let $f \in \mathcal{C}\left(\left[a^{2}, b^{2}\right]\right)$ be the restriction of a function $f$ analytic in the interior of the ellipse $\mathcal{E}_{a^{2}, b^{2}}$ with foci in $a^{2}, b^{2}$ and a vertex at the origin. Then, for all $\xi$ with
$$
1<\xi<\bar{\xi}:=\frac{a+b}{a-b}
$$
there exists a constant $K$ such that
$$
E_{k}\left(f,\left[a^{2}, b^{2}\right]\right) \leq K\left(\frac{1}{\xi}\right)^{k}
$$

Proof. The proof closely parallels the argument given in [92] for the proof of Theorem 8.7. First of all, observe that the ellipse $\mathcal{E}_{\chi}$ in Bernstein's theorem has foci in $\pm 1$ and vertices in $\pm(\chi+1 / \chi) / 2$ and $\pm(\chi-1 / \chi) / 2$. The parameter $\chi$ is the sum of the lengths of the semiaxes. Similarly, the ellipse $\mathcal{E}_{a^{2}, b^{2}}$ has foci in $a^{2}, b^{2}$ and vertices in $0, a^{2}+b^{2}$, and $\left(a^{2}+b^{2}\right) / 2 \pm \mathrm{i} a b$. Also observe that $\bar{\xi}$ is the sum of the lengths of the semiaxes of $\mathcal{E}_{a^{2}, b^{2}}$, normalized with respect to the semifocal length, so that it plays
exactly the same role as $\chi$ for $\mathcal{E}_{\chi}$. Now we look for a conformal map that sends an annulus in the complex plane to the ellipse where $f$ is analytic. When this ellipse is $\mathcal{E}_{\chi}$, a suitable map is $u=c(v)=(v+1 / v) / 2$, which sends the annulus $\chi^{-1}<|v|<\chi$ to $\mathcal{E}_{\chi}$. When the desired ellipse has foci in $a^{2}, b^{2}$, we compose $c(v)$ with the change of variable
$$
x=\psi(u)=\left(u+\frac{a^{2}+b^{2}}{b^{2}-a^{2}}\right) \frac{b^{2}-a^{2}}{2},
$$
thus obtaining a function that maps the annulus $\mathcal{A}=\left\{\xi^{-1}<|v|<\xi\right\}$ to an ellipse. Denote this ellipse as $\mathcal{E}_{a^{2}, b^{2}, \xi}$ and observe that it is contained in the interior of $\mathcal{E}_{a^{2}, b^{2}}$. Therefore we have that the function
$$
f(\psi(c(v)))=f\left(\left[\frac{1}{2}\left(v+\frac{1}{v}\right)+\frac{a^{2}+b^{2}}{b^{2}-a^{2}}\right] \frac{b^{2}-a^{2}}{2}\right)
$$
is analytic on $\mathcal{A}$ and continuous on $|v|=\xi$. The proof now proceeds as in the original Bernstein theorem. The Laurent expansion
$$
f(\psi(c(v)))=\sum_{\nu=-\infty}^{\infty} \alpha_{\nu} v^{\nu}
$$
converges in $\mathcal{A}$ with $\alpha_{-\nu}=\alpha_{\nu}$. Moreover, we have the bound
$$
\left|\alpha_{\nu}\right|=\left|\frac{1}{2 \pi \mathrm{i}} \int_{|v|=\xi} \frac{f(\psi(c(v)))}{v^{\nu+1}} d v\right| \leq \frac{M(\xi)}{\xi^{\nu}}
$$
where $M(\xi)$ is the maximum value (in modulus) taken by $f$ on the ellipse $\mathcal{E}_{a^{2}, b^{2}, \xi}$.
Now observe that $u=c(v)$ describes the real interval $[-1,1]$ for $|v|=1$, so for $u \in[-1,1]$ we have
$$
f(\psi(u))=\alpha_{0}+2 \sum_{\nu=1}^{\infty} \alpha_{\nu} T_{\nu}(u),
$$
where $T_{\nu}(u)$ is the $\nu$ th Chebyshev polynomial. Since $\psi(u)$ is a linear transformation, we have $E_{k}\left(f(z),\left[a^{2}, b^{2}\right]\right)=E_{k}(f(u),[-1,1])$, so from the theory of Chebyshev approximation [92] we obtain
$$
E_{k}\left(f,\left[a^{2}, b^{2}\right]\right)=E_{k}(f(u),[-1,1]) \leq 2 M(\xi) \sum_{\nu=k+1}^{\infty} \xi^{-\nu}=\frac{2 M(\xi)}{\xi-1} \xi^{-k},
$$
hence the thesis. Note that the explicit value of $K$ is computable.
The following result is based on [29, Theorem 1].
Theorem 8.12. Let $f \in \mathcal{C}(I)$ be as in Theorem 8.10. Then, for all $\xi$ with
$$
1<\xi<\bar{\xi}:=\frac{a+b}{a-b}
$$
there exists $C>0$ independent of $k$ such that
$$
E_{k}(f, I) \leq C \xi^{-\frac{k}{2}}
$$

Proof. Let $P_{k}$ and $Q_{k}$ be polynomials of best uniform approximation of degree $k$ on the interval $\left[a^{2}, b^{2}\right]$ for the functions $f_{2}(\sqrt{x})$ and $f_{2}(\sqrt{x}) / \sqrt{x}$, respectively. Then by Theorem 8.11 there are constants $K_{1}$ and $K_{2}$ such that
$$
\begin{equation*}
\max _{x \in\left[a^{2}, b^{2}\right]}\left|P_{k}(x)-f_{2}(\sqrt{x})\right| \leq K_{1} \xi^{-k} \tag{8.13}
\end{equation*}
$$
and
$$
\begin{equation*}
\max _{x \in\left[a^{2}, b^{2}\right]}\left|Q_{k}(x)-f_{2}(\sqrt{x}) / \sqrt{x}\right| \leq K_{2} \xi^{-k} \tag{8.14}
\end{equation*}
$$

We use the polynomials $P_{k}$ and $Q_{k}$ to define a third polynomial $R_{2 k+1}(x):=\left[P_{k}\left(x^{2}\right)+\right. \left.x Q_{k}\left(x^{2}\right)\right] / 2$, of degree $\leq 2 k+1$, which approximates $f(x)$ on $[a, b]$ and has small norm on $[-b,-a]$. Indeed, from (8.13) and (8.14) we have
$$
\begin{align*}
\max _{x \in[a, b]}\left|R_{2 k+1}(x)-f(x)\right| & \leq \frac{1}{2} \max _{x \in[a, b]}\left|P_{k}\left(x^{2}\right)-f(x)\right|+\frac{1}{2} \max _{x \in[a, b]}\left|x Q_{k}\left(x^{2}\right)-f(x)\right|  \tag{8.15}\\
& \leq \frac{1}{2} K_{1} \xi^{-k}+\frac{1}{2} b K_{2} \xi^{-k}=\frac{K_{1}+b K_{2}}{2} \xi^{-k}
\end{align*}
$$
and
$$
\begin{array}{r}
\max _{x \in[-b,-a]}\left|R_{2 k+1}(x)\right| \leq \frac{1}{2} \max _{x \in[a, b]}\left|P_{k}\left(x^{2}\right)-f(x)+f(x)-x Q_{k}\left(x^{2}\right)\right| \\
\leq \frac{1}{2} \max _{x \in[a, b]}\left|P_{k}\left(x^{2}\right)-f(x)\right|+\frac{1}{2} \max _{x \in[a, b]}\left|x Q_{k}\left(x^{2}\right)-f(x)\right| \leq \frac{K_{1}+b K_{2}}{2} \xi^{-k} \tag{8.17}
\end{array}
$$

Similarly, we can find another polynomial $S_{2 k+1}(x)$ such that
$$
\begin{equation*}
\max _{x \in[-b,-a]}\left|S_{2 k+1}(x)-f(x)\right| \leq \frac{K_{3}+b K_{4}}{2} \xi^{-k} \tag{8.18}
\end{equation*}
$$
and
$$
\begin{equation*}
\max _{x \in[a, b]}\left|S_{2 k+1}(x)\right| \leq \frac{K_{3}+b K_{4}}{2} \xi^{-k} \tag{8.19}
\end{equation*}
$$

Then from the inequalities (8.15)-(8.19) we have
$$
\begin{array}{r}
\max _{x \in I}\left|R_{2 k+1}(x)+S_{2 k+1}(x)-f(x)\right| \leq \max _{x \in[a, b]}\left|R_{2 k+1}(x)-f(x)\right|+\max _{x \in[a, b]}\left|S_{2 k+1}(x)\right| \\
+\max _{x \in[-b,-a]}\left|S_{2 k+1}(x)-f(x)\right|+\max _{x \in[-b,-a]}\left|R_{2 k+1}(x)\right| \\
\leq\left(K_{1}+K_{3}+b\left(K_{2}+K_{4}\right)\right) \xi^{-k}
\end{array}
$$
and therefore
$$
E_{k}(f, I) \leq \sqrt{\xi}\left(K_{1}+K_{3}+b\left(K_{2}+K_{4}\right)\right) \xi^{-\frac{k}{2}}
$$
for odd values of $k$ and
$$
E_{k}(f, I) \leq \xi\left(K_{1}+K_{3}+b\left(K_{2}+K_{4}\right)\right) \xi^{-\frac{k}{2}}
$$
for even values of $k$. This completes the proof.

In the following we assume, without loss of generality, that $k$ is odd. In order to obtain bounds on the density matrix, we apply Theorem 8.12 to the step function $f$ defined on $I$ as
$$
f(x)=\left\{\begin{array}{ccc}
1 & \text { for } & -b \leq x \leq-a \\
0 & \text { for } & a \leq x \leq b
\end{array}\right.
$$
i.e., $f$ is the restriction of $f_{1}(z) \equiv 1$ on $[-b,-a]$ and the restriction of $f_{2}(z) \equiv 0$ on $[a, b]$. Here the polynomial approximation of $f_{2}(\sqrt{x}), f_{2}(\sqrt{x}) / \sqrt{x}$, and $f_{1}(\sqrt{-x})$ is exact, so we have $K_{1}=K_{2}=K_{3}=0$. As for $K_{4}$, observe that $|1 / \sqrt{z}|$ achieves its maximum on the vertex of $\mathcal{E}_{a^{2}, b^{2}, \xi}$ with smallest abscissa; therefore we have
$$
K_{4}=\frac{2 M(\xi)}{\xi-1}
$$
where
$$
M(\xi)=\frac{1}{\sqrt{z_{0}}} \quad \text { with } \quad z_{0}=\left[-\frac{1}{2}\left(\xi+\frac{1}{\xi}\right)+\frac{a^{2}+b^{2}}{b^{2}-a^{2}}\right] \frac{b^{2}-a^{2}}{2} .
$$

Moreover, we find $R_{2 k+1}(x) \equiv 0$ and $S_{2 k+1}(x)=\left(1+x V_{k}\left(x^{2}\right)\right) / 2$, where $V_{k}(x)$ is the polynomial of best uniform approximation for $1 / \sqrt{x}$ on $\left[a^{2}, b^{2}\right]$. Thus, we obtain the bound
$$
E_{k}(f, I) \leq C \xi^{-\frac{k}{2}},
$$
where $C$ is given by
$$
C=\sqrt{\xi} K_{4} b
$$

Let us now apply this result to our sequence of Hamiltonians. We will assume that the matrices are shifted so that $\mu=0$, that is, we replace each $H_{n}$ by $H_{n}-\mu I_{n}$. Under this hypothesis, the natural choice for $a$ is $a=\gamma$, whereas $b$ is the smallest number such that $\sigma\left(H_{n}\right) \subset[-b,-a] \cup[a, b]$ for every $n$.

Using the same argument used in section 8.3 for the derivation of matrix decay bounds (see also [12] and [14]), we can obtain bounds on the off-diagonal entries of $f\left(H_{n}\right)$. If $H_{n}$ is banded with bandwidth $m$ independent of $n$, we have
$$
\begin{equation*}
\left|\left[P_{n}\right]_{i j}\right|=\left|\left[f\left(H_{n}\right)\right]_{i j}\right| \leq \sqrt{\xi} \frac{2 M(\xi)}{\xi-1} b \xi^{-\frac{|i-j|}{2 m}} \tag{8.20}
\end{equation*}
$$
whereas if $H_{n}$ has a more general sparsity pattern, we obtain
$$
\begin{equation*}
\left|\left[P_{n}\right]_{i j}\right|=\left|\left[f\left(H_{n}\right)\right]_{i j}\right| \leq \sqrt{\xi} \frac{2 M(\xi)}{\xi-1} b \xi^{-\frac{d_{n}(i, j)}{2}} \tag{8.21}
\end{equation*}
$$
where $d_{n}(i, j)$ is the distance between nodes $i$ and $j$ in the graph $G_{n}$ associated with $H_{n}$.

Next, we compare the bounds derived in this section with those for the FermiDirac approximation of the step function obtained in section 8.1, using a suitable choice of the inverse temperature $\beta$. Recall that if $\mathcal{E}_{\chi}$ denotes the regularity ellipse for the Fermi-Dirac function, the earlier bounds for the banded case are
$$
\begin{equation*}
\left|\left[P_{n}\right]_{i j}\right| \leq \frac{2 M(\chi)}{\chi-1}\left(\frac{1}{\chi}\right)^{\frac{|i-j|}{m}} \tag{8.22}
\end{equation*}
$$

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/1bc7e2d7-f39d-466c-801c-6b7da6d033c3-38.jpg?height=631&width=912&top_left_y=330&top_left_x=433}
\captionsetup{labelformat=empty}
\caption{Fig. 8.10 Comparison of parameters $1 / \bar{\xi}$ and $1 / \bar{\chi}$ for several values of the spectral gap. Here $\delta=10^{-5}$.}
\end{figure}

For ease of computation, we assume in this section that $\mu=0$ and that the spectrum of each matrix $H_{n}$ is contained in $[-1,1]$. As explained in section 8.1, once $\gamma$ is known, we pick a tolerance $\delta$ and compute $\beta$ so that the Fermi-Dirac function provides a uniform approximation of the step function with error $\leq \delta$ outside the gap:
$$
\beta \geq \frac{2}{\gamma} \ln \left(\frac{1-\delta}{\delta}\right) .
$$

Then the supremum of the set of admissible values of $\chi$, which ensures optimal asymptotic decay in this framework, is
$$
\bar{\chi}=\left(\pi+\sqrt{\beta^{2}+\pi^{2}}\right) / \beta .
$$

Figures 8.10 and 8.11 compare the values of $1 / \bar{\xi}$ and $1 / \bar{\chi}$ (which characterize the behavior of the bounds (8.20) and (8.22), respectively). Note that in general we find $1 / \bar{\xi}<1 / \bar{\chi}$; this means that the asymptotic decay rate is higher for the bound based on disjoint interval approximation. Moreover, the disjoint interval method directly approximates the step function and therefore does not require one to choose a tolerance for "intermediate" approximation. As a result, the bounds based on disjoint interval approximation prescribe a smaller truncation bandwidth $\bar{m}$ in the approximation to the spectral projector in order to achieve a given level of error. For instance, in the tridiagonal case ( $m=1$ ) we observed a factor of three reduction in $\bar{m}$ compared to the previous bounds, independent of the size of the gap.
8.6 Dependence of the Rate of Decay on the Spectral Gap. As already mentioned in section 4, the functional dependence of the decay length (governing the rate of decay in the density matrix) on the spectral gap has been the subject of some discussion; see, for instance, [3, 70, 73, 104, 127, 140]. Some of these authors have argued that the decay length decreases like the square root of the gap if the Fermi level is located near one of the gap edges (i.e., close to either $\varepsilon_{n_{e}}$ or $\varepsilon_{n_{e}+1}$ ) and like the gap itself if the Fermi level falls in the middle of the gap. These estimates hold for the small gap limit.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/1bc7e2d7-f39d-466c-801c-6b7da6d033c3-39.jpg?height=628&width=910&top_left_y=333&top_left_x=433}
\captionsetup{labelformat=empty}
\caption{Fig. 8.II Logarithmic plot of parameters $1 / \bar{\xi}$ and $1 / \bar{\chi}$ with respect to several values of the spectral gap. Here $\delta=10^{-5}$.}
\end{figure}

In this section we address this problem by studying how the decay described by the bounds (8.20) and (8.21) behaves asymptotically with respect to $\gamma$ or, equivalently, with respect to $a$ (see the notation introduced in the previous section). Note that we are assuming here that the Fermi level falls exactly in the middle of the gap.

Let us rewrite (8.20) in the form
$$
\left|\left[P_{n}\right]_{i j}\right| \leq C \mathrm{e}^{-\alpha|i-j| / m},
$$
where
$$
\alpha=\frac{1}{2} \ln \xi=\frac{1}{2} \ln \left(\frac{a+b}{b-a}\right) .
$$

For a fixed $m$, the decay behavior is essentially described by the parameter $\alpha$. Let us assume for simplicity of notation that $b=1$, so that the spectral gap is normalized and the expression for $\alpha$ becomes
$$
\alpha=\frac{1}{2} \ln \left(\frac{1+a}{1-a}\right) .
$$

The Taylor expansion of $\alpha$ for $a$ small yields
$$
\alpha=a+\frac{a^{3}}{3}+o\left(a^{3}\right) .
$$

Therefore, for small values of $\gamma$, the decay behavior is described at first order by the gap itself, rather than by a more complicated function of $\gamma$. This result is consistent with similar ones found in the literature [70, 73, 140]. The fact that some systems exhibit density matrix decay lengths proportional to the square root of the gap (see, e.g., [73]) does not contradict our result: since we are dealing here with upper bounds, a square root dependence, which corresponds to faster decay for small $a$, is still consistent with our bounds. Given that our bounds are completely general, it does not come as a surprise that we obtain the more conservative estimate among the alternatives discussed in the literature.
8.7. Dependence of the Rate of Decay on the Temperature. Another issue that has stirred up some controversy in the literature concerns the precise rate of decay in the density matrix for metals at positive temperature; see, e.g., the results and discussion in [3,52,70]. Recall that in metals at positive temperatures $T$, the density matrix $F_{n}=f_{F D}\left(H_{n}\right)$ decays exponentially. The question is whether the decay length is proportional to $T$ or to $\sqrt{T}$ for small $T$. Our approach shows that the decay length is proportional to $T$.

Indeed, from the analysis in section 8.1, in particular Theorems 8.1 and 8.4, we find that the decay length $\alpha$ in the exponential decay bound (8.1) (or, more generally, the decay length $\theta$ in the bound (8.5)) behaves like $\ln \chi$, where-assuming for simplicity that $\mu=0$, as before - the parameter $\chi$ is any number satisfying
$$
1<\chi<\bar{\chi}, \quad \bar{\chi}=\left(\pi+\sqrt{\beta^{2}+\pi^{2}}\right) / \beta
$$

Letting $x=\pi / \beta=\pi k_{B} T$ and observing that for small $x$
$$
\ln \left(x+\sqrt{1+x^{2}}\right)=x+o\left(x^{2}\right)
$$
we conclude that, at low temperatures, the decay length is proportional to $k_{B} T$. This conclusion is in complete agreement with the results in [52, 70]. To the best of our knowledge, this is the first time this result has been established in a fully rigorous and completely general manner.
8.8. Other Approaches. Decay bounds on the entries of spectral projectors can also be obtained from the contour integral representation
$$
\begin{equation*}
P_{n}=\frac{1}{2 \pi \mathrm{i}} \int_{\Gamma}\left(z I_{n}-H_{n}\right)^{-1} d z \tag{8.23}
\end{equation*}
$$
where $\Gamma$ is a simple closed curve (counterclockwise oriented) in $\mathbb{C}$ surrounding a portion of the real axis containing those eigenvalues of $H_{n}$ that correspond to the occupied states, and only those. Componentwise, (8.23) becomes
$$
\left[P_{n}\right]_{i j}=\frac{1}{2 \pi \mathrm{i}} \int_{\Gamma}\left[\left(z I_{n}-H_{n}\right)^{-1}\right]_{i j} d z, \quad 1 \leq i, j \leq n
$$
from which we obtain
$$
\left|\left[P_{n}\right]_{i j}\right| \leq \frac{1}{2 \pi} \int_{\Gamma}\left|\left[\left(z I_{n}-H_{n}\right)^{-1}\right]_{i j}\right| d z, \quad 1 \leq i, j \leq n
$$

Assume the matrices $H_{n}$ are banded, with uniformly bounded spectra and bandwidths as $n \rightarrow \infty$. By [34, Proposition 2.3] there exist, for all $z \in \Gamma$, explicitly computable constants $c(z) \geq 0$ and $0<\lambda(z)<1$ (independent of $n$ ) such that
$$
\begin{equation*}
\left|\left[\left(z I_{n}-H_{n}\right)^{-1}\right]_{i j}\right| \leq c(z)[\lambda(z)]^{|i-j|} \tag{8.24}
\end{equation*}
$$
for all $i, j=1, \ldots, n$. Moreover, $c$ and $\lambda$ depend continuously on $z \in \Gamma$. Since $\Gamma$ is compact we can set
$$
\begin{equation*}
c=\max _{z \in \Gamma} c(z) \quad \text { and } \quad \lambda=\max _{z \in \Gamma} \lambda(z) \tag{8.25}
\end{equation*}
$$

Now let us assume that the matrices $H_{n}$ have spectral gaps $\gamma_{n}$ satisfying $\inf _{n} \gamma_{n}>0$. It is then clear that $c$ is finite and that $\lambda \in(0,1)$. Hence, we obtain the following bound:
$$
\begin{equation*}
\left|\left[P_{n}\right]_{i j}\right| \leq\left(c \cdot \frac{\ell(\Gamma)}{2 \pi}\right) \lambda^{|i-j|} \tag{8.26}
\end{equation*}
$$
for all $i, j=1, \ldots, n$, where $\ell(\Gamma)$ denotes the length of $\Gamma$. Finally, letting $C=c \cdot \frac{\ell(\Gamma)}{2 \pi}$ and $\alpha=-\ln \lambda$, we obtain the exponential decay bounds
$$
\begin{equation*}
\left|\left[P_{n}\right]_{i j}\right| \leq C \cdot \mathrm{e}^{-\alpha|i-j|}, \quad 1 \leq i, j \leq n \tag{8.27}
\end{equation*}
$$
with both $C>0$ and $\alpha>0$ independent of $n$. As usual, the bounds can be easily extended to the case of general sparsity patterns. One disadvantage of this approach is that explicit evaluation of the constants $C$ and $\alpha$ is rather complicated.

The integral representation (8.23) is useful not only as a theoretical tool, but also increasingly as a computational tool. Indeed, quadrature rules with suitably chosen nodes $z_{1}, \ldots, z_{k} \in \Gamma$ can be used to approximate the integral in (8.23), leading to
$$
\begin{equation*}
P_{n} \approx \sum_{i=1}^{k} w_{i}\left(z_{i} I_{n}-H_{n}\right)^{-1} \tag{8.28}
\end{equation*}
$$
for suitable quadrature weights $w_{1}, \ldots, w_{k}$. Note that this amounts to a rational approximation of $P_{n}=h\left(H_{n}\right)$. In practice, using the trapezoidal rule with a small number of nodes suffices to achieve high accuracy, due to the exponential convergence of this quadrature rule for analytic functions [33]. Note that if $P_{n}$ is real, then it is sufficient to use just the $z_{i}$ in the upper half-plane and then take the real part of the result [65, p. 307]. If the spectral gap $\gamma_{n}$ for $H_{n}$ is not too small, all the resolvents $\left(z_{i} I_{n}-H_{n}\right)^{-1}$ decay rapidly away from the main diagonal, with exponential rate independent of $n_{e}$. Hence, $O(n)$ approximation is possible, at least in principle. Rational approximations of the type (8.28) are especially useful in those situations where only selected entries of $P_{n}$ are required. Then only the corresponding entries of the resolvents $\left(z_{i} I_{n}-H_{n}\right)^{-1}$ need to be computed. For instance, in some cases only the diagonal entries of $P_{n}$ are needed [116]. In others, only entries in positions corresponding to the nonzero entries in the Hamiltonian $H_{n}$ must be computed; this is the case, for instance, when computing the objective function $\langle E\rangle=\operatorname{Tr}\left(P_{n} H_{n}\right)$ in density matrix minimization algorithms. Computing selected entries of a resolvent is not an easy problem. However, progress has been made on this front in several recent papers; see, e.g., [78, 82, 83, 124, 126].
8.9. Computational Considerations. In the preceding sections we have rigorously established exponential decay bounds for zero-temperature density matrices corresponding to finite-range Hamiltonians with nonvanishing spectral gap ("insulators"), as well as for density matrices corresponding to arbitrary finite-range Hamiltonians at positive electronic temperatures. Our results are very general and apply to a wide variety of physical systems and discretizations. Hence, a mathematical justification of the physical phenomenon of "nearsightedness" has been obtained, and the possibility of $O(n)$ methods firmly established. ${ }^{8}$

\footnotetext{
${ }^{8}$ Heuristics relating the "nearsightedness range of electronic matter" and the linear complexity of the divide-and-conquer method of Yang [138], essentially a domain decomposition approach to DFT, were given by Kohn himself; see, e.g., [75, 105].
}

Having thus achieved our main purpose, the question remains whether our estimates can be of practical use in the design of $O(n)$ algorithms. As shown in section 6, having estimated the rate of decay in the density matrix $P$ allows one to prescribe a priori a sparsity pattern for the computed approximation $\tilde{P}$ to $P$. Estimating an "envelope" for the nonnegligible entries in $P$ means that one can estimate beforehand the storage requirements and set up static data structures for the computation of the approximate density matrix $\tilde{P}$. An added advantage is the possibility of using the prescribed sparsity pattern to develop efficient parallel algorithms; it is well known that adaptive computations, in which the sparsity pattern is determined "on the fly," may lead to load imbalances and loss of parallel efficiency due to the need for large amounts of communication and unpredictable memory accesses. This is completely analogous to prescribing a sparsity pattern vs. using an adaptive one when computing sparse approximate inverses for use as preconditioners when solving linear systems; see [10].

Most of the $O(n)$ algorithms currently in use consist of iterative schemes producing increasingly accurate approximations to the density matrix. These approximations may correspond to successive terms in an expansion of $P$ with respect to a prescribed basis [54, 80, 81], or they may be the result of a gradient or descent method in density matrix minimization approaches [23, 24, 79, 93]. Closely related methods include purification and algorithms based on approximating the sign function [95]; we refer the reader again to [20, 97, 113] for recent surveys on state-of-the-art linear scaling methods for electronic structure. Most of these algorithms construct a sequence of approximations
$$
P^{(0)}, P^{(1)}, \ldots, P^{(k)}, \ldots
$$
which, under appropriate conditions, converge to $P$. Each iterate is obtained from the preceding one by some matrix-matrix multiplication, or powering, scheme; each step introduces new nonzeros (fill-in), and the matrices $P^{(k)}$ become increasingly dense. The exponential decay property, however, implies that most of these nonzeros will be negligible, with only $O(n)$ of them being above any prescribed threshold $\delta>$ 0 . Clearly, knowing a priori the location of the nonnegligible entries in $P$ can be used to drastically reduce the computational burden and to achieve linear scaling, since only those entries need to be computed. Negligible entries that fall within the prescribed sparsity pattern may be removed using a drop tolerance; this strategy further decreases storage and arithmetic complexity, but its implementation demands the use of dynamic data structures.

An illustration of this use of the decay estimates can be found, for instance, in [14], where a Chebyshev expansion of the Fermi-Dirac function $f_{F D}(H)$ was used to approximate the density matrix at finite temperatures. Given a prescribed error tolerance, exponential decay bounds were applied to the Fermi-Dirac function to determine the truncation bandwidth needed to satisfy the required approximation error. When computing the polynomial $p_{k}(H) \approx f_{F D}(H)$ using the Chebyshev expansion, only entries within the prescribed bandwidth were retained. Combined with an estimate of the approximation error obtained by monitoring the magnitude of the coefficients in the Chebyshev expansion, this approach worked well for some simple 1D model problems resulting in linear scaling computations. A related approach, based on qualitative decay estimates for the density matrix, was used in [4], whose authors present computational results for a variety of 1D and 2D systems including insulators at zero temperature and metals at finite temperature; see further [80].

Unfortunately, the practical usefulness of our bounds for more realistic calculations is limited. The bounds are generally pessimistic and tend to be overly conserva- tive, especially for the case of zero or low temperatures. This is to be expected, since the bounds were obtained by estimating the degree of a polynomial approximation to the Fermi-Dirac matrix function needed to satisfy a prescribed error tolerance. These bounds tend to be rather pessimistic because they do not take into account the possibility of numerical cancellation when evaluating the matrix polynomial. For instance, the bounds must apply in the worst-case scenario where the Hamiltonian has nonnegative entries and the approximating polynomial has nonnegative coefficients. Moreover, the bounds do not take into account the size of the entries in the Hamiltonian, particularly the fact that the nonzeros within the band (or sparsity pattern) are not of uniform size but may be spread out over several orders of magnitude. It should be emphasized that the presence of a gap is only a sufficient condition for localization of the density matrix, not a necessary one: it has been pointed out, for example, in [90], that disordered systems may exhibit strong localization even in the absence of a well-defined gap. This is the case, for instance, of the Anderson model of localization in condensed matter physics [2]. Obviously, our approach is unable to account for such phenomena in the zero-temperature case. The theory reviewed in this paper is primarily a qualitative one; nevertheless, it captures many of the features of actual physical systems, like the asymptotic dependence of the decay rate on the gap size or on the electronic temperature.

A natural question is whether the bounds can be improved to the point where they can be used to obtain practical estimates of the entries in the density matrix. In order to achieve this, additional assumptions on the Hamiltonians would be needed, making the theory less general. In other words, the price we pay for the generality of our theory is that we get pessimistic bounds. Recall that for a given sparsity pattern in the normalized Hamiltonians $H_{n}$ our decay bounds depend on just one essential parameter, the gap $\gamma$. Our bounds are the same no matter what the eigenvalue distribution is to the left of the highest occupied level, $\varepsilon_{n_{e}}$, and to the right of the lowest unoccupied one, $\varepsilon_{n_{e}+1}$. If more spectral information were at hand, the bounds could be improved. The situation is very similar to that arising in the derivation of error bounds for the convergence of Krylov methods, such as the conjugate gradient (CG) method for solving symmetric positive definite (SPD) linear systems $A x=b$; see, e.g., [57, Theorem 10.2.6]. Bounds based on the spectral condition number $\kappa_{2}(A)$ alone, while sharp, do not in general capture the actual convergence behavior of CG. They represent the worst-case behavior, which is rarely observed in practice. Much more accurate bounds can be obtained by making assumptions on the distribution of the eigenvalues of $A$. For instance, if $A$ has only $k$ distinct eigenvalues, then the CG method converges (in exact arithmetic) to the solution $x_{*}=A^{-1} b$ in at most $k$ steps. Similarly, suppose the Hamiltonian $H_{n}$ has only $k<n$ distinct eigenvalues (with $\mu$ not one of them), and that the multiplicities of the eigenvalues to the left of $\mu$ add up to $n_{e}$, the number of electrons. Then there is a polynomial $p_{k}(\lambda)$ of degree at most $k-1$ such that $p_{k}\left(H_{n}\right)=P_{n}$, the density matrix. This is just the interpolation polynomial that takes the value 1 on the eigenvalues to the left of $\mu$, and zero on the eigenvalues to the right of $\mu$. This polynomial "approximation" is actually exact. If $k \ll n$ and is independent of $n$, then $P_{n}$ will be a matrix with $O(n)$ nonzero entries; moreover, the sparsity pattern of $P_{n}$ can be determined a priori from the graph structure of $H_{n}$. Another situation is that in which the eigenvalues of $H_{n}$ fall in a small number $k$ of narrow bands, or tight clusters, with the rightmost band to the left of $\mu$ well separated from the leftmost band to the right of $\mu$. In this case
we can find again a low-degree polynomial $p_{k}(\lambda)$ with $p_{k}\left(H_{n}\right) \approx P_{n}$, and improved bounds can be obtained.

The problem, of course, is that these are rather special eigenvalue distributions, and it is difficult to know a priori whether or not such conditions hold.

Another practical issue that should be at least briefly mentioned is the fact that our bounds assume knowledge of lower and upper bounds on the spectra of the Hamiltonians $H_{n}$, as well as estimates for the size and location of the spectral gap (this is also needed in order to determine the Fermi level $\mu$ ). These issues have received a great deal of attention in the literature, and here we limit ourselves to observing that $O(n)$ procedures exist to obtain sufficiently accurate estimates of these quantities; see, e.g., [53].
9. Transformation to an Orthonormal Basis. In this section we discuss the transformation of a Hamiltonian from a nonorthogonal to an orthogonal basis. The main point is that while this transformation results in matrices with less sparsity, the transformed matrices retain the decay properties of the original matrices, only with (possibly) different constants. What is important, from the point of view of asymptotic complexity, is that the rate of decay remains independent of system size.

We begin with a discussion of decay in the inverse of the overlap matrix. To this end, consider a sequence $\left\{S_{n}\right\}$ of overlap matrices of size $n=n_{b} \cdot n_{e}$, with $n_{b}$ constant and $n_{e}$ increasing to infinity. We make the following assumptions:
1. Each $S_{n}$ is a banded SPD matrix with unit diagonal entries and with bandwidth uniformly bounded with respect to $n$.
2. The spectral condition number (ratio of the largest to the smallest eigenvalue) of each $S_{n}, \kappa_{2}\left(S_{n}\right)$, is uniformly bounded with respect to $n$. Because of assumption 1, this is equivalent to requiring that the smallest eigenvalue of $S_{n}$ remain bounded away from zero for all $n$.
As always in this paper, the bandedness assumption in item 1 is not essential and can be replaced by the weaker hypothesis that each $S_{n}$ is sparse and that the corresponding graphs $\left\{G_{n}\right\}$ have bounded maximal degree with respect to $n$. Actually, it would be enough to require that the sequence $\left\{S_{n}\right\}$ have the exponential decay property relative to a sequence of graphs $\left\{G_{n}\right\}$ of bounded maximal degree. In order to simplify the discussion, and also in view of the fact that overlap matrices usually exhibit exponential or even superexponential decay, we assume from the outset that each $S_{n}$ has already been truncated to a sparse (or banded) matrix. Again, this is for notational convenience only, and it is straightforward to modify the following arguments to account for the more general case. On the other hand, the assumption on condition numbers in item 2 is essential and cannot be weakened.

Remark 9.1. We note that assumption 2 above is analogous to the condition that the sequence of Hamiltonians $\left\{H_{n}\right\}$ have spectral gap bounded below uniformly in $n$; while this condition ensures (as we have shown) the exponential decay property in the associated spectral projectors $P_{n}$, assumption 2 above ensures exponential decay in the inverses (or inverse factors) of the overlap matrices. Both conditions amount to requiring that the corresponding problems be uniformly well-conditioned in $n$. The difference is that the decay on the spectral projectors depends on the spectral gap of the Hamiltonians and therefore on the nature of the system under study (i.e., insulator vs. metallic system), whereas the sparsity and spectral properties of the overlap matrices depend on other features of the system, mainly the interatomic distances.

In the following we shall need some basic results on the decay of the inverses [34], inverse Cholesky factors [15], and inverse square roots (Löwdin factors) [12] of banded SPD matrices; see also [71].

Let $A$ be SPD and $m$-banded, and let $a$ and $b$ denote the smallest and largest eigenvalues of $A$, respectively. Write $\kappa$ for the spectral condition number $\kappa_{2}(A)$ of $A$ (hence, $\kappa=b / a$ ). Define
$$
q:=\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1} \quad \text { and } \quad \lambda:=q^{1 / m}
$$

Furthermore, let $K_{0}:=(1+\sqrt{\kappa})^{2} /(2 b)$. In [34], Demko, Moss, and Smith obtained the following bound on the entries of $A^{-1}$ :
$$
\begin{equation*}
\left|\left[A^{-1}\right]_{i j}\right| \leq K \lambda^{|i-j|}, \quad 1 \leq i, j \leq n, \tag{9.1}
\end{equation*}
$$
where $K:=\max \left\{a^{-1}, K_{0}\right\}$. Note that the bound (9.1) "blows up" as $\kappa \rightarrow \infty$, as one would expect.

As shown in [15], the decay bound (9.1) and the bandedness assumption on $A$ imply a similar decay bound on the inverse Cholesky factor $Z=R^{-1}=L^{-T}$, where $A=R^{T} R=L L^{T}$ with $R$ upper triangular ( $L$ lower triangular). Assuming that $A$ has been scaled so that $\max _{1 \leq i \leq n} A_{i i}=1$ (which is automatically true if $A$ is an overlap matrix corresponding to a set of normalized basis functions), we have
$$
\begin{equation*}
\left|Z_{i j}\right| \leq K_{1} \lambda^{j-i}, \quad j \geq i \tag{9.2}
\end{equation*}
$$
with $K_{1}=K \frac{1-\lambda^{m}}{1-\lambda}$; here $K, \lambda$ are the same as before. We further note that while $K_{1}>K$, for some classes of matrices it is possible to show that the actual magnitude of the ( $i, j$ ) entry of $Z$ (as opposed to the bound (9.2)) is actually less than the magnitude of the corresponding entry of $A^{-1}$. This is true, for instance, for an irreducible $M$-matrix; see [15].

Finally, let us consider the inverse square root, $A^{-1 / 2}$. In [12] the following bound is established:
$$
\begin{equation*}
\left|\left[A^{-1 / 2}\right]_{i j}\right| \leq K_{2} \lambda^{|i-j|}, \quad 1 \leq i, j \leq n . \tag{9.3}
\end{equation*}
$$

Here $K_{2}$ depends again on the extreme eigenvalues $a$ and $b$ of $A$, whereas $\lambda=q^{1 / m}$, where now $q$ is any number satisfying the inequalities
$$
\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}<q<1
$$

As before, the bound (9.3) blows up as $\kappa \rightarrow \infty$, as one would expect.
Introducing the positive scalar $\alpha=-\ln \lambda$, we can rewrite all these bounds in the form
$$
\left|B_{i j}\right| \leq K \mathrm{e}^{-\alpha|i-j|}, \quad 1 \leq i, j \leq n
$$
for the appropriate matrix $B$ and suitable constants $K$ and $\alpha>0$.
Now let $\left\{S_{n}\right\}$ be a sequence of $n \times n$ overlap matrices, where $n=n_{b} \cdot n_{e}$ with $n_{b}$ fixed and $n_{e} \rightarrow \infty$. Assuming that the matrices $S_{n}$ satisfy assumptions 1-2 above, then their inverses satisfy the uniform exponential decay bounds (9.1), with $K$ and $\lambda$ constant and independent of $n$. Hence, as discussed in section 6 , for any given $\epsilon>0$ there exists an integer $\bar{m}$ independent of $n$ such that each matrix $S_{n}$ in the sequence can be approximated, in norm, by an $\bar{m}$-banded matrix with an error less than $\epsilon$. As usual, this result can be extended from the banded case to the sparse case,
assuming that the corresponding graphs $G_{n}$ have bounded maximal degree as $n \rightarrow \infty$. Moreover, given assumptions $1-2$ above, the inverse Cholesky factors $Z_{n}$ satisfy a uniform (in $n$ ) exponential decay bound of the type (9.2), and therefore uniform approximation with banded triangular matrices is possible. Again, generalization to more general sparsity patterns is possible, provided the usual assumption on the maximum degree of the corresponding graphs $G_{n}$ holds. Similarly, under the same conditions we obtain a uniform rate of exponential decay for the entries of the inverse square roots $S_{n}^{-1 / 2}$, with a corresponding result on the existence of a banded (or sparse) approximation.

Let us now consider the sequence of transformed Hamiltonians, $\tilde{H}_{n}=Z_{n}^{T} H_{n} Z_{n}$. Here $Z_{n}$ denotes either the inverse Cholesky factor or the inverse square root of the corresponding overlap matrix $S_{n}$. Assuming that the sequence $\left\{H_{n}\right\}$ satisfies the offdiagonal exponential decay property and that $\left\{S_{n}\right\}$ satisfies assumptions 1-2 above, it follows from the decay properties of the matrix sequence $\left\{Z_{n}\right\}$ that the sequence $\left\{\tilde{H}_{n}\right\}$ also enjoys off-diagonal exponential decay. This is a straightforward consequence of the following result, which is adapted from a similar one for infinite matrices due to Jaffard [71, Proposition 1].

Theorem 9.2. Consider two sequences $\left\{A_{n}\right\}$ and $\left\{B_{n}\right\}$ of $n \times n$ matrices (where $n \rightarrow \infty$ ) whose entries satisfy
$$
\left|\left[A_{n}\right]_{i j}\right| \leq c_{1} \mathrm{e}^{-\alpha|i-j|} \quad \text { and } \quad\left|\left[B_{n}\right]_{i j}\right| \leq c_{2} \mathrm{e}^{-\alpha|i-j|}, \quad 1 \leq i, j \leq n
$$
where $c_{1}, c_{2}$, and $\alpha>0$ are independent of $n$. Then the sequence $\left\{C_{n}\right\}$, where $C_{n}=A_{n} B_{n}$, satisfies a similar bound:
$$
\begin{equation*}
\left|\left[C_{n}\right]_{i j}\right| \leq c \mathrm{e}^{-\alpha^{\prime}|i-j|}, \quad 1 \leq i, j \leq n \tag{9.4}
\end{equation*}
$$
for any $0<\alpha^{\prime}<\alpha$, with $c$ independent of $n$.
Proof. First note that the entries of each $A_{n}$ clearly satisfy
$$
\left|\left[A_{n}\right]_{i j}\right| \leq c_{1} \mathrm{e}^{-\alpha^{\prime}|i-j|} \quad \text { for any } \quad \alpha^{\prime}<\alpha
$$

Let $\omega=\alpha-\alpha^{\prime}$. Then $\omega>0$ and the entries $\left[C_{n}\right]_{i j}$ of $C_{n}=A_{n} B_{n}$ satisfy
$$
\left|\left[C_{n}\right]_{i j}\right| \leq \sum_{k=1}^{n}\left|\left[A_{n}\right]_{i k}\right|\left|\left[B_{n}\right]_{k j}\right| \leq c_{1} c_{2}\left(\sum_{k=1}^{n} \mathrm{e}^{-\omega|k-j|}\right) \mathrm{e}^{-\alpha^{\prime}|i-j|}
$$

To complete the proof just observe that for any $j$,
$$
\sum_{k=1}^{n} \mathrm{e}^{-\omega|k-j|}=\sum_{k=0}^{j-1} \mathrm{e}^{-\omega k}+\sum_{k=1}^{n-j} \mathrm{e}^{-\omega k}<\sum_{k=0}^{\infty} \mathrm{e}^{-\omega k}+\sum_{k=1}^{\infty} \mathrm{e}^{-\omega k}=\frac{1+\mathrm{e}^{-\omega}}{1-\mathrm{e}^{-\omega}} .
$$

Since the last term is independent of $n$, the entries of $C_{n}$ satisfy (9.4) with a constant $c$ that is also independent of $n$.

The foregoing result can obviously be extended to the product of three matrices. Thus, the entries of the matrix sequence $\left\{\tilde{H}_{n}\right\}$, where $\tilde{H}_{n}=Z_{n}^{T} H_{n} Z_{n}$, enjoy the exponential off-diagonal decay property
$$
\left|\left[\tilde{H}_{n}\right]_{i j}\right| \leq c \mathrm{e}^{-\alpha|i-j|}, \quad 1 \leq i, j \leq n
$$
for suitable constants $c$ and $\alpha>0$.

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/1bc7e2d7-f39d-466c-801c-6b7da6d033c3-47.jpg?height=1204&width=584&top_left_y=337&top_left_x=575}
\captionsetup{labelformat=empty}
\caption{Fig. 9. 1 Magnitude of the entries in the Hamiltonian for the $\mathrm{C}_{52} \mathrm{H}_{106}$ linear alkane. Top: Nonorthogonal (Gaussian-type orbital) basis. Bottom: Orthogonal basis. White: $<10^{-8}$; yellow: $10^{-8}-10^{-6}$; green: $10^{-6}-10^{-4}$; blue: $10^{-4}-10^{-2}$; black: $>10^{-2}$. Note: nz refers to the number of "black" entries.}
\end{figure}

Alternatively, one could first approximate $H_{n}$ and $Z_{n}$ with banded matrices $\bar{H}_{n}$ and $\bar{Z}_{n}$ and then define the (approximate) transformed Hamiltonian as $\tilde{H}_{n}:= \bar{Z}_{n}^{T} \bar{H}_{n} \bar{Z}_{n}$, possibly subject to further truncation. Using the fact that both $H_{n}$ and $Z_{n}$ have 2 -norm bounded independently of $n$, it is easy to show that the final approximation error can be reduced below any prescribed tolerance by reducing the error in $\bar{H}_{n}$ and $\bar{Z}_{n}$. Hence, with either approach, the transformed Hamiltonians $\tilde{H}_{n}$ can be approximated uniformly in $n$ within a prescribed error by banded matrices of constant bandwidth, just like the original ("nonorthogonal") Hamiltonians. While the bandwidth of the approximations will be larger than for the original Hamiltonians, the truncated matrices retain a good deal of sparsity and asymptotically contain $O(n)$ nonzeros. Hence, we have a justification of the statement (see section 1) that in our theory we can assume from the outset that the basis set $\left\{\phi_{i}\right\}_{i=1}^{n}$ is orthonormal.

In Figure 9.1 we show the Hamiltonian $H$ for the already mentioned linear alkane $\mathrm{C}_{52} \mathrm{H}_{106}$ (see section 8.2) discretized in a Gaussian-type orbital basis (top) and the "orthogonalized" Hamiltonian $\tilde{H}=\bar{Z}^{T} \bar{H} \bar{Z}$ (bottom). This figure shows that while
the transformation to the orthogonal basis alters the magnitude of the entries in the Hamiltonian, the bandwidth of $\tilde{H}$ (truncated to a tolerance of $10^{-8}$ ) is only slightly wider than that of $H$. In this case the overlap matrix $S$ is well-conditioned, hence the entries of $Z$ exhibit fast decay. An ill-conditioned overlap matrix would lead to a less sparse transformed Hamiltonian $\tilde{H}$.

As usual, the bandedness assumption was made for simplicity of exposition only; similar bounds can be obtained for more general sparsity patterns, assuming the matrices $H_{n}$ and $S_{n}$ have the exponential decay property relative to a sequence $\left\{G_{n}\right\}$ of graphs having maximal degree uniformly bounded with respect to $n$.

It is important to emphasize that, in practice, the explicit formation of $\tilde{H}_{n}$ from $H_{n}$ and $Z_{n}$ is not needed and is never carried out. Indeed, in all algorithms for electronic structure computation the basic matrix operations are matrix-matrix and matrix-vector products, which can be performed without explicit transformation of the Hamiltonian to an orthonormal basis. On the other hand, for the study of the decay properties it is convenient to assume that all the relevant matrices are explicitly given in an orthogonal representation.

One last issue to be addressed is whether the transformation to an orthonormal basis should be effected via the inverse Cholesky factor or via the Löwdin (inverse square root) factor of the overlap matrix. Comparing the decay bounds for the two factors suggests that the inverse Cholesky factor should be preferred (smaller $\alpha$ ). Also note that the inverse Cholesky factor is triangular, and its sparsity can be increased by suitable reorderings of the overlap matrix. The choice of ordering may also be influenced by the computer architecture used. We refer the reader to [30] for the use of bandwidth-reducing orderings like reverse Cuthill-McKee, and to [25] for the use of space-filling curve orderings like the 3D Hilbert curve to improve load balancing and data locality on parallel architectures. In contrast, the Löwdin factor is a full symmetric matrix, regardless of the ordering. On the other hand, the multiplicative constant $c$ is generally smaller for the Löwdin factor. Closer examination of a few examples suggests that in practice there is no great difference in the actual decay behavior of these two factors. However, approximating $S_{n}^{-1 / 2}$ is generally more expensive and considerably more involved than approximating the inverse Cholesky factor. For the latter, the AINV algorithm [13] and its variants [24, 110, 136] are quite efficient and have been successfully used in various quantum chemistry codes. For other $O(n)$ algorithms for transformation to an orthonormal basis, see [72, 100, 122]. In all these algorithms, sparsity is preserved by dropping small entries in the course of the computation. Explicit decay bounds for the $Z_{n}$ factors could be used, in principle, to establish a priori which matrix elements not to compute, thus reducing the amount of overhead. Notice, however, that even if asymptotically bounded, the condition numbers $\kappa_{2}\left(S_{n}\right)$ can be fairly large, leading to rather pessimistic decay estimates. This is again perfectly analogous to the situation with the condition-number-based error bounds for the CG method applied to a linear system $A x=b$. Indeed, both the CG error bounds and the estimates (9.1) are obtained using Chebyshev polynomial approximation for the function $f(\lambda)=\lambda^{-1}$.
10. The Vanishing Gap Case. In this section we discuss the case of a sequence $\left\{H_{n}\right\}$ of bounded, finite-range Hamiltonians for which the spectral gap around the Fermi level $\mu$ vanishes as $n \rightarrow \infty$. Recall that this means that $\inf _{n} \gamma_{n}=0$, where $\gamma_{n}:=\varepsilon_{n_{e}+1}^{(n)}-\varepsilon_{n_{e}}^{(n)}$ is the HOMO-LUMO gap for the $n$th Hamiltonian; it is assumed here that $\varepsilon_{n_{e}}^{(n)}<\mu<\varepsilon_{n_{e}+1}^{(n)}$ for all $n=n_{b} \cdot n_{e}$. The reciprocal $\gamma_{n}^{-1}$ of the gap can be interpreted as the condition number of the problem [109], so a vanishing spectral
gap means that the conditioning deteriorates as $n_{e} \rightarrow \infty$ and the problem becomes increasingly difficult.

As already mentioned, in the zero-temperature limit our decay bounds blow up and therefore lose all meaning as $\gamma_{n} \rightarrow 0$. On the other hand, we know a priori that some type of decay should be present, in view of the results in section 7. A general treatment of the vanishing gap case appears to be rather difficult, for the main reason that in the limit as $\beta \rightarrow \infty$ the Fermi-Dirac approximation to the Heaviside function becomes discontinuous, and therefore we can no longer make use of tools from classical approximation theory for analytic functions. Similarly, in the vanishing gap case the decay bounds (8.27) based on the resolvent estimates (8.24) break down since $c \rightarrow \infty$ and $\lambda \rightarrow 1$ in (8.26).

Rather than attacking the problem in general, in this section we give a complete analysis of what is perhaps the simplest nontrivial example of a sequence $\left\{H_{n}\right\}$ with vanishing gap. While this is only a special case, this example captures some of the essential features of the "metallic" case, such as the rather slow off-diagonal decay of the entries of the density matrix. The simple model studied in this section may appear at first sight to be too simple and unrealistic to yield any useful information about actual physical systems. However, calculation of the density matrix at zero temperature on a system composed of 500 Al atoms reported in [140] reveals a decay behavior which is essentially identical to that obtained analytically for a free electron gas, a model very close to ours (which is essentially a discrete variant of the one in [140]). We believe that our analysis will shed some light on more general situations in which a slowly decaying density matrix occurs.

We begin by considering the infinite tridiagonal Toeplitz matrix
$$
H=\left(\begin{array}{cccccc}
0 & \frac{1}{2} & & & &  \tag{10.1}\\
\frac{1}{2} & 0 & \frac{1}{2} & & & \\
& \ddots & \ddots & \ddots & & \\
& & \frac{1}{2} & 0 & \frac{1}{2} & \\
& & & \ddots & \ddots & \ddots
\end{array}\right)
$$
which defines a bounded, banded, self-adjoint operator on $\ell^{2}$. The graph of this matrix is just a (semi-infinite) path. The operator can be interpreted as an averaging operator or as a centered second-difference operator with a zero Dirichlet condition at one end, shifted and scaled so as to have spectrum contained in $[-1,1]$. From a physical standpoint, $H$ is the shifted and scaled discrete one-electron Hamiltonian, where the electron is constrained to the half-line $[0, \infty)$.

For $n$ even ( $n=2 \cdot n_{e}$, with $n_{e} \in \mathbb{N}$ ) consider the $n$-dimensional approximation
$$
H_{n}=\left(\begin{array}{ccccc}
0 & \frac{1}{2} & & &  \tag{10.2}\\
\frac{1}{2} & 0 & \frac{1}{2} & & \\
& \ddots & \ddots & \ddots & \\
& & \frac{1}{2} & 0 & \frac{1}{2} \\
& & & \frac{1}{2} & 0
\end{array}\right)
$$

This corresponds to truncating the semi-infinite path and imposing zero Dirichlet conditions at both ends. Now let $\left\{e_{1}, e_{2} \ldots\right\}$ denote the standard basis of $\ell^{2}$, and let $\hat{I}$ denote the identity operator restricted to the subspace of $\ell^{2}$ spanned by $e_{n+1}, e_{n+2}, \ldots$.

Letting
$$
H_{(n)}:=\left(\begin{array}{cc}
H_{n} & 0 \\
0 & \hat{I}
\end{array}\right)
$$
the sequence $\left\{H_{(n)}\right\}$ is now a sequence of bounded self-adjoint linear operators on $\ell^{2}$ that converges strongly to $H$. Note that $\sigma\left(H_{n}\right) \subset[-1,1]$ for all $n$; also, $0 \notin \sigma\left(H_{n}\right)$ for all even $n$. It is easy to see that half of the eigenvalues of $H_{n}$ lie in $[-1,0)$ and the other half in $(0,1]$. We set $\mu=0$ and we label as "occupied" the states corresponding to negative eigenvalues. The spectral gap of each $H_{n}$ is then $\varepsilon_{n / 2+1}^{(n)}-\varepsilon_{n / 2}^{(n)}$.

The eigenvalues and eigenvectors of $H_{n}$ are known explicitly [35, Lemma 6.1]. Indeed, the eigenvalues, in descending order, are given by $\varepsilon_{k}^{(n)}=\cos \left(\frac{k \pi}{n+1}\right)$ (with $1 \leq k \leq n)$ and the corresponding normalized eigenvectors are given by $v_{k}^{(n)}=\left(v_{k}^{(n)}(j)\right)$ with entries
$$
v_{k}^{(n)}(j)=\sqrt{\frac{2}{n+1}} \sin \left(\frac{j k \pi}{n+1}\right), \quad 1 \leq j \leq n
$$

Note that the eigenvalues are symmetric with respect to the origin, and that the spectral gap at 0 vanishes, since $\varepsilon_{n / 2+1}^{(n)}=-\varepsilon_{n / 2}^{(n)} \rightarrow 0$ as $n \rightarrow \infty$. We also point to the well-known fact that the eigenvectors of this operator are strongly delocalized. Nevertheless, as we will see, some localization (decay) is present in the density matrix, owing to cancellation (i.e., destructive interference).

Now let $P_{n}$ be the zero-temperature density matrix associated with $H_{n}$, i.e., the spectral projector onto the subspace of $\mathbb{C}^{n}$ spanned by the eigenvectors of $H_{n}$ associated with the lowest $n_{e}$ eigenvalues (the occupied subspace). We extend $P_{n}$ to a projector acting on $\ell^{2}$ by embedding $P_{n}$ into an infinite matrix $P_{(n)}$ as follows:
$$
P_{(n)}:=\left(\begin{array}{cc}
P_{n} & 0 \\
0 & 0
\end{array}\right) .
$$

Note that $P_{(n)}$ is just the orthogonal projector onto the subspace of $\ell^{2}$ spanned by the eigenvectors of $H_{(n)}$ associated with eigenvalues in the interval $[-1,0)$. Moreover, $\operatorname{Tr}\left(P_{(n)}\right)=\operatorname{Tr}\left(P_{n}\right)=\operatorname{rank}\left(P_{n}\right)=\frac{n}{2}=n_{e}$. The limiting behavior of the sequence $\left\{P_{(n)}\right\}$ (hence, of $\left\{P_{n}\right\}$ ) is completely described by the following result.

Theorem 10.1. Let $H, H_{n}$, and $P_{(n)}$ be as described above. Then the following hold:
(i) $H$ has purely absolutely continuous spectrum, ${ }^{9}$ given by the interval $[-1,1]$. In particular, $H$ has no eigenvalues.
(ii) The union of the spectra of the $n$-dimensional sections $H_{n}$ of $H$ is everywhere dense in $\sigma(H)=[-1,1]$. In other words, every point in $[-1,1]$ is the limit of a sequence of the form $\left\{\varepsilon_{k}^{(n)}\right\}$ for $n \rightarrow \infty$, where $\varepsilon_{k}^{(n)} \in \sigma\left(H_{n}\right)$ and $k=k(n)$.
(iii) The sequence $\left\{H_{n}\right\}$ has vanishing gap: $\inf _{n} \gamma_{n}=0$.
(iv) The spectral projectors $P_{(n)}$ converge strongly to $P=h(H)$, where $h(x)= \chi_{[-1,0)}(x)$, the characteristic function of the interval $[-1,0)$.
(v) $P$ is the orthogonal projector onto an infinite-dimensional subspace of $\ell^{2}$.

\footnotetext{
${ }^{9}$ The absolutely continuous spectrum of a self-adjoint linear operator $H$ on a Hilbert space $\mathscr{H}$ is the spectrum of the restriction of $H$ to the subspace $\mathscr{H}_{\mathrm{ac}} \subseteq \mathscr{H}$ of vectors $\psi$ whose spectral measures $\mu_{\psi}$ are absolutely continuous with respect to the Lebesgue measure. For details, see [107, pp. 224-231].
}

Proof. Statements (i)-(ii) are straightforward consequences of classical results on the asymptotic eigenvalue distribution of Toeplitz matrices, while (iv)-(v) follow from general results in spectral theory. Statement (iii) was already noted (the eigenvalues of $H_{n}$ are explicitly known) and it also follows from (i)-(ii). In more detail, statement (i) is a special case of Rosenblum's theorem on the spectra of banded infinite Toeplitz matrices; see [108] or [17, Theorem 1.31]. For the facts that the spectrum of $H$ coincides with the interval $[-1,1]$ and that the finite section eigenvalues $\varepsilon_{k}^{(n)}$ are dense in $\sigma(H)=[-1,1]$ (statement (ii)), see the paper by Hartman and Wintner [60] or the book by Grenander and Szegö [58, Chapter 5]. Statement (iv) can be proved as follows. For a linear operator $A$ on $\ell^{2}$, write $R_{\lambda}(A)=(A-\lambda I)^{-1}$, with $\lambda \notin \sigma(A)$. A sequence $\left\{A_{n}\right\}$ of self-adjoint (Hermitian) operators is said to converge in the strong resolvent sense to $A$ if $R_{\lambda}\left(A_{n}\right) \longrightarrow R_{\lambda}(A)$ strongly for all $\lambda \in \mathbb{C}$ with $\operatorname{Re} \lambda \neq 0$, that is,
$$
\lim _{n \rightarrow \infty}\left\|R_{\lambda}\left(A_{n}\right) x-R_{\lambda}(A) x\right\|=0 \quad \text { for all } \quad x \in \ell^{2} .
$$

It is easy to check, using, for instance, the results in [18, Chapter 2], that the sequence $\left\{H_{n}\right\}$ converges in the strong resolvent sense to $H$. Statement (iv) (as well as (ii)) now follows from [107, Theorem VIII.24]. The fact (v) that $P=h(H)$ is an orthogonal projector onto an infinite-dimensional subspace of $\ell^{2}$ follows from the fact that $\mu=0$ is not an eigenvalue of $H$ (because of (i)) and from the spectral theorem for self-adjoint operators in Hilbert space; see, e.g., [107, Chapter VII] or [115, Chapter 12].

The foregoing result implies that the Toeplitz matrix sequence $\left\{H_{n}\right\}$ given by (10.2) exhibits some of the key features of the discrete Hamiltonians describing metallic systems, in particular, the vanishing gap property and the fact that the eigenvalues tend to fill the entire energy spectrum. The sequence $\left\{H_{n}\right\}$ can be thought of as a 1D "toy model" that can be solved analytically to gain some insight into the decay properties of the density matrix of such systems. Indeed, from the knowledge of the eigenvectors of $H_{n}$ we can write down the spectral projector corresponding to the lowest $n_{e}=n / 2$ eigenvalues explicitly. Recalling that the eigenvalues $\varepsilon_{k}^{(n)}$ are given in descending order, it is convenient to compute $P_{n}$ as the projector onto the orthogonal complement of the subspace spanned by the eigenvectors corresponding to the $n / 2$ largest eigenvalues:
$$
P_{n}=I_{n}-\sum_{k=1}^{n_{e}} v_{k}^{(n)}\left(v_{k}^{(n)}\right)^{T}
$$

The ( $i, j$ ) entry of $P_{n}$ is therefore given by
$$
\left[P_{n}\right]_{i j}=e_{i}^{T} P_{n} e_{j}=\delta_{i j}-\frac{2}{n+1} \sum_{k=1}^{n_{e}} \sin \left(\frac{i k \pi}{n+1}\right) \sin \left(\frac{j k \pi}{n+1}\right) .
$$

For $i=j$, we find
$$
\begin{equation*}
\left[P_{n}\right]_{i i}=1-\frac{2}{n+1} \sum_{k=1}^{n_{e}} \sin ^{2}\left(\frac{i k \pi}{n+1}\right)=\frac{1}{2} \quad \text { for all } i=1, \ldots, n \text { and for all } n . \tag{10.3}
\end{equation*}
$$

Hence, for this system the charge density $P_{i i}$ is constant and the system essentially behaves like a noninteracting electron gas; see, for example, [50]. We note in passing that this example confirms that the bound (7.2) is sharp, since equality is attained for this particular projector. Moreover, the trigonometric identity
$$
\begin{equation*}
\sin \theta \sin \phi=-\frac{1}{2}[\cos (\theta+\phi)-\cos (\theta-\phi)] \tag{10.4}
\end{equation*}
$$
implies for all $i, j=1, \ldots, n$ that
$$
\begin{equation*}
\left[P_{n}\right]_{i j}=\frac{1}{n+1} \sum_{k=1}^{n_{e}}\left[\cos \left(\frac{(i+j) k \pi}{n+1}\right)-\cos \left(\frac{(i-j) k \pi}{n+1}\right)\right] . \tag{10.5}
\end{equation*}
$$

From (10.5) it immediately follows, for all $i$ and for all $n$, that
$$
\begin{equation*}
\left[P_{n}\right]_{i, i+2 l}=0, \quad l=1,2, \ldots \tag{10.6}
\end{equation*}
$$

Since (10.3) and (10.6) hold for all $n$, they also hold in the limit as $n \rightarrow \infty$. Hence, the strong limit $P$ of the sequence of projectors $\left\{P_{(n)}\right\}$ satisfies $P_{i i}=1 / 2$ and $P_{i, j}=0$ for all $j=i+2 l$, where $i, l=1,2, \ldots$. To determine the remaining off-diagonal entries $P_{i j}$ (with $j \neq i$ and $j \neq i+2 l$ ) we directly compute the limit of $\left[P_{n}\right]_{i j}$ as $n \rightarrow \infty$, as follows. Observe that using the substitution $x=k /(n+1)$ and taking the limit as $n \rightarrow \infty$ in (10.5), we obtain for all $i \geq 1$ and for all $j \neq i+2 l(l=0,1, \ldots)$
$$
\begin{align*}
P_{i j} & =\int_{0}^{\frac{1}{2}} \cos [(i+j) \pi x] d x-\int_{0}^{\frac{1}{2}} \cos [(i-j) \pi x] d x \\
& =\frac{1}{\pi}\left[\frac{(-1)^{\frac{i+j-1}{2}}}{i+j}+\frac{(-1)^{\frac{i-j+1}{2}}}{i-j}\right] \tag{10.7}
\end{align*}
$$

It follows from (10.7) that $\left|P_{i j}\right|$ is bounded by a quantity that decays only linearly in the distance from the main diagonal. As a result, $O(n)$ approximation of $P_{n}$ for large $n$ involves a huge prefactor. Therefore, from this very simple example we can gain some insight into the vanishing gap case. The analytical results obtained show that the density matrix can exhibit rather slow decay, confirming the well-known fact that $O(n)$ approximations pose a formidable challenge in the vanishing gap case.

The 2D case is easily handled as follows. We consider for simplicity the case of a square lattice consisting of $n^{2}$ points in the plane. The 2D Hamiltonian is given by
$$
H_{n^{2}}=\frac{1}{2}\left(H_{n} \otimes I_{n}+I_{n} \otimes H_{n}\right)
$$
where the scaling factor $\frac{1}{2}$ is needed so as to have $\sigma\left(H_{n^{2}}\right) \subset[-1,1]$. The eigenvalues and eigenvectors of $H_{n^{2}}$ can be explicitly written in terms of those of $H_{n}$; see, e.g., [35]. Assuming again that $n$ is even, exactly half of the $n^{2}$ eigenvalues of $H_{n^{2}}$ are negative (counting multiplicities), the other half positive. As before, we are interested in finding the spectral projector associated with the eigenvectors corresponding to negative eigenvalues. Note again that the spectral gap tends to zero as $n \rightarrow \infty$. If $P_{n^{2}}$ denotes the spectral projector onto the occupied states, it is not difficult to show that
$$
\begin{equation*}
P_{n^{2}}=P_{n} \otimes\left(I_{n}-P_{n}\right)+\left(I_{n}-P_{n}\right) \otimes P_{n} \tag{10.8}
\end{equation*}
$$

It follows from (10.8) that the spectral projector $P_{n^{2}}$ has a natural $n \times n$ block structure, where the following hold:
- Each diagonal block is equal to $\frac{1}{2} I_{n}$; note that this gives the correct trace, $\operatorname{Tr}\left(P_{n^{2}}\right)=\frac{n^{2}}{2}$.
- The $(k, l)$ off-diagonal block $\Pi_{k l}$ is given by $\Pi_{k l}=\left[P_{n}\right]_{k l}\left(I_{n}-2 P_{n}\right)$. Hence, each off-diagonal block has a "striped" structure, with the main diagonal as well as the third, fifth, etc., off-diagonals identically zero. Moreover, every block $\Pi_{k l}$ with $l=k+2 m(m \geq 1)$ is zero.

This shows that in the 2D case, the rate of decay in the spectral projector is essentially the same as in the 1D case. The 3D case can be handled in a similar manner, leading to the same conclusion.

For this simple example we can also compute the entries of the density matrix at positive electronic temperature $T>0$. Recalling that the density matrix in this case is given by the Fermi-Dirac function with parameter $\beta=1 /\left(k_{B} T\right)$, we have in the 1D case (assuming $\mu=0$ )
$$
\begin{equation*}
P_{i j}=\frac{2}{n+1} \sum_{k=1}^{n} \frac{\sin \left(\frac{i k \pi}{n+1}\right) \sin \left(\frac{j k \pi}{n+1}\right)}{1+\exp \left[\beta \cos \left(\frac{k \pi}{n+1}\right)\right]} \tag{10.9}
\end{equation*}
$$

Making use again of the trigonometric identity (10.4) and using the same substitution $x=k /(n+1)$, we can reduce the computation of the density matrix element $P_{i j}$ for $n \rightarrow \infty$ to the evaluation of the integral
$$
\begin{equation*}
P_{i j}=\int_{0}^{1} \frac{\cos [(i-j) \pi x]-\cos [(i+j) \pi x]}{1+\exp (\beta \cos \pi x)} d x \tag{10.10}
\end{equation*}
$$

Unfortunately, this integral cannot be evaluated explicitly in terms of elementary functions. Note, however, that the integral
$$
I_{k}=\int_{0}^{1} \frac{\cos (k \pi x)}{1+\exp (\beta \cos \pi x)} d x
$$
(where $k$ is an integer) becomes, under the change of variable $\pi x=\arccos t$,
$$
I_{k}=\frac{1}{\pi} \int_{-1}^{1} \frac{\cos (k \arccos t)}{1+\mathrm{e}^{\beta t}} \frac{d t}{\sqrt{1-t^{2}}}
$$

Hence, up to a constant factor, $I_{k}$ is just the $k$ th coefficient in the Chebyshev expansion of the Fermi-Dirac function $1 /\left(1+\mathrm{e}^{\beta t}\right)$. Since the Fermi-Dirac function is analytic on the interior of an ellipse containing the interval $[-1,1]$ and continuous on the boundary of such an ellipse, it follows from the general theory of Chebyshev approximation that the coefficients $I_{k}$ decay at least exponentially fast as $k \rightarrow \infty$; see, e.g, [92]. This in turn implies that the entries $P_{i j}$ given by (10.10) decay at least exponentially fast away from the main diagonal, the faster the larger the temperature is, as already discussed in section 8.7. Hence, for this special case we have established in a more direct way the exponential decay behavior already proved in general in section 8.1. In the present case, however, for any value of $\beta$ the decay rate of the entries $P_{i j}$ given by (10.10) can be determined to arbitrary accuracy by numerically computing the Chebyshev coefficients of the Fermi-Dirac function.

We mention that a simple, 1D model of a system with arbitrarily small gap was described in [49]. The (continuous) Hamiltonian in [49] consists of the kinetic term plus a potential given by a sum of Gaussian wells located at the nuclei sites $X_{i}$ :
$$
\mathcal{H}=-\frac{1}{2} \frac{d^{2}}{d x^{2}}+V(x), \quad V(x)=-\sum_{i=-\infty}^{\infty} \frac{a}{\sqrt{2 \pi \sigma^{2}}} \exp \left(-\left(x-X_{i}\right)^{2} / 2 \sigma^{2}\right),
$$
with $a>0$ and $\sigma>0$ tunable parameters. The spectra of this family of Hamiltonians present a band structure with band gap proportional to $\sqrt{a} / \sigma$. Note that the model
essentially reduces to ours for $a \rightarrow 0$ and/or for $\sigma \rightarrow \infty$. On the other hand, while the gap can be made arbitrarily small by tuning the parameters in the model, for any choice of $a>0$ and $\sigma>0$ the gap does not vanish; therefore, no approximation of the infinite-size system with a sequence of finite-size ones can lead to a vanishing gap in the thermodynamic limit. This means that our bounds, when applied to this model, will yield exponential decay, albeit very slow (since the correlation lengths will be quite large for small $a \rightarrow 0$ and/or for large $\sigma$ ). The model in [49], on the other hand, can be useful for testing purposes when developing algorithms for metal-like systems with slowly decaying density matrices.
II. Other Applications. In this section we sketch a few possible applications of our decay results to areas other than electronic structure computations.
II.I. Density Matrices for Thermal States. In quantum statistical mechanics, the equilibrium density matrix for a system of particles subject to a heat bath at absolute temperature $T$ is defined as
$$
\begin{equation*}
P=\frac{\mathrm{e}^{-\beta H}}{Z}, \quad \text { where } \quad Z=\operatorname{Tr}\left(\mathrm{e}^{-\beta H}\right) \tag{11.1}
\end{equation*}
$$

As usual, $\beta=\left(k_{B} T\right)^{-1}$, where $k_{B}$ denotes the Boltzmann constant; see [102]. The matrix $P$ is the quantum analogue of the canonical Gibbs state. The Hamiltonian $H$ is usually assumed to have been shifted so that the smallest eigenvalue is zero [87, p. 112]. Note that $P$ as defined in (11.1) is not an orthogonal projector. It is, however, Hermitian and positive semidefinite. Normalization by the partition function $Z$ ensures that $\sigma(P) \subset[0,1]$ and that $\operatorname{Tr}(P)=1$.

It is clear that for increasing temperature, i.e., for $T \rightarrow \infty$ (equivalently, for $\beta \rightarrow 0$ ), the canonical density matrix $P$ approaches the identity matrix, normalized by the matrix size $n$. In particular, the off-diagonal entries tend to zero. The physical interpretation of this is that in the limit of large temperatures the system states become totally uncorrelated. For temperatures approaching absolute zero, on the other hand, the canonical matrix $P$ tends to the orthogonal projector associated with the zero eigenvalue (ground state). In this limit, the correlation between state $i$ and state $j$ is given by the $(i, j)$ entry of the orthogonal projector onto the eigenspace corresponding to the zero eigenvalue, normalized by $n$.

For finite, positive values of $T$, the canonical density matrix $P$ is full but decays away from the main diagonal (or, more generally, away from the sparsity pattern of $H)$. The rate of decay depends on $\beta$ : the smaller it is, the faster the decay. Application of the bounds developed in section 8 to the matrix exponential is straightforward. For instance, the bounds based on Bernstein's theorem take the form
$$
\begin{equation*}
\left|\left[\mathrm{e}^{-\beta H}\right]_{i j}\right| \leq C(\beta) \mathrm{e}^{-\alpha d(i, j)}, \quad i \neq j \tag{11.2}
\end{equation*}
$$
where
$$
C(\beta)=\frac{2 \chi}{\chi-1} \mathrm{e}^{\beta\|H\|_{2}\left(\kappa_{1}-1\right) / 2} \quad \text { and } \quad \alpha=2 \ln \chi .
$$

In these expressions, $\chi>1$ and $\kappa_{1}>1$ are the parameters associated with the Bernstein ellipse with foci in -1 and 1 and major semiaxis $\kappa_{1}$, as described in section 8. Choosing $\chi$ large makes the exponential term decay $\mathrm{e}^{-\alpha d(i, j)}$ very fast, but causes $C(\beta)$ to grow larger. Clearly, a smaller $\beta$ makes the upper bound (11.2) smaller. Bounds on the entries of the canonical density matrix $P$ can be obtained by dividing
through the upper bounds by $Z$. Techniques for estimating $Z$ can be developed using the techniques described in [56]; see also [11].

Although the bound (11.2) is an exponentially decaying one, it can be shown that the decay in the entries of a banded or sparse matrix is actually superexponential. This can be shown by expanding the exponential in a series of Chebyshev polynomials and using the fact that the coefficients in the expansion, which can be expressed in terms of Bessel functions, decay to zero superexponentially; see [92] and also [69]. The decay bounds obtained in this way are, however, less transparent and more complicated to evaluate than (11.2).

Finally, exponential decay bounds for spectral projectors and other matrix functions might provide a rigorous justification for $O(n)$ algorithms recently developed for disordered systems; see, e.g., [117, 118].
11.2. Quantum Information Theory. A related area of research where our decay bounds for matrix functions have proven useful is the study of quantum manybody systems in information theory; see, e.g., [31, 32, 38, 120, 121]. In particular, relationships between spectral gaps and rates of decay for functions of finite-range Hamiltonians have been established in [31] using the techniques introduced in [12]. The exponential decay of correlations and its relation to the spectral gap have also been studied in [62, 63].

As shown in [32], exponential decay bounds for matrix functions play a crucial role in establishing so-called area laws for the entanglement entropy of ground states associated with bosonic systems. These area laws essentially state that the entanglement entropy associated with a 3D bosonic lattice is proportional to the surface area, rather than to the volume, of the lattice. Intriguingly, such area laws are analogous to those governing the Beckenstein-Hawking black hole entropy. We refer the interested reader to the recent, comprehensive survey paper [38] for additional information.
II.3. Complex Networks. The study of complex networks is an emerging field of science currently undergoing vigorous development. Researchers in this highly interdisciplinary field include mathematicians, computer scientists, physicists, chemists, engineers, neuroscientists, biologists, social scientists, etc. Among the mathematical tools used in this field, linear algebra and graph theory, in particular spectral graph theory, play a major role. Also, statistical mechanics concepts and techniques have been found to be ideally suited to the study of large-scale networks.

In recent years, quantitative methods of network analysis have increasingly made use of matrix functions. This approach has been spearheaded in the works of Estrada, Rodríguez-Velázquez, D. Higham, and Hatano; see, e.g., [39, 40, 41, 42, 43, 46], as well as the recent surveys [45, 44] and the references therein. Functions naturally arising in the context of network analysis include the exponential, the resolvent, and hyperbolic functions, among others. Physics-based justifications for the use of these matrix functions in the analysis of complex networks are thoroughly discussed in [44].

For example, the exponential of the adjacency matrix $A$ associated with a simple, undirected graph $G=(V, E)$ can be used to give natural definitions of important measures associated with nodes in $G$, such as the subgraph centrality associated with node $i$, defined as $C(i)=\left[\mathrm{e}^{A}\right]_{i i}$, and the communicability associated with two distinct nodes $i$ and $j$, defined as $C(i, j)=\left[\mathrm{e}^{A}\right]_{i j}$. Other network quantities that can be expressed in terms of the entries in appropriate matrix functions of $A$ include betweenness, returnability, vulnerability, and so forth. The graph Laplacian $L=D-A$, where $D=\operatorname{diag}\left(d_{1}, \ldots, d_{n}\right)$ with $d_{i}$ denoting the degree of node $i$, is sometimes used instead of the adjacency matrix, as well as weighted analogues of both $A$ and $L$.

Most networks arising in real-world applications are sparse, often with degree distributions closely approximated by power laws. Because the maximum degree in such "scale-free" networks increases as the number of nodes tends to infinity, one cannot expect uniform exponential decay rates to hold asymptotically for the matrix functions associated with such graphs unless additional structure is imposed, for instance, in the form of weights. Nevertheless, our bounds for the entries of functions of sparse matrices can be used to obtain estimates on quantities such as the communicability between two nodes. A discussion of locality (or the lack thereof) in matrix functions used in the analysis of complex networks can be found in [44]. We also refer the reader to [11] for a description of quadrature-rule-based bounds for the entries of matrix functions associated with complex networks.
II.4. Tridiagonal Eigensolvers. The solution of symmetric tridiagonal eigenvalue problems plays an important role in many fields of computational science. As noted, for example, in [130], solving such problems is key for most dense real symmetric (and complex Hermitian) eigenvalue computations and therefore plays a central role in standard linear algebra libraries such as LAPACK and ScaLAPACK. Even in the sparse case, the symmetric tridiagonal eigenvalue problem appears as a step in the Lanczos algorithm.

The efficiency of symmetric tridiagonal eigensolvers can be significantly increased by exploiting localization in the eigenvectors (more generally, invariant subspaces) associated with an isolated cluster of eigenvalues. It would be highly desirable to identify beforehand any localization in the eigenspace in a cost-effective manner, as this would lead to reduced computational costs $[101,130]$. It is clear that this problem is essentially the same as the one considered in this paper, with the additional assumption that the matrix $H$ is tridiagonal. Given estimates on the location of the cluster of eigenvalues and on the size of the gaps separating it from the remainder of the spectrum, the techniques described in this paper can be used to bound the entries in the spectral projector associated to the cluster of interest; in turn, the bounds can be used to identify banded approximations to the spectral projectors with guaranteed prescribed error. Whether the estimates obtained in this manner are accurate enough to lead to practical algorithms with run times and storage demands substantially improved over current ones remains an open question for further research.

Finally, in the recent paper [139] the exponential decay results in [12] are used to derive error bounds and stopping criteria for the Lanczos method applied to the computation of $\mathrm{e}^{-t A} v$, where $A$ is a large SPD matrix, $v$ is a vector, and $t>0$. The bounds are applied to the exponential of the tridiagonal matrix $T_{k}$ generated after $k$ steps by the Lanczos process in order to obtain the approximation error after $k$ steps.
II.5. Non-Hermitian Extensions. Although the main focus of the paper has been the study of functions of sparse Hermitian matrices, many of our results can be extended, under appropriate conditions, to non-Hermitian matrices. The generalizations of our decay bounds to normal matrices, including, for example, skew-Hermitian matrices, is relatively straightforward; see, e.g., the results in [14] and [106]. Further generalizations to diagonalizable matrices were given in [14], although the bounds now contain additional terms taking into account the departure from normality. These bounds may be difficult to use in practice, as knowledge of the eigenvectors or of the field of values of the matrix is needed. Bounds for functions of general sparse matrices can also be obtained using contour integration; see, e.g., [106] and [91]. It is quite possible that these bounds will prove useful in applications involving functions of sparse, nonnormal matrices. Examples include functions of digraphs in network analysis,
like returnability, or functions of the Hamiltonians occurring in the emerging field of non-Hermitian quantum mechanics; see, respectively, [42] and [8, 9, 94].
12. Conclusions and Open Problems. In this paper we have described a general theory of localization for the density matrices associated with certain sequences of banded or sparse discrete Hamiltonians of increasing size. We have obtained, under very general conditions, exponential decay bounds for the off-diagonal entries of zero-temperature density matrices for gapped systems ("insulators") and for density matrices associated with systems at positive electronic temperature. The theory, while purely mathematical, recovers well-known physical phenomena such as the fact that the rate of decay is faster at higher temperatures and for larger gaps, and even captures the correct asymptotics for small gaps and low temperatures. Thus, we have provided a theoretical justification for the development of $O(n)$ methods for electronic structure computations. As an integral part of this theory, we have also surveyed the approximation of rapidly decaying matrices by banded or sparse ones, the effects of transforming a Hamiltonian from a nonorthogonal to an orthogonal basis, and some general properties of orthogonal projectors.

In the case of zero-temperature and vanishing gaps, our bounds deteriorate for increasing $n$. In the limit as $n \rightarrow \infty$ we no longer have exponentially decaying bounds, which is entirely consistent with the physics. For metallic systems at zero temperature the decay in the spectral projector follows a power law, and we have exhibited a simple model Hamiltonian for which the decay in the corresponding density matrix is only linear in the distance from the main diagonal.

Because of the slow decay, the development of $O(n)$ methods in the metallic case at zero temperature is problematic. We refer the reader to [5, 19, 81, 134] for some attempts in this direction, but the problem remains essentially open. In the metallic case it may be preferable to keep $P$ in the factorized form $P=X X^{*}$, where $X \in \mathbb{C}^{n \times n_{e}}$ is any matrix whose columns span the occupied subspace, and to seek a maximally localized $X$. Note that
$$
P=X X^{*}=(X U)(X U)^{*}
$$
for any unitary $n_{e} \times n_{e}$ matrix $U$, so the question is whether the occupied subspace admits a set of basis vectors that can be rotated so as to become as localized as possible. Another possibility is to research the use of rank-structured approximations (such as hierarchical matrix techniques [59]) to the spectral projector. Combinations of tensor product approximations and wavelets appear to be promising. We refer here to [55] for a study of the decay properties of density matrices in a wavelet basis (see also [119]), and to [16] for an early attempt to exploit near low-rank properties of spectral projectors. See also the more recent works by W. Hackbusch and collaborators [26, 27, 28, 47, 48, 86].

Besides the motivating application of electronic structure, our theory is also applicable to other problems where localization plays a prominent role. We hope that this paper will stimulate further research in this fascinating and important area at the crossroads of mathematics, physics, and computing.

Acknowledgments. We are indebted to three anonymous referees for carefully reading the original manuscript and for suggesting a number of corrections and improvements. Thanks also to the handling editor, Fadil Santosa, for useful feedback and for his patience and understanding during the several months it took us to revise the paper. We would also like to acknowledge useful discussions with David Borthwick, Matt Challacombe, Jean-Luc Fattebert, Roberto Grena, Daniel Kressner, and

Maxim Olshanskii. Finally, we are grateful to Jacek Jakowski for providing the data for the linear alkane.

\section*{REFERENCES}
[1] N. I. Achieser, Theory of Approximation, Frederick Ungar, New York, 1956.
[2] P. W. Anderson, Absence of diffusion in certain random lattices, Phys. Rev., 109 (1958), pp. 1492-1505.
[3] R. Baer and M. Head-Gordon, Sparsity of the density matrix in Kohn-Sham density functional theory and an assessment of linear system-size scaling methods, Phys. Rev. Lett., 79 (1997), pp. 3962-3965.
[4] R. Baer and M. Head-Gordon, Chebyshev expansion methods for electronic structure calculations on large molecular systems, J. Chem. Phys., 107 (1997), pp. 10003-10013.
[5] R. Baer and M. Head-Gordon, Energy renormalization-group method for electronic structure of large systems, Phys. Rev. B, 58 (1998), pp. 15296-15299.
[6] K. R. Bates, A. D. Daniels, and G. E. Scuseria, Comparison of conjugate gradient density matrix search and Chebyshev expansion methods for avoiding diagonalization in largescale electronic structure calculations, J. Chem. Phys., 109 (1998), pp. 3308-3312.
[7] C. Bekas, E. Kokiopoulou, and Y. Saad, Computation of large invariant subspaces using polynomial filtered Lanczos iterations with applications in density functional theory, SIAM J. Matrix Anal. Appl., 30 (2008), pp. 397-418.
[8] C. M. Bender, S. Boettcher, and P. N. Meisinger, PT-symmetric quantum mechanics, J. Math. Phys., 40 (1999), pp. 2201-2229.
[9] C. M. Bender, D. C. Brody, and H. F. Jones, Must a Hamiltonian be Hermitian?, Amer. J. Phys., 71 (2003), pp. 1095-1102.
[10] M. Benzi, Preconditioning techniques for large linear systems: A survey, J. Comput. Phys., 182 (2002), pp. 418-477.
[11] M. Benzi and P. Boito, Quadrature rule-based bounds for functions of adjacency matrices, Linear Algebra Appl., 433 (2010), pp. 637-652.
[12] M. Benzi and G. H. Golub, Bounds for the entries of matrix functions with applications to preconditioning, BIT, 39 (1999), pp. 417-438.
[13] M. Benzi, C. D. Meyer, and M. Tůma, A sparse approximate inverse preconditioner for the conjugate gradient method, SIAM J. Sci. Comput., 17 (1996), pp. 1135-1149.
[14] M. Benzi and N. Razouk, Decay bounds and $O(n)$ algorithms for approximating functions of sparse matrices, Electr. Trans. Numer. Anal., 28 (2007), pp. 16-39.
[15] M. Benzi and M. Tůma, Orderings for factorized sparse approximate inverse preconditioners, SIAM J. Sci. Comput., 21 (2000), pp. 1851-1868.
[16] G. Beylkin, N. Coult, and M. J. Mohlenkamp, Fast spectral projection algorithms for density-matrix computations, J. Comput. Phys., 152 (1999), pp. 32-54.
[17] A. Böttcher and S. M. Grudsky, Spectral Properties of Banded Toeplitz Matrices, SIAM, Philadelphia, PA, 2005.
[18] A. Böttcher and B. Silbermann, Introduction to Large Truncated Toeplitz Matrices, Springer, New York, 1998.
[19] D. R. Bowler, J.-L. Fattebert, M. J. Gillan, P.-D. Haynes, and C.-K. Skylaris, Introductory remarks: Linear scaling methods, J. Phys. Condensed Matter, 20 (2008), 290301.
[20] D. R. Bowler and T. Miyazaki, $O(N)$ methods in electronic structure calculations, Rep. Progr. Phys., 75 (2012), 036503.
[21] C. Brouder, G. Panati, M. Calandra, C. Mourougane, and N. Marzari, Exponential localization of Wannier functions in insulators, Phys. Rev. Lett., 98 (2007), 046402.
[22] K. Burke et al., The $A B C$ of $D F T$, available online from http://chem.uci.edu/~kieron/ dftold2/literature.php, accessed October 2012.
[23] M. Ceriotti, T. D. Kühne, and M. Parrinello, An efficient and accurate decomposition of the Fermi operator, J. Chem. Phys., 129 (2008), 024707.
[24] M. Challacombe, A simplified density matrix minimization for linear scaling self-consistent field theory, J. Chem. Phys., 110 (1999), pp. 2332-2342.
[25] M. Challacombe, A general parallel sparse-blocked matrix multiply for linear scaling SCF theory, Comput. Phys. Commun., 128 (2000), pp. 93-107.
[26] S. R. Chinnamsetty, Wavelet Tensor Product Approximation in Electronic Structure Calculations, Ph.D. thesis, Universität Leipzig, 2008.
[27] S. R. Chinnamsetty, M. Espig, H.-J. Flad, and W. Hackbusch, Canonical tensor products as a generalization of Gaussian-type orbitals, Z. Phys. Chem., 224 (2010), pp. 681-694.
[28] S. R. Chinnamsetty, M. Espig, B. N. Khoromskij, W. Hackbusch, and H.-J. Flad, Tensor product approximation with optimal rank in quantum chemistry, J. Chem. Phys., 127 (2007), 084110.
[29] C. K. Chui and M. Hasson, Degree of uniform approximation on disjoint intervals, Pacific J. Math., 105 (1983), pp. 291-297.
[30] L. Colombo and W. Sawyer, A parallel implementation of tight-binding molecular dynamics, Mater. Sci. Engrg. B, 37 (1996), pp. 228-231.
[31] M. Cramer and J. Eisert, Correlations, spectral gap and entanglement in harmonic quantum systems on generic lattices, New J. Phys., 8 (2006), 71.
[32] M. Cramer, J. Eisert, M. B. Plenio, and J. Dreissig, Entanglement-area law for general bosonic harmonic lattice systems, Phys. Rev. A, 73 (2006), 012309.
[33] P. J. Davis and P. Rabinowitz, Methods of Numerical Integration, 2nd ed., Academic Press, London, 1984.
[34] S. Demko, W. F. Moss, and P. W. Smith, Decay rates for inverses of band matrices, Math. Comp., 43 (1984), pp. 491-499.
[35] J. W. Demmel, Applied Numerical Linear Algebra, SIAM, Philadelphia, PA, 1997.
[36] J. des Cloizeaux, Energy bands and projection operators in a crystal: Analytic and asymptotic properties, Phys. Rev., 135 (1964), pp. A685-A697.
[37] R. Diestel, Graph Theory, Springer, Berlin, 2000.
[38] J. Eisert, M. Cramer, and M. B. Plenio, Colloquium: Area laws for the entanglement entropy, Rev. Modern Phys., 82 (2010), pp. 277-306.
[39] E. Estrada, Generalized walks-based centrality measures for complex biological networks, J. Theoret. Biol., 263 (2010), pp. 556-565.
[40] E. Estrada, The Structure of Complex Networks: Theory and Applications, Oxford University Press, Oxford, UK, 2012.
[41] E. Estrada and N. Hatano, Communicability in complex networks, Phys. Rev. E, 77 (2008), 036111.
[42] E. Estrada and N. Hatano, Returnability in complex directed networks (digraphs), Linear Algebra Appl., 430 (2009), pp. 1886-1896.
[43] E. Estrada and N. Hatano, A vibrational approach to node centrality and vulnerability in complex networks, Phys. A, 389 (2010), pp. 3648-3660.
[44] E. Estrada, N. Hatano, and M. Benzi, The physics of communicability in complex networks, Phys. Rep., 514 (2012), pp. 89-119.
[45] E. Estrada and D. J. Higham, Network properties revealed through matrix functions, SIAM Rev., 52 (2010), pp. 696-714.
[46] E. Estrada and J. A. Rodríguez-Velázquez, Subgraph centrality in complex networks, Phys. Rev. E, 71 (2005), 056103.
[47] H.-J. Flad, W. Hackbusch, D. Kolb, and R. Schneider, Wavelet approximation of correlated wavefunctions. I. Basics, J. Chem. Phys., 116 (2002), pp. 9641-9657.
[48] H.-J. Flad, W. Hackbusch, B. N. Khoromskij, and R. Schneider, Concepts of data-sparse tensor-product approximation in many-particle modelling, in Matrix Methods: Theory, Algorithms and Applications: Dedicated to the Memory of Gene Golub, V. Olshevsky and E. Tyrtyshnikov, eds., World Scientific, Hackensack, NJ, 2010, pp. 313-343.
[49] C. J. García-Cervera, J. Lu, Y. Xuan, and W. E, A linear scaling subspace iteration algorithm with optimally localized non-orthogonal wave functions for Kohn-Sham density functional theory, Phys. Rev. B, 79 (2009), 115110.
[50] G. Giuliani and G. Vignale, Quantum Theory of the Electron Liquid, Cambridge University Press, Cambridge, UK, 2005.
[51] S. Goedecker, Low complexity algorithms for electronic structure calculations, J. Comput. Phys., 118 (1995), pp. 261-268.
[52] S. Goedecker, Decay properties of the finite-temperature density matrix in metals, Phys. Rev. B, 58 (1998), pp. 3501-3502.
[53] S. Goedecker, Linear scaling electronic structure methods, Rev. Modern Phys., 71 (1999), pp. 1085-1123.
[54] S. Goedecker and L. Colombo, Efficient linear scaling algorithm for tight-binding molecular dynamics, Phys. Rev. Lett., 73 (1994), pp. 122-125.
[55] S. Goedecker and O. V. Ivanov, Frequency localization properties of the density matrix and its resulting hypersparsity in a wavelet representation, Phys. Rev. B, 59 (1999), pp. 72707273.
[56] G. H. Golub and G. Meurant, Matrices, Moments and Quadrature with Applications, Princeton University Press, Princeton, NJ, 2010.
[57] G. H. Golub and C. F. Van Loan, Matrix Computations, 3rd ed., Johns Hopkins University Press, Baltimore, MD, 1996.
[58] U. Grenander and G. Szegö, Toeplitz Forms and Their Applications, Chelsea, New York, 1958.
[59] W. Hackbusch, Hierarchische Matrizen: Algorithmen und Analysis, Springer, Berlin, 2009.
[60] P. Hartman and A. Wintner, The spectra of Toeplitz's matrices, Amer. J. Math., 76 (1954), pp. 867-882.
[61] M. Hasson, The degree of approximation by polynomials on some disjoint intervals in the complex plane, J. Approx. Theory, 144 (2007), pp. 119-132.
[62] M. B. Hastings, Locality in quantum and Markov dynamics on lattices and networks, Phys. Rev. Lett., 93 (2004), 140402.
[63] M. B. Hastings and T. Koma, Spectral gap and exponential decay of correlations, Comm. Math. Phys., 265 (2006), pp. 781-804.
[64] L. He and D. Vanderbilt, Exponential decay properties of Wannier functions and related quantities, Phys. Rev. Lett., 86 (2001), pp. 5341-5344.
[65] N. J. Higham, Functions of Matrices: Theory and Computation, SIAM, Philadelphia, PA, 2008.
[66] P. Hohenberg and W. Kohn, Inhomogeneous electron gas, Phys. Rev., 136 (1964), pp. B864871.
[67] R. A. Horn and C. R. Johnson, Matrix Analysis, Cambridge University Press, Cambridge, UK, 1991.
[68] R. A. Horn and C. R. Johnson, Topics in Matrix Analysis, Cambridge University Press, Cambridge, UK, 1994.
[69] A. Iserles, How large is the exponential of a banded matrix?, New Zealand J. Math., 29 (2000), pp. 177-192.
[70] S. Ismail-Beigi and T. A. Arias, Locality of the density matrix in metals, semiconductors, and insulators, Phys. Rev. Lett., 82 (1999), pp. 2127-2130.
[71] S. Jaffard, Propriétés des matrices "bien localisées" près de leur diagonale et quelques applications, Ann. Inst. Henri Poincarè, 7 (1990), pp. 461-476.
$[72]$ B. Jansik, S. Høst, P. Jørgensen, and J. Olsen, Linear-scaling symmetric square-root decomposition of the overlap matrix, J. Chem. Phys., 126 (2007), 124104.
[73] J. Jędrzejewski and T. Krokhmalskii, Exact results for spatial decay of the one-body density matrix in low-dimensional insulators, Phys. Rev. B, 70 (2004), 153102.
[74] W. Kohn, Analytic properties of Bloch waves and Wannier functions, Phys. Rev., 115 (1959), pp. 809-821.
[75] W. Kohn, Density functional and density matrix method scaling linearly with the number of atoms, Phys. Rev. Lett., 76 (1996), pp. 3168-3171.
[76] W. Kohn and L. J. Sham, Self-consistent equations including exchange and correlation effects, Phys. Rev. (2), 140 (1965), pp. A1133-A1138.
[77] C. Le Bris, Computational chemistry from the perspective of numerical analysis, Acta Numer., 14 (2005), pp. 363-444.
[78] S. Li, S. Ahmed, G. Glimieck, and E. Darve, Computing entries of the inverse of a sparse matrix using the FIND algorithm, J. Comput. Phys., 227 (2008), pp. 9408-9427.
[79] X.-P. Li, R. W. Nunes, and D. Vanderbilt, Density-matrix electronic structure method with linear system-size scaling, Phys. Rev. B, 47 (1993), pp. 10891-10894.
[80] W. Liang, C. Saravanan, Y. Shao, R. Baer, A. T. Bell, and M. Head-Gordon, Improved Fermi operator expansion methods for fast electronic structure calculations, J. Chem. Phys., 119 (2003), pp. 4117-4124.
[81] L. Lin, J. Lu, L. Ying, R. Car, and W. E, Multipole representation of the Fermi operator with application to the electronic structure analysis of metallic systems, Phys. Rev. B, 79 (2009), 115133.
[82] L. Lin, C. Yang, J. Lu, L. Ying, and W. E, A fast parallel algorithm for selected inversion of structured sparse matrices with application to $2 D$ electronic structure calculations, SIAM J. Sci. Comput., 33 (2011), pp. 1329-1351.
[83] L. Lin, C. Yang, J. C. Meza, J. Lu, and L. Ying, SelInv-An algorithm for selected inversion of a sparse symmetric matrix, ACM Trans. Math. Software, 37 (2011), pp. 1-19.
[84] G. G. Lorentz, Approximation of Functions, Holt, Rinehart and Winston, New York, 1966.
[85] P.-O. Löwdin, Linear Algebra for Quantum Theory, John Wiley and Sons, New York, 1998.
[86] H. luo, D. Kolb, H.-J. Flad, W. Hackbusch, and T. Koprucki, Wavelet approximation of correlated wavefunctions. II. Hyperbolic wavelets and adaptive approximation schemes, J. Chem. Phys., 117 (2002), pp. 3625-3638.
[87] G. Mackey, Mathematical Foundations of Quantum Mechanics, Dover, New York, 2004.
[88] N. H. March, W. H. Young, and S. Sampanthar, The Many-Body Problem in Quantum Mechanics, Cambridge University Press, Cambridge, UK, 1967.
[89] R. M. Martin, Electronic Structure. Basic Theory and Practical Methods, Cambridge University Press, Cambridge, UK, 2004.
[90] P. E. Maslen, C. Ochsenfeld, C. A. White, M. S. Lee, and M. Head-Gordon, Locality and sparsity of ab initio one-particle density matrices and localized orbitals, J. Phys. Chem. A, 102 (1998), pp. 2215-2222.
[91] N. Mastronardi, M. Ng, and E. E. Tyrtyshnikov, Decay in functions of multiband matrices, SIAM J. Matrix Anal. Appl., 31 (2010), pp. 2721-2737.
[92] G. Meinardus, Approximation of Functions: Theory and Numerical Methods, Springer Tracts in Natural Philosophy 13, Springer, New York, 1967.
[93] J. M. Millam and G. Scuseria, Linear scaling conjugate gradient density matrix search as an alternative to diagonalization for first principles electronic structure calculations, J. Chem. Phys., 106 (1997), pp. 5569-5577.
[94] N. Moiseyev, Non-Hermitian Quantum Mechanics, Cambridge University Press, Cambridge, UK, 2011.
[95] K. Nemeth and G. Scuseria, Linear scaling density matrix search based on sign matrices, J. Chem. Phys., 113 (2000), pp. 6035-6041.
[96] G. Nenciu, Existence of the exponentially localised Wannier functions, Comm. Math. Phys., 91 (1983), pp. 81-85.
[97] A. M. N. Niklasson, Density matrix methods in linear scaling electronic structure theory, in Linear-Scaling Techniques in Computational Chemistry and Physics, R. Zaleśny, M. G. Papadopoulos, P. G. Mezey, and J. Leszczynski, eds., Springer, New York, 2011, pp. 439473.
[98] P. Ordejón, Order- $N$ tight-binding methods for electronic-structure and molecular dynamics, Comput. Materials Sci., 12 (1998), pp. 157-191.
[99] P. Ordejón, D. A. Drabold, R. M. Martin, and M. P. Grumbach, Linear system-size scaling methods for electronic-structure calculations, Phys. Rev. B, 51 (1995), pp. 14561476.
[100] T. Ozaki, Efficient recursion method for inverting an overlap matrix, Phys. Rev. B, 64 (2001), 195110.
[101] B. Parlett, Invariant subspaces for tightly clustered eigenvalues of tridiagonals, BIT, 36 (1996), pp. 542-562.
[102] R. K. Pathria, Statistical Mechanics, Internat. Ser. Natural Philos. 45, Pergamon Press, Oxford, UK, 1986.
[103] E. Prodan, Nearsightedness of electronic matter in one dimension, Phys. Rev. B, 73 (2006), 085108.
[104] E. Prodan, S. R. Garcia, and M. Putinar, Norm estimates of complex symmetric operators applied to quantum systems, J. Phys. A Math. Gen., 39 (2006), pp. 389-400.
[105] E. Prodan and W. Kohn, Nearsightedness of electronic matter, Proc. Natl. Acad. Sci. USA, 102 (2005), pp. 11635-11638.
[106] N. Razouk, Localization Phenomena in Matrix Functions: Theory and Algorithms, Ph.D. thesis, Emory University, Atlanta, GA, 2008.
[107] M. Reed and B. Simon, Methods of Modern Mathematical Physics. Volume I: Functional Analysis, Academic Press, New York, London, 1972.
[108] M. Rosenblum, The absolute continuity of Toeplitz's matrices, Pacific J. Math., 10 (1960), pp. 987-996.
[109] E. H. Rubensson, Controlling errors in recursive Fermi-Dirac operator expansions with applications in electronic structure theory, SIAM J. Sci. Comput., 34 (2012), pp. B1-B23.
[110] E. H. Rubensson, N. Bock, E. Holmström, and A. M. N. Niklasson, Recursive inverse factorization, J. Chem. Phys., 128 (2008), 104105.
[111] E. H. Rubensson, E. Rudberg, and P. Salek, Rotations of occupied invariant subspaces in self-consistent field calculations, J. Math. Phys., 49 (2008), 032103.
[112] E. H. Rubensson, E. Rudberg, and P. Salek, Truncation of small matrix elements based on the Euclidean norm for blocked data structures, J. Comput. Chem., 30 (2009), pp. 974977.
[113] E. H. Rubensson, E. Rudberg, and P. Salek, Methods for Hartree-Fock and density functional theory electronic structure calculations with linearly scaling processor time and memory usage, in Linear-Scaling Techniques in Computational Chemistry and Physics, R. Zaleśny, M. G. Papadopoulos, P. G. Mezey, and J. Leszczynski, eds., Springer, New York, 2011, pp. 269-300.
[114] E. H. Rubensson and P. Salek, Systematic sparse matrix error control for linear scaling electronic structure calculations, J. Comput. Chem., 26 (2005), pp. 1628-1637.
[115] W. Rudin, Functional Analysis, McGraw-Hill, New York, 1973.
[116] Y. Saad, J. R. Chelikowsky, and S. M. Shontz, Numerical methods for electronic structure calculations of materials, SIAM Rev., 52 (2010), pp. 3-54.
[117] V. E. Sacksteder, Linear Algebra with Disordered Sparse Matrices That Have Spatial Structure: Theory and Computation, Ph.D. thesis, Department of Physics, Università degli Studi di Roma "La Sapienza," Rome, Italy, 2004.
[118] V. E. Sacksteder, $O(N)$ algorithms for disordered systems, Numer. Linear Algebra Appl., 12 (2005), pp. 827-838.
[119] R. Schneider and T. Weber, Wavelets for density matrix computation in electronic structure calculations, Appl. Numer. Math., 56 (2006), pp. 1383-1396.
[120] N. Schuch, Quantum Entanglement: Theory and Applications, Ph.D. thesis, Department of Physics, Technischen Universität München, Munich, Germany, 2007.
[121] N. Schuch, J. I. Cirac, and M. M. Wolf, Quantum states on harmonic lattices, Comm. Math. Phys., 267 (2006), pp. 65-92.
[122] S. Schweizer, J. Kussmann, B. Doser, and C. Hochsenfeld, Linear-scaling Cholesky decomposition, J. Comput. Chem., 29 (2008), pp. 1004-1010.
[123] G. Scuseria, Linear scaling density functional calculations with Gaussian orbitals, J. Phys. Chem. A, 25 (1999), pp. 4782-4790.
[124] R. B. Sidje and Y. Saad, Rational approximation to the Fermi-Dirac function with applications in density functional theory, Numer. Algorithms, 56 (2011), pp. 455-479.
[125] A. Szabo and N. Ostlund, Modern Quantum Chemistry. Introduction to Advanced Electronic Structure Theory, Dover, New York, 1996.
[126] J. Tang and Y. Saad, A probing method for computing the diagonal of a matrix inverse, Numer. Linear Algebra Appl., 19 (2012), pp. 485-501.
[127] S. N. Taraskin, D. A. Drabold, and S. R. Elliott, Spatial decay of the single-particle density matrix in insulators: Analytic results in two and three dimensions, Phys. Rev. Lett., 88 (2002), 196495.
[128] S. N. Taraskin, P. A. Fry, X. Zhang, D. A. Drabold, and S. R. Elliott, Spatial decay of the single-particle density matrix in tight-binding metals: Analytic results in two dimensions, Phys. Rev. B, 66 (2002), 233101.
[129] T. Thonhauser and D. Vanderbilt, Insulator/Chern-insulator transition in the Haldane model, Phys. Rev. B, 74 (2006), 23511.
[130] C. Vömel and B. N. Parlett, Detecting localization in an invariant subspace, SIAM J. Sci. Comput., 33 (2011), pp. 3447-3467.
[131] J. von Neumann, Wahrscheinlichkeitstheoretischer Aufbau der Quantenmechanik, Gött. Nach., 1927. See also Collected Works, Volume I, A. H. Taub, ed., Pergamon Press, New York, Oxford, London, Paris, 1961, pp. 208-235.
[132] J. von Neumann, Approximative properties of matrices of high finite order, Port. Math., 3 (1942), pp. 1-62. See also Collected Works, Volume I, A. H. Taub, ed., Pergamon Press, New York, Oxford, London, Paris, 1961, pp. 270-331.
[133] J. von Neumann, Mathematical Foundations of Quantum Mechanics, Princeton University Press, Princeton, NJ, 1955.
[134] S. C. Watson and E. Carter, Linear-scaling parallel algorithms for the first principles treatment of metals, Comput. Phys. Commun., 128 (2000), pp. 67-92.
[135] S. Y. Wu and C. S. Jayanthi, Order-N methodologies and their applications, Phys. Rep., 358 (2002), pp. 1-74.
[136] H. J. Xiang, J. Yang, J. G. Hou, and Q. Zhu, Linear scaling calculation of band edge states and doped semiconductors, J. Chem. Phys., 126 (2007), 244707.
[137] C. Yang, W. Gao, and J. C. Meza, On the convergence of the self-consistent field iteration for a class of nonlinear eigenvalue problems, SIAM J. Matrix Anal. Appl., 30 (2009), pp. 1773-1788.
[138] W. Yang, Direct calculation of electron density in density-functional theory, Phys. Rev. Lett., 66 (1991), pp. 1438-1441.
[139] Q. Ye, Error bounds for the Lanczos method for approximating matrix exponentials, SIAM J. Numer. Anal., 51 (2013), pp. 68-87.
[140] X. Zhang and D. A. Drabold, Properties of the density matrix from realistic calculations, Phys. Rev. B, 63 (2001), 233109.