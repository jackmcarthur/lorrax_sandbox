\title{
Efficient full frequency GW for metals using a multipole approach for the dielectric screening
}

\author{
Dario A. Leon ©, ${ }^{1,2,3, *}$ Andrea Ferretti ${ }^{2}$ Daniele Varsano ${ }^{(2)}$ Elisa Molinari ${ }^{(1,2}$ and Claudia Cardoso ${ }^{(2)}$ \\ ${ }^{1}$ FIM Department, University of Modena \& Reggio Emilia, 41125, Modena, Italy \\ ${ }^{2}$ S3 Centre, Istituto Nanoscienze, CNR, 41125, Modena, Italy \\ ${ }^{3}$ Department of Mechanical Engineering and Technology Management, Norwegian University of Life Sciences, 1430, Ås, Norway
}

\begin{abstract}
The properties of metallic systems with important and structured excitations at low energies, such as Cu , are challenging to describe with simple models like the plasmon pole approximation (PPA), and more accurate and sometimes prohibitive full frequency approaches are usually required. In this paper we propose a numerical approach to $G W$ calculations on metals that takes into account the frequency dependence of the screening via the multipole approximation (MPA), an accurate and efficient alternative to current full frequency methods that was recently developed and validated for semiconductors and overcomes several limitations of PPA. We now demonstrate that MPA can be successfully extended to metallic systems by optimizing the frequency sampling for this class of materials and introducing a simple method to include the long-wavelength limit of the intraband contributions. The good agreement between MPA and full frequency results for the calculations of quasiparticle energies, polarizability, self-energy, and spectral functions in different metallic systems confirms the accuracy and computational efficiency of the method. Finally, we discuss the physical interpretation of the MPA poles through a comparison with experimental electron energy loss spectra for Cu .
\end{abstract}

DOI: 10.1103/PhysRevB.107.155130

\section*{I. INTRODUCTION}

Many-body perturbation theory provides accurate methods to study the spectroscopic properties of condensed-matter systems from first principles [1-3]. Calculations often adopt the so-called $G W$ approximation $[2,4-8]$, for which the frequency integration in the evaluation of the self-energy is crucial to the deployment of the method. The frequency dependence of the screened potential $W$ is often described within the plasmon pole approximation (PPA) [9-14], successfully applied to the calculation of quasiparticle energies of semiconductors [9], the homogeneous electron gas [15], and simple metals such as Al and Na [16-19], especially for quasiparticles with energies close to the Fermi level. However, the description of the self-energy and the spectral functions for the whole range of frequencies is still challenging and requires expensive full frequency ( FF ) approaches.

Despite its success, the use of PPA is problematic when complex metals are concerned, even for the calculations of quasiparticle energies [6]. Its applicability for transition and noble metals has often been disputed [6,20], since the approximation is based on the homogeneous electron gas, for which PPA becomes exact in the long-wavelength limit [4,21,22], while it is in principle not strictly valid in the presence of strongly localized $d$ bands. In fact, these metals present complex screening effects due to collective excitations [23,24], which result in highly structured energy-loss spectra whose description is unattainable with a single plasmon peak [24]. Moreover, metals with relevant excitations at low energies,

\footnotetext{
*dario.alejandro.leon.valido@nmbu.no
}
such as Cu , require a specially accurate description of the lowfrequency regime, which makes it difficult to determine the PPA parameters since it requires sampling the polarizability at zero frequency [20].

In this context, we have recently developed a multipole approach (MPA) that naturally bridges from PPA to FF treatments of the $G W$ self-energy [25]. The method has been implemented in the yambo code [26,27] and was validated for bulk semiconductors. We have shown that, for semiconductors, MPA attains an accuracy comparable to that of FF methods at a much lower computational cost, while also circumventing several of the PPA shortcomings. Here we extend the assessment of MPA validity and performance to the case of metals. We do so by computing quasiparticle energies, together with the full frequency dependence of the self-energy and the spectral function. The approach is similar to the one used for semiconductors [25], with only slight changes in the frequency sampling strategy used in the multipole interpolation. In the following, we show that MPA is accurate for metallic systems, even in cases in which the use of PPA is challenging. In addition to MPA, we also propose a simple $a b$ initio method to include intraband contributions [28-32] to the dielectric function in the long-wavelength $\mathbf{q} \rightarrow 0$ limit, absent in semiconductors. Despite its virtually zero computational cost, it significantly accelerates the convergence of quasiparticle energies with respect to the $\mathbf{k}$-points grid in systems where the intraband contributions are dominant.

The paper is organized as follows: In Sec. II, we briefly summarize the $G W$ approximation and the MPA approach. In the same section, we further extend the strategy used in the frequency sampling for the multipole interpolation, with respect to the MPA implementation presented in Ref. [25] for
semiconductors. We also discuss the relevance of the inclusion of the intraband contribution to the dielectric function in the limit $\mathbf{q} \rightarrow 0$. In Sec. III we first present MPA calculations for simple metals and propose a simple way of including the aforementioned intraband limit. We then describe in detail the results obtained for Cu , a prototype challenging system for PPA. Finally, in Sec. IV we summarize and discuss the main conclusions of this work.

\section*{II. METHODS}

\section*{A. Quasiparticle energies within $\boldsymbol{G W}$}

We adopt the $G W$ approximation [2,4-8] for the evaluation of the electron-electron self-energy, which is computed via a frequency convolution of the one-particle Green's function $G(\omega)$ and the dynamical screened interaction potential $W(\omega)$ :
$$
\begin{equation*}
\Sigma^{G W}(\omega)=\frac{i}{2 \pi} \int_{-\infty}^{+\infty} d \omega^{\prime} e^{-i \omega^{\prime} \eta} G\left(\omega-\omega^{\prime}\right) W\left(\omega^{\prime}\right) \tag{1}
\end{equation*}
$$

In the present work we limit ourselves to the $G_{0} W_{0}$ approximation, although MPA, the method we want to discuss here, can be exploited also within more advanced approaches such as different self-consistent $G W$ schemes [33-40], or methods including vertex corrections [36,41-44] and cumulant expansions [45,46]. A more comprehensive discussion of these aspects can be found, e.g., in Refs. [7,8]. The present implementation uses as a starting point single-particle energies and wave functions computed within Kohn-Sham (KS) density-functional theory (DFT) to then build the noninteracting single-particle Green's function $G_{0}(\omega)$ and the irreducible polarizability $X_{0}(\omega)$.

The dressed polarizability $X(\omega)$ and the screened interaction, $W(\omega)$, are then numerically evaluated by solving the Dyson equation for each given frequency:
$$
\begin{align*}
X(\omega) & =X_{0}(\omega)+X_{0}(\omega) v X(\omega) \\
W(\omega) & =\varepsilon^{-1}(\omega) v=v+v X(\omega) v \tag{2}
\end{align*}
$$
where $v$ is the bare Coulomb potential, $\varepsilon$ the dielectric function and, for simplicity, we have omitted the spatial, nonlocal, degrees of freedom. All the quantities have to be thought as frequency-dependent operators or matrices of the form $X(\omega)=X\left(\mathbf{r}, \mathbf{r}^{\prime}, \omega\right)$, or, when using a plane-wave basis set, $X_{\mathbf{G G}^{\prime}}(\mathbf{q}, \omega)$. The quasiparticle ( QP ) energies $\epsilon_{m}^{\mathrm{QP}}$ are then computed either by numerically solving the exact QP equation,
$$
\begin{equation*}
\epsilon_{m}^{\mathrm{QP}}=\epsilon_{m}^{\mathrm{KS}}+\left\langle\psi_{m}^{\mathrm{KS}}\right| \Sigma\left(\epsilon_{m}^{\mathrm{QP}}\right)-v_{x c}^{\mathrm{KS}}\left|\psi_{m}^{\mathrm{KS}}\right\rangle, \tag{3}
\end{equation*}
$$
or its linearized form,
$$
\begin{equation*}
\epsilon_{m}^{\mathrm{QP}} \approx \epsilon_{m}^{\mathrm{KS}}+Z_{m}\left\langle\psi^{\mathrm{KS}}\right| \Sigma\left(\epsilon_{m}^{\mathrm{KS}}\right)-v_{x c}^{\mathrm{KS}}\left|\psi_{m}^{\mathrm{KS}}\right\rangle, \tag{4}
\end{equation*}
$$
with the renormalization factors $Z_{m}$ given by
$$
\begin{equation*}
Z_{m}=\left[1-\left.\left\langle\psi_{m}^{\mathrm{KS}}\right| \frac{\partial \Sigma(\omega)}{\partial \omega}\right|_{\omega=\epsilon_{m}^{\mathrm{KS}}}\left|\psi_{m}^{\mathrm{KS}}\right\rangle\right]^{-1} \tag{5}
\end{equation*}
$$

In the above equations we have made reference to the Kohn-Sham eigenvalues and eigenvectors, $\epsilon_{m}^{\mathrm{KS}}$ and $\left|\psi_{m}^{\mathrm{KS}}\right\rangle$, respectively.

A key quantity in the above formulation is the dynamical part of the inverse dielectric function, $Y \equiv \varepsilon^{-1}-I= v X$, which determines the correlation part of $W, W_{c} \equiv W-$
$v=Y v$, and, through Eq. (1), the correlation part of the self-energy, $\Sigma_{c}$. With the purpose of avoiding the expensive numerical evaluation of the frequency convolution in $\Sigma_{c}$, Eq. (1), as required, e.g., by full frequency real axis (FF-RA) approaches [20,47] or contour deformation (FF-CD) techniques [35,48,49], $Y$ or $X$ have been the target of several analytical simplifications like the plasmon pole approximation (PPA) [9-13], analytic continuations [50-54] or our multipole approach (MPA) [25], briefly sketched below.

\section*{B. The multipole approach}

The multipole approximation is inspired by the Lehmann representation of the polarizability $X$. At the independentparticle level, $X$ (equal to $X_{0}$ ) is written in a compact way as a sum of poles with vanishing imaginary part corresponding to all possible single-particle transitions (here considered at the Kohn-Sham level for simplicity) of energy $\Omega^{\mathrm{KS}}$ and probability amplitude $R^{\text {KS }}$ :
$$
\begin{equation*}
X_{0}(\omega)=\sum_{n}^{N_{T}} \frac{2 R_{n}^{\mathrm{KS}} \Omega_{n}^{\mathrm{KS}}}{\omega^{2}-\left(\Omega_{n}^{\mathrm{KS}}\right)^{2}}, \tag{6}
\end{equation*}
$$
where $\operatorname{Re}\left[\Omega_{n}^{K S}\right]$ is positive defined and $\operatorname{Im}\left[\Omega_{n}^{K S}\right] \rightarrow 0^{-}$to ensure the correct time ordering. The sum is truncated at a finite number of transitions ( $N_{T}$ ) determined by the number of bands included in the calculation.

The MPA approach provides an analytic continuation for the dressed polarizability $X$ to the complex-frequency plane, $z \equiv \omega+i \varpi$, by representing it as a sum of a few complex poles $n_{p}$ (usually of the order of 10 to 15 ), as
$$
\begin{equation*}
X^{\mathrm{MP}}(z)=\sum_{n}^{n_{p}} \frac{2 R_{n} \Omega_{n}}{z^{2}-\Omega_{n}^{2}} . \tag{7}
\end{equation*}
$$

Note that this representation is applied to each matrix element in reciprocal space, $X_{\mathbf{G G}^{\prime}}^{\mathrm{MP}}(\mathbf{q}, z)$.

By considering Eq. (7) and the Lehmann representation for $G_{0}$, the correlation part of the $G W$ self-energy is then integrated analytically and reads:
$$
\begin{align*}
\Sigma_{c}^{\mathrm{MP}}(\omega)= & \sum_{m}^{N_{B}} \sum_{n}^{n_{p}} P_{m} v R_{n}\left[\frac{f_{m}}{\omega-E_{m}+\Omega_{n}-i \eta}\right. \\
& \left.+\frac{\left(1-f_{m}\right)}{\omega-E_{m}-\Omega_{n}+i \eta}\right] v \tag{8}
\end{align*}
$$
where $P_{m}$ are projectors over KS states, $E_{m}$ their eigenenergies, and $f_{m}$ their occupations. The sum over states is truncated at the maximum number of bands, $N_{B}$. This expression generalizes the PPA solution to the case of a multipole expansion for $X(z)$ and bridges between PPA and an exact full frequency approach by increasing the number of poles in $X$. More details about this procedure can be found in Ref. [25].

\section*{C. Multipole approach sampling for metals}

As detailed in Ref. [25], the poles and residues in Eq. (7) are obtained by numerically evaluating $X$ for a number of frequencies, $z_{j}$, equal to twice the number of poles and solving
the following system of equations:
$$
\begin{equation*}
\sum_{n=1}^{n_{p}} \frac{2 \Omega_{n} R_{n}}{z_{j}^{2}-\Omega_{n}^{2}}=X\left(z_{j}\right), \quad j=1, \ldots, 2 n_{p} \tag{9}
\end{equation*}
$$

Since the number of poles used in the MPA model, $n_{p}$, is much smaller than the total number of electron-hole transitions of the target polarizability, $N_{T}$, the representation, and therefore the efficiency of the method, depends critically on the frequency sampling used in the interpolation. For semiconductors, the so-called double parallel sampling proved to be the most robust and accurate with respect to FF calculations, with the fastest convergence with respect to the number of poles. It runs along two parallel lines above the real axis:
$$
s^{\mathrm{DP}}=\left\{\begin{array}{l}
\mathbf{z}^{1}: z_{n}^{1}=\omega_{n}+i \varpi_{1}  \tag{10}\\
\mathbf{z}^{2}: z_{n}^{2}=\omega_{n}+i \varpi_{2}
\end{array} \quad n=1, \ldots, n_{p}\right.
$$

The first of the two branches is closer to the real axis (e.g., with $\varpi_{1}=0.1 \mathrm{Ha}$ ), except for the first point, set exactly at the origin of coordinates, $z_{1}^{1}=0$. The second branch is located further away, typically at $\varpi_{2}=1 \mathrm{Ha}$. In a simplified view, $X$ sampled along the first line preserves some of the structure of $X$ in a region close to its poles, while $X$ sampled along the second line is simple enough to be described with a few poles and accounts for the overall structure of $X$. A more detailed description can be found in Ref. [25].

To obtain a numerically stable and effective sampling for metals we found that, at variance with the semiconductor case [25], a small shift of the $z_{1}^{1}$ point (in the origin) along the imaginary axis is needed, resulting in $z_{1}^{1}=i \varpi_{1}$, where $\varpi_{1}=10^{-5} \mathrm{Ha}$. The shift is done in order to avoid numerical instabilities due to intraband transitions with energies close to zero. This is similar to the PPA implementation for metals [26,27], which adopts a $10^{-8} \mathrm{Ha}$ shift, but in this case along the positive real axis instead of the imaginary axis.

A second difference with respect the strategy used for semiconductors concerns the distribution of the frequency sampling of $X$ along the real axis. For semiconductors [25], the frequency sampling is done in nonuniform grids, in particular, a semihomogeneous partition in powers of two that ranges from zero to $\omega_{m}$, called linear partition. Here, we generalize it to any possible exponent $\alpha$ :
$$
\left\{\omega_{n}\right\}_{\alpha}:\left\{\begin{array}{l}
(0), n_{p}=1  \tag{11}\\
(0,1) \omega_{m}, n_{p}=2 \\
\left(0, \frac{1}{2}, 1\right)^{\alpha} \omega_{m}, n_{p}=3 \\
\left(0, \frac{1}{4}, \frac{1}{2}, 1\right)^{\alpha} \omega_{m}, n_{p}=4 \\
\left(0, \frac{1}{8}, \frac{1}{4}, \frac{1}{2}, 1\right)^{\alpha} \omega_{m}, n_{p}=5 \\
\left(0, \frac{1}{8}, \frac{1}{4}, \frac{1}{2}, \frac{3}{4}, 1\right)^{\alpha} \omega_{m}, n_{p}=6 \\
\left(0, \frac{1}{8}, \frac{1}{4}, \frac{3}{8}, \frac{1}{2}, \frac{3}{4}, 1\right)^{\alpha} \omega_{m}, n_{p}=7 \\
\ldots
\end{array}\right.
$$

The distribution described in Ref. [25] corresponds to $\alpha=$ 1. As discussed below, there are cases (see, for example, the case of copper in Fig. 4), in which $X$ presents a more complex structure at low frequencies and therefore a denser sampling grid in that region is convenient. The distribution
corresponding to $\alpha=2$ concentrates more points at low frequencies than the linear case, $\alpha=1$, and permits us to increase the accuracy of the $X$ description without changing the frequency range $\omega_{m}$ or increase the number of poles used in MPA. In this work, we adopt a quadratic partition, corresponding to $\alpha=2$, for Al and Cu , and a linear one, $\alpha=1$, for Na , with $\omega_{m}$ ranging up to 4,5 , and 10 Ha for $\mathrm{Al}, \mathrm{Na}$, and Cu , respectively.

\section*{D. Intraband contributions}

Despite the success of the $G W$ approximation, systems with metallic screening present specific methodological challenges, one being the inclusion of intraband transitions [31,55]. Specifically, for partially filled bands, there is a nonvanishing probability that an electron is excited within the same band, i.e., within states with quantum numbers $\mathbf{k}, n$ and $\mathbf{k}-\mathbf{q}, m$, with $n=m$. Notably, these transitions play an important role, for example, in noble metals [20,56]. Both inter- and intraband transitions contribute to the irreducible polarizability as defined in Eq. (6). However, the energy of the pole corresponding to intraband transitions decreases with $\mathbf{q}$ until it vanishes in the $\mathbf{q} \rightarrow 0$ limit. Despite this behavior, the contribution to the inverse dielectric function in the case of bulk metals is still finite, due to the divergence of the Coulomb potential, which makes $Y=v X$ not vanishing for $\mathbf{q} \rightarrow 0$. For this reason, in the case of metals it is important to properly take this term into account, since it cannot be simply evaluated as in the case of the interband contributions.

In principle, it is possible to decrease the weight of the $\mathbf{q}=$ 0 element, that contains only interband terms, by systematically increasing the number of $\mathbf{k}$-points in the Brillouin-zone (BZ) sampling. However, the contributions from the Fermi surface can dramatically slow down the convergence with respect to the $\mathbf{k}$-space sampling [29], resulting in spurious gaps at the Fermi level that vanish very slowly with increasing number of $\mathbf{k}$ points [32]. Several approaches to include the intraband limit have been proposed. Those based on explicit Fermi-surface integration [28,30,31] are, as explained above, computationally expensive since they require dense $\mathbf{k}$ grids. Alternatively, analytical models based on a Taylor expansion of the dielectric function in the small-q region, avoiding explicit Fermi-surface calculations, are able to remove the spurious gap at the Fermi level with a limited number of $\mathbf{k}$ points [32,57,58]. Nevertheless, some of them may depend on ad hoc external parameters.

A common approach to include the missing intraband contribution relies on the use of a phenomenological Drudelike term added to the head of the irreducible dielectric matrix in the $\mathbf{q} \rightarrow 0$ limit, $Y_{\mathbf{G}=\mathbf{G}^{\prime}=0}(\mathbf{q}=0, \omega)$ [30]. In the long-wavelength limit $\mathbf{q} \rightarrow 0$, the Drude term for the independent-particle dielectric function can be written in the form [24,28,30,59,60]
$$
\begin{equation*}
Y_{D}(\omega)=\frac{\omega_{D}^{2}}{\omega(\omega+i \gamma)}+O\left[\mathbf{q}^{2}\right] \tag{12}
\end{equation*}
$$
where the Drude frequency $\omega_{D}$ (see Table I) is an input parameter of the model and the relaxation frequency $\gamma$ is usually a

\begin{table}
\captionsetup{labelformat=empty}
\caption{TABLE I. Summary of the notation concerning frequency related quantities introduced in this work. The plasma frequency is defined in terms of the electronic density $\rho_{e}$. The Drude frequency or pole are model parameters used to describe the plasmon or only its intraband contribution, as described in Eq. (12).}
\begin{tabular}{lcc}
\hline \hline Contribution & Pole (complex) & Frequency (real) \\
\hline Intra-band & $\Omega_{A}$ & $\omega_{A}=\operatorname{Re}\left[\Omega_{A}\right]$ \\
Inter-band & $\Omega_{E}$ & $\omega_{E}=\operatorname{Re}\left[\Omega_{E}\right]$ \\
Plasmon (intra + inter) & $\Omega_{p}$ & $\omega_{p}=\operatorname{Re}\left[\Omega_{p}\right]$ \\
Plasma & & $\omega_{\mathrm{pl}}=\sqrt{4 \pi \rho_{e}}$ \\
Drude (model) & $\omega_{D}+i \gamma$ & $\omega_{D}$ \\
\hline \hline
\end{tabular}
\end{table}
free parameter set typically to $\gamma=0.1 \mathrm{eV}$. In principle $\omega_{D}$ can be determined fully $a b$ initio, resorting to very dense $\mathbf{k}$-point grids [20,30] or to an interpolation of the BZ, for instance with Wannier functions [61-64] or the tetrahedron method [29,30,40,65]. Alternatively, experimental values can also be used when available.

In the next sections we discuss the possibility to extrapolate a complex plasmon frequency (see Table I) in the $\mathbf{q} \rightarrow 0$ limit from the frequency structure of $Y(\mathbf{q}, \omega)$ at finite $\mathbf{q}$, which in general is a superposition of intra- and interband contributions. In a second step, we will use a $f$-sum rule [24] in the same spirit of Ref. [30], in order to estimate the intraband contribution to the plasmon frequency. We also propose a simple and virtually zero-cost method to include an approximate treatment of the missing intraband limit from first-principles, without the need to resort to any add-on model.

\section*{III. RESULTS AND DISCUSSION}

In the following, we present the results for three bulk metallic systems highlighting different issues arising when applying the $G W$ approach to metals. We start by studying the case of two simple metals, Al and Na (see, e.g., Refs. [66-68] for a description of their band structures). Next, we focus our attention on Cu , a more challenging system whose electronic structure has been thoroughly studied, both experimentally [69,70] and theoretically [20,71-73]. The use of PPA for Cu has been shown to be problematic [20] and, for this reason, copper is not only an important test case for the application of MPA and the description of intraband effects, but also provides a better understanding of the applicability of PPA.

As a starting point for our $G W$ simulations, we use DFT calculations performed at the PBE [74] level using scalar-relativistic optimized norm-conserving Vanderbilt pseudopotentials [75], as implemented in the QUANTUM ESPRESSO package [76,77]. The kinetic-energy cutoff is set to 100 , 70 , and 150 Ry for $\mathrm{Al}, \mathrm{Na}$, and Cu , respectively. The $\mathbf{k}$ grids were determined by the convergence requirements of the $G W$ calculations, considering, in particular, the specific treatment of the intraband limit. When reporting quasiparticle energies, we use $\mathbf{k}$-point grids of $16 \times 16 \times 16$ for Al and Na , and $12 \times 12 \times 12$ for Cu . Moreover, the GW correction to the Fermi level is linearly interpolated from the corresponding corrections to the closer quasiparticles present in the specific k mesh.

\begin{table}
\captionsetup{labelformat=empty}
\caption{TABLE II. Al and Na quasiparticle energies (eV) with respect the Fermi level computed within DFT-PBE, GW-PPA, and GWMPA using a $16 \times 16 \times 16 \mathbf{k}$ grid including the $\mathbf{q} \rightarrow 0$ intraband contribution through the constant approximation (CA) method (see Sec. III C).}
\begin{tabular}{lcccc}
\hline \hline & & DFT-PBE & GW-PPA & GW-MPA \\
\hline Al & $\Gamma_{1}$ & -11.12 & -10.79 & -10.94 \\
Al & $\Gamma_{25^{\prime}}$ & 12.71 & 12.30 & 12.48 \\
Al & $X_{4^{\prime}}$ & -2.93 & -2.91 & -2.86 \\
Al & $W_{3}$ & -0.85 & -0.83 & -0.82 \\
Na & $\Gamma_{1}$ & -3.27 & -2.85 & -2.97 \\
Na & $\Gamma_{25^{\prime}}$ & 11.76 & 11.19 & 10.81 \\
\hline \hline
\end{tabular}
\end{table}

The DFT results are in good agreement with previous results obtained with the same method [72] and in reasonable agreement with the results reported for Cu in Ref. [20], performed using the LDA [31]. In fact, the $G W$ results for Cu have shown to be very sensitive to the choice of the DFT starting point [72], although we will not address this point here. The $G W$ calculations were done using the yambo [26,27] code. The numerical convergence of the $G W$ results has been checked with care, and the resulting parameters, being system dependent, are detailed in the sections below when discussing the results.

\section*{A. Multipole approach for simple metals}

We start by computing quasiparticle energies of Al and Na using MPA. Here the frequency dependence of the polarizability presents a structure with mainly one strong plasmon peak, similar to that of silicon computed in Ref. [25]. As expected, the double parallel sampling ensures convergence with a similar number of poles, $n_{p}=8$. The present results were obtained considering 300 bands for both $X$ and $\Sigma$ and an energy cutoff for $X$ of 20 and 15 Ry for Al and Na , respectively.

In Table II we report the quasiparticle energies for Al and Na , including $\Gamma_{1}$ (the lowest QP peak at $\Gamma$, corresponding to the valence bandwidth) and other reference quasiparticles, computed using PPA and MPA. MPA QPs are generally in very good agreement with FF values from the literature (see, e.g., Ref. [19] and references therein). According to our calculations, the computed quasiparticles values for Al and Na with MPA are estimated to differ by less than 8 meV from the corresponding FF-RA results (comparison done using 10 Ry cutoff to represent $X_{0}$ for both MPA and FF-RA), as found for semiconductors [25]. Instead, PPA QPs show deviations that are systematically larger for states further from Fermi.

Previous $G W$ calculations for Al and Na [19] have shown that PPA describes well the tail of the self-energy, i.e., the frequency region around the Kohn-Sham energies, and gives reasonable QP solutions for both Al and Na . However, if we consider the whole frequency range, the agreement between PPA and FF-CD is less satisfactory. PPA shows sharp fluctuations in the self-energy and spectral functions that result in several spurious solutions of the quasiparticle equation, evinced by multiple small peaks in the spectral function (see,

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3d8b6fda-3544-4cf1-8ed7-c015c8fb8cd8-05.jpg?height=1148&width=1782&top_left_y=197&top_left_x=139}
\captionsetup{labelformat=empty}
\caption{FIG. 1. (a), (c) Frequency dependence of the real part of the self-energy and (b), (d) spectral function computed with MPA for three quasiparticles of (a), (b) Al and (c), (d) two of Na , including the intraband limit using the constant approximation (CA) (presented in Sec. III C). In the case of Na , we also show the corresponding curves without any intraband correction (nD).}
\end{figure}
e.g., Fig. 4 of Ref. [19]). In Fig. 1 we show the self-energy and spectral function for Al and Na , this time computed with MPA. The comparison with results obtained within FF-CD [19] shows that MPA not only describes well the tail of the $X(\omega)$ and $\Sigma(\omega)$ functions but also correctly describes the positions of the peaks and their relative intensities in the whole frequency range.

The left panels of Fig. 1 correspond to Al plots of the MPA self-energy, $\left\langle\psi_{m \mathbf{k}}\right| \Sigma(\omega)\left|\psi_{m \mathbf{k}}\right\rangle$, and the spectral function, $\left\langle\psi_{m \mathbf{k}}\right| \operatorname{Im}[G(\omega)]\left|\psi_{m \mathbf{k}}\right\rangle$, as a function of the frequency. These quantities have been projected on three Al states, one corresponding to the bottom of the valence band at $\Gamma$ and two other Kohn-Sham states closer to the Fermi level. Comparing the three self-energy functions, there is a more effective pole superposition for states at energies further away from the Fermi level. Indeed, for the lowest energy state with $\mathrm{E}^{K S}=-11.2 \mathrm{eV}$, this leads to a frequency dependence of $\Sigma$ with an intense single pole (at about -15 eV with respect to $\mathrm{E}^{K S}$ ) and consequently a very broad and shallow QP peak in the corresponding spectral function. At the same time the satellite structure is enhanced to the point of becoming a second peak, originating from a second solution of the quasiparticle equation (intersections of the dashed line with the self-energy function in the upper panel). This scenario is consistent with the so-called "plasmaron" peak, a sharp satellite
feature emerging as an artifact of the $G_{0} W_{0}$ approximation to the self-energy [2,45,78].

The situation is similar for the two QPs computed for Na shown in the rights panel of Fig. 1, with the lowest state presenting again two solutions for the QP equation.

\section*{B. Analysis of the intraband contribution}

In common $G W$ implementations, especially those targeting semiconductors, the intraband contribution to the dielectric function in the $\mathbf{q} \rightarrow 0$ limit, Eq. (6), is often not included, as explained in Sec. II D. In the case of Al, where a substantial part of the Fermi surface is very close to the BZ boundary, one can expect [32] that many of the metallic contributions are effectively interband rather than intraband terms, resulting in a small error when the intraband limit is neglected [32], while for Na it is found to be more relevant.

For both Al and Na , in Fig. 2 we show how this affects the frequency dependence of the $Y_{\mathbf{G}=\mathbf{G}^{\prime}=0}$ matrix elements computed for different $\mathbf{q}$ vectors along an arbitrary direction. The curves in green shades correspond to $Y(\omega)$ computed for finite but small $\mathbf{q}$. The orange curve corresponds to the $\mathbf{q} \rightarrow 0$ limit evaluated only for the interband term. There are two main differences between the green and orange curves. The first difference is the limit of $\operatorname{Re}[Y]$ as the frequency tends to

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3d8b6fda-3544-4cf1-8ed7-c015c8fb8cd8-06.jpg?height=957&width=1778&top_left_y=197&top_left_x=124}
\captionsetup{labelformat=empty}
\caption{FIG. 2. Frequency dependence of $Y_{\mathbf{G}=\mathbf{G}^{\prime}=0}$ matrix elements computed with MPA for different $\mathbf{q}$ vectors of modulus $q \equiv|\mathbf{q}|$ tending to zero for (a), (b) Al, and for (c), (d) Na. For $q=0$ (orange curves) the intraband transitions are not included. The insets in panels (a) and (c) show the region around $\omega=0$. Panel (e) shows the $q$ dispersion of the real and the imaginary parts of the main pole of $Y$ for $q_{0}-q_{4}\left(q_{n}=\frac{n}{8}\right.$ in units of $2 \pi / a$, being $a$ the respective Al and Na lattice parameters). The solid lines show the corresponding parabolic fits consistent with a Lindhard (bulk) plasmon dispersion [30,32,79,80]. The black dashed lines correspond to the experimental plasmon frequency of Al [30] and Na [79]. Dash-dot purple lines correspond to the values of the intraband frequency, $\omega_{A}$, computed in Ref. [19] using the method described in Ref. [57], while the violet one corresponds to our estimate for Al , computed by means of Eq. (14).}
\end{figure}
zero (static limit), which evolves smoothly for finite $\mathbf{q}$ but in general tends to a value different from the one corresponding to $\mathbf{q}=0$. As shown in the insets of Fig. 2, the smallest finite $\mathbf{q}$ provides a static limit very similar to the value for $\mathbf{q}=0$ in the case of Al , while it is considerably larger in the case of Na (both results in agreement with previous studies [32]).

This difference has been commonly used as a measure of the missing intraband term [19,32], since for metals in the limit $\mathbf{q} \rightarrow 0, \varepsilon_{\mathbf{G}=\mathbf{G}^{\prime}=0}^{-1}(\mathbf{q}, \omega=0)$ vanishes, meaning that $Y_{00}(\mathbf{q}, \omega=0) \rightarrow-1$, as apparent from the progression of the curves with finite $\mathbf{q}$, that include intraband transitions. In fact, in the independent-particle picture, the $\mathbf{q} \rightarrow 0$ limit of $\operatorname{Re}[Y]$ at $\omega=0$ is related to a nonvanishing probability of vertical transitions within the same band [30], and can therefore be used to estimate a Drude frequency [81,82]. However, this probability alone does not determine the plasmon frequency (see Table I for a summary of the nomenclature) or the position of the pole of $\operatorname{Re}[Y]$ for $\mathbf{q} \rightarrow 0$.

In fact, the second difference between the orange ( $\mathbf{q}=0$, no intraband contribution) and the green curves (finite $\mathbf{q}$, intraband included) in Fig. 2 is the position of the main pole of $Y(\omega)$, here called $\Omega_{p}$, or in the case of Na , to the apparent absence of poles for $\mathbf{q}=0$, whose small amplitudes cannot be seen in the plot. If the whole frequency range is considered, we see that the behavior of $\operatorname{Re}[Y(\omega \rightarrow 0)]$ depends on the position of $\Omega_{p}$. Following the green curves at finite $\mathbf{q}$, it is clear that $Y_{\mathbf{G}=\mathbf{G}^{\prime}=0}$ for both Al and Na change smoothly with $\mathbf{q}$. The curves present a pole $\Omega_{p}(\mathbf{q})$ of decreasing energy and
increasing amplitude, just above 0.5 Ha for Al and 0.2 Ha for Na . As shown in Fig. 2(e), both the real and imaginary part of this pole can be easily extrapolated to $\mathbf{q}=0$, by means of the Lindhard plasmon dispersion [30,32,79,80].

In the same plot we show, as a reference, the Drude frequency corresponding to the $\mathbf{q} \rightarrow 0$ limit of the intraband contributions, $\omega_{A}$ (see Table I), as computed in Ref. [19] for Al and Na , in addition to the experimental plasmon frequency $\omega_{p}$ of Al [60,79,83-86] and Na [79]. In the simulations we can also extrapolate, already with a $8 \times 8 \times 8 \mathbf{k}$-point mesh, the plasmon frequency at $\mathbf{q} \rightarrow 0$ from the position of the main structure of the response functions, namely $\omega_{p} \equiv \operatorname{Re}\left[\Omega_{p}\right]$. This procedure provides $\omega_{p}=0.55 \mathrm{Ha}(15.01 \mathrm{eV})$ for Al , in excellent agreement with the experimental value of 15.0 eV [30]. Similarly, the value extrapolated for $\mathrm{Na}, \omega_{p}=0.21 \mathrm{Ha} (5.79 \mathrm{eV})$, matches very well the experimental value of 5.9 eV [79] and both compare well with the Drude intraband frequency computed in Ref. [19] ( 6.18 eV ). The small difference between our theoretical result and Ref. [19] can be attributed to methodological differences (e.g., the DFT functional on top of which the $G W$ calculations are performed). In contrast, the difference between $\omega_{p}$ and $\omega_{A}$ for Al is larger than 2.5 eV since the plasmon frequency $\omega_{p}$ has non-negligible contributions from both intra- and interband transitions, as previously reported in Refs. [30,60]. Note however that the interband contributions are not included in the Drude frequency computed in Ref. [19].

To discriminate between the intra- and interband contributions to the plasmon frequency, we have used a simple expression based on the $f$-sum rule [2,30,87], but separating the two contributions:
$$
\begin{equation*}
\Omega_{A}^{2}=-\lim _{\mathbf{q} \rightarrow 0} \frac{2}{\pi} \int_{0}^{\infty} d \omega \omega \operatorname{Im}\left[Y(\mathbf{q}, \omega)-Y_{E}(\mathbf{q}, \omega)\right] \tag{13}
\end{equation*}
$$
where $Y_{E}$ corresponds to interband transitions only, while $Y$ accounts for the complete response. Within MPA the integral is solved analytically (derivation in Sec. I of the Supplemental Material [88]), leading to
$$
\begin{equation*}
\Omega_{A}^{2}=2 v\left(R_{p} \Omega_{p}-R_{E} \Omega_{E}\right) \tag{14}
\end{equation*}
$$
where $\Omega_{E}$ and $v R_{E}$ are the position and the residue of the most relevant pole of $Y_{E}(\mathbf{q}=0)$, while $\Omega_{p}$ and $v R_{p}$ are the corresponding values for $Y(\mathbf{q}=0)$.

In principle, the product $v R_{p} \Omega_{p}$ should be computed in the $\mathbf{q} \rightarrow 0$ limit. We have instead considered the extrapolation of $\Omega_{p}^{2}$, which is equivalent in our model (see Sec. I in the Supplemental Material [88]) and significantly more stable. The values of $v R_{E} \Omega_{E}$ are taken directly from the calculation at $\mathbf{q}=0$ [orange curves in Figs. 2(a) and 2(b)], since no intraband transitions are considered, as explained above. For Al , the real part of $\Omega_{E}$ is $\omega_{E}=0.37 \mathrm{Ha}(10.08 \mathrm{eV})$ and thus, applying Eq. (14), the real part of the intraband pole $\Omega_{A}$ is $\omega_{A}=0.43 \mathrm{Ha}(11.72 \mathrm{eV})$. For $\mathrm{Na}, \omega_{p}$ and $\omega_{A}$ are similar. The comparison of $\omega_{p}$ and $\omega_{A}$ confirms that the experimental plasmon frequency, $\omega_{p}$, in the case of Na corresponds mainly to intraband contributions, while for Al there is an important interband contribution [60], and its use as a Drude intraband frequency would result in an overestimation of the actual $\omega_{A}$.

Making use of the extrapolation procedures described above in the context of the MPA framework, and of a simple $f$-sum rule, it is possible to determine not only the real but also the imaginary part of both the plasmon and the intraband pole, usually not considered in other $a b$ initio methods. It is also worth noticing that the extrapolation is done with points from a much coarser $\mathbf{k}$ grid ( $8 \times 8 \times 8$ for both Al and Na ), with respect to the grids required to compute the intraband frequency with an independent-particle formulation [31,32].

Despite the limited accuracy of the computed imaginary values, they are meaningful and provide a qualitative understanding of how intra- and interband terms, linearly summed at the independent-particle level, are combined after the inversion of the Dyson equation. While the Na case is trivial, since the interband contribution is negligible, in the case of Al the small difference between $\omega_{A}$ and $\omega_{E}$, comparable to their imaginary parts, explains the presence of a single pole in $Y(\omega)$ located roughly at $\omega_{p}^{2} \sim \omega_{A}^{2}+\omega_{E}^{2}$ (see Sec. I of the Supplemental Material [88]).

\section*{C. Modeling of the intraband limit}

Our analysis of the dressed response function $Y(\omega)$ suggests that an alternative to the direct evaluation of the intraband limit, usually determined from $X$ at the independentparticle level [31], can be obtained, either by (1) including a complex Drude pole $Y_{D}(\omega)$, according to Eq. (12), in the head $\left(\mathbf{G}=\mathbf{G}^{\prime}=0\right)$ of the independent-particle dielectric function, with the Drude frequency given by the computed intraband
pole; or (2) approximating the full $Y(\mathbf{q}=0)$ matrix element by its nearest neighbor $Y(\mathbf{q} \neq 0)$, i.e., with the $\mathbf{q}$-vector closest to zero according to the adopted $\mathbf{k}$-point grid.

The first method builds on using an estimate of the Drude intraband frequency, similar to the extrapolations used in Ref. [82], but here considering the whole frequency range and both intra- and interband contributions. The second method, which we will call from now on constant dielectric function approximation (CA), assumes that the whole $Y(\mathbf{q})$ matrix is constant in a small region around $\mathbf{q}=0$. This approach is inspired by the leading term of the Taylor expansion for small $\mathbf{q}$ of the Thomas-Fermi distribution, and is corroborated by the small difference of $0.006 \mathrm{Ha}(0.17 \mathrm{eV})$ found for both, Al and Na , between the extrapolated value of $\Omega_{p}$ and its value at the first finite $\mathbf{q}$, as shown in Fig. 2(e). Both methods simultaneously correct the position of the plasmon pole and the limit of $Y_{\mathbf{G}=\mathbf{G}^{\prime}=0}$ for $\omega=0$ and add virtually no computational cost to the calculation. In addition, CA also corrects other matrix elements for which the intraband limit may be important.

In Sec. II of the Supplemental Material [88] we report plots similar to those in Fig. 2 for $Y$ matrix elements of Na other than the head, showing that after the head $\left(Y_{\mathbf{G}=\mathbf{G}^{\prime}=0}\right)$, intraband contributions are relevant also for the so-called wing elements $\left(Y_{\mathbf{G}=0 \neq \mathbf{G}^{\prime}}\right.$ and $\left.Y_{\mathbf{G} \neq 0=\mathbf{G}^{\prime}}\right)$, while less important for the diagonal elements $\left(Y_{\mathbf{G}=\mathbf{G}^{\prime} \neq 0}\right)$, especially at increasing $|\mathbf{G}|$. For finite $|\mathbf{G}|$ the evolution of the $Y(\mathbf{q})$ matrix elements when $\mathbf{q} \rightarrow 0$ is less smooth and the position of the poles does not always change monotonically, meaning that an extrapolation would require a denser $\mathbf{k}$-point grid. Even if the constant dielectric function approximation has limited accuracy for some of these matrix elements, it still provides a significant overall improvement. In particular in materials such as Cu , as discussed below, the CA method presents some clear advantages regarding the estimation of $\omega_{A}$.

To assess the effect of this approximation in the QP solution, in Fig. 3 we show Al and Na QP energies computed without (nD) and with (CA) intraband corrections. When the number of $\mathbf{k}$ points is increased, the weight of the $Y(\mathbf{q}=0)$ element in the self-energy decreases and both methods eventually converge to the same quasiparticle values, but only very slowly, as discussed above. In fact, Fig. 3 shows that for two selected QPs of Na the intraband term is fundamental due to the importance of this contribution to the screening properties of the system. In contrast, for Al the difference is small, and the convergence is governed by the inter- rather than intraband contributions for all four QPs considered. In the bottom panel of Fig. 3 one can see the significant acceleration introduced by CA in the convergence of the bandwidth of Na , where, besides a small oscillation in the $20 \times 20 \times 20$ grid (caused by oscillations in the DFT eigenvalues), the first point corresponding to the $8 \times 8 \times 8$ mesh already provides very accurate results. In the CA scheme the convergence benefits simultaneously from the decrease of the weight of the $Y(\mathbf{q}=0)$ contribution and from the fact that the correction itself improves for denser grids in reciprocal space, since the first $\mathbf{q} \neq 0$ is closer to zero.

In Fig. 1 we show the frequency dependence of the real part of the self-energy (top) and spectral function (bottom) computed for two quasiparticles of Na , within MPA with and without the CA intraband correction. The correction does not

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3d8b6fda-3544-4cf1-8ed7-c015c8fb8cd8-08.jpg?height=1166&width=770&top_left_y=199&top_left_x=180}
\captionsetup{labelformat=empty}
\caption{FIG. 3. (top panel) Difference between GW-MPA corrections computed with the (CA) intraband term and without (nD) as a function of the number of $\mathbf{k}$ points, for two quasiparticles of Na (light and dark yellow) and four of Al (green shades). (bottom panel) Convergence of the $G W$ correction for the QP at $\Gamma_{1}$ of Na (yellow) and Al (green) with CA (solid) and nD (dashed).}
\end{figure}
change dramatically the shape of the self-energy, but introduces an extra pole in the real part of the self-energy at the intraband frequency ( $\approx-6 \mathrm{eV}$ ) and renormalizes the peaks of the spectral function. The inclusion of this term promotes the pole overlapping around the plasmon frequency, which is already accurate without CA. However, CA increases the intensity of the plasmon affecting the tail of the self-energy and thus the QP solution, as illustrated in the insets of Fig. 1, differently for each quasiparticle.

A similar picture is found for Al , for which the intraband transitions are less important. In this case, the change in intensity of the plasmon is smaller and the plasmon is located farther from $\omega=0$. Therefore, the effects of CA on the tail of the self-energy and consequently, on the QP solution are much less important.

For both Al and Na , the QP energies computed with the Drude model, Eq. (12) using as input $\omega_{D}=\omega_{A}$, and the CA schemes are very similar, with differences below 20 meV when using the $8 \times 8 \times 8 \mathbf{k}$ grid. This leads us to conclude that the CA scheme could replace the usual Drude correction, replacing a semiempirical scheme by a simple $a b$ initio approximation. This is particularly relevant when the Drude intraband frequency is difficult to estimate either from
experiments or calculations, since the CA scheme has virtually zero computational cost and, as the extrapolation presented in the previous section, describes both the real and the imaginary part of $Y$.

To summarize this section, the inclusion of the intraband limit through the proposed CA scheme requires no extra computational cost with respect to the standard $G W$ calculation and accelerates the $\mathbf{k}$-grid convergence of the QP energies for systems where the intraband contribution dominates, like Na , without resorting to semi-empirical corrections such as the Drude model or computationally costly ab initio approaches.

\section*{D. Frequency representation of the response function of copper}

As mentioned before, the case of copper presents several challenges for an accurate $G W$ description. The Cu band structure features a series of flat $d$ bands around 2 eV below the Fermi level, leading to strong transitions in $Y_{\mathbf{G}, \mathbf{G}^{\prime}}(\mathbf{q}, \omega)$ spread over a large energy range [20]. As shown in Fig. 4 for $\mathbf{q}=0$, even for small values of $\mathbf{G}$ and $\mathbf{G}^{\prime}, Y_{\mathbf{G}, \mathbf{G}^{\prime}}(\mathbf{q}, \omega)$ can behave very differently from a single pole case, hindering the use of PPA but suggesting that a multipole approach could prevent resorting to more expensive FF methods.

When considering PPA or in general MPA with only a few poles, one of the main issues is that the interpolation of $X$ or $Y$ may give rise to nonphysical poles, posing representability problems. Within the Godby and Needs (GN) PPA scheme implemented in yambo [11,26,89,90], the condition used to identify these so-called unfulfilled modes is the following:
$$
\begin{equation*}
\operatorname{Re}\left[\frac{Y_{\mathbf{G G}^{\prime}}(\mathbf{q}, 0)}{Y_{\mathbf{G G}^{\prime}}\left(\mathbf{q}, i \varpi_{\mathrm{pl}}\right)}-1\right]<0, \tag{15}
\end{equation*}
$$
$\varpi_{\mathrm{pl}}$ being a frequency on the imaginary axis used to perform the GN interpolation, typically set to $\varpi_{\mathrm{pl}}=1 \mathrm{Ha}$ or to a value of the order of the plasma frequency ( $\varpi_{\mathrm{pl}} \gtrsim \omega_{\mathrm{pl}}$ ), computed from the electronic density, $\rho_{e}$ (see Table I). As an example, for the diagonal elements ( $\mathbf{G}=\mathbf{G}^{\prime}$ ), the polarizability evaluated on the imaginary axis should be real and therefore unfulfilled modes are those for which the resulting pole is instead imaginary. In these cases, the position of the pole is typically set to $\Omega_{\text {fail }}^{\mathrm{GN}}=1 \mathrm{Ha}$.

Setting the pole at $\Omega_{\text {fail }}^{\mathrm{GN}}$ usually works well for simple semiconductors [25,89]. However, in more complex systems it can compromise the PPA approach. In fact, when performing GW calculations using GN-PPA for Cu , we found that no less than $48 \%$ of the matrix elements are unfulfilled modes. This means that, for almost half of the matrix elements, the position of the pole is spuriously set to 1 Ha , severely affecting the selfenergy and the quasiparticle solution, as shown in the insets of Fig. 5. Within the MPA, increasing the number of poles in the description of $Y$, together with the generalized condition to assign the position of the poles of the unfulfilled modes, as described in Ref. [25], leads to a significant improvement in the representability of $Y$, as illustrated in Sec. III of the Supplemental Material [88].

In Fig. 4 we compare selected $Y$ matrix elements computed within MPA with 1 and 12 poles, with the FF results computed with a frequency grid of 1000 points (all other convergence parameters being the same: $\mathbf{k}$ grid, number of empty bands, etc.) At first glance, the enveloping structure

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3d8b6fda-3544-4cf1-8ed7-c015c8fb8cd8-09.jpg?height=953&width=1782&top_left_y=197&top_left_x=139}
\captionsetup{labelformat=empty}
\caption{FIG. 4. Selected $\mathrm{Cu} Y(\mathbf{q}=0)$ matrix elements computed within MPA with 1 and 12 poles compared with the corresponding FF results. The $y$ axes are scaled with the factors indicated on top of each panel.}
\end{figure}
of diagonal elements presents a strong overall peak, as in the case of semiconductors such as $\mathrm{Si}, \mathrm{hBN}$, and $\mathrm{TiO}_{2}$, which are well-described within the PPA and MPA [25]. However, in the case of Cu , there are other important peaks close to the origin not captured by a single-pole model. In this case, PPA
quasiparticle energies are not just numerically inaccurate, as in the case of the discussed semiconductors, but PPA becomes an inadequate model. Increasing the number of poles from 1 to 12 significantly improves the agreement between $Y$ computed with MPA and FF, reproducing the overall frequency

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3d8b6fda-3544-4cf1-8ed7-c015c8fb8cd8-09.jpg?height=903&width=1771&top_left_y=1557&top_left_x=146}
\captionsetup{labelformat=empty}
\caption{FIG. 5. Frequency dependence of the real part of the self-energy (top) and spectral function (bottom) of three quasiparticle states of Cu : (a), (b) one close to the Fermi energy; (c), (d) $\Gamma_{12}$; and (e), (f) $\Gamma_{1}$ computed with PPA, MPA, and FF. The intersections with the dotted lines represent the graphical solutions of Eq. (3).}
\end{figure}
dependence even if MPA presents a much smoother shape. While the rapid oscillations in the FF response function are enhanced by the discretization of the Brillouin zone, the origin of such fluctuations can be related to the topology of the flat $d$ bands of Cu [20], consistently, e.g., with the very structured $W(\omega)$ computed for Ni [91]. In fact, regardless of the overall simple shape of $X$, numerous interband transitions, close in energy and not effectively overlapped, contribute to the fluctuations of the polarizability $X$ and of the inverse dielectric function $Y$, when computed within FF. Nevertheless, as discussed in the next section, they do not significantly influence the computed $G W$ quasiparticle energies.

\section*{E. Quasiparticles and spectral function of copper}

In Fig. 5 (top panels) we show the frequency dependence of the self-energy projected on three selected quasiparticle states of Cu calculated within PPA, MPA, and FF-RA. The details of $\Sigma$ computed within the FF approach, better appreciated in Fig. 5(c), depend on the fine structure of $W$, which requires a dense frequency grid when computing the polarizability, as shown in Sec. V of the Supplemental Material [88]. Since these calculations are very expensive, the curves shown in Fig. 5 were computed including 200 bands for all the three methods, and using a frequency grid with 1000 points for FF and no intraband correction. Fully converged MPA results and intraband corrections are discussed at the end of this section.

The FF-RA self-energy presents a rather flat structure with no dominant peaks. Since $\Sigma$ is obtained from the convolution of $G$ and $W$ in Eq. (1), the oscillations of $W$ are attenuated, resulting in a much smoother function. Nevertheless, the convergence of the QP solution is challenging, since it requires an accurate description of the tail of the self-energy, as shown in the insets of Fig. 5. This could explain, at least in part, the variety of results present in the literature.

PPA results (blue curves in Fig. 5) show that the quasiparticle solution (insets of Fig. 5) obtained with a single pole model for $W$ deviates from the FF-RA solution. Besides the deviations at the tail of $\Sigma$, PPA fails to describe the frequency dependence of $\Sigma$ and the spectral function (bottom panels). On the other hand, the MPA results, here obtained with 12 poles and the quadratic sampling, are very accurate, not only in the tail region, that determines the QP corrections, but also for the whole frequency range of both the self-energy and the spectral function. The difference between MPA with 12 poles and FF-RA QP energies, computed as the graphical solution of Eq. (3), are smaller than 8 meV for $\Gamma_{12}$ and 30 meV for $\Gamma_{1}$, while in the case of PPA they range from 180 to 420 meV .

Comparing the three selected quasiparticle states in Fig. 5, the effect of the overlapping of the independent-particle excitations (due to the inclusion of local field effects via the Dyson equation for $W$ ) on the self-energy of Cu is more relevant for $\Gamma_{1}$ than for $\Gamma_{12}$ and the QPs around the Fermi energy. Indeed, as shown in the bottom panels of Fig. 5, for the QPs closer to the Fermi level, the shape of the spectral function has a very narrow quasiparticle peak and three satellites. When compared with the QPs close to Fermi, the QPs at deeper energies ( $\Gamma_{12}$ and $\Gamma_{1}$ ) present a broader quasiparticle peak and more intense satellites. The shallower satellite (above
-10 eV ) forms a shoulder structure for $\Gamma_{12}$ (central panel) and eventually merges with the QP peak to form a single broader peak for $\Gamma_{1}$ (right panel). Despite its complexity, the Cu states at different energies present similar trends as the cases of Al and Na discussed in Sec. III A.

It is worth emphasizing the importance of the frequency sampling in MPA. Since copper $X$ and $Y$ present a rich structure at low frequencies, but the energy range $\omega_{m}$ in Eq. (11) is still large, the quadratic sampling has shown to be more efficient than the linear one. Specifically, it provides, with the same number of poles and the same $\omega_{m}$, a larger density of points in the low-frequency region and therefore higher accuracy. The comparison between the computational cost of MPA and the FF-RA method can be done in a simplified way by comparing the number of frequencies for which $X$ is numerically computed in each approach. Here, for MPA we use 24 frequency points, corresponding to 12 poles, while the FF-RA frequency grid has 1000 points, corresponding to a 40 times gain in computational efficiency of MPA with respect to FF-RA.

The convergence with respect to the number of bands and the size of the $X$ matrices is particularly challenging, as already reported for example for other systems with $d$ states [92-94], with a slow, nonmonotonic convergence that hinders the use of extrapolations (see more detail in Sec. V of the Supplemental Material [88]). For this reason, the computational efficiency of MPA is particularly beneficial as it allows for the use of fine $G W$ convergence parameters, thereby increasing the overall accuracy of the results.

In Table III we show the MPA results obtained with 60 Ry of energy cutoff and 1000 bands for both $X$ and $\Sigma$. These parameters are comparable to the largest ones used within a static subspace approximation [73]. The reported MPA quasiparticle energies are in good agreement with previous calculations using different FF approaches and are summarized in Table III. The main differences can be explained by the use of different starting points for the $G W$ calculation, i.e., different exchange-correlation functionals and/or pseudopotentials in the DFT ground state, and possibly to an incomplete convergence of some of the results. While the use of converged parameters is essential when comparing the computed QP energies with experiments, $G W$ corrections do not always improve over DFT or PBE results, as also observed in Refs. [72,73]. In the present case, $G W$ significantly improves $\Gamma_{1}$, while for $\Gamma_{12}$ and other QPs, the $G W$ correction is rather small and slightly worsens the DFT results. The localized nature of the $d$ states in Cu may require methods beyond $G W$ in order to further improve the agreement with experiments [37,46,95,96].

\section*{F. Intraband effects in copper}

To investigate the intraband contributions of copper, in Fig. 6 we show the frequency dependence of the $Y_{\mathbf{G}=\mathbf{G}^{\prime}=0}$ matrix elements computed for the smallest $\mathbf{q}$ vectors along one direction of a $16 \times 16 \times 16 \mathbf{k}$ grid. Since $Y(\omega)$ of Cu is very structured at small frequencies, where the effects of the intraband contributions are expected to be stronger, we have used MPA with a quadratic sampling, Eq. (11) with $\alpha=2$ and $n_{p}=15$, a number of poles slightly larger than the value

\begin{table}
\captionsetup{labelformat=empty}
\caption{TABLE III. DFT and $G W$ quasiparticle energies of Cu computed with different methodologies by different groups and compared with the experimental values. All the $G W$ calculations correspond to FF approaches ran on top of LDA [20] and PBE [72].}
\begin{tabular}{|l|l|l|l|l|l|l|l|}
\hline QP (eV) & DFT/LDA Ref. [20] & DFT/PBE Ref. [72] & DFT/PBE (current work) & GW@LDA Ref. [20] & GW@PBE Ref. [72] & GW@PBE (current work) & Expt. Ref. [69] \\
\hline $\Gamma_{12}$ & -2.27 & -2.05 & -2.18 & -2.81 & -1.92 to -2.11 & -2.12 & -2.78 \\
\hline $\Gamma_{1}$ & -9.79 & -9.29 & -9.27 & -9.24 & -9.14 to -9.20 & -9.06 & -8.60 \\
\hline $X_{5}$ & -1.40 & -1.33 & -1.49 & -2.04 & -1.45 to-1.22 & -1.39 & -2.01 \\
\hline $L_{2^{\prime}}$ & -1.12 & -0.92 & -0.99 & -0.57 & -0.98 to -1.02 & -1.05 & -0.85 \\
\hline $L_{3}$ & -1.63 & -1.47 & -1.63 & -2.24 & -1.58 to -1.36 & -1.57 & -2.25 \\
\hline $L$ gap & 5.40 & 4.80 & 4.66 & 4.76 & 4.98 to 5.09 & 4.88 & 4.95 \\
\hline
\end{tabular}
\end{table}
needed to converge the quasiparticle energies. In contrast with Na , the orange curve ( $\mathbf{q}=0$, no intraband contribution) presents a similar shape and scale with respect to the green curves (small but finite $\mathbf{q}$, with intraband contributions), even if with less intense peaks.

In the right panel of Fig. 6 we show the position of the first four poles of $Y(\omega)$ as a function of $\mathbf{q}$, which present a rather flat dispersion, when compared with the plasmon dispersion of Al in Fig. 2. As expected, for $\mathbf{q}=0$ the position of some poles does not correspond exactly to the limit given by the curves with finite $\mathbf{q}$. However, the main difference between
the zero and finite $\mathbf{q}$ curves of $Y(\omega)$ is not in the position but rather in the value of the residues of the poles, which is reflected in the intensity of some of the peaks, as shown in Fig. 6.

To compare the computed results with experiments, we used electron energy-loss data extracted from a compilation of optical measurements found in Table I of the chapter Optical Constants of Metals of Ref. [24] (see, e.g., Fig. 8 of Ref. [31]), after interpolation with a multipole model. For this, we chose 18 points of the spectra, with a frequency distribution corresponding to the quadratic sampling of Eq. (11) and used them

\begin{figure}
\includegraphics[alt={},max width=\textwidth]{https://cdn.mathpix.com/cropped/3d8b6fda-3544-4cf1-8ed7-c015c8fb8cd8-11.jpg?height=1091&width=1621&top_left_y=1258&top_left_x=214}
\captionsetup{labelformat=empty}
\caption{FIG. 6. (left panels) Frequency dependence of the (a) real and (b) imaginary part of $Y_{\mathbf{G}=\mathbf{G}^{\prime}=0}$ for Cu computed within MPA for different $q$ values tending to zero ( $q_{n}=\frac{n}{16}$ in units of $2 \pi / a$, where $a$ is the lattice parameter of Cu ). For $q=0$ (orange curves) the intraband term is not included. (right panels) (c) Real and (d) imaginary parts of the four most relevant poles at low energies in the $Y$ curves for different $q$ values. The purple dashed lines lines correspond to the position of the poles extracted from optical measurements collected in Ref. [24], as explained in the main text. The blue dashed lines correspond to the values of the intraband frequency $\omega_{A}$ computed by means of Eq. (14) and reported in Ref. [31].}
\end{figure}
to interpolate a nine-pole model. We then analyzed the four poles with the highest residues in the frequency interval we are interested in. In the upper panel of Fig. 6 we show, as horizontal lines, the corresponding experimental energies of the poles. Interestingly, the experimental poles are very similar to the poles computed at the RPA level within MPA. This supports the interpretation (see, e.g., Ref. [97]) that the MPA poles of $Y$ are not a mere mathematical construct aimed at improving representability but indeed correspond to physical collective excitations, each of them describing the envelope of a set of single-particle transitions, with a finite imaginary part corresponding to the width of the excitation. We emphasize that the agreement with the experiment is achieved without resorting to any ad hoc parameters such as the damping in the case of the full frequency mesh on the real axis used in Ref. [31].

In simpler systems, the inclusion of the intraband limit, even with a simple Drude tail fitted from the experimental spectra, is expected to correct the residues and thus the intensity of the peaks at $\mathbf{q}=0$. However, in systems for which the intra- and interband contributions are superimposed in a more structured frequency dependence, the description of the experimental spectra with only the Drude term from Eq. (12) is not possible [59,81,98], and indeed models often resort to variable or multiple relaxation frequencies [59,84,98]. In fact, as shown in Fig. 6, the $\mathbf{q}$ dependence of $Y$ does not allow one to discriminate between peaks with an intra- or an interband character. To circumvent this difficulty, in Ref. [31] the intraband frequency is evaluated numerically as the limit of an intraband integral at the independent-particle level, while in Ref. [58] it is estimated within a noninteracting uniform-gas theory.

Here we use again the $f$-sum rule by integrating Eq. (13), but generalizing Eq. (14) to the case where, in contrast with Al and Na , more than one pole contributes to the intraband term (see Sec. I of the Supplemental Material [88]). The resulting intraband frequency, $\omega_{A}=0.36 \mathrm{Ha}(9.80 \mathrm{eV})$, compares well with the corresponding result of $0.34 \mathrm{Ha}(9.27 \mathrm{eV})$ from Ref. [31] and both values are very close in energy to the second pole shown in Fig. 6. We find that intraband contributions represent around the $25 \%$ of the corresponding $f$-sum rule of this pole ( $R \Omega$ product), being the largest ratio among all the poles. However, as can be appreciated in Fig. 6 from the change of intensity of the peaks, the interband contributions are dominant. In fact, the intraband contributions to the total $f$-sum rule (sum of all $R \Omega$ products) is rather small, less than $4 \%$.

Using the frequency determined in Ref. [31] ( 9.27 eV ) and the relaxation frequency fixed to 0.1 eV as the inputs to the Drude correction of Eq. (12), in our MPA calculations, we find that the Drude tail overlaps with the several interband peaks of $Y(\omega)$, without affecting the position of the poles while changing their residues (Sec. IV of the Supplemental Material [88]), similar to the effect of the CA correction, $Y(\mathbf{q}= 0) \sim Y\left(\mathbf{q}_{\text {min }}\right)$, as proposed in Sec. III B. In any case, CA is general and independent of the complexity of the frequency structure of the inverse dielectric function $Y$. It works well for Cu , as confirmed by the comparison with the experimental data and constitutes a very simple procedure. Despite these considerations and similarly to the case of Al , the intraband
correction has a small effect on the Cu QP energies, that present differences of the order of 5 meV when computed with and without CA in a $12 \times 12 \times 12 \mathbf{k}$ grid.

\section*{IV. SUMMARY AND CONCLUSIONS}

In this work we address the accuracy of the MPA scheme as applied to the full frequency $G W$ calculation of metals. This approach, previously validated for semiconductors [25], is now applied to metals using $\mathrm{Al}, \mathrm{Na}$, and Cu as prototype systems. Also in the case of metals, MPA is shown to deliver results with an accuracy similar to other FF methods at a much lower computational cost, comparable to other analytical continuation approaches [25,54].

After presenting the MPA theoretical framework, we have applied the approach to simple metals and discussed the role of inter- and intraband contributions to the dielectric functions of bulk Al and Na . To evaluate the response function and the $G W$ corrections in metals, we have proposed two simple methods to include the intraband terms in the inverse dielectric function in the $\mathbf{q} \rightarrow 0$ limit: (1) by extrapolating the position of the main pole in $Y_{00}(\mathbf{q}, \omega)$, from small $\mathbf{q}$ to $\mathbf{q}=0$, and computing the intraband pole through the $f$-sum rule of Eq. (14), which can then be used as an input value in a Drude model to correct $Y_{0}$. This approach is generalized for a multipole structure of $Y(\mathbf{q}, \omega)$ in the case of Cu . And (2) by approximating $Y(\mathbf{q}=0)$ by $Y\left(\mathbf{q}_{\text {min }}\right)$. The second method, here called CA, is simpler and spares the determination of the intraband frequency.

Both methods significantly accelerate the convergence of the QP energies with respect to the $\mathbf{k}$-point grid. In addition, CA simultaneously corrects all $Y$ matrix elements. CA works equally within PPA, MPA, and FF and can be used independently of the dimensionality of the system under study, even if the leading power of series expansion of the inverse dielectric function in the $\mathbf{q} \rightarrow 0$ limit depends on dimensionality. In fact, it can be thought of as the most trivial case of a polynomial interpolation (a constant) [99,100]. A similar approach can be applied in situations where the $\mathbf{q} \rightarrow 0$ limit of $Y$ (or other many-body operators, such as $W$ ) is difficult to evaluate. Even if the proposed methodologies were exemplified for three isotropic metals, the extension to nonisotropic systems is straightforward.

Eventually, $G W \mathrm{QP}$ corrections for $\mathrm{Na}, \mathrm{Al}$, and Cu were evaluated, showing an excellent agreement with existing theoretical literature and experimental data, further stressing the accuracy of the proposed approach. Notably, the case of Cu was discussed with particular detail, since PPA calculations present several drawbacks. In fact, for Cu , the PPA quasiparticle solutions deviate significantly from the FF results and completely fail to describe the frequency dependence of $\Sigma$ and the spectral function. In contrast, MPA reproduces very accurately the FF results, not only in the tail region that determines the quasiparticles corrections, but in the whole frequency range for both the self-energy and the spectral function. The frequency representation of the polarizability and the inverse dielectric function present strong oscillations within FF . In contrast, MPA results are much more stable, leading to a smooth frequency representation of $X$ and $Y$.

Importantly, the smoother structure of the MPA dielectric function does not necessarily result in a loss of accuracy in the subsequent calculation of the self-energy, the QP energies, and the spectral function. In fact, the frequency dependence of $Y$ given by MPA is meaningful and reproduces the main peaks of the experimental energy-loss spectra. This leads us to conclude that the MPA poles of $Y$ may be seen not only as a mathematical tool, but also as an efficient description of collective excitations, with each pole representing the envelope of a set of single-particle transitions.

In conclusion, MPA reproduces well the overall frequency dependence of the polarizability, the inverse dielectric function, the self-energy and the spectral function in metallic systems, and gives results for the quasiparticle energies similar to those obtained within FF methods. Moreover, the favorable computational performance allows for the use of more stringent convergence parameters such as denser $\mathbf{k}$ grids and larger number of bands and polarizability matrices. The
use of the proposed intraband corrections further accelerates the convergence with the $\mathbf{k}$ grid and the accuracy of the final results.

\section*{ACKNOWLEDGMENTS}

We acknowledge stimulating discussions with Massimo Rontani, Pino D'Amico, Alberto Guandalini and Giacomo Sesti. This work was partially supported by MaX - MAterials design at the eXascale-a European Centre of Excellence funded by the European Union's program HORIZON-EUROHPC-JU-2021-COE-01 (Grant No. 101093374), ICSC Centro Nazionale di Ricerca in High Performance Computing, Big Data and Quantum Computing, funded by European Union NextGenerationEU - PNRR, Missione 4 Componente 2 Investimento 1.4. Computational time on the Marconi100 machine at CINECA was provided by the Italian ISCRA program (HP10BAYEFL), and on Meluxina was provided by the EuroHPC Regular Access program (EHPC-REG-2021R0008).
[1] G. Onida, L. Reining, and A. Rubio, Rev. Mod. Phys. 74, 601 (2002).
[2] R. M. Martin, L. Reining, and D. M. Ceperley, Interacting Electrons (Cambridge University Press, Cambridge, 2016).
[3] N. Marzari, A. Ferretti, and C. Wolverton, Nat. Mater. 20, 736 (2021).
[4] L. Hedin, Phys. Rev. 139, A796 (1965).
[5] G. Strinati, H. J. Mattausch, and W. Hanke, Phys. Rev. B 25, 2867 (1982).
[6] F. Aryasetiawan and O. Gunnarsson, Rep. Prog. Phys. 61, 237 (1998).
[7] L. Reining, Wiley Interdiscip. Rev. Comput. Mol. Sci. 8, e1344 (2018).
[8] D. Golze, M. Dvorak, and P. Rinke, Front. Chem. (Lausanne, Switz.) 7, 377 (2019).
[9] M. S. Hybertsen and S. G. Louie, Phys. Rev. B 34, 5390 (1986).
[10] S. B. Zhang, D. Tománek, M. L. Cohen, S. G. Louie, and M. S. Hybertsen, Phys. Rev. B 40, 3162 (1989).
[11] R. W. Godby and R. J. Needs, Phys. Rev. Lett. 62, 1169 (1989).
[12] W. von der Linden and P. Horsch, Phys. Rev. B 37, 8351 (1988).
[13] G. E. Engel and B. Farid, Phys. Rev. B 47, 15931 (1993).
[14] P. Larson, M. Dvorak, and Z. Wu, Phys. Rev. B 88, 125205 (2013).
[15] L. Hedin, B. I. Lundqvist, and S. Lundqvist, Int. J. Quantum Chem. 1, 791 (1967).
[16] J. E. Northrup, M. S. Hybertsen, and S. G. Louie, Phys. Rev. Lett. 59, 819 (1987).
[17] M. P. Surh, J. E. Northrup, and S. G. Louie, Phys. Rev. B 38, 5976 (1988).
[18] J. E. Northrup, M. S. Hybertsen, and S. G. Louie, Phys. Rev. B 39, 8198 (1989).
[19] M. Cazzaniga, Phys. Rev. B 86, 035120 (2012).
[20] A. Marini, G. Onida, and R. Del Sole, Phys. Rev. Lett. 88, 016403 (2001).
[21] A. L. Fetter and J. D. Walecka, Quantum Theory of ManyParticle Systems (McGraw-Hill, New York, 1971).
[22] G. Giuliani and G. Vignale, Quantum Theory of the Electron Liquid (Cambridge University Press, 2005).
[23] P. O. Nilsson and C. G. Larsson, Phys. Rev. B 27, 6143 (1983).
[24] E. Palik, Handbook of Optical Constants of Solids, 1st ed. (Academic Press, College Park, 1985), p. 283.
[25] D. A. Leon, C. Cardoso, T. Chiarotti, D. Varsano, E. Molinari, and A. Ferretti, Phys. Rev. B 104, 115157 (2021).
[26] A. Marini, C. Hogan, M. Grüning, and D. Varsano, Comput. Phys. Commun. 180, 1392 (2009).
[27] D. Sangalli, A. Ferretti, H. Miranda, C. Attaccalite, I. Marri, E. Cannuccia, P. Melo, M. Marsili, F. Paleari, A. Marrazzo, G. Prandini, P. Bonfà, M. O. Atambo, F. Affinito, M. Palummo, A. Molina-Sánchez, C. Hogan, M. Grüning, D. Varsano, and A. Marini, J. Phys.: Condens. Matter 31, 325902 (2019).
[28] E. G. Maksimov, I. I. Mazin, S. N. Rashkeev, and Y. A. Uspenski, J. Phys. F: Met. Phys. 18, 833 (1988).
[29] M. Methfessel and A. T. Paxton, Phys. Rev. B 40, 3616 (1989).
[30] K.-H. Lee and K. J. Chang, Phys. Rev. B 49, 2362 (1994).
[31] A. Marini, G. Onida, and R. Del Sole, Phys. Rev. B 64, 195125 (2001).
[32] M. Cazzaniga, N. Manini, L. G. Molinari, and G. Onida, Phys. Rev. B 77, 035117 (2008).
[33] M. van Schilfgaarde, T. Kotani, and S. Faleev, Phys. Rev. Lett. 96, 226402 (2006).
[34] F. Bruneval, N. Vast, and L. Reining, Phys. Rev. B 74, 045102 (2006).
[35] T. Kotani, M. van Schilfgaarde, and S. V. Faleev, Phys. Rev. B 76, 165106 (2007).
[36] M. Shishkin, M. Marsman, and G. Kresse, Phys. Rev. Lett. 99, 246403 (2007).
[37] A. Kutepov, K. Haule, S. Y. Savrasov, and G. Kotliar, Phys. Rev. B 85, 155129 (2012).
[38] A. Kutepov, V. Oudovenko, and G. Kotliar, Comput. Phys. Commun. 219, 407 (2017).
[39] M. Grumet, P. Liu, M. Kaltak, J. c. v. Klimeš, and G. Kresse, Phys. Rev. B 98, 155143 (2018).
[40] C. Friedrich, S. Blügel, and D. Nabok, Nanomaterials 12, 3660 (2022).
[41] E. L. Shirley, Phys. Rev. B 54, 7758 (1996).
[42] W. Chen and A. Pasquarello, Phys. Rev. B 92, 041115(R) (2015).
[43] X. Ren, N. Marom, F. Caruso, M. Scheffler, and P. Rinke, Phys. Rev. B 92, 081104(R) (2015).
[44] E. Maggio and G. Kresse, J. Chem. Theory Comput. 13, 4765 (2017).
[45] M. Guzzo, G. Lani, F. Sottile, P. Romaniello, M. Gatti, J. J. Kas, J. J. Rehr, M. G. Silly, F. Sirotti, and L. Reining, Phys. Rev. Lett. 107, 166401 (2011).
[46] J. S. Zhou, L. Reining, A. Nicolaou, A. Bendounan, K. Ruotsalainen, M. Vanzini, J. J. Kas, J. J. Rehr, M. Muntwiler, V. N. Strocov, F. Sirotti, and M. Gatti, Proc. Natl. Acad. Sci. U. S. A. 117, 28596 (2020).
[47] F. Hüser, T. Olsen, and K. S. Thygesen, Phys. Rev. B 87, 235132 (2013).
[48] R. W. Godby, M. Schlüter, and L. J. Sham, Phys. Rev. B 37, 10159 (1988).
[49] F. Aryasetiawan, in Strong Coulomb Correlations in Electronic Structure Calculations, 1st ed. (CRC Press, London, 2000), p. 96.
[50] R. Daling, W. van Haeringen, and B. Farid, Phys. Rev. B 44, 2952 (1991).
[51] G. E. Engel, B. Farid, C. M. M. Nex, and N. H. March, Phys. Rev. B 44, 13356 (1991).
[52] K.-H. Lee and K. J. Chang, Phys. Rev. B 54, R8285 (1996).
[53] J. A. Soininen, J. J. Rehr, and E. L. Shirley, Phys. Scr. 2005, 243 (2005).
[54] I. Duchemin and X. Blase, J. Chem. Theory Comput. 16, 1742 (2020).
[55] F. Wooten, Optical Properties of Solids (Academic Press, 1972).
[56] K. Kolwas and A. Derkachova, Nanomaterials 10, 1411 (2020).
[57] M. Cazzaniga, L. Caramella, N. Manini, and G. Onida, Phys. Rev. B 82, 035104 (2010).
[58] O. K. Orhan and D. D. O'Regan, J. Phys.: Condens. Matter 31, 315901 (2019).
[59] J. W. Allen and J. C. Mikkelsen, Phys. Rev. B 15, 2952 (1977).
[60] D. Y. Smith and B. Segall, Phys. Rev. B 34, 5191 (1986).
[61] W. Kohn, Phys. Rev. B 10, 382 (1974).
[62] B. Sporkmann and H. Bross, Phys. Rev. B 49, 10869 (1994).
[63] G. Prandini, M. Galante, N. Marzari, and P. Umari, Comput. Phys. Commun. 240, 106 (2019).
[64] G. Prandini, G.-M. Rignanese, and N. Marzari, npj Comput. Mater. 5, 129 (2019).
[65] P. E. Blöchl, O. Jepsen, and O. K. Andersen, Phys. Rev. B 49, 16223 (1994).
[66] H. J. Levinson, F. Greuter, and E. W. Plummer, Phys. Rev. B 27, 727 (1983).
[67] E. Jensen and E. W. Plummer, Phys. Rev. Lett. 55, 1912 (1985).
[68] I.-W. Lyo and E. W. Plummer, Phys. Rev. Lett. 60, 1558 (1988).
[69] R. Courths and S. Hüfner, Phys. Rep. 112, 53 (1984).
[70] M. Vos, A. S. Kheifets, C. Bowles, C. Chen, E. Weigold, and F. Aryasetiawan, Phys. Rev. B 70, 205111 (2004).
[71] V. P. Zhukov, E. V. Chulkov, and P. M. Echenique, Phys. Rev. B 68, 045102 (2003).
[72] P. Liu, M. Kaltak, J. Klineš, and G. Kresse, Phys. Rev. B 94, 165109 (2016).
[73] M. Del Ben, F. H. da Jornada, G. Antonius, T. Rangel, S. G. Louie, J. Deslippe, and A. Canning, Phys. Rev. B 99, 125128 (2019).
[74] J. P. Perdew, K. Burke, and M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996).
[75] D. R. Hamann, Phys. Rev. B 88, 085117 (2013).
[76] P. Giannozzi, S. Baroni, N. Bonini, M. Calandra, R. Car, C. Cavazzoni, D. Ceresoli, G. L. Chiarotti, M. Cococcioni, I. Dabo, A. D. Corso, S. de Gironcoli, S. Fabris, G. Fratesi, R. Gebauer, U. Gerstmann, C. Gougoussis, A. Kokalj, M. Lazzeri, L. Martin-Samos et al., J. Phys.: Condens. Matter 21, 395502 (2009).
[77] P. Giannozzi, O. Andreussi, T. Brumme, O. Bunau, M. B. Nardelli, M. Calandra, R. Car, C. Cavazzoni, D. Ceresoli, M. Cococcioni, N. Colonna, I. Carnimeo, A. D. Corso, S. de Gironcoli, P. Delugas, R. A. DiStasio, Jr., A. Ferretti, A. Floris, G. Fratesi, G. Fugallo et al., J. Phys.: Condens. Matter 29, 465901 (2017).
[78] F. Caruso and F. Giustino, Eur. Phys. J. B 89, 238 (2016).
[79] A. vom Felde, J. Sprösser-Prou, and J. Fink, Phys. Rev. B 40, 10181 (1989).
[80] S. Huotari, M. Cazzaniga, H.-C. Weissker, T. Pylkkänen, H. Müller, L. Reining, G. Onida, and G. Monaco, Phys. Rev. B 84, 075108 (2011).
[81] P. B. Johnson and R. W. Christy, Phys. Rev. B 6, 4370 (1972).
[82] P. D'Amico, M. Gibertini, D. Prezzi, D. Varsano, A. Ferretti, N. Marzari, and E. Molinari, Phys. Rev. B 101, 161410(R) (2020).
[83] A. G. Mathewson and H. P. Myers, J. Phys. F: Met. Phys. 2, 403 (1972).
[84] R. L. Benbow and D. W. Lynch, Phys. Rev. B 12, 5615 (1975).
[85] K. J. Krane, J. Phys. F: Met. Phys. 8, 2133 (1978).
[86] H. Möller and A. Otto, Phys. Rev. Lett. 45, 2140 (1980).
[87] G. Stefanucci and R. van Leeuwen, Nonequilibrium ManyBody Theory of Quantum Systems: A Modern Introduction (Cambridge University Press, 2013).
[88] See Supplemental Material at http://link.aps.org/supplemental/ 10.1103/PhysRevB.107.155130 for a detailed description.
[89] T. Rangel, M. Del Ben, D. Varsano, G. Antonius, F. Bruneval, F. H. da Jornada, M. J. van Setten, O. K. Orhan, D. D. O'Regan, A. Canning, A. Ferretti, A. Marini, G.-M. Rignanese, J. Deslippe, S. G. Louie, and J. B. Neaton, Comput. Phys. Commun. 255, 107242 (2020).
[90] A. Oschlies, R. W. Godby, and R. J. Needs, Phys. Rev. B 51, 1527 (1995).
[91] M. Springer and F. Aryasetiawan, Phys. Rev. B 57, 4364 (1998).
[92] B.-C. Shih, Y. Xue, P. Zhang, M. L. Cohen, and S. G. Louie, Phys. Rev. Lett. 105, 146401 (2010).
[93] C. Friedrich, M. C. Müller, and S. Blügel, Phys. Rev. B 83, 081101(R) (2011).
[94] J. A. Berger, L. Reining, and F. Sottile, Phys. Rev. B 85, 085126 (2012).
[95] A. Marini, R. Del Sole, A. Rubio, and G. Onida, Phys. Rev. B 66, 161104(R) (2002).
[96] T. Rangel, D. Kecik, P. E. Trevisanutto, G.-M. Rignanese, H. Van Swygenhoven, and V. Olevano, Phys. Rev. B 86, 125125 (2012).
[97] B. Farid, G. E. Engel, R. Daling, and W. van Haeringen, Phys. Rev. B 44, 13349 (1991).
[98] M. Suffczynski, Phys. Rev. 117, 663 (1960).
[99] J. Deslippe, G. Samsonidze, D. A. Strubbe, M. Jain, M. L. Cohen, and S. G. Louie, Comput. Phys. Commun. 183, 1269 (2012).
[100] A. Guandalini, P. D'Amico, A. Ferretti, and D. Varsano, npj Comput. Mater. 9, 44 (2023).