\title{
Spectral properties from an efficient analytical representation of the $G W$ self-energy within a multipole approximation
}

\author{
Dario A. Leon, ${ }^{1, *}$ Kristian Berland, ${ }^{1}$ and Claudia Cardoso ${ }^{2}$ \\ ${ }^{1}$ Department of Mechanical Engineering and Technology Management, Norwegian University of Life Sciences, NO-1432 Ås, Norway \\ ${ }^{2}$ S3 Centre, Istituto Nanoscienze, CNR, 41125 Modena, Italy
}
(Dated: May 16, 2025)

\begin{abstract}
We propose an efficient analytical representation of the frequency-dependent $G W$ self-energy $\Sigma$ via a multipole approximation (MPA- $\Sigma$ ). The MPA self-energy model is interpolated from a small set of numerical evaluations of $\Sigma$ in the complex frequency plane, similar to the MPA interpolation developed for the screened Coulomb interaction (MPA-W) [D. A. Leon et al., Phys. Rev. B 104, 115157 (2021)]. Crucially, MPA- $\Sigma$ enables a multipole representation for the interacting Green's function $G$ (MPA- $G$ ), and in turn, access to all the spectral properties, including quasiparticle energies ( QP ) and renormalization factors beyond the linearized QP equation. We validate the MPA- $\Sigma$ and MPA- $G$ approaches for a diverse set of systems: bulk $\mathrm{Si}, \mathrm{Na}$ and Cu , monolayer $\mathrm{MoS}_{2}$, the NaCl ion-pair, and the $\mathrm{F}_{2}$ molecule. We show that, just as for MPA- $W$, an appropriate choice of frequency sampling in MPA- $\Sigma$ is critical to guarantee computational efficiency and high accuracy. Moreover, the combined MPA- $W$ and MPA- $\Sigma$ scheme considerably reduces the cost of full-frequency self-energy calculations, especially for spectral band structures over a wide energy range.
\end{abstract}

\section*{I. INTRODUCTION}

In condensed matter physics, first principle methods such as density functional theory (DFT) in the KohnSham (KS) approximation provide accurate ground-state properties and have been immensely useful for understanding the electronic structure of materials. However, they fail to reliably provide accurate band structures, which requires including many-body effects beyond the mean field DFT level. The description of electron addition or removal energies and related excited-state properties is usually treated with methods such as the $G W$ approximation, based on the Green's function formalism [1$6]$.

In common $G W$ implementations, the Green's function $G$ and the screened Coulomb potential $W$ are constructed perturbatively. Starting from DFT, the KS quasiparticle (QP) energies are corrected by an exchange-correlation self-energy $\Sigma$. This correction can be obtained iteratively within a self-consistent $G W$ approach, or done in a computationally cheaper one-shot $G_{0} W_{0}$ procedure. Since the imaginary part of $G$ is closely related to the spectral function obtained from photoemission experiments $[7,8]$, a dynamical self-energy $\Sigma(\omega)$ can account for many-body features, such as finite QP lifetimes and satellite structures $[2,5,9-16]$.
$G_{0} W_{0}$ is the state-of-the-art $a b$ initio method for the description of angle-resolved photoemission and inverse photoemission spectroscopy measurements, giving generally a very accurate agreement with experiment (see, e.g., Refs. [17-20]). More accurate spectral functions can be obtained with self-consistent approaches, including cumulant expansions of $\Sigma$ and vertex corrections (see, e.g.,

\footnotetext{
* dario.alejandro.leon.valido@nmbu.no
}

Refs. [4, 5, 13, 16, 21-23]).
The $G W$ self-energy $\Sigma(\omega)$ is given by a frequency convolution of $G(\omega)$ and $W(\omega)$. This convolution can be evaluated with different full-frequency (FF) methods, based on numerical integrations along the frequency real axis [19,24-26], or through an integration in the complex frequency plane using contour deformation and analytic continuation techniques $[27-32]$. Such numerical FF evaluations tend to be computationally expensive. A less costly alternative is to represent $W$ (or the dielectric function) with a simple model, such as in the plasmon pole approximation (PPA), that allows for an analytical integration of the frequency convolution in $\Sigma[33-37]$, but in many cases this has limited accuracy. Higher accuracy can be obtained with multipole models and Padé approximants [31, 38-40], including the recently developed MPA- $W$ method [41-43].

Analytical models of the dielectric response are also widely used in the study of optical and electronic properties of materials [44-46]. They have been used in the study of, e.g., optical excitations [47, 48], electron energy loss [49, 50], and x-ray absorption spectra [51, 52]. Simple models have also been used to account for dynamical effects arising from electron-hole interactions in doped systems [53, 54], or in the ab initio description of plasmonphonon hybridization in doped semiconductors [55]. Less common is the use of models for $\Sigma[56-59]$, the interacting $G$ [59-62], or for total energy calculations [59-64], and usually these approaches mainly aim at improving the computational efficiency of such calculations.

In this work, we present an efficient multipole approximation for the self-energy (MPA- $\Sigma$ ). This method yields simple analytical representations of all $G W$ operators, including a multipole-Padé representation of the Green's function (MPA-G). Moreover, the combination of MPA$\Sigma$ with the previous MPA- $W$ method considerably reduces the cost of evaluating $\Sigma$ and $G$ in its full-frequency
domain. Although this study is limited to the $G_{0} W_{0}$ approximation, extending it to higher levels of theory, e.g., self-consistent $G W$, and the inclusion of vertex corrections or cumulant expansions, is straightforward.

The paper is organized as follows: In the methods section (Sec. II) we summarize the general $G W$ equations (Sec. II A), and the previously introduced MPA- $W$ method (Sec. II B). Thereafter, we present the new MPA$\Sigma$ (Sec. II C) and MPA-G (Sec. II D) approaches, and provide computational details of the $G W$ calculations (Sec. IIE). In the results section (Sec. III) we benchmark MPA- $\Sigma$ and MPA- $G$ on different prototypical materials (Sec. III A), and build spectral band structures (Sec. III B). Last Sec. IV holds our conclusions. In addition, Appendix A provides an analysis of the QP particle and the renormalization factor in terms of two MPA toy models, while Appendix B presents numerical details for interpolating spectral functions in momentum space.

\section*{II. METHODS}

\section*{A. Quasiparticle $G W$ equations}

In terms of KS states, the non-interacting time-ordered Green's function can be written in the Lehmann representation [4, 65], analytically continued to the complex frequency plane, as
$$
\begin{equation*}
G_{0}(z)=\sum_{m} \rho_{m}^{\mathrm{KS}}\left[\frac{f_{m}^{\mathrm{KS}}}{z-\varepsilon_{m}^{\mathrm{KS}}-i 0^{+}}+\frac{1-f_{m}^{\mathrm{KS}}}{z-\varepsilon_{m}^{\mathrm{KS}}+i 0^{+}}\right] \tag{1}
\end{equation*}
$$
where the sum runs over the KS states, $m$, with the projectors $\rho_{m}^{\mathrm{KS}}=\left|\psi_{m}^{\mathrm{KS}}\right\rangle\left\langle\psi_{m}^{\mathrm{KS}}\right|$, KS energies $\varepsilon_{m}^{\mathrm{KS}}$, and occupation numbers $f_{m}^{\mathrm{KS}} \in[0,1]$. The complex frequency is given by $z \equiv \omega+i \varpi$, which is evaluated in the first and third quadrants ( $\omega \varpi>0$ ), opposite to the pole position according to the time ordering (see notation in Table I).

The projection of $G_{0}$ onto the KS states ( $n \mathbf{k}$ ) is given by
$$
\begin{align*}
G_{0 n \mathbf{k}}(z) & \equiv\left\langle\psi_{n \mathbf{k}}^{\mathrm{KS}}\right| G_{0}(z)\left|\psi_{n \mathbf{k}}^{\mathrm{KS}}\right\rangle \\
& =\frac{f_{n \mathbf{k}}^{\mathrm{KS}}}{z-\varepsilon_{n \mathbf{k}}^{\mathrm{KS}}-i 0^{+}}+\frac{1-f_{n \mathbf{k}}^{\mathrm{KS}}}{z-\varepsilon_{n \mathbf{k}}^{\mathrm{KS}}+i 0^{+}} \tag{2}
\end{align*}
$$
where the spectral function $\operatorname{Im}\left[G_{0 n \mathbf{k}}\right]$ is a Dirac delta function centered on $\epsilon_{n \mathbf{k}}^{\mathrm{KS}}$. The interacting Green's function is given by the Dyson equation for this operator, in which the DFT exchange and correlation potential is subtracted from the self-energy:
$$
\begin{equation*}
G_{n \mathbf{k}}^{-1}(z)=G_{0 n \mathbf{k}}^{-1}(z)-\Sigma_{n \mathbf{k}}(z)+v_{x c}^{\mathrm{KS}} \tag{3}
\end{equation*}
$$
where, as commonly done, the off-diagonal elements ( $n \mathbf{k} \neq n^{\prime} \mathbf{k}^{\prime}$ ) have been neglected. At the $G_{0} W_{0}$ level, $\Sigma$ is given by the convolution of $G_{0}$ and $W_{0}$ :
$$
\begin{equation*}
\Sigma(z)=\frac{i}{2 \pi} \int_{-\infty}^{+\infty} d \omega^{\prime} e^{-i \omega^{\prime} 0^{+}} G_{0}\left(z-\omega^{\prime}\right) W_{0}\left(\omega^{\prime}\right) \tag{4}
\end{equation*}
$$

\begin{table}
\begin{tabular}{lcc}
\hline \hline Complex quantity & Energy/Poles & Residues \\
\hline Energy/frequency & $z=\omega+i \varpi$ & - \\
$G_{0}(z)$ & $\epsilon^{\mathrm{KS}}=\varepsilon^{\mathrm{KS}} \pm i 0^{+}$ & 1 \\
$G(z)$ & $\epsilon_{p}=\varepsilon_{p}+i \eta_{p}$ & $Z_{p}$ \\
$W(z)$ & $\Omega_{p}=\omega_{p}+i \varpi_{p}$ & $R_{p}$ \\
$\Sigma(z)$ & $\xi_{p}=\zeta_{p}+i \varsigma_{p}$ & $S_{p}$ \\
\hline \hline
\end{tabular}
\captionsetup{labelformat=empty}
\caption{TABLE I. List of complex quantities relevant for this work, and definition of the used notation. In the case of $G_{0}$, each state is represented by a single pole with a vanishing imaginary part whose sign follows the time ordering. The residue of such pole carries all the spectral weight.}
\end{table}

The QP energies correspond to the poles of $G$, which are determined by solving the QP equation:
$$
\begin{equation*}
\epsilon_{n \mathbf{k}}=\epsilon_{n \mathbf{k}}^{\mathrm{KS}}+\left\langle\psi_{n \mathbf{k}}^{\mathrm{KS}}\right| \Sigma\left(\epsilon_{n \mathbf{k}}\right)-v_{x c}^{\mathrm{KS}}\left|\psi_{n \mathbf{k}}^{\mathrm{KS}}\right\rangle . \tag{5}
\end{equation*}
$$

The frequency dependence of $G$ has a structure typically dominated by a well-defined main peak, the QP pole [8], and satellite structures at larger energies [4, 5, 66]. Like the QP pole, the satellites are also formal solutions of Eq. (5). They arise from many-body excitations accounted for in $\Sigma$, such as the plasmonic structures in $G W$, and give rise to replicas of the QP band structure [4, 5]. In the so-called QP picture, satellites are disregarded and only energies around the QP pole are considered. As such, the QP picture resembles the independent particle picture, but with the KS energies corrected by the real part of $\Sigma$, while the finite imaginary part accounts for the broadening of the QP pole, according to its lifetime.

Due to the non-linearity of Eq. (5), its numerical evaluation requires a recursive procedure, such as the secant method. Alternatively, it can be approximated by a linearized equation:
$$
\begin{equation*}
\epsilon_{n \mathbf{k}} \approx \epsilon_{n \mathbf{k}}^{\operatorname{lin}} \equiv \epsilon_{n \mathbf{k}}^{\mathrm{KS}}+Z_{n \mathbf{k}}^{\operatorname{lin}}\left\langle\psi_{n \mathbf{k}}^{\mathrm{KS}}\right| \Sigma\left(\epsilon_{n \mathbf{k}}^{\mathrm{KS}}\right)-v_{x c}^{\mathrm{KS}}\left|\psi_{n \mathbf{k}}^{\mathrm{KS}}\right\rangle, \tag{6}
\end{equation*}
$$
with the corresponding linearized renormalization factor, $Z_{n \mathbf{k}}^{\operatorname{lin}}$, given by
$$
\begin{equation*}
Z_{n \mathbf{k}}^{\operatorname{lin}}=\left[1-\left.\left\langle\psi_{n \mathbf{k}}^{\mathrm{KS}}\right| \frac{\partial \Sigma(z)}{\partial z}\right|_{z=\epsilon_{n \mathbf{k}}^{\mathrm{KS}}}\left|\psi_{n \mathbf{k}}^{\mathrm{KS}}\right\rangle\right]^{-1} \tag{7}
\end{equation*}
$$
which approximates the spectral weight of the QP pole, $Z_{n \mathbf{k}}$, based on the assumption that the QP correction $\left(\epsilon_{n \mathbf{k}}-\epsilon_{n \mathbf{k}}^{\mathrm{KS}}\right)$ is small. Since the satellite structures have a non-vanishing weight, $Z_{n \mathbf{k}}$ usually has a very small imaginary part and a real part ranging from 0.5 to 1 . This interval is taken as a typical validity range of the QP picture [5, 66, 67], while strong correlation effects can lead to situations where the spectral weight is concentrated in the satellites, like in Mott insulators (see, e.g., Refs. [4, 68]). When computed in a consistent way, the spectral weights of the QP pole and the satellites sum exactly to one, since they comply with the sum rule for
the number of particles and holes [69]:
$$
\begin{equation*}
\frac{1}{\pi} \int_{-\infty}^{\infty} \operatorname{Im} G_{n \mathbf{k}}(\omega) d \omega=1 \tag{8}
\end{equation*}
$$

\section*{B. MPA for the screening interaction}

The screened Coulomb potential can be separated in a static bare Coulomb and a correlation term: $W(\omega)= v+W_{\mathrm{c}}(\omega)$. As detailed in Ref. [41], the frequency dependence of each matrix element $W_{\mathbf{q G G}}^{\mathrm{c}}$ can be described by a multipole model with a small number of complex poles for each transferred momentum, $\mathbf{q}$, and reciprocal lattice vectors, $\mathbf{G G}^{\prime} . W$ is then given by:
$$
\begin{align*}
W_{\mathbf{G G}^{\prime}}^{\mathrm{MPA}}(\mathbf{q}, z)= & v_{\mathbf{G G}^{\prime}}(\mathbf{q})+\sum_{p=1}^{n_{W}} R_{p \mathbf{q} \mathbf{G}^{\prime}} \\
& \times\left[\frac{1}{z-\Omega_{p \mathbf{q} \mathbf{G}^{\prime}}}-\frac{1}{z+\Omega_{p \mathbf{q} \mathbf{G}^{\prime}}}\right] \tag{9}
\end{align*}
$$
where $\Omega_{p \mathbf{q} \mathbf{G}^{\prime}}$ are the MPA poles and $R_{p \mathbf{q} \mathbf{G}^{\prime}}$ their residues, and $n_{W}$, the number of poles. The time ordering of $W$ implies that $\operatorname{Re}\left[\Omega_{p \mathbf{q} \mathbf{G}^{\prime}}\right] \times \operatorname{Im}\left[\Omega_{p \mathbf{q} \mathbf{G}^{\prime}}\right]<0$. Such poles represent effective plasmon-like quasiparticles emerging from a large set of single-particle transitions from valence to conduction states $[38,42]$.

For each $W_{\mathbf{q G G}}^{\mathbf{c}}$ matrix element, all poles and residues are obtained through a non-linear interpolation of values numerically evaluated in a conveniently selected set of complex frequencies $\left\{z_{i}, i=1, \ldots, 2 n_{W}\right\}$. We use a frequency sampling along two lines parallel to the real axis (double-parallel sampling), typically along $\operatorname{Im} z=0.1$ and $\operatorname{Im} z=1 \mathrm{Ha}$, respectively, with Re $z_{i}$ distributed inhomogeneously. The double-parallel sampling, in particular the line of points with the largest imaginary part, reduces the noise resulting from the coarse Brillouin zone sampling of $W$. The inhomogeneous sampling distributions along the real axis are denser closer to the origin, following Eq. (10) of Ref. [42], which limits the number of poles needed [41, 42]. We use two types of distributions, a linear and a quadratic semi-homogeneous partition, depending on the given system. The frequency range is also specific for each system, since it must encompass the main structures of $W$. The classical plasmon energy, or the maximum single-particle transition from the valence to the conduction bands can be used as a reference energy scale in setting the sampling. More practical details, including a measure of the representability error, can be found in Refs. [41, 42].

With such an MPA representation, the frequency integral in the $G_{0} W_{0}$ self-energy of Eq. (4) can be solved
analytically. In a plane-wave basis set, it results in
$$
\begin{align*}
& \Sigma_{n \mathbf{k}}^{\mathrm{MPA}-W}(z)=\Sigma_{n \mathbf{k}}^{\mathrm{x}}+\sum_{m} \sum_{\mathbf{G G}^{\prime}} \sum_{p=1}^{n_{W}} \int \frac{d \mathbf{q}}{(2 \pi)^{3}} S_{p \mathbf{G G}^{\prime}}^{n m}(\mathbf{k}, \mathbf{q}) \\
& \times\left[\frac{f_{m \mathbf{k}-\mathbf{q}}^{\mathrm{KS}}}{z-\epsilon_{m \mathbf{k}-\mathbf{q}}^{\mathrm{KS}}+\Omega_{p \mathbf{q} \mathbf{G G}^{\prime}}}+\frac{1-f_{m \mathbf{k}-\mathbf{q}}^{\mathrm{KS}}}{z-\epsilon_{m \mathbf{k}-\mathbf{q}}^{\mathrm{KS}}-\Omega_{p \mathbf{q} \mathbf{G G}^{\prime}}}\right] \tag{10}
\end{align*}
$$
where
$$
\begin{align*}
S_{p \mathbf{G} \mathbf{G}^{\prime}}^{n m}(\mathbf{k}, \mathbf{q}) & \equiv-2 \rho_{n m}^{\mathrm{KS}}(\mathbf{k}, \mathbf{q}, \mathbf{G}) R_{p \mathbf{q G}^{\prime}} \rho_{n m}^{\mathrm{KS}^{*}}\left(\mathbf{k}, \mathbf{q}, \mathbf{G}^{\prime}\right) \\
\rho_{n m}^{\mathrm{KS}}(\mathbf{k}, \mathbf{q}, \mathbf{G}) & \equiv\langle n \mathbf{k}| e^{i(\mathbf{q}+\mathbf{G}) \cdot \mathbf{r}}|m \mathbf{k}-\mathbf{q}\rangle \tag{11}
\end{align*}
$$

Note that the time ordering of $W$ carries over to the time ordering of $\Sigma$, and since $\operatorname{Im}\left[\Omega_{p \mathbf{q} \mathbf{G}^{\prime}}\right]$ is finite the vanishing imaginary part of the $\epsilon_{m \mathbf{k}-\mathbf{q}}^{\mathrm{KS}}$ poles of $G_{0}$ can be disregarded. The derivative of $\Sigma^{\mathrm{MPA}-W}(z)$ and therefore the linearized renormalization factor [see Eq. (7)] can also be computed analytically as
$$
\begin{gather*}
\frac{\partial \Sigma_{n \mathbf{k}}^{\mathrm{MPA}-W}(z)}{\partial z}=-\sum_{m} \sum_{\mathbf{G G}^{\prime}} \sum_{p=1}^{n_{W}} \int \frac{d \mathbf{q}}{(2 \pi)^{3}} S_{p \mathbf{G} \mathbf{G}^{\prime}}^{n m}(\mathbf{k}, \mathbf{q}) \\
\times\left[\frac{f_{m \mathbf{k}-\mathbf{q}}^{\mathrm{KS}}}{\left(z-\epsilon_{m \mathbf{k}-\mathbf{q}}^{\mathrm{KS}}+\Omega_{p \mathbf{q} \mathbf{G}^{\prime}}\right)^{2}}+\frac{1-f_{m \mathbf{k}-\mathbf{q}}^{\mathrm{KS}}}{\left(z-\epsilon_{m \mathbf{k}-\mathbf{q}}^{\mathrm{KS}}-\Omega_{p \mathbf{q} \mathbf{G}^{\prime}}\right)^{2}}\right] . \tag{12}
\end{gather*}
$$

The case $n_{W}=1$ is analogous to the PPA approach, which only uses one or two frequency evaluations of $W$. On the other hand, FF approaches on the real-axis can require as much as 1000 frequency points. By increasing $n_{W}$, MPA- $W$ typically provides an accuracy similar to FF with about 10 poles, interpolated from 20 frequency points. Thus, MPA- $W$ can be viewed as an effective FF approach requiring around 50 times fewer $W$ evaluations, with the corresponding savings in memory allocation [41]. The MPA- $W$ method is currently implemented in YAMBO [70, 71] and GPAW [72].

\section*{C. MPA for the self-energy}

The MPA- $W$ representation of Eq. (10) shows that $\Sigma_{\mathrm{c}}$, the correlation part of $\Sigma$, can be written as a sum of poles. However, the evaluation of Eq. (10) for each frequency point still requires a large number of matrix multiplications due to the dependence of $\rho_{n m}^{\mathrm{KS}}, \Omega_{p}$ and $R_{p}$ on the $\mathbf{q} \mathbf{G G}^{\prime}$ indices. To solve the linearized QP equation in Eq. (6), $\Sigma$ only needs to be evaluated at two frequencies, or one if Eq. (12) is used to compute the renormalization factor. However, to obtain spectral properties beyond the QP pole and the renormalization factor, a wide frequency range is needed.

To avoid the direct evaluation of the self-energy projection for each KS state on a dense frequency grid, $\Sigma_{n \mathbf{k}}$ can be modeled as a simple multipole-Padé approximant

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_17_755328709aa12b82ff93g-04.jpg?height=475&width=828&top_left_y=200&top_left_x=205}
\captionsetup{labelformat=empty}
\caption{FIG. 1. Example of an asymmetric MPA- $\Sigma$ sampling in the complex frequency plane with two branches close and far from the real axis, with imaginary part $\varpi= \pm 1 \mathrm{eV}$ (orange circles) and $\varpi= \pm 20 \mathrm{eV}$ (blue squares), each having six points in the positive side and eight in the negative one distributed according to the linear semi-homogeneous partition of Refs. [41, 42].}
\end{figure}
with a small number of $n_{\Sigma}$ poles that do not depend explicitly on $\mathbf{q G G}$, but are consistent with Eq. (10):
$$
\begin{equation*}
\Sigma_{n \mathbf{k}}^{\mathrm{MPA}-\Sigma}(z)=\Sigma_{n \mathbf{k}}^{\mathrm{x}}+\sum_{p=1}^{n_{\Sigma}} \frac{S_{n \mathbf{k} p}}{z-\xi_{n \mathbf{k} p}} \tag{13}
\end{equation*}
$$

The corresponding derivative is given by:
$$
\begin{equation*}
\frac{\partial \Sigma_{n \mathbf{k}}^{\mathrm{MPA}-\Sigma}(z)}{\partial z}=-\sum_{p=1}^{n_{\Sigma}} \frac{S_{n \mathbf{k} p}}{\left(z-\xi_{n \mathbf{k} p}\right)^{2}} \tag{14}
\end{equation*}
$$

Therefore, analogously to MPA- $W, \Sigma$ can be explicitly computed for a small number of frequency points used to interpolate the $\Sigma^{\mathrm{MPA}-\Sigma}$ model. For each $n \mathbf{k}$ (omitted for simplicity), the poles, $\xi_{p}$, and residues, $S_{p}$, are obtained by solving the following system of $2 n_{\Sigma}$ equations and variables:
$$
\begin{equation*}
\Sigma_{\mathrm{c}}^{\mathrm{MPA}-\Sigma}\left(z_{i}\right) \equiv \sum_{p}^{n_{\Sigma}} \frac{S_{p}}{z_{i}-\xi_{p}}=\Sigma_{c}\left(z_{i}\right), i=1, \ldots, 2 n_{\Sigma} \tag{15}
\end{equation*}
$$

The solution of Eq. (15) is obtained with a procedure similar to the one used in MPA- $W$, in which the correlation part of the MPA- $\Sigma$ model is rewritten in its Padé form, i.e., as a fraction of two polynomials:
$$
\begin{equation*}
\sum_{p}^{n_{\Sigma}} \frac{S_{p}}{z-\xi_{p}}=\frac{A_{n_{\Sigma}-1}(z)}{B_{n_{\Sigma}}(z)} \tag{16}
\end{equation*}
$$

The coefficients of the polynomial $B_{n_{\Sigma}}(z)$ can be evaluated from the numerical reference data, $\left\{z_{i}, \Sigma_{c}\left(z_{i}\right)\right\}$, using one of the two methods developed in Ref. [41], based on linear algebra and Thiele's Padé interpolation (see details in Sec. I of Ref. [73]). Moreover, its factorization can be performed using the companion matrix method [41]
and is given by
$$
\begin{equation*}
B_{n_{\Sigma}}(z)=\prod_{p}^{n_{\Sigma}}\left(z-\xi_{p}\right) \tag{17}
\end{equation*}
$$

In both methods the sampling points are divided into two sets, used to separate the problem of finding the poles, $\xi_{p}$, from the much simpler problem of finding the residues, $S_{p}$, once the poles are known. Such separation is computationally advantageous since the nonlinear problem of $2 n_{\Sigma}$ variables in Eq. (15), is reduced to two problems of size $n_{\Sigma}$, one nonlinear for the poles and the other linear for the residues. Moreover, by first obtaining the poles, it is then possible to apply physical constraints. We impose the time ordering to the complex poles $\xi_{p}=\zeta_{p}+i \varsigma_{p}$, and that they lay in the vicinity of the real frequency axis, as done for MPA- $W$ [41], which results in $\zeta_{p} / \varsigma_{p}<-1$. The residues $S_{p}$ can then be found by solving a simple linear least-squares problem (see details in Sec. I of Ref. [73]).

As for MPA- $W$, an adequate frequency sampling of $\Sigma$ in the complex plane is essential to obtain an effective MPA- $\Sigma$ representation. We adopted the same type of inhomogeneous samplings parallel to the real axis used for MPA- $W$. Unlike $W$ [Eq. (9)], $\Sigma$ is not symmetric in $\omega$ and therefore consists of single poles rather than pairs at $z= \pm \Omega_{p}$. For this reason, $\Sigma$ requires sampling along both the positive and negative axes, with a denser sampling in the region with the maximum variability. This corresponds to negative frequencies for the valence states, and positive for the conduction. A illustrated in Fig. 1, the sampling is chosen so that it complies with time ordering, having a small positive (negative) imaginary part for energies larger (smaller) than the KS energies, typically of $\varpi= \pm 0.1 \mathrm{eV}$. The parallel sampling is done along the orange line, while the double parallel would use both the orange and the blue points. Since $\Sigma$ has a smoother structure than $W$, it is sufficient to sample it along a single line parallel to the real frequency axis.

The pole structure of $\Sigma$ is expected to resemble that of $W$, therefore the same sampling distribution can be used for $\Sigma$, as long as it is replicated on the negative side of the imaginary axis, as illustrated in the example of Fig. 1. The use of an even number of points including the origin results in an asymmetric distribution. Despite the need to sample $\Sigma$ along both the positive and negative parts of the real axis, using a single line allows us to use around the same number of sampling points as in the double-parallel sampling of $W$, typically about 20 , for both MPA- $W$ and MPA- $\Sigma$. More details on the MPA- $\Sigma$ sampling and the convergence with the number of poles $n_{\Sigma}$ can be found in Sec. II of Ref. [73].

MPA- $\Sigma$ requires a particularly accurate interpolation around $z=0$ for obtaining accurate QP energies, which sometimes requires a more precise sampling. This can be done by benchmarking the sampling for one or a few selected QP states against the corresponding FF calculations. The sampling can then be replicated for the remaining QPs, without extending the FF calculations. As
in the case of MPA- $W$, the computational cost of the $\Sigma$ evaluation can be compared in terms of the number of frequency points for which $\Sigma$ is explicitly evaluated. Therefore, given a FF grid spacing of $\Delta \omega=0.1 \mathrm{eV}$ in a frequency interval of 100 eV , MPA- $\Sigma$ is typically around 50 times more efficient than the FF $\Sigma$ evaluation.

\section*{D. MPA for the Green's function}

As in MPA- $W$ and MPA- $\Sigma$, one could construct a multipole-Padé representation of the Green's function from the interpolation of the numerical data, $\left\{z_{i}, G\left(z_{i}\right)\right\}$. However, with MPA- $\Sigma$ in place, it is more convenient to obtain an MPA-G representation from the Dyson equation in Eq. (3) (see also Refs. [31, 74] and the algorithmic-inversion-method in a sum-over-poles (AIM-SOP) representation of Refs. [59, 60, 62]).

Given Eq. (16), the total MPA- $\Sigma$ can be written in its Padé representation as
$$
\begin{equation*}
\Sigma^{\mathrm{MPA}-\Sigma}(z)=\frac{\Sigma_{x} B_{n_{\Sigma}}(z)+A_{n_{\Sigma}-1}(z)}{B_{n_{\Sigma}}(z)} \tag{18}
\end{equation*}
$$

Notice that if we apply physical constraints after the factorization in Eq. (17), and fit the residues thereafter, we will need to reconstruct both the $A_{n_{\Sigma}-1}(z)$ and $B_{n_{\Sigma}}(z)$ polynomials from the new poles and residues using Eq. (18), which is straightforward. We can then obtain MPA- $G$ as
$$
\begin{align*}
G^{\mathrm{MPA}-\Sigma}(z) & \equiv \frac{B_{n_{\Sigma}}(z)}{C_{n_{\Sigma}+1}(z)}  \tag{19}\\
C_{n_{\Sigma}+1}(z) & =z B_{n_{\Sigma}}(z)-\Sigma_{x} B_{n_{\Sigma}}(z)-A_{n_{\Sigma}-1}(z)
\end{align*}
$$

The poles of $G^{\mathrm{MPA}-\Sigma}$ will then correspond to the zeros of $C_{n_{\Sigma}+1}(z)$. Analogous to $B_{n_{\Sigma}}(z)$, the $C_{n_{\Sigma}+1}(z)$ polynomial can be factorized, e.g., using the companion matrix method [41]:
$$
\begin{equation*}
C_{n_{\Sigma}+1}(z)=\prod_{p}^{n_{\Sigma}+1}\left(z-\epsilon_{p}\right) \tag{20}
\end{equation*}
$$
while the residues can be computed using the residue theorem. Constructed in this fashion, the poles of $G^{\mathrm{MPA}-\Sigma}$ do not necessarily respect the time ordering, although this can be imposed in a second step.

The resulting multipole-Padé representation of $G$ for each state $n \mathbf{k}$ is then given by
$$
\begin{equation*}
G_{n \mathbf{k}}^{\mathrm{MPA}-\Sigma}(z)=\sum_{p=1}^{n_{\Sigma}+1} \frac{Z_{n \mathbf{k} p}}{z-\epsilon_{n \mathbf{k} p}} \tag{21}
\end{equation*}
$$
where the poles are obtained (see also Ref. [59]) as
$$
\begin{equation*}
Z_{n \mathbf{k} p}=\frac{\prod_{i}^{n_{\Sigma}}\left(\epsilon_{n \mathbf{k} p}-\xi_{n \mathbf{k} i}\right)}{\prod_{i \neq p}^{n_{\Sigma}+1}\left(\epsilon_{n \mathbf{k} p}-\epsilon_{n \mathbf{k} i}\right)} \tag{22}
\end{equation*}
$$

Notice that $G^{\mathrm{MPA}-\Sigma}$ has one pole more than $\Sigma^{\mathrm{MPA}-\Sigma}$, corresponding to the QP pole. As mentioned in Sec. II A and as will be illustrated in Appendix. A, the remaining poles correspond to satellites that emerge from the poles of $\Sigma^{\mathrm{MPA}-\Sigma}$. All the QP and satellites $\epsilon_{n \mathbf{k} p}$ poles are solutions of the QP equation in Eq. (5).

The MPA- $G$ representation overcomes the limitations of the approximation introduced by the linearized QP equation in Eq. (6), for both the positions and the residues of the QP poles. It also preserves important analytical properties of the solutions of non-linear eigenvalue equations with rational self-energy potentials [75]. From Eq. (22), it follows that
$$
\begin{equation*}
Z_{n \mathbf{k} p}=\left[1-\left.\frac{\partial \Sigma_{n \mathbf{k}}^{\mathrm{MPA}-\Sigma}(z)}{\partial z}\right|_{z=\epsilon_{n \mathbf{k} p}}\right]^{-1} \tag{23}
\end{equation*}
$$
while the sum rule of Eq. (8) is obeyed and simplifies to
$$
\begin{equation*}
\sum_{p} Z_{n \mathbf{k} p}=1 . \tag{24}
\end{equation*}
$$

Therefore, the numerical accuracy of the QP and satellites spectral weights of Eq. (22) depends only on the quality of the MPA- $\Sigma$ interpolation, while always comply exactly with the sum rule for the number of particles and holes.

\section*{E. Computational details}

DFT calculations were performed using the planewave Quantum Espresso package [76, 77] with the Perdew-Burke-Ernzerhof (PBE) variant of the generalized gradient approximation (GGA) [78]. We adopted the norm-conserving optimized Vanderbilt pseudopotentials of Ref. [79], with a kinetic energy cutoff for the wavefunctions of $70,30,100$, and 60 Ry respectively for Na, $\mathrm{Si}, \mathrm{Cu}$, and monolayer $\mathrm{MoS}_{2}$, and 85 Ry for both the NaCl ion-pair and the $\mathrm{F}_{2}$ molecule. The Brillouin zone was sampled with a $16 \times 16 \times 16$ Monkhorst-Pack grid for $\mathrm{Si}, \mathrm{Na}$, and $\mathrm{Cu}, 18 \times 18 \times 1$ for the monolayer $\mathrm{MoS}_{2}$ and $\Gamma$-only for NaCl and $\mathrm{F}_{2}$.

The $G_{0} W_{0}$ calculations were performed with yambo [70, 71]. In all the cases, the screened Coulomb potential was computed within MPA- $W$, using Eq. (9) with $n_{W}=8$ for Si and Na , and $n_{W}=12$ for Cu , the same sampling as in Refs. [41, 42]. Similarly, for $\mathrm{MoS}_{2}$, NaCl and $\mathrm{F}_{2}$ we used a sampling with 8 poles and a linear distribution. The method to evaluate the self-energy $\Sigma$, using full-frequency or the new MPA- $\Sigma$ method, is specified in each case. Since we are considering a $\mathrm{MoS}_{2}$ monolayer, we used the Monte-Carlo based averaging method ( $W$-av) for 2D semiconductors, first developed in Ref. [80] and then merged with MPA- $W$ in Ref. [43]. For metals, we used the constant approximation (CA) method [42] to treat the long-wavelength limit of the
intraband contributions. Both, $W$-av and CA are methods that can greatly accelerate the $\mathbf{k}$-point convergence of $G W$.

\section*{III. RESULTS}

\section*{A. Self-energy and Green's function of prototypical materials}

Figure 2 shows the real and imaginary parts of the $G_{0} W_{0}$ self-energy, i.e. $\operatorname{Re} \Sigma(\mathrm{a})-(\mathrm{c})$ and $\operatorname{Im} \Sigma(\mathrm{d})-(\mathrm{f})$, and the imaginary part of the Green's function, $\operatorname{Im} G$ (g)-(i), for a selected valence and a conduction state of $\mathrm{Si}, \mathrm{Na}$, and Cu . The solid lines give the results computed with a full-frequency evaluation of $\Sigma$ and $G$ (FF), serving as a benchmark, and the (dashed) dotted lines, the MPA- $\Sigma$ approach. The FF approach is evaluated on a homogeneous grid of 2000 frequency points. For MPA$\Sigma, 18$ frequencies are used for Si and 22 for Na and Cu , corresponding to $n_{\Sigma}=9$ and 11, respectively. The sampling frequencies used in the interpolation are indicated with red (valence) and blue (conduction) ticks along the horizontal panel edges.

The self-energies of Si and Na have the typical twopole structure characteristic of systems with a screening potential dominated by a single plasmon pole. As seen in the denominators of Eq. (10), the two $\Sigma$ poles are the result of the plasmon convoluted, respectively, with valence and conduction states. Cu presents a similar picture, but with several plasmon-like poles in $W_{0}$, as can be seen in Ref. [42], coupled with the single-particle poles of $G_{0}$, resulting in a richer structure of $\Sigma$.

We also tested the MPA- $\Sigma$ description for the $\mathrm{MoS}_{2}$ monolayer, the NaCl ion-pair, and the $\mathrm{F}_{2}$ molecule, i.e., materials with lower dimensionality. The results are shown in Fig. 3. We used an MPA- $\Sigma$ representation with 10, 11 and up to 14 poles for $\mathrm{MoS}_{2}, \mathrm{NaCl}$ and $\mathrm{F}_{2}$, respectively. Both Figs. 2 and 3 demonstrate an excellent agreement of MPA- $\Sigma$ and MPA- $G$ with the FF results.

In order to obtain an accurate MPA- $\Sigma$ representation while preserving the physical meaning of the main poles, in the above calculations, the number of sampling points of $\Sigma$ at each side of the frequency axis was adapted to each particular system and state, using the FF calculations as a reference. It is however convenient to establish a general sampling scheme that does not require a FF reference and can be applied to all the states of each system. For this purpose, we select a frequency distribution and number of poles similar to the one used for MPA- $W$. As mentioned in Sec. II C, in the case of $\Sigma$, the distribution is centered on each KS energy, with a small asymmetry on the frequency sampling, i.e., 1 or 2 frequency points more on the negative (positive) side for valence (conduction) states. The total number of frequency points is given by $2 n_{\Sigma}$, with $n_{\Sigma}=9$ for Si and Na , and $n_{\Sigma}=11$ for Cu .

Figure 4 shows the QP energies and Re $Z$ of a set of valence and conduction bands of $\mathrm{Si}, \mathrm{Na}$, and Cu , in a
wide range of energies and momenta. Figures 4 (a)-4(c) show the $G_{0} W_{0}$ energies found by recursively solving the QP equation in Eq. (5) with the FF $\Sigma$ representation, used here as a reference. They show the expected $G W$ stretching of the KS bands and, for Si, a band gap opening. Figures $4(\mathrm{~d})-4(\mathrm{f})$ compare the difference between the results of the non-linearized FF and the linearized FF QP equation, $\Delta \varepsilon^{\text {lin-FF }}=\varepsilon^{\mathrm{FF}}-\varepsilon^{\text {lin-FF }}$, (red squares), and the difference between the non-linearized FF and the analytical MPA- $\Sigma$ results, $\Delta \varepsilon^{\mathrm{MPA}-\Sigma}=\varepsilon^{\mathrm{FF}}-\varepsilon^{\mathrm{MPA}-\Sigma}$, (blue circles).

For valence states $\Delta \varepsilon^{\operatorname{lin}-\mathrm{FF}}$ increases with the distance from the Fermi level, consistent with the general trends for materials (see, e.g., Ref. [67]). In contrast, $\Delta \varepsilon^{\mathrm{MPA}-\Sigma}$ is almost zero for most of the valence and conduction states, with the exception of a few quasiparticles with more structure around $\Sigma\left(\omega=\varepsilon^{\mathrm{KS}}\right)$. This good agreement between MPA- $\Sigma$ and FF demonstrates the accuracy of the MPA- $\Sigma$ method, even with a simple frequency sampling scheme. As mentioned in Sec. II C, solving the linearized QP equation requires $\Sigma$ to be computed for one or two frequency points, whereas MPA- $\Sigma$ requires about 20. While the additional sampling points do increase computational costs, this comes with a significant improvement in accuracy. Critically, MPA- $\Sigma$ also allows for a straightforward evaluation of $\Sigma$ in its full-frequency range and gives access to an analytical representation of $G$.

Figures $4(\mathrm{~g})-4(\mathrm{i})$ show the linearized renormalization factor of Eq. (7), $Z^{\text {lin }}$, corresponding to the reference FF data evaluated with Eq. (12) (red squares) and the MPA$\Sigma$ representation of Eq. (14) (blue circles). They differ by less than 0.003, 0.010, and 0.025 for all the quasiparticles of $\mathrm{Si}, \mathrm{Na}$, and Cu , respectively. The results labeled MPA$G$ (yellow triangles) correspond to the non-linearized $Z$, obtained as the residue $Z_{p}$ of the QP pole in Eqs. (22) and (23). Therefore, comparing the MPA- $\Sigma$ and MPA- $G$ results corresponds to comparing $Z^{\text {lin }}$ with $Z$. For most of the quasiparticles, the linearization is a good approximation and $Z^{\mathrm{MPA}-\Sigma}$ is quite similar to $Z^{\mathrm{MPA}-G}$; however, for the Na and Cu states with more intense satellites ( $\operatorname{Re} Z \lesssim 0.8$ ), the deviation can increase to 0.045 and 0.070 , respectively. A detailed analysis of the limitations of Eq. (7) as an approximation of the spectral weight of the QP pole in different scenarios is presented in Appendix A . In particular, aside from the QP correction to the single-particle energies, it is shown that the deviation can increase with the QP broadening.

Figure 5 shows the QP band structure of Na (a), Si (b), monolayer $\mathrm{MoS}_{2}$ (c), and Cu (d), interpolated in kspace, as described in Appendix B. The width of the lines gives the imaginary part of the QP poles $\operatorname{Im}\left[\epsilon_{n \mathbf{k}}\right]$, while the color shade indicates the value of the renormalization factors $\operatorname{Re}\left[Z_{n \mathbf{k}}\right]$ for valence (orange shades) and conduction (purple shades) bands. In general, we find that both $\operatorname{Im}\left[\epsilon_{n \mathbf{k}}\right]$ and $\operatorname{Re}\left[Z_{n \mathbf{k}}\right]$ increase further away from the Fermi level, as a result of the QP pole and the satellites broadening and merging in a single peak. However, in the

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_17_755328709aa12b82ff93g-07.jpg?height=1288&width=1725&top_left_y=192&top_left_x=203}
\captionsetup{labelformat=empty}
\caption{FIG. 2. MPA- $\Sigma$ and MPA- $G$ results compared with the corresponding full-frequency (FF) $\Sigma$ and interacting $G$, for semiconducting and metallic materials. The plots show the spectra of a selected valence (yellow) and a conduction (purple) state of Si (left), Na (middle), and Cu (right), with KS energies relative to the valence band minimum (VBM), as specified in their respective panels. (a)-(c) and (d)-(f) Compares the real and imaginary parts of $\Sigma$, and (g)-(i) the imaginary part of $G$, for the respective materials. The solid orange and purple curves indicate the FF results, while the corresponding MPA- $\Sigma$ and MPA- $G$ results are plotted as black dotted (valence) and dashed-dotted (conduction) lines. In the top and center panels, the red (blue) ticks along the horizontal panel borders indicate the distribution of sampling points along the real frequency axis used as input in the MPA- $\Sigma$ interpolation of the given valence (conduction) state.}
\end{figure}
energy range in which multiple bands of different character cross, the picture becomes more complex showing a non-monotonic behavior. The metallic band of Na is an exception, with $\operatorname{Im}\left[\epsilon_{n \mathbf{k}}\right]$ and $\operatorname{Re}\left[Z_{n \mathbf{k}}\right]$ decreasing without crossing any other band. This is in line with the increased weight of the satellites, as discussed in Sec. III B.

\section*{B. Spectral band structures}

The spectral functions probed in photoemission and inverse photoemission experiments account for the contributions of each state, according to the polarization of the incoming light [8]. In a typical ab initio calculation, the total $\Sigma$ and $G$ spectral functions, $A_{\Sigma}(\mathbf{k}, \omega) \equiv$
$1 / \pi \sum_{n} \operatorname{Im}\left[\sum_{n \mathbf{k}}(\omega)\right]$ and $A_{G}(\mathbf{k}, \omega) \equiv 1 / \pi \sum_{n} \operatorname{Im}\left[G_{n \mathbf{k}}(\omega)\right]$, are evaluated with a finite number of bands. Since we use time-ordered operators, the spectra have opposite signs for valence and conduction states, which makes the background intensity sensitive to the number of bands included. In Sec. III of Ref. [73] we provide detailed descriptions on how we plot spectral functions, especially when few bands are included.

Fig. 6 shows the computed $A_{\Sigma}(\mathbf{k}, \omega)$ (top panels) and $A_{G}(\mathbf{k}, \omega)$ (bottom panels) spectral band structures of Na [Figs. 6(a) and 6(e)], Si [Figs. 6(b) and 6(f)], monolayer $\mathrm{MoS}_{2}$ [Figs. 6(c) and 6(g)] and Cu [Figs. 6(d) and 6(h)], obtained with MPA- $\Sigma$ and MPA- $G$. Their accuracy is comparable with the spectral functions obtained with a FF evaluation (see Sec. III of Ref. [73]). As in Fig. 5,

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_17_755328709aa12b82ff93g-08.jpg?height=1281&width=1720&top_left_y=197&top_left_x=208}
\captionsetup{labelformat=empty}
\caption{FIG. 3. MPA- $\Sigma$ and MPA- $G$ results compared with the corresponding full-frequency (FF) $\Sigma$ and interacting $G$, for materials with reduced dimensionality. The plots show the spectra of a selected valence (yellow) and a conduction (purple) state of the monolayer $\mathrm{MoS}_{2}$ (left), the NaCl ion-pair (middle), and the $\mathrm{F}_{2}$ (right) molecule, with KS energies relative to the valence band minimum (VBM), as specified in their respective panels. As in Fig. 2, the MPA- $\Sigma$ and MPA- $G$ results are plotted as black dotted (valence) and dashed-dotted (conduction) lines, while the red (blue) ticks along the horizontal panel borders indicate the distribution of sampling points along the real frequency axis used as input in the MPA- $\Sigma$ interpolation of the given valence (conduction) state.}
\end{figure}
to build such spectral band structures we use the spline interpolation in $\mathbf{k}$-space detailed in Appendix B. Due to the time ordering, each state has positive (orange shades) and negative (purple shades) $1 / \pi \operatorname{Im}\left[\Sigma_{n \mathbf{k}}(\omega)\right]$ components, even if positive (negative) intensities are predominant for valence (conduction) states. Therefore, the spectral contribution of each state to the total spectral function $A_{\Sigma}(\mathbf{k}, \omega)$ is not always trivial, as in the case of Na (see Fig. S2 of Ref. [73]).
$A_{\Sigma}(\mathbf{k}, \omega)$ exhibits bands that arise from the coupling of the plasmon with the single KS states, as discussed in the previous section. From now on we will call them $\Sigma$ bands. Essentially, the position of the $\Sigma$ valence (conduction) bands corresponds to the independent-particle bands of $G_{0}$ shifted down (up) by the plasmon energy, $\zeta_{n \mathbf{k}} \sim \varepsilon_{n \mathbf{k}}^{\mathrm{KS}} \pm \omega_{\mathrm{pl}}$, where $\omega_{\mathrm{pl}}$ has a value of $5.8,16.6,10.98$, and 26.5 eV
for $\mathrm{Na}, \mathrm{Si}, \mathrm{MoS}_{2}$, and Cu respectively. Therefore, in semiconducting materials like Si , the $\Sigma$ bands present a gap given by the KS gap plus twice the plasmon energy $\left(\varepsilon_{\text {gap }}^{\Sigma}=\varepsilon_{\text {gap }}^{\mathrm{KS}}+2 \omega_{\mathrm{pl}}\right)$.
$A_{G}(\mathbf{k}, \omega)$ (panels (e-h)) also exhibits bands that we will call $G$ bands. There are two types, those coming from the QP pole, $\varepsilon_{n \mathbf{k}}^{\mathrm{QP}}$, and those formed from the satellites, $\varepsilon_{n \mathbf{k}}^{\mathrm{sat}}$. The QP bands are shifted with respect to the $G_{0}$ bands by the $G_{0} W_{0}$ correction, according to Eq. (5). The satellite bands (called sidebands in Ref. [4]) correspond to replicas of the QP bands located at larger energies. As already seen in Figs. 2(g)-2(i), the QP peak typically dominates the spectral function. To better discern the satellite structures, we impose thresholds to the $A_{G}(\mathbf{k}, \omega)$ color maps, corresponding to $\pm 0.25, \pm 0.10, \pm 0.16$, and $\pm 0.22 \mathrm{eV}^{-1}$ for $\mathrm{Na}, \mathrm{Si}, \mathrm{MoS}_{2}$, and Cu , respectively. In

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_17_755328709aa12b82ff93g-09.jpg?height=1294&width=1725&top_left_y=189&top_left_x=200}
\captionsetup{labelformat=empty}
\caption{FIG. 4. QP energies and renormalization factors of Si [left: (a), (d), (g)], Na [middle: (b), (e), (h)] and Cu [right: (c), (f), (i)] as a function of their KS energies. (a)-(c) Show the QP solution obtained from the numerical full-frequency (FF) self-energy using a recursive method. (d)-(f) Show the difference between the FF QP equation and the FF linearized results (red squares) and the analytical MPA- $\Sigma$ solution (blue circles). (g)-(i) Correspond to the linearized $\operatorname{Re}\left[Z^{\text {lin }}\right]$ obtained with the FF MPA- $W$ approach of Eq. (12) (red squares) and the MPA- $\Sigma$ representation of Eq. (14) (blue circles), and the non-linearized $\operatorname{Re}[Z]$ computed as the residue of the QP pole in the MPA-G representation of Eq. (22) (yellow triangles).}
\end{figure}
the case of the metallic band of Na , the intensity of the satellite around the $\Gamma$ point is similar in magnitude to the QP peak, which is interpreted as a plasmaron [4, 5, 81, 82]. Since the satellites emerge from the plasmonic structures in $\Sigma[5]$, the satellite bands are shifted with respect to the QP bands by roughly the energy of the plasmon [21], $\varepsilon_{n \mathbf{k}}^{\text {sat }} \sim \varepsilon_{n \mathbf{k}}^{\text {QP }} \pm \omega_{\text {pl }}$, and, in turn, are shifted with respect to the $\Sigma$ bands by the $G_{0} W_{0}$ correction, $\varepsilon_{n \mathbf{k}}^{\mathrm{sat}}-\zeta_{n \mathbf{k}} \sim \varepsilon_{n \mathbf{k}}^{\mathrm{QP}}-\varepsilon_{n \mathbf{k}}^{\mathrm{KS}}$. One of the advantages of the MPA- $G$ representation is the possibility to analytically separate the spectral contributions of the QP pole and the satellites, as done in Fig. S4 of Ref. [73].

Figure 6 exhibits a qualitative difference between the $\Sigma$ bands of Na [Fig. 6(a)] and Si [Fig. 6(b)], which are easy to isolate, compared to those of $\mathrm{MoS}_{2}$ [Fig. 6(c)] and Cu [Fig. 6(d)], which are generally broader and overlap more
with each other. This is in line with both the more complex $\Sigma$ structure of $\mathrm{MoS}_{2}$ and Cu , and the fact that the energy separation of the bands is smaller than the width of the main plasmon. As a result, $\mathrm{MoS}_{2}$ exhibits rather flat and broadened valence and conduction $\Sigma$ bands, with an apparent gap given by secondary peaks at energies smaller than the main plasmon. The $\Sigma$ bands of Cu also shows a main flat and broadened dispersion, even if some secondary bands are still visible. As a consequence, the satellites bands of $\mathrm{MoS}_{2}$ and Cu are also broadened, resulting in the diffuse background of Fig. $6(\mathrm{~g}, \mathrm{~h})$.

The results in Fig. 6 illustrate the connection between the poles of $\Sigma$ and $G$, whose analyses allow to identify the QP pole from the satellites, as also illustrated with the toy models introduced in Appendix A. Due to this connection, the accuracy of both the $\Sigma$ and $G$ bands requires a good description of the screening, which can be
-1.00
-0.75
-0.50
-0.25
0.00
0.25
0.50
0.75
1.00

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_17_755328709aa12b82ff93g-10.jpg?height=695&width=1779&top_left_y=276&top_left_x=178}
\captionsetup{labelformat=empty}
\caption{FIG. 5. Band structure of Na , Si , monolayer $\mathrm{MoS}_{2}$, and Cu , with a variable width representing the imaginary part of the quasiparticles $\operatorname{Im}\left[\epsilon_{n \mathbf{k}}\right]$. The color map is given by $-\operatorname{sign}\left(\operatorname{Re}\left[\epsilon_{n \mathbf{k}}\right]\right) \operatorname{Re}\left[Z_{n \mathbf{k}}\right]$, representing the renormalization factor for valence (orange shades) and conduction (purple shades) states.}
\end{figure}
improved with vertex corrections $[21,81]$ or cumulant expansions [16, 21, 22, 82]. At finite temperature, electronphonon interactions are expected to further renormalize and broaden the QP peak [16, 23], as shown for example in Refs. [12, 83, 84] for the case of Cu .

\section*{IV. CONCLUSIONS}

We have presented MPA- $\Sigma$, a robust method to efficiently approximate the frequency dependence of the $G W$ self-energy as a multipole-Padé representation, typically with around 10 poles. Such representation, similar to the multipole approximation for the screening interaction, MPA- $W$, is built from numerical data evaluated on around 20 frequency points in the complex plane, thus avoiding explicit evaluations of the self-energy on dense frequency grids (of the order of 1000 frequencies in our cases). MPA- $\Sigma$ allows also to solve the QP equation analytically and obtain an MPA- $G$ representation of the interacting Green's function, from which all the spectral properties can be easily extracted, including the positions, broadenings, and the spectral weights of the QP pole and its satellites.

Combining MPA- $W$ and MPA- $\Sigma$ is a computationally powerful approach, reducing the number of frequency evaluations by a factor $\sim 50^{2}$, compared to full-frequency approaches, while providing spectra with comparable numerical accuracy. The excellent accuracy of this method has been verified for several materials: bulk Si, Na, and Cu , monolayer $\mathrm{MoS}_{2}$, the NaCl ion pair, and the $\mathrm{F}_{2}$ molecule. The efficiency of the MPA method allows us to compute $\Sigma$ and $G$ spectra in a wide energy range.

In particular, our results for NaCl and $\mathrm{F}_{2}$ exhibit features beyond the typical energy range of the $G W$ spectra found in the literature for such molecular species. We have also presented a method for interpolating the spectra of higher-dimensional systems in momentum space, and construct $\Sigma$ and $G$ spectral band structures. For Na , Si, monolayer $\mathrm{MoS}_{2}$, and Cu , we report full $\Sigma$ and $G$ spectral band structures, while isolating the contributions from the QP pole and the satellites.

The spectral weights of both the QP pole and the satellites computed with MPA- $G$ comply with the sum rule for the number of particles and holes, providing an accurate way to evaluate the renormalization factor beyond the linearized QP equation. Therefore, such analytical representations can be useful in understanding the physical nature of the QP picture and the renormalization factor. In the following Appendixes, we present toy models that capture the most typical situations when solving the QP equation in prototypical materials, exposing the limitations of its linearization in different regimes from weak to strong correlation.

\section*{ACKNOWLEDGMENTS}

This work was funded by the Research Council of Norway through the MORTY project (315330). Access to high performance computing resources was provided by UNINETT Sigma2 (NN9711K) in Norway, and by EuroHPC Joint Undertaking through the project EHPC-EXT-2022E01-022 that grants access to Leonardo-Booster@Cineca, Italy. We acknowledge Andrea Ferretti for insightful discussions and comments on

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_17_755328709aa12b82ff93g-11.jpg?height=1460&width=1781&top_left_y=178&top_left_x=176}
\captionsetup{labelformat=empty}
\caption{FIG. 6. Spectral band structures $A_{\Sigma}(\mathbf{k}, \omega)$ (a)-(d) and $A_{G}(\mathbf{k}, \omega)$ (e)-(h) of Na (a), (e); Si (b), (f); $\mathrm{MoS}_{2}$ (c), (g); and Cu (d), (h); computed with MPA- $\Sigma$ and MPA- $G$. We considered the metallic band and the next valence and 2 conduction bands of Na along the $\Gamma N$ path, 4 valence and 6 conduction bands of Si along the $L \Gamma X$ path, 9 valence and 7 conduction bands of $\mathrm{MoS}_{2}$ along the $\Gamma M K \Gamma$ path, and 6 valence and 9 conduction bands of Cu along the $L \Gamma X$ path.}
\end{figure}
the manuscript, and Kristian S. Thygesen and Mikael Kuisma for stimulating discussions.

\section*{V. DATA AVAILABILITY STATEMENTS}

The data generated in this article are openly available [85].

\section*{Appendix A: The quasiparticle picture in the MPA representation}

As discussed in Sec. II, it is well known that the linearized renormalization factor in Eq. (7) is not always
a good approximation for the QP spectral weight, while there is no established method to compute the spectral weight of satellites. Equation (7) is expected to work when the QP correction is small, although it is still widely used in ab initio calculations even in the case of large corrections, especially in high-throughput studies (see, e.g., Ref. [67]). In the following sections, we expose in detail some of the limitations of such linearization, while highlighting the advantages of the MPA- $G$ representation, consistent with Eqs. (23) and (24), to analyze the QP picture in the limit of weak and strong correlation. To that aim, we use toy MPA- $\Sigma$ models in which the number of poles is limited to one, $\Sigma^{\mathrm{s} 1}$, and two, $\Sigma^{\mathrm{s} 2}$, and their corresponding MPA- $G$ solutions, $G^{\mathrm{s} 1}$ and $G^{\mathrm{s} 2}$. Such models, although simple, still capture the most typical
situations when solving the QP equation in prototypical materials.

In the next sections we show that (1) the linearized QP equation can incorrectly predict values of $Z^{\text {lin }}<0.5$ even when the QP picture holds ( $Z>0.5$ ). (2) The error introduced by the linearization, not only increases with the magnitude of the QP correction, but also with the QP broadening. (3) The static term in the self-energy plays an important role in the QP picture, affecting the distribution of the spectral weight between the QP pole and the satellites. For certain conditions, the satellites can have a larger spectral weight than the QP pole. (4) In the case of a self-energy with multiple poles, the linearized $Z^{\text {lin }}$ can deviate from the exact value $Z$, even in the limit $Z^{\text {lin }} \rightarrow 1$, which does not necessarily correspond to a vanishing linearized QP correction.

\section*{1. Toy MPA- $\Sigma$ model with a single pole}

We consider a self-energy model with one pole:
$$
\begin{equation*}
\Sigma^{\mathrm{s} 1}(\omega)=x \xi+\frac{\lambda}{1-\lambda} \frac{\xi^{2}}{\omega-\xi} \tag{A1}
\end{equation*}
$$
where $\omega$ is centered on the given QP energy, $x \xi$ accounts for static contributions such as the exchange interaction and vertex corrections, and the second term accounts for correlation with a single pole at the plasmon energy $\xi$. The residue of the pole, $S=\xi^{2} \lambda /(1-\lambda)$, is defined in terms of a parameter $\lambda$ so that the linearized renormalization factor in Eq. (7), independently of the other two parameters, is given by:
$$
\begin{equation*}
Z_{\mathrm{s} 1}^{\mathrm{lin}} \equiv\left[1-\left.\frac{\partial \Sigma^{\mathrm{s} 1}(\omega)}{\partial \omega}\right|_{\omega=0}\right]^{-1}=1-\lambda \tag{A2}
\end{equation*}
$$

The interacting Green's function corresponding to Eq. (A1) is obtained by inverting the Dyson equation, $G(\omega)=[\omega-\Sigma(\omega)]^{-1}$, resulting in:
$$
\begin{equation*}
G^{\mathrm{s} 1}(\omega)=\sum_{p=1}^{2} \frac{Z_{p}^{\mathrm{s} 1}(x, \lambda)}{\omega-\epsilon_{p}^{\mathrm{s1}}(x, \lambda, \xi)} \tag{A3}
\end{equation*}
$$
which has two poles, $\epsilon_{p}^{\mathrm{s} 1}$, with residues $Z_{p}^{\mathrm{s} 1}$, given by
$$
\begin{align*}
\epsilon_{1,2}^{\mathrm{s} 1} & =\frac{\xi}{2}(1+x \mp \sqrt{D}) \\
Z_{1,2}^{\mathrm{s} 1} & =\frac{1}{2}\left(1 \pm \frac{1-x}{\sqrt{D}}\right) \tag{A4}
\end{align*}
$$
where
$$
\begin{equation*}
D \equiv(1-x)^{2}+4 \lambda /(1-\lambda) \tag{A5}
\end{equation*}
$$

The poles are proportional to $\xi$ and can be rescaled to obtain dimensionless units. For all the considered values of the parameters, $\left|\operatorname{Re}\left[\epsilon_{1}^{\mathrm{s} 1}\right]\right|<\left|\operatorname{Re}\left[\epsilon_{2}^{\mathrm{s} 1}\right]\right|$. The pole labeled
as 1 is identified as the QP pole and the one labeled as 2 is the satellite, thus, $Z_{\mathrm{s} 1}^{\mathrm{MPA}-G}=Z_{1}^{\mathrm{s} 1}$.

We first focus on the correlation effects by setting $x=0$. Notice that $\lambda=0$ corresponds to $S=0$ (zero correlation), while $\lambda \rightarrow 1$ corresponds to the limit of infinite correlation. Figure 7 shows the real and imaginary parts of the scaled poles $\epsilon_{p}^{\mathrm{s} 1} / \xi$ [Figs. 7(a) and 7(c)], and the residues $Z_{p}^{\mathrm{s} 1}$ [Figs. 7(b) and 7(d)] of $G^{\mathrm{s} 1}$ as functions of $\operatorname{Re} \lambda$, for $\operatorname{Im} \lambda=0$ (blue curves) and $\operatorname{Im} \lambda=0.3$ (orange curves). Since $x=0$, $\operatorname{Re} \lambda=0$ and $\operatorname{Im} \lambda=0$ also correspond to the independent-particle limit, in which the QP state has a zero self-energy correction $\left(\epsilon_{1}^{\mathrm{s} 1}=0\right)$ and carries the whole spectral weight ( $Z_{\mathrm{s} 1}^{\mathrm{MPA}-G}=1$ ). As a consequence, the satellite $\epsilon_{2}^{\mathrm{s} 1}=\xi$ vanishes ( $Z_{2}^{\mathrm{s} 1}=0$ ).

In both the cases of $\operatorname{Im} \lambda=0$ and 0.3 , as $\operatorname{Re} \lambda$ increases, $\operatorname{Re}\left[\epsilon_{1}^{\mathrm{s} 1}\right]$ goes to negative values with decreasing spectral weight $\operatorname{Re}\left[Z_{1}^{\mathrm{s} 1}\right]$, while $\operatorname{Re}\left[\epsilon_{2}^{\mathrm{s} 1}\right]$ increases from $\xi$ (solid gray line in panel (a)), with increasing $\operatorname{Re}\left[Z_{2}^{s 1}\right]$. The finite $\operatorname{Im} \lambda$ induces a finite imaginary part in the poles and residues. As $\operatorname{Re} \lambda$ increases, both $\operatorname{Im}\left[\epsilon_{p}^{\mathrm{s} 1}\right]$ increase in modulus, while $\operatorname{Im}\left[Z_{p}^{\mathrm{s} 1}\right]$ decrease despite $\operatorname{Im} \lambda$ being constant. The broadening of the poles avoids the infinite correlation limit, affecting the curvature of $\operatorname{Re}\left[\epsilon_{p}^{\mathrm{s} 1}\right]$ and $\operatorname{Re}\left[Z_{p}^{\mathrm{s} 1}\right]$ (orange vs. blue curves in panels (a, b)).

As illustrated by the dotted black line in Fig. 7(b), when we move from the independent-particle limit, the renormalization factor $Z_{\mathrm{s} 1}^{\mathrm{lin}}$ starts deviating from $Z_{\mathrm{s} 1}^{\mathrm{MPA}-G}$ (dark blue curve), while for $\operatorname{Im} \lambda=0.3$ (dark orange curve), both definitions already differ at $\operatorname{Re} \lambda=0$ and only their real parts coincide around $\operatorname{Re} \lambda=0.2$. The plot shows that, $Z^{\text {lin }}$ cannot be taken as an indicator of the validity of the QP picture beyond the regime of weak correlation ( $\lambda \approx 0$ ), since $\operatorname{Re}\left[Z_{\mathrm{s} 1}^{\mathrm{MPA}-G}\right]>0.5$ in the whole interval, even when $\operatorname{Re}\left[Z_{\mathrm{s} 1}^{\text {lin }}\right]<0.5$. In the large correlation limit ( $\lambda \rightarrow 1$ ), the energy position of the poles diverges $\left(\operatorname{Re}\left[\epsilon_{1,2}^{\mathrm{s} 1}\right] \rightarrow \mp \infty\right)$ with similar spectral weight $\left(\operatorname{Re}\left[Z_{1,2}^{\mathrm{s} 1}\right] \rightarrow 0.5\right)$.

In Fig. 8 we analyze the effects of the static term. The plots are analogous to Fig. 7. Notice that we have extended the range of the plots to $\operatorname{Re} \lambda=1.1$, to test the QP picture even for extreme values. In this case, we have fixed $\operatorname{Im} \lambda=0.01$ and considered three values of $x$, resulting in a picture similar to the one described in Fig. 7, where $\operatorname{Re}\left[Z_{1}^{s 1}\right]$ is always larger than 0.5 . The value $x=0$ corresponds to the previously discussed case of a self-energy with only the correlation term. The main effect of a finite value, illustrated with $x= \pm 0.3$, is to change the concavity of the residues according to its sign (orange/green vs. blue curves in panel (b)). At variance with $x \leq 0$, for positive values $\operatorname{Re}\left[Z_{\mathrm{s} 1}^{\mathrm{MPA}-G}\right]<\operatorname{Re}\left[Z_{\mathrm{s} 1}^{\operatorname{lin}}\right]$ close to $\operatorname{Re} \lambda \rightarrow 0$. In the case of $x>1$, the spectral weight of the satellite is larger than the QP pole, as $\operatorname{Re}\left[Z_{1}^{\mathrm{s} 1}\right]<\operatorname{Re}\left[Z_{2}^{\mathrm{s} 1}\right]$.

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_17_755328709aa12b82ff93g-13.jpg?height=998&width=1736&top_left_y=187&top_left_x=200}
\captionsetup{labelformat=empty}
\caption{FIG. 7. Real and imaginary parts of the scaled poles (a), (c) and residues (b), (d) of the Green's function model of Eq. (A3) as a function of the $\lambda$ parameter, where we have fixed $x=0$. The solid horizontal gray line in (a) corresponds to $\operatorname{Re}\left[\epsilon_{p} / \xi\right]=1$, while a dashed line is drawn at zero. In (b), the gray dashed horizontal line at $\operatorname{Re}\left[Z_{p}\right]=0.5$ represents the limit of the QP picture, while the gray solid line at 1 indicates the sum rule of Eq. (24) ( $Z_{1}+Z_{2}=1$ ). The dotted black line corresponds to $\operatorname{Re}\left[Z^{\text {lin }}\right]=1-\operatorname{Re} \lambda$, contrasted with $\operatorname{Re}\left[Z_{\mathrm{s} 1}^{\mathrm{MPA}-G}\right]=\operatorname{Re}\left[Z_{1}\right]$.}
\end{figure}

\section*{2. Toy MPA- $\Sigma$ model with two poles}

Similarly to the $\Sigma^{\mathrm{s} 1}$ model discussed in Sec. A, here we introduce a toy MPA- $\Sigma$ model with two poles, one at $\epsilon_{1}=\xi$ and the other at a larger energy $\epsilon_{2}=(1+a) \xi$ :
$$
\begin{align*}
\Sigma^{\mathrm{s} 2}(\omega) & =x \xi+ \\
& \frac{2 \lambda-1}{2(1-\lambda)} \frac{\xi^{2}}{\omega-\xi}+\frac{(1+a)^{2}}{2(1-\lambda)} \frac{\xi^{2}}{\omega-(1+a) \xi} \tag{A6}
\end{align*}
$$
where the residues, $S_{1}=\xi^{2}(2 \lambda-1) /[2(1-\lambda)]$ and $S_{2}= \xi^{2}(1+a)^{2} /[2(1-\lambda)]$ are constrained so that the following expression remains invariant, as for $\Sigma^{\mathrm{s} 1}$ :
$$
\begin{equation*}
Z_{\mathrm{s} 2}^{\operatorname{lin}} \equiv\left[1-\left.\frac{\partial \Sigma^{\mathrm{s} 2}(\omega)}{\partial \omega}\right|_{\omega=0}\right]^{-1}=1-\lambda . \tag{A7}
\end{equation*}
$$

The corresponding MPA- $G$ representation has three poles and a similar form, only with the additional parameter $a$ :
$$
\begin{equation*}
G^{\mathrm{s} 2}(\omega)=\sum_{p=1}^{3} \frac{Z_{p}^{\mathrm{s} 2}(x, \lambda, a)}{\omega-\epsilon_{p}^{\mathrm{s} 2}(x, \lambda, a, \xi)} . \tag{A8}
\end{equation*}
$$

Notice that for $a=0, \Sigma^{\mathrm{s} 2}$ simplifies to $\Sigma^{\mathrm{s} 1}$, and therefore this parameter can be used to turn on the second
pole. We then fix $x=0$ and $\operatorname{Im} \lambda=0$, and compare the two models. Figure 9 shows the real part of the scaled poles, $\epsilon_{p}^{\mathrm{s} 2} / \xi$ [Fig. 9(a)], and the residues, $Z_{p}^{\mathrm{s} 2}$ [Fig. 9(b)], of $G^{\mathrm{s} 2}$ as functions of $\operatorname{Re} \lambda$, for $a=0$ (blue curves) and $a=1$ (orange curves). The overall picture discussed in Sec. A is similar for the two models, the first one having a satellite increasing from $\epsilon_{2}^{\mathrm{s} 1}=\xi$, while the second has two, one increasing from $\epsilon_{2}^{\mathrm{s} 1}=(1+a) \xi$ and the second remaining around $\epsilon_{3}^{\mathrm{s} 2}=\xi$ with a vanishing residue $Z_{3}^{\mathrm{s} 2}$. However, a finite $a$ induces a finite $Z_{3}^{\mathrm{s}^{2}}$, with its larger modulus at $\operatorname{Re} \lambda=0$, which introduces a deviation between $Z_{\mathrm{s} 2}^{\operatorname{lin}}$ and $Z_{\mathrm{s} 2}^{\mathrm{MPA}-G}$ around $\lambda=0$. Notice that even if $Z_{\mathrm{s} 2}^{\operatorname{lin}}=1$ for $\lambda=0$, the linearized QP correction does not vanish for finite $a$, as $\Sigma^{\mathrm{s} 2}(\omega=0) / \xi=-(2 l+a) /[2(1-l)]$. The weak correlation limit is found for both $\lambda \rightarrow 0$ and $a \rightarrow 0$ simultaneously.

\section*{Appendix B: Spline interpolation in k-space}

To obtain smooth plots of the $\Sigma$ and $G$ spectral band structures, similar to the band structures of the response function computed in Ref. [86], we need to perform an interpolation on the $\mathbf{k}$ space for each frequency $z_{i}$. Such an interpolation can be cumbersome due to the disper-

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_17_755328709aa12b82ff93g-14.jpg?height=996&width=1744&top_left_y=189&top_left_x=197}
\captionsetup{labelformat=empty}
\caption{FIG. 8. Real and imaginary parts of the scaled poles (a), (c) and residues (b), (d) of the Green's function model of Eq. (A3) as a function of the $\lambda$ parameter, where we have fixed its imaginary part to a small value of $\operatorname{Im} \lambda=0.01$. The solid and dashed horizontal gray lines and the black dotted line in (b) are analogous to Fig. 7.}
\end{figure}

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_17_755328709aa12b82ff93g-14.jpg?height=505&width=1722&top_left_y=1417&top_left_x=203}
\captionsetup{labelformat=empty}
\caption{FIG. 9. Real part of the poles (a) and residues (b) of the Green's function model of Eq. (A8) as a function of Re $\lambda$, where we have fixed $\operatorname{Im} \lambda=0$.}
\end{figure}
sion of the different bands. The MPA- $\Sigma$ and MPA- $G$ representations can simplify this interpolation, since it is sufficient to interpolate the poles and the residues of each multipole-Padé model. The interpolation of numerical data can also be simplified by considering an auxiliary set of frequencies centered on the KS energies $z_{i}^{\prime} \equiv z_{i}-\epsilon_{n \mathbf{k}}^{K S}$. Since $z_{i}^{\prime}$ carries the dispersion of the bands, $\Sigma\left(z_{i}^{\prime}, \mathbf{k}\right)$ and $G\left(z_{i}^{\prime}, \mathbf{k}\right)$ are much smoother than the original $\Sigma\left(z_{i}, \mathbf{k}\right)$ and $G\left(z_{i}, \mathbf{k}\right)$.

We start by interpolating each KS band $n$ :
$$
\begin{equation*}
\forall n: \epsilon_{n}^{\mathrm{KS}}(\mathbf{k} \text {-grid }) \rightarrow f_{n}^{\epsilon}(\mathbf{k}), \tag{B1}
\end{equation*}
$$
where $f_{n}^{\epsilon}$ is the interpolating function. Similarly, we interpolate $\Sigma$ and $G$ for all frequencies along the direction of each band:
$$
\begin{array}{ll}
\forall n, i: \quad & \Sigma_{n}\left(z_{i}^{\prime}, \mathbf{k} \text {-grid }\right) \rightarrow f_{n}^{\Sigma}\left(z_{i}^{\prime}, \mathbf{k}\right) \\
& G_{n}\left(z_{i}^{\prime}, \mathbf{k} \text {-grid }\right) \rightarrow f_{n}^{G}\left(z_{i}^{\prime}, \mathbf{k}\right) \tag{B2}
\end{array}
$$
where $f_{n}^{\Sigma}$ and $f_{n}^{G}$ are the interpolating functions of $\Sigma$ and $G$, respectively. The interpolation for the $\left\{z_{i}\right\}$ frequencies can then be obtained as
$$
\begin{equation*}
f_{n}^{\Sigma / G}\left(z_{i}, \mathbf{k}\right)=f_{n}^{\Sigma / G}\left[z_{i}^{\prime}+f_{n}^{\epsilon}(\mathbf{k}), \mathbf{k}\right] \tag{B3}
\end{equation*}
$$
[1] F. Aryasetiawan and O. Gunnarsson, The GW method, Rep. Prog. Phys. 61, 237 (1998).
[2] L. Hedin, On correlation effects in electron spectroscopies and the gw approximation, Journal of Physics: Condensed Matter 11, R489 (1999).
[3] G. Onida, L. Reining, and A. Rubio, Electronic excitations: density-functional versus many-body green'sfunction approaches, Rev. Mod. Phys. 74, 601 (2002).
[4] R. M. Martin, L. Reining, and D. M. Ceperley, Interacting Electrons (Cambridge University Press, Cambridge, 2016).
[5] L. Reining, The gw approximation: content, successes and limitations, WIREs Computational Molecular Science 8, e1344 (2018).
[6] N. Marzari, A. Ferretti, and C. Wolverton, Electronicstructure methods for materials design, Nature Materials 20, 736-749 (2021).
[7] A. Damascelli, Probing the electronic structure of complex systems by arpes, Physica Scripta 2004, 61 (2004).
[8] S. Hüfner, Photoelectron spectroscopy: principles and applications (Springer Science \& Business Media, 2013).
[9] G. Strinati, H. J. Mattausch, and W. Hanke, Dynamical correlation effects on the quasiparticle bloch states of a covalent crystal, Phys. Rev. Lett. 45, 290 (1980).
[10] G. Strinati, H. J. Mattausch, and W. Hanke, Dynamical aspects of correlation corrections in a covalent crystal, Phys. Rev. B 25, 2867 (1982).
[11] J. S. Dolado, V. M. Silkin, M. A. Cazalilla, A. Rubio, and P. M. Echenique, Lifetimes and mean-free paths of hot electrons in the alkali metals, Phys. Rev. B 64, 195128 (2001).
[12] A. Marini, R. Del Sole, A. Rubio, and G. Onida, Quasiparticle band-structure effects on the d hole lifetimes of copper within the gw approximation, Phys. Rev. B 66, 161104 (2002).
[13] A. S. Kheifets, V. A. Sashin, M. Vos, E. Weigold, and F. Aryasetiawan, Spectral properties of quasiparticles in silicon: A test of many-body theory, Phys. Rev. B 68, 233205 (2003).
[14] B. Arnaud, S. Lebègue, and M. Alouani, Excitonic and quasiparticle lifetime effects on silicon electron energy loss spectra from first principles, Phys. Rev. B 71, 035308 (2005).
[15] M. Cazzaniga, $g w$ and beyond approaches to quasiparticle properties in metals, Phys. Rev. B 86, 035120 (2012).
[16] J. S. Zhou, L. Reining, A. Nicolaou, A. Bendounan, K. Ruotsalainen, M. Vanzini, J. J. Kas, J. J. Rehr, M. Muntwiler, V. N. Strocov, F. Sirotti, and M. Gatti, Unraveling intrinsic correlation effects with angleresolved photoemission spectroscopy, Proceedings of the National Academy of Sciences 117, 28596 (2020).
[17] M. S. Hybertsen and S. G. Louie, First-principles theory of quasiparticles: Calculation of band gaps in semiconductors and insulators, Phys. Rev. Lett. 55, 1418 (1985).
which can be used to evaluate the final spectra on a much denser k-grid. We have used first-order splines as interpolating functions, which is sufficient to obtain smooth spectra for our calculations, although this approach can be applied with other interpolating functions as well.
[18] M. van Schilfgaarde, T. Kotani, and S. Faleev, Quasiparticle self-consistent GW theory, Phys. Rev. Lett. 96, 226402 (2006).
[19] F. Hüser, T. Olsen, and K. S. Thygesen, Quasiparticle gw calculations for solids, molecules, and two-dimensional materials, Phys. Rev. B 87, 235132 (2013).
[20] D. Golze, M. Dvorak, and P. Rinke, The GW Compendium: A Practical Guide to Theoretical Photoemission Spectroscopy, Front. Chem. 7, 377 (2019).
[21] B. Gumhalter, V. Kovač, F. Caruso, H. Lambert, and F. Giustino, On the combined use of gw approximation and cumulant expansion in the calculations of quasiparticle spectra: The paradigm of si valence bands, Phys. Rev. B 94, 035103 (2016).
[22] J. S. Zhou, M. Gatti, J. J. Kas, J. J. Rehr, and L. Reining, Cumulant green's function calculations of plasmon satellites in bulk sodium: Influence of screening and the crystal environment, Phys. Rev. B 97, 035137 (2018).
[23] J. P. Nery, P. B. Allen, G. Antonius, L. Reining, A. Miglio, and X . Gonze, Quasiparticles and phonon satellites in spectral functions of semiconductors and insulators: Cumulants applied to the full first-principles theory and the fröhlich polaron, Phys. Rev. B 97, 115145 (2018).
[24] A. Marini, G. Onida, and R. D. Sole, Quasiparticle electronic structure of copper in the gw approximation, Phys. Rev. Lett. 88, 016403 (2002).
[25] M. Shishkin and G. Kresse, Implementation and performance of the frequency-dependent $g w$ method within the paw framework, Phys. Rev. B 74, 035101 (2006).
[26] F. Liu, L. Lin, D. Vigil-Fowler, J. Lischner, A. F. Kemper, S. Sharifzadeh, F. H. da Jornada, J. Deslippe, C. Yang, J. B. Neaton, and S. G. Louie, Numerical integration for ab initio many-electron self energy calculations within the GW approximation, J. Comput. Phys. 286, 1 (2015).
[27] R. W. Godby, M. Schlüter, and L. J. Sham, Self-energy operators and exchange-correlation potentials in semiconductors, Phys. Rev. B 37, 10159 (1988).
[28] F. Aryasetiawan in, Strong coulomb correlations in electronic structure calculations (CRC Press., London, 2000) p. 96, 1st ed., https://doi.org/10.1201/ 9781482296877.
[29] T. Kotani, M. van Schilfgaarde, and S. V. Faleev, Quasiparticle self-consistent $g w$ method: A basis for the independent-particle approximation, Phys. Rev. B 76, 165106 (2007).
[30] R. Daling, W. van Haeringen, and B. Farid, Plasmon dispersion in silicon obtained by analytic continuation of the random-phase-approximation dielectric matrix, Phys. Rev. B 44, 2952 (1991).
[31] G. E. Engel, B. Farid, C. M. M. Nex, and N. H. March, Calculation of the gw self-energy in semiconducting crystals, Phys. Rev. B 44, 13356 (1991).
[32] I. Duchemin and X. Blase, Robust Analytic-Continuation Approach to Many-Body GW Calculations, J. Chem. Theory Comput. 16, 1742 (2020).
[33] M. S. Hybertsen and S. G. Louie, Electron correlation in semiconductors and insulators: Band gaps and quasiparticle energies, Phys. Rev. B 34, 5390 (1986).
[34] S. B. Zhang, D. Tománek, M. L. Cohen, S. G. Louie, and M. S. Hybertsen, Evaluation of quasiparticle energies for semiconductors without inversion symmetry, Phys. Rev. B 40, 3162 (1989).
[35] R. W. Godby and R. J. Needs, Metal-insulator transition in kohn-sham theory and quasiparticle theory, Phys. Rev. Lett. 62, 1169 (1989).
[36] W. von der Linden and P. Horsch, Precise quasiparticle energies and hartree-fock bands of semiconductors and insulators, Phys. Rev. B 37, 8351 (1988).
[37] G. E. Engel and B. Farid, Generalized plasmon-pole model and plasmon band structures of crystals, Phys. Rev. B 47, 15931 (1993).
[38] B. Farid, G. E. Engel, R. Daling, and W. van Haeringen, Plasmon excitations in crystals, Phys. Rev. B 44, 13349 (1991).
[39] K.-H. Lee and K. J. Chang, First-principles study of the optical properties and the dielectric response of al, Phys. Rev. B 49, 2362 (1994).
[40] J. A. Soininen, J. J. Rehr, and E. L. Shirley, Electron self-energy calculation using a general multi-pole approximation, J. Phys.: Condens. Matter 15, 2573 (2003).
[41] D. A. Leon, C. Cardoso, T. Chiarotti, D. Varsano, E. Molinari, and A. Ferretti, Frequency dependence in $g w$ made simple using a multipole approximation, Phys. Rev. B 104, 115157 (2021).
$[42]$ D. A. Leon, A. Ferretti, D. Varsano, E. Molinari, and C. Cardoso, Efficient full frequency gw for metals using a multipole approach for the dielectric screening, Phys. Rev. B 107, 155130 (2023).
[43] A. Guandalini, D. A. Leon, P. D'Amico, C. Cardoso, A. Ferretti, M. Rontani, and D. Varsano, Efficient GW calculations via interpolation of the screened interaction in momentum and frequency space: The case of graphene, Phys. Rev. B 109, 075120 (2024).
[44] H. Raether, Excitation of Plasmons and Interband Transitions by Electrons, 1st ed., Springer Tracts in Modern Physics 88, Vol. 88 (Springer, Berlin, Heidelberg, 1980).
[45] G. Giuliani and G. Vignale, Quantum Theory of the Electron Liquid (Cambridge University Press, 2005).
[46] D. Pines, Theory of quantum liquids (CRC Press, 2018).
[47] J. W. Allen and J. C. Mikkelsen, Optical properties of crsb, mnsb, nisb, and nias, Phys. Rev. B 15, 2952 (1977).
[48] D. Y. Smith and B. Segall, Intraband and interband processes in the infrared spectrum of metallic aluminum, Phys. Rev. B 34, 5191 (1986).
[49] K.-H. Lee and K. J. Chang, Analytic continuation of the dynamic response function using an N -point Padé approximant, Phys. Rev. B 54, R8285 (1996).
[50] Y.-G. Jin and K. J. Chang, Dynamic response function and energy-loss spectrum for Li using an N-point Padé approximant, Phys. Rev. B 59, R8285 (1999).
[51] J. J. Kas, A. P. Sorini, M. P. Prange, L. W. Cambell, J. A. Soininen, and J. J. Rehr, Many-pole model of inelastic losses in x-ray absorption spectra, Phys. Rev. B 76, 195116 (2007).
[52] J. J. Kas, J. Vinson, N. Trcera, D. Cabaret, E. L. Shirley, and J. J. Rehr, Many-Pole Model of Inelastic Losses Ap-
plied to Calculations of XANES, J. Phys. Conf. Ser. 190, 012009 (2009).
[53] Y. Liang and L. Yang, Carrier plasmon induced nonlinear band gap renormalization in two-dimensional semiconductors, Phys. Rev. Lett. 114, 063001 (2015).
[54] A. Champagne, J. B. Haber, S. Pokawanvit, D. Y. Qiu, S. Biswas, H. A. Atwater, F. H. da Jornada, and J. B. Neaton, Quasiparticle and optical properties of carrierdoped monolayer mote2 from first principles, Nano Letters 23, 4274 (2023), pMID: 37159934.
[55] J.-M. Lihm and C.-H. Park, Plasmon-phonon hybridization in doped semiconductors from first principles, Phys. Rev. Lett. 133, 116402 (2024).
[56] M. M. Riegera, L. Steinbeck, I. D. White, H. N. Rojas, and R. W. Godby, The gw space-time method for the selfenergy of large systems, Comput. Phys. Commun. 117, 211 (1999).
[57] J. A. Soininen, J. J. Rehr, and E. L. Shirley, Multipole representation of the dielectric matrix, Phys. Scripta 2005, 243 (2005).
[58] M. J. van Setten, F. Caruso, S. Sharifzadeh, X. Ren, M. Scheffler, F. Liu, J. Lischner, L. Lin, J. R. Deslippe, S. G. L. C. Yang, F. Weigend, J. B. Neaton, F. Evers, and P. Rinke, GW100: Benchmarking G0W0 for Molecular Systems, J. Chem. Theory Comput. 11, 5665 (2015).
[59] T. Chiarotti, N. Marzari, and A. Ferretti, Unified green's function approach for spectral and thermodynamic properties from algorithmic inversion of dynamical potentials, Phys. Rev. Res. 4, 013242 (2022).
[60] T. Chiarotti, A. Ferretti, and N. Marzari, Energies and spectra of solids from the algorithmic inversion of dynamical hubbard functionals, Phys. Rev. Res. 6, L032023 (2024).
[61] A. Ferretti, T. Chiarotti, and N. Marzari, Green's function embedding using sum-over-pole representations, Phys. Rev. B 110, 045149 (2024).
[62] M. Quinzi, T. Chiarotti, M. Gibertini, and A. Ferretti, Broken symmetry solutions in one-dimensional lattice models via many-body perturbation theory, Phys. Rev. B 111, 125148 (2025).
[63] S. Ismail-Beigi, Correlation energy functional within the GW-RPA: Exact forms, approximate forms, and challenges, Phys. Rev. B 81, 195126 (2010).
[64] Z. Guo and J. Liu, Beyond-mean-field studies of wigner crystal transitions in various interacting two-dimensional systems, https://arxiv.org/abs/2409.14658v2 (2024).
[65] G. Lehmann and M. Taut, On the numerical calculation of the density of states and related properties, Phys. Status Solidi (b) 54, 469 (1972).
[66] B. Farid, Dynamical correlation functions expressed in terms of many-particle ground-state wavefunction; the dynamical self-energy operator, Philosophical Magazine B 82, 1413 (2002).
[67] A. Rasmussen, T. Deilmann, and K. S. Thygesen, Towards fully automatized GW band structure calculations: What we can learn from 60.000 self-energy evaluations, NPJ Comput. Mater. 7 (2021).
[68] A. Georges, G. Kotliar, W. Krauth, and M. J. Rozenberg, Dynamical mean-field theory of strongly correlated fermion systems and the limit of infinite dimensions, Rev. Mod. Phys. 68, 13 (1996).
[69] U. von Barth and B. Holm, Self-consistent $g w_{0}$ results for the electron gas: Fixed screened potential $w_{0}$ within the random-phase approximation, Phys. Rev. B 54, 8411
(1996).
[70] A. Marini, C. Hogan, M. Grüning, and D. Varsano, yambo: An ab initio tool for excited state calculations, Comput. Phys. Commun. 180, 1392 (2009).
[71] D. Sangalli, A. Ferretti, H. Miranda, C. Attaccalite, I. Marri, E. Cannuccia, P. Melo, M. Marsili, F. Paleari, A. Marrazzo, G. Prandini, P. Bonfà, M. O. Atambo, F. Affinito, M. Palummo, A. Molina-Sánchez, C. Hogan, M. Grüning, D. Varsano, and A. Marini, Many-body perturbation theory calculations using the yambo code, J. Phys.: Condens. Matter 31, 325902 (2019).
[72] J. J. Mortensen, A. H. Larsen, M. Kuisma, A. V. Ivanov, A. Taghizadeh, A. Peterson, A. Haldar, A. O. Dohn, C. Schäfer, E. Ö. Jónsson, E. D. Hermes, F. A. Nilsson, G. Kastlunger, G. Levi, H. Jónsson, H. Häkkinen, J. Fojt, J. Kangsabanik, J. Sødequist, J. Lehtomäki, J. Heske, J. Enkovaara, K. T. Winther, M. Dulak, M. M. Melander, M. Ovesen, M. Louhivuori, M. Walter, M. Gjerding, O. Lopez-Acevedo, P. Erhart, R. Warmbier, R. Würdemann, S. Kaappa, S. Latini, T. M. Boland, T. Bligaard, T. Skovhus, T. Susi, T. Maxson, T. Rossi, X. Chen, Y. L. A. Schmerwitz, J. Schiøtz, T. Olsen, K. W. Jacobsen, and K. S. Thygesen, GPAW: An open Python package for electronic structure calculations, The Journal of Chemical Physics 160, 092503 (2024).
[73] See Supplemental Materials for a detailed description.
[74] J. Gesenhues, D. Nabok, M. Rohlfing, and C. Draxl, Analytical representation of dynamical quantities in $g w$ from a matrix resolvent, Phys. Rev. B 96, 245124 (2017).
[75] S. Güttel and F. Tisseur, The nonlinear eigenvalue problem, Acta Numerica 26, 1-94 (2017).
[76] P. Giannozzi, S. Baroni, N. Bonini, M. Calandra, R. Car, C. Cavazzoni, D. Ceresoli, G. L. Chiarotti, M. Cococcioni, I. Dabo, A. D. Corso, S. de Gironcoli, S. Fabris, G. Fratesi, R. Gebauer, U. Gerstmann, C. Gougoussis, A. Kokalj, M. Lazzeri, L. Martin-Samos, N. Marzari, F. Mauri, R. Mazzarello, S. Paolini, A. Pasquarello, L. Paulatto, C. Sbraccia, S. Scandolo, G. Sclauzero, A. P. Seitsonen, A. Smogunov, P. Umari, and R. M. Wentzcovitch, QUANTUM ESPRESSO: a modular and opensource software project for quantum simulations of materials, J. Phys.: Condens. Matter 21, 395502 (2009).
[77] P. Giannozzi, O. Andreussi, T. Brumme, O. Bunau, M. B. Nardelli, M. Calandra, R. Car, C. Cavazzoni, D. Ceresoli, M. Cococcioni, N. Colonna, I. Carnimeo,
A. D. Corso, S. de Gironcoli, P. Delugas, R. A. DiStasio, A. Ferretti, A. Floris, G. Fratesi, G. Fugallo, R. Gebauer, U. Gerstmann, F. Giustino, T. Gorni, J. Jia, M. Kawamura, H.-Y. Ko, A. Kokalj, E. Küçükbenli, M. Lazzeri, M. Marsili, N. Marzari, F. Mauri, N. L. Nguyen, H.V. Nguyen, A. O. de-la Roza, L. Paulatto, S. Poncé, D. Rocca, R. Sabatini, B. Santra, M. Schlipf, A. P. Seitsonen, A. Smogunov, I. Timrov, T. Thonhauser, P. Umari, N. Vast, X. Wu, and S. Baroni, Advanced capabilities for materials modelling with Quantum ESPRESSO, J. Phys.: Condens. Matter 29, 465901 (2017).
[78] J. P. Perdew, K. Burke, and M. Ernzerhof, Generalized gradient approximation made simple, Phys. Rev. Lett. 77, 3865 (1996).
[79] D. R. Hamann, Optimized norm-conserving vanderbilt pseudopotentials, Phys. Rev. B 88, 085117 (2013).
[80] A. Guandalini, P. D'Amico, A. Ferretti, and D. Varsano, Efficient GW calculations in two dimensional materials through a stochastic integration of the screened potential, npj Computational Materials 9, 44 (2023).
[81] M. Guzzo, G. Lani, F. Sottile, P. Romaniello, M. Gatti, J. J. Kas, J. J. Rehr, M. G. Silly, F. Sirotti, and L. Reining, Valence electron photoemission spectrum of semiconductors: Ab initio description of multiple satellites, Phys. Rev. Lett. 107, 166401 (2011).
[82] F. Caruso and F. Giustino, The GW plus cumulant method and plasmonic polarons: application to the homogeneous electron gas*, Eur. Phys. J. B 89, https://doi.org/10.1140/epjb/e2016-70028-4 (2016).
[83] A. Gerlach, K. Berge, A. Goldmann, I. Campillo, A. Rubio, J. M. Pitarke, and P. M. Echenique, Lifetime of d holes at cu surfaces: Theory and experiment, Phys. Rev. B 64, 085423 (2001).
[84] A. Tamai, W. Meevasana, P. D. C. King, C. W. Nicholson, A. de la Torre, E. Rozbicki, and F. Baumberger, Spin-orbit splitting of the shockley surface state on Cu(111), Phys. Rev. B 87, 075113 (2013).
[85] D. A. Leon, MPA-Sigma_data (2025).
[86] D. A. Leon, C. Elgvin, P. D. Nguyen, O. Prytz, F. S. Hage, and K. Berland, Unraveling many-body effects in ZnO : Combined study using momentum-resolved electron energy-loss spectroscopy and first-principles calculations, Phys. Rev. B 109, 115153 (2024).

\title{
Spectral properties from an efficient analytical representation of the $G W$ self-energy within a multipole approximation: Supplemental materials
}

\author{
Dario A. Leon, ${ }^{1, *}$ Kristian Berland, ${ }^{1}$ and Claudia Cardoso ${ }^{2}$ \\ ${ }^{1}$ Department of Mechanical Engineering and Technology Management, Norwegian University of Life Sciences, NO-1432 Ås, Norway \\ ${ }^{2}$ S3 Centre, Istituto Nanoscienze, CNR, 41125 Modena, Italy
}

\section*{I. MPA- $\Sigma$ INTERPOLATION}

The procedure we use to obtain the MPA- $\Sigma$ representation is analogous to the one for MPA- $W$ (see Appendix A of Ref. [1]). We need to solve the following non linear system of $2 n_{\Sigma}$ equations and variables:
$$
\Sigma_{c}^{\mathrm{MPA}}\left(z_{i}\right) \equiv \sum_{p}^{n_{\Sigma}} \frac{S_{p}}{z_{i}-\xi_{p}}=\Sigma_{c}\left(z_{i}\right), i=1, \ldots, 2 n_{\Sigma}(\mathrm{S} 1)
$$
where $n_{\Sigma}$ is the number of poles and $\left\{z_{i}, \Sigma_{c}\left(z_{i}\right)\right\}$ correspond to the numerical data according to the given sampling (see Sec. II). Notice that finding the residues $S_{p}$ if the poles $\xi_{p}$ are known is a simple linear least square problem:
$$
\begin{equation*}
\min _{\mathbf{S}_{\mathbf{n}_{\Sigma}}}\left\|\mathbf{M}_{2 \mathbf{n}_{\Sigma} \mathbf{n}_{\Sigma}} \cdot \mathbf{S}_{\mathbf{n}_{\Sigma}}-\boldsymbol{\Sigma}_{2 \mathbf{n}_{\Sigma}}\right\| \tag{S2}
\end{equation*}
$$
where we have defined the following vectors and matrix:
$$
\begin{align*}
\mathbf{S}_{\mathbf{n}_{\boldsymbol{\Sigma}}} & \equiv\left[\begin{array}{llll}
S_{1} & S_{2} & \ldots & S_{n_{\Sigma}}
\end{array}\right]  \tag{S3}\\
\boldsymbol{\Sigma}_{\mathbf{2 n}_{\boldsymbol{\Sigma}}} & \equiv\left[\begin{array}{llll}
\Sigma_{c}\left(z_{1}\right) & \Sigma_{c}\left(z_{2}\right) & \ldots & \Sigma_{c}\left(z_{2 n_{\Sigma}}\right)
\end{array}\right]  \tag{S4}\\
\mathbf{M}_{\mathbf{2 n}_{\boldsymbol{\Sigma}} \mathbf{n}_{\boldsymbol{\Sigma}}} & \equiv\left[\begin{array}{cccc}
\frac{1}{z_{1}-\xi_{1}} & \frac{1}{z_{1}-\xi_{2}} & \ldots & \frac{1}{z_{1}-\xi_{n_{\Sigma}}} \\
\frac{1}{z_{2}-\xi_{1}} & \frac{1}{z_{2}-\xi_{2}} & \cdots & \frac{1}{z_{2}-\xi_{n_{\Sigma}}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{1}{z_{2 n_{\Sigma}}-\xi_{1}} & \frac{1}{z_{2 n_{\Sigma}}-\xi_{2}} & \cdots & \frac{1}{z_{2 n_{\Sigma}}-\xi_{n_{\Sigma}}}
\end{array}\right] \tag{S5}
\end{align*}
$$

To find the poles, we start by writing the MPA selfenergy in its Padé form:
$$
\begin{equation*}
\sum_{p}^{n_{\Sigma}} \frac{S_{p}}{z-\xi_{p}}=\frac{A_{n_{\Sigma}-1}(z)}{B_{n_{\Sigma}}(z)} \equiv \frac{\mathbf{a}_{\mathbf{n}_{\Sigma}} \cdot \mathbf{z}_{\mathbf{n}_{\Sigma}}}{\mathbf{b}_{\mathbf{n}_{\Sigma}} \cdot \mathbf{z}_{\mathbf{n}_{\Sigma}}+z^{n_{\Sigma}}} \tag{S6}
\end{equation*}
$$
where we have defined the following vectors:
$$
\begin{align*}
\mathbf{a}_{\mathbf{n}_{\Sigma}} & \equiv\left[\begin{array}{llll}
a_{0} & a_{1} & \ldots & a_{n_{\Sigma}-1}
\end{array}\right] \\
\mathbf{b}_{\mathbf{n}_{\Sigma}} & \equiv\left[\begin{array}{llll}
b_{0} & b_{1} & \ldots & b_{n_{\Sigma}-1}
\end{array}\right]  \tag{S7}\\
\mathbf{z}_{\mathbf{n}_{\Sigma}}(z) & \equiv\left[\begin{array}{llll}
1 & z & \ldots & z^{n_{\Sigma}-1}
\end{array}\right] .
\end{align*}
$$

Notice that finding the poles means to find $\mathbf{b}_{\mathbf{n}_{\boldsymbol{\Sigma}}}$ and factorize the $B_{n_{\Sigma}}(z)$ polynomial, which can be done by di-

\footnotetext{
* dario.alejandro.leon.valido@nmbu.no
}
agonalizing its corresponding companion matrix:
$$
\mathbf{C}_{\mathbf{n}_{\boldsymbol{\Sigma} \mathbf{n}_{\boldsymbol{\Sigma}}}}=\left[\begin{array}{ccccc}
0 & 0 & \ldots & 0 & -b_{0}  \tag{S8}\\
1 & 0 & \ldots & 0 & -b_{1} \\
0 & 1 & \ldots & 0 & -b_{2} \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & \ldots & 1 & -b_{n_{\Sigma}-1}
\end{array}\right]
$$

To compute $\mathbf{b}_{\mathbf{n}_{\boldsymbol{\Sigma}}}$, we can then take one of the two routes in the next sections.

\section*{A. Linear solver}

With the definitions in Eq. (S6) we can transform our non linear problem into an equivalent linear one:
$$
\begin{equation*}
\Sigma_{c}\left(z_{i}\right) z_{i}^{n_{\Sigma}}+\Sigma_{c}\left(z_{i}\right) \mathbf{z}_{\mathbf{n}_{\boldsymbol{\Sigma}}}\left(z_{i}\right) \cdot \mathbf{b}_{\mathbf{n}_{\boldsymbol{\Sigma}}}=\mathbf{z}_{\mathbf{n}_{\boldsymbol{\Sigma}}}\left(z_{i}\right) \cdot \mathbf{a}_{\mathbf{n}_{\boldsymbol{\Sigma}}} \tag{S9}
\end{equation*}
$$

We can split the sampled points into two sets with the same number of elements $n_{\Sigma}$ :
$$
\begin{align*}
& \text { set } 1: i=1, \ldots, n_{\Sigma}, \\
& \text { set } 2: i=n_{\Sigma}+1, \ldots, 2 n_{\Sigma}, \tag{S10}
\end{align*}
$$
and define the following vectors and matrices with the first set of Eq. (S10):
$$
\begin{align*}
\mathbf{v}_{\mathbf{n}_{\mathbf{\Sigma}}}^{\mathbf{1}} & =\left[\begin{array}{llll}
\Sigma_{c}\left(z_{1}\right) z_{1}^{n_{\Sigma}} & \Sigma_{c}\left(z_{2}\right) z_{2}^{n_{\Sigma}} & \ldots & \Sigma_{c}\left(z_{n_{\Sigma}}\right) z_{n_{\Sigma}}^{n_{\Sigma}}
\end{array}\right]  \tag{S11}\\
\mathbf{z}_{\mathbf{n}_{\boldsymbol{\Sigma}} \mathbf{n}_{\boldsymbol{\Sigma}}}^{\mathbf{1}} & =\left[\begin{array}{cccc}
1 & z_{1} & \ldots & z_{1}^{n_{\Sigma}-1} \\
1 & z_{2} & \ldots & z_{2}^{n_{\Sigma}-1} \\
\vdots & \vdots & \ddots & \vdots \\
1 & z_{n_{\Sigma}} & \ldots & z_{n_{\Sigma}}^{n_{\Sigma}-1}
\end{array}\right]  \tag{S12}\\
\mathbf{M}_{\mathbf{n}_{\boldsymbol{\Sigma}} \mathbf{n}_{\boldsymbol{\Sigma}}}^{\mathbf{1}} & =\left[\begin{array}{cccc}
\Sigma_{c}\left(z_{1}\right) & \Sigma_{c}\left(z_{1}\right) z_{1} & \ldots & \Sigma_{c}\left(z_{1}\right) z_{1}^{n_{\Sigma}-1} \\
\Sigma_{c}\left(z_{2}\right) & \Sigma_{c}\left(z_{2}\right) z_{2} & \ldots & \Sigma_{c}\left(z_{2}\right) z_{2}^{n_{\Sigma}-1} \\
\vdots & \vdots & \ddots & \vdots \\
\Sigma_{c}\left(z_{n_{\Sigma}}\right) & \Sigma_{c}\left(z_{n_{\Sigma}}\right) z_{n_{\Sigma}} & \ldots & \Sigma_{c}\left(z_{n_{\Sigma}}\right) z_{n_{\Sigma}}^{n_{\Sigma}-1}
\end{array}\right] \tag{S13}
\end{align*}
$$
likewise, we can define $\mathbf{v}_{\mathbf{n}_{\Sigma}}^{\mathbf{2}}, \mathbf{z}_{\mathbf{n}_{\Sigma} \mathbf{n}_{\Sigma}}^{\mathbf{2}}$ and $\mathbf{M}_{\mathbf{n}_{\Sigma} \mathbf{n}_{\Sigma}}^{\mathbf{2}}$ with the second set. The system of equations is then simplified as
$$
\begin{gather*}
v_{n_{\Sigma}}^{1}+M_{n_{\Sigma} n_{\Sigma}}^{1} \cdot b_{n_{\Sigma}}=z_{n_{\Sigma} n_{\Sigma}}^{1} \cdot a_{n_{\Sigma}}  \tag{S14}\\
v_{n_{\Sigma}}^{2}+M_{n_{\Sigma} n_{\Sigma}}^{2} \cdot b_{n_{\Sigma}}=z_{n_{\Sigma} n_{\Sigma}}^{2} \cdot a_{n_{\Sigma}} .
\end{gather*}
$$

By combining the two equations in Eq. (S14) we obtain a linear system for $\mathbf{b}_{\mathbf{n}_{\boldsymbol{\Sigma}}}$ :
$$
\begin{equation*}
\mathbf{M}_{\mathbf{n}_{\Sigma} \mathbf{n}_{\Sigma}} \cdot \mathbf{b}_{\mathbf{n}_{\Sigma}}=\mathbf{v}_{\mathbf{n}_{\Sigma}}, \tag{S15}
\end{equation*}
$$
where $\mathbf{M}_{\mathbf{n}_{\boldsymbol{\Sigma}} \mathbf{n}_{\boldsymbol{\Sigma}}}$ and $\mathbf{v}_{\mathbf{n}_{\boldsymbol{\Sigma}}}$ are defined as
$$
\begin{array}{rr}
\mathbf{M}_{\mathbf{n}_{\Sigma} \mathbf{n}_{\Sigma}} \equiv & \mathbf{z}_{\mathbf{n}_{\Sigma} \mathbf{n}_{\Sigma}}^{\mathbf{2}} \cdot\left(\mathbf{z}_{\mathbf{n}_{\Sigma} \mathbf{n}_{\Sigma}}^{\mathbf{1}}\right)^{-1} \cdot \mathbf{M}_{\mathbf{n}_{\Sigma} \mathbf{n}_{\Sigma}}^{\mathbf{1}}-\mathbf{M}_{\mathbf{n}_{\Sigma} \mathbf{n}_{\Sigma}}^{\mathbf{2}} \\
\mathbf{v}_{\mathbf{n}_{\Sigma}} \equiv & -\mathbf{z}_{\mathbf{n}_{\Sigma} \mathbf{n}_{\Sigma}}^{\mathbf{2}} \cdot\left(\mathbf{z}_{\mathbf{n}_{\Sigma} \mathbf{n}_{\Sigma}}^{\mathbf{1}}\right)^{-1} \cdot \mathbf{v}_{\mathbf{n}_{\Sigma}}^{\mathbf{1}}+\mathbf{v}_{\mathbf{n}_{\Sigma}}^{\mathbf{2}} \tag{S16}
\end{array}
$$

\section*{B. Padé/Thiele solver}

It is also possible to write the Padé representation as a $2 n_{\Sigma}$-point continued fraction of reciprocal differences and use Thiele's interpolation formula [2]:
$$
\begin{align*}
& \frac{A_{n_{\Sigma}-1}(z) / b_{0}}{B_{n_{\Sigma}}(z) / b_{0}}= \\
& \quad \frac{c_{1}}{1+} \frac{c_{2}\left(z-z_{1}\right)}{1+} \cdots \frac{c_{2 n_{\Sigma}}\left(z-z_{2 n_{\Sigma}-1}\right)}{1+\left(z-z_{2 n_{\Sigma}-1}\right) g_{2 n_{\Sigma}}(z)} \tag{S17}
\end{align*}
$$
where the coefficients $c_{i}$ and functions $g_{i}(z)$ are given by the following recursion relations:
$$
\begin{align*}
c_{i} & =g_{i}\left(z_{i}\right) \\
g_{i}(z) & =\left\{\begin{array}{l}
\Sigma_{c}\left(z_{i}\right), \quad i=1 \\
\frac{g_{i-1}\left(z_{i-1}\right)-g_{i-1}(z)}{\left(z-z_{i-1}\right) g_{i-1}(z)}, s \geq 2
\end{array}\right. \tag{S18}
\end{align*}
$$
where index $i=1, \ldots, 2 n_{\Sigma}$ corresponds to both the iteration step and the index of the sampled point.

Since we are primarily interested in the polynomial in the denominator of Eq. (S17), we can define the following vectors:
$$
\begin{align*}
\mathbf{d}_{\mathbf{n}_{\boldsymbol{\Sigma}}+\mathbf{1}} & =\left[\begin{array}{lllll}
1 & \frac{b_{1}}{b_{0}} & \ldots & \frac{b_{n_{\Sigma}-1}}{b_{0}} & \frac{1}{b_{0}}
\end{array}\right]  \tag{S19}\\
\mathbf{c}_{\mathbf{2 n}_{\boldsymbol{\Sigma}}} & =\left[\begin{array}{llll}
g\left(z_{1}\right) & g\left(z_{2}\right) & \ldots & g\left(z_{2 n_{\Sigma}}\right)
\end{array}\right]
\end{align*}
$$
and recast the recursivity in vectorial form [1]:
$$
\begin{align*}
\mathbf{d}_{\mathbf{n}_{\mathbf{\Sigma}}+\mathbf{1}}^{\mathbf{s}} & = \begin{cases}d_{j}^{s}=\delta_{j 1}, & s=0,1 \\
d_{j}^{s}=d_{j}^{s-1}+c_{s} z_{s} d_{j+1}^{s-2} & -c_{s} z_{s-1} d_{j}^{s-2}, s \geq 2\end{cases} \\
\mathbf{c}_{\mathbf{2 n}_{\boldsymbol{\Sigma}}}^{\mathbf{s}} & = \begin{cases}c_{i}^{s}=\Sigma_{c}\left(z_{i}\right), & s=1 \\
c_{i}^{s}=\frac{c_{i-1}^{s-1}-c_{i}^{s-1}}{\left(z_{i}-z_{i-1}\right) c_{i}^{s-1}}, & s \geq 2,\end{cases} \tag{S20}
\end{align*}
$$
where $j=1, \ldots, n_{\Sigma}+1, s=0, \ldots, 2 n_{\Sigma}$ is the iteration step, and $\delta_{j 1}$ is a Kronecker delta.

Once $\mathbf{d}_{\mathbf{n}_{\boldsymbol{\Sigma}}+\mathbf{1}}$ has been computed in the last iteration, we can retrieve the vector $\mathbf{b}_{\mathbf{n}_{\boldsymbol{\Sigma}}}$ as
$$
\mathbf{b}_{\mathbf{n}_{\Sigma}}=\left[\begin{array}{llll}
\frac{1}{d_{n_{\Sigma}+1}} & \frac{d_{1}}{d_{n_{\Sigma}+1}} & \ldots & \frac{d_{n_{\Sigma}}}{d_{n_{\Sigma}+1}} \tag{S21}
\end{array}\right]
$$

\section*{II. MPA- $\Sigma$ SAMPLING}

As mentioned in the main text, MPA- $\Sigma$ uses a sampling in the complex frequency plane $z^{\prime} \equiv z-\epsilon_{n \mathbf{k}}^{\mathrm{KS}}$, that depends on the number of poles $n_{\Sigma}$. The set of frequency points, $\left\{z_{i}^{\prime}: i=1, . ., 2 n_{\Sigma}\right\}$, is divided into two subsets corresponding to the first and third quadrants of the complex frequency plane:
$$
\left\{z_{i}^{\prime}\right\}: \begin{cases}z_{n}^{+}=\omega_{n}+i \varpi ; & n=1, . ., n_{\Sigma}^{+}  \tag{S22}\\ z_{n}^{-}=-\omega_{n}-i \varpi ; & n=1, . ., n_{\Sigma}^{-}\end{cases}
$$
where the imaginary part is typically set to $\varpi=0.1 \mathrm{eV}$, $n_{\Sigma}^{+}+n_{\Sigma}^{-}=2 n_{\Sigma}$, and the frequency points are distributed inhomogeneously along the real axis (see also Eq. (10) of Ref. [3]) as
$$
\left\{\omega_{n}\right\}_{\alpha}:\left\{\begin{array}{r}
(0), n_{\Sigma}^{+}=1  \tag{S23}\\
(0,1) \times \omega_{m}, n_{\Sigma}^{+}=2 \\
\left(0, \frac{1}{2}, 1\right)^{\alpha} \times \omega_{m}, n_{\Sigma}^{+}=3 \\
\left(0, \frac{1}{4}, \frac{1}{2}, 1\right)^{\alpha} \times \omega_{m}, n_{\Sigma}^{+}=4 \\
\left(0, \frac{1}{8}, \frac{1}{4}, \frac{1}{2}, 1\right)^{\alpha} \times \omega_{m}, n_{\Sigma}^{+}=5 \\
\left(0, \frac{1}{8}, \frac{1}{4}, \frac{1}{2}, \frac{3}{4}, 1\right)^{\alpha} \times \omega_{m}, n_{\Sigma}^{+}=6 \\
\left(0, \frac{1}{8}, \frac{1}{4}, \frac{3}{8}, \frac{1}{2}, \frac{3}{4}, 1\right)^{\alpha} \times \omega_{m}, n_{\Sigma}^{+}=7 \\
\cdots
\end{array}\right.
$$

In Eq. (S23), the maximum frequency $\omega_{m}$ defines the desired sampling interval. The exponent $\alpha$ is usually set to $\alpha=1$ or $\alpha=2$ respectively corresponding to a linear or a quadratic semi-homogeneous partition in powers of 2 [3]. In the main manuscript, it is common practice to set the sampling with a minimum of asymmetry, using $n_{\Sigma}^{-}=n_{\Sigma}^{+}+1\left(n_{\Sigma}^{+}=n_{\Sigma}^{-}+1\right)$ for valence (conduction) states.

In Fig. S1 we show the convergence of $\Sigma$ and the interacting $G$ corresponding to the top valence state of Si , with respect to the total number of poles $n_{\Sigma}$. For the MPA- $\Sigma$ sampling, we set $n_{\Sigma}^{-}=n_{\Sigma}^{+}+1$ with a linear partition on both the positive and negative sides.

\section*{III. INSIGHTS INTO SPECTRAL BAND STRUCTURES}

Here we provide spectral band structures of $\mathrm{Na}, \mathrm{Si}$, $\mathrm{MoS}_{2}$, and Cu , computed with a full-frequency evaluation of $\Sigma$ and $G$, to benchmark the analogous ones in Fig. 6 of the main manuscript that correspond to MPA$\Sigma$. We show the details on how they are built, by isolating the bands of Na as an example. We also make

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_17_755328709aa12b82ff93g-20.jpg?height=465&width=1760&top_left_y=159&top_left_x=187}
\captionsetup{labelformat=empty}
\caption{FIG. S1. Re $\Sigma(\mathrm{a}), \operatorname{Im} \Sigma(\mathrm{b})$, and $\operatorname{Im} G$ (c) functions corresponding to the top valence state of Si computed with MPA- $\Sigma$ with a variable number of poles, $n_{\Sigma}$, compared to the full-frequency approach (FF).}
\end{figure}
use of the MPA- $G$ representation, to separate the spectral contributions of the QP pole and the satellites from the total MPA- $G$ spectral function computed in the main manuscript.

As mentioned in the main manuscript, the $\Sigma$ and $G$ spectral functions, $A_{\Sigma}$ and $A_{G}$, are computed for a finite number of valence and conduction states, respecting the time ordering. For visualization purposes, we have divided the intensity of the valence (conduction) contribution to $A_{\Sigma}$ and $A_{G}$ by the corresponding number of valence (conduction) bands, to ensure a more uniform background intensity:
$$
\begin{align*}
A_{\Sigma}(\mathbf{k}, \omega) & \equiv \frac{1}{n_{v} \pi} \sum_{v}^{n_{v}} \operatorname{Im} \Sigma_{v \mathbf{k}}(\omega)+\frac{1}{n_{c} \pi} \sum_{c}^{n_{c}} \operatorname{Im} \Sigma_{c \mathbf{k}}(\omega)  \tag{S24}\\
A_{G}(\mathbf{k}, \omega) & \equiv \frac{1}{n_{v} \pi} \sum_{v}^{n_{v}} \operatorname{Im} G_{v \mathbf{k}}(\omega)+\frac{1}{n_{c} \pi} \sum_{c}^{n_{c}} \operatorname{Im} G_{c \mathbf{k}}(\omega) \tag{S25}
\end{align*}
$$
where $v$ and $c$ run over valence and conduction states respectively and $n_{v}$ and $n_{c}$ correspond to the total number of valence and conduction bands considered. The normalization with respect to $n_{v}$ and $n_{c}$ is not necessary if a
[1] D. A. Leon, C. Cardoso, T. Chiarotti, D. Varsano, E. Molinari, and A. Ferretti, Frequency dependence in $g w$ made simple using a multipole approximation, Phys. Rev. B 104, 115157 (2021).
[2] H. J. Vidberg and J. W. Serene, Solving the eliashberg equations by means of n-point padé approximants, J. Low
large enough number of bands is included in Eqs. (S24) and (S25).

Fig. S2 shows $1 / \pi \operatorname{Im} \Sigma_{n \mathbf{k}}$ (a-d) and $1 / \pi \operatorname{Im} G_{n \mathbf{k}}$ (e-h) of Na along the $\Gamma N$ k-path, where $n$ corresponds to the highest non-metallic valence band (v1), the metallic band that crosses the Fermi level close to $N(\mathrm{~m} 1)$, the two lowest conduction bands (c1-c2), and the combination of all these 4 bands (all). The last panels ( $\mathrm{d}, \mathrm{h}$ ) are analogous to panels (a, e) of Fig. 6 in the main manuscript. Panel (b) shows discernible intensities in both the positive and negative sides. The same happens for other states at smaller intensities, as a consequence of the time ordering. Therefore, the total spectral function $A_{\Sigma}(\mathbf{k}, \omega)$ (d) shows a non trivial superposition in the region corresponding to the conduction bands.

In Fig. S3 we provide full-frequency (FF) $\Sigma$ and $G$ spectral band structures of Si (a, d), monolayer $\mathrm{MoS}_{2}$ (b, e ), and $\mathrm{Cu}(\mathrm{c}, \mathrm{f})$, analogous to the MPA- $\Sigma$ and MPA- $G$ ones presented in Figs. 6 of the main manuscript.

In Fig. S4 we have separated the spectral contributions of the QP pole (a-d) and the satellites (e-h) from the total MPA- $G$ spectral function plotted in panels (e-h) of Fig. 6 of the main manuscript. By a simple inspection, we can see that the superposition of the QP pole and the satellites spectra gives the total spectral $G$ function.

Temp. Phys. 29, 179-192 (1977).
[3] D. A. Leon, A. Ferretti, D. Varsano, E. Molinari, and C. Cardoso, Efficient full frequency gw for metals using a multipole approach for the dielectric screening, Phys. Rev. B 107, 155130 (2023).

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_17_755328709aa12b82ff93g-21.jpg?height=825&width=1760&top_left_y=178&top_left_x=187}
\captionsetup{labelformat=empty}
\caption{FIG. S2. Spectral $\Sigma$ (a-d) and $G$ (e-h) bands of Na, for the first valence state (a, e), the metallic band (b, f), the first two valence bands ( $c, g$ ) and their combinations ( $d, h$ ).}
\end{figure}

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_17_755328709aa12b82ff93g-22.jpg?height=1717&width=1773&top_left_y=184&top_left_x=184}
\captionsetup{labelformat=empty}
\caption{FIG. S3. Spectral band structures $A_{\Sigma}(\mathbf{k}, \omega)(\mathrm{a}-\mathrm{c})$ and $A_{G}(\mathbf{k}, \omega)(\mathrm{d}-\mathrm{f})$ of $\mathrm{Si}(\mathrm{a}, \mathrm{d}), \mathrm{MoS}_{2}(\mathrm{~b}, \mathrm{e})$, and $\mathrm{Cu}(\mathrm{c}, \mathrm{f})$, computed with a FF $\Sigma$ evaluation.}
\end{figure}

\begin{figure}
\includegraphics[width=\textwidth]{https://cdn.mathpix.com/cropped/2025_10_17_755328709aa12b82ff93g-23.jpg?height=1464&width=1787&top_left_y=176&top_left_x=176}
\captionsetup{labelformat=empty}
\caption{FIG. S4. MPA-G spectral band structures of $\mathrm{Na}(\mathrm{a}, \mathrm{e}), \mathrm{Si}(\mathrm{b}, \mathrm{f}), \mathrm{MoS}_{2}(\mathrm{c}, \mathrm{g})$, and $\mathrm{Cu}(\mathrm{d}, \mathrm{h})$, analogous to panels (e-h) of Fig. 6 of the main manuscript, but corresponding to the QP pole only (a-d) and satellites (e-h).}
\end{figure}