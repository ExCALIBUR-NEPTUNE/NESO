**Tamain equations (System 2-6) simplified**

**Ed Threlfall**

The main point of this document is to get an initial version of the equations into a form easily implemented in e.g. *Nektar++*. The end goal is a set of equations with a clear demarcation of variables and constants.

Simplifying assumptions
=======================

-   Uniform B-field in the $\mathbf{\hat{z}}$-direction. This means the grad-B drift velocities vanish.
    
-   All derivatives only act in transverse plane $(x,y)$. It is therefore assumed that $\nabla_{\perp}$ and $\nabla$ are identical
    in all of these equations.

Comment on v-space effects: the equations seem to be fluid equations coupled to a diffusion equation for the neutrals - not clear to me where the velocity-space effects come from \...

Eq. 95 (continuity equation)
============================


$$
\partial_t  n_e + \nabla \cdot (n_e \mathbf{u}_e) = S_e^n-\frac{n_e}{\tau_{n_e}}
$$


Where


$$
\mathbf{u}_e=\mathbf{u}_{E \times B} +\mathbf{u}_{\nabla B_e}+\mathbf{u}_{diff}.
$$


Of the term $\nabla \cdot (n_e \mathbf{u}_e) \equiv \mathbf{u}_e \cdot \nabla n_e + n_e \nabla \cdot \mathbf{u}_e$ (identity trivial to prove by writing out 2D components), the first piece is just an advection which presumably can be handled by the *Nektar++* Advection object. The second piece gives the nonlinear term $-n_e \nabla \cdot (D_{\perp} \nabla n_e)$ (the grad B drift vanishes, and the divergence of the $E \times B$ drift vanishes, see below).

(One has


$$
\mathbf{u}_{E \times B} \propto \frac{\mathbf{B} \times \nabla \Phi}{B^2}
$$

the divergence of which which may be simplified with the aid of the
vector identity

$$
\nabla \cdot (\mathbf{F} \times \mathbf{G}) \equiv \mathbf{G} \cdot (\nabla \times \mathbf{F}) - \mathbf{F} \cdot (\nabla \times \mathbf{G});
$$


noting that $\mathbf{B}$ has zero curl, as does $\nabla \Phi$, one sees $\nabla \cdot \mathbf{u}_{E \times B}=0$. Also trivial to show by writing in Cartesians.)

In all, the first equation reduces to


$$
\partial_t n_e + \mathbf{u}_e \cdot \nabla n_e - n_e \nabla \cdot (D_{\perp} \nabla n_e ) = S_e^n - \frac{n_e}{\tau_{n_e}}.
$$


Of the terms here, I cannot find a definition of $D_{\perp}$ in the equations document; I presume $\tau_e$ can be taken as constant for a first attempt, and $S_e^n$ I later take as a constant time $n_e n_n$, plus another constant times $n_e^2$ i.e. constant ionization and recombination coefficients).

I suppose it needs clarifying whether the *Nektar++* advection solver solves $\partial_t f + \mathbf{u} \cdot \nabla f = 0$ or $\partial_t f + \nabla \cdot (\mathbf{u} f) = 0$; presumably, the former (the two are only equivalent if $\mathbf{u}$ is divergence-free).

Eq. 96 Vorticity equation
=========================

The equation for the vorticity $\nabla \cdot \mathbf{E}^+$ is, ignoring vanishing grad-B drifts,


$$
\partial_t \nabla \cdot \mathbf{E}^+ + \nabla \cdot ( \nabla \cdot ( \mathbf{u}_i \otimes \mathbf{E}^+ )) = \nabla \cdot \left ( n_e \mathbf{u}_{cx}\right )+
n_e \left ( \frac{1}{\tau_{n_e}} - \frac{1}{\tau_{n_i}} \right ) + \nabla \cdot (\nu \nabla_{\perp} (\nabla \cdot \mathbf{E}^+))
$$

One notes that $\mathbf{u}_{cx} = -\frac{\mu_{cx}}{1+\mu_{cx}^2} \frac{B e}{m_i n_e} \mathbf{E}^+$. Note also $n_e \mathbf{u}_{cx} = - \frac{\mu_{cx}}{1+\mu_{cx}^2} \frac{Be}{m_i} \mathbf{E}^+$ and so $\nabla \cdot (n_e \mathbf{u}_{cx})$ is given by $- \frac{\mu_cx}{1+\mu_{cx}^2} \frac{Be}{m_i} \nabla \cdot \mathbf{E}^+$, a simple damping term.

The second term in the original equation, which I take to mean the fully-contracted expression $\nabla_m \nabla_n u_i^m E^i{+n}$, or, expanded (e.g. by writing out indices)


$$
\nabla \cdot ( \nabla \cdot ( \mathbf{u}_i \otimes \mathbf{E}^+ )) \equiv \partial_n (u_i)_m \partial_m E^+_n + 2 (\nabla \cdot \mathbf{u}_i) (\nabla \cdot \mathbf{E}^+) + \mathbf{E}^+ \cdot \nabla (\nabla \cdot \mathbf{u}_i)
$$


There does not seem to be a nice vector calculus expression for thefirst term on the right-hand side. Written out in full, the intended meaning of that term is


$$
.\partial_n (u_i)_m \partial_m E^+_n \equiv \frac{\partial (u_i)_x}{\partial x} \frac{\partial E^+_x}{\partial x}+ \frac{\partial (u_i)_y}{\partial x} \frac{\partial E^+_x}{\partial y} + \frac{\partial (u_i)_x}{\partial y} \frac{\partial E^+_y}{\partial x}+\frac{\partial (u_i)_y}{\partial y} \frac{\partial E^+_y}{\partial y}.
$$


It may or may not help to use the expression $\nabla \cdot \mathbf{u}_i = - \nabla \cdot (D_{\perp} \nabla n_e) + \frac{\mu_{cx}}{1+\mu_{cx}^2} \frac{Be}{m_i} \left ( \frac{\nabla \cdot \mathbf{E}^+}{n_e} - \frac{\mathbf{E}^+ \cdot \nabla n_e}{n_e^2} \right )$.

It seems the electric field itself is needed to update the vorticity equation (one recalls that the electrostatic potential is needed for the vorticity equation in the Hasegawa-Wakatani system).

Eq. 97 (electron energy flux equation)
======================================

This is


$$
\partial_t \mathcal{E}_e+\nabla \cdot (\mathcal{E}_e \mathbf{u}_e+ p_e \mathbf{u}_e) = S_e^{\mathcal{E}}-\frac{\mathcal{E}_e}{\tau_{\mathcal{E}_e}}+Q_{ie}+\nabla \cdot(\kappa_{\perp,e} n_e \nabla_{\perp} k T_e)
$$


One simplification comes from the equation of state: $\mathcal{E}_e + p_e = \frac{5}{3} \mathcal{E}_e$.

Another comes from equipartition $k T_e = \frac{2}{3} \frac{\mathcal{E}_e}{n_e}$.

The friction term, after Braginskii, is


$$
Q_{ie} = 3 \frac{m_e}{m_i} \frac{n_e}{\tau_e} (k T_e - kT_i) = 2 \frac{m_e}{m_i} \frac{1}{\tau_e} (\mathcal{E}_e-\mathcal{E}_i),
$$


and one must later insert $\tau_e$ for additional dependence on the variables.

As before, do $\nabla \cdot (\mathcal{E}_e \mathbf{u}_e ) = \mathbf{u_e} \cdot \nabla \mathcal{E}_e + \mathcal{E}_e \nabla \cdot \mathbf{u}_e$ and use $\nabla \cdot \mathbf{u}_e = - \nabla \cdot (D_{\perp} \nabla u_e)$.

In all, and using the expression for $\nabla \cdot \mathbf{u}_e$ obtained earlier, the equation simplifies to


$$
\partial_t \mathcal{E}_e+\frac{5}{3} \left ( \mathbf{u}_e \cdot \nabla \mathcal{E}_e - \mathcal{E}_e \nabla \cdot (D_{\perp} \nabla n_e) \right ) = S_e^{\mathcal{E}}-\frac{\mathcal{E}_e}{\tau_{\mathcal{E}_e}}+2 \frac{m_e}{m_i} \frac{1}{\tau_e} (\mathcal{E}_e-\mathcal{E}_i) +\nabla \cdot(\kappa_{\perp,e} n_e \nabla_{\perp} \left (\frac{\mathcal{E}_e}{n_e} \right ))
$$


Eq. 98 (ion energy flux equation)
=================================

This is very similar to the electron one (note $n_i=n_e$, quasineutrality).


$$
\partial_t \mathcal{E}_i+\nabla \cdot (\mathcal{E}_i \mathbf{u}_i+ p_i \mathbf{u}_i) = S_i^{\mathcal{E}}-\frac{\mathcal{E}_i}{\tau_{\mathcal{E}_i}}-Q_{ie}+\nabla \cdot(\kappa_{\perp,i} n_e \nabla_{\perp} k T_i)
$$


There is an additional term in the drift-velocity, coming from charge exchange:


$$
\mathbf{u}_{cx} = \frac{\mu_{cx}}{1+\mu_{cx^2}} \frac{1}{B} ( -\nabla_{\perp} \Phi - \frac{1}{e n_e} \nabla p_i ).
$$


Here, note I have set $Z_i=1$ as seems to be implicit in $n_i=n_e$.

Following the same procedures as above, the equation becomes

Eq. 99 (neutral diffusion)
==========================

The equation is


$$
\partial_t n_n = S_n^n + \nabla \cdot (D_n \nabla_{\perp} p_n).
$$


Neutrals obviously experience zero drift velocity, only diffusion.

I am not sure what to do with this equation as there is no equation of state for the neutrals. Is the final $p_n$ actually meant to be $n_n$ -
?

Eq. 102 (elliptic equation for electrostatic potential)
=======================================================

The modified electric field is given by


$$
\mathbf{E}^+ = \frac{m_i}{e B^2} \left ( n_e \nabla_{\perp} \Phi + \frac{1}{e} \nabla_{\perp} p_i \right ).
$$


Taking the divergence one has


$$
\nabla \cdot \mathbf{E}^+ = \frac{m_i}{e B^2} \left ( \nabla \cdot (n_e \nabla_{\perp} \Phi) +\frac{2}{3e} \nabla \cdot \nabla_{\perp} \mathcal{E}_i \right ).
$$


This needs to be solved for $\Phi$ - question is whether the *Nektar++* HelmSolve (or related routine) can handle what is basically a steady diffusion with spatial variation in the diffusion coefficient.

Attempt at final system for implementation in *Nektar++*
========================================================

This is 2D with $\nabla \equiv \left ( \frac{\partial}{\partial x}, \frac{\partial}{\partial y} \right )$,hence I have explicitly removed the $\perp$ symbol from $\nabla$. The variables are $n_e$, $\varpi \equiv \nabla \cdot \mathbf{E}^+$, $\mathcal{E}_e$, $\mathcal{E}_i$, $n_n$, $\Phi$. All non-variable expressions are to be interpreted as constants in the first implementation; Braginskii expressions have been inserted where available. There are probably mistakes in the constants but the main point is whether the equations can be implemented in terms of variables
times constants \... see also the following section for a further attempt at reduction.

$$
\partial_t n_e + \mathbf{u}_e \cdot \nabla n_e - n_e \nabla \cdot (D_{\perp} \nabla n_e ) = n_e n_n K_i - n_e^2 K_r - \frac{n_e}{\tau_{n_e}}.
$$

(the above equation may contain mixed CGS and SI units \... also I used $\nu_{e \perp}$ for $\nu$) in the above
$\nabla \cdot \mathbf{u}_i = - \nabla \cdot (D_{\perp} \nabla n_e) + \frac{\mu_{cx}}{1+\mu_{cx}^2} \frac{Be}{m_i} \left ( \frac{\nabla \cdot \mathbf{E}^+}{n_e} - \frac{\mathbf{E}^+ \cdot \nabla n_e}{n_e^2} \right )$.

$$
\partial_t n_n = - n_e n_n K_i + n_e^2 K_r + \nabla \cdot (D_n \nabla p_n).
$$



$$
\nabla \cdot (n_e \nabla \Phi) = \frac{m_i}{e B^2} \varpi - \frac{2}{3e} \nabla^2 \mathcal{E}_i.
$$



$$
\mathbf{E}^+ = \frac{m_i}{e B^2} \left ( n_i \nabla \Phi + \frac{2}{3e} \nabla \mathcal{E}_i \right )
$$


Equations as above, with long expressions for constants removed
===============================================================


$$
\partial_t n_e + \mathbf{u}_e \cdot \nabla n_e - n_e \nabla \cdot (D_{\perp} \nabla n_e ) = K_i n_e n_n - K_r n_e^2 - c_1 n_e.
$$



$$
\partial_t \varpi + \partial_n (u_i)_m \partial_m E^+_n + 2 (\nabla \cdot \mathbf{u}_i) (\nabla \cdot \mathbf{E}^+) + \mathbf{E}^+ \cdot \nabla (\nabla \cdot \mathbf{u}_i) = - c_2 \varpi +
c_3 n_e + c_4 \nabla \cdot \left ( \frac{\mathcal{E}_e^{\frac{5}{2}}}{n_e^{\frac{7}{2}}} \nabla (\varpi) \right);
$$

in which $\mathbf{u}_i \equiv \frac{\mu_{cx}^2}{1+\mu_{cx}^2} \frac{\mathbf{B} \times \nabla \Phi}{B^2} -D_{\perp} \nabla n_e + \frac{\mu_{cx}}{1+\mu_{cx}^2}\frac{1}{B} \left ( -\nabla \Phi -\frac{2}{3n_e} \nabla \mathcal{E}_i\right )$
and $\nabla \cdot \mathbf{u}_i = - \nabla \cdot (D_{\perp} \nabla n_e) + \frac{\mu_{cx}}{1+\mu_{cx}^2} \frac{Be}{m_i} \left ( \frac{\nabla \cdot \mathbf{E}^+}{n_e} - \frac{\mathbf{E}^+ \cdot \nabla n_e}{n_e^2} \right )$.

$$
\partial_t \mathcal{E}_e+\frac{5}{3} \left ( \mathbf{u}_e \cdot \nabla \mathcal{E}_e - \mathcal{E}_e \nabla \cdot (D_{\perp} \nabla n_e) \right ) = \\ -K_i n_e n_n \mathcal{E}_i-K_r n_e \mathcal{E}_e-c_5\mathcal{E}_e+ c_6 \frac{n_e^{\frac{5}{2}}}{\mathcal{E}_e^{\frac{3}{2}}} (\mathcal{E}_e-\mathcal{E}_i) +c_7 \nabla \cdot \left ( \frac{n_e^{\frac{5}{2}}}{\mathcal{E}_e^{\frac{1}{2}}} \nabla \left (\frac{\mathcal{E}_e}{n_e} \right ) \right ).
$$



$$
\partial_t n_n = - n_e n_n K_i + n_e^2 K_r + \frac{1}{m_i(K_{cx}+K_i)}\nabla \cdot \left ( \frac{\nabla p_n}{n_e} \right ).
$$



$$
\nabla \cdot (n_e \nabla \Phi) = \frac{m_i}{e B^2} \varpi - \frac{2}{3e} \nabla^2 \mathcal{E}_i.
$$



$$
\mathbf{E}^+ = \frac{m_i}{e B^2} \left ( n_i \nabla \Phi + \frac{2}{3e} \nabla \mathcal{E}_i \right ).
$$


Questions
=========
