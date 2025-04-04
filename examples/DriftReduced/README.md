# The DriftReduced solver

## Overview
The following describes the examples that are currently available for the `DriftReduced` solver.
To build the solver executable, follow the instructions for building NESO in the [top-level README](../../README.md).

<!--
## Prerequisites

In order to generate Nektar++ xml meshes, you'll need `gmsh` and `NekMesh`.
If NESO was installed with spack, then `NekMesh` should already be built.  It can be added to your path with:

    export PATH=$PATH:$(spack location -i nektar%[compiler])/bin

where [compiler] is either 'gcc' or 'oneapi' (both should work if 'spack install' completed without errors.)
-->

## Examples

### 2DHW (unfinished)

### 2Din3DHW_fluid_only

Solves the 2D Hasegawa-Wakatani (HW) equations on a 3D domain. That is:

$$
\begin{align}
    \frac{\partial n}{\partial t} + [\phi, n]  & = \alpha (\phi - n) - \kappa \frac{\partial\phi}{\partial y} \\
    \frac{\partial{\zeta}}{\partial t} + [\phi, \zeta] & = \alpha (\phi - n)
\end{align}
$$

where $n$ is number density, $\zeta$ is vorticity and $\phi$ is the electrostatic potential.

$[a,b]$ is the Poisson bracket operator, defined as

$$
\begin{equation}
    [a,b] = \frac{\partial a}{\partial x} \frac{\partial b}{\partial y} - \frac{\partial a}{\partial y} \frac{\partial 
b}{\partial x}.
\end{equation}
$$

<!--
Generate the mesh with

    ./scripts/geo_to_xml.sh examples/DriftReduced/2Din3DHW_fluid_only/cuboid.geo -x 1,2 -y 3,4 -z 5,6 -o cuboid_5x5x10_5x5x10hexes.xml
-->

The example can be run with

    ./scripts/run_eg.sh DriftReduced 2Din3DHW_fluid_only

This script expects to find mpirun on the path and executes with four MPI ranks by default. It looks for a solver executable in the most recently modified spack-build* directory, but this can be overridden using the '-b' option.

### 2Din3DHW

Solves equations (1) and (2), as in the previous example, but also enables a system of neutral particles that are coupled to the fluid solver. Particles deposit density into the (plasma) fluid via ionization.

<!--
Generate the mesh with

    ./scripts/geo_to_xml.sh examples/DriftReduced/2Din3DHW/cuboid.geo -x 1,2 -y 3,4 -z 5,6 -o cuboid_5x5x10_8x8x16hexes.xml
-->

The example can be run with

    ./scripts/run_eg.sh DriftReduced 2Din3DHW

This script expects to find mpirun on the path and executes with four MPI ranks by default. It looks for a solver executable in the most recently modified spack-build* directory, but this can be overridden using the '-b' option.

### 2DRogersRicci (unfinished)

Model based on the **2D** finite difference implementation described in "*Low-frequency turbulence in a linear magnetized plasma*", B.N. Rogers and P. Ricci, PRL **104**, 225002, 2010 ([link](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.104.225002)); see equations 5-7.

#### Equations

In SI units:

$$
\begin{aligned}
\frac{d n}{dt} &= -\sigma\frac{n c_s}{R}\exp(\Lambda - e\phi/T_e) + S_n ~~~(1)\\
\frac{d T_e}{dt} &= -\sigma\frac{2}{3}\frac{T_e c_s}{R}\left[1.71\exp(\Lambda - e\phi/T_e)-0.71\right] + S_T ~~~(2)\\
\frac{d \nabla^2\phi}{dt} &= \sigma \frac{c_s m_i \Omega_{ci}^2}{eR}\left[1-\exp(\Lambda - e\phi/T_e)\right] ~~~(3)\\
\end{aligned}
$$

where

$$
\begin{aligned}
\sigma &= \frac{1.5 R}{L_z} \\
\frac{df}{dt} &= \frac{\partial f}{\partial t} - \frac{1}{B}\left[\phi,f\right] \\
\end{aligned}
$$

and the source terms have the form

$$
\begin{aligned}
S_n &= S_{0n}\frac{1-{\rm tanh[(r-r_s)/L_s]}}{2} \\
S_T &= S_{0T}\frac{1-{\rm tanh[(r-r_s)/L_s]}}{2} \\
\end{aligned}
$$

where $r = \sqrt{x^2 + y^2}$

#### Model parameters

| Parameter     | Value               | Comment                                                                                                         |
| ------------- | ------------------- | --------------------------------------------------------------------------------------------------------------- |
| $T_{e0}$      | 6 eV                |                                                                                                                 |
| $L_z$         | 18 m                |                                                                                                                 |
| $n_0$         | 2e18 m<sup>-3</sup> |                                                                                                                 |
| $m_i$         | 6.67e-27 kg         | Inferred from the value of $c_{s0}$ quoted in the paper. Value is $\sim 4 m_p$, consistent with a Helium plasma |
| $\Omega_{ci}$ | $9.6e5$             |                                                                                                                 |
| $\Lambda$     | 3                   | Couloumb Logarithm                                                                                              |
| R             | 0.5 m               | Approx radius of the plasma column                                                                              |

Derived values
| Parameter   | Calculated as            | Value                               | Comment                                           |
| ----------- | ------------------------ | ----------------------------------- | ------------------------------------------------- |
| B           | $\Omega_{ci} m_i q_E$    | 40 mT                               |                                                   |
| $c_{s0}$    | $\sqrt{T_{e0}/m_i}$      | 1.2e4 ms<sup>-1</sup>               |                                                   |
| $\rho_{s0}$ | $c_{s0}/\Omega{ci}$      | 1.2e-2 m                            | Paper has 1.4e-2 m ... implies $m_i\sim 3 m_p$ !? |
| $S_{0n}$    | 0.03 $n_0 c_{s0}/R$      | 4.8e22 m<sup>-3</sup>s<sup>-1</sup> |                                                   |
| $S_{0T}$    | 0.03 $T_{e0} c_{s0} / R$ | 4318.4 Ks<sup>-1</sup>              |                                                   |
| $\sigma$    | $1.5 R/L_z$              | 1/24                                |                                                   |
| $L_s$       | $0.5\rho_{s0}$           | 6e-3 m                              |                                                   |
| $r_s$       | $20\rho_{s0}$            | 0.24 m                              | Approx radius of the LAPD plasma chamber          |

#### Initial conditions

The default initial conditions are

| Field    | Default ICs (uniform)                          |
| -------- | ---------------------------------------------- |
| n        | $2\times10^{14} m^{-3}$ ($10^{-4}$ normalised) |
| T        | $6\times10^{-4}$ eV ($10^{-4}$ normalised)     |
| $\omega$ | 0                                              |

#### Boundary conditions

All fields have Dirichlet for the potential on all boundaries; (homogeneous) Neumann for all other fields.

| Field    | Dirichlet BC value |
| -------- | ------------------ |
| n        | $10^{-4}$          |
| T        | $10^{-4}$          |
| $\omega$ | 0                  |
| $\phi$   | $\phi_{\rm bdy}$   |

$\phi_{\rm bdy}$ is set to 0.03 by default. This value was chosen to keep the value of $\phi$ relatively flat outside the central source region and avoid boundary layers forming in $\omega$ and $\phi$. 

#### Domain and mesh

The mesh is a square with the origin at the centre and size $\sqrt{T_{e0}/m_i}/\Omega{ci} = 100\rho_{s0} = 1.2$ m.

By default, there are 64x64 quadrilateral (square) elements, giving sizes of 1.875 cm = 25/16 $\rho_{s0}$

Default res is substantially lower than that used in the finite difference model, which has 1024x1024 elements. (I think)

#### Normalisation

Normalisations follow those in Rogers & Ricci, that is:

|                       | Normalised to   |
| --------------------- | --------------- |
| Charge                | $e$             |
| Electric potential    | $e/T_{e0}$      |
| Energy                | $T_{e0}$        |
| Number densities      | $n_0$           |
| Perpendicular lengths | $100 \rho_{s0}$ |
| Parallel lengths      | $R$             |
| Time                  | $R/c_{S0}$      |


The normalised forms of the equations are:

$$
\begin{align}
\frac{\partial n}{\partial t} &= 40\left[\phi,n\right] -\frac{1}{24}\exp(3 - \phi/T_e)n + S_n  ~~~({\bf 4}) \\
\frac{\partial T_e}{\partial t} &= 40\left[\phi,T_e\right] -\frac{1}{36}\left[1.71\exp(3 - \phi/T_e)-0.71\right]T_e + S_T  ~~~({\bf 5}) \\
\frac{\partial  \nabla^2\phi}{\partial t} &= 40\left[\phi,\nabla^2\phi\right] + \frac{1}{24}\left[1-\exp(3 - \phi/T_e)\right] ~~~({\bf 6})\\
\nabla^2\phi &= \omega ~~({\bf 7}) \\
\end{align}
$$

with 

$$
\begin{equation}
S_n = S_T = 0.03\left\\{1-\tanh[(\rho_{s0}r-r_s)/L_s]\right\\}
\end{equation}
$$

where $\rho_{s0}$, $r_s$ and $Ls$ have the (SI) values listed in the tables above.
<!-- This system can be be obtained by applying the normalisation factors, then simplifying; see [here](./details/rogers-ricci-2d-normalised.md) for details. Note that the prime notation used in the derivations is dropped in the equations above for readability. -->

#### Simulation time

Based on Fig 4. of Rogers & Ricci, the simulation time for the 3D version might be $\sim 12$ in normalised units (= $500~{\rm{\mu}s}$), but it's not clear if the full duration is being shown. 
$500~{\rm{\mu}s}$ doesn't seem enough time for anything interesting to happen - we (arbitrarily) choose to run for ten times longer - $5~{\rm ms}$, or  $\sim 120$ in normalised units.

#### Decoupled density
Note that, since the density only features in equation 4, it is effectively decoupled from the rest of system. Implementing equations 5-7 only is therefore sufficient to capture most of the interesting behaviour.

#### Example output

<figure>
  <img src="../../docs/media/rr2D_64x64_CG_temperature.gif" align="left" width="50%" >
  <img src="../../docs/mediamedia/rr2D_64x64_CG_vorticity.gif" align="left" width="50%" >
  <figcaption>Temperature (left) and vorticity (right) in normalised units, run with the CG implementation on a 64x64 quad mesh for 5 ms (120 normalised time units).</figcaption>
</figure>

<figure>
  <img src="../../docs/mediamedia/rr2D_64x64_DG_temperature.gif" align="left" width="50%" >
  <img src="../../docs/mediamedia/rr2D_64x64_DG_vorticity.gif" align="left" width="50%" >
  <figcaption>Temperature (left) and vorticity (right) in normalised units, run with the DG implementation on a 64x64 quad mesh for 5 ms (120 normalised time units).</figcaption>
</figure>

### 3DHW (unfinished)

## Diagnostics
For the Hasegawa-Wakatani examples (`2DHW`,` 2Din3DHW_fluid_only`, `2Din3DHW`, `3DHW`), the solver can be made to output the total fluid energy ($E$) and enstrophy ($W$), which are defined as:  

$$
\begin{align}
E&=\frac{1}{2}\int (n^2 + |\nabla\phi|^2)~\mathbf{dx}\\ 
W&=\frac{1}{2}\int (n-\zeta)^2~\mathbf{dx}
\end{align}
$$

In the `2Din3DHW_fluid_only` example, the expected growth rates of $E$ and $W$ can be calculated analytically according to:

$$
\begin{align}
\frac{dE}{dt} &= \Gamma_n-\Gamma_\alpha \\
\frac{dW}{dt} &= \Gamma_n
\end{align}
$$

where

$$
\begin{align}
\Gamma_\alpha &= \alpha \int (n - \phi)^2~\mathbf{dx}\\
\Gamma_n &= -\kappa \int n \frac{\partial{\phi}}{\partial y}~\mathbf{dx}
\end{align}
$$

To change the frequency of this output modify the value of `growth_rates_recording_step` inside the `<PARAMETERS>` node in the example's configuration file`.
When that parameter is set, the values of $E$ and $W$ are written to `<run_directory>/growth_rates.h5` at each simulation step $^*$.  Expected values of $\frac{dE}{dt}$ and $\frac{dW}{dt}$, calculated with equations (6) and (7) are also written to file, but note that these are only meaningful when particle coupling is disabled.

$^*$ Note that the file will appear empty until the file handle is closed at the end of simulation.