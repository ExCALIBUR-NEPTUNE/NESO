The following describes the examples that are currently available for the `H3LAPD` solver.
To build the solver executable, follow the instructions for building NESO in the [top-level README](../../README.md).


## Prerequisites

In order to generate Nektar++ xml meshes, you'll need `gmsh` and `NekMesh`.
If NESO was installed with spack, then `NekMesh` should already be built.  It can be added to your path with:

    export PATH=$PATH:$(spack location -i nektar%[compiler])/bin

where [compiler] is either 'gcc' or 'oneapi' (both should work if 'spack install' completed without errors.)

## Examples

### 2Din3D-hw_fluid-only

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

Generate the mesh with

    ./scripts/geo_to_xml.sh examples/H3LAPD/2Din3D-hw_fluid-only/cuboid_periodic_5x5x10.geo -x 1,2 -y 3,4 -z 5,6 -o cuboid.xml

Then run the example with

    ./scripts/run_eg.sh H3LAPD 2Din3D-hw_fluid-only

This script expects to find mpirun on the path and executes with four MPI ranks by default. It looks for a solver executable in the most recently modified spack-build* directory, but this can be overridden using the '-b' option.

### 2Din3D-hw

Solves equations (1) and (2), as in the previous example, but also enables a system of neutral particles that are coupled to the fluid solver. Particles deposit density into the (plasma) fluid via ionization.

Generate the mesh with

    ./scripts/geo_to_xml.sh examples/H3LAPD/2Din3D-hw/cuboid_periodic_8x8x16.geo -x 1,2 -y 3,4 -z 5,6 -o cuboid.xml

Then run the example with

    ./scripts/run_eg.sh H3LAPD 2Din3D-hw

This script expects to find mpirun on the path and executes with four MPI ranks by default. It looks for a solver executable in the most recently modified spack-build* directory, but this can be overridden using the '-b' option.

## Diagnostics
For the '2Din3DHW' equation system (used in the `2Din3D-hw` and `2Din3D-hw_fluid-only` examples), the solver can be made to output the total fluid energy ($E$) and enstrophy ($W$), which are defined as:  

$$
\begin{align}
E&=\frac{1}{2}\int (n^2 + |\nabla\phi|^2)~\mathbf{dx}\\ 
W&=\frac{1}{2}\int (n-\zeta)^2~\mathbf{dx}
\end{align}
$$

In the `2Din3D-hw_fluid-only` example, the expected growth rates of $E$ and $W$ can be calculated analytically according to:

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

To change the frequency of this output modify the value of `growth_rates_recording_step` inside the `<PARAMETERS>` node in `<example_directory>/hw.xml`.
When that parameter is set, the values of $E$ and $W$ are written to `<run_directory>/growth_rates.csv` at each simulation step $^*$.  Expected values of $\frac{dE}{dt}$ and $\frac{dW}{dt}$, calculated with equations (6) and (7) are also written to file, but note that these are only meaningful when particle coupling is disabled.

$^*$ Note that the file will appear empty until the file handle is closed at the end of simulation.