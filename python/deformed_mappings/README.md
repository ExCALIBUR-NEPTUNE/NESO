# Deformed Mapping Code Generation
## Introduction

In FEM codes there exist (non)-linear maps ($X(\xi)$) from reference element space ($\xi$) to physical space.
The non-linearity of these maps complicate the process of finding a mesh element which contains a given physical point. 
This "cell binning" problem is formulated as follows; given an input point $x$, find an element e and reference position $xi$ such that $X_e(\xi) = x$.
For each candidate mesh element $e$ a Newton iteration is applied to evaluate $\xi = X_e^{-1}(x)$.
If $\xi$ falls inside the reference element, for the geometry type of $e$, then $e$ is the element that "contains" the point $x$.

Each (linear sided) element type has a different map $X_e$ which is a function of the vertices of a particular instance of the element.
The functional form of the map $X$ ranges between linear and tri-linear depending on the element type.
This repository symbolically represents the maps $X$ for each of the 3D element types and uses the automatic differentiation and code generation capabilities of Sympy to generate the C++ code required to perform the Newton iterations.

## Installation
The requirements are listed in the `requirements.txt` file and can be installed into a Python virtual environment with
```
pip install -r requirements.txt
```

## Testing and Generating Output
The generation script will run the self tests for each geometry type before generating output.
The tests can be ran without generating output by running
```
make test
```

To run the self tests and generate code output in the NESO include directory run
```
# output dir will be "../../include/nektar_interface/particle_cell_mapping/generated_linear"
make
```

To place output in `./generated_linear` for development/testing run
```
make dev
```


