This is a test implmentation of a PIC solver for 1+1D Vlasov Poisson, written
in C++/DPC++.
This is primarily designed to test the use of multiple repos/workflows for
different code components.

## Installation

A relatively straightforward way of installing the dependencies is via spack, but be aware that this will take several hours!

```
spack install hipscyl@0.9.1 boost@1.78.0 llvm-openmp@12.0.1 fftw@3.3.10
spack load hipsycl@0.9.1 boost@1.78.0 llvm-openmp@12.0.1 fftw@3.3.10
```

## Build

TODO fix this

```
cmake -DCMAKE_CXX_COMPILER=icpx . -B build
cmake --build build
cd build
ctest ..
```

## Testing

Tests will be in `./build/test/`. TODO: fix this.

## Licence

This is licenced under MIT.

In order to comply with the licences of dependencies, this software is not to be released as a binary.
