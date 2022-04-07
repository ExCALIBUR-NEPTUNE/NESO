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

To build, do

```
cmake . -B build
cmake --build build
```

It may be necessary to specify the SYCL compiler and location of Boost, e.g.

```
cmake -DCMAKE_CXX_COMPILER=icpx -DBoost_INCLUDE_DIR=/root/code/boost_1_78_0 . -B build
cmake --build build
```

The executable `PolyrepoPracticeCore` is created in `bin`. 

## NEKTAR++ Dependence

To build with Nektar++, ensure that Nektar++ is installed on your system.
There are instructions on the 
Compile it following the instructions at
[https://www.nektar.info](https://www.nektar.info),
but briefly,

```bash
git clone https://gitlab.nektar.info/nektar/nektar
cd nektar
mkdir build && cd build
cmake .. 
cmake --build .
make install
```

should install Nektar++.

To build NEPTUNE with Nektar++, set the `Nektar++_DIR` flag in cmake, e.g.

```
cmake -DNektar++_DIR=/path/to/nektar/build/dist/lib/nektar++/cmake . -B build
cmake --build build
```

where `/path/to/nektar/build/dist/lib/nektar++/cmake` is the folder containing
the *Nektar++Config.cmake* file. 
Note that for this file to exist, you must do `make install` at the end of the
Nektar++ build.


## Testing

Tests will be in `./build/test/`. TODO: fix this.

## Address Sanitizers

To debug for memory leaks, compile with the options

```
cmake . -B -DENABLE_SANITIZER_ADDRESS=on -DENABLE_SANITIZER_LEAK=on
```

## Licence

This is licenced under MIT.

In order to comply with the licences of dependencies, this software is not to be released as a binary.
