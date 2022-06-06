This is a test implmentation of a PIC solver for 1+1D Vlasov Poisson, written
in C++/DPC++.
This is primarily designed to test the use of multiple repos/workflows for
different code components.

## Dependencies

* CMake
* Boost >= 1.78 (for tests)
* SYCL implementation Hipsycl and fftw or OneAPI and MKL.
* Nektar++

### Building dependencies with Spack

A relatively straightforward way of installing the dependencies is via spack, but be aware that this will take several hours!

```
spack install hipscyl@0.9.2 boost@1.78.0 llvm-openmp@12.0.1 fftw@3.3.10
spack load hipsycl@0.9.2 boost@1.78.0 llvm-openmp@12.0.1 fftw@3.3.10
```

### CMake 

Ensure a recent version of [CMake](https://cmake.org/download/) is available.
If necessary, install with

```
wget https://github.com/Kitware/CMake/releases/download/v3.23.1/cmake-3.23.1.tar.gz
tar xvf cmake-3.23.1.tar.gz
cd cmake-3.23.1/
./configure
make
```

### Boost

The test suite requires the [Boost library](https://www.boost.org/) (version >= 1.78).
If this is not available on your system, this can be build from source by doing

```
wget https://boostorg.jfrog.io/artifactory/main/release/1.79.0/source/boost_1_79_0.tar.gz
tar xvf boost_1_79_0.tar.gz
cd boost_1_79_0/
./bootstrap.sh
./b2
```

If the install is not automatically found by cmake, specify the path to the
install dir at configure time:

```
cmake -DBoost_INCLUDE_DIR=/path/to/boost_1_79_0/ . -B build
```

### Nektar++

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
cmake -DNektar++_DIR=/path/to/nektar/build/dist/lib64/nektar++/cmake . -B build
cmake --build build
```

where `/path/to/nektar/build/dist/lib64/nektar++/cmake` is the folder containing
the `Nektar++Config.cmake` file. 
Note that for this file to exist, you must do `make install` at the end of the
Nektar++ build.

### NEPTUNE

To build the code and the tests, do

```
cmake . -B build
cmake --build build
```

It may be necessary to tell CMake the location of dependencies:

* Boost by setting `-DBoost_INCLUDE_DIR`
* SYCL compiler by setting `-DCMAKE_CXX_COMPILER`
* Nektar++ by setting the location of `Nektar++Config.cmake` in `-DNeektar++_DIR`

It may be necessary to specify the SYCL compiler and location of Boost, e.g.

```
cmake -DCMAKE_CXX_COMPILER=icpx -DBoost_INCLUDE_DIR=/root/code/boost_1_78_0 -DNektar++_DIR=/root/code/nektar/build/dist/lib64/nektar++/cmake . -B build
cmake --build build
```

The executable `NESO` is created in `bin`. 


## System-specific information

### Dirac (CSD3 @ Cambridge)

```
module unload intel/compilers/2017.4
module unload intel/mkl/2017.4
module load gcc/11
module load intel/oneapi/2022.1.0/compiler
module load intel/oneapi/2022.1.0/mkl
module load intel/oneapi/2022.1.0/tbb
export  LD_LIBRARY_PATH=/usr/local/software/intel/oneapi/2022.1/compiler/latest/linux/lib:$LD_LIBRARY_PATH
```

## Testing

cmake also builds a test suite `bin/testNESO`.

## Address Sanitizers

To debug for memory leaks, compile with the options

```
cmake . -B -DENABLE_SANITIZER_ADDRESS=on -DENABLE_SANITIZER_LEAK=on
```

## Licence

This is licenced under MIT.

In order to comply with the licences of dependencies, this software is not to be released as a binary.

