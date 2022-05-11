This is a test implmentation of a PIC solver for 1+1D Vlasov Poisson, written
in C++/DPC++.
This is primarily designed to test the use of multiple repos/workflows for
different code components.

## Requirements

* Boost >= 1.78 (for tests)

## Building

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

## Testing

Run tests by doing

```
cd test
cmake -S -B build
cmake --build build
cd build && ctest
```

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

