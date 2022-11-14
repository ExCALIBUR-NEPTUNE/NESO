This is a test implementation of a PIC solver for 1+1D Vlasov Poisson, written
in C++/DPC++.
This is primarily designed to test the use of multiple repos/workflows for
different code components.

## Dependencies

* CMake
* Boost >= 1.74 (for tests)
* SYCL implementation Hipsycl and fftw or OneAPI and MKL.
* Nektar++

### Building with Spack

The easiest way to install NESO is using the
[Spack package manager](https://spack.readthedocs.io/en/latest/index.html), although
this can take a few hours the first time you do it or if you change
compilers. This repository has been set up so you can use a
variation of the [Spack developer
workflow](https://spack-tutorial.readthedocs.io/en/latest/tutorial_developer_workflows.html). Simply
run the following commands at the top level of the repository:

```bash
git submodule update --init  # Probably not necessary, but just to be safe
spack env activate -p -d .
spack install
```

This creates an [anonymous Spack
environment](https://spack.readthedocs.io/en/latest/environments.html#anonymous-environments)
based on the settings in [the spack.yaml file](spack.yaml). These
configurations tell Spack to install NESO and all of its
dependencies. Rather
than pulling fresh NESO source code from GitHub,
it will use the copy of the code in your current working
directory. These packages will be installed in the usual Spack
locations. They will also be linked into a ["filesystem
view"](https://spack.readthedocs.io/en/latest/environments.html#filesystem-views)
located at `.spack-env/view/`. Environment variables such as `PATH`
and `CMAKE_PREFIX_PATH` were updated to include the view directory and
its subdirectories when you called `spack env activate...`.
The NESO build will be done in a directory called something
like `spack-build-6gyyv2t` (the hash at the end will differ).
Binaries, however, are placed in the `bin` directory at the
top level of the repository. They are not currently installed. This is
a bug and will likely change in future.

#### Using GCC

By default, the build (as set in `spack.yaml`) uses GCC, along with the
FFTW and hipSYCL libraries.

#### Building with OneAPI

To build with the Intel OneAPI compilers, modify `spack.yaml` by
commenting out the `neso%gcc` spec and uncommenting the `neso%oneapi`
spec. The latter spec will use MKL instead of FFTW and OpenBLAS and
DPC++ instead of hipSYCL. It has been found that the oneAPI and clang
compilers struggle to build NumPy and Boost due to very large memory
requirements. As such, the spec has been set to use other compilers
for this (the Intel Classic compilers by default). Feel free to
experiment with changing these or seeing if there is a way to make the
builds work with oneAPI.

For this to work, you must have the Intel compilers installed
and [registered with
Spack](https://spack.readthedocs.io/en/latest/getting_started.html#compiler-configuration). Note
that the Intel implementation of SYCL currently requires the
environment variable `LD_LIBRARY_PATH` to be set at run-time. As such,
the binaries will only run when the environment is active.

Sometimes when switching between compilers, CMake doesn't seem to
fully reset on the first attempt of the build. It is not clear why
this is the case, but the issue seems to be resolved by running the
`spack install` command again.

#### Developing
As you develop the code, there are a few options for how you
recompile. One is simply to run `spack install` again. This will
reuse the existing build directory and reinstall the results of the
build. The build environment used is determined by the packages
configuration for NESO (as specificed in the NESO [package
repository](https://github.com/ExCALIBUR-NEPTUNE/NESO-Spack) and is
the same as if you were doing a traditional Spack installation of a
named version of NESO. The main disadvantage of this approach is that
Spack hides the output of CMake during the build process and will only
show you any information on the build if there is an error. This means
you will likely miss any compiler warnings, unless you check the build
logs. Spack is also quite slow when running like this, as it needs to
run its full dependency concretization process.

An alternative approach is to prefix your usual build commands with
`spack build-env neso`. This will cause the commands to be run in the
same build environment used for `spack install`. For example, you
could run

```bash
spack build-env neso cmake . -B build
spack build-env neso cmake --build build
```
This would cause the build to occur in the directory `build`. This
approach works quite well. The only slight downside is the commands are
a bit cumbersome.

Finally, you can take advantage of the fact that the filesystem view
which was loaded when you activated the environment gives you access
to all of the resources for the build that you need. As such, you
could just run

```bash
cmake . -B build
cmake --build build
```
CMake will automatically be able to find all of the packages it needs
in `.spack-env/view/`. The downside of this approach is that there is
a risk CMake will end up using a different compiler or compiler
version than was used to build all of the dependencies. This is
especially likely if not using a system compiler. You should
ensure you are aware of what compilers you have installed and, if
necessary, explicitly specify to CMake which you want to use.

### Manually Installing Dependencies

#### CMake 

Ensure a recent version of [CMake](https://cmake.org/download/) is available.
If necessary, install with

```
wget https://github.com/Kitware/CMake/releases/download/v3.23.1/cmake-3.23.1.tar.gz
tar xvf cmake-3.23.1.tar.gz
cd cmake-3.23.1/
./configure
make
```

#### Boost

The test suite requires the [Boost library](https://www.boost.org/) (version >= 1.74).
If this is not available on your system, it can be built from source by doing

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

#### Nektar++

To build with Nektar++, ensure that Nektar++ is installed on your system.
Detailed instructions can be found in the [Nektar user guide](https://doc.nektar.info/userguide/latest/user-guidese3.html#x7-60001.3),
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

To build NESO with Nektar++, set the `Nektar++_DIR` flag in cmake, e.g.

```
cmake -DNektar++_DIR=/path/to/nektar/build/dist/lib64/nektar++/cmake . -B build
cmake --build build
```

where `/path/to/nektar/build/dist/lib64/nektar++/cmake` is the folder containing
the `Nektar++Config.cmake` file. 
Note that for this file to exist, you must do `make install` at the end of the
Nektar++ build.

### Manually building NESO

To build the code and the tests, do

```
cmake . -B build
cmake --build build
```

It may be necessary to tell CMake the location of dependencies:

* Boost by setting `-DBoost_INCLUDE_DIR`
* SYCL compiler by setting `-DCMAKE_CXX_COMPILER`
* Nektar++ by setting the location of `Nektar++Config.cmake` using `-DNektar++_DIR`

For example:

```
cmake -DCMAKE_CXX_COMPILER=dpcpp -DBoost_INCLUDE_DIR=/root/code/boost_1_78_0 -DNektar++_DIR=/root/code/nektar/build/dist/lib64/nektar++/cmake . -B build
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

CMake also builds a suite unit tests (`<build_dir>/test/unitTests`)
and integration tests (`<build_dir>/test/integrationTests`).

A subset of the tests may be run using appropriate flags:
e.g. `path/to/testExecutable --gtest_filter=TestSuiteName.TestName`.
See the [googletest user guide](http://google.github.io/googletest/)
for more info, especially with regards to running [specific
tests](https://google.github.io/googletest/advanced.html#selecting-tests).

Alternatively, you can call
[CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html) from
within the build directory to execute your tests.

## Address Sanitizers

To debug for memory leaks, compile with the options

```
cmake . -B -DENABLE_SANITIZER_ADDRESS=on -DENABLE_SANITIZER_LEAK=on
```

## Licence

This is licenced under MIT.

In order to comply with the licences of dependencies, this software is not to be released as a binary.

