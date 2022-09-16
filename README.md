This is a test implmentation of a PIC solver for 1+1D Vlasov Poisson, written
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
like `spack-build-6gyyv2t` (the hash may differ).  FFTW and hipSYCL
will be used; OneAPI currently isn't supported in Spack (but hopefully
will be soon). It will also build NESO for you in a directory called
something like `spack-build-6gyyv2t` (the exact hash may
differ). Binaries, however, are placed in the `bin` directory at the
top level of the repository. They are not currently installed. This is
likely to change in future.

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
logs.

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
a small risk CMake will end up using a different compiler or compiler
version than was used to build all of the dependencies. You should
ensure you are aware of what compilers you have installed and, if
necessary, explicitely specify to CMake which you want to use.

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

#### Nektar++

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

### Manually building NEPTUNE

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

cmake also builds a test suite `bin/testNESO`.

## Address Sanitizers

To debug for memory leaks, compile with the options

```
cmake . -B -DENABLE_SANITIZER_ADDRESS=on -DENABLE_SANITIZER_LEAK=on
```

## Licence

This is licenced under MIT.

In order to comply with the licences of dependencies, this software is not to be released as a binary.

