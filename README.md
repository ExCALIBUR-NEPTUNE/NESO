# NESO (Neptune Exploratory SOftware)

This is a work-in-progress respository for exploring the implementation of 
a series of tokamak exhaust relevant models combining high order finite
elements with particles, written in C++ and SYCL.

## Dependencies

* CMake
* Boost >= 1.74 (for tests)
* SYCL implementation Hipsycl and fftw or OneAPI and MKL.
* Nektar++
* NESO-Particles

### Building with Spack

The easiest way to install NESO is using the
[Spack package manager](https://spack.readthedocs.io/en/latest/index.html), although
this can take a few hours the first time you do it or if you change
compilers. This repository has been set up so you can use a
variation of the [Spack developer
workflow](https://spack-tutorial.readthedocs.io/en/latest/tutorial_developer_workflows.html). Simply
run the following commands at the top level of the repository:

If you don't already have Spack available on your computer, install it
according to the [official documentation](https://spack.readthedocs.io/en/latest/getting_started.html#installation), or as follows:
```bash
# Ensure all prerequisites are installed
apt update # For Ubuntu; other distros have their own commands
apt install build-essential ca-certificates coreutils curl environment-modules gfortran git gpg lsb-release python3 python3-distutils python3-venv unzip zip

git clone -c feature.manyFiles=true -b v0.19.0 https://github.com/spack/spack.git $HOME/.spack
echo 'export SPACK_ROOT=$HOME/.spack' >> $HOME/.bashrc
echo 'source $SPACK_ROOT/share/spack/setup-env.sh' >> $HOME/.bashrc
export SPACK_ROOT=$HOME/.spack
source $SPACK_ROOT/share/spack/setup-env.sh
```

Next, install the Intel compilers if they are not already present on
your computer.
```bash
spack install intel-oneapi-compilers
spack load intel-oneapi-compilers
spack compiler find
spack unload intel-oneapi-compilers
```

Now, activate the NESO development environment and build it and its
dependencies. This must be done from the top level of this
repository.
```
git submodule update --init
. activate
spack install
```
The `activate` script sets some useful environment variables and runs
`spack activate ...`. This activates an [anonymous Spack
environment](https://spack.readthedocs.io/en/latest/environments.html#anonymous-environments)
based on the settings in [the spack.yaml file](spack.yaml). These
configurations tell Spack to install NESO and all of its
dependencies. Rather than pulling fresh NESO and Nektar++ source code
from GitHub, it will use the copy of the code in your current working
directory and the Nektar++ submodule (respectively). You can leave
this environment at any time by running `deactivate`.

`spack install` will build two copies of NESO: one with
GCC/hipSYCL/FFTW3 and one with Intel's OneAPI/DPC++/MKL. These
packages and their dependencies will be installed in the usual Spack
locations. They will also be linked into ["filesystem
views"](https://spack.readthedocs.io/en/latest/environments.html#filesystem-views)
`view/gcc-hipsycl` and `view/oneapi-dpcpp`. The NESO builds will be
done in directories called something like `spack-build-abc1234` (the
hashes at the end will differ). If you change your spack installation
in some way (e.g., upgrading the version of a dependency) then the
hash will change and NESO and/or Nektar++ will be rebuilt. The
activation provides the convenience command `cleanup` to delete these
old builds.

In order to tell which build is which, symlinks `builds/gcc-hipsycl`
and `builds/oneapi-dpcpp` are provided. As Nektar++ is being built
from the submodule, its build trees are located at
`nektar/spack-build-abc1234` (the hashes at the end will differ) and
can be accessed with symlinks `nektar/builds/gcc` and
`nektar/builds/oneapi`. Test binaries will be contained within
these build directories. The activation script launches a background
task which regularly checks whether the hashes of your NESO and
Nektar++ builds has changed. If they have, it will update the
symlinks. They will also be checked whenever the environment is
activated or deactivated.

It has been found that the oneAPI and clang
compilers struggle to build NumPy and Boost due to very large memory
requirements. As such, the oneAPI build of NESO compiles these
dependencies using another compiler  (the Intel Classic compilers by default). Feel free to
experiment with changing these or seeing if there is a way to make the
builds work with oneAPI.

#### Developing
As you develop the code, there are a few options for how you
recompile. One is simply to run `spack install` again. This will reuse
the existing build directory and reinstall the results of the
build. The build environment used is determined by the package
configuration for NESO (as specificed in the NESO [package
repository](https://github.com/ExCALIBUR-NEPTUNE/NESO-Spack) and is
the same as if you were doing a traditional Spack installation of a
named version of NESO. This has the particular advantage of building
with all toolchaings (i.e., GCC, OneAPI) at one time. It also works
well if you are developing NESO and Nektar++ simultaneously, as it
will rebuild both. The main disadvantage of this approach is that
Spack hides the output of CMake during the build process and will only
show you any information on the build if there is an error. This means
you will likely miss any compiler warnings, unless you check the build
logs. There is also some overhead associated with running Spack, which
makes the build a little slow.

An alternative approach is to prefix your usual build commands with
`spack build-env neso%gcc` or `spack build-env neso%oneapi` (depending
on which compiler you want to use). This will cause the commands to be run in the
same build environment as when installing that version of NESO. For example, you
could run

```bash
spack build-env neso%gcc cmake . -B build
spack build-env neso%gcc cmake --build build
```
This would cause the build to occur in the directory `build`. This
approach works quite well. A slight downside is the commands are
a bit cumbersome. It also won't build for both compilers at once,
although you might not want to do that anyway, early during the development
of a new feature.

Finally, you could take advantage of the filesystem views created 
when you installed the environment and which give you access
to all of the resources for the build that you need. For example, you
could run

```bash
cmake -DCMAKE_PREFIX_PATH=$(pwd)/gcc-hipsycl . -B build
cmake --build build
```
CMake will automatically be able to find all of the packages it needs
in `gcc-hipsycl`. The downside of this approach is that there is a
risk CMake will end up using a different compiler or compiler version
than intended. This is especially likely if not using a system
compiler. You should ensure you are aware of what compilers you have
installed and, if necessary, explicitly specify to CMake which you
want to use.

In general, it is recommended to use the `spack build-env ...`
command for compiling earlier in the development process (if you
aren't simultaneously making changes to Nektar++), to ensure you see
any warnings and so you aren't spending extra times compiling things
twice. Once you believe you have a working implementation with one
compiler (of if you are working on NESO and Nektar++ simultaneously),
you can use `spack install` to test the build against other
compilers. It is *not* recommended to use the Spack views, as building
this way is less likely to be reproducible.

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

### NESO-Particles

Install NESO-Particles by following the installation instructions at [https://github.com/ExCALIBUR-NEPTUNE/NESO-Particles](https://github.com/ExCALIBUR-NEPTUNE/NESO-Particles). Additional configuration options for NESO-Particles can be passed when NESO is configured through cmake.


### NESO

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

## Handling Submodules when Developing

This repository contains some git submodules, for code which is likely
to undergo development tightly-coupled with NESO (e.g., Nektar) or
which it is convenient to distribute this way (NESO-spack). When
checking out different branches, on which submodules are at different
commits, it can be easy to end up with working against different
versions of submodules than you think. This can cause headaches when
trying to reproduce certain behaviour. Run `git submodule update
--recursive` to ensure everything is at the correct commit. You can
avoid this issue altogether by using `git checkout
--recurse-submodules`. You can also set this as a default for your
copy of the repository by running `git config --local
submodule.recurse true`. Note, however, that these latter two options
can cause git to complain if trying to switch between two branches,
only one of which contains a submodule.


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

CMake also builds a suite unit tests (e.g. `<build_dir>/test/unitTests`)
and integration tests (`<build_dir>/test/integrationTests`). The build
directories are `builds/gcc-hipsycl` and `builds/oneapi-dpcpp`.

A subset of the tests may be run using appropriate flags:
e.g. `path/to/testExecutable --gtest_filter=TestSuiteName.TestName`.
See the [googletest user guide](http://google.github.io/googletest/)
for more info, especially with regards to running [specific
tests](https://google.github.io/googletest/advanced.html#selecting-tests).

Alternatively, you can call
[CTest](https://cmake.org/cmake/help/latest/manual/ctest.1.html) from
within the build directory to execute your tests.

# Solvers
Each solver has
- Source code: `solvers/<solver_name>`
- Integration tests: `test/integration/solvers/<solver_name>`
- Examples: `examples/<solver_name>/<example_name>`

To run a solver example:
```
./scripts/run_eg.sh  [solver_name] [example_name] <-n num_MPI> <-b build_dir>
```
which will look for the solver executable in the most recently modified spack-build-* directory, unless one is supplied with `-b`.  Output is generated in `example-runs/<solver_name>/<example_name>`.

## Address Sanitizers

To debug for memory leaks, compile with the options

```
cmake . -B -DENABLE_SANITIZER_ADDRESS=on -DENABLE_SANITIZER_LEAK=on
```

## Licence

This is licenced under MIT.

In order to comply with the licences of dependencies, this software is not to be released as a binary.

