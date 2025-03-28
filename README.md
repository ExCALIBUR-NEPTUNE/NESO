# NESO (Neptune Exploratory SOftware)

This is a work-in-progress respository for exploring the implementation of 
a series of tokamak exhaust relevant models combining high order finite
elements with particles, written in C++ and SYCL.

## Dependencies

* CMake
* Boost >= 1.74 (for tests)
* SYCL implementation AdaptiveCpp or DPC++.
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

git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git
echo 'export SPACK_ROOT=$HOME/spack' >> $HOME/.bashrc
echo 'source $SPACK_ROOT/share/spack/setup-env.sh' >> $HOME/.bashrc
export SPACK_ROOT=$HOME/spack
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
GCC/AdaptiveCpp and one with Intel's OneAPI/DPC++. These
packages and their dependencies will be installed in the usual Spack
locations. They will also be linked into ["filesystem
views"](https://spack.readthedocs.io/en/latest/environments.html#filesystem-views)
`view/adaptivecpp` and `view/oneapi-dpcpp`. The NESO builds will be
done in directories called something like `build-arch-abc1234` (the
hashes at the end will differ). If you change your spack installation
in some way (e.g., upgrading the version of a dependency) then the
hash will change and NESO and/or Nektar++ will be rebuilt. The
activation provides the convenience command `cleanup` to delete these
old builds.

In order to tell which NESO build is which, symlinks are generated at
`builds/gcc-<hash>` and `builds/oneapi-<hash>`. Similar links are generated for
Nektar++ and neso-particles in their own `builds` subdirectories. The activation
script also launches a background task which regularly checks whether the hashes
of your NESO, Nektar++ and neso-particles builds have changed. If they have, it
will update the symlinks. They will also be checked whenever the environment is
activated or deactivated.

#### Developing
As you develop the code, there are a few options for how you
recompile. One is simply to run `spack install` again. This will reuse
the existing build directory and reinstall the results of the
build. The build environment used is determined by the package
configuration for NESO (as specificed in the NESO [package
repository](https://github.com/ExCALIBUR-NEPTUNE/NESO-Spack) and is
the same as if you were doing a traditional Spack installation of a
named version of NESO. This has the particular advantage of building
with all toolchains (i.e., GCC, OneAPI) at one time. It also works
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
cmake -DCMAKE_PREFIX_PATH=$(pwd)/adaptivecpp . -B build
cmake --build build
```
CMake will automatically be able to find all of the packages it needs
in `adaptivecpp`. The downside of this approach is that there is a
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
directories are `builds/gcc-<hash>` and `builds/oneapi-<hash>`.

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
- Integration tests: `solvers/test/integration/<solver_name>`
- Examples: `examples/<solver_name>/<example_name>`

To run a solver example:
```
./scripts/run_eg.sh  [solver_name] [example_name] <-n num_MPI> <-b build_dir>
```
which will look for the solver executable in the most recently modified build
directory (or symlink) in `builds`, unless an alternative location is supplied
with `-b`.  Output is generated in `runs/<solver_name>/<example_name>`.

## Address Sanitizers

To debug for memory leaks, compile with the options

```
cmake . -B -DENABLE_SANITIZER_ADDRESS=on -DENABLE_SANITIZER_LEAK=on
```

## Generating meshes

One pipeline for creating simple Nektar++ meshes is through Gmsh and NekMesh.
  .geo -> .msh in Gmsh
  .msh -> .xml with NekMesh

A .geo file can be created using the Gmsh UI (each command adds a new line to the .geo file).  For simple meshes it may be easier to produce the .geo file in a text editor.  .geo files can also be loaded into the UI to edit.

The script at `scripts/geo_to_xml.sh` can be used to convert .geo files to
Nektar++ XML meshes. Execute the script with no arguments to see the available
options. Note that no `<EXPANSIONS>` node is generated; users need to add one
themselves, either to the mesh file itself, or to another xml file that gets
passed to Nektar++. Alternatively, an xml mesh can be generated by running Gmsh
and NekMesh separately, using the instructions below.


An example .geo file is shown below:
<details>
  <summary>Expand</summary>
  mesh.geo
  
  ```cpp
//Create construction points for the corners of the mesh
//Point(i) = (x, y, z, scale)
Point(1) = {0, 0, 0, 1.0};
Point(2) = {0, 1, 0, 1.0};
Point(3) = {1, 1, 0, 1.0};
Point(4) = {1, 0, 0, 1.0};

//Create construction lines for the edges of the mesh
//Line(i) = (Start_i, End_i) 
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

//Add physical lines from the construction lines
Physical Line(5) = {1};
Physical Line(6) = {2};
Physical Line(7) = {3};
Physical Line(8) = {4};

//Close the lines with a curve loop
Curve Loop(1) = {1,2,3,4};
//Create a construction surface out of the curve loop
Plane Surface(1) = {1};
//Create a physical surface out of the construction surface
Physical Surface(1) = {1};

//A transfinite line means that 65 points are created along lines
// 1 and 3, which will form the corners of 64 mesh elements.
//Progression 1 means they are uniformly spaced
Transfinite Line {1, 3} = 65 Using Progression 1;
Transfinite Line {2, 4} = 2 Using Progression 1;
//Set the surface
Transfinite Surface {1};

//The default is tris, this line is necessary to produce quads
Recombine Surface "*";
```

</details>
Loading the above file in the Gmsh GUI, then selecting `2D mesh` and saving will produce a .msh file.
Alternatively for simple meshes one can jump straight to the next step by writing the .msh file directly in a text editor.

<details>
  <summary>Expand</summary>
   mesh.msh
  
  ```
$MeshFormat
4.1 0 8
$EndMeshFormat
$Entities
4 4 1 0
.
.
.
$EndEntities
$Nodes
9 130 1 130
0 1 0 1
.
.
.
$EndNodes
$Elements
5 194 1 194
1 1 1 64
.
.
.
$EndElements
```
</details>

Next call NekMesh on the .msh file to convert it to .xml format

```
./NekMesh path/to/src/mesh.msh path/to/dst/mesh.xml
```
This will produce a .xml file like the following:
<details>
  <summary>Expand</summary>
   mesh.xml
  
```
<NEKTAR>
  <GEOMETRY DIM="2" SPACE="2">
    <VERTEX COMPRESSED="B64Z-LittleEndian" BITSIZE="64">eJx912dsjWEYxvFjz9qUmq3aW+1Zu0VttalVWlup7TwSMT4UH3wQiZFIDoKQVFqCKEWt2tUqpWZRW+1RrqfJfee9Ivrljvz7u5/Tc95z3sPl+v9PPse/3nXlnp96YDDmJvm9AvT7a0/k/v3ZKL3gP/1O6YXI5xyH3yG9MPnrszD3SC9CPugS/G7pRcm3s+cfkl6M/KIj8AelFye/bgzmYeklyFdPgY+VXpJ8mn38R6R7kZ94BT5eeinyvgbzuPTS5D1J8MeklyHfwf79CdLLkvdLgD8pvRz5sCGYp6WXJ18kHf6U9Ark19vn74z0iuS33YRPlF6J/MEpmOeke5NvmQx/Vnpl8in2+T8vvQr5pHPwSdJ9yP+IxrwovSr5sfb5uyC9Gnl/g3lZenXy74/CX5Jeg/yA9ZhXpNckvyYWPll6LfJ51/816b7kvQ/AX5XuRz45BPO69NrkQ+3rp/v9yeddPzek1yHvew1e99clnxOKeVN6PfLZl+F1f33ya+31d0t6A/JxF+B1f0PyPmGYKdIbkTf2+tH9jckfsNfvbelNyPdLhNf9Tcl3j8BMld6MfCV7/en+5uRT7fWfJr0F+Uz7+av7W5KPnI95R3oA+X32+tX9rcj/su+fdOmtyS+Mg9f9bchvXIZ5V3pb8oH2+tf97cj7G8x70tuTL3oIXvd3IB+/GjNDekfyKfvhdX8n8n3t+/e+9M7kt++F1/1dyD+IwXwgnW/YkR543R9IPjoIM1N6N/JfdsHr/u7kI3r9HUZ9D/JeeP+7tfckv8Ue+FB6L/LmFs5X35v8eTxhRn0f8h9x/3VrDyL/1b7gj6QHkw9Pxfnq+5KvjwvSqO9HPj0N52vvT36kfUM9lh5CPsTeP9UPIL8OHwhG/UDyCXdxvvZB5OPtB9oT6YPJB2TgfPVDyGfhA9moH0recx/nax9G3tveMJ5KH07eJxPnqw8l3xs3PKN+BPmYhzhf+0jy0faG/Uz6KPKuxzhf/WjyHnzhMOrHEI96gvO1jyV/235hypI+jvyzpzhf/XjyhfGFzqifQH50Fs7XHka+jf1C+lz6RPLJz3G++knkp+ILtVE/mXzgS5yvfQr5zfY/BC+kTyUfm43z1YeTT8Qw6qeRr/sa52ufTv7DT+zXHkF+6xtnjyTv9w37X0qfQd7rnbPPJD/4E/Zrn0XevHf22eRtd2dLn0P+4wdnn0s+yv592ueRD89x9vnkv7/F/lfSo8inf3L2BeR74vHlal9IPuSLs0eT34DH534tfRH5hK/Ovph82mfs176EfMB3Z19KPu/1fyN9GXnPD2dfTn4meq72FeR9fjn7SvJx6O630t3kY347uyHvnC7XKte/frT/AaQ8q4UA</VERTEX>
    <EDGE COMPRESSED="B64Z-LittleEndian" BITSIZE="64">eJx1j1VXFlEARbG7i1BRQhAFgxBUQpBSVDCwaFAwABWwQezu7u7ujp/mg+c8uNf65mWvWfvsuXeCgv49XYL+f/zeNcB7twDvQXjvDnrfA7Tvid6+F7z3vQP4PgH6vui874fOvj96+wEBvjcQ37MfhN5+MM7xfgjOsR+K3n4Yzvd+OM63H4HefiTu5f0o3Ms+GL19CO7rfSjuax+G3n40/sP7MfgP+7Ho7cPFkdiPE0fBj0dvHyEGYx8phsBHobePFkOxnyCGwcegt48VR2M/URwDH4fefpI4FvvJYjh8PHr7BHEc9lPE8fBT0dtPEyOwny5Gwieit08So7BPFqPhU9DbzxAnYJ8qxsCnobefKcZiP0ucCD8bvX26GId9hjgJPhO9fZY4Gfs5Yjx8Nnr7HDEB+7niFPhc9PZ54lTs88Vp8AXo7QvF6djPExPh56O3LxKTsF8gJsMvRG+/SEzBvlicAV+C3n6xmIr9EjENfil6+2XiTOxLxVnwy9HbrxBnY79STIdfhd5+tZiBfZmYCV+O3r5CzMK+UpwDX4XevlrMxr5GzIGvRW9fJ87Ffo2YC78WvX29mId9g5gPvw69/XqxAPsNYiH8RvT2jeI87JvE+fDN6O03iUXYbxYXwG9Bb98iLsS+VVwE34befqtYjP02sQR+O3r7HeJi7HeKS+B3obffLS7Fvl1cBt+B3n6PWIp9p7gcfi96+33iCuz3iyvhD6C3Pyiuwv6QuBr+MHr7I2IZ9kfFcvhj6O2PixXYnxAr4U+itz8lVmF/WqyGP4Pe/qxYg/05sRb+PHr7C2Id9hfFNfCX0NtfFtdif0Wsh7+K3v6a2ID9dXEd/A309jfF9djfEjfA30Zvf0fciP1dsRH+Hnr7+2IT9g/EZviH6O0fiZuwfyxuhn+C3v6puAX7Z2IL/HP09i/EVuxfim3wr9Dbvxa3Yv9G3Ab/Fr39O3E79u/FHfAf0Nt/FHdi/0ncBf8Zvf0XcTf2X8V2+G/o7b+LHdj/EPfA/0Rv/0vsxP63uBf+D3r7v21uqaIA</EDGE>
    <ELEMENT>
      <Q COMPRESSED="B64Z-LittleEndian" BITSIZE="64">eJx1zuc30GEYh3EkZe+99957k5REA1EaGtpZlbIpKqSFtFRaEmkplSR/mjfX74XvOT1vPudc9znPfZuYrH+maIYb/tPNcSNayNzom3AzWsq/RrdCa7SRPUa3RTu0l31Gd0BHdJI7je6MLugqdxvdDd3RQ+43uid6oTdaSvdBX/RDK+n+GICBaC09CIMxBG2kh2IYhqOt9AiMxCi0kx6NMRiL9tLjMB4T0EF6IiZhMjpKT8FUTEMn6emYgZnoLD0LszEHXaTnYh7mo6v0AizELegmvQi3YjG6S9+G27EEPaTvwFLciZ7Sy7Acd6GX9N24B/eit/QKrMQq9JG+D6uxBn2l78cDWIt+0g/iITyM/tKPYB0exQDpx/A4nsBA6fV4Ek9hkPTTeAbPYrD0c3geL2CI9AZsxCYMld6MLXgRw6RfwsvYiuHSr+BVbMMI6e3YgZ0YKb0Lu7EHo6T3Yh9ew2jp17EfBzBG+g28ibcwVvogDuEwxkm/jSN4B+Ol38V7eB8TpD/AURzDROnj+BAnMEn6I3yMTzBZ+lN8hpOYIv05vsCXmCp9Cl/ha0yT/gbf4jtMlz6N73EGM6R/wFmcw0zpH3EeP2GW9M/4Bb9itvRvuIDfMUf6D1zEn5gr/Rf+xiXMk/4Hl/Ev5ktfwX+4imuOL2ei</Q>
    </ELEMENT>
    <COMPOSITE>
      <C ID="1"> Q[0-63] </C>
      <C ID="5"> E[3,6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,66,69,72,75,78,81,84,87,90,93,96,99,102,105,108,111,114,117,120,123,126,129,132,135,138,141,144,147,150,153,156,159,162,165,168,171,174,177,180,183,186,189,192] </C>
      <C ID="6"> E[191] </C>
      <C ID="7"> E[190,187,184,181,178,175,172,169,166,163,160,157,154,151,148,145,142,139,136,133,130,127,124,121,118,115,112,109,106,103,100,97,94,91,88,85,82,79,76,73,70,67,64,61,58,55,52,49,46,43,40,37,34,31,28,25,22,19,16,13,10,7,4,1] </C>
      <C ID="8"> E[0] </C>
    </COMPOSITE>
    <DOMAIN>
<D ID="0"> C[1] </D>
    </DOMAIN>
  </GEOMETRY>
<Metadata>
  <Provenance>
    <GitBranch>refs/heads/master</GitBranch>
    <GitSHA1>X</GitSHA1>
    <Hostname>X</Hostname>
    <NektarVersion>5.5.0</NektarVersion>
    <Timestamp>T</Timestamp>
  </Provenance>
  <NekMeshCommandLine>path/to/src/mesh.msh path/to/dst/mesh.xml </NekMeshCommandLine>
</Metadata>
  <EXPANSIONS>
    <E COMPOSITE="C[1]" NUMMODES="4" TYPE="MODIFIED" FIELDS="u"/>
  </EXPANSIONS>
</NEKTAR>
```
</details>

Note the edge composites run in opposite directions.  To align them to the same direction use the NekMesh module peralign as follows:

```
./NekMesh -m peralign:surf1=5:surf2=7:dir=x path/to/src/mesh.xml path/to/dst/mesh_aligned.xml
```

## Licence

This is licenced under MIT.

In order to comply with the licences of dependencies, this software is not to be released as a binary.

