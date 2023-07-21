Cppyy is a library for automatically generating python bindings for C++ code. This allows us to call any function from NESO in python code.
As an example, the whole of NESO main is replicated in `cppyy_run.py`.

Cppyy requires llvm / clang to build. I've had problems setting up the build
environment automatically, but this set up works for me.

## Activate the spack environment

```
git submodule update --init
. activate
```

## Set up the Python environment

Install the python dependencies in a virtual environment by doing

```
python3 -m venv venv_cppyy
. venv_cppyy/bin/activate
pip3 install -r requirements
```

This will install the dependencies:

* cppyy==2.4.2
* libclang
* clang-format
* setuptools
* wheel

# Install the spack environment

Set up the `spack.yaml` to the supported toolchain:

```
spack:
  specs:
  - neso%clang@12.0.1 ^openblas ^hipsycl ^scotch@6 ^llvm@12.0.1 ^python@3.9.12%gcc@11.2.0
  - hipsycl@0.9.2%gcc@11.2.0 ^llvm@12.0.1%gcc@11.2.0
  - llvm@12.0.1%gcc@11.2.0 +python +clang
  - python@3.9.12%gcc@11.2.0
  - gcc@11.2.0
```

These versions are pinned to what works for me - other versions may work too.

Install the environment with

```
spack concretize -f
spack install
```

This command might fail if the clang compiler is not installed. 
In that case, try replacing the first line with

```
spack:
  specs:
  - neso%clang@12.0.1 ^openblas ^hipsycl ^scotch@6 ^llvm@12.0.1 ^python@3.9.12%gcc@11.2.0
```

do `spack concretize -f; spack install`, and then register the clang compiler with spack by doing

```
spack load llvm@12.0.1%gcc@11.2.0
spack compiler find
```

then reinstating the first line and doing `spack concretize -f; spack install` again.

This will probably fail too... 

# Building manually

To build manually, do 

```
spack build-env neso%clang cmake . -B build_new -DCPPYY_MODULE_PATH=venv_cppyy/lib/python3.9/site-packages/cppyy_backend/cmake/
spack build-env neso%clang cmake --build build_new -j 1 -v
```

This will fail with errors that look like:
```
In file included from input_line_3:2:
In file included from /home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-12.2.0/gcc-11.2.0-54m6goknrjplkcc6tvobxnijkssbx4qg/lib/gcc/x86_64-pc-linux-gnu/11.2.0/../../../../include/c++/11.2.0/string:40:
In file included from /home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-12.2.0/gcc-11.2.0-54m6goknrjplkcc6tvobxnijkssbx4qg/lib/gcc/x86_64-pc-linux-gnu/11.2.0/../../../../include/c++/11.2.0/bits/char_traits.h:699:
/home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-12.2.0/gcc-11.2.0-54m6goknrjplkcc6tvobxnijkssbx4qg/lib/gcc/x86_64-pc-linux-gnu/11.2.0/../../../../include/c++/11.2.0/cstdint:47:11: error: no member named 'int8_t' in the global namespace
  using ::int8_t;
```

Copy the last line from the verbose build and execute on the command line.
For me, this is:
```
cd /home/jparker/code/NESO/build_new/PyNESO && /home/jparker/code/NESO/venv_3_9_12/bin/rootcling -f /home/jparker/code/NESO/build_new/PyNESO.cpp -s PyNESO -rmf /home/jparker/code/NESO/build_new/PyNESO/PyNESO.rootmap -rml libPyNESOCppyy.so -cxxflags='-DNESO_HIPSYCL -I /home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-11.2.0/hipsycl-0.9.2-nfvpn6i7t3b25zbzgw3lpx35yuo3khbq/include/ -DBINDING_BUILD=on -pthread -std=c++1z -m64 -I/home/jparker/code/NESO/venv_3_9_12/lib/python3.9/site-packages/cppyy_backend/include -std=c++17 -I/home/jparker/code/NESO -I/home/jparker/code/NESO/include -I -I/home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-11.2.0/hipsycl-0.9.2-nfvpn6i7t3b25zbzgw3lpx35yuo3khbq/include/ -I/home/jparker/code/NESO/include -I/home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-11.2.0/fftw-3.3.10-u6j4gsbk2z3s6iosn32aw3crgsitva5v/include' /home/jparker/code/NESO/include/custom_types.hpp /home/jparker/code/NESO/include/diagnostics.hpp /home/jparker/code/NESO/include/fft_fftw.hpp /home/jparker/code/NESO/include/fft_mkl.hpp /home/jparker/code/NESO/include/fft_wrappers.hpp /home/jparker/code/NESO/include/mesh.hpp /home/jparker/code/NESO/include/plasma.hpp /home/jparker/code/NESO/include/simulation.hpp /home/jparker/code/NESO/include/species.hpp /home/jparker/code/NESO/include/velocity.hpp /home/jparker/code/NESO/build_new/linkdef.h
```

This works. Then do

```
cd ../..
spack build-env neso%clang cmake --build build_new -j 1 -v
```

This fails with error like:

```
ERROR: Unknown container kind CursorKind.OMP_PARALLEL_MASKED_DIRECTIVE
Traceback (most recent call last):
  File "/home/jparker/.local/lib/python3.9/site-packages/cppyy_backend/_cppyy_generator.py", line 737, in main
    json.dump(mapping, f, indent=1, sort_keys=True)
  File "/home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-11.2.0/python-3.9.12-zoa7syazdqtlpe7bhp735modrmwwz27r/lib/python3.9/json/__init__.py", line 179, in dump
    for chunk in iterable:
  File "/home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-11.2.0/python-3.9.12-zoa7syazdqtlpe7bhp735modrmwwz27r/lib/python3.9/json/encoder.py", line 429, in _iterencode
    yield from _iterencode_list(o, _current_indent_level)
  File "/home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-11.2.0/python-3.9.12-zoa7syazdqtlpe7bhp735modrmwwz27r/lib/python3.9/json/encoder.py", line 325, in _iterencode_list
    yield from chunks
  File "/home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-11.2.0/python-3.9.12-zoa7syazdqtlpe7bhp735modrmwwz27r/lib/python3.9/json/encoder.py", line 405, in _iterencode_dict
    yield from chunks
  File "/home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-11.2.0/python-3.9.12-zoa7syazdqtlpe7bhp735modrmwwz27r/lib/python3.9/json/encoder.py", line 438, in _iterencode
    o = _default(o)
  File "/home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-11.2.0/python-3.9.12-zoa7syazdqtlpe7bhp735modrmwwz27r/lib/python3.9/json/encoder.py", line 179, in default
    raise TypeError(f'Object of type {o.__class__.__name__} '
TypeError: Object of type CursorKind is not JSON serializable

gmake[2]: *** [PyNESO/PyNESO.map] Error 1
gmake[2]: *** Deleting file `PyNESO/PyNESO.map'
gmake[2]: Leaving directory `/home/jparker/code/NESO/build_new'
gmake[1]: *** [CMakeFiles/PyNESOCppyy.dir/all] Error 2
gmake[1]: Leaving directory `/home/jparker/code/NESO/build_new'
gmake: *** [all] Error 2
```

Again execute the last displayed command, for me:

```
cd /home/jparker/code/NESO/build_new/PyNESO && python3 /home/jparker/code/NESO/venv_3_9_12/bin/cppyy-generator --libclang /home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-11.2.0/llvm-12.0.1-oomlmdehchcg455mfixdplvmluxogicz/lib/libclang.so --flags "\-DNESO_HIPSYCL;\-I;/home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-11.2.0/hipsycl-0.9.2-nfvpn6i7t3b25zbzgw3lpx35yuo3khbq/include/;\-DBINDING_BUILD=on;\-pthread;\-std=c++1z;\-m64;\-I/home/jparker/code/NESO/venv_3_9_12/lib/python3.9/site-packages/cppyy_backend/include;\-std=c++17;\-I/home/jparker/code/NESO;\-I/home/jparker/code/NESO/include;\-I;\-I/home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-11.2.0/hipsycl-0.9.2-nfvpn6i7t3b25zbzgw3lpx35yuo3khbq/include/;\-I/home/jparker/code/NESO/include;\-I/home/jparker/spack/opt/spack/linux-centos7-haswell/gcc-11.2.0/fftw-3.3.10-u6j4gsbk2z3s6iosn32aw3crgsitva5v/include" /home/jparker/code/NESO/build_new/PyNESO/PyNESO.map /home/jparker/code/NESO/include/custom_types.hpp /home/jparker/code/NESO/include/diagnostics.hpp /home/jparker/code/NESO/include/fft_fftw.hpp /home/jparker/code/NESO/include/fft_mkl.hpp /home/jparker/code/NESO/include/fft_wrappers.hpp /home/jparker/code/NESO/include/mesh.hpp /home/jparker/code/NESO/include/plasma.hpp /home/jparker/code/NESO/include/simulation.hpp /home/jparker/code/NESO/include/species.hpp /home/jparker/code/NESO/include/velocity.hpp
```

This works. Again do

```
cd ../..
spack build-env neso%clang cmake --build build_new -j 1 -v
```

This fails with

```
[ 72%] Generating dist/PyNESO-0.0.1-py3-none-linux_x86_64.whl
python3 setup.py bdist_wheel
Traceback (most recent call last):
  File "/home/jparker/code/NESO/build_new/setup.py", line 7, in <module>
    import setuptools
ModuleNotFoundError: No module named 'setuptools'
gmake[2]: *** [dist/PyNESO-0.0.1-py3-none-linux_x86_64.whl] Error 1
gmake[2]: Leaving directory `/home/jparker/code/NESO/build_new'
gmake[1]: *** [CMakeFiles/wheel.dir/all] Error 2
gmake[1]: Leaving directory `/home/jparker/code/NESO/build_new'
gmake: *** [all] Error 2
```

Now do

```
cd build_new
python3 setup.py bdist_wheel
cd ..
spack build-env neso%clang cmake --build build_new -j 1 -v
```

which now should finish successfully.

## Executing the code

To run the code, do

```
python3 cppyy_run.py
```

This will execute the python mock up of NESO's main, and should finish with the same final values:

```
0.882496 0.75386 0.128635
```
