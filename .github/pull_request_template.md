# Description

Please include a summary of changes, motivation and context for this PR.

Fixes # (issue)

## Type of change

Please delete options that are not relevant.

- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] API change (breaks downstream workflows and requires major semver bump)
- [ ] Requires documentation updates

# Testing

Please describe the tests that you ran to verify your changes and provide instructions for reproducibility. Please also list any relevant details for your test configuration.

- [ ] Test Foo in /test/path/to/file_for_test_Foo.cpp
 - Description of test Foo
- [ ] Test Bar in /test/path/to/file_for_test_Bar.cpp
 - Description of test Bar

**Test Configuration**:

* OS:
* SYCL implementation:
* MPI details:
* Hardware:

# Checklist:

- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any new dependencies are automatically built for users via `cmake`
- [ ] I have used understandable variable names
- [ ] I have run `clang-format` against my `*.hpp` and `*.cpp` changes
- [ ] I have run `cmake-format` against my changes to `CMakeLists.txt`
- [ ] I have run `black` against changes to `*.py`
- [ ] I have made corresponding changes to the documentation
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings

######### DELETE THIS AND BELOW #########

# How-to:

1. Format C++ source code with clang-format

```
find ./src ./include ./test -iname \*.hpp -o -iname \*.cpp | xargs clang-format -i
```

2. Format cmake files with cmake-format

```
cmake-format -i CMakeLists.txt src/nektar/CMakeLists.txt test/CMakeLists.txt
```

3. Format python with black

```
find ./python -iname \*.py | xargs black
```

