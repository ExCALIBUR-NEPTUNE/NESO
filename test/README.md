To build:

```
cmake -DCMAKE_CXX_COMPILER=icpx . -B build
cmake --build build
```

To run the tests:

```
cd build
ctest
```
