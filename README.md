This is a test implmentation of a PIC solver for 1+1D Vlasov Poisson, written
in C++/DPC++.
This is primarily designed to test the use of multiple repos/workflows for
different code components.

## Testing

Run tests by doing

```
cmake -S -B build
cmake --build build
cd build && ctest
```
