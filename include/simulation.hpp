#include <iostream>

#include "mesh.hpp"
#include "plasma.hpp"
#include "diagnostics.hpp"

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

void initialize();
void evolve(sycl::queue &q, Mesh &mesh, Plasma &plasma, FFT &fft, Diagnostics &diagnostics);
