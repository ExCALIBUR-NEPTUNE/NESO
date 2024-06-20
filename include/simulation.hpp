#include <iostream>

#include "diagnostics.hpp"
#include "mesh.hpp"
#include "plasma.hpp"
#include "sycl_typedefs.hpp"

void initialize();
void evolve(sycl::queue &q, Mesh &mesh, Plasma &plasma, FFT &fft,
            Diagnostics &diagnostics);
