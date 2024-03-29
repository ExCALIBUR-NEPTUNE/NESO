#include "diagnostics.hpp"
#include "fft_wrappers.hpp"
#include "mesh.hpp"
#include "plasma.hpp"
#include "revision.hpp"
#include "run_info.hpp"
#include "simulation.hpp"
#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <iostream>
#include <string>

// Value passed at compile time
// Used for REVISION
#define BUILDFLAG1_(x) #x
#define BUILDFLAG(x) BUILDFLAG1_(x)

int main() {

  try {
    auto asyncHandler = [&](sycl::exception_list exceptionList) {
      for (auto &e : exceptionList) {
        std::rethrow_exception(e);
      }
    };
    auto Q = sycl::queue{sycl::default_selector{}, asyncHandler};

    RunInfo run_info(Q, NESO::version::revision, NESO::version::git_state);

    // initialize();
    // Initialize by calling Mesh and Particle constructors
    Mesh mesh(128, 0.05, 40);
    Species ions(mesh, false, 2.0, -1, 1836.2, 1);
    Species electrons(mesh, true, 2.0, 1, 1, 12800);
    std::vector<Species> species_list;
    species_list.push_back(ions);
    species_list.push_back(electrons);
    Plasma plasma(species_list);

    Diagnostics diagnostics;
    FFT fft(Q, mesh.nintervals);

    mesh.set_initial_field(Q, mesh, plasma, fft);
    evolve(Q, mesh, plasma, fft, diagnostics);
  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

  return 0;
}
