#include "include/args.hpp"
#include "include/create_data.hpp"
#include <benchmark/benchmark.h>
#include <nektar_interface/projection/algorithm_types.hpp>
#include <nektar_interface/projection/device_data.hpp>
#include <nektar_interface/projection/shapes.hpp>
#include <neso_constants.hpp>
#include <neso_particles/sycl_typedefs.hpp>

static CmdArgs args;

// mock the nektar shapes (not really used anyway)

using namespace NESO::Project;

template <int nmode, typename T, typename Shape>
void bm_project(benchmark::State &state) {
  auto Q = sycl::queue{sycl::default_selector_v};
  auto [data, ptrs] = create_data<nmode, T, Shape>(
      Q, args.ncell, args.min_per_cell, args.max_per_cell);
  std::optional<sycl::event> ev;
  for (auto _ : state) {
    if ((ev = Shape::algorithm::template project<nmode, T, 1, 1, Shape, NoFilter>(data, 0,
                                                                        Q))) {
      ev.value().wait();
    } else {
      state.SkipWithError("Projection failed");
      break;
    }
  }
  free_data(Q, ptrs);
}


#define MAKE_BENCH_SET(type, shape, alg)                                       \
  BENCHMARK(bm_project<3, type, shape<alg>>);                                  \
  BENCHMARK(bm_project<4, type, shape<alg>>);                                  \
  BENCHMARK(bm_project<5, type, shape<alg>>);                                  \
  BENCHMARK(bm_project<6, type, shape<alg>>);                                  \
  BENCHMARK(bm_project<7, type, shape<alg>>);                                  \
  BENCHMARK(bm_project<8, type, shape<alg>>);

MAKE_BENCH_SET(double, eQuad, ThreadPerCell);
MAKE_BENCH_SET(double, eTriangle, ThreadPerCell);
MAKE_BENCH_SET(double, eHex, ThreadPerCell);
MAKE_BENCH_SET(double, ePyramid, ThreadPerCell);
MAKE_BENCH_SET(double, ePrism, ThreadPerCell);
MAKE_BENCH_SET(double, eTet, ThreadPerCell);

MAKE_BENCH_SET(float, eQuad, ThreadPerCell);
MAKE_BENCH_SET(float, eTriangle, ThreadPerCell);
MAKE_BENCH_SET(float, eHex, ThreadPerCell);
MAKE_BENCH_SET(float, ePyramid, ThreadPerCell);
MAKE_BENCH_SET(float, ePrism, ThreadPerCell);
MAKE_BENCH_SET(float, eTet, ThreadPerCell);

MAKE_BENCH_SET(double, eQuad, ThreadPerDof);
MAKE_BENCH_SET(double, eTriangle, ThreadPerDof);
MAKE_BENCH_SET(double, eHex, ThreadPerDof);
MAKE_BENCH_SET(double, ePyramid, ThreadPerDof);
MAKE_BENCH_SET(double, ePrism, ThreadPerDof);
MAKE_BENCH_SET(double, eTet, ThreadPerDof);

MAKE_BENCH_SET(float, eQuad, ThreadPerDof);
MAKE_BENCH_SET(float, eTriangle, ThreadPerDof);
MAKE_BENCH_SET(float, eHex, ThreadPerDof);
MAKE_BENCH_SET(float, ePyramid, ThreadPerDof);
MAKE_BENCH_SET(float, ePrism, ThreadPerDof);
MAKE_BENCH_SET(float, eTet, ThreadPerDof);

int main(int argc, char **argv) {
  args = get_args(argc, argv, true);

  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
}
