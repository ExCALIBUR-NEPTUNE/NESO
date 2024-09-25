#include <benchmark/benchmark.h>
#include <sycl/sycl.hpp>

namespace Nektar::LibUtilities {
enum ShapeType {
  eNoShapeType,
  ePoint,
  eSegment,
  eTriangle,
  eQuadrilateral,
  eTetrahedron,
  ePyramid,
  ePrism,
  eHexahedron
};
}
#include <nektar_interface/projection/shapes.hpp>
#include <nektar_interface/projection/algorithm_types.hpp>
#include <nektar_interface/projection/auto_switch.hpp>
#include <nektar_interface/projection/constants.hpp>
#include <nektar_interface/projection/device_data.hpp>
#include "include/create_data.hpp"
#include "include/args.hpp"

static CmdArgs args;

//mock the nektar shapes (not really used anyway)



using namespace NESO::Project;

template<int nmode, typename Shape>
void
bm_project(benchmark::State &state) {
	auto Q = sycl::queue{sycl::default_selector_v};
	auto [data, ptrs] = create_data<nmode,Shape>(Q, args.ncell, args.min_per_cell, args.max_per_cell);
    
    for(auto _: state)	
	{
		Shape::algorithm::template project<nmode,double,1,1,Shape>(data,0,Q).wait();
	}
	free_data(Q,ptrs);	
}

BENCHMARK(bm_project<3,eQuad<ThreadPerCell2D>>);
BENCHMARK(bm_project<4,eQuad<ThreadPerCell2D>>);
BENCHMARK(bm_project<5,eQuad<ThreadPerCell2D>>);
BENCHMARK(bm_project<6,eQuad<ThreadPerCell2D>>);
BENCHMARK(bm_project<7,eQuad<ThreadPerCell2D>>);
BENCHMARK(bm_project<8,eQuad<ThreadPerCell2D>>);

BENCHMARK(bm_project<3,eTriangle<ThreadPerCell2D>>);
BENCHMARK(bm_project<4,eTriangle<ThreadPerCell2D>>);
BENCHMARK(bm_project<5,eTriangle<ThreadPerCell2D>>);
BENCHMARK(bm_project<6,eTriangle<ThreadPerCell2D>>);
BENCHMARK(bm_project<7,eTriangle<ThreadPerCell2D>>);
BENCHMARK(bm_project<8,eTriangle<ThreadPerCell2D>>);

BENCHMARK(bm_project<3,eTet<ThreadPerCell3D>>);
BENCHMARK(bm_project<4,eTet<ThreadPerCell3D>>);
BENCHMARK(bm_project<5,eTet<ThreadPerCell3D>>);
BENCHMARK(bm_project<6,eTet<ThreadPerCell3D>>);
BENCHMARK(bm_project<7,eTet<ThreadPerCell3D>>);
BENCHMARK(bm_project<8,eTet<ThreadPerCell3D>>);

BENCHMARK(bm_project<3,eHex<ThreadPerCell3D>>);
BENCHMARK(bm_project<4,eHex<ThreadPerCell3D>>);
BENCHMARK(bm_project<5,eHex<ThreadPerCell3D>>);
BENCHMARK(bm_project<6,eHex<ThreadPerCell3D>>);
BENCHMARK(bm_project<7,eHex<ThreadPerCell3D>>);
BENCHMARK(bm_project<8,eHex<ThreadPerCell3D>>);

BENCHMARK(bm_project<3,ePyramid<ThreadPerCell3D>>);
BENCHMARK(bm_project<4,ePyramid<ThreadPerCell3D>>);
BENCHMARK(bm_project<5,ePyramid<ThreadPerCell3D>>);
BENCHMARK(bm_project<6,ePyramid<ThreadPerCell3D>>);
BENCHMARK(bm_project<7,ePyramid<ThreadPerCell3D>>);
BENCHMARK(bm_project<8,ePyramid<ThreadPerCell3D>>);

BENCHMARK(bm_project<3,ePrism<ThreadPerCell3D>>);
BENCHMARK(bm_project<4,ePrism<ThreadPerCell3D>>);
BENCHMARK(bm_project<5,ePrism<ThreadPerCell3D>>);
BENCHMARK(bm_project<6,ePrism<ThreadPerCell3D>>);
BENCHMARK(bm_project<7,ePrism<ThreadPerCell3D>>);
BENCHMARK(bm_project<8,ePrism<ThreadPerCell3D>>);

BENCHMARK(bm_project<3,eQuad<ThreadPerDof2D>>);
BENCHMARK(bm_project<4,eQuad<ThreadPerDof2D>>);
BENCHMARK(bm_project<5,eQuad<ThreadPerDof2D>>);
BENCHMARK(bm_project<6,eQuad<ThreadPerDof2D>>);
BENCHMARK(bm_project<7,eQuad<ThreadPerDof2D>>);
BENCHMARK(bm_project<8,eQuad<ThreadPerDof2D>>);

BENCHMARK(bm_project<3,eTriangle<ThreadPerDof2D>>);
BENCHMARK(bm_project<4,eTriangle<ThreadPerDof2D>>);
BENCHMARK(bm_project<5,eTriangle<ThreadPerDof2D>>);
BENCHMARK(bm_project<6,eTriangle<ThreadPerDof2D>>);
BENCHMARK(bm_project<7,eTriangle<ThreadPerDof2D>>);
BENCHMARK(bm_project<8,eTriangle<ThreadPerDof2D>>);

BENCHMARK(bm_project<3,eTet<ThreadPerDof3D>>);
BENCHMARK(bm_project<4,eTet<ThreadPerDof3D>>);
BENCHMARK(bm_project<5,eTet<ThreadPerDof3D>>);
BENCHMARK(bm_project<6,eTet<ThreadPerDof3D>>);
BENCHMARK(bm_project<7,eTet<ThreadPerDof3D>>);
BENCHMARK(bm_project<8,eTet<ThreadPerDof3D>>);

BENCHMARK(bm_project<3,eHex<ThreadPerDof3D>>);
BENCHMARK(bm_project<4,eHex<ThreadPerDof3D>>);
BENCHMARK(bm_project<5,eHex<ThreadPerDof3D>>);
BENCHMARK(bm_project<6,eHex<ThreadPerDof3D>>);
BENCHMARK(bm_project<7,eHex<ThreadPerDof3D>>);
BENCHMARK(bm_project<8,eHex<ThreadPerDof3D>>);

BENCHMARK(bm_project<3,ePyramid<ThreadPerDof3D>>);
BENCHMARK(bm_project<4,ePyramid<ThreadPerDof3D>>);
BENCHMARK(bm_project<5,ePyramid<ThreadPerDof3D>>);
BENCHMARK(bm_project<6,ePyramid<ThreadPerDof3D>>);
BENCHMARK(bm_project<7,ePyramid<ThreadPerDof3D>>);
BENCHMARK(bm_project<8,ePyramid<ThreadPerDof3D>>);

BENCHMARK(bm_project<3,ePrism<ThreadPerDof3D>>);
BENCHMARK(bm_project<4,ePrism<ThreadPerDof3D>>);
BENCHMARK(bm_project<5,ePrism<ThreadPerDof3D>>);
BENCHMARK(bm_project<6,ePrism<ThreadPerDof3D>>);
BENCHMARK(bm_project<7,ePrism<ThreadPerDof3D>>);
BENCHMARK(bm_project<8,ePrism<ThreadPerDof3D>>);

int
main(int argc, char **argv)
{
	args = get_args(argc, argv, true);
    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();
}
