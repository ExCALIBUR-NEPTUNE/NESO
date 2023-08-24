#include "nektar_interface/particle_interface.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <nektar_interface/utilities.hpp>
#include <particle_utility/particle_initialisation_line.hpp>

using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

// Test advecting particles between ranks
TEST(ParticleInitialisationLine, Points) {

  const double tol = 1.0e-10;
  int argc = 2;
  char *argv[2];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  std::filesystem::path mesh_file =
      test_resources_dir / "square_triangles_quads.xml";
  copy_to_cstring(std::string(mesh_file), &argv[1]);

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);
  graph = SpatialDomains::MeshGraph::Read(session);

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 2;

  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("POINT_RANK"), 1),
                             ParticleProp(Sym<INT>("POINT_INDEX"), 1),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);
  CellIDTranslation cell_id_translation(sycl_target, A->cell_id_dat, mesh);

  std::vector<double> line_start = {pbc.global_origin[0], pbc.global_origin[1]};
  std::vector<double> line_end = {pbc.global_origin[0] + pbc.global_extent[0],
                                  pbc.global_origin[1] + pbc.global_extent[1]};

  // Create a line initialisation object
  const int npoints_total = 1000;
  auto line_initialisation = std::make_shared<ParticleInitialisationLine>(
      domain, sycl_target, line_start, line_end, npoints_total);

  auto lambda_line_eq = [&](const double x) {
    const double x0 = pbc.global_origin[0];
    const double y0 = pbc.global_origin[1];
    const double x1 = pbc.global_origin[0] + pbc.global_extent[0];
    const double y1 = pbc.global_origin[1] + pbc.global_extent[1];
    const double a = (y1 - y0) / (x1 - x0);
    const double b = y0 - a * x0;
    const double y = a * x + b;
    return y;
  };

  Array<OneD, NekDouble> coord_phys{3};
  Array<OneD, NekDouble> coord_ref{3};

  for (int px = 0; px < line_initialisation->npoints_local; px++) {

    const double ref_p0 = line_initialisation->point_ref_positions[0][px];
    const double ref_p1 = line_initialisation->point_ref_positions[1][px];
    const double p0 = line_initialisation->point_phys_positions[0][px];
    const double p1 = line_initialisation->point_phys_positions[1][px];

    // is this point actually on a line?
    const double y_test = lambda_line_eq(p0);
    ASSERT_NEAR(y_test, p1, 1.0e-6);

    // Does the nektar++ cell agree that it contains this point
    const int neso_cell = line_initialisation->point_neso_cells[px];
    const int nektar_geom_id = cell_id_translation.map_to_nektar[neso_cell];
    coord_phys[0] = p0;
    coord_phys[1] = p1;
    coord_ref[0] = 0.0;
    coord_ref[1] = 0.0;
    coord_ref[2] = 0.0;
    auto geom = graph->GetGeometry2D(nektar_geom_id);
    auto geom_found = geom->ContainsPoint(coord_phys, coord_ref, 1.0e-8);

    ASSERT_TRUE(geom_found);
    ASSERT_NEAR(ref_p0, coord_ref[0], 1.0e-10);
    ASSERT_NEAR(ref_p1, coord_ref[1], 1.0e-10);
  }

  A->free();
  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
}
