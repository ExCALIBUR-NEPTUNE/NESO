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
      std::make_shared<NektarGraphLocalMapperT>(sycl_target, mesh, tol);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 2;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("POINT_RANK"), 1),
                             ParticleProp(Sym<INT>("POINT_INDEX"), 1),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);
  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);

  CellIDTranslation cell_id_translation(sycl_target, A->cell_id_dat, mesh);
  const int cell_count = domain->mesh->get_cell_count();

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::vector<double> line_start = {pbc.global_origin[0], pbc.global_origin[1]};
  std::vector<double> line_end = {pbc.global_origin[0] + pbc.global_extent[0],
                                  pbc.global_origin[1] + pbc.global_extent[1]};

  // Create a line initialisation object
  const int npoints_total = 1000;
  auto line_initialisation = std::make_shared<ParticleInitialisationLine>(
      domain, sycl_target, line_start, line_end, npoints_total);

  // create a particle at each point on the line
  const int N = line_initialisation->npoints_local;
  if (N > 0) {
    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_point =
            line_initialisation->point_phys_positions[dimx][px];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_point;
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
      initial_distribution[Sym<INT>("POINT_RANK")][px][0] = rank;
      initial_distribution[Sym<INT>("POINT_INDEX")][px][0] = px;
    }
    A->add_particles_local(initial_distribution);
  }

  // distribute the particles to where neso/neso-particles thinks they live
  A->hybrid_move();
  cell_id_translation.execute();
  A->cell_move();

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

  // check the particle cell/reference position matches the point
  for (int cellx = 0; cellx < cell_count; cellx++) {

    auto positions = A->position_dat->cell_dat.get_cell(cellx);
    auto cell_ids = A->cell_id_dat->cell_dat.get_cell(cellx);
    auto point_indices =
        (*A)[Sym<INT>("POINT_INDEX")]->cell_dat.get_cell(cellx);
    auto point_ranks = (*A)[Sym<INT>("POINT_RANK")]->cell_dat.get_cell(cellx);
    auto ref_positions =
        (*A)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]->cell_dat.get_cell(cellx);

    for (int rowx = 0; rowx < cell_ids->nrow; rowx++) {
      const int point_rank = (*point_ranks)[0][rowx];
      ASSERT_EQ(point_rank, rank);
      const int point_index = (*point_indices)[0][rowx];
      ASSERT_EQ(cellx, line_initialisation->point_neso_cells[point_index]);

      const double px = (*positions)[0][rowx];
      const double py = (*positions)[1][rowx];
      const double point_px =
          line_initialisation->point_phys_positions[0][point_index];
      const double point_py =
          line_initialisation->point_phys_positions[1][point_index];
      ASSERT_NEAR(px, point_px, 1.0e-14);
      ASSERT_NEAR(py, point_py, 1.0e-14);

      const double rpx = (*ref_positions)[0][rowx];
      const double rpy = (*ref_positions)[1][rowx];
      const double point_rpx =
          line_initialisation->point_ref_positions[0][point_index];
      const double point_rpy =
          line_initialisation->point_ref_positions[1][point_index];
      ASSERT_NEAR(rpx, point_rpx, 1.0e-14);
      ASSERT_NEAR(rpy, point_rpy, 1.0e-14);

      // is this point actually on a line?
      const double y_test = lambda_line_eq(px);
      ASSERT_NEAR(y_test, py, 1.0e-6);
    }
  }

  A->free();
  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
}
