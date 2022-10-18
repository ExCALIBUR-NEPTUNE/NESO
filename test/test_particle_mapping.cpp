#include "nektar_interface/particle_interface.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

// Test advecting particles between ranks
TEST(ParticleGeometryInterface, LocalMapping) {

  const int N_total = 2000;
  const double tol = 1.0e-10;
  int argc = 2;
  char *argv[2];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);
  copy_to_cstring(std::string("test/test_resources/square_triangles_quads.xml"),
                  &argv[1]);

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);
  graph = SpatialDomains::MeshGraph::Read(session);

  ParticleMeshInterface mesh(graph);
  SYCLTarget sycl_target{0, mesh.get_comm()};

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapperT>(sycl_target, mesh, tol);
  Domain domain(mesh, nektar_graph_local_mapper);

  const int ndim = 2;
  const double extent[2] = {1.0, 1.0};
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  ParticleGroup A(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A.position_dat);

  CellIDTranslation cell_id_translation(sycl_target, A.cell_id_dat, mesh);

  const int rank = sycl_target.comm_pair.rank_parent;
  const int size = sycl_target.comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);

  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  const int N = rend - rstart;

  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");

  const int cell_count = domain.mesh.get_cell_count();

  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);

    ParticleSet initial_distribution(N, A.get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
      }

      initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
      initial_distribution[Sym<INT>("ID")][px][0] = px;
    }
    A.add_particles_local(initial_distribution);
  }
  reset_mpi_ranks(A[Sym<INT>("NESO_MPI_RANK")]);

  MeshHierarchyGlobalMap mesh_hierarchy_global_map(
      sycl_target, domain.mesh, A.position_dat, A.cell_id_dat, A.mpi_rank_dat);

  auto lambda_check_owning_cell = [&] {
    auto point = std::make_shared<PointGeom>(ndim, -1, 0.0, 0.0, 0.0);
    Array<OneD, NekDouble> global_coord(3);
    Array<OneD, NekDouble> local_coord(3);
    for (int cellx = 0; cellx < cell_count; cellx++) {

      auto positions = A.position_dat->cell_dat.get_cell(cellx);
      auto cell_ids = A.cell_id_dat->cell_dat.get_cell(cellx);
      auto reference_positions =
          A[Sym<REAL>("NESO_REFERENCE_POSITIONS")]->cell_dat.get_cell(cellx);

      for (int rowx = 0; rowx < cell_ids->nrow; rowx++) {

        const int cell_neso = (*cell_ids)[0][rowx];
        ASSERT_EQ(cell_neso, cellx);
        const int cell_nektar = cell_id_translation.map_to_nektar[cell_neso];

        global_coord[0] = (*positions)[0][rowx];
        global_coord[1] = (*positions)[1][rowx];

        NekDouble dist;
        auto geom = graph->GetGeometry2D(cell_nektar);
        auto is_contained =
            geom->ContainsPoint(global_coord, local_coord, tol, dist);

        ASSERT_TRUE(is_contained);

        // check the local coordinate matches the one on the particle
        for (int dimx = 0; dimx < ndim; dimx++) {
          ASSERT_TRUE(ABS(local_coord[dimx] -
                          (*reference_positions)[dimx][rowx]) <= tol);
        }
      }
    }
  };

  pbc.execute();
  mesh_hierarchy_global_map.execute();
  A.hybrid_move();
  cell_id_translation.execute();
  A.cell_move();
  lambda_check_owning_cell();

  mesh.free();

  delete[] argv[0];
  delete[] argv[1];
}
