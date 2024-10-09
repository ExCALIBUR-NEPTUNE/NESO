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

using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

// Test advecting particles between ranks
TEST(ParticleGeometryInterface, Advection2D) {

  const int N_total = 1000;
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

  auto config = std::make_shared<ParameterStore>();
  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh, config);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 2;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("P_ORIG"), ndim),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);

  CellIDTranslation cell_id_translation(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::mt19937 rng_rank(18241);

  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  const int N = rend - rstart;

  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");

  const int Nsteps = 2000;
  const REAL dt = 0.10;
  const int cell_count = domain->mesh->get_cell_count();

  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);
    auto velocities =
        NESO::Particles::normal_distribution(N, 3, 0.0, 0.5, rng_vel);
    std::uniform_int_distribution<int> uniform_dist(
        0, sycl_target->comm_pair.size_parent - 1);
    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
        initial_distribution[Sym<REAL>("P_ORIG")][px][dimx] = pos_orig;
      }
      for (int dimx = 0; dimx < 3; dimx++) {
        initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
      initial_distribution[Sym<INT>("ID")][px][0] = px;
      const auto px_rank = uniform_dist(rng_rank);
      initial_distribution[Sym<INT>("NESO_MPI_RANK")][px][0] = px_rank;
    }
    A->add_particles_local(initial_distribution);
  }
  reset_mpi_ranks((*A)[Sym<INT>("NESO_MPI_RANK")]);

  MeshHierarchyGlobalMap mesh_hierarchy_global_map(
      sycl_target, domain->mesh, A->position_dat, A->cell_id_dat,
      A->mpi_rank_dat);

  auto lambda_advect = [&] {
    auto t0 = profile_timestamp();
    particle_loop(
        A,
        [=](auto P, auto V) {
          for (int dimx = 0; dimx < ndim; dimx++) {
            P.at(dimx) += dt * V.at(dimx);
          }
        },
        Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")))
        ->execute();
    sycl_target->profile_map.inc("Advect", "Execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));
  };

  auto lambda_check_owning_cell = [&] {
    Array<OneD, NekDouble> global_coord(3);
    Array<OneD, NekDouble> local_coord(3);
    Array<OneD, NekDouble> eta(3);
    for (int cellx = 0; cellx < cell_count; cellx++) {

      auto positions = A->position_dat->cell_dat.get_cell(cellx);
      auto cell_ids = A->cell_id_dat->cell_dat.get_cell(cellx);
      auto reference_positions =
          A->get_cell(Sym<REAL>("NESO_REFERENCE_POSITIONS"), cellx);

      for (int rowx = 0; rowx < cell_ids->nrow; rowx++) {

        const int cell_neso = (*cell_ids)[0][rowx];
        ASSERT_EQ(cell_neso, cellx);
        const int cell_nektar = cell_id_translation.map_to_nektar[cell_neso];

        auto geom = graph->GetGeometry2D(cell_nektar);
        local_coord[0] = reference_positions->at(rowx, 0);
        local_coord[1] = reference_positions->at(rowx, 1);
        global_coord[0] = geom->GetCoord(0, local_coord);
        global_coord[1] = geom->GetCoord(1, local_coord);

        geom->GetXmap()->LocCoordToLocCollapsed(local_coord, eta);
        // check the global coordinate matches the one on the particle
        for (int dimx = 0; dimx < ndim; dimx++) {
          const double err_abs =
              ABS(positions->at(rowx, dimx) - global_coord[dimx]);
          ASSERT_TRUE(err_abs <= tol);
          ASSERT_TRUE(std::fabs((double)eta[dimx]) < (1.0 + tol));
        }
      }
    }
  };

  REAL T = 0.0;

  for (int stepx = 0; stepx < Nsteps; stepx++) {

    pbc.execute();
    mesh_hierarchy_global_map.execute();
    A->hybrid_move();
    cell_id_translation.execute();
    A->cell_move();
    lambda_check_owning_cell();

    lambda_advect();

    T += dt;
    // if ((stepx % 100 == 0) && (rank == 0)) {
    //   std::cout << stepx << std::endl;
    // }
  }

  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
}

class ParticleAdvection3D : public testing::TestWithParam<
                                std::tuple<std::string, std::string, double>> {
};
TEST_P(ParticleAdvection3D, Advection3D) {
  // Test advecting particles between ranks

  std::tuple<std::string, std::string, double> param = GetParam();

  const int N_total = 1000;
  const double tol = std::get<2>(param);

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";

  std::filesystem::path condtions_file_basename =
      static_cast<std::string>(std::get<0>(param));
  std::filesystem::path mesh_file_basename =
      static_cast<std::string>(std::get<1>(param));
  std::filesystem::path conditions_file =
      test_resources_dir / condtions_file_basename;
  std::filesystem::path mesh_file = test_resources_dir / mesh_file_basename;

  int argc = 3;
  char *argv[3];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);
  copy_to_cstring(std::string(conditions_file), &argv[1]);
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);
  graph = SpatialDomains::MeshGraph::Read(session);

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  auto config = std::make_shared<ParameterStore>();
  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh, config);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 3;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<REAL>("P_ORIG"), ndim),
                             ParticleProp(Sym<REAL>("V"), 3),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);

  CellIDTranslation cell_id_translation(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);
  std::mt19937 rng_vel(52234231 + rank);
  std::mt19937 rng_rank(18241);

  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  const int N = rend - rstart;

  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");

  const int Nsteps = 2000;
  const REAL dt = 0.1;
  const int cell_count = domain->mesh->get_cell_count();

  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);
    auto velocities =
        NESO::Particles::normal_distribution(N, 3, 0.0, 0.5, rng_vel);
    std::uniform_int_distribution<int> uniform_dist(
        0, sycl_target->comm_pair.size_parent - 1);
    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
        initial_distribution[Sym<REAL>("P_ORIG")][px][dimx] = pos_orig;
      }
      for (int dimx = 0; dimx < 3; dimx++) {
        initial_distribution[Sym<REAL>("V")][px][dimx] = velocities[dimx][px];
      }
      initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
      initial_distribution[Sym<INT>("ID")][px][0] = px;
      const auto px_rank = uniform_dist(rng_rank);
      initial_distribution[Sym<INT>("NESO_MPI_RANK")][px][0] = px_rank;
    }
    A->add_particles_local(initial_distribution);
  }
  reset_mpi_ranks((*A)[Sym<INT>("NESO_MPI_RANK")]);

  MeshHierarchyGlobalMap mesh_hierarchy_global_map(
      sycl_target, domain->mesh, A->position_dat, A->cell_id_dat,
      A->mpi_rank_dat);

  auto lambda_advect = [&] {
    auto t0 = profile_timestamp();
    particle_loop(
        A,
        [=](auto P, auto V) {
          for (int dimx = 0; dimx < ndim; dimx++) {
            P.at(dimx) += dt * V.at(dimx);
          }
        },
        Access::write(Sym<REAL>("P")), Access::read(Sym<REAL>("V")))
        ->execute();
    sycl_target->profile_map.inc("Advect", "Execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));
  };
  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> geoms_3d;
  get_all_elements_3d(graph, geoms_3d);

  auto lambda_check_owning_cell = [&] {
    Array<OneD, NekDouble> global_coord(3);
    Array<OneD, NekDouble> local_coord(3);
    Array<OneD, NekDouble> eta(3);
    for (int cellx = 0; cellx < cell_count; cellx++) {

      auto positions = A->position_dat->cell_dat.get_cell(cellx);
      auto cell_ids = A->cell_id_dat->cell_dat.get_cell(cellx);
      auto reference_positions =
          A->get_cell(Sym<REAL>("NESO_REFERENCE_POSITIONS"), cellx);

      for (int rowx = 0; rowx < cell_ids->nrow; rowx++) {

        const int cell_neso = (*cell_ids)[0][rowx];
        ASSERT_EQ(cell_neso, cellx);
        const int cell_nektar = cell_id_translation.map_to_nektar[cell_neso];
        auto geom = geoms_3d[cell_nektar];
        local_coord[0] = reference_positions->at(rowx, 0);
        local_coord[1] = reference_positions->at(rowx, 1);
        local_coord[2] = reference_positions->at(rowx, 2);
        global_coord[0] = geom->GetCoord(0, local_coord);
        global_coord[1] = geom->GetCoord(1, local_coord);
        global_coord[2] = geom->GetCoord(2, local_coord);

        geom->GetXmap()->LocCoordToLocCollapsed(local_coord, eta);

        // check the global coordinate matches the one on the particle
        for (int dimx = 0; dimx < ndim; dimx++) {
          const double err_abs =
              ABS(positions->at(rowx, dimx) - global_coord[dimx]);
          ASSERT_TRUE(err_abs <= tol);
          ASSERT_TRUE(std::fabs((double)eta[dimx]) < (1.0 + tol));
        }
      }
    }
  };

  // H5Part h5part("trajectory.h5part", A, Sym<REAL>("P"),
  //               Sym<INT>("NESO_MPI_RANK"),
  //               Sym<REAL>("NESO_REFERENCE_POSITIONS"));

  REAL T = 0.0;
  for (int stepx = 0; stepx < Nsteps; stepx++) {

    pbc.execute();
    mesh_hierarchy_global_map.execute();
    A->hybrid_move();
    cell_id_translation.execute();
    A->cell_move();
    lambda_check_owning_cell();

    lambda_advect();
    T += dt;
    // h5part.write();
  }

  // h5part.close();
  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

INSTANTIATE_TEST_SUITE_P(
    MultipleMeshes, ParticleAdvection3D,
    testing::Values(std::tuple<std::string, std::string, double>(
                        "reference_all_types_cube/conditions.xml",
                        "reference_all_types_cube/linear_non_regular_0.5.xml",
                        1.0e-4 // The non-linear exit tolerance in Nektar is
                               // like (err_x * err_x
                               // + err_y * err_y) < 1.0e-8
                        ),
                    std::tuple<std::string, std::string, double>(
                        "reference_all_types_cube/conditions.xml",
                        "reference_all_types_cube/mixed_ref_cube_0.5.xml",
                        1.0e-10)));
