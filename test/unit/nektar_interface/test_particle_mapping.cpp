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

#include "nektar_interface/particle_cell_mapping/newton_geom_interfaces.hpp"

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::SpatialDomains;
using namespace NESO::Particles;

static inline void copy_to_cstring(std::string input, char **output) {
  *output = new char[input.length() + 1];
  std::strcpy(*output, input.c_str());
}

class ParticleGeometryInterface2D
    : public testing::TestWithParam<std::tuple<std::string, double>> {};
TEST_P(ParticleGeometryInterface2D, LocalMapping2D) {

  const int N_total = 2000;
  /* nektar++ maps to a scaled tolerance of 1.0-8 in Geometry2D.cpp
   * with an exit condition like
   *  (rx*rx + ry*ry) < tol
   *
   *  where rx, ry are x,y direction residuals. This exist condition means that
   *  the absolute error in rx and ry is approximately 1.0e-4.
   */

  std::tuple<std::string, double> param = GetParam();
  const double tol = std::get<1>(param);
  int argc = 2;
  char *argv[2];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";

  std::filesystem::path mesh_file = test_resources_dir / std::get<0>(param);

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
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);

  CellIDTranslation cell_id_translation(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);

  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  const int N = rend - rstart;

  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");

  const int cell_count = domain->mesh->get_cell_count();

  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);

    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
      }

      initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
      initial_distribution[Sym<INT>("ID")][px][0] = px;
    }
    A->add_particles_local(initial_distribution);
  }
  reset_mpi_ranks((*A)[Sym<INT>("NESO_MPI_RANK")]);

  MeshHierarchyGlobalMap mesh_hierarchy_global_map(
      sycl_target, domain->mesh, A->position_dat, A->cell_id_dat,
      A->mpi_rank_dat);

  auto lambda_check_owning_cell = [&] {
    auto point = std::make_shared<PointGeom>(ndim, -1, 0.0, 0.0, 0.0);
    Array<OneD, NekDouble> global_coord(3);
    Array<OneD, NekDouble> local_coord(3);
    for (int cellx = 0; cellx < cell_count; cellx++) {

      auto positions = A->position_dat->cell_dat.get_cell(cellx);
      auto cell_ids = A->cell_id_dat->cell_dat.get_cell(cellx);
      auto reference_positions =
          (*A)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]->cell_dat.get_cell(cellx);

      for (int rowx = 0; rowx < cell_ids->nrow; rowx++) {

        const int cell_neso = (*cell_ids)[0][rowx];
        ASSERT_EQ(cell_neso, cellx);
        const int cell_nektar = cell_id_translation.map_to_nektar[cell_neso];

        global_coord[0] = (*positions)[0][rowx];
        global_coord[1] = (*positions)[1][rowx];

        NekDouble dist;
        auto geom = graph->GetGeometry2D(cell_nektar);
        auto is_contained =
            geom->ContainsPoint(global_coord, local_coord, 1.0e-14, dist);

        ASSERT_TRUE(is_contained);

        // check the local coordinate matches the one on the particle
        for (int dimx = 0; dimx < ndim; dimx++) {
          const double err =
              ABS(local_coord[dimx] - (*reference_positions)[dimx][rowx]);

          ASSERT_TRUE(err <= tol);
        }
      }
    }
  };

  pbc.execute();
  mesh_hierarchy_global_map.execute();
  A->hybrid_move();
  cell_id_translation.execute();
  A->cell_move();
  lambda_check_owning_cell();

  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
}

INSTANTIATE_TEST_SUITE_P(
    MultipleMeshes, ParticleGeometryInterface2D,
    testing::Values(
        std::tuple<std::string, double>("square_triangles_quads.xml", 1.0e-8),
        std::tuple<std::string, double>(
            "reference_squared_deformed_quads/"
            "reference_square_deformed_quads.xml",
            2.0e-4 // The non-linear exit tolerance in Nektar is like (err_x *
                   // err_x
                   // + err_y * err_y) < 1.0e-8
            )));

template <typename T, typename U, typename R>
inline void check_geom_map(T &n, U &geom, R &rng) {

  const int N_test = 5;
  std::uniform_real_distribution<double> ref_distribution(-1.0, 1.0);
  Array<OneD, NekDouble> xi(3);
  Array<OneD, NekDouble> cg(3);
  REAL g[3];

  for (int testx = 0; testx < N_test; testx++) {

    // Get a point in the reference element
    cg[0] = ref_distribution(rng);
    cg[1] = ref_distribution(rng);
    cg[2] = ref_distribution(rng);
    geom->GetXmap()->LocCollapsedToLocCoord(cg, xi);

    n.x(xi[0], xi[1], xi[2], g, g + 1, g + 2);

    // check the map from reference space to global space
    for (int dx = 0; dx < 3; dx++) {
      cg[dx] = geom->GetCoord(dx, xi);
      if (std::isfinite(cg[dx])) {
        const REAL err_abs = abs(cg[dx] - g[dx]);
        const REAL err = std::min(err_abs, err_abs / abs(cg[dx]));
        ASSERT_TRUE(err < 1.0e-12);
      }
    }

    // check the map from global space back to reference space
    REAL xi_check[3];
    n.x_inverse(g[0], g[1], g[2], xi_check, xi_check + 1, xi_check + 2);
    for (int dx = 0; dx < 3; dx++) {
      const REAL err_abs = abs(xi_check[dx] - xi[dx]);
      const REAL err = std::min(err_abs, err_abs / abs(xi[dx]));
      // The exit tol on the newton method is 1E-10 so we test against 1E-8.
      ASSERT_TRUE(err < 1.0e-8);
    }
  }
}

class ParticleGeometryInterface
    : public testing::TestWithParam<
          std::tuple<std::string, std::string, double>> {};
TEST_P(ParticleGeometryInterface, LocalMapping3D) {

  std::tuple<std::string, std::string, double> param = GetParam();

  const int N_total = 2000;
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

  std::mt19937 rng{182348};

  for (auto &geom : graph->GetAllTetGeoms()) {
    auto n = Newton::XMapNewton<Newton::MappingTetLinear3D>(sycl_target,
                                                            geom.second);
    check_geom_map(n, geom.second, rng);
  }
  for (auto &geom : graph->GetAllPyrGeoms()) {
    auto n = Newton::XMapNewton<Newton::MappingPyrLinear3D>(sycl_target,
                                                            geom.second);
    check_geom_map(n, geom.second, rng);
  }
  for (auto &geom : graph->GetAllPrismGeoms()) {
    auto n = Newton::XMapNewton<Newton::MappingPrismLinear3D>(sycl_target,
                                                              geom.second);
    check_geom_map(n, geom.second, rng);
  }
  for (auto &geom : graph->GetAllHexGeoms()) {
    auto n = Newton::XMapNewton<Newton::MappingHexLinear3D>(sycl_target,
                                                            geom.second);
    check_geom_map(n, geom.second, rng);
  }
  auto config = std::make_shared<ParameterStore>();
  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh, config);

  auto domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);

  const int ndim = 3;
  ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                             ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                             ParticleProp(Sym<INT>("ID"), 1)};

  auto A = std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

  NektarCartesianPeriodic pbc(sycl_target, graph, A->position_dat);

  CellIDTranslation cell_id_translation(sycl_target, A->cell_id_dat, mesh);

  const int rank = sycl_target->comm_pair.rank_parent;
  const int size = sycl_target->comm_pair.size_parent;

  std::mt19937 rng_pos(52234234 + rank);

  int rstart, rend;
  get_decomp_1d(size, N_total, rank, &rstart, &rend);
  const int N = rend - rstart;

  int N_check = -1;
  MPICHK(MPI_Allreduce(&N, &N_check, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  NESOASSERT(N_check == N_total, "Error creating particles");

  const int cell_count = domain->mesh->get_cell_count();

  if (N > 0) {
    auto positions =
        uniform_within_extents(N, ndim, pbc.global_extent, rng_pos);

    ParticleSet initial_distribution(N, A->get_particle_spec());
    for (int px = 0; px < N; px++) {
      for (int dimx = 0; dimx < ndim; dimx++) {
        const double pos_orig = positions[dimx][px] + pbc.global_origin[dimx];
        initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
      }

      initial_distribution[Sym<INT>("CELL_ID")][px][0] = 0;
      initial_distribution[Sym<INT>("ID")][px][0] = px;
    }
    A->add_particles_local(initial_distribution);
  }
  reset_mpi_ranks((*A)[Sym<INT>("NESO_MPI_RANK")]);

  MeshHierarchyGlobalMap mesh_hierarchy_global_map(
      sycl_target, domain->mesh, A->position_dat, A->cell_id_dat,
      A->mpi_rank_dat);

  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> geoms_3d;
  get_all_elements_3d(graph, geoms_3d);
  auto lambda_check_owning_cell = [&] {
    Array<OneD, NekDouble> global_coord(3);
    Array<OneD, NekDouble> local_coord(3);
    for (int cellx = 0; cellx < cell_count; cellx++) {

      auto positions = A->position_dat->cell_dat.get_cell(cellx);
      auto cell_ids = A->cell_id_dat->cell_dat.get_cell(cellx);
      auto reference_positions =
          (*A)[Sym<REAL>("NESO_REFERENCE_POSITIONS")]->cell_dat.get_cell(cellx);

      for (int rowx = 0; rowx < cell_ids->nrow; rowx++) {

        const int cell_neso = (*cell_ids)[0][rowx];
        ASSERT_EQ(cell_neso, cellx);
        const int cell_nektar = cell_id_translation.map_to_nektar[cell_neso];

        global_coord[0] = (*positions)[0][rowx];
        global_coord[1] = (*positions)[1][rowx];
        global_coord[2] = (*positions)[2][rowx];

        NekDouble dist;
        auto geom = geoms_3d[cell_nektar];
        auto is_contained =
            geom->ContainsPoint(global_coord, local_coord, tol, dist);

        ASSERT_TRUE(is_contained);

        // check the local coordinate matches the one on the particle
        for (int dimx = 0; dimx < ndim; dimx++) {
          const double err_abs =
              ABS(local_coord[dimx] - (*reference_positions)[dimx][rowx]);
          ASSERT_TRUE(err_abs <= tol);
        }
      }
    }
  };

  pbc.execute();
  mesh_hierarchy_global_map.execute();
  A->hybrid_move();
  cell_id_translation.execute();
  A->cell_move();
  lambda_check_owning_cell();

  mesh->free();

  delete[] argv[0];
  delete[] argv[1];
  delete[] argv[2];
}

INSTANTIATE_TEST_SUITE_P(
    MultipleMeshes, ParticleGeometryInterface,
    testing::Values(
        std::tuple<std::string, std::string, double>(
            "reference_all_types_cube/conditions.xml",
            "reference_all_types_cube/mixed_ref_cube_0.5_perturbed.xml",
            2.0e-4 // The non-linear exit tolerance in Nektar is like (err_x *
                   // err_x
                   // + err_y * err_y) < 1.0e-8
            ),
        std::tuple<std::string, std::string, double>(
            "reference_all_types_cube/conditions.xml",
            "reference_all_types_cube/mixed_ref_cube_0.5.xml", 1.0e-10)));
