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

// Test advecting particles between ranks
TEST(ParticleGeometryInterface, LocalMapping2DRegular) {

  const int N_total = 2000;
  /* nektar++ maps to a scaled tolerance of 1.0-8 in Geometry2D.cpp
   * with an exit condition like
   *  (rx*rx + ry*ry) < tol
   *
   *  where rx, ry are x,y direction residuals. This exist condition means that
   *  the absolute error in rx and ry is approximately 1.0e-4.
   */
  const double tol = 1.0e-8;
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

// Test advecting particles between ranks
TEST(ParticleGeometryInterface, LocalMapping2DDeformed) {

  const int N_total = 2000;
  /* nektar++ maps to a scaled tolerance of 1.0-8 in Geometry2D.cpp
   * with an exit condition like
   *  (rx*rx + ry*ry) < tol
   *
   *  where rx, ry are x,y direction residuals. This exist condition means that
   *  the absolute error in rx and ry is approximately 1.0e-4.
   */
  const double tol = 2.0 * 1.0e-4;
  int argc = 2;
  char *argv[2];
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);

  std::filesystem::path source_file = __FILE__;
  std::filesystem::path source_dir = source_file.parent_path();
  std::filesystem::path test_resources_dir =
      source_dir / "../../test_resources";
  // std::filesystem::path mesh_file =
  //     test_resources_dir / "square_triangles_quads.xml";
  //   TODO
  std::filesystem::path mesh_file =
      "/home/js0259/git-ukaea/NESO-workspace/reference_square/"
      "reference_square_deformed_quads.xml";
  copy_to_cstring(std::string(mesh_file), &argv[1]);
  //   TODO

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
    Array<OneD, NekDouble> local_coord_test(3);
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
            geom->ContainsPoint(global_coord, local_coord, 1.0e-10, dist);

        ASSERT_TRUE(is_contained);

        local_coord_test[0] = (*reference_positions)[0][rowx];
        local_coord_test[1] = (*reference_positions)[1][rowx];
        local_coord_test[2] = 0.0;

        // check the local coordinate matches the one on the particle
        for (int dimx = 0; dimx < ndim; dimx++) {
          const double err =
              ABS(local_coord[dimx] - (*reference_positions)[dimx][rowx]);

          const NekDouble map_back_phys = geom->GetCoord(dimx, local_coord);
          ASSERT_NEAR(map_back_phys, global_coord[dimx], tol);
          const NekDouble map_back_phys_test =
              geom->GetCoord(dimx, local_coord_test);
          ASSERT_NEAR(map_back_phys_test, global_coord[dimx], tol);
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

template <typename T>
inline double ARG(T geom, double *coords, double *Lcoords) {
  auto m_shapeType = geom->GetShapeType();

  std::vector<PointGeomSharedPtr> m_verts;
  for (int vx = 0; vx < geom->GetNumVerts(); vx++) {
    m_verts.push_back(geom->GetVertex(vx));
  }

  int v1, v2, v3;
  if (m_shapeType == LibUtilities::eHexahedron ||
      m_shapeType == LibUtilities::ePrism ||
      m_shapeType == LibUtilities::ePyramid) {
    v1 = 1;
    v2 = 3;
    v3 = 4;
  } else if (m_shapeType == LibUtilities::eTetrahedron) {
    v1 = 1;
    v2 = 2;
    v3 = 3;
  } else {
    v1 = 1;
    v2 = 2;
    v3 = 3;
    ASSERTL0(false, "unrecognized 3D element type");
  }
  // Point inside tetrahedron
  PointGeom r(3, 0, coords[0], coords[1], coords[2]);

  // Edges
  PointGeom er0, e10, e20, e30;
  er0.Sub(r, *m_verts[0]);
  e10.Sub(*m_verts[v1], *m_verts[0]);
  e20.Sub(*m_verts[v2], *m_verts[0]);
  e30.Sub(*m_verts[v3], *m_verts[0]);

  // Cross products (Normal times area)
  PointGeom cp1020, cp2030, cp3010;
  cp1020.Mult(e10, e20);
  cp2030.Mult(e20, e30);
  cp3010.Mult(e30, e10);

  // Barycentric coordinates (relative volume)
  NekDouble iV = 2. / e30.dot(cp1020); // Hex Volume = {(e30)dot(e10)x(e20)}
  Lcoords[0] = er0.dot(cp2030) * iV - 1.0;
  Lcoords[1] = er0.dot(cp3010) * iV - 1.0;
  Lcoords[2] = er0.dot(cp1020) * iV - 1.0;
  nprint("AA Lcoords", Lcoords[0], Lcoords[1], Lcoords[2]);

  double eta[3] = {0.0, 0.0, 0.0};
  GeometryInterface::Hexahedron hex{};
  hex.loc_coord_to_loc_collapsed(Lcoords, eta);
  const bool clamp =
      GeometryInterface::clamp_loc_coords(&eta[0], &eta[1], &eta[2], 0.0);
  double dist = 0.0;
  if (clamp) {
    double xi[3] = {0.0, 0.0, 0.0};
    hex.loc_collapsed_to_loc_coord(eta, xi);
    xi[0] = (xi[0] + 1.) * 0.5; // re-scaled to ratio [0, 1]
    xi[1] = (xi[1] + 1.) * 0.5;
    xi[2] = (xi[2] + 1.) * 0.5;
    for (int i = 0; i < 3; ++i) {
      NekDouble tmp = xi[0] * e10[i] + xi[1] * e20[i] + xi[2] * e30[i] - er0[i];
      dist += tmp * tmp;
    }
    dist = sqrt(dist);
  }
  return dist;
}

template <typename T>
inline bool BB(T geom, double *coords, double *Lcoords,
               const double tol = 0.0) {
  // Convert to the local (xi) coordinates.
  double dist = ARG(geom, coords, Lcoords);
  nprint("dist", dist);
  if (dist <= tol + NekConstants::kNekMachineEpsilon) {
    return true;
  }
  double eta[3];
  GeometryInterface::Hexahedron hex{};
  hex.loc_coord_to_loc_collapsed(Lcoords, eta);
  if (GeometryInterface::clamp_loc_coords(&eta[0], &eta[1], &eta[2], tol)) {
    // m_xmap->LocCollapsedToLocCoord(eta, locCoord);
    return false;
  } else {
    return true;
  }
}

#include <stdio.h>
template <typename T> inline void TODOprinter(T &geom) {

  const int nverts = geom->GetNumVerts();
  nprint("nverts", nverts);
  for (int vx = 0; vx < nverts; vx++) {
    NekDouble x, y, z;
    auto v = geom->GetVertex(vx);
    v->GetCoords(x, y, z);
    nprint(vx, "(", x, ",", y, ",", z, "),");
  }

  NekDouble p[8][3] = {{-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {1.0, 1.0, -1.0},
                       {-1.0, 1.0, -1.0},  {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0},
                       {1.0, 1.0, 1.0},    {-1.0, 1.0, 1.0}};

  Array<OneD, NekDouble> local_coord(3);
  for (int px = 0; px < 8; px++) {
    local_coord[0] = p[px][0];
    local_coord[1] = p[px][1];
    local_coord[2] = p[px][2];
    NekDouble p0 = geom->GetCoord(0, local_coord);
    NekDouble p1 = geom->GetCoord(1, local_coord);
    NekDouble p2 = geom->GetCoord(2, local_coord);
    nprint(p[px][0], p[px][1], p[px][2], "\t|", p0, p1, p2);
  };

  local_coord[0] = -0.6;
  local_coord[1] = -0.5;
  local_coord[2] = -0.2;
  NekDouble p0 = geom->GetCoord(0, local_coord);
  NekDouble p1 = geom->GetCoord(1, local_coord);
  NekDouble p2 = geom->GetCoord(2, local_coord);
  nprint("test coord = (", local_coord[0], ",", local_coord[1], ",",
         local_coord[2], ")");
  printf("correct_global_coord = ( %.16f, %.16f, %.16f)\n", p0, p1, p2);

  nprint("-------------------------------------------");
}

// Test advecting particles between ranks
TEST(ParticleGeometryInterface, LocalMapping3D) {

  const int N_total = 2000;
  const double tol = 1.0e-10;

  // int argc = 2;
  // char *argv[2];
  // copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);
  // std::filesystem::path source_file = __FILE__;
  // std::filesystem::path source_dir = source_file.parent_path();
  // std::filesystem::path test_resources_dir =
  //     source_dir / "../../test_resources";
  // std::filesystem::path mesh_file =
  //     test_resources_dir / "square_triangles_quads.xml";
  // copy_to_cstring(std::string(mesh_file), &argv[1]);

  int argc = 3;
  char *argv[3];
  // copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);
  // std::filesystem::path conditions_file =
  //     "/home/js0259/git-ukaea/NESO-workspace/3D/conditions.xml";
  // copy_to_cstring(std::string(conditions_file), &argv[1]);
  // std::filesystem::path mesh_file =
  //     "/home/js0259/git-ukaea/NESO-workspace/3D/reference_cube.xml";
  // copy_to_cstring(std::string(mesh_file), &argv[2]);

  std::filesystem::path conditions_file =
      //"/home/js0259/git-ukaea/NESO-workspace/3D/conditions.xml";
      "/home/js0259/git-ukaea/NESO-workspace/reference_all_types_cube/"
      "condition.xml";
  copy_to_cstring(std::string(conditions_file), &argv[1]);
  std::filesystem::path mesh_file =
      //"/home/js0259/git-ukaea/NESO-workspace/3D/reference_cube_0.5.xml";
      "/home/js0259/git-ukaea/NESO-workspace/reference_all_types_cube/"
      "mixed_ref_cube_0.2.xml";
  copy_to_cstring(std::string(mesh_file), &argv[2]);

  LibUtilities::SessionReaderSharedPtr session;
  SpatialDomains::MeshGraphSharedPtr graph;
  // Create session reader.
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);
  graph = SpatialDomains::MeshGraph::Read(session);

  auto mesh = std::make_shared<ParticleMeshInterface>(graph);
  auto sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());

  nprint("TET");
  TODOprinter(graph->GetAllTetGeoms().begin()->second);
  nprint("PYR");
  TODOprinter(graph->GetAllPyrGeoms().begin()->second);
  nprint("PRISM");
  TODOprinter(graph->GetAllPrismGeoms().begin()->second);
  nprint("HEX");
  TODOprinter(graph->GetAllHexGeoms().begin()->second);

  Array<OneD, NekDouble> xi(3);
  Array<OneD, NekDouble> cg(3);
  for (auto &geom : graph->GetAllTetGeoms()) {

    auto n = Newton::XMapNewton<Newton::MappingTetLinear3D>(sycl_target,
                                                            geom.second);

    REAL g0, g1, g2;
    xi[0] = -0.1;
    xi[1] = -0.1;
    xi[2] = -0.1;
    for (int dx = 0; dx < 3; dx++) {
      cg[dx] = geom.second->GetCoord(dx, xi);
    }

    n.x(xi[0], xi[1], xi[2], &g0, &g1, &g2);

    nprint("T:", g0, g1, g2);
    nprint("C:", cg[0], cg[1], cg[2]);
  }

  auto nektar_graph_local_mapper =
      std::make_shared<NektarGraphLocalMapperT>(sycl_target, mesh, tol);
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
          ASSERT_TRUE(ABS(local_coord[dimx] -
                          (*reference_positions)[dimx][rowx]) <= tol);
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
