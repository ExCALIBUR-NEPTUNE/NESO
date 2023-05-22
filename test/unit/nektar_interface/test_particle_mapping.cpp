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
TEST(ParticleGeometryInterface, LocalMapping2D) {

  const int N_total = 2000;
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

template<typename T>
inline double ARG(
  T geom,
  double * coords,
  double * Lcoords
){
  auto m_shapeType = geom->GetShapeType();

  std::vector<PointGeomSharedPtr> m_verts;
  for(int vx=0 ; vx<geom->GetNumVerts() ; vx++){
    m_verts.push_back(geom->GetVertex(vx));
  }


         int v1, v2, v3;
         if (m_shapeType == LibUtilities::eHexahedron ||
             m_shapeType == LibUtilities::ePrism ||
             m_shapeType == LibUtilities::ePyramid)
         {
             v1 = 1;
             v2 = 3;
             v3 = 4;
         }
         else if (m_shapeType == LibUtilities::eTetrahedron)
         {
             v1 = 1;
             v2 = 2;
             v3 = 3;
         }
         else
         {
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
         NekDouble iV =
             2. / e30.dot(cp1020); // Hex Volume = {(e30)dot(e10)x(e20)}
         Lcoords[0] = er0.dot(cp2030) * iV - 1.0;
         Lcoords[1] = er0.dot(cp3010) * iV - 1.0;
         Lcoords[2] = er0.dot(cp1020) * iV - 1.0;
         nprint("AA Lcoords", Lcoords[0], Lcoords[1], Lcoords[2]);

        
          double eta[3] = {0.0, 0.0, 0.0};
          GeometryInterface::Hexahedron hex{};
          hex.loc_coord_to_loc_collapsed(Lcoords, eta);
          const bool clamp = GeometryInterface::clamp_loc_coords(&eta[0], &eta[1], &eta[2], 0.0);
          double dist = 0.0;
         if (clamp)
         {
           double xi[3] = {0.0, 0.0, 0.0};
             hex.loc_collapsed_to_loc_coord(eta, xi);
             xi[0] = (xi[0] + 1.) * 0.5; // re-scaled to ratio [0, 1]
             xi[1] = (xi[1] + 1.) * 0.5;
             xi[2] = (xi[2] + 1.) * 0.5;
             for (int i = 0; i < 3; ++i)
             {
                 NekDouble tmp =
                     xi[0] * e10[i] + xi[1] * e20[i] + xi[2] * e30[i] - er0[i];
                 dist += tmp * tmp;
             }
             dist = sqrt(dist);
         }
          return dist;
}

template <typename T>
inline bool BB(
  T geom,
  double * coords,
  double * Lcoords,
  const double tol = 0.0
)
 {
     // Convert to the local (xi) coordinates.
     double dist = ARG(geom, coords, Lcoords);
     nprint("dist", dist);
     if (dist <= tol + NekConstants::kNekMachineEpsilon)
     {
         return true;
     }
     double eta[3];
      GeometryInterface::Hexahedron hex{};
      hex.loc_coord_to_loc_collapsed(Lcoords, eta);
     if (GeometryInterface::clamp_loc_coords(&eta[0], &eta[1], &eta[2], tol))
     {
         //m_xmap->LocCollapsedToLocCoord(eta, locCoord);
         return false;
     }
     else
     {
         return true;
     }
 }








// Test advecting particles between ranks
TEST(ParticleGeometryInterface, LocalMapping3D) {

  const int N_total = 50;
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
  copy_to_cstring(std::string("test_particle_geometry_interface"), &argv[0]);
  std::filesystem::path conditions_file =
      "/home/js0259/git-ukaea/NESO-workspace/3D/conditions.xml";
  copy_to_cstring(std::string(conditions_file), &argv[1]);
  std::filesystem::path mesh_file =
      "/home/js0259/git-ukaea/NESO-workspace/3D/reference_cube.xml";
  copy_to_cstring(std::string(mesh_file), &argv[2]);

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

  int cells[18] = {362,  361, 259,  210, 206, 208, 854, 258, 204,
                   1423, 855, 1421, 852, 113, 55,  115, 257, 256};

  double point[3] = {0.645735, 0.118957, 0.285289};

  Array<OneD, NekDouble> coord(3);
  Array<OneD, NekDouble> loc_coord(3);
  Array<OneD, NekDouble> col_coord(3);

  coord[0] = point[0];
  coord[1] = point[1];
  coord[2] = point[2];

  for (int cx = 0; cx < 18; cx++) {

    const int geom_id = cells[cx];
    auto geom = geoms_3d[geom_id];
    double disti;
    const bool contains_point = geom->ContainsPoint(coord, loc_coord, 0.0, disti);
    
    if (contains_point) {
      nprint("disti", disti);
      nprint("is regular:", geom->GetMetricInfo()->GetGtype() == eRegular);
      nprint("is deformed:", geom->GetMetricInfo()->GetGtype() == eDeformed);
      geom->GetXmap()->LocCoordToLocCollapsed(loc_coord, col_coord);

      auto bb = geom->GetBoundingBox();
      nprint(bb[0], bb[3]);
      nprint(bb[1], bb[4]);
      nprint(bb[2], bb[5]);


      nprint("FOUND HOST:", geom_id, loc_coord[0], loc_coord[1], loc_coord[2]);
      nprint("xi :", loc_coord[0], loc_coord[1], loc_coord[2]);
      nprint("eta:", col_coord[0], col_coord[1], col_coord[2]);
      nprint("shape type hex ", shape_type_to_int(eHexahedron));
      nprint("shape type geom", shape_type_to_int(geom->GetShapeType()));

      auto v0 = geom->GetVertex(0);
      auto v1 = geom->GetVertex(1);
      auto v2 = geom->GetVertex(3);
      auto v3 = geom->GetVertex(4);

      NekDouble x, y, z;

      v0->GetCoords(x, y, z);
      nprint("v0", x, y, z);
      v1->GetCoords(x, y, z);
      nprint("v1", x, y, z);
      v2->GetCoords(x, y, z);
      nprint("v2", x, y, z);
      v3->GetCoords(x, y, z);
      nprint("v3", x, y, z);

      PointGeom r(3, 0, point[0], point[1], point[2]);
      
      nprint("r", point[0], point[1], point[2]);


      // Edges
      PointGeom er0, e10, e20, e30;
      er0.Sub(r, *v0);
      e10.Sub(*v1, *v0);
      e20.Sub(*v2, *v0);
      e30.Sub(*v3, *v0);

      // Cross products (Normal times area)
      PointGeom cp1020, cp2030, cp3010;
      cp1020.Mult(e10, e20);
      cp2030.Mult(e20, e30);
      cp3010.Mult(e30, e10);

      cp1020.GetCoords(x, y, z);
      nprint("cp1020", x, y, z);
      cp2030.GetCoords(x, y, z);
      nprint("cp2030", x, y, z);
      cp3010.GetCoords(x, y, z);
      nprint("cp3010", x, y, z);

      NekDouble iV = 2. / e30.dot(cp1020); // Hex Volume = {(e30)dot(e10)x(e20)}
      nprint("iV", iV);

      double Lcoords[3];
      Lcoords[0] = er0.dot(cp2030) * iV - 1.0;
      Lcoords[1] = er0.dot(cp3010) * iV - 1.0;
      Lcoords[2] = er0.dot(cp1020) * iV - 1.0;
      nprint("Lcoords", Lcoords[0], Lcoords[1], Lcoords[2]);
      
      double xi[3];
      double eta[3];
      GeometryInterface::Hexahedron hex{};
      hex.loc_coord_to_loc_collapsed(Lcoords, eta);

      const bool clamp = GeometryInterface::clamp_loc_coords(
        &eta[0],
        &eta[1],
        &eta[2],
        0.0
      );
      nprint("clamped?", clamp);
      
      double Lcoords2[3];
      const double contained2 = BB(
        geom,
        point,
        Lcoords2
      );

      nprint("contained2:", contained2);

    }
  }

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
