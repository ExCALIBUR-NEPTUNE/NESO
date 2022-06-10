#include "nektar_interface/geometry_transport_2d.hpp"
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>
#include <gtest/gtest.h>
#include <memory>

using namespace std;
using namespace Nektar;
using namespace Nektar::SolverUtils;
using namespace Nektar::SpatialDomains;

TEST(GeometryTransport2DTest, QuadClone) {

  int size, rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  PointGeomSharedPtr edge_points[2];
  SegGeomSharedPtr edges[4];
  QuadGeomMap map_quads{};

  const double E = 1.0;

  // Create a quad on this rank
  // E x E quad on each rank shifted by E * rank in x

  for (int yx = 0; yx < 2; yx++) {
    PointGeomSharedPtr p0 = std::make_shared<PointGeom>(2, 0, (rank * E) + 0.0,
                                                        (yx + 0.0) * E, 0.0);
    PointGeomSharedPtr p1 = std::make_shared<PointGeom>(
        2, 1, (rank * E) + 1.0 * E, (yx + 0.0) * E, 0.0);
    PointGeomSharedPtr p2 = std::make_shared<PointGeom>(
        2, 2, (rank * E) + 1.0 * E, (yx + 1.0) * E, 0.0);
    PointGeomSharedPtr p3 = std::make_shared<PointGeom>(2, 3, (rank * E) + 0.0,
                                                        (yx + 1.0) * E, 0.0);
    edge_points[0] = p0;
    edge_points[1] = p1;
    edges[0] = std::make_shared<SpatialDomains::SegGeom>(0, 2, edge_points);
    edge_points[0] = p1;
    edge_points[1] = p2;
    edges[1] = std::make_shared<SpatialDomains::SegGeom>(1, 2, edge_points);
    edge_points[0] = p2;
    edge_points[1] = p3;
    edges[2] = std::make_shared<SpatialDomains::SegGeom>(2, 2, edge_points);
    edge_points[0] = p3;
    edge_points[1] = p0;
    edges[3] = std::make_shared<SpatialDomains::SegGeom>(3, 2, edge_points);
    map_quads[yx] = std::make_shared<SpatialDomains::QuadGeom>(0, edges);

    auto geom = map_quads[yx];
    // std::cout << "setup start" << std::endl;
    geom->GetGeomFactors();
    geom->Setup();
    // std::cout << "setup passed" << std::endl;
  }

  auto remote_quads = NESO::get_all_remote_geoms_2d(MPI_COMM_WORLD, map_quads);

  Array<OneD, NekDouble> point(3);

  point[0] = 0.25;
  point[1] = 0.25;
  point[2] = 0.0;

  for (auto &rg : remote_quads) {
    auto num_verts = rg->geom->GetNumVerts();
    //   cout << "Num verts: " << num_verts << " remote rank " << rg->rank << "
    //   remote id " << rg->id << endl;
    for (int vx = 0; vx < num_verts; vx++) {
      const auto vert = rg->geom->GetVertex(vx);

      NekDouble x, y, z;
      vert->GetCoords(x, y, z);
      // cout << "x: " << x << " y: " << y << endl;
    }

    auto bounds = rg->geom->GetBoundingBox();
    /*
    std::cout << "\t"
        << bounds[0] << ", "
        << bounds[1] << ", "
        << bounds[2] << ", "
        << bounds[3] << ", "
        << bounds[4] << ", "
        << bounds[5] << std::endl;

    std::cout << "contains point? " << rg->geom->ContainsPoint(point) <<
    std::endl;
    */
  }
}

TEST(GeometryTransport2DTest, TriClone) {

  int size, rank;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  PointGeomSharedPtr edge_points[2];
  SegGeomSharedPtr edges[3];
  TriGeomMap map_tris{};

  const double E = 1.0;

  for (int yx = 0; yx < 2; yx++) {
    PointGeomSharedPtr p0 = std::make_shared<PointGeom>(2, 0, (rank * E) + 0.0,
                                                        (yx + 0.0) * E, 0.0);
    PointGeomSharedPtr p1 = std::make_shared<PointGeom>(
        2, 1, (rank * E) + 1.0 * E, (yx + 0.0) * E, 0.0);
    PointGeomSharedPtr p2 = std::make_shared<PointGeom>(
        2, 2, (rank * E) + 1.0 * E, (yx + 1.0) * E, 0.0);
    edge_points[0] = p0;
    edge_points[1] = p1;
    edges[0] = std::make_shared<SpatialDomains::SegGeom>(0, 2, edge_points);
    edge_points[0] = p1;
    edge_points[1] = p2;
    edges[1] = std::make_shared<SpatialDomains::SegGeom>(1, 2, edge_points);
    edge_points[0] = p2;
    edge_points[1] = p0;
    edges[2] = std::make_shared<SpatialDomains::SegGeom>(2, 2, edge_points);

    map_tris[yx] = std::make_shared<SpatialDomains::TriGeom>(0, edges);
  }

  auto remote_tris = NESO::get_all_remote_geoms_2d(MPI_COMM_WORLD, map_tris);

  Array<OneD, NekDouble> point(3);

  point[0] = 0.1;
  point[1] = 0.1;
  point[2] = 0.0;

  for (auto &rg : remote_tris) {
    auto num_verts = rg->geom->GetNumVerts();
    // cout << "Num verts: " << num_verts << " remote rank " << rg->rank << "
    // remote id " << rg->id << endl;
    for (int vx = 0; vx < num_verts; vx++) {
      const auto vert = rg->geom->GetVertex(vx);

      NekDouble x, y, z;
      vert->GetCoords(x, y, z);
      // cout << "x: " << x << " y: " << y << endl;
    }

    auto bounds = rg->geom->GetBoundingBox();
    /*
    std::cout << "\t"
        << bounds[0] << ", "
        << bounds[1] << ", "
        << bounds[2] << ", "
        << bounds[3] << ", "
        << bounds[4] << ", "
        << bounds[5] << std::endl;

    std::cout << "contains point? " << rg->geom->ContainsPoint(point) <<
    std::endl;
    */
  }
}
