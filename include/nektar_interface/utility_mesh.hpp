#ifndef __UTILITY_MESH_H_
#define __UTILITY_MESH_H_

#include <LibUtilities/Foundations/PolyEPoints.h>
#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

#include <memory>
#include <vector>
using namespace Nektar;

namespace NESO {

/**
 * Create a curved HexGeom from three X maps and a number of modes. Each X map
 * should map the reference space [-1,1]^3 to a point in R.
 *
 * @param num_modes Number of modes in the coordinate mapping for the Xmap, i.e.
 * polynomial order plus 1.
 * @param xmapx Callable that defines the X map in the x direction with
 * signature NekDouble(std::array<NekDouble,3>) where the single argument is a
 * subscriptable type.;
 * @param xmapy Callable that defines the X map in the y direction with
 * signature NekDouble(std::array<NekDouble,3>) where the single argument is a
 * subscriptable type.;
 * @param xmapz Callable that defines the X map in the z direction with
 * signature NekDouble(std::array<NekDouble,3>) where the single argument is a
 * subscriptable type.;
 * @returns HexGeom constructed from X maps.
 */
template <typename T, typename U, typename V>
inline std::shared_ptr<SpatialDomains::HexGeom>
make_hex_geom(const int num_modes, T xmapx, U xmapy, V xmapz) {

  /**
   * Vertices:
   *
   *  7 - 6
   *  |   |   z = 1
   *  4 - 5
   *
   *  3 - 2
   *  |   |   z = -1
   *  0 - 1
   *
   *  Edges:
   *
   *  * 10 *
   *  11   9  z = 1
   *  *  8 *
   *
   *  7 - 6
   *  |   |   z = 0
   *  4 - 5
   *
   *  * 2 *
   *  3   1   z = -1
   *  * 0 *
   *
   *  Faces:
   *
   *  * - *
   *  | 5 |   Top face, z = 1
   *  * - *
   *
   *  * 3 *
   *  4   2   Sides, z = 0
   *  * 1 *
   *
   *  * - *
   *  | 0 |   Bottom face, z = -1
   *  * - *
   *
   *  auto pts = LibUtilities::PointsManager()[*points_key];
   *  Triangle is something like trievenlyspaced
   *  quad will x fastest then y
   *  Triangles are not the expansion looping ordering - look at nektmesh
   *  top eta0, eta1, eta2=0
   *  eta0 eta1
   *
   *  make meshgraph, meshgraphio. view -> adavnced "nonlinear subdivisions
   *  (slider)"
   */

  std::array<NekDouble, 3> coords_vertices[8] = {
      {-1.0, -1.0, -1.0}, {1.0, -1.0, -1.0}, {1.0, 1.0, -1.0},
      {-1.0, 1.0, -1.0},  {-1.0, -1.0, 1.0}, {1.0, -1.0, 1.0},
      {1.0, 1.0, 1.0},    {-1.0, 1.0, 1.0}};

  int map_edge_to_vertices[12][2] = {
      {0, 1}, {1, 2}, {2, 3}, {3, 0},

      {0, 4}, {1, 5}, {2, 6}, {3, 7},

      {4, 5}, {5, 6}, {6, 7}, {7, 4},
  };

  int map_face_to_edges[6][4] = {
      {0, 1, 2, 3},  {0, 5, 8, 4},  {1, 6, 9, 5},
      {2, 7, 10, 6}, {3, 4, 11, 7}, {8, 9, 10, 11},
  };

  int map_face_to_vertices[6][4] = {{0, 1, 2, 3}, {0, 1, 4, 5}, {1, 2, 6, 5},
                                    {2, 3, 7, 6}, {1, 3, 7, 4}, {4, 5, 6, 7}};

  auto points_key =
      LibUtilities::PointsKey(num_modes, LibUtilities::ePolyEvenlySpaced);
  auto control_points = std::make_shared<LibUtilities::PolyEPoints>(points_key);
  control_points->Initialize();
  auto Z = control_points->GetZ();
  std::vector<NekDouble> Znormalised(Z.size());
  for (int ix = 0; ix < num_modes; ix++) {
    Znormalised.at(ix) = (Z[ix] + 1.0) / 2.0;
  }

  std::map<int, std::shared_ptr<SpatialDomains::PointGeom>> v;
  std::map<int, std::shared_ptr<SpatialDomains::SegGeom>> e;
  std::map<int, std::shared_ptr<SpatialDomains::Curve>> c;
  std::map<int, std::shared_ptr<SpatialDomains::QuadGeom>> q;

  // Create the vertices
  int vx_index = 0;
  int cx_index = 0;
  for (int vx = 0; vx < 8; vx++) {
    const NekDouble px = xmapx(coords_vertices[vx]);
    const NekDouble py = xmapy(coords_vertices[vx]);
    const NekDouble pz = xmapz(coords_vertices[vx]);
    v[vx] =
        std::make_shared<SpatialDomains::PointGeom>(3, vx_index++, px, py, pz);
  }

  // Create the edges
  auto lambda_get_1D_qpoint = [&](const int point_id, const auto a,
                                  const auto b) -> std::array<NekDouble, 3> {
    NekDouble ax, ay, az, bx, by, bz;
    ax = a[0];
    ay = a[1];
    az = a[2];
    bx = b[0];
    by = b[1];
    bz = b[2];
    const NekDouble dx = bx - ax;
    const NekDouble dy = by - ay;
    const NekDouble dz = bz - az;

    std::array<NekDouble, 3> out;
    out[0] = ax + Znormalised[point_id] * dx;
    out[1] = ay + Znormalised[point_id] * dy;
    out[2] = az + Znormalised[point_id] * dz;
    return out;
  };

  for (int ex = 0; ex < 12; ex++) {
    auto cx = std::make_shared<SpatialDomains::Curve>(
        cx_index++, LibUtilities::ePolyEvenlySpaced);

    for (int mx = 0; mx < num_modes; mx++) {
      auto ref_coord =
          lambda_get_1D_qpoint(mx, coords_vertices[map_edge_to_vertices[ex][0]],
                               coords_vertices[map_edge_to_vertices[ex][1]]);
      const NekDouble px = xmapx(ref_coord);
      const NekDouble py = xmapy(ref_coord);
      const NekDouble pz = xmapz(ref_coord);
      cx->m_points.push_back(std::make_shared<SpatialDomains::PointGeom>(
          3, vx_index++, px, py, pz));
    }
    std::shared_ptr<SpatialDomains::PointGeom> vertices_array[2] = {
        v.at(map_edge_to_vertices[ex][0]), v.at(map_edge_to_vertices[ex][1])};
    c[ex] = cx;
    e[ex] =
        std::make_shared<SpatialDomains::SegGeom>(ex, 3, vertices_array, cx);
    e[ex]->GetGeomFactors();
    e[ex]->Setup();
  }

  // Create the faces
  auto lambda_get_2D_qpoint = [&](const int point_id0, const int point_id1,
                                  const auto a, const auto b,
                                  const auto c) -> std::array<NekDouble, 3> {
    /**
     * c
     * |
     * a - b
     */

    NekDouble ax, ay, az, bx, by, bz, cx, cy, cz;
    ax = a[0];
    ay = a[1];
    az = a[2];
    bx = b[0];
    by = b[1];
    bz = b[2];
    cx = c[0];
    cy = c[1];
    cz = c[2];
    const NekDouble d0x = bx - ax;
    const NekDouble d0y = by - ay;
    const NekDouble d0z = bz - az;
    const NekDouble d1x = cx - ax;
    const NekDouble d1y = cy - ay;
    const NekDouble d1z = cz - az;

    std::array<NekDouble, 3> out;
    out[0] = ax + Znormalised[point_id0] * d0x + Znormalised[point_id1] * d1x;
    out[1] = ay + Znormalised[point_id0] * d0y + Znormalised[point_id1] * d1y;
    out[2] = az + Znormalised[point_id0] * d0z + Znormalised[point_id1] * d1z;
    return out;
  };

  for (int fx = 0; fx < 6; fx++) {
    std::shared_ptr<SpatialDomains::PointGeom> vertices_array[4] = {
        v.at(map_face_to_vertices[fx][0]), v.at(map_face_to_vertices[fx][1]),
        v.at(map_face_to_vertices[fx][2]), v.at(map_face_to_vertices[fx][3])};

    auto cx = std::make_shared<SpatialDomains::Curve>(
        cx_index++, LibUtilities::ePolyEvenlySpaced);
    for (int mx = 0; mx < num_modes; mx++) {
      for (int my = 0; my < num_modes; my++) {
        auto ref_coord = lambda_get_2D_qpoint(
            mx, my, coords_vertices[map_face_to_vertices[fx][0]],
            coords_vertices[map_face_to_vertices[fx][1]],
            coords_vertices[map_face_to_vertices[fx][3]]);
        const NekDouble px = xmapx(ref_coord);
        const NekDouble py = xmapy(ref_coord);
        const NekDouble pz = xmapz(ref_coord);
        cx->m_points.push_back(std::make_shared<SpatialDomains::PointGeom>(
            3, vx_index++, px, py, pz));
      }
    }

    std::shared_ptr<SpatialDomains::SegGeom> edges_array[4] = {
        e.at(map_face_to_edges[fx][0]), e.at(map_face_to_edges[fx][1]),
        e.at(map_face_to_edges[fx][2]), e.at(map_face_to_edges[fx][3])};
    q[fx] = std::make_shared<SpatialDomains::QuadGeom>(fx, edges_array, cx);
    q[fx]->GetGeomFactors();
    q[fx]->Setup();
  }

  std::shared_ptr<SpatialDomains::QuadGeom> quads[6] = {q[0], q[1], q[2],
                                                        q[3], q[4], q[5]};

  auto hex = std::make_shared<SpatialDomains::HexGeom>(0, quads);
  hex->GetGeomFactors();
  hex->Setup();

  return hex;
}

} // namespace NESO

#endif
