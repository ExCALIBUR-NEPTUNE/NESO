#ifndef __PARTICLE_CELL_MAPPING_COMMON_H__
#define __PARTICLE_CELL_MAPPING_COMMON_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

#include "../particle_mesh_interface.hpp"
#include "map_particles_common.hpp"
#include "nektar_interface/parameter_store.hpp"
#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar;
using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {

#ifndef MAPPING_CROSS_PRODUCT_3D
#define MAPPING_CROSS_PRODUCT_3D(a1, a2, a3, b1, b2, b3, c1, c2, c3)           \
  c1 = ((a2) * (b3)) - ((a3) * (b2));                                          \
  c2 = ((a3) * (b1)) - ((a1) * (b3));                                          \
  c3 = ((a1) * (b2)) - ((a2) * (b1));
#endif

#ifndef MAPPING_DOT_PRODUCT_3D
#define MAPPING_DOT_PRODUCT_3D(a1, a2, a3, b1, b2, b3)                         \
  ((a1) * (b1) + (a2) * (b2) + (a3) * (b3))
#endif

/**
 *  Map a global coordinate to a local coordinate in reference space (xi).
 *
 *  @param geom 2D Geometry object to map.
 *  @param coords Global coordinates of point (physical space).
 *  @param Lcoords Local coordinates (xi) in reference space.
 *  @returns Maximum distance from geometry object to point (in refence space)
 * if not contained.
 */
template <typename T>
inline double get_local_coords_2d(std::shared_ptr<T> geom,
                                  const Array<OneD, const NekDouble> &coords,
                                  Array<OneD, NekDouble> &Lcoords) {

  NESOASSERT(geom->GetMetricInfo()->GetGtype() == eRegular,
             "Not a regular geometry object");

  int last_point_index = -1;
  if (geom->GetShapeType() == LibUtilities::eTriangle) {
    last_point_index = 2;
  } else if (geom->GetShapeType() == LibUtilities::eQuadrilateral) {
    last_point_index = 3;
  } else {
    NESOASSERT(false, "get_local_coords_2d Unknown shape type.");
  }

  NESOASSERT(geom->GetCoordim() == 2, "Expected coordim == 2");

  const double last_coord = (geom->GetCoordim() == 3) ? coords[2] : 0.0;
  const double r0 = coords[0];
  const double r1 = coords[1];
  const double r2 = last_coord;

  const auto v0 = geom->GetVertex(0);
  const auto v1 = geom->GetVertex(1);
  const auto v2 = geom->GetVertex(last_point_index);

  const double er_0 = r0 - (*v0)[0];
  const double er_1 = r1 - (*v0)[1];
  const double er_2 = (v0->GetCoordim() == 3) ? r2 - (*v0)[2] : 0.0;

  const double e10_0 = (*v1)[0] - (*v0)[0];
  const double e10_1 = (*v1)[1] - (*v0)[1];
  const double e10_2 = (v0->GetCoordim() == 3 && v1->GetCoordim() == 3)
                           ? (*v1)[2] - (*v0)[2]
                           : 0.0;

  const double e20_0 = (*v2)[0] - (*v0)[0];
  const double e20_1 = (*v2)[1] - (*v0)[1];
  const double e20_2 = (v0->GetCoordim() == 3 && v2->GetCoordim() == 3)
                           ? (*v2)[2] - (*v0)[2]
                           : 0.0;

  MAPPING_CROSS_PRODUCT_3D(e10_0, e10_1, e10_2, e20_0, e20_1, e20_2,
                           const double norm_0, const double norm_1,
                           const double norm_2)
  MAPPING_CROSS_PRODUCT_3D(norm_0, norm_1, norm_2, e10_0, e10_1, e10_2,
                           const double orth1_0, const double orth1_1,
                           const double orth1_2)
  MAPPING_CROSS_PRODUCT_3D(norm_0, norm_1, norm_2, e20_0, e20_1, e20_2,
                           const double orth2_0, const double orth2_1,
                           const double orth2_2)

  const double scale0 =
      MAPPING_DOT_PRODUCT_3D(er_0, er_1, er_2, orth2_0, orth2_1, orth2_2) /
      MAPPING_DOT_PRODUCT_3D(e10_0, e10_1, e10_2, orth2_0, orth2_1, orth2_2);
  Lcoords[0] = 2.0 * scale0 - 1.0;
  const double scale1 =
      MAPPING_DOT_PRODUCT_3D(er_0, er_1, er_2, orth1_0, orth1_1, orth1_2) /
      MAPPING_DOT_PRODUCT_3D(e20_0, e20_1, e20_2, orth1_0, orth1_1, orth1_2);
  Lcoords[1] = 2.0 * scale1 - 1.0;

  double eta0 = -2;
  double eta1 = -2;
  if (geom->GetShapeType() == LibUtilities::eTriangle) {
    NekDouble d1 = 1. - Lcoords[1];
    if (fabs(d1) < NekConstants::kNekZeroTol) {
      if (d1 >= 0.) {
        d1 = NekConstants::kNekZeroTol;
      } else {
        d1 = -NekConstants::kNekZeroTol;
      }
    }
    eta0 = 2. * (1. + Lcoords[0]) / d1 - 1.0;
    eta1 = Lcoords[1];

  } else if (geom->GetShapeType() == LibUtilities::eQuadrilateral) {
    eta0 = Lcoords[0];
    eta1 = Lcoords[1];
  }

  double dist = 0.0;
  bool contained =
      ((eta0 <= 1.0) && (eta0 >= -1.0) && (eta1 <= 1.0) && (eta1 >= -1.0));
  if (!contained) {
    dist = (eta0 < -1.0) ? (-1.0 - eta0) : 0.0;
    dist = std::max(dist, (eta0 > 1.0) ? (eta0 - 1.0) : 0.0);
    dist = std::max(dist, (eta1 < -1.0) ? (-1.0 - eta1) : 0.0);
    dist = std::max(dist, (eta1 > 1.0) ? (eta1 - 1.0) : 0.0);
  }

  return dist;
}

/**
 *  Test if a 2D Geometry object contains a point. Returns the computed
 * reference coordinate (xi).
 *
 *  @param geom 2D Geometry object, e.g. QuadGeom, TriGeom.
 *  @param global_coord Global coordinate to map to local coordinate.
 *  @param local_coord Output, computed locate coordinate in reference space.
 *  @param tol Input tolerance for geometry containing point.
 */
template <typename T>
inline bool
contains_point_2d(std::shared_ptr<T> geom, Array<OneD, NekDouble> &global_coord,
                  Array<OneD, NekDouble> &local_coord, const NekDouble tol) {
  if (geom->GetMetricInfo()->GetGtype() == eRegular) {
    const double dist = get_local_coords_2d(geom, global_coord, local_coord);
    bool contained = dist <= tol;
    return contained;
  } else {
    return geom->ContainsPoint(global_coord, local_coord, tol);
  }
}

/**
 *  Test if a 3D Geometry object contains a point. Returns the computed
 * reference coordinate (xi).
 *
 *  @param geom 3D Geometry object.
 *  @param global_coord Global coordinate to map to local coordinate.
 *  @param local_coord Output, computed locate coordinate in reference space.
 *  @param tol Input tolerance for geometry containing point.
 */
template <typename T>
inline bool
contains_point_3d(std::shared_ptr<T> geom, Array<OneD, NekDouble> &global_coord,
                  Array<OneD, NekDouble> &local_coord, const NekDouble tol) {
  bool contained = geom->ContainsPoint(global_coord, local_coord, tol);
  return contained;
}

inline Geometry3DSharedPtr get_geometry_3d(MeshGraphSharedPtr graph,
                                           const int geom_id) {
  {
    auto geoms0 = graph->GetAllTetGeoms();
    auto it0 = geoms0.find(geom_id);
    if (it0 != geoms0.end()) {
      return it0->second;
    }
  }
  {
    auto geoms1 = graph->GetAllPyrGeoms();
    auto it1 = geoms1.find(geom_id);
    if (it1 != geoms1.end()) {
      return it1->second;
    }
  }
  {
    auto geoms2 = graph->GetAllPrismGeoms();
    auto it2 = geoms2.find(geom_id);
    if (it2 != geoms2.end()) {
      return it2->second;
    }
  }
  {
    auto geoms3 = graph->GetAllHexGeoms();
    auto it3 = geoms3.find(geom_id);
    if (it3 != geoms3.end()) {
      return it3->second;
    }
  }

  NESOASSERT(false, "Could not find geom in graph.");
  return nullptr;
}

/**
 *  Count the number of Regular and Deformed geometry objects in a container.
 *  Counts are incremented.
 *
 *  @param[in] geoms Container of geometry objects. Either RemoteGeom2D or
 * RemoteGeom3D.
 *  @param[in, out] count_regular Number of regular geometry objects (eRegular).
 *  @param[in, out] count_deformed Number of deformed geometry objects
 * (eDeformed).
 *
 */
template <typename T>
inline void count_geometry_types(std::vector<T> &geoms, int *count_regular,
                                 int *count_deformed) {

  for (auto &geom : geoms) {
    auto t = geom->geom->GetMetricInfo()->GetGtype();
    if (t == eRegular) {
      (*count_regular)++;
    } else if (t == eDeformed) {
      (*count_deformed)++;
    } else {
      NESOASSERT(false, "Unknown geometry type - not Regular or Deformed.");
    }
  }
}

/**
 *  Count the number of Regular and Deformed geometry objects in a container.
 *  Counts are incremented.
 *
 *  @param[in] geoms Container of geometry objects.
 *  @param[in, out] count_regular Number of regular geometry objects (eRegular).
 *  @param[in, out] count_deformed Number of deformed geometry objects
 * (eDeformed).
 *
 */
template <typename T>
inline void count_geometry_types(std::map<int, T> &geoms, int *count_regular,
                                 int *count_deformed) {

  for (auto &geom : geoms) {
    auto t = geom.second->GetMetricInfo()->GetGtype();
    if (t == eRegular) {
      (*count_regular)++;
    } else if (t == eDeformed) {
      (*count_deformed)++;
    } else {
      NESOASSERT(false, "Unknown geometry type - not Regular or Deformed.");
    }
  }
}

} // namespace NESO

#endif
