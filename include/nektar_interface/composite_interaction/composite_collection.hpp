#ifndef __COMPOSITE_COLLECTION_H_
#define __COMPOSITE_COLLECTION_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <neso_particles.hpp>
using namespace NESO::Particles;

#include <nektar_interface/geometry_transport/packed_geom_2d.hpp>
#include <nektar_interface/particle_mesh_interface.hpp>

#include "line_plane_intersection.hpp"

#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace NESO::CompositeInteraction {

/**
 * Struct pointed to by a BlockedBinaryTree for each MeshHierarchy cell.
 */
struct CompositeCollection {
  int num_quads;
  int num_tris;
  LinePlaneIntersection *lpi_quads;
  LinePlaneIntersection *lpi_tris;
  size_t stride_quads;
  size_t stride_tris;
  unsigned char *buf_quads;
  unsigned char *buf_tris;
  int *composite_ids_quads;
  int *composite_ids_tris;
  int *geom_ids_quads;
  int *geom_ids_tris;
};

} // namespace NESO::CompositeInteraction

#endif
