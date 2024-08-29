#ifndef __NEKTAR_INTERFACE_COMPOSITE_INTERACTION_COMPOSITE_COLLECTION_H_
#define __NEKTAR_INTERFACE_COMPOSITE_INTERACTION_COMPOSITE_COLLECTION_H_

#include "line_line_intersection.hpp"
#include "line_plane_intersection.hpp"

namespace NESO::CompositeInteraction {

/**
 * Struct pointed to by a BlockedBinaryTree for each MeshHierarchy cell.
 */
struct CompositeCollection {
  // Face members
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
  // Segment members
  int num_segments;
  LineLineIntersection *lli_segments;
  int *composite_ids_segments;
  int *geom_ids_segments;
};

} // namespace NESO::CompositeInteraction

#endif
