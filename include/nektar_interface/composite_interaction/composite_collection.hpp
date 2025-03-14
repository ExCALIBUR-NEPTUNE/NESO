#ifndef __NEKTAR_INTERFACE_COMPOSITE_INTERACTION_COMPOSITE_COLLECTION_H_
#define __NEKTAR_INTERFACE_COMPOSITE_INTERACTION_COMPOSITE_COLLECTION_H_

#include "../particle_cell_mapping/newton_quad_embed_3d.hpp"
#include "../particle_cell_mapping/newton_triangle_embed_3d.hpp"
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
  Newton::MappingQuadLinear2DEmbed3D::DataDevice *buf_quads;
  Newton::MappingTriangleLinear2DEmbed3D::DataDevice *buf_tris;
  int *composite_ids_quads;
  int *composite_ids_tris;
  int *geom_ids_quads;
  int *geom_ids_tris;
  // Segment members
  int num_segments;
  LineLineIntersection *lli_segments;
  int *composite_ids_segments;
  int *geom_ids_segments;
  // group ids
  int *group_ids_quads;
  int *group_ids_tris;
  int *group_ids_segments;
};

} // namespace NESO::CompositeInteraction

#endif
