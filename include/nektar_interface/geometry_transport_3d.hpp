#ifndef __GEOMETRY_TRANSPORT_3D_H__
#define __GEOMETRY_TRANSPORT_3D_H__

// Nektar++ Includes
#include <SpatialDomains/MeshGraph.h>

// System includes
#include <iostream>
#include <mpi.h>

using namespace std;
using namespace Nektar;
using namespace Nektar::SpatialDomains;
using namespace Nektar::LibUtilities;

#include "geometry_transport_2d.hpp"

namespace NESO {

class RemoteGeom3D {
protected:
public:
  /// The remote rank that owns the geometry object (i.e. holds it in its
  /// MeshGraph).
  int rank = -1;
  /// The geometry id on the remote rank.
  int id = -1;
  /// A local copy of the geometry object.
  std::shared_ptr<Nektar::SpatialDomains::Geometry3D> geom;
  /// The underlying Nektar++ shape type
  LibUtilities::ShapeType shape_type;

  /**
   * Wrapper around remote 3D geometry object.
   *
   * @param rank Owning rank.
   * @param id Geometry id on remote rank.
   * @param geom Shared pointer to geometry object.
   */
  template <typename T>
  RemoteGeom3D(const int rank, const int id, std::shared_ptr<T> geom)
      : rank(rank), id(id),
        geom(std::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry3D>(
            geom)),
        shape_type(geom->GetShapeType()){};

  /**
   * Get the Nektar++ bounding box for the geometry object.
   *
   * @returns Bounding box.
   */
  std::array<NekDouble, 6> GetBoundingBox() {
    return this->geom->GetBoundingBox();
  }
};

/**
 * Get all 3D geometry objects from a Nektar++ MeshGraph
 *
 * @param[in] graph MeshGraph instance.
 * @param[in,out] std::map of Nektar++ Geometry3D pointers.
 */
inline void get_all_elements_3d(
    Nektar::SpatialDomains::MeshGraphSharedPtr &graph,
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> &geoms) {
  geoms.clear();

  for (auto &e : graph->GetAllTetGeoms()) {
    geoms[e.first] =
        std::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry3D>(e.second);
  }
  for (auto &e : graph->GetAllPyrGeoms()) {
    geoms[e.first] =
        std::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry3D>(e.second);
  }
  for (auto &e : graph->GetAllPrismGeoms()) {
    geoms[e.first] =
        std::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry3D>(e.second);
  }
  for (auto &e : graph->GetAllHexGeoms()) {
    geoms[e.first] =
        std::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry3D>(e.second);
  }
}

/**
 * Get the unqiue integer (cast) that corresponds to a Nektar++ shape type.
 *
 * @param shape_type ShapeType Enum value.
 * @returns Static cast of enum value.
 */
inline int shape_type_to_int(Nektar::LibUtilities::ShapeType shape_type) {
  return static_cast<int>(shape_type);
}

/**
 * Get the unqiue Enum (cast) that corresponds to an integer returned form
 * shape_type_to_int.
 *
 * @param type_int Int returned from shape_type_to_int.
 * @returns Static cast to Enum value.
 */
inline Nektar::LibUtilities::ShapeType int_to_shape_type(const int type_int) {
  return static_cast<Nektar::LibUtilities::ShapeType>(type_int);
}

/**
 * Create 3D geometry objects from integer descriptions in the format:
 * <owning rank>, <geometry id>, <faces as list of 2D geometry ids>
 *
 * @param[in,out] base_ptr Pointer to pointer of start of integer array. The
 * array pointer is incremented as the integers are consumed to create 3D
 * geometry objects.
 * @param[in] rank_element_map Map from MPI rank to a map from geometry id to
 * 2D geom shared pointer.
 */
inline std::shared_ptr<RemoteGeom3D> reconstruct_geom_3d(
    int **base_ptr,
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>>
        &rank_element_map_2d) {

  int *ptr = *base_ptr;
  const int rank = *ptr++;
  const int geom_id = *ptr++;
  const int shape_int = *ptr++;
  const Nektar::LibUtilities::ShapeType shape_type =
      int_to_shape_type(shape_int);
  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>
      &element_map = rank_element_map_2d[rank];

  switch (shape_type) {
  case ShapeType::eHexahedron: {
    std::vector<QuadGeomSharedPtr> faces(6);
    for (int facex = 0; facex < 6; facex++) {
      const int face_id = *ptr++;
      auto face_ptr = element_map[face_id];
      ASSERTL0(face_ptr != nullptr, "No face in element map.");
      ASSERTL0(face_ptr->GetGlobalID() == face_id, "Element id missmatch.");
      ASSERTL0(face_ptr->GetShapeType() == ShapeType::eQuadrilateral,
               "Element type missmatch.");
      faces[facex] = std::dynamic_pointer_cast<QuadGeom>(face_ptr);
    }
    auto new_geom = std::make_shared<HexGeom>(geom_id, faces.data());
    new_geom->GetGeomFactors();
    new_geom->Setup();
    *base_ptr = ptr;
    return std::make_shared<RemoteGeom3D>(rank, geom_id, new_geom);
  }
  case ShapeType::ePrism: {
    std::vector<Geometry2DSharedPtr> faces(5);
    for (int facex = 0; facex < 5; facex++) {
      const int face_id = *ptr++;
      auto face_ptr = element_map[face_id];
      ASSERTL0(face_ptr != nullptr, "No face in element map.");
      ASSERTL0(face_ptr->GetGlobalID() == face_id, "Element id missmatch.");
      faces[facex] = face_ptr;
    }
    auto new_geom = std::make_shared<PrismGeom>(geom_id, faces.data());
    new_geom->GetGeomFactors();
    new_geom->Setup();
    *base_ptr = ptr;
    return std::make_shared<RemoteGeom3D>(rank, geom_id, new_geom);
  }
  case ShapeType::ePyramid: {
    std::vector<Geometry2DSharedPtr> faces(5);
    for (int facex = 0; facex < 5; facex++) {
      const int face_id = *ptr++;
      auto face_ptr = element_map[face_id];
      ASSERTL0(face_ptr != nullptr, "No face in element map.");
      ASSERTL0(face_ptr->GetGlobalID() == face_id, "Element id missmatch.");
      faces[facex] = face_ptr;
    }
    auto new_geom = std::make_shared<PyrGeom>(geom_id, faces.data());
    new_geom->GetGeomFactors();
    new_geom->Setup();
    *base_ptr = ptr;
    return std::make_shared<RemoteGeom3D>(rank, geom_id, new_geom);
  }
  case ShapeType::eTetrahedron: {
    std::vector<TriGeomSharedPtr> faces(4);
    for (int facex = 0; facex < 4; facex++) {
      const int face_id = *ptr++;
      auto face_ptr = element_map[face_id];
      ASSERTL0(face_ptr != nullptr, "No face in element map.");
      ASSERTL0(face_ptr->GetGlobalID() == face_id, "Element id missmatch.");
      ASSERTL0(face_ptr->GetShapeType() == ShapeType::eTriangle,
               "Element type missmatch.");
      faces[facex] = std::dynamic_pointer_cast<TriGeom>(face_ptr);
    }
    auto new_geom = std::make_shared<TetGeom>(geom_id, faces.data());
    new_geom->GetGeomFactors();
    new_geom->Setup();
    *base_ptr = ptr;
    return std::make_shared<RemoteGeom3D>(rank, geom_id, new_geom);
  }
  default: {
    return nullptr;
  }
  }
}

/**
 * Reconstruct multiple 3D geometry objects from a vector of ints.
 *
 * @param[in] rank_element_map_2d Map from MPI rank to element map (geom id to
 * shared ptr) required to rebuild geoms.
 * @param[in] packed_geoms Int vector of 3D geometry objects described by 2D
 * geometry ids.
 * @param[in,out] output_container Output vector in which to place constructed
 * 3D geometry objects.
 */
inline void reconstruct_geoms_3d(
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>>
        &rank_element_map_2d,
    std::vector<int> &packed_geoms,
    std::vector<std::shared_ptr<RemoteGeom3D>> &output_container) {

  // rebuild the 3D geometry objects
  int *base_ptr = packed_geoms.data();
  int *end_ptr = base_ptr + packed_geoms.size();
  while (base_ptr < end_ptr) {
    auto new_geom_3d = reconstruct_geom_3d(&base_ptr, rank_element_map_2d);
    ASSERTL0(new_geom_3d != nullptr,
             "Could not recreate a 3D geometry object.");
    output_container.push_back(new_geom_3d);
  }
}

/**
 * Desconstruct a 3D geometry object into a series of integers that describe
 * the owning rank, shape type and constituent faces.
 *
 * @param[in] rank MPI rank that owns the 3D object.
 * @param[in] geometry_id Global id of geometry object.
 * @param[in] geom 3D geometry object.
 * @param[in,out] deconstructed_geoms Output vector to push description onto.
 * @param[in,out] face_ids Set of face ids to add faces to.
 */
inline void deconstruct_geoms_3d(const int rank, const int geometry_id,
                                 std::shared_ptr<Geometry3D> geom,
                                 std::vector<int> &deconstructed_geoms,
                                 std::set<int> &face_ids) {
  deconstructed_geoms.push_back(rank);
  deconstructed_geoms.push_back(geometry_id);
  deconstructed_geoms.push_back(shape_type_to_int(geom->GetShapeType()));
  const int num_faces = geom->GetNumFaces();
  for (int facex = 0; facex < num_faces; facex++) {
    auto geom_2d = geom->GetFace(facex);
    // record this 2D geom as one to be sent to this remote rank
    const int geom_2d_gid = geom_2d->GetGlobalID();
    face_ids.insert(geom_2d_gid);
    // push this face onto the construction list
    deconstructed_geoms.push_back(geom_2d_gid);
  }
}

/**
 *  TODO
 */
inline void deconstuct_per_rank_geoms_3d(
    const int comm_rank,
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>
        &geoms_2d,
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>
        &geoms_3d,
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>>>
        &rank_element_map,
    const int num_send_ranks, std::vector<int> &send_ranks,
    std::vector<int> &send_sizes,
    std::map<int, std::vector<int>> &deconstructed_geoms,
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::TriGeom>>>
        &rank_triangle_map,
    std::map<int,
             std::map<int, std::shared_ptr<Nektar::SpatialDomains::QuadGeom>>>
        &rank_quad_map) {

  // the face ids to be sent to each remote rank
  std::map<int, std::set<int>> face_ids;
  int rank_index = 0;
  for (int rank : send_ranks) {
    deconstructed_geoms[rank].reserve(7 * rank_element_map[rank].size());
    for (auto &geom_pair : rank_element_map[rank]) {
      deconstruct_geoms_3d(comm_rank, geom_pair.first, geom_pair.second,
                           deconstructed_geoms[rank], face_ids[rank]);
    }
    send_sizes[rank_index++] = deconstructed_geoms[rank].size();
  }

  //  We have now collected the set of 2D geoms to send to each remote rank
  //  that are required to recreate the 3D geoms we will send to each rank.
  //  The deconstructed geoms array describes how to recreate the 3D geoms.

  // The 2D exchange routines exchange triangles and quads seperately so we
  // create the seperate maps based on the 2D geom id sets

  for (int rank : send_ranks) {
    for (const int &geom_id : face_ids[rank]) {
      ASSERTL0(geoms_2d.count(geom_id) == 1,
               "Geometry id not found in geoms_2d.");
      const auto geom = geoms_2d[geom_id];
      const auto shape_type = geom->GetShapeType();
      if (shape_type == LibUtilities::ShapeType::eQuadrilateral) {
        rank_quad_map[rank][geom_id] =
            std::dynamic_pointer_cast<QuadGeom>(geom);
      } else if (shape_type == LibUtilities::ShapeType::eTriangle) {
        rank_triangle_map[rank][geom_id] =
            std::dynamic_pointer_cast<TriGeom>(geom);
      } else {
        ASSERTL0(false, "Unknown 2D shape type.");
      }
    }
  }
}

} // namespace NESO

#endif
