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

} // namespace NESO

#endif
