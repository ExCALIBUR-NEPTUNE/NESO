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
 * Struct to hold local and remote 3D geoms.
 */
struct GeometryLocalRemote3D {
  /// Local geometry objects which are owned by this MPI rank.
  std::map<int, std::shared_ptr<Geometry3D>> local;
  /// Remote geometry objects where a copy is stored on this MPI rank.
  std::vector<std::shared_ptr<RemoteGeom3D>> remote;

  /**
   * Push a geometry object onto the container depending on if the geometry
   * object is local or remote.
   *
   * @param geom Geometry object.
   */
  inline void push_back(std::pair<int, std::shared_ptr<Geometry3D>> geom) {
    this->local[geom.first] = geom.second;
  }

  /**
   * Push a geometry object onto the container depending on if the geometry
   * object is local or remote.
   *
   * @param geom Geometry object.
   */
  inline void push_back(std::shared_ptr<RemoteGeom3D> geom) {
    this->remote.push_back(geom);
  }

  /**
   *  Returns number of remote and local geometry objects.
   */
  inline size_t size() { return this->local.size() + this->remote.size(); }
};

/**
 * Struct to holder shared pointers to the different types of 3D geometry
 * objects in terms of classification of shape.
 */
class GeometryTypes3D {
protected:
  inline GeometryLocalRemote3D &classify(std::shared_ptr<Geometry3D> &geom) {

    auto shape_type = geom->GetShapeType();
    if (shape_type == eTetrahedron) {
      return this->tet;
    } else if (shape_type == ePyramid) {
      return this->pyr;
    } else if (shape_type == ePrism) {
      return this->prism;
    } else if (shape_type == eHexahedron) {
      return this->hex;
    } else {
      NESOASSERT(false, "could not classify geometry type");
      return this->tet; // supresses warnings, unreachable.
    }
  }

public:
  /// Store of local and remote tetrahedrons.
  GeometryLocalRemote3D tet;
  /// Store of local and remote pyramids.
  GeometryLocalRemote3D pyr;
  /// Store of local and remote prism.
  GeometryLocalRemote3D prism;
  /// Store of local and remote hexahedrons.
  GeometryLocalRemote3D hex;

  /**
   * Push a geometry object onto the correct container.
   *
   * @param geom Geometry object to push onto correct container.
   */
  inline void push_back(std::pair<int, std::shared_ptr<Geometry3D>> &geom) {
    auto &container = this->classify(geom.second);
    container.push_back(geom);
  }
  /**
   * Push a geometry object onto the correct container.
   *
   * @param geom Geometry object to push onto correct container.
   */
  inline void push_back(std::shared_ptr<RemoteGeom3D> &geom) {
    auto &container = this->classify(geom->geom);
    container.push_back(geom);
  }

  /**
   *  @returns Number of elements accross all types.
   */
  inline size_t size() {
    return this->tet.size() + this->pyr.size() + this->prism.size() +
           this->hex.size();
  }
};

/**
 * Struct to holder shared pointers to the different types of 3D geometry
 * objects in terms of classification of shape and type, e.g. Regular,
 * Deformed, linear, non-linear.
 */
class GeometryContainer3D {
protected:
  inline GeometryTypes3D &classify(std::shared_ptr<Geometry3D> &geom) {

    auto g_type = geom->GetMetricInfo()->GetGtype();
    if (g_type == eRegular) {
      return this->regular;
    } else {
      const std::map<LibUtilities::ShapeType, int> expected_num_verts{
          {{eTetrahedron, 4}, {ePyramid, 5}, {ePrism, 6}, {eHexahedron, 8}}};
      const int linear_num_verts = expected_num_verts.at(geom->GetShapeType());
      const int num_verts = geom->GetNumVerts();
      if (num_verts == linear_num_verts) {
        return this->deformed_linear;
      } else {
        return this->deformed_non_linear;
      }
    }
  }

public:
  /// Elements with linear sides that are considered eRegular by Nektar++.
  GeometryTypes3D regular;
  /// Elements with linear sides that are considered eDeformed by Nektar++.
  GeometryTypes3D deformed_linear;
  /// Elements with non-linear sides that are considered eDeformed by Nektar++.
  GeometryTypes3D deformed_non_linear;

  /**
   * Push a geometry object onto the correct container.
   *
   * @param geom Geometry object to push onto correct container.
   */
  inline void push_back(std::pair<int, std::shared_ptr<Geometry3D>> geom) {
    auto &container = this->classify(geom.second);
    container.push_back(geom);
  }
  /**
   * Push a geometry object onto the correct container.
   *
   * @param geom Geometry object to push onto correct container.
   */
  inline void push_back(std::shared_ptr<RemoteGeom3D> &geom) {
    auto &container = this->classify(geom->geom);
    container.push_back(geom);
  }
};

/**
 * Catagorise geometry types by shape, local or remote and X-map type.
 *
 * @param[in] graph Nektar MeshGraph of locally owned geometry objects.
 * @param[in] remote_geoms_3d Vector of remotely owned geometry objects.
 * @param[in, out] output_container Geometry objects cataorised by type..
 */
inline void assemble_geometry_container_3d(
    Nektar::SpatialDomains::MeshGraphSharedPtr &graph,
    std::vector<std::shared_ptr<RemoteGeom3D>> &remote_geoms_3d,
    GeometryContainer3D &output_container) {

  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry3D>> geoms;
  get_all_elements_3d(graph, geoms);
  for (auto geom : geoms) {
    output_container.push_back(geom);
  }
  for (auto geom : remote_geoms_3d) {
    output_container.push_back(geom);
  }
}

/**
 * Get the unqiue integer (cast) that corresponds to a Nektar++ shape type.
 *
 * @param shape_type ShapeType Enum value.
 * @returns Static cast of enum value.
 */
inline constexpr int
shape_type_to_int(Nektar::LibUtilities::ShapeType shape_type) {
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
 * the owning rank and shape type.
 *
 * @param[in] rank MPI rank that owns the 3D object.
 * @param[in] geometry_id Global id of geometry object.
 * @param[in] geom 3D geometry object.
 * @param[in,out] deconstructed_geoms Output vector to push description onto.
 */
inline void deconstruct_geoms_base_3d(const int rank, const int geometry_id,
                                      std::shared_ptr<Geometry3D> geom,
                                      std::vector<int> &deconstructed_geoms) {
  deconstructed_geoms.push_back(rank);
  deconstructed_geoms.push_back(geometry_id);
  deconstructed_geoms.push_back(shape_type_to_int(geom->GetShapeType()));
  const int num_faces = geom->GetNumFaces();
  for (int facex = 0; facex < num_faces; facex++) {
    auto geom_2d = geom->GetFace(facex);
    // record this 2D geom as one to be sent to this remote rank
    const int geom_2d_gid = geom_2d->GetGlobalID();
    // push this face onto the construction list
    deconstructed_geoms.push_back(geom_2d_gid);
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
  deconstruct_geoms_base_3d(rank, geometry_id, geom, deconstructed_geoms);
  const int num_faces = geom->GetNumFaces();
  for (int facex = 0; facex < num_faces; facex++) {
    auto geom_2d = geom->GetFace(facex);
    const int geom_2d_gid = geom_2d->GetGlobalID();
    face_ids.insert(geom_2d_gid);
  }
}

/**
 *  Deconstruct, i.e. serialise to integers, 3D Nektar++ geometry objects such
 *  that they can be sent over MPI. Geometry objects are packed into an output
 *  map for each remote MPI rank.
 *
 *  @param[in] original_rank MPI rank which owns the geometry objects in
 * `rank_element_map`.
 *  @param[in] geoms_2d Map from geometry id to 2D geometry objects required by
 *  the 3D geometry objects which are t o be packed.
 *  @param[in] rank_element_map Map from remote MPI rank to map from geometry id
 * to 3D geometry object to pack for the remote rank.
 *  @param[in] send_ranks Vector of remote MPI ranks.
 *  @param[in, out] send_sizes Output vector of packed sizes per MPI rank.
 *  @param[in, out] deconstructed_geoms Map from remote MPI rank to vector of
 *  ints describing the packed 3D geoms in terms of the 2D geometry ids that
 *  build the faces of the 3D object.
 *  @param[in, out] rank_triangle_map Map from remote MPI rank to TriGeom ids
 * and objects to pack and send.
 *  @param[in, out] rank_quad Map from remote MPI rank to QuadGeom ids and
 * objects to pack and send.
 */
inline void deconstuct_per_rank_geoms_3d(
    const int original_rank,
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>
        &geoms_2d,
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
      deconstruct_geoms_3d(original_rank, geom_pair.first, geom_pair.second,
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
      const auto geom = geoms_2d.at(geom_id);
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

/**
 *  Exchange 3D geometry objects between MPI ranks. Must be called collectively
 * on the communicator.
 *
 *  @param[in] comm MPI communicator.
 *  @param[in] deconstructed_geoms Map from remote MPI rank to packed 3D
 * geometry objects to send to that MPI rank.
 *  @param[in] send_ranks Vector of MPI ranks to send 3D geometry objects to.
 *  @param[in] send_sizes The number of ints to send to each remote MPI rank.
 *  @param[in] num_recv_ranks Number of MPI ranks to receive 3D geometry objects
 * from.
 *  @param[in] recv_rank Vector of MPI ranks to receive 3D geometry objects
 * from.
 *  @param[in] recv_sizes Vector containing the number of ints expected to be
 * received from each remote MPI rank.
 *  @param[in, out] packed_geoms Received 3D geometry objects in packed int
 * form.
 */
inline void
sendrecv_geoms_3d(MPI_Comm comm,
                  std::map<int, std::vector<int>> &deconstructed_geoms,
                  const int num_send_ranks, std::vector<int> &send_ranks,
                  std::vector<int> &send_sizes, const int num_recv_ranks,
                  std::vector<int> &recv_ranks, std::vector<int> &recv_sizes,
                  std::vector<int> &packed_geoms) {

  int recv_total_size = 0;
  std::vector<int> recv_offsets(num_recv_ranks);
  int recv_index = 0;
  for (const int rank : recv_ranks) {
    const int rank_recv_count = recv_sizes[recv_index];
    recv_offsets[recv_index] = recv_total_size;
    recv_total_size += rank_recv_count;
    recv_index++;
  }
  packed_geoms.resize(recv_total_size);

  // exchange deconstructed 3D geom descriptions
  std::vector<MPI_Request> recv_requests(num_recv_ranks);
  // non-blocking recv packed geoms
  for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
    const int offset = recv_offsets.at(rankx);
    const int num_ints = recv_sizes.at(rankx);
    const int remote_rank = recv_ranks.at(rankx);
    MPICHK(MPI_Irecv(packed_geoms.data() + offset, num_ints, MPI_INT,
                     remote_rank, 135, comm, recv_requests.data() + rankx));
  }
  // send geoms to remote ranks
  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    const int remote_rank = send_ranks.at(rankx);
    const int num_ints = send_sizes.at(rankx);
    MPICHK(MPI_Send(deconstructed_geoms.at(remote_rank).data(), num_ints,
                    MPI_INT, remote_rank, 135, comm));
  }
  // wait for geoms to be recvd
  MPICHK(
      MPI_Waitall(num_recv_ranks, recv_requests.data(), MPI_STATUSES_IGNORE));
}

} // namespace NESO

#endif
