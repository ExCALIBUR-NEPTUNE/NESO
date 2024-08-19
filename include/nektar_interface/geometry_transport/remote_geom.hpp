#ifndef __REMOTE_GEOM_H__
#define __REMOTE_GEOM_H__

// Nektar++ Includes
#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <cstddef>
#include <vector>

#include "geometry_packing_utility.hpp"
#include "shape_mapping.hpp"

#include <neso_particles.hpp>

namespace NESO::GeometryTransport {

/**
 *  Description of a geometry object that is owned by a remote rank.
 */
template <typename T>
class RemoteGeom : public Particles::MeshHierarchyData::SerialInterface {

protected:
  template <typename U>
  static inline void push_offset(std::size_t *offset, U *const data) {
    const std::size_t size = sizeof(U);
    *offset += size;
  }

  template <typename U>
  static inline void push(std::byte *buf, std::size_t *offset, U *const data) {
    const std::size_t size = sizeof(U);
    std::memcpy(buf + (*offset), data, size);
    *offset += size;
  }

  template <typename U>
  static inline void pop(const std::byte *buf, std::size_t *offset, U *data) {
    const std::size_t size = sizeof(U);
    std::memcpy(data, buf + (*offset), size);
    *offset += size;
  }

public:
  /// The remote rank that owns the geometry object (i.e. holds it in its
  /// MeshGraph).
  int rank = -1;
  /// The geometry id on the remote rank.
  int id = -1;
  /// A local copy of the geometry object.
  std::shared_ptr<T> geom;

  RemoteGeom() = default;

  /**
   *  Constructor for remote geometry object.
   *
   *  @param rank Remote rank that owns the object.
   *  @param id Remote id of this geometry object.
   *  @param geom Shared pointer to local copy of the geometry object.
   */
  RemoteGeom(int rank, int id, std::shared_ptr<T> geom)
      : rank(rank), id(id), geom(geom){};

  /**
   * Get the Nektar++ bounding box for the geometry object.
   *
   * @returns Bounding box.
   */
  std::array<NekDouble, 6> GetBoundingBox() {
    return this->geom->GetBoundingBox();
  }

  /**
   * @returns The number of bytes required to serialise this instance.
   */
  virtual inline std::size_t get_num_bytes() const override {

    std::size_t offset = 0;
    auto shape_type = this->geom->GetShapeType();
    const int shape_type_int = shape_type_to_int(shape_type);
    GeomPackSpec gs;

    auto lambda_push_point = [&](auto point) {
      PointStruct ps;
      this->push_offset(&offset, &ps);
    };

    // Push the members which are not the geom
    this->push_offset(&offset, &this->rank);
    this->push_offset(&offset, &this->id);
    this->push_offset(&offset, &shape_type_int);

    // Push the description of the geom
    int gid = -1;
    this->push_offset(&offset, &gid);
    if (shape_type == LibUtilities::ShapeType::eSegment) {

      const int coordim = -1;
      this->push_offset(&offset, &coordim);
      const int num_verts = geom->GetNumVerts();
      this->push_offset(&offset, &num_verts);
      for (int vx = 0; vx < num_verts; vx++) {
        auto point = geom->GetVertex(vx);
        lambda_push_point(point);
      }
      // curve of the edge
      auto curve = geom->GetCurve();
      ASSERTL0(curve == nullptr, "Not implemented for curved edges");
      // A curve with n_points = -1 will be a taken as non-existant.
      gs.a = 0;
      gs.b = 0;
      gs.n_points = -1;
      this->push_offset(&offset, &gs);

    } else if ((shape_type == LibUtilities::ShapeType::eTriangle) ||
               (shape_type == LibUtilities::ShapeType::eQuadrilateral)) {
      TODO
    } else { // Assume a 3D geom
    }

    return offset;
  }

  /**
   * Serialise this instance into the provided space.
   *
   * @param buffer[in, out] Pointer to space that the calling function
   * guarantees to be at least get_num_bytes in size.
   * @param num_bytes Size of allocated buffer passed (get_num_bytes).
   */
  virtual inline void serialise(std::byte *buffer,
                                const std::size_t num_bytes) const override {
    std::size_t offset = 0;
    auto shape_type = this->geom->GetShapeType();
    const int shape_type_int = shape_type_to_int(shape_type);
    GeomPackSpec gs;

    auto lambda_push_point = [&](auto point) {
      PointStruct ps;
      ps.coordim = point->GetCoordim();
      ps.vid = point->GetVid();
      point->GetCoords(ps.x, ps.y, ps.z);
      this->push(buffer, &offset, &ps);
    };

    // Push the members which are not the geom
    this->push(buffer, &offset, &this->rank);
    this->push(buffer, &offset, &this->id);
    this->push(buffer, &offset, &shape_type_int);

    auto lambda_push_edge = [&](auto edge) {
      int gid = edge->GetGlobalID();
      this->push(buffer, &offset, &gid);
      const int coordim = edge->GetCoordim();
      this->push(buffer, &offset, &coordim);
      const int num_verts = edge->GetNumVerts();
      this->push(buffer, &offset, &num_verts);
      for (int vx = 0; vx < num_verts; vx++) {
        auto point = edge->GetVertex(vx);
        lambda_push_point(point);
      }
      // curve of the edge
      auto curve = edge->GetCurve();
      ASSERTL0(curve == nullptr, "Not implemented for curved edges");
      // A curve with n_points = -1 will be a taken as non-existant.
      gs.a = 0;
      gs.b = 0;
      gs.n_points = -1;
      this->push(buffer, &offset, &gs);
    };

    auto lambda_push_face = [&](auto face) {
      int gid = face->GetGlobalID();
      this->push(buffer, &offset, &gid);
      const int num_edges = face->GetNumEdges();
      for (int ex = 0; ex < num_edges; ex++) {
        auto edge = face->GetEdge(ex);
        lambda_push_edge(edge);
      }
      // curve of the face
      auto curve = face->GetCurve();
      ASSERTL0(curve == nullptr, "Not implemented for curved edges");
      // A curve with n_points = -1 will be a taken as non-existant.
      gs.a = 0;
      gs.b = 0;
      gs.n_points = -1;
      this->push(buffer, &offset, &gs);
    };

    // Push the description of the geom
    if (shape_type == LibUtilities::ShapeType::eSegment) {
      lambda_push_edge(geom);
    } else if ((shape_type == LibUtilities::ShapeType::eTriangle) ||
               (shape_type == LibUtilities::ShapeType::eQuadrilateral)) {
      lambda_push_face(geom);
    } else { // Assume a 3D geom
             // TODO
    }

    NESOASSERT(offset == num_bytes, "Different offset from expected value.");
  }

  /**
   * Deserialise, i.e. reconstruct, an instance of the class from the byte
   * buffer.
   *
   * @param buffer Pointer to space that the calling function guarantees to be
   * at least get_num_bytes in size from which this object should be recreated.
   * @param num_bytes Size of allocated buffer passed (get_num_bytes).
   */
  virtual inline void deserialise(const std::byte *buffer,
                                  const std::size_t num_bytes) override {

    std::size_t offset = 0;
    GeomPackSpec gs;
    PointStruct ps;
    int shape_type_int;
    std::vector<SpatialDomains::Geometry2D> faces;
    std::vector<SpatialDomains::PointGeomSharedPtr> vertices;

    this->pop(buffer, &offset, &this->rank);
    this->pop(buffer, &offset, &this->id);
    this->pop(buffer, &offset, &shape_type_int);
    auto shape_type = int_to_shape_type(shape_type_int);

    auto lambda_pop_edge = [&]() {
      int gid;
      this->pop(buffer, &offset, &gid);

      int coordim;
      this->pop(buffer, &offset, &coordim);
      int num_verts;
      this->pop(buffer, &offset, &num_verts);

      for (int vx = 0; vx < num_verts; vx++) {
        this->pop(buffer, &offset, &ps);
        vertices.push_back(std::make_shared<SpatialDomains::PointGeom>(
            ps.coordim, ps.vid, ps.x, ps.y, ps.z));
      }
      // In future the edge might have a corresponding curve
      this->pop(buffer, &offset, &gs);
      ASSERTL0(gs.n_points == -1, "unpacking routine did not expect a curve");
      auto g = std::make_shared<SpatialDomains::SegGeom>(gid, coordim,
                                                         vertices.data());
      g->GetGeomFactors();
      g->Setup();
      return g;
    };

    auto lambda_pop_face = [&](const auto shape_type) {
      std::vector<SpatialDomains::SegGeomSharedPtr> edges;
      int gid;
      this->pop(buffer, &offset, &gid);
      int num_edges;
      this->pop(buffer, &offset, &num_edges);

      for (int ex = 0; ex < num_edges; ex++) {
        edges.push_back(lambda_pop_edge());
      }
      // curve of the face
      this->pop(buffer, &offset, &gs);
      ASSERTL0(gs.n_points == -1, "unpacking routine did not expect a curve");

      std::shared_ptr<Geometry2D> g;
      if (shape_type == LibUtilities::ShapeType::eTriangle) {
        g = std::dynamic_pointer_cast<Geometry2D>(
            std::make_shared<TriGeom>(gid, edges.data()));
      } else {
        g = std::dynamic_pointer_cast<Geometry2D>(
            std::make_shared<QuadGeom>(gid, edges.data()));
      }
      g->GetGeomFactors();
      g->Setup();
      return g;
    };

    if (shape_type == LibUtilities::ShapeType::eSegment) {
      this->geom = std::dynamic_pointer_cast<T>(lambda_pop_edge());
    } else if ((shape_type == LibUtilities::ShapeType::eTriangle) ||
               (shape_type == LibUtilities::ShapeType::eQuadrilateral)) {
      this->geom = std::dynamic_pointer_cast<T>(lambda_pop_face());
    } else { // Assume a 3D geom
      NESOASSERT(false, "not implemented yet");
    }

    this->geom->GetGeomFactors();
    this->geom->Setup();
    NESOASSERT(offset == num_bytes, "Not all data was deserialised");
  }
};

} // namespace NESO::GeometryTransport

#endif
