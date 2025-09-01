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
  static inline void push(std::byte *buf, std::size_t *offset,
                          const std::vector<U> &data) {
    const std::size_t size = sizeof(U) * data.size();
    std::memcpy(buf + (*offset), data.data(), size);
    *offset += size;
  }

  template <typename U>
  static inline void pop(const std::byte *buf, std::size_t *offset, U *data) {
    const std::size_t size = sizeof(U);
    std::memcpy(data, buf + (*offset), size);
    *offset += size;
  }

  template <typename U>
  static inline void pop(const std::byte *buf, std::size_t *offset,
                         std::vector<U> &data) {
    const std::size_t size = sizeof(U) * data.size();
    std::memcpy(data.data(), buf + (*offset), size);
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
  /// Additional int properties.
  std::vector<int> aux_int_properties;

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
    const int num_aux_int_properties = this->aux_int_properties.size();
    this->push_offset(&offset, &num_aux_int_properties);
    offset += num_aux_int_properties * sizeof(int);

    auto lambda_push_edge = [&](auto edge) {
      int gid = -1;
      this->push_offset(&offset, &gid);
      const int coordim = -1;
      this->push_offset(&offset, &coordim);
      const int num_verts = edge->GetNumVerts();
      this->push_offset(&offset, &num_verts);
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
      this->push_offset(&offset, &gs);
    };

    auto lambda_push_face = [&](auto face) {
      int gid = face->GetGlobalID();
      this->push_offset(&offset, &gid);
      const int num_edges = face->GetNumEdges();
      this->push_offset(&offset, &num_edges);
      for (int ex = 0; ex < num_edges; ex++) {
        auto edge = std::dynamic_pointer_cast<SegGeom>(face->GetEdge(ex));
        NESOASSERT(edge.get() != nullptr,
                   "Face edge could not be cast to SegGeom");
        lambda_push_edge(edge);
      }
      // curve of the face
      auto curve = face->GetCurve();
      ASSERTL0(curve == nullptr, "Not implemented for curved edges");
      // A curve with n_points = -1 will be a taken as non-existant.
      gs.a = 0;
      gs.b = 0;
      gs.n_points = -1;
      this->push_offset(&offset, &gs);
    };

    auto lambda_push_polyhedron = [&](auto poly) {
      const int gid = -1;
      this->push_offset(&offset, &gid);
      const int num_faces = poly->GetNumFaces();
      this->push_offset(&offset, &num_faces);
      for (int fx = 0; fx < num_faces; fx++) {
        auto face = poly->GetFace(fx);
        const int face_shape_type_int = -1;
        this->push_offset(&offset, &face_shape_type_int);
        lambda_push_face(face);
      }
    };

    if (shape_type == LibUtilities::ShapeType::eSegment) {
      lambda_push_edge(std::dynamic_pointer_cast<SegGeom>(geom));
    } else if ((shape_type == LibUtilities::ShapeType::eTriangle) ||
               (shape_type == LibUtilities::ShapeType::eQuadrilateral)) {
      lambda_push_face(std::dynamic_pointer_cast<Geometry2D>(geom));
    } else { // Assume a 3D geom
      lambda_push_polyhedron(std::dynamic_pointer_cast<Geometry3D>(geom));
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
    const int num_aux_int_properties = this->aux_int_properties.size();
    this->push(buffer, &offset, &num_aux_int_properties);
    this->push(buffer, &offset, this->aux_int_properties);

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
      const int gid = face->GetGlobalID();
      this->push(buffer, &offset, &gid);
      const int num_edges = face->GetNumEdges();
      this->push(buffer, &offset, &num_edges);
      for (int ex = 0; ex < num_edges; ex++) {
        // The TriGeoms and QuadGeoms are constructed with SegGeoms so this
        // should be fine.
        auto edge = std::dynamic_pointer_cast<SegGeom>(face->GetEdge(ex));
        NESOASSERT(edge.get() != nullptr,
                   "Face edge could not be cast to SegGeom");
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

    auto lambda_push_polyhedron = [&](auto poly) {
      const int gid = poly->GetGlobalID();
      this->push(buffer, &offset, &gid);
      const int num_faces = poly->GetNumFaces();
      this->push(buffer, &offset, &num_faces);
      for (int fx = 0; fx < num_faces; fx++) {
        auto face = poly->GetFace(fx);
        const int face_shape_type_int = shape_type_to_int(face->GetShapeType());
        this->push(buffer, &offset, &face_shape_type_int);
        lambda_push_face(face);
      }
    };

    // Push the description of the geom
    if (shape_type == LibUtilities::ShapeType::eSegment) {
      lambda_push_edge(std::dynamic_pointer_cast<SegGeom>(geom));
    } else if ((shape_type == LibUtilities::ShapeType::eTriangle) ||
               (shape_type == LibUtilities::ShapeType::eQuadrilateral)) {
      lambda_push_face(std::dynamic_pointer_cast<Geometry2D>(geom));
    } else { // Assume a 3D geom
      lambda_push_polyhedron(std::dynamic_pointer_cast<Geometry3D>(geom));
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

    this->pop(buffer, &offset, &this->rank);
    this->pop(buffer, &offset, &this->id);
    this->pop(buffer, &offset, &shape_type_int);
    auto shape_type = int_to_shape_type(shape_type_int);

    int num_aux_int_properties = -1;
    this->pop(buffer, &offset, &num_aux_int_properties);
    ASSERTL0(num_aux_int_properties > -1,
             "num_aux_int_properties failed to unpack sensible value");
    this->aux_int_properties.resize(num_aux_int_properties);
    this->pop(buffer, &offset, this->aux_int_properties);

    auto lambda_pop_edge = [&]() {
      int gid;
      this->pop(buffer, &offset, &gid);
      int coordim;
      this->pop(buffer, &offset, &coordim);
      int num_verts;
      this->pop(buffer, &offset, &num_verts);
      std::vector<SpatialDomains::PointGeomSharedPtr> vertices;
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
      edges.reserve(num_edges);
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

    auto lambda_pop_polyhedron = [&](const auto shape_type) {
      int gid;
      this->pop(buffer, &offset, &gid);
      int num_faces;
      this->pop(buffer, &offset, &num_faces);
      std::vector<SpatialDomains::Geometry2DSharedPtr> faces;
      faces.reserve(num_faces);
      for (int fx = 0; fx < num_faces; fx++) {
        int face_shape_type_int;
        this->pop(buffer, &offset, &face_shape_type_int);
        const auto face_shape_type = int_to_shape_type(face_shape_type_int);
        faces.push_back(lambda_pop_face(face_shape_type));
      }
      // Polyhedra don't seem to have a curve in Nektar++
      std::shared_ptr<Geometry3D> g;
      if (shape_type == LibUtilities::ShapeType::eTetrahedron) {
        std::vector<TriGeomSharedPtr> tmp_faces;
        tmp_faces.reserve(num_faces);
        for (auto fx : faces) {
          tmp_faces.push_back(std::dynamic_pointer_cast<TriGeom>(fx));
        }
        g = std::dynamic_pointer_cast<Geometry3D>(
            std::make_shared<TetGeom>(gid, tmp_faces.data()));
      } else if (shape_type == LibUtilities::ShapeType::ePyramid) {
        g = std::dynamic_pointer_cast<Geometry3D>(
            std::make_shared<PyrGeom>(gid, faces.data()));
      } else if (shape_type == LibUtilities::ShapeType::ePrism) {
        g = std::dynamic_pointer_cast<Geometry3D>(
            std::make_shared<PrismGeom>(gid, faces.data()));
      } else {
        std::vector<QuadGeomSharedPtr> tmp_faces;
        tmp_faces.reserve(num_faces);
        for (auto fx : faces) {
          tmp_faces.push_back(std::dynamic_pointer_cast<QuadGeom>(fx));
        }
        g = std::dynamic_pointer_cast<Geometry3D>(
            std::make_shared<HexGeom>(gid, tmp_faces.data()));
      }
      g->GetGeomFactors();
      g->Setup();
      return g;
    };

    if (shape_type == LibUtilities::ShapeType::eSegment) {
      this->geom = std::dynamic_pointer_cast<T>(lambda_pop_edge());
    } else if ((shape_type == LibUtilities::ShapeType::eTriangle) ||
               (shape_type == LibUtilities::ShapeType::eQuadrilateral)) {
      this->geom = std::dynamic_pointer_cast<T>(lambda_pop_face(shape_type));
    } else { // Assume a 3D geom
      this->geom =
          std::dynamic_pointer_cast<T>(lambda_pop_polyhedron(shape_type));
    }

    this->geom->GetGeomFactors();
    this->geom->Setup();

    NESOASSERT(offset == num_bytes, "Not all data was deserialised");
  }
};

} // namespace NESO::GeometryTransport

#endif
