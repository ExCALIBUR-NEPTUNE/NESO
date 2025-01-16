#ifndef __PACKED_GEOM_2D_H__
#define __PACKED_GEOM_2D_H__

// Nektar++ Includes
#include "remote_geom_2d.hpp"
#include "shape_mapping.hpp"
#include <SpatialDomains/MeshGraph.h>

#include "geometry_packing_utility.hpp"
#include <neso_particles.hpp>
using namespace NESO::Particles;

using namespace Nektar;
using namespace Nektar::LibUtilities;

namespace NESO::GeometryTransport {

class PackedGeom2D {
private:
  // Push data onto the buffer.
  template <typename T> inline void push(T *data) {
    const std::size_t size = sizeof(T);
    const int offset_new = offset + size;
    buf.resize(offset_new);
    std::memcpy(((unsigned char *)buf.data()) + offset, data, size);
    offset = offset_new;
  }

  // Pop data from the buffer.
  template <typename T> inline void pop(T *data) {
    const std::size_t size = sizeof(T);
    const int offset_new = offset + size;
    ASSERTL0((offset_new <= input_length) || (input_length == -1),
             "Unserialiation overflows buffer.");
    std::memcpy(data, buf_in + offset, size);
    offset = offset_new;
  }

  /*
   * 2D Geoms are created with an id, edges and an optional curve. Edges
   * are constructed with an id, coordim, vertices and an optional curve.
   * Pack by looping over the edges and serialising each one into a
   * buffer.
   *
   */
  template <typename T> inline void pack_general(T &geom) {

    this->offset = 0;
    this->buf.reserve(512);
    this->id = geom.GetGlobalID();
    this->num_edges = geom.GetNumEdges();

    push(&shape_type_int);
    push(&rank);
    push(&local_id);
    push(&id);
    push(&num_edges);

    auto lambda_pack_curve = [&](const auto curve) {
      GeomPackSpec gs;
      if (curve == nullptr) {
        gs.a = 0;
        gs.b = 0;
        gs.n_points = -1;
      } else {
        gs.a = curve->m_curveID;
        gs.b = static_cast<int>(curve->m_ptype);
        gs.n_points = curve->m_points.size();
      }
      push(&gs);
      if (curve != nullptr) {
        const int n_points = curve->m_points.size();
        for (int pointx = 0; pointx < n_points; pointx++) {
          auto point = curve->m_points.at(pointx);
          PointStruct ps;
          ps.coordim = point->GetCoordim();
          ps.vid = point->GetVid();
          point->GetCoords(ps.x, ps.y, ps.z);
          push(&ps);
        }
      }
    };

    GeomPackSpec gs;
    PointStruct ps;

    for (int edgex = 0; edgex < num_edges; edgex++) {

      auto seg_geom = geom.GetSegGeom(edgex);
      const int num_points = seg_geom->GetNumVerts();

      gs.a = seg_geom->GetGlobalID();
      gs.b = seg_geom->GetCoordim();
      gs.n_points = num_points;
      push(&gs);

      for (int pointx = 0; pointx < num_points; pointx++) {
        auto point = seg_geom->GetVertex(pointx);
        ps.coordim = point->GetCoordim();
        ps.vid = point->GetVid();
        point->GetCoords(ps.x, ps.y, ps.z);
        push(&ps);
      }

      // curve of the edge
      auto curve = seg_geom->GetCurve();
      // A curve with n_points = -1 will be a taken as non-existent.
      lambda_pack_curve(curve);
    }

    // curve of geom
    auto curve = geom.GetCurve();
    lambda_pack_curve(curve);
  };

  // Unpack the data common to both Quads and Triangles.
  inline void unpack_general() {
    ASSERTL0(offset == 0, "offset != 0 - cannot unpack twice");
    ASSERTL0(buf_in != nullptr, "source buffer has null pointer");

    // pop the metadata
    pop(&shape_type_int);
    pop(&rank);
    pop(&local_id);
    pop(&id);
    pop(&num_edges);

    ASSERTL0(rank >= 0, "unreasonable rank");
    ASSERTL0((num_edges == 4) || (num_edges == 3),
             "Bad number of edges expected 4 or 3");

    // helper lambda to unpack a curve
    auto lambda_unpack_curve =
        [&](const auto gs) -> SpatialDomains::CurveSharedPtr {
      if (gs.n_points < 0) {
        return SpatialDomains::CurveSharedPtr();
      }
      const int m_curveID = gs.a;
      const int m_ptype_int = gs.b;
      const LibUtilities::PointsType m_ptype =
          static_cast<LibUtilities::PointsType>(m_ptype_int);

      const int n_points = gs.n_points;
      std::vector<SpatialDomains::PointGeomSharedPtr> m_points;
      m_points.reserve(n_points);
      for (int pointx = 0; pointx < n_points; pointx++) {
        PointStruct ps;
        pop(&ps);
        m_points.push_back(std::make_shared<SpatialDomains::PointGeom>(
            ps.coordim, ps.vid, ps.x, ps.y, ps.z));
      }
      auto curve = std::make_shared<SpatialDomains::Curve>(m_curveID, m_ptype);
      curve->m_points = m_points;
      return curve;
    };

    PointStruct ps;
    GeomPackSpec gs;

    edges.reserve(num_edges);
    vertices.reserve(num_edges * 2);

    // pop and construct the edges
    for (int edgex = 0; edgex < num_edges; edgex++) {
      pop(&gs);
      const int edge_id = gs.a;
      const int edge_coordim = gs.b;
      const int n_points = gs.n_points;

      ASSERTL0(n_points >= 2, "Expected at least two points for an edge");

      const int points_offset = vertices.size();
      for (int pointx = 0; pointx < n_points; pointx++) {
        pop(&ps);
        vertices.push_back(std::make_shared<SpatialDomains::PointGeom>(
            ps.coordim, ps.vid, ps.x, ps.y, ps.z));
      }

      pop(&gs);
      auto curve = lambda_unpack_curve(gs);

      // actually construct the edge
      auto edge_tmp = std::make_shared<SpatialDomains::SegGeom>(
          edge_id, edge_coordim, vertices.data() + points_offset, curve);
      // edge_tmp->Setup();
      edges.push_back(edge_tmp);
    }

    // Unpack the curve for the geom if it exists.
    pop(&gs);
    this->curve = lambda_unpack_curve(gs);
  }

  /*
   *  Pack a 2DGeom into the buffer.
   */
  template <typename T> void pack(std::shared_ptr<T> &geom) {
    auto extern_geom = std::static_pointer_cast<GeomExtern<T>>(geom);
    this->pack_general(*extern_geom);
  }

  int shape_type_int;
  int rank;
  int local_id;
  int id;
  int num_edges;

  unsigned char *buf_in = nullptr;
  int input_length = 0;
  int offset = 0;

  std::vector<SpatialDomains::SegGeomSharedPtr> edges;
  std::vector<SpatialDomains::PointGeomSharedPtr> vertices;
  SpatialDomains::CurveSharedPtr curve;

public:
  std::vector<unsigned char> buf;

  PackedGeom2D(std::vector<unsigned char> &buf)
      : buf_in(buf.data()), input_length(buf.size()){};
  PackedGeom2D(unsigned char *buf_in, const int input_length = -1)
      : buf_in(buf_in), input_length(input_length){};

  template <typename T>
  PackedGeom2D(int rank, int local_id, std::shared_ptr<T> &geom) {
    this->shape_type_int = shape_type_to_int(geom->GetShapeType());
    this->rank = rank;
    this->local_id = local_id;
    pack(geom);
  }

  /*
   *  This offset is to help pointer arithmetric into the buffer for the next
   *  Geom.
   */
  inline int get_offset() { return this->offset; };
  /*
   * The rank that owns this geometry object.
   */
  inline int get_rank() { return this->rank; };
  /*
   * The local id of this geometry object on the remote rank.
   */
  inline int get_local_id() { return this->local_id; };

  /*
   *  Unpack the data as a 2DGeom.
   */
  template <typename T> std::shared_ptr<RemoteGeom2D<T>> unpack() {
    unpack_general();
    std::shared_ptr<T> geom =
        std::make_shared<T>(this->id, this->edges.data(), this->curve);
    geom->GetGeomFactors();
    geom->Setup();

    auto remote_geom = std::make_shared<RemoteGeom2D<T>>(rank, local_id, geom);
    return remote_geom;
  }
};

} // namespace NESO::GeometryTransport

#endif
