#ifndef __GEOMETRY_TRANSPORT_2D_H__
#define __GEOMETRY_TRANSPORT_2D_H__

// Nektar++ Includes
#include <SpatialDomains/MeshGraph.h>

// System includes
#include <iostream>
#include <map>
#include <mpi.h>
#include <vector>

using namespace std;
using namespace Nektar;
using namespace Nektar::SpatialDomains;

#include <neso_particles.hpp>

using namespace NESO::Particles;

namespace NESO {

/*
 * Mirrors the existing PointGeom for packing.
 */
struct PointStruct {
  int coordim;
  int vid;
  NekDouble x;
  NekDouble y;
  NekDouble z;
};

/*
 *  General struct to hold the description of the arguments for SegGeoms and
 *  Curves.
 */
struct GeomPackSpec {
  int a;
  int b;
  int n_points;
};

/**
 *  Helper class to access segments and curves in Nektar geometry classes.
 *  These attributes are protected in the base class - this class provides
 *  accessors.
 */
template <class T> class GeomExtern : public T {
private:
protected:
public:
  SpatialDomains::SegGeomSharedPtr GetSegGeom(int index) {
    return this->m_edges[index];
  };
  SpatialDomains::CurveSharedPtr GetCurve() { return this->m_curve; };
};

/**
 *  Description of a 2D geometry object that is owned by a remote rank.
 *
 */
template <typename T> class RemoteGeom2D {
public:
  /// The remote rank that owns the geometry object (i.e. holds it in its
  /// MeshGraph).
  int rank = -1;
  /// The geometry id on the remote rank.
  int id = -1;
  /// A local copy of the geometry object.
  std::shared_ptr<T> geom;
  /**
   *  Constructor for remote geometry object.
   *
   *  @param rank Remote rank that owns the object.
   *  @param id Remote id of this geometry object.
   *  @param geom Shared pointer to local copy of the geometry object.
   */
  RemoteGeom2D(int rank, int id, std::shared_ptr<T> geom)
      : rank(rank), id(id), geom(geom){};

  /**
   * Get the Nektar++ bounding box for the geometry object.
   *
   * @returns Bounding box.
   */
  std::array<NekDouble, 6> GetBoundingBox() {
    return this->geom->GetBoundingBox();
  }
};

class PackedGeom2D {
private:
  // Push data onto the buffer.
  template <typename T> void push(T *data) {
    const std::size_t size = sizeof(T);
    const int offset_new = offset + size;
    buf.resize(offset_new);
    std::memcpy(((unsigned char *)buf.data()) + offset, data, size);
    offset = offset_new;
  }

  // Pop data from the buffer.
  template <typename T> void pop(T *data) {
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
  template <typename T> void pack_general(T &geom) {

    this->offset = 0;
    this->buf.reserve(512);
    this->id = geom.GetGlobalID();
    this->num_edges = geom.GetNumEdges();

    push(&rank);
    push(&local_id);
    push(&id);
    push(&num_edges);

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
      ASSERTL0(curve == nullptr, "Not implemented for curved edges");
      // A curve with n_points = -1 will be a taken as non-existant.
      gs.a = 0;
      gs.b = 0;
      gs.n_points = -1;
      push(&gs);
    }

    // curve of geom
    auto curve = geom.GetCurve();
    ASSERTL0(curve == nullptr, "Not implemented for curved edges");
    // A curve with n_points = -1 will be a taken as non-existant.
    gs.a = 0;
    gs.b = 0;
    gs.n_points = -1;
    push(&gs);
  };

  // Unpack the data common to both Quads and Triangles.
  void unpack_general() {
    ASSERTL0(offset == 0, "offset != 0 - cannot unpack twice");
    ASSERTL0(buf_in != nullptr, "source buffer has null pointer");

    // pop the metadata
    pop(&rank);
    pop(&local_id);
    pop(&id);
    pop(&num_edges);

    ASSERTL0(rank >= 0, "unreasonable rank");
    ASSERTL0((num_edges == 4) || (num_edges == 3),
             "Bad number of edges expected 4 or 3");

    GeomPackSpec gs;
    PointStruct ps;

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

      // In future the edge might have a corresponding curve
      pop(&gs);
      ASSERTL0(gs.n_points == -1, "unpacking routine did not expect a curve");

      // actually construct the edge
      auto edge_tmp = std::make_shared<SpatialDomains::SegGeom>(
          edge_id, edge_coordim, vertices.data() + points_offset);
      // edge_tmp->Setup();
      edges.push_back(edge_tmp);
    }

    // In future the geom might have a curve
    pop(&gs);
    ASSERTL0(gs.n_points == -1, "unpacking routine did not expect a curve");
  }

  /*
   *  Pack a 2DGeom into the buffer.
   */
  template <typename T> void pack(std::shared_ptr<T> &geom) {
    auto extern_geom = std::static_pointer_cast<GeomExtern<T>>(geom);
    this->pack_general(*extern_geom);
  }

  int rank;
  int local_id;
  int id;
  int num_edges;

  unsigned char *buf_in = nullptr;
  int input_length = 0;
  int offset = 0;

  std::vector<SpatialDomains::SegGeomSharedPtr> edges;
  std::vector<SpatialDomains::PointGeomSharedPtr> vertices;

public:
  std::vector<unsigned char> buf;

  PackedGeom2D(std::vector<unsigned char> &buf)
      : buf_in(buf.data()), input_length(buf.size()){};
  PackedGeom2D(unsigned char *buf_in, const int input_length = -1)
      : buf_in(buf_in), input_length(input_length){};

  template <typename T>
  PackedGeom2D(int rank, int local_id, std::shared_ptr<T> &geom) {
    this->rank = rank;
    this->local_id = local_id;
    pack(geom);
  }

  /*
   *  This offset is to help pointer arithmetric into the buffer for the next
   *  Geom.
   */
  int get_offset() { return this->offset; };
  /*
   * The rank that owns this geometry object.
   */
  int get_rank() { return this->rank; };
  /*
   * The local id of this geometry object on the remote rank.
   */
  int get_local_id() { return this->local_id; };

  /*
   *  Unpack the data as a 2DGeom.
   */
  template <typename T> std::shared_ptr<RemoteGeom2D<T>> unpack() {
    unpack_general();
    std::shared_ptr<T> geom = std::make_shared<T>(this->id, this->edges.data());
    geom->GetGeomFactors();
    geom->Setup();
    auto remote_geom = std::make_shared<RemoteGeom2D<T>>(rank, local_id, geom);
    return remote_geom;
  }
};

/*
 * Class to pack and unpack a set of 2D Geometry objects.
 *
 */
class PackedGeoms2D {
private:
  unsigned char *buf_in;
  int input_length = 0;
  int offset = 0;

  int get_packed_count() {
    int packed_count;
    ASSERTL0(input_length >= sizeof(int), "Input buffer has no count.");
    std::memcpy(&packed_count, buf_in, sizeof(int));
    ASSERTL0(((packed_count >= 0) && (packed_count <= input_length)),
             "Packed count is either negative or unrealistic.");
    return packed_count;
  }

public:
  std::vector<unsigned char> buf;

  PackedGeoms2D(){};

  /*
   * Pack a set of geometry objects collected by calling GetAllQuadGeoms or
   * GetAllTriGeoms on a MeshGraph object.
   */
  template <typename T>
  PackedGeoms2D(int rank, std::map<int, std::shared_ptr<T>> &geom_map) {
    const int num_geoms = geom_map.size();
    buf.reserve(512 * num_geoms);

    buf.resize(sizeof(int));
    std::memcpy(buf.data(), &num_geoms, sizeof(int));

    for (auto &geom_item : geom_map) {
      auto geom = geom_item.second;
      PackedGeom2D pg(rank, geom_item.first, geom);
      buf.insert(buf.end(), pg.buf.begin(), pg.buf.end());
    }
  };

  /*
   * Pack a set of geometry objects collected by calling GetAllQuadGeoms or
   * GetAllTriGeoms on a MeshGraph object.
   */
  template <typename T>
  PackedGeoms2D(std::vector<std::shared_ptr<RemoteGeom2D<T>>> &geoms) {
    const int num_geoms = geoms.size();
    buf.reserve(512 * num_geoms);

    buf.resize(sizeof(int));
    std::memcpy(buf.data(), &num_geoms, sizeof(int));

    for (auto &remote_geom : geoms) {
      auto geom = remote_geom->geom;
      const int rank = remote_geom->rank;
      const int gid = remote_geom->id;
      PackedGeom2D pg(rank, gid, geom);
      buf.insert(buf.end(), pg.buf.begin(), pg.buf.end());
    }
  };

  /*
   * Initialise this object in unpacking mode with a byte buffer of
   * serialised objects.
   */
  PackedGeoms2D(unsigned char *buf_in, int input_length) {
    this->offset = 0;
    this->buf_in = buf_in;
    this->input_length = input_length;
  };

  /*
   * Unserialise the held byte buffer as 2D geometry type T.
   */
  template <typename T>
  void unpack(std::vector<std::shared_ptr<RemoteGeom2D<T>>> &geoms) {
    const int packed_count = this->get_packed_count();
    geoms.reserve(geoms.size() + packed_count);
    for (int cx = 0; cx < packed_count; cx++) {
      auto new_packed_geom =
          PackedGeom2D(buf_in + sizeof(int) + offset, input_length);
      auto new_geom = new_packed_geom.unpack<T>();
      offset += new_packed_geom.get_offset();
      geoms.push_back(new_geom);
    }
    ASSERTL0(offset <= input_length, "buffer overflow occured");
  }
};

static inline int pos_mod(int i, int n) { return (i % n + n) % n; }

/*
 *  Collect all the 2D Geometry objects of type T from all the other MPI ranks.
 */
template <typename T>
std::vector<std::shared_ptr<RemoteGeom2D<T>>>
get_all_remote_geoms_2d(MPI_Comm comm,
                        std::map<int, std::shared_ptr<T>> &geom_map) {
  int rank, size;
  MPICHK(MPI_Comm_rank(comm, &rank));
  MPICHK(MPI_Comm_size(comm, &size));

  PackedGeoms2D local_packed_geoms(rank, geom_map);

  int max_buf_size, this_buf_size;
  this_buf_size = local_packed_geoms.buf.size();

  MPICHK(
      MPI_Allreduce(&this_buf_size, &max_buf_size, 1, MPI_INT, MPI_MAX, comm));

  // simplify send/recvs by choosing a buffer size of the maximum over all
  // ranks
  local_packed_geoms.buf.resize(max_buf_size);

  std::vector<unsigned char> buf_send(max_buf_size);
  std::vector<unsigned char> buf_recv(max_buf_size);

  memcpy(buf_send.data(), local_packed_geoms.buf.data(), max_buf_size);

  unsigned char *ptr_send = buf_send.data();
  unsigned char *ptr_recv = buf_recv.data();
  MPI_Status status;

  std::vector<std::shared_ptr<RemoteGeom2D<T>>> remote_geoms{};

  // cyclic passing of remote geometry objects.
  for (int shiftx = 0; shiftx < (size - 1); shiftx++) {

    int rank_send = pos_mod(rank + 1, size);
    int rank_recv = pos_mod(rank - 1, size);

    MPICHK(MPI_Sendrecv(ptr_send, max_buf_size, MPI_BYTE, rank_send, rank,
                        ptr_recv, max_buf_size, MPI_BYTE, rank_recv, rank_recv,
                        comm, &status));

    PackedGeoms2D remote_packed_geoms(ptr_recv, max_buf_size);
    remote_packed_geoms.unpack(remote_geoms);

    unsigned char *ptr_tmp = ptr_send;
    ptr_send = ptr_recv;
    ptr_recv = ptr_tmp;
  }

  return remote_geoms;
}

/**
 * Get all 2D geometry objects from a Nektar++ MeshGraph
 *
 * @param[in] graph MeshGraph instance.
 * @param[in,out] std::map of Nektar++ Geometry2D pointers.
 */
inline void get_all_elements_2d(
    Nektar::SpatialDomains::MeshGraphSharedPtr &graph,
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>> &geoms) {
  geoms.clear();

  for (auto &e : graph->GetAllTriGeoms()) {
    geoms[e.first] =
        std::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry2D>(e.second);
  }
  for (auto &e : graph->GetAllQuadGeoms()) {
    geoms[e.first] =
        std::dynamic_pointer_cast<Nektar::SpatialDomains::Geometry2D>(e.second);
  }
}

/**
 *  Add remote 2D objects to a map from geometry ids to shared pointers.
 *
 *  @param[in] remote_geoms Vector of remote geometry objects.
 *  @param[in,out] new_map Output element map (appended to).
 */
template <typename T>
inline void combine_remote_geoms_2d(
    std::vector<std::shared_ptr<RemoteGeom2D<T>>> &remote_geoms,
    std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>
        &new_map) {
  for (const auto &ix : remote_geoms) {
    new_map[ix->id] = std::dynamic_pointer_cast<Geometry2D>(ix.geom);
  }
}

/**
 *  Add remote 2D objects (typed) to a vector of remote geometry objects
 * (generic 2D type).
 *
 *  @param[in] remote_geoms Vector of remote geometry objects.
 *  @param[in,out] remote_geoms_2d Output vector of 2D geometry types.
 */
template <typename T>
inline void combine_remote_geoms_2d(
    std::vector<std::shared_ptr<RemoteGeom2D<T>>> &remote_geoms,
    std::vector<std::shared_ptr<RemoteGeom2D<Geometry2D>>> &remote_geoms_2d) {
  remote_geoms_2d.reserve(remote_geoms_2d.size() + remote_geoms.size());
  for (const auto &ix : remote_geoms) {
    remote_geoms_2d.push_back(
        std::make_shared<RemoteGeom2D<Geometry2D>>(ix->rank, ix->id, ix->geom));
  }
}

} // namespace NESO

#endif
