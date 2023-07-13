#ifndef __PACKED_GEOMS_2D_H__
#define __PACKED_GEOMS_2D_H__

// Nektar++ Includes
#include "packed_geom_2d.hpp"
#include <SpatialDomains/MeshGraph.h>

using namespace Nektar;

namespace NESO {

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
      GeometryTransport::PackedGeom2D pg(rank, geom_item.first, geom);
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
      GeometryTransport::PackedGeom2D pg(rank, gid, geom);
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
      auto new_packed_geom = GeometryTransport::PackedGeom2D(
          buf_in + sizeof(int) + offset, input_length);
      auto new_geom = new_packed_geom.unpack<T>();
      offset += new_packed_geom.get_offset();
      geoms.push_back(new_geom);
    }
    ASSERTL0(offset <= input_length, "buffer overflow occured");
  }
};

} // namespace NESO

#endif
