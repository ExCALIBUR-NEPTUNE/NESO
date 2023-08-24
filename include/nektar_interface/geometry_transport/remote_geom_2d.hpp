#ifndef __REMOTE_GEOM_2D_H__
#define __REMOTE_GEOM_2D_H__

// Nektar++ Includes
#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

namespace NESO {

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

} // namespace NESO

#endif
