#ifndef __REMOTE_GEOM_3D_H__
#define __REMOTE_GEOM_3D_H__

// Nektar++ Includes
#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

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

} // namespace NESO

#endif
