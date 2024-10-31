#ifndef __GEOMETRY_PACKING_UTILITY_H__
#define __GEOMETRY_PACKING_UTILITY_H__

// Nektar++ Includes
#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <cstddef>
#include <vector>

namespace NESO::GeometryTransport {

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

inline std::ostream &operator<<(std::ostream &os, const PointStruct &ps) {
  os << "coordim: " << ps.coordim;
  os << " vid: " << ps.vid;
  os << " x: " << ps.x;
  os << " y: " << ps.y;
  os << " z: " << ps.z;
  return os;
}
inline std::ostream &operator<<(std::ostream &os, const GeomPackSpec &gps) {
  os << "a: " << gps.a;
  os << " b: " << gps.b;
  os << " n_points: " << gps.n_points;
  return os;
}

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

} // namespace NESO::GeometryTransport

#endif
