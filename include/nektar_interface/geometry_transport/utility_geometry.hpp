#ifndef _NESO_GEOMETRY_TRANSPORT_UTILITY_GEOMETRY_HPP_
#define _NESO_GEOMETRY_TRANSPORT_UTILITY_GEOMETRY_HPP_

#include <memory>

namespace NESO {

/**
 * @returns True if the geometry object is linear.
 */
template <typename GEOM_TYPE>
inline bool geometry_is_linear(std::shared_ptr<GEOM_TYPE> geom) {
  const int ndim = geom->GetCoordim();
  const auto xmap = geom->GetXmap();
  for (int dimx = 0; dimx < ndim; dimx++) {
    if (xmap->GetBasisNumModes(dimx) != 2) {
      return false;
    }
  }
  return true;
}

} // namespace NESO

#endif
