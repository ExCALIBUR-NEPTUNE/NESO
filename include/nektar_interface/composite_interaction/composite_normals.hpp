#ifndef __NESO_COMPOSITE_INTERACTION_COMPOSITE_NORMALS_HPP_
#define __NESO_COMPOSITE_INTERACTION_COMPOSITE_NORMALS_HPP_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <nektar_interface/typedefs.hpp>

namespace NESO::CompositeInteraction {

/**
 * Get the unit normal vector to a linear Nektar++ geometry object.
 *
 * @param[in] geom 1D geometry object.
 * @param[in, out] normal On return contains the normal vector.
 */
void get_normal_vector(std::shared_ptr<SpatialDomains::Geometry1D> geom,
                       std::vector<REAL> &normal);

/**
 * Get the unit normal vector to a linear Nektar++ geometry object.
 *
 * @param[in] geom 1D geometry object.
 * @param[in, out] normal On return contains the normal vector.
 */
void get_normal_vector(std::shared_ptr<SpatialDomains::Geometry2D> geom,
                       std::vector<REAL> &normal);

} // namespace NESO::CompositeInteraction

#endif
