#ifndef __NESO_COMPOSITE_INTERACTION_COMPOSITE_UTILITY_HPP_
#define __NESO_COMPOSITE_INTERACTION_COMPOSITE_UTILITY_HPP_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <nektar_interface/typedefs.hpp>

namespace NESO {

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

/**
 * Get the average of the vertices of a geometry object.
 *
 * @param[in] geom Geometry object.
 * @param[in, out] average On return contains the average of the vertices.
 */
void get_vertex_average(std::shared_ptr<SpatialDomains::Geometry> geom,
                        std::vector<REAL> &average);

} // namespace NESO

#endif
