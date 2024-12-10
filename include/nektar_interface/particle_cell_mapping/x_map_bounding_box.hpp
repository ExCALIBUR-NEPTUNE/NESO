#ifndef __NESO_PARTICLE_CELL_MAPPING_X_MAP_BOUNDING_BOX_HPP__
#define __NESO_PARTICLE_CELL_MAPPING_X_MAP_BOUNDING_BOX_HPP__

#include "../geometry_transport/utility_geometry.hpp"
#include "../parameter_store.hpp"
#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

namespace {
namespace SD = Nektar::SpatialDomains;
}

namespace NESO::BoundingBox {

/**
 * Return a Nektar++ style bounding box for the geometry object. Padding is
 * added to each end of each dimension. i.e. a padding of 5% (pad_rel = 0.05)
 * at each end is 10% globally.
 *
 * Parameters:
 *  get_bounding_box/linear_pad_rel
 *  get_bounding_box/linear_pad_abs
 *  get_bounding_box/nonlinear_pad_rel
 *  get_bounding_box/nonlinear_pad_abs
 *  get_bounding_box/nonlinear_grid_size_factor
 *
 * @param sycl_target Compute device to compute bounding box on.
 * @param geom Geometry object to get bounding box for.
 * @param parameter_store ParameterStore containing parameters.
 * @returns Bounding box in format [minx, miny, minz, maxx, maxy, maxz];
 */
std::array<double, 6>
get_bounding_box(Particles::SYCLTargetSharedPtr sycl_target,
                 SD::Geometry3DSharedPtr geom,
                 ParameterStoreSharedPtr parameter_store);

} // namespace NESO::BoundingBox

#endif
