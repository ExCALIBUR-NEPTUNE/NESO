#ifndef __NESO_PARTICLE_CELL_MAPPING_X_MAP_BOUNDING_BOX_HPP__
#define __NESO_PARTICLE_CELL_MAPPING_X_MAP_BOUNDING_BOX_HPP__

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
 * @param sycl_target Compute device to compute bounding box on.
 * @param geom Geometry object to get bounding box for.
 * @param grid_size Resolution of grid to use on each face of the collapsed
 * reference space. Default 32.
 * @param pad_rel Relative padding to add to computed bounding box, default
 * 0.05, i.e. 5%.
 * @param pad_abs Absolute padding to add to computed bounding box, default
 * 0.0.
 * @returns Bounding box in format [minx, miny, minz, maxx, maxy, maxz];
 */
std::array<double, 6>
get_bounding_box(Particles::SYCLTargetSharedPtr sycl_target,
                 SD::Geometry3DSharedPtr geom, std::size_t grid_size = 32,
                 const Particles::REAL pad_rel = 0.05,
                 const Particles::REAL pad_abs = 0.0);

} // namespace NESO::BoundingBox

#endif
