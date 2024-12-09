#include <nektar_interface/particle_cell_mapping/newton_geom_interfaces.hpp>
#include <nektar_interface/particle_cell_mapping/x_map_bounding_box.hpp>
#include <nektar_interface/particle_cell_mapping/x_map_newton.hpp>

namespace NESO::BoundingBox {

std::array<double, 6> get_bounding_box(SYCLTargetSharedPtr sycl_target,
                                       SD::Geometry3DSharedPtr geom,
                                       std::size_t grid_size,
                                       const REAL pad_rel, const REAL pad_abs) {

  NESOASSERT(geom != nullptr, "Bad geometry object passed.");

  Newton::XMapNewton<Newton::MappingGeneric3D> x_map(sycl_target, geom);
  return x_map.get_bounding_box(grid_size, pad_rel, pad_abs);
}

} // namespace NESO::BoundingBox
