#include <nektar_interface/particle_cell_mapping/newton_geom_interfaces.hpp>
#include <nektar_interface/particle_cell_mapping/x_map_bounding_box.hpp>
#include <nektar_interface/particle_cell_mapping/x_map_newton.hpp>

namespace NESO::BoundingBox {

std::array<double, 6>
get_bounding_box(Particles::SYCLTargetSharedPtr sycl_target,
                 SD::Geometry3DSharedPtr geom,
                 ParameterStoreSharedPtr parameter_store) {

  NESOASSERT(geom != nullptr, "Bad geometry object passed.");
  const bool is_linear = geometry_is_linear(geom);
  const std::size_t max_num_modes =
      static_cast<std::size_t>(geom->GetXmap()->EvalBasisNumModesMax());
  Newton::XMapNewton<Newton::MappingGeneric3D> x_map(sycl_target, geom);

  REAL pad_rel = 0.0;
  REAL pad_abs = 0.0;
  std::size_t grid_size = 2;

  if (is_linear) {
    pad_rel =
        parameter_store->get<REAL>("get_bounding_box/linear_pad_rel", 0.0);
    pad_abs =
        parameter_store->get<REAL>("get_bounding_box/linear_pad_abs", 0.0);
    grid_size = 2;
  } else {
    pad_rel =
        parameter_store->get<REAL>("get_bounding_box/nonlinear_pad_rel", 0.05);
    pad_abs =
        parameter_store->get<REAL>("get_bounding_box/nonlinear_pad_abs", 0.0);
    const auto grid_size_factor = parameter_store->get<INT>(
        "get_bounding_box/nonlinear_grid_size_factor", 4);
    grid_size = static_cast<std::size_t>(grid_size_factor) * max_num_modes;
  }

  NESOASSERT(grid_size >= 2, "Bad bounding box grid size (< 2).");
  NESOASSERT(pad_rel >= 0.0, "Bad bounding box pad_rel (< 0.0)");
  NESOASSERT(pad_abs >= 0.0, "Bad bounding box pad_abs (< 0.0)");

  return x_map.get_bounding_box(grid_size, pad_rel, pad_abs);
}

} // namespace NESO::BoundingBox
