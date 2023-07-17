#include <nektar_interface/particle_cell_mapping/map_particles_3d.hpp>

namespace NESO {

MapParticles3D::MapParticles3D(
    SYCLTargetSharedPtr sycl_target,
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    ParameterStoreSharedPtr config)
    : sycl_target(sycl_target),
      particle_mesh_interface(particle_mesh_interface) {

  this->map_particles_common =
      std::make_unique<MapParticlesCommon>(sycl_target);

  this->map_particles_3d_regular = nullptr;
  this->map_particles_host = nullptr;
  std::get<0>(this->map_particles_3d_deformed_linear) = nullptr;
  std::get<1>(this->map_particles_3d_deformed_linear) = nullptr;
  std::get<2>(this->map_particles_3d_deformed_linear) = nullptr;
  std::get<3>(this->map_particles_3d_deformed_linear) = nullptr;

  GeometryContainer3D geometry_container_3d;
  assemble_geometry_container_3d(particle_mesh_interface->graph,
                                 particle_mesh_interface->remote_geoms_3d,
                                 geometry_container_3d);

  // Create a mapper for 3D regular geometry objects
  if (geometry_container_3d.regular.size()) {
    this->map_particles_3d_regular = std::make_unique<MapParticles3DRegular>(
        sycl_target, particle_mesh_interface, config);
  }

  // Create mappers for the deformed geometry objects with linear faces
  if (geometry_container_3d.deformed_linear.tet.size()) {
    std::get<0>(this->map_particles_3d_deformed_linear) = std::make_unique<
        Newton::MapParticlesNewton<Newton::MappingTetLinear3D>>(
        Newton::MappingTetLinear3D{}, this->sycl_target,
        geometry_container_3d.deformed_linear.tet.local,
        geometry_container_3d.deformed_linear.tet.remote, config);
  }
  if (geometry_container_3d.deformed_linear.prism.size()) {
    std::get<1>(this->map_particles_3d_deformed_linear) = std::make_unique<
        Newton::MapParticlesNewton<Newton::MappingPrismLinear3D>>(
        Newton::MappingPrismLinear3D{}, this->sycl_target,
        geometry_container_3d.deformed_linear.prism.local,
        geometry_container_3d.deformed_linear.prism.remote, config);
  }
  if (geometry_container_3d.deformed_linear.hex.size()) {
    std::get<2>(this->map_particles_3d_deformed_linear) = std::make_unique<
        Newton::MapParticlesNewton<Newton::MappingHexLinear3D>>(
        Newton::MappingHexLinear3D{}, this->sycl_target,
        geometry_container_3d.deformed_linear.hex.local,
        geometry_container_3d.deformed_linear.hex.remote, config);
  }
  if (geometry_container_3d.deformed_linear.pyr.size()) {
    std::get<3>(this->map_particles_3d_deformed_linear) = std::make_unique<
        Newton::MapParticlesNewton<Newton::MappingPyrLinear3D>>(
        Newton::MappingPyrLinear3D{}, this->sycl_target,
        geometry_container_3d.deformed_linear.pyr.local,
        geometry_container_3d.deformed_linear.pyr.remote, config);
  }

  // Create a mapper for 3D deformed non-linear geometry objects
  // as a sycl version is not written yet we reuse the host mapper
  if (geometry_container_3d.deformed_non_linear.size()) {
    this->map_particles_host = std::make_unique<MapParticlesHost>(
        sycl_target, particle_mesh_interface, config);
  }
}

void MapParticles3D::map(ParticleGroup &particle_group, const int map_cell) {

  if (this->map_particles_3d_regular) {
    // attempt to bin particles into regular geometry objects
    this->map_particles_3d_regular->map(particle_group, map_cell);
  }

  map_newton_internal(std::get<0>(this->map_particles_3d_deformed_linear),
                      particle_group, map_cell);
  map_newton_internal(std::get<1>(this->map_particles_3d_deformed_linear),
                      particle_group, map_cell);
  map_newton_internal(std::get<2>(this->map_particles_3d_deformed_linear),
                      particle_group, map_cell);
  map_newton_internal(std::get<3>(this->map_particles_3d_deformed_linear),
                      particle_group, map_cell);

  bool particles_not_mapped = true;
  if (this->map_particles_host) {

    // are there particles whcih are not yet mapped into cells
    particles_not_mapped =
        this->map_particles_common->check_map(particle_group, map_cell, false);

    // attempt to bin the remaining particles into deformed cells if there are
    // deformed cells.
    if (particles_not_mapped) {
      this->map_particles_host->map(particle_group, map_cell);
    }
  }

  // if there are particles not yet mapped this may be an error depending on
  // which stage of NESO-Particles hybrid move we are at.
  particles_not_mapped =
      this->map_particles_common->check_map(particle_group, map_cell, true);

  NESOASSERT(!particles_not_mapped,
             "Failed to find cell containing one or more particles.");
}

} // namespace NESO
