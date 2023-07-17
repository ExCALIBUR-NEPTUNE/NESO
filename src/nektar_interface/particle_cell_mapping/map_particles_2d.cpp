#include <nektar_interface/particle_cell_mapping/map_particles_2d.hpp>

namespace NESO {

MapParticles2D::MapParticles2D(
    SYCLTargetSharedPtr sycl_target,
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    ParameterStoreSharedPtr config)
    : sycl_target(sycl_target),
      particle_mesh_interface(particle_mesh_interface) {

  // determine if there are regular and deformed geometry objects
  this->count_regular = 0;
  this->count_deformed = 0;

  std::map<int, std::shared_ptr<Nektar::SpatialDomains::Geometry2D>>
      geoms_local;
  get_all_elements_2d(particle_mesh_interface->graph, geoms_local);
  count_geometry_types(geoms_local, &count_regular, &count_deformed);
  count_geometry_types(particle_mesh_interface->remote_triangles,
                       &count_regular, &count_deformed);
  count_geometry_types(particle_mesh_interface->remote_quads, &count_regular,
                       &count_deformed);

  this->map_particles_common =
      std::make_unique<MapParticlesCommon>(sycl_target);

  if (this->count_deformed > 0) {

    std::map<int, std::shared_ptr<QuadGeom>> quads_local;
    std::vector<std::shared_ptr<RemoteGeom2D<QuadGeom>>> quads_remote;

    for (auto &geom : geoms_local) {
      auto t = geom.second->GetMetricInfo()->GetGtype();
      auto s = geom.second->GetShapeType();
      if ((t == eDeformed) && (s == eQuadrilateral)) {
        quads_local[geom.first] =
            std::dynamic_pointer_cast<QuadGeom>(geom.second);
      }
    }
    for (auto &geom : particle_mesh_interface->remote_quads) {
      auto t = geom->geom->GetMetricInfo()->GetGtype();
      auto s = geom->geom->GetShapeType();
      if ((t == eDeformed) && (s == eQuadrilateral)) {
        quads_remote.push_back(geom);
      }
    }

    this->map_particles_newton_linear_quad = std::make_unique<
        Newton::MapParticlesNewton<Newton::MappingQuadLinear2D>>(
        Newton::MappingQuadLinear2D{}, this->sycl_target, quads_local,
        quads_remote, config);
  }

  this->map_particles_2d_regular = std::make_unique<MapParticles2DRegular>(
      sycl_target, particle_mesh_interface, config);
}

void MapParticles2D::map(ParticleGroup &particle_group, const int map_cell) {

  if (this->count_regular > 0) {
    // attempt to bin particles into regular geometry objects
    this->map_particles_2d_regular->map(particle_group, map_cell);
  }

  bool particles_not_mapped = true;
  if (this->count_deformed > 0) {

    // are there particles which are not yet mapped into cells
    particles_not_mapped =
        this->map_particles_common->check_map(particle_group, map_cell, false);

    // attempt to bin the remaining particles into deformed cells if there are
    // deformed cells.
    if (particles_not_mapped) {
      this->map_particles_newton_linear_quad->map(particle_group, map_cell);
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
