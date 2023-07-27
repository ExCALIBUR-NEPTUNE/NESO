#include "nektar_interface/particle_cell_mapping/nektar_graph_local_mapper.hpp"

namespace NESO {

void NektarGraphLocalMapper::particle_group_callback(
    ParticleGroup &particle_group) {

  particle_group.add_particle_dat(
      ParticleDat(particle_group.sycl_target,
                  ParticleProp(Sym<REAL>("NESO_REFERENCE_POSITIONS"),
                               particle_group.domain->mesh->get_ndim()),
                  particle_group.domain->mesh->get_cell_count()));
};

NektarGraphLocalMapper::NektarGraphLocalMapper(
    SYCLTargetSharedPtr sycl_target,
    ParticleMeshInterfaceSharedPtr particle_mesh_interface,
    ParameterStoreSharedPtr config)
    : sycl_target(sycl_target),
      particle_mesh_interface(particle_mesh_interface) {

  const int ndim = this->particle_mesh_interface->ndim;
  if (ndim == 2) {
    this->map_particles_2d = std::make_unique<MapParticles2D>(
        this->sycl_target, this->particle_mesh_interface, config);
  } else if (ndim == 3) {
    this->map_particles_3d = std::make_unique<MapParticles3D>(
        this->sycl_target, this->particle_mesh_interface, config);
  } else {
    NESOASSERT(false, "Unsupported number of dimensions.");
  }
};

/**
 *  Called internally by NESO-Particles to map positions to Nektar++
 *  geometry objects.
 */
void NektarGraphLocalMapper::map(ParticleGroup &particle_group,
                                 const int map_cell) {
  const int ndim = this->particle_mesh_interface->ndim;
  if (ndim == 2) {
    this->map_particles_2d->map(particle_group, map_cell);
  } else if (ndim == 3) {
    this->map_particles_3d->map(particle_group, map_cell);
  }
}

} // namespace NESO
