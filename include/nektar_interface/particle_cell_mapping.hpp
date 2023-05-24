#ifndef __PARTICLE_CELL_MAPPING_H__
#define __PARTICLE_CELL_MAPPING_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

#include "candidate_cell_mapping.hpp"
#include "coordinate_mapping.hpp"
#include "particle_cell_mapping_2d.hpp"
#include "particle_cell_mapping_3d.hpp"
#include "particle_mesh_interface.hpp"

#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {

/**
 * Class to map particle positions to Nektar++ cells. Implemented for triangles
 * and quads.
 */
class NektarGraphLocalMapperT : public LocalMapper {
private:
  SYCLTargetSharedPtr sycl_target;
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;
  const double tol;
  std::unique_ptr<MapParticles2D> map_particles_2d;
  std::unique_ptr<MapParticles3D> map_particles_3d;

public:
  ~NektarGraphLocalMapperT(){};

  /**
   * Callback for ParticleGroup to execute for additional setup of the
   * LocalMapper that may involve the ParticleGroup.
   *
   * @param particle_group ParticleGroup instance.
   */
  inline void particle_group_callback(ParticleGroup &particle_group) {

    particle_group.add_particle_dat(
        ParticleDat(particle_group.sycl_target,
                    ParticleProp(Sym<REAL>("NESO_REFERENCE_POSITIONS"),
                                 particle_group.domain->mesh->get_ndim()),
                    particle_group.domain->mesh->get_cell_count()));
  };

  /**
   *  Construct a new mapper object.
   *
   *  @param sycl_target SYCLTarget to use.
   *  @param particle_mesh_interface Interface between NESO-Particles and
   * Nektar++ mesh.
   *  @param tol Tolerance to pass to Nektar++ to bin particles into cells.
   */
  NektarGraphLocalMapperT(
      SYCLTargetSharedPtr sycl_target,
      ParticleMeshInterfaceSharedPtr particle_mesh_interface,
      const double tol = 1.0e-10)
      : sycl_target(sycl_target),
        particle_mesh_interface(particle_mesh_interface), tol(tol) {

    const int ndim = this->particle_mesh_interface->ndim;
    if (ndim == 2) {
      this->map_particles_2d = std::make_unique<MapParticles2D>(
          this->sycl_target, this->particle_mesh_interface);
    } else if (ndim == 3) {
      this->map_particles_3d = std::make_unique<MapParticles3D>(
          this->sycl_target, this->particle_mesh_interface);
    }
  };

  /**
   *  Called internally by NESO-Particles to map positions to Nektar++
   *  geometry objects.
   */
  inline void map(ParticleGroup &particle_group, const int map_cell = -1) {
    const int ndim = this->particle_mesh_interface->ndim;
    if (ndim == 2) {
      this->map_particles_2d->map(particle_group, map_cell);
    } else if (ndim == 3) {
      this->map_particles_3d->map(particle_group, map_cell);
    } else {
      NESOASSERT(false, "Unsupported number of dimensions.");
    }
  }
};

} // namespace NESO

#endif
