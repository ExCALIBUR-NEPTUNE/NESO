#ifndef __MAP_PARTICLES_COMMON_H__
#define __MAP_PARTICLES_COMMON_H__

#include <cmath>
#include <map>
#include <memory>
#include <mpi.h>
#include <vector>

#include "../particle_mesh_interface.hpp"
#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar;
using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {

/**
 *  Helper class to determine if particles have been binned into cells.
 * Particles map be binned into local cells at the intermediate step of the
 * hybrid transfer and must be binned into cells at the end of hybrid transfer.
 */
class MapParticlesCommon {
protected:
  /// Disable (implicit) copies.
  MapParticlesCommon(const MapParticlesCommon &st) = delete;
  /// Disable (implicit) copies.
  MapParticlesCommon &operator=(MapParticlesCommon const &a) = delete;

  SYCLTargetSharedPtr sycl_target;
  std::unique_ptr<ErrorPropagate> ep;

public:
  /**
   *  Create new instance for given SYCLTarget.
   *
   *  @param sycl_target SYCLTarget to use.
   */
  MapParticlesCommon(SYCLTargetSharedPtr sycl_target);

  /**
   *  Returns true if there are particles that were not binned into cells.
   *
   *  @param particle_group ParticleGroup to check particles in.
   *  @param map_cell Only check particles within a particular NESO::Particles
   * cell.
   *  @param final_map Is this check the final or intermediate step of the
   * hybrid move.
   */
  bool check_map(ParticleGroup &particle_group, const int map_cell = -1,
                 const bool final_map = true);
};

} // namespace NESO

#endif
