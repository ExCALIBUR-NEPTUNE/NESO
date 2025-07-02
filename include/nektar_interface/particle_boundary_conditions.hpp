#ifndef __PARTICLE_BOUNDARY_CONDITIONS_H__
#define __PARTICLE_BOUNDARY_CONDITIONS_H__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <set>
#include <stack>
#include <vector>

#include <mpi.h>

#include "bounding_box_intersection.hpp"
#include "composite_interaction/composite_intersection.hpp"
#include "parameter_store.hpp"
#include "special_functions.hpp"
#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {

/**
 * Periodic boundary conditions implementation designed to work with a
 * CartesianHMesh.
 */
class NektarCartesianPeriodic {
private:
  BufferDevice<double> d_origin;
  BufferDevice<double> d_extents;
  SYCLTargetSharedPtr sycl_target;
  ParticleDatSharedPtr<REAL> position_dat;
  const int ndim;
  ParticleLoopSharedPtr loop;

public:
  double global_origin[3];
  double global_extent[3];
  ~NektarCartesianPeriodic(){};

  /**
   * Construct instance to apply periodic boundary conditions to particles
   * within the passed ParticleDat.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param graph Nektar++ MeshGraph on which particles move.
   * @param position_dat ParticleDat containing particle positions.
   */
  NektarCartesianPeriodic(SYCLTargetSharedPtr sycl_target,
                          Nektar::SpatialDomains::MeshGraphSharedPtr graph,
                          ParticleDatSharedPtr<REAL> position_dat);

  /**
   * Apply periodic boundary conditions to the particle positions in the
   * ParticleDat this instance was created with.
   */
  void execute();
};

/**
 * Implementation of a reflection process which truncates the particle
 * trajectory at the mesh boundary.
 *
 * If particle positions are stored in a ParticleDat with Sym "P" then P will
 * be set to a value just inside the domain at the intersection point of the
 * particle trajectory and the composite which the particle hits.
 *
 * This implementation assumes that each particle carries an additional
 * property of two components that stores the proportion through the time step
 * the particle is at. The first component of this property holds the total
 * proportion of the time step that the particle has been integrated through.
 * The second component holds the amount of that proportion though which the
 * particle was integrated in the last modification to the properties of the
 * particle (i.e. position).
 *
 * This time property allows the time of the particle to be re-wound to the
 * point of intersection with a composite.
 *
 */
class NektarCompositeTruncatedReflection {
protected:
  SYCLTargetSharedPtr sycl_target;
  std::shared_ptr<CompositeInteraction::CompositeIntersection>
      composite_intersection;
  std::vector<int> composite_indices;
  Sym<REAL> velocity_sym;
  Sym<REAL> time_step_prop_sym;
  REAL reset_distance;
  int ndim;

  // The standard NESO-Particles reflection implementation
  std::shared_ptr<BoundaryReflection> boundary_reflection;

public:
  /**
   * Implementation of a reflection process which truncates the particle
   * trajectory at the mesh boundary.
   *
   * @param velocity_sym Symbol of ParticleDat which contains particle
   * velocities.
   * @param time_step_prop_sym Symbol of ParticleDat which contains the time
   * step of the particle. This property should have at least two components.
   * @param sycl_target Compute device for all ParticleGroups which will use the
   * instance.
   * @param mesh Mesh for all ParticleGroups which will use the instance.
   * @param composite_indices Vector of boundary composites for which particles
   * should be reflected when an intersection occurs.
   * @param config Configuration to pass to composite intersection routines.
   */
  NektarCompositeTruncatedReflection(
      Sym<REAL> velocity_sym, Sym<REAL> time_step_prop_sym,
      SYCLTargetSharedPtr sycl_target,
      std::shared_ptr<ParticleMeshInterface> mesh,
      std::vector<int> &composite_indices,
      ParameterStoreSharedPtr config = std::make_shared<ParameterStore>());

  /**
   * Apply the reflection process. This method should be called after a time
   * step has been performed.
   *
   * @param particle_sub_group ParticleSubGroup of particles to apply truncated
   * reflection to.
   */
  void execute(ParticleSubGroupSharedPtr particle_sub_group);

  /**
   * Method to call before particle positions are updated by a time stepping
   * process.
   *
   * @param particle_sub_group ParticleSubGroup of particles to apply truncated
   * reflection to.
   */
  void pre_advection(ParticleSubGroupSharedPtr particle_sub_group);
};

} // namespace NESO
#endif
