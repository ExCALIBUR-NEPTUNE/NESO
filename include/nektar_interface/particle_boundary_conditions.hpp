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
#include "composite_interaction/composite_collections.hpp"
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

class NektarCompositeReflection {
protected:
  struct NormalType {
    REAL x;
    REAL y;
    REAL z;

    inline NormalType &operator=(const int v) {
      this->x = v;
      this->y = v;
      this->z = v;
      return *this;
    }
  };

  SYCLTargetSharedPtr sycl_target;
  std::shared_ptr<CompositeInteraction::CompositeCollections>
      composite_collections;
  std::vector<int> composite_indices;
  std::map<int, std::set<int>> collected_geoms;
  std::unique_ptr<BlockedBinaryTree<INT, NormalType, 8>> map_geoms_normals;
  std::shared_ptr<LocalArray<BlockedBinaryNode<INT, NormalType, 8> *>> la_root;
  std::unique_ptr<ErrorPropagate> ep;
  Sym<REAL> velocity_sym;

  void collect();

public:
  /**
   * TODO
   */
  NektarCompositeReflection(
      Sym<REAL> velocity_sym, SYCLTargetSharedPtr sycl_target,
      std::shared_ptr<CompositeInteraction::CompositeCollections>
          composite_collections,
      std::vector<int> &composite_indices);

  void execute(std::map<int, ParticleSubGroupSharedPtr> &particle_groups);
};

} // namespace NESO
#endif
