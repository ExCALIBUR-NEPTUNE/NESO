#ifndef __GLOBAL_BOUNDING_BOX_H__
#define __GLOBAL_BOUNDING_BOX_H__

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

//#include "bounding_box_intersection.hpp"
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
class GlobalBoundingBox {
private:
  const int ndim;
  double m_global_origin[3];
  double m_global_upper[3];
  double m_global_extent[3];
  double m_volume;
public:
  ~GlobalBoundingBox(){};

  /**
   * Construct instance to apply periodic boundary conditions to particles
   * within the passed ParticleDat.
   *
   * @param sycl_target SYCLTargetSharedPtr to use as compute device.
   * @param graph Nektar++ MeshGraph on which particles move.
   */
  GlobalBoundingBox(SYCLTargetSharedPtr sycl_target,
                    Nektar::SpatialDomains::MeshGraphSharedPtr graph)
      : ndim(graph->GetMeshDimension()) {

    NESOASSERT(this->ndim <= 3, "bad mesh ndim");

    auto verticies = graph->GetAllPointGeoms();

    double origin[3];
    double upper[3];
    for (int dimx = 0; dimx < 3; dimx++) {
      origin[dimx] = std::numeric_limits<double>::max();
      upper[dimx] = std::numeric_limits<double>::min();
    }

    for (auto &vx : verticies) {
      Nektar::NekDouble x, y, z;
      vx.second->GetCoords(x, y, z);
      origin[0] = std::min(origin[0], x);
      origin[1] = std::min(origin[1], y);
      origin[2] = std::min(origin[2], z);
      upper[0] = std::max(upper[0], x);
      upper[1] = std::max(upper[1], y);
      upper[2] = std::max(upper[2], z);
    }

    MPICHK(MPI_Allreduce(origin, m_global_origin, 3, MPI_DOUBLE, MPI_MIN,
                         sycl_target->comm_pair.comm_parent));
    MPICHK(MPI_Allreduce(upper, m_global_upper, 3, MPI_DOUBLE, MPI_MAX,
                         sycl_target->comm_pair.comm_parent));

    for (int dimx = 0; dimx < 3; dimx++) {
      m_global_extent[dimx] = m_global_upper[dimx] - m_global_origin[dimx];
    }

    m_volume = 1.0;
    for (int dimx = 0; dimx < this->ndim; dimx++){
      m_volume *= m_global_origin[dimx];
    }
  };

  double* global_origin() { return m_global_origin; }
  double* global_upper() { return m_global_upper; }
  double* global_extent() { return m_global_extent; }
  const double* global_origin() const { return m_global_origin; }
  const double* global_upper() const { return m_global_upper; }
  const double* global_extent() const { return m_global_extent; }

  double global_origin(int i) { return m_global_origin[i]; } // TODO const
  double global_upper(int i) { return m_global_upper[i]; } // TODO const
  double global_extent(int i) { return m_global_extent[i]; } // TODO const

  double global_volume() { return m_volume; } // TODO const

};

} // namespace NESO
#endif
