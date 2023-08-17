#ifndef __COMPOSITE_COLLECTIONS_H_
#define __COMPOSITE_COLLECTIONS_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <neso_particles.hpp>
using namespace NESO::Particles;

#include "composite_transport.hpp"
#include <nektar_interface/geometry_transport/packed_geom_2d.hpp>
#include <nektar_interface/particle_mesh_interface.hpp>

#include <map>
#include <memory>
#include <set>
#include <utility>
#include <vector>

namespace NESO::CompositeInteraction {

/**
 * TODO
 */
class CompositeCollections {
protected:
  ParticleMeshInterfaceSharedPtr particle_mesh_interface;

  inline void collect_cell(const INT cell) {

    std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::QuadGeom>>>
        remote_quads;
    std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::TriGeom>>>
        remote_tris;

    this->composite_transport->get_geometry(cell, remote_quads, remote_tris);
    nprint("cell:", cell, "qsize:", remote_quads.size(),
           "tsize:", remote_tris.size());

    for (auto gx : remote_quads) {
      nprint("\t quad:", gx->rank, gx->id);
    }
    for (auto gx : remote_tris) {
      nprint("\t tri:", gx->rank, gx->id);
    }
  }

public:
  /// Disable (implicit) copies.
  CompositeCollections(const CompositeCollections &st) = delete;
  /// Disable (implicit) copies.
  CompositeCollections &operator=(CompositeCollections const &a) = delete;

  /// SYCLTarget to use for computation.
  SYCLTargetSharedPtr sycl_target;

  /// The composite transport instance.
  std::unique_ptr<CompositeTransport> composite_transport;

  /**
   * TODO
   */
  inline void free() { this->composite_transport->free(); }

  /**
   *  TODO
   */
  CompositeCollections(SYCLTargetSharedPtr sycl_target,
                       ParticleMeshInterfaceSharedPtr particle_mesh_interface,
                       std::vector<int> &composite_indices)
      : sycl_target(sycl_target),
        particle_mesh_interface(particle_mesh_interface) {

    this->composite_transport = std::make_unique<CompositeTransport>(
        particle_mesh_interface, composite_indices);

    for (auto cx : this->composite_transport->held_cells) {
      this->collect_cell(cx);
    }
  }

  /**
   * TODO
   */
  inline void collect_geometry(std::set<INT> &cells) {

    this->composite_transport->collect_geometry(cells);
    for (auto cx : cells) {
      this->collect_cell(cx);
    }
  }
};

} // namespace NESO::CompositeInteraction

#endif
