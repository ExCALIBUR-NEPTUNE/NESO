#ifndef __COMPOSITE_TRANSPORT_H_
#define __COMPOSITE_TRANSPORT_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <neso_particles.hpp>
using namespace NESO::Particles;

#include <nektar_interface/geometry_transport/halo_extension.hpp>
#include <nektar_interface/geometry_transport/packed_geom_2d.hpp>
#include <nektar_interface/particle_mesh_interface.hpp>

#include <cstdint>
#include <map>
#include <mpi.h>
#include <set>
#include <utility>
#include <vector>

#include "composite_communication.hpp"

namespace NESO::CompositeInteraction {

/**
 * TODO
 */
class CompositeTransport {
protected:
  const int ndim;
  std::unique_ptr<CompositeCommunication> composite_communication;
  MPI_Comm comm;
  bool allocated;
  // map from mesh hierarchy cells to the packed geoms for that cell
  std::map<INT, std::vector<unsigned char>> packed_geoms;
  // map from mesh hierarchy cells to the number of packed geoms for that cell
  std::map<INT, CompositeCommunication::ExplicitZeroInitInt> packed_geoms_count;

  int rank;
  // the maximum size of the packed geoms accross all ranks
  std::uint64_t max_buf_size;

  ParticleMeshInterfaceSharedPtr particle_mesh_interface;

public:
  /// Disable (implicit) copies.
  CompositeTransport(const CompositeTransport &st) = delete;
  /// Disable (implicit) copies.
  CompositeTransport &operator=(CompositeTransport const &a) = delete;

  /// The composite indices for which the class detects intersections with.
  const std::vector<int> composite_indices;

  // are the geoms in the mesh hierarchy cell owned or already requested
  std::set<INT> held_cells;

  ~CompositeTransport() { this->free(); }

  /**
   *  TODO
   */
  void free();

  /**
   * TODO
   */
  void get_geometry(
      const INT cell,
      std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::QuadGeom>>>
          &remote_quads,
      std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::TriGeom>>>
          &remote_tris);

  /**
   * TODO
   * @param[in, out] cells_in MeshHierarchy cells which are required. On exit
   * hold the cells which are new to this MPI rank.
   * @returns Number of cells collected.
   */
  int collect_geometry(std::set<INT> &cells_in);

  /**
   *  TODO
   */
  CompositeTransport(ParticleMeshInterfaceSharedPtr particle_mesh_interface,
                     std::vector<int> &composite_indices);
};

} // namespace NESO::CompositeInteraction

#endif
