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
 * High level class that collects geometry information for geometry objects
 * which are members of certain composites of interest.
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
   * Free the data structure. Must be called collectively on the communicator.
   */
  void free();

  /**
   * Unpack the geometry for a given cell.
   *
   * @param[in] cell MeshHierarchy cell to unpack geometry for.
   * @param[in, out] remote_quads On return contains the unpacked remote quads
   * for the cell.
   * @param[in, out] remote_tris On return contains the unpacked remote
   * triangles for the cell.
   */
  void get_geometry(
      const INT cell,
      std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::QuadGeom>>>
          &remote_quads,
      std::vector<std::shared_ptr<RemoteGeom2D<SpatialDomains::TriGeom>>>
          &remote_tris);

  /**
   * Collect on this MPI rank geometry information for requested MeshHierarchy
   * cells. Must be called collectively on the communicator.
   *
   * @param[in, out] cells_in MeshHierarchy cells which are required. On exit
   * hold the cells which are new to this MPI rank.
   * @returns Number of cells collected.
   */
  int collect_geometry(std::set<INT> &cells_in);

  /**
   *  Construct new transport instance for a given mesh and set of composite
   * indices.
   *
   *  @param particle_mesh_interface Mesh to collect geometry information on.
   *  @param composite_indices Composite indices to collect geometry objects
   * for.
   */
  CompositeTransport(ParticleMeshInterfaceSharedPtr particle_mesh_interface,
                     std::vector<int> &composite_indices);
};

} // namespace NESO::CompositeInteraction

#endif
