#ifndef __COMPOSITE_COMMUNICATION_H_
#define __COMPOSITE_COMMUNICATION_H_

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

#include <neso_particles.hpp>
using namespace NESO::Particles;

#include <nektar_interface/geometry_transport/packed_geom_2d.hpp>
#include <nektar_interface/particle_mesh_interface.hpp>

#include <map>
#include <set>
#include <utility>
#include <vector>

#include <mpi.h>

namespace NESO::CompositeInteraction {

/**
 * Developer oriented type for communicating geometry objects between ranks.
 */
class CompositeCommunication {
protected:
  MPI_Comm comm;
  bool allocated;
  MPI_Win win;
  int *win_data;
  int rank;

public:
  struct ExplicitZeroInitInt {
    int value{0};
  };

  /// Disable (implicit) copies.
  CompositeCommunication(const CompositeCommunication &st) = delete;
  /// Disable (implicit) copies.
  CompositeCommunication &operator=(CompositeCommunication const &a) = delete;

  ~CompositeCommunication() { this->free(); }

  /**
   *  Create new communication object on a communicator. Must be called
   *  collectively on the communicator.
   *
   *  @param comm_in Communicator to use.
   */
  CompositeCommunication(MPI_Comm comm_in);

  /**
   * Must be called collectively on the communicator.
   *
   *  @param ranks_out The remote ranks which this MPI rank will send to.
   *  @returns The number of remote MPI ranks which will send to this MPI rank.
   */
  int get_num_in_edges(const std::vector<int> &ranks_out);

  /**
   *   Determine the remote MPI ranks which will send to this MPI rank. Must be
   * called collectively on the communicator.
   *
   *  @param[in] ranks_out The remote ranks which this MPI rank will send to.
   *  @param[in, out] ranks_in The vector to populate with remote MPI ranks
   *  which will send to this MPI rank.
   */
  void get_in_edges(const std::vector<int> &ranks_out,
                    std::vector<int> &ranks_in);

  /**
   *  Exchange send/recv counts between ranks. Must be called collectively on
   * the communicator.
   *
   *  @param[in] send_ranks The remote MPI ranks which this rank will send to.
   *  @param[in] recv_ranks The remote MPI ranks which will send data to this
   *  rank.
   *  @param[in] send_counts The send counts for each rank in send_ranks that
   *  this rank will send.
   *  @param[in, out] recv_counts The counts which each rank in recv_ranks will
   *  send to this rank.
   */
  void exchange_send_counts(const std::vector<int> &send_ranks,
                            const std::vector<int> &recv_ranks,
                            const std::vector<int> &send_counts,
                            std::vector<int> &recv_counts);

  /**
   *  Exchange the MeshHierarchy cells required be each MPI rank. Must be called
   * collectively on the communicator.
   *
   *  @param[in] send_ranks The remote MPI ranks which this rank will send to.
   *  @param[in] recv_ranks The remote MPI ranks which will send data to this
   *  rank.
   *  @param[in] send_counts The send counts for each rank in send_ranks that
   *  this rank will send.
   *  @param[in] recv_counts The counts which each rank in recv_ranks will
   *  send to this rank.
   *  @param[in] rank_send_cells_map Map from each send rank to the
   *  MeshHierarchy cells which are required.
   *  @param[in, out] rank_recv_cells_map Map from rank in recv_ranks to the
   *  cells which are requested.
   */
  void exchange_requested_cells(
      const std::vector<int> &send_ranks, const std::vector<int> &recv_ranks,
      const std::vector<int> &send_counts, const std::vector<int> &recv_counts,
      const std::map<int, std::vector<std::int64_t>> &rank_send_cells_map,
      std::map<int, std::vector<std::int64_t>> &rank_recv_cells_map);

  /**
   *  Exchange the number of geometry objects in each  MeshHierarchy cell which
   *  was requested. Must be called collectively on the communicator.
   *
   *  @param[in] send_ranks The remote MPI ranks which this rank will send to.
   *  @param[in] recv_ranks The remote MPI ranks which will send data to this
   *  rank.
   *  @param[in] rank_send_cells_map Map from each send rank to the
   *  MeshHierarchy cells which are required.
   *  @param[in] rank_recv_cells_map Map from rank in recv_ranks to the
   *  cells which are requested.
   *  @param[in, out] packed_geoms_count On output for each requested cell this
   *  map contains the number of geometry objects in the cell.
   */
  void exchange_requested_cells_counts(
      const std::vector<int> &send_ranks, const std::vector<int> &recv_ranks,
      const std::map<int, std::vector<std::int64_t>> &rank_send_cells_map,
      const std::map<int, std::vector<std::int64_t>> &rank_recv_cells_map,
      std::map<INT, CompositeCommunication::ExplicitZeroInitInt>
          &packed_geoms_count);

  /**
   *  Exchange the geometry objects in each  MeshHierarchy cell which
   *  was requested. Must be called collectively on the communicator.
   *
   *  @param[in] send_ranks The remote MPI ranks which this rank will send to.
   *  @param[in] recv_ranks The remote MPI ranks which will send data to this
   *  rank.
   *  @param[in] rank_send_cells_map Map from each send rank to the
   *  MeshHierarchy cells which are required.
   *  @param[in] rank_recv_cells_map Map from rank in recv_ranks to the
   *  cells which are requested.
   *  @param[in] packed_geoms_count On output for each requested cell this
   *  map contains the number of geometry objects in the cell.
   *  @param[in, out] packed_geoms Populated on return with packed geometry
   *  objects for each cell.
   */
  void exchange_packed_cells(
      const std::uint64_t max_buf_size, const std::vector<int> &send_ranks,
      const std::vector<int> &recv_ranks,
      const std::map<int, std::vector<std::int64_t>> &rank_send_cells_map,
      const std::map<int, std::vector<std::int64_t>> &rank_recv_cells_map,
      std::map<INT, CompositeCommunication::ExplicitZeroInitInt>
          &packed_geoms_count,
      std::map<INT, std::vector<unsigned char>> &packed_geoms);

  /**
   *  Free the container. Must be called collectively on the communicator.
   */
  void free();
};

} // namespace NESO::CompositeInteraction

#endif
