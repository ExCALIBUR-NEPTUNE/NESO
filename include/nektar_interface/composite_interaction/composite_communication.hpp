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
 * TODO
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
   *  TODO
   */
  CompositeCommunication(MPI_Comm comm_in);

  /**
   *  TODO
   */
  int get_num_in_edges(const std::vector<int> &ranks_out);

  /**
   *  TODO
   */
  void get_in_edges(const std::vector<int> &ranks_out,
                    std::vector<int> &ranks_in);

  /**
   *  TODO
   */
  void exchange_send_counts(const std::vector<int> &send_ranks,
                            const std::vector<int> &recv_ranks,
                            const std::vector<int> &send_counts,
                            std::vector<int> &recv_counts);

  /**
   *  TODO
   */
  void exchange_requested_cells(
      const std::vector<int> &send_ranks, const std::vector<int> &recv_ranks,
      const std::vector<int> &send_counts, const std::vector<int> &recv_counts,
      const std::map<int, std::vector<std::int64_t>> &rank_send_cells_map,
      std::map<int, std::vector<std::int64_t>> &rank_recv_cells_map);

  /**
   *  TODO
   */
  void exchange_requested_cells_counts(
      const std::vector<int> &send_ranks, const std::vector<int> &recv_ranks,
      const std::map<int, std::vector<std::int64_t>> &rank_send_cells_map,
      const std::map<int, std::vector<std::int64_t>> &rank_recv_cells_map,
      std::map<INT, CompositeCommunication::ExplicitZeroInitInt>
          &packed_geoms_count);

  /**
   *  TODO
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
   *  TODO
   */
  void free();
};

} // namespace NESO::CompositeInteraction

#endif
