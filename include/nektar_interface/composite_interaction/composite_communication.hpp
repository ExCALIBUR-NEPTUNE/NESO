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
  /// Disable (implicit) copies.
  CompositeCommunication(const CompositeCommunication &st) = delete;
  /// Disable (implicit) copies.
  CompositeCommunication &operator=(CompositeCommunication const &a) = delete;

  ~CompositeCommunication() { this->free(); }

  /**
   *  TODO
   */
  CompositeCommunication(MPI_Comm comm_in) {
    MPICHK(MPI_Comm_dup(comm_in, &this->comm));
    MPICHK(MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm,
                            &this->win_data, &this->win));
    this->allocated = true;
    MPICHK(MPI_Comm_rank(this->comm, &this->rank));
  }

  /**
   *  TODO
   */
  inline int get_num_in_edges(const std::vector<int> &ranks_out) {
    this->win_data[0] = 0;
    MPICHK(MPI_Barrier(comm));
    const int one[1] = {1};
    int recv[1];
    for (int rank : ranks_out) {
      MPICHK(MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, this->win));
      MPICHK(MPI_Get_accumulate(one, 1, MPI_INT, recv, 1, MPI_INT, rank, 0, 1,
                                MPI_INT, MPI_SUM, this->win));
      MPICHK(MPI_Win_unlock(rank, this->win));
    }
    MPICHK(MPI_Barrier(comm));
    const int num_ranks = this->win_data[0];
    return num_ranks;
  }

  /**
   *  TODO
   */
  inline void get_in_edges(const std::vector<int> &ranks_out,
                           std::vector<int> &ranks_in) {
    // number of incoming edges in this pattern
    const int num_incoming = this->get_num_in_edges(ranks_out);

    ranks_in.resize(num_incoming);
    std::vector<MPI_Request> recv_requests(num_incoming);
    for (int rankx = 0; rankx < num_incoming; rankx++) {
      MPICHK(MPI_Irecv(ranks_in.data() + rankx, 1, MPI_INT, MPI_ANY_SOURCE, 77,
                       this->comm, recv_requests.data() + rankx));
    }
    for (const int rankx : ranks_out) {
      MPICHK(MPI_Send(&this->rank, 1, MPI_INT, rankx, 77, this->comm));
    }
    MPICHK(
        MPI_Waitall(num_incoming, recv_requests.data(), MPI_STATUSES_IGNORE));
  }

  /**
   *  TODO
   */
  inline void free() {
    if (this->allocated) {
      MPICHK(MPI_Win_free(&this->win));
      MPICHK(MPI_Comm_free(&this->comm));
      this->allocated = false;
    }
  }
};

} // namespace NESO::CompositeInteraction

#endif
