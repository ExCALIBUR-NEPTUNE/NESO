#include <nektar_interface/composite_interaction/composite_communication.hpp>

namespace NESO::CompositeInteraction {

CompositeCommunication::CompositeCommunication(MPI_Comm comm_in) {
  this->comm = comm_in;
  MPICHK(MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL, this->comm,
                          &this->win_data, &this->win));
  this->allocated = true;
  MPICHK(MPI_Comm_rank(this->comm, &this->rank));
}

int CompositeCommunication::get_num_in_edges(
    const std::vector<int> &ranks_out) {
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

void CompositeCommunication::get_in_edges(const std::vector<int> &ranks_out,
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
  MPICHK(MPI_Waitall(num_incoming, recv_requests.data(), MPI_STATUSES_IGNORE));

  MPICHK(MPI_Barrier(comm));
}

void CompositeCommunication::exchange_send_counts(
    const std::vector<int> &send_ranks, const std::vector<int> &recv_ranks,
    const std::vector<int> &send_counts, std::vector<int> &recv_counts) {

  const int num_send_ranks = send_ranks.size();
  const int num_recv_ranks = recv_ranks.size();

  NESOASSERT(num_send_ranks <= send_counts.size(),
             "Missmatch in send buffer sizes.");
  NESOASSERT(num_recv_ranks <= recv_counts.size(),
             "Missmatch in recv buffer sizes.");

  std::vector<MPI_Request> recv_requests(num_recv_ranks);
  for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
    const int remote_rank = recv_ranks[rankx];
    MPICHK(MPI_Irecv(&recv_counts[rankx], 1, MPI_INT, remote_rank, 78,
                     this->comm, recv_requests.data() + rankx));
  }

  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    const int remote_rank = send_ranks[rankx];
    MPICHK(
        MPI_Send(&send_counts[rankx], 1, MPI_INT, remote_rank, 78, this->comm));
  }

  std::vector<MPI_Status> status(num_recv_ranks);
  MPICHK(MPI_Waitall(num_recv_ranks, recv_requests.data(), status.data()));
}

void CompositeCommunication::exchange_requested_cells(
    const std::vector<int> &send_ranks, const std::vector<int> &recv_ranks,
    const std::vector<int> &send_counts, const std::vector<int> &recv_counts,
    const std::map<int, std::vector<std::int64_t>> &rank_send_cells_map,
    std::map<int, std::vector<std::int64_t>> &rank_recv_cells_map) {
  const int num_send_ranks = send_ranks.size();
  const int num_recv_ranks = recv_ranks.size();

  std::vector<MPI_Request> recv_requests(num_recv_ranks);
  for (int rankx = 0; rankx < num_recv_ranks; rankx++) {
    const int remote_rank = recv_ranks[rankx];
    const int num_cells = recv_counts[rankx];
    MPICHK(MPI_Irecv(rank_recv_cells_map.at(remote_rank).data(), num_cells,
                     MPI_INT64_T, remote_rank, 79, this->comm,
                     recv_requests.data() + rankx));
  }

  for (int rankx = 0; rankx < num_send_ranks; rankx++) {
    const int remote_rank = send_ranks[rankx];
    const int num_cells = send_counts[rankx];
    MPICHK(MPI_Send(rank_send_cells_map.at(remote_rank).data(), num_cells,
                    MPI_INT64_T, remote_rank, 79, this->comm));
  }

  MPICHK(
      MPI_Waitall(num_recv_ranks, recv_requests.data(), MPI_STATUSES_IGNORE));
}

void CompositeCommunication::exchange_requested_cells_counts(
    const std::vector<int> &send_ranks, const std::vector<int> &recv_ranks,
    const std::map<int, std::vector<std::int64_t>> &rank_send_cells_map,
    const std::map<int, std::vector<std::int64_t>> &rank_recv_cells_map,
    std::map<INT, CompositeCommunication::ExplicitZeroInitInt>
        &packed_geoms_count) {

  int num_recv_requests = 0;
  for (auto &rx : rank_send_cells_map) {
    num_recv_requests += rx.second.size();
  }
  // start the recv operations for packed geoms
  std::vector<MPI_Request> recv_requests(num_recv_requests);
  MPI_Request *recv_requests_ptr = recv_requests.data();
  num_recv_requests = 0;
  for (const int remote_rank : send_ranks) {
    if (remote_rank != this->rank) {
      const auto &cells = rank_send_cells_map.at(remote_rank);
      for (const auto &cellx : cells) {
        MPICHK(MPI_Irecv(&(packed_geoms_count[cellx].value), 1, MPI_INT,
                         remote_rank, 80, this->comm, recv_requests_ptr++));
        num_recv_requests++;
      }
    }
  }

  // send geoms
  for (const int remote_rank : recv_ranks) {
    if (remote_rank != this->rank) {
      for (const auto &cellx : rank_recv_cells_map.at(remote_rank)) {
        MPICHK(MPI_Send(&(packed_geoms_count[cellx].value), 1, MPI_INT,
                        remote_rank, 80, this->comm));
      }
    }
  }

  MPICHK(MPI_Waitall(num_recv_requests, recv_requests.data(),
                     MPI_STATUSES_IGNORE));
}

void CompositeCommunication::exchange_packed_cells(
    const std::uint64_t max_buf_size, const std::vector<int> &send_ranks,
    const std::vector<int> &recv_ranks,
    const std::map<int, std::vector<std::int64_t>> &rank_send_cells_map,
    const std::map<int, std::vector<std::int64_t>> &rank_recv_cells_map,
    std::map<INT, CompositeCommunication::ExplicitZeroInitInt>
        &packed_geoms_count,
    std::map<INT, std::vector<unsigned char>> &packed_geoms) {

  int num_recv_requests = 0;
  for (auto &rx : rank_send_cells_map) {
    num_recv_requests += rx.second.size();
    for (auto &cellx : rx.second) {
      NESOASSERT(packed_geoms.count(cellx) == 0,
                 "trying to recv a cell we already hold?");
      packed_geoms[cellx] = std::vector<unsigned char>(max_buf_size);
    }
  }
  const int max_buf_sizei = static_cast<int>(max_buf_size);

  // start the recv operations for packed geoms
  std::vector<MPI_Request> recv_requests(num_recv_requests);
  MPI_Request *recv_requests_ptr = recv_requests.data();
  num_recv_requests = 0;
  for (const int remote_rank : send_ranks) {
    const auto &cells = rank_send_cells_map.at(remote_rank);
    for (const auto &cellx : cells) {
      if (packed_geoms_count[cellx].value > 0) {
        NESOASSERT(packed_geoms.at(cellx).size() >= max_buf_size,
                   "recv buffer incorrectly sized");
        MPICHK(MPI_Irecv(packed_geoms.at(cellx).data(), max_buf_sizei,
                         MPI_UNSIGNED_CHAR, remote_rank, 81, this->comm,
                         recv_requests_ptr++));
        num_recv_requests++;
      }
    }
  }

  // send geoms
  for (const int remote_rank : recv_ranks) {
    for (const auto &cellx : rank_recv_cells_map.at(remote_rank)) {
      if (packed_geoms_count[cellx].value > 0) {
        NESOASSERT(packed_geoms.at(cellx).size() >= max_buf_size,
                   "trying to send a cell we don't hold geoms for");

        MPICHK(MPI_Send(packed_geoms.at(cellx).data(), max_buf_sizei,
                        MPI_UNSIGNED_CHAR, remote_rank, 81, this->comm));
      }
    }
  }

  MPICHK(MPI_Waitall(num_recv_requests, recv_requests.data(),
                     MPI_STATUSES_IGNORE));
}

void CompositeCommunication::free() {
  if (this->allocated) {
    MPICHK(MPI_Win_free(&this->win));
    this->allocated = false;
  }
}

} // namespace NESO::CompositeInteraction
