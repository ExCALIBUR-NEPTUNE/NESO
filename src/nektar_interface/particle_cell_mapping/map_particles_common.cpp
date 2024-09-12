
#include "nektar_interface/particle_cell_mapping/map_particles_common.hpp"

namespace NESO {

MapParticlesCommon::MapParticlesCommon(SYCLTargetSharedPtr sycl_target)
    : sycl_target(sycl_target),
      ep(std::make_unique<ErrorPropagate>(sycl_target)) {}

bool MapParticlesCommon::check_map(ParticleGroup &particle_group,
                                   const int map_cell, const bool final_map) {
  this->ep->reset();
  auto k_ep = this->ep->device_ptr();
  auto mpi_ranks = particle_group.mpi_rank_dat;

  auto loop = particle_loop(
      mpi_ranks,
      [=](auto k_part_mpi_ranks) {
        if (final_map) {
          // if a geom is not found and there is a non-null global
          // MPI rank then this function was called after the global
          // move and the lack of a local cell / mpi rank is a fatal
          // error.
          if ((k_part_mpi_ranks.at(0) > -1) && (k_part_mpi_ranks.at(1) < 0)) {
            NESO_KERNEL_ASSERT(false, k_ep);
          }
        } else {
          // This loop was called at an intermediate state to
          // determine if there exist particles which are not
          // mapped into cells. Hence only the local component of
          // the mapping dat is checked.
          if (k_part_mpi_ranks.at(1) < 0) {
            NESO_KERNEL_ASSERT(false, k_ep);
          }
        }
      },
      Access::read(mpi_ranks));

  if (map_cell > -1) {
    loop->execute(map_cell);
  } else {
    loop->execute();
  }

  if (this->ep->get_flag()) {
    // If the return flag is true there are particles which were not binned
    // into cells.
    return true;
  } else {
    // If the return flag is false all particles were binned into cells.
    return false;
  }
}

} // namespace NESO
