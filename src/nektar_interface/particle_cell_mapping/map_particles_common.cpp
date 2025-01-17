
#include "nektar_interface/particle_cell_mapping/map_particles_common.hpp"

namespace NESO {

MapParticlesCommon::MapParticlesCommon(SYCLTargetSharedPtr sycl_target)
    : sycl_target(sycl_target),
      ep(std::make_unique<ErrorPropagate>(sycl_target)) {}

bool MapParticlesCommon::check_map(ParticleGroup &particle_group,
                                   const int map_cell) {
  this->ep->reset();
  auto k_ep = this->ep->device_ptr();
  auto mpi_ranks = particle_group.mpi_rank_dat;

  auto loop = particle_loop(
      mpi_ranks,
      [=](auto k_part_mpi_ranks) {
        NESO_KERNEL_ASSERT(k_part_mpi_ranks.at(1) > -1, k_ep);
      },
      Access::read(mpi_ranks));

  if (map_cell < 0) {
    loop->execute();
  } else {
    loop->execute(map_cell);
  }

  // If the return flag is true there are particles which were not binned into
  // cells. If the return flag is false all particles were binned into cells.
  const bool flag = this->ep->get_flag();
  return flag;
}

} // namespace NESO
