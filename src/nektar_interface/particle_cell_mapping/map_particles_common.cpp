
#include "nektar_interface/particle_cell_mapping/map_particles_common.hpp"

namespace NESO {

MapParticlesCommon::MapParticlesCommon(SYCLTargetSharedPtr sycl_target)
    : sycl_target(sycl_target),
      ep(std::make_unique<ErrorPropagate>(sycl_target)) {}

bool MapParticlesCommon::check_map(ParticleGroup &particle_group,
                                   const int map_cell, const bool final_map) {

  // Get kernel pointers to the ParticleDats
  auto cell_id_dat = particle_group.cell_id_dat;
  auto k_part_cell_ids = cell_id_dat->cell_dat.device_ptr();
  auto k_part_mpi_ranks = particle_group.mpi_rank_dat->cell_dat.device_ptr();

  // Get iteration set for particles, two cases single cell case or all cells
  const int max_cell_occupancy = (map_cell > -1)
                                     ? cell_id_dat->h_npart_cell[map_cell]
                                     : cell_id_dat->cell_dat.get_nrow_max();

  const int k_cell_offset = (map_cell > -1) ? map_cell : 0;
  const std::size_t local_size = 256;
  const auto div_mod = std::div(max_cell_occupancy, local_size);
  const int outer_size = div_mod.quot + (div_mod.rem == 0 ? 0 : 1);
  const std::size_t cell_count =
      (map_cell > -1) ? 1
                      : static_cast<std::size_t>(cell_id_dat->cell_dat.ncells);
  sycl::range<2> outer_iterset{local_size * outer_size, cell_count};
  sycl::range<2> local_iterset{local_size, 1};
  const auto k_npart_cell = cell_id_dat->d_npart_cell;

  this->ep->reset();
  auto k_ep = this->ep->device_ptr();

  this->sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(
            sycl::nd_range<2>(outer_iterset, local_iterset),
            [=](sycl::nd_item<2> idx) {
              const int cellx = idx.get_global_id(1) + k_cell_offset;
              const int layerx = idx.get_global_id(0);
              if (layerx < k_npart_cell[cellx]) {
                if (k_part_mpi_ranks[cellx][1][layerx] < 0) {

                  if (final_map) {
                    // if a geom is not found and there is a non-null global
                    // MPI rank then this function was called after the global
                    // move and the lack of a local cell / mpi rank is a fatal
                    // error.
                    if (((k_part_mpi_ranks)[cellx][0][layerx] > -1) &&
                        (k_part_mpi_ranks[cellx][1][layerx] < 0)) {
                      NESO_KERNEL_ASSERT(false, k_ep);
                    }
                  } else {
                    // This loop was called at an intermediate state to
                    // determine if there exist particles which are not
                    // mapped into cells. Hence only the local component of
                    // the mapping dat is checked.
                    if (k_part_mpi_ranks[cellx][1][layerx] < 0) {
                      NESO_KERNEL_ASSERT(false, k_ep);
                    }
                  }
                }
              }
            });
      })
      .wait_and_throw();

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
