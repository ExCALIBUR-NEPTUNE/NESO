#ifndef __COMPOSITE_INTERSECTION_H_
#define __COMPOSITE_INTERSECTION_H_

#include <neso_particles.hpp>
using namespace NESO::Particles;

namespace NESO::CompositeInteraction {

/**
 *  High-level class to detect and compute the intersection of a particle
 *  trajectory and a Nektar++ composite.
 */
class CompositeIntersection {
protected:
public:
  /// Disable (implicit) copies.
  CompositeIntersection(const CompositeIntersection &st) = delete;
  /// Disable (implicit) copies.
  CompositeIntersection &operator=(CompositeIntersection const &a) = delete;

  /// The NESO::Particles Sym<REAL> used to store the previous particle
  /// position.
  const static inline Sym<REAL> previous_position_sym =
      Sym<REAL>("NESO_COMP_INT_PREV_POS");

  /**
   *  TODO
   */
  CompositeIntersection() {}

  /**
   *  Method to store the current particle positions before an integration step.
   *
   *  @param particle_group Particles to store current positions of.
   */
  inline void pre_integration(ParticleGroupSharedPtr particle_group) {
    const auto position_dat = particle_group->position_dat;
    const int ndim = position_dat->ncomp;
    const auto sycl_target = particle_group->sycl_target;
    // If the previous position dat does not already exist create it here
    if (!particle_group->contains_dat(previous_position_sym)) {
      particle_group->add_particle_dat(
          ParticleDat(sycl_target, ParticleProp(previous_position_sym, ndim),
                      particle_group->domain->mesh->get_cell_count()));
    }

    // copy the current position onto the previous position
    auto pl_iter_range = position_dat->get_particle_loop_iter_range();
    auto pl_stride = position_dat->get_particle_loop_cell_stride();
    auto pl_npart_cell = position_dat->get_particle_loop_npart_cell();
    const auto k_P = position_dat->cell_dat.device_ptr();
    auto k_PP =
        particle_group->get_dat(previous_position_sym)->cell_dat.device_ptr();
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                for (int dimx = 0; dimx < ndim; dimx++) {
                  k_PP[cellx][dimx][layerx] = k_PP[cellx][dimx][layerx];
                }
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
  }
};

} // namespace NESO::CompositeInteraction

#endif
