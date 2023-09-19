#include "compute_target.hpp"
#include "particle_group.hpp"
#include "profiling.hpp"
#include <neso_particles.hpp>
#include <NEC_abstract_reaction/reaction_kernel.hpp>

using namespace NESO::Particles;

template <typename kernelType>
class ReactionController {
    public:
        ReactionController(
            const ParticleGroupSharedPtr& particle_group_,
            const kernelType& reactionKernel_,
            const SYCLTargetSharedPtr& sycl_target_
        ):
            particle_group(particle_group_),
            reactionKernel(reactionKernel_),
            sycl_target(sycl_target_)
        {};

        ParticleGroupSharedPtr particle_group;
        kernelType reactionKernel;
        SYCLTargetSharedPtr sycl_target;

    void apply() {
        auto t0 = profile_timestamp();

        const auto pl_iter_range =
            particle_group->mpi_rank_dat->get_particle_loop_iter_range();
        const auto pl_stride =
            particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
        const auto pl_npart_cell =
            particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

        sycl_target->profile_map.inc("NeutralParticleSystem", "Ionisation_Prepare",
                                    1, profile_elapsed(t0, profile_timestamp()));
        
        auto reactionMem = sycl::malloc_device<ionise_reaction>(sizeof(ionise_reaction), sycl_target->queue);

        sycl_target->queue.submit([&](sycl::handler &cgh) {
            cgh.memcpy(reactionMem, &reactionKernel, sizeof(ionise_reaction));
        }).wait_and_throw();

        sycl_target->queue
            .submit([&](sycl::handler &cgh) {
            cgh.parallel_for<>(
                sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                    NESO_PARTICLES_KERNEL_START
                    const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                    const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                    // get the temperatue in eV. TODO: ensure not unit conversion is
                    // required

                    reactionMem->select_params(cellx, layerx);

                    reactionMem->set_invratio();

                    REAL rate = reactionKernel.calc_rate(reactionMem);

                    REAL weight_fraction = -rate * reactionMem->k_dt_SI *
                                        reactionMem->real_fields[1];

                    REAL deltaweight =
                        weight_fraction * reactionMem->particle_properties[0];

                    if ((reactionMem->particle_properties[0] + deltaweight) <=
                        0) {
                    reactionMem->int_fields[0] = -1;
                    deltaweight = -reactionMem->particle_properties[0];
                    }

                    reactionKernel.apply_kernel(weight_fraction);

                    reactionKernel.feedback_kernel(reactionMem, weight_fraction);

                    reactionMem->particle_properties[0] += deltaweight;

                    reactionMem->update_params(cellx, layerx);

                    NESO_PARTICLES_KERNEL_END
                });
        }).wait_and_throw();

        sycl_target->queue.submit([&](sycl::handler &cgh) {
        cgh.memcpy(&reactionKernel, reactionMem, sizeof(ioniseData));
        }).wait_and_throw();

        sycl::free(reactionMem, sycl_target->queue);

        sycl_target->profile_map.inc("NeutralParticleSystem", "Ionisation_Execute",
                                    1, profile_elapsed(t0, profile_timestamp()));        

    }
};