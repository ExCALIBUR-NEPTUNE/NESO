#include "compute_target.hpp"
#include "particle_group.hpp"
#include "profiling.hpp"
#include <neso_particles.hpp>
#include <NEC_abstract_reaction/reaction_kernel.hpp>

using namespace NESO::Particles;

template <typename kernelType, typename dataType>
class ReactionController {
    public:
        ReactionController(
            const ParticleGroupSharedPtr& particle_group_,
            const kernelType& reactionKernel_,
            const dataType& reactionData_,
            const SYCLTargetSharedPtr& sycl_target_
        ):
            particle_group(particle_group_),
            reactionKernel(reactionKernel_),
            reactionData(reactionData_),
            sycl_target(sycl_target_)
        {};

        ParticleGroupSharedPtr particle_group;
        kernelType reactionKernel;
        dataType reactionData;
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
        
        auto reactionDataMem = sycl::malloc_device<ioniseData>(sizeof(ioniseData), sycl_target->queue);

        sycl_target->queue.submit([&](sycl::handler &cgh) {
            cgh.memcpy(reactionDataMem, &reactionData, sizeof(ioniseData));
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

                    reactionDataMem->select_params(cellx, layerx);

                    reactionDataMem->set_invratio();

                    REAL rate = reactionKernel.calc_rate(reactionDataMem);

                    REAL weight_fraction = -rate * reactionDataMem->k_dt_SI *
                                        reactionDataMem->real_fields[1];

                    REAL deltaweight =
                        weight_fraction * reactionDataMem->particle_properties[0];

                    if ((reactionDataMem->particle_properties[0] + deltaweight) <=
                        0) {
                    reactionDataMem->int_fields[0] = -1;
                    deltaweight = -reactionDataMem->particle_properties[0];
                    }

                    reactionKernel.apply_kernel();

                    reactionKernel.feedback_kernel(reactionDataMem, weight_fraction);

                    reactionDataMem->particle_properties[0] += deltaweight;

                    reactionDataMem->update_params(cellx, layerx);

                    NESO_PARTICLES_KERNEL_END
                });
        }).wait_and_throw();

        sycl_target->queue.submit([&](sycl::handler &cgh) {
        cgh.memcpy(&reactionData, reactionDataMem, sizeof(ioniseData));
        }).wait_and_throw();

        sycl::free(reactionDataMem, sycl_target->queue);

        sycl_target->profile_map.inc("NeutralParticleSystem", "Ionisation_Execute",
                                    1, profile_elapsed(t0, profile_timestamp()));        

    }
};