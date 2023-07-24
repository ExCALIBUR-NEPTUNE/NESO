#ifndef __E2D3V_PARALLEL_INITIALISATION_H_
#define __E2D3V_PARALLEL_INITIALISATION_H_

#include <memory>
#include <nektar_interface/particle_interface.hpp>
#include <neso_particles.hpp>

namespace NESO {

inline void get_point_in_local_domain(ParticleGroupSharedPtr particle_group,
                                      double *point) {

  auto domain = particle_group->domain;
  const int space_ncomp = particle_group->position_dat->ncomp;
  auto mesh = std::dynamic_pointer_cast<ParticleMeshInterface>(domain->mesh);
  auto graph = mesh->graph;

  NESOASSERT(space_ncomp == 2, "Expected 2 position components");

  auto trigeoms = graph->GetAllTriGeoms();
  if (trigeoms.size() > 0) {

    auto tri = trigeoms.begin()->second;
    auto v0 = tri->GetVertex(0);
    auto v1 = tri->GetVertex(1);
    auto v2 = tri->GetVertex(2);

    double mid[2];
    mid[0] = 0.5 * ((*v1)[0] - (*v0)[0]);
    mid[1] = 0.5 * ((*v1)[1] - (*v0)[1]);
    mid[0] += (*v0)[0];
    mid[1] += (*v0)[1];

    point[0] = 0.5 * ((*v2)[0] - mid[0]);
    point[1] = 0.5 * ((*v2)[1] - mid[1]);
    point[0] += mid[0];
    point[1] += mid[1];

    Array<OneD, NekDouble> test(3);
    test[0] = point[0];
    test[1] = point[1];
    test[2] = 0.0;
    NESOASSERT(tri->ContainsPoint(test), "Triangle should contain this point");

  } else {
    auto quadgeoms = graph->GetAllQuadGeoms();
    NESOASSERT(quadgeoms.size() > 0, "could not find any 2D geometry objects");

    auto quad = quadgeoms.begin()->second;
    auto v0 = quad->GetVertex(0);
    auto v2 = quad->GetVertex(2);

    Array<OneD, NekDouble> mid(3);
    mid[0] = 0.5 * ((*v2)[0] - (*v0)[0]);
    mid[1] = 0.5 * ((*v2)[1] - (*v0)[1]);
    mid[2] = 0.0;
    mid[0] += (*v0)[0];
    mid[1] += (*v0)[1];

    NESOASSERT(quad->ContainsPoint(mid), "Quad should contain this point");

    point[0] = mid[0];
    point[1] = mid[1];
  }
}

inline void parallel_advection_store(ParticleGroupSharedPtr particle_group) {

  auto sycl_target = particle_group->sycl_target;
  const int space_ncomp = particle_group->position_dat->ncomp;

  double local_point[3];
  get_point_in_local_domain(particle_group, local_point);

  auto k_P = particle_group->position_dat->cell_dat.device_ptr();
  auto k_ORIG_POS =
      (*particle_group)[Sym<REAL>("ORIG_POS")]->cell_dat.device_ptr();
  auto pl_iter_range =
      particle_group->mpi_rank_dat->get_particle_loop_iter_range();
  auto pl_stride =
      particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
  auto pl_npart_cell =
      particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

  sycl::buffer<double, 1> b_local_point(local_point, 3);

  // store the target position
  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        auto a_local_point =
            b_local_point.get_access<sycl::access::mode::read>(cgh);
        cgh.parallel_for<>(sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
          NESO_PARTICLES_KERNEL_START
          const INT cellx = NESO_PARTICLES_KERNEL_CELL;
          const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
          for (int nx = 0; nx < space_ncomp; nx++) {
            k_ORIG_POS[cellx][nx][layerx] = k_P[cellx][nx][layerx];
            k_P[cellx][nx][layerx] = a_local_point[nx];
          }

          NESO_PARTICLES_KERNEL_END
        });
      })
      .wait_and_throw();
}

inline void parallel_advection_restore(ParticleGroupSharedPtr particle_group) {

  auto sycl_target = particle_group->sycl_target;
  const int space_ncomp = particle_group->position_dat->ncomp;
  auto k_P = particle_group->position_dat->cell_dat.device_ptr();
  auto k_ORIG_POS =
      (*particle_group)[Sym<REAL>("ORIG_POS")]->cell_dat.device_ptr();
  auto pl_iter_range =
      particle_group->mpi_rank_dat->get_particle_loop_iter_range();
  auto pl_stride =
      particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
  auto pl_npart_cell =
      particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

  ErrorPropagate ep(sycl_target);
  auto k_ep = ep.device_ptr();

  // restore the target position
  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
          NESO_PARTICLES_KERNEL_START
          const INT cellx = NESO_PARTICLES_KERNEL_CELL;
          const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

          for (int nx = 0; nx < space_ncomp; nx++) {

            const bool valid = ABS(k_ORIG_POS[cellx][nx][layerx] -
                                   k_P[cellx][nx][layerx]) < 1.0e-6;

            NESO_KERNEL_ASSERT(valid, k_ep);
            k_P[cellx][nx][layerx] = k_ORIG_POS[cellx][nx][layerx];
          }

          NESO_PARTICLES_KERNEL_END
        });
      })
      .wait_and_throw();

  ep.check_and_throw("Advected particle was very far from intended position.");
}

inline void parallel_advection_step(ParticleGroupSharedPtr particle_group,
                                    const int num_steps, const int step) {
  auto sycl_target = particle_group->sycl_target;
  const int space_ncomp = particle_group->position_dat->ncomp;
  auto k_P = particle_group->position_dat->cell_dat.device_ptr();
  auto k_ORIG_POS =
      (*particle_group)[Sym<REAL>("ORIG_POS")]->cell_dat.device_ptr();
  auto pl_iter_range =
      particle_group->mpi_rank_dat->get_particle_loop_iter_range();
  auto pl_stride =
      particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
  auto pl_npart_cell =
      particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

  const double steps_left = ((double)num_steps) - ((double)step);
  const double inverse_steps_left = 1.0 / steps_left;

  sycl_target->queue
      .submit([&](sycl::handler &cgh) {
        cgh.parallel_for<>(sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
          NESO_PARTICLES_KERNEL_START
          const INT cellx = NESO_PARTICLES_KERNEL_CELL;
          const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

          for (int nx = 0; nx < space_ncomp; nx++) {
            const double offset =
                k_ORIG_POS[cellx][nx][layerx] - k_P[cellx][nx][layerx];
            k_P[cellx][nx][layerx] += inverse_steps_left * offset;
          }

          NESO_PARTICLES_KERNEL_END
        });
      })
      .wait_and_throw();
}

inline void
parallel_advection_initialisation(ParticleGroupSharedPtr particle_group) {

  const int space_ncomp = particle_group->position_dat->ncomp;
  auto domain = particle_group->domain;
  auto sycl_target = particle_group->sycl_target;
  particle_group->add_particle_dat(
      ParticleDat(sycl_target, ParticleProp(Sym<REAL>("ORIG_POS"), space_ncomp),
                  domain->mesh->get_cell_count()));
}

} // namespace NESO

#endif
