#ifndef __EVALUATE_PROJECT_SYSTEM_H__
#define __EVALUATE_PROJECT_SYSTEM_H__
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <cstdio>
#include <iostream>
#include <map>
#include <mpi.h>
#include <string>

#include "nektar_interface/function_evaluation.hpp"
#include <nektar_interface/equation_system_wrapper.hpp>
#include <nektar_interface/geometry_transport/halo_extension.hpp>
#include <nektar_interface/particle_interface.hpp>
using namespace NESO;

#include <neso_particles.hpp>
using namespace NESO::Particles;

#include <SpatialDomains/MeshGraph.h>
using namespace Nektar;

namespace NESO::Benchmark {

class EvaluateProjectSystem {
protected:
public:
  SessionReaderSharedPtr session;
  MeshGraphSharedPtr graph;
  ParticleMeshInterfaceSharedPtr mesh;
  SYCLTargetSharedPtr sycl_target;
  std::shared_ptr<NektarGraphLocalMapper> nektar_graph_local_mapper;
  DomainSharedPtr domain;
  std::shared_ptr<EquationSystemWrapper> eq_sys_wrapper;
  int ndim;
  int size;
  int rank;
  MPI_Comm comm;
  DisContFieldSharedPtr field;
  ParticleGroupSharedPtr particle_group;
  int num_particles_total;
  std::shared_ptr<NektarCartesianPeriodic> pbc;
  std::shared_ptr<CellIDTranslation> cell_id_translation;

  EvaluateProjectSystem(SessionReaderSharedPtr session,
                        MeshGraphSharedPtr graph)
      : session(session), graph(graph) {

    mesh = std::make_shared<ParticleMeshInterface>(graph);
    extend_halos_fixed_offset(1, mesh);
    ndim = mesh->get_ndim();
    comm = mesh->get_comm();
    MPICHK(MPI_Comm_rank(comm, &rank));
    MPICHK(MPI_Comm_size(comm, &size));
    sycl_target = std::make_shared<SYCLTarget>(0, mesh->get_comm());
    if (rank == 0) {
      sycl_target->print_device_info();
    }

    nektar_graph_local_mapper =
        std::make_shared<NektarGraphLocalMapper>(sycl_target, mesh);
    domain = std::make_shared<Domain>(mesh, nektar_graph_local_mapper);
    eq_sys_wrapper = std::make_shared<EquationSystemWrapper>(session, graph);

    auto fields = eq_sys_wrapper->UpdateFields();
    NESOASSERT(fields.size(), "No field found to evaluate.");
    field = std::dynamic_pointer_cast<DisContField>(fields[0]);

    // Create a function to evaluate
    NESOASSERT(ndim == 2 || ndim == 3, "Unexpected number of dimensions");
    if (ndim == 2) {
      auto lambda_f = [&](const NekDouble x, const NekDouble y) {
        return std::pow((x + 1.0) * (x - 1.0) * (y + 1.0) * (y - 1.0), 4);
      };
      interpolate_onto_nektar_field_2d(lambda_f, field);
    } else {
      auto lambda_f = [&](const NekDouble x, const NekDouble y,
                          const NekDouble z) {
        return std::pow((x + 1.0) * (x - 1.0) * (y + 1.0) * (y - 1.0) *
                            (z + 1.0) * (z - 1.0),
                        4);
      };
      interpolate_onto_nektar_field_3d(lambda_f, field);
    }

    // create particle system
    ParticleSpec particle_spec{ParticleProp(Sym<REAL>("P"), ndim, true),
                               ParticleProp(Sym<INT>("CELL_ID"), 1, true),
                               ParticleProp(Sym<REAL>("V"), ndim),
                               ParticleProp(Sym<REAL>("E"), 1)};
    particle_group =
        std::make_shared<ParticleGroup>(domain, particle_spec, sycl_target);

    // Determine total number of particles
    const int num_particles_total_default = 16000;
    session->LoadParameter("num_particles_total", num_particles_total,
                           num_particles_total_default);
    // Determine how many particles this rank initialises
    int rstart, rend;
    get_decomp_1d(size, num_particles_total, rank, &rstart, &rend);
    const int N = rend - rstart;
    int num_particles_check = -1;
    MPICHK(MPI_Allreduce(&N, &num_particles_check, 1, MPI_INT, MPI_SUM,
                         MPI_COMM_WORLD));
    NESOASSERT(num_particles_check == num_particles_total,
               "Error creating particles");

    // Initialise particles on this rank
    int seed;
    const int seed_default = 34672835;
    session->LoadParameter("seed", seed, seed_default);
    std::mt19937 rng_pos(seed + rank);

    pbc = std::make_shared<NektarCartesianPeriodic>(
        sycl_target, graph, particle_group->position_dat);
    std::vector<std::uniform_real_distribution<double>> velocity_distributions;
    for (int dimx = 0; dimx < ndim; dimx++) {
      velocity_distributions.push_back(std::uniform_real_distribution<double>(
          -1.0 * pbc->global_extent[dimx], pbc->global_extent[dimx]));
    }

    const int cell_count = mesh->get_cell_count();
    // uniformly distributed positions
    if (N > 0) {
      auto positions =
          uniform_within_extents(N, ndim, pbc->global_extent, rng_pos);

      ParticleSet initial_distribution(N, particle_group->get_particle_spec());
      for (int px = 0; px < N; px++) {
        for (int dimx = 0; dimx < ndim; dimx++) {
          const double pos_orig =
              positions[dimx][px] + pbc->global_origin[dimx];
          initial_distribution[Sym<REAL>("P")][px][dimx] = pos_orig;
          initial_distribution[Sym<REAL>("V")][px][dimx] =
              velocity_distributions[dimx](rng_pos);
        }
        initial_distribution[Sym<INT>("CELL_ID")][px][0] = px % cell_count;
      }
      particle_group->add_particles_local(initial_distribution);
    }

    cell_id_translation = std::make_shared<CellIDTranslation>(
        sycl_target, particle_group->cell_id_dat, mesh);
    transfer_particles();
  }

  inline void transfer_particles() {
    // send the particles to the correct rank
    // parallel_advection_initialisation(A, 32);
    pbc->execute();
    particle_group->hybrid_move();
    cell_id_translation->execute();
    particle_group->cell_move();
  }

  inline void advect() {
    const REAL dt = 0.5;
    const auto k_V =
        particle_group->get_dat(Sym<REAL>("V"))->cell_dat.device_ptr();
    const auto k_P =
        particle_group->get_dat(Sym<REAL>("P"))->cell_dat.device_ptr();
    const auto pl_iter_range =
        particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        particle_group->mpi_rank_dat->get_particle_loop_npart_cell();
    const auto k_ndim = ndim;
    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;
                for (int dimx = 0; dimx < k_ndim; dimx++) {
                  k_P[cellx][0][layerx] += dt * k_V[cellx][0][layerx];
                }
                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    transfer_particles();
  };

  inline void free() {
    particle_group->free();
    sycl_target->free();
    mesh->free();
  }
};

} // namespace NESO::Benchmark

#endif
