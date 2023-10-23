#ifndef H3LAPD_MASS_RECORDER_H
#define H3LAPD_MASS_RECORDER_H

#include "../ParticleSystems/NeutralParticleSystem.hpp"
#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>

namespace LU = Nektar::LibUtilities;

namespace NESO::Solvers::H3LAPD {

template <typename T> class MassRecorder {
protected:
  const LU::SessionReaderSharedPtr session;
  std::shared_ptr<NeutralParticleSystem> particle_sys;
  std::shared_ptr<T> n;

  SYCLTargetSharedPtr sycl_target;
  BufferDeviceHost<double> dh_particle_total_weight;
  bool initial_mass_computed;
  double initial_mass_fluid;
  int mass_recording_step;
  int rank;
  ofstream fh;

public:
  MassRecorder(const LU::SessionReaderSharedPtr session,
               std::shared_ptr<NeutralParticleSystem> particle_sys,
               std::shared_ptr<T> n)
      : session(session), particle_sys(particle_sys), n(n),
        sycl_target(particle_sys->sycl_target),
        dh_particle_total_weight(sycl_target, 1), initial_mass_computed(false) {

    session->LoadParameter("mass_recording_step", mass_recording_step, 0);
    rank = sycl_target->comm_pair.rank_parent;
    if ((rank == 0) && (mass_recording_step > 0)) {
      fh.open("mass_recording.csv");
      fh << "step,relative_error,mass_particles,mass_fluid\n";
    }
  };

  ~MassRecorder() {
    if ((rank == 0) && (mass_recording_step > 0)) {
      fh.close();
    }
  }

  inline double compute_particle_mass() {
    auto particle_group = this->particle_sys->particle_group;
    auto k_W = (*particle_group)[Sym<REAL>("COMPUTATIONAL_WEIGHT")]
                   ->cell_dat.device_ptr();

    this->dh_particle_total_weight.h_buffer.ptr[0] = 0.0;
    this->dh_particle_total_weight.host_to_device();
    auto k_particle_weight = this->dh_particle_total_weight.d_buffer.ptr;

    const auto pl_iter_range =
        particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                const double contrib = k_W[cellx][0][layerx];

                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    energy_atomic(k_particle_weight[0]);
                energy_atomic.fetch_add(contrib);

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    this->dh_particle_total_weight.device_to_host();
    const double tmp_weight = this->dh_particle_total_weight.h_buffer.ptr[0];
    double total_particle_weight;
    MPICHK(MPI_Allreduce(&tmp_weight, &total_particle_weight, 1, MPI_DOUBLE,
                         MPI_SUM, sycl_target->comm_pair.comm_parent));

    return total_particle_weight;
  }

  inline double compute_fluid_mass() {
    return this->n->Integral(this->n->GetPhys()) * this->particle_sys->n_to_SI;
  }

  inline double compute_total_added_mass() {
    // N.B. in this case, total_num_particles_added already accounts for all MPI
    // ranks - no need for an Allreduce
    double added_mass =
        ((double)this->particle_sys->total_num_particles_added) *
        this->particle_sys->particle_init_weight;
    return added_mass;
  }

  inline void compute_initial_fluid_mass() {
    if (mass_recording_step > 0) {
      if (!this->initial_mass_computed) {
        this->initial_mass_fluid = this->compute_fluid_mass();
        this->initial_mass_computed = true;
      }
    }
  }

  inline double get_initial_mass() {
    NESOASSERT(this->initial_mass_computed == true,
               "initial mass not computed");
    return this->initial_mass_fluid;
  }

  inline void compute(int step) {
    if (mass_recording_step > 0) {
      if (step % mass_recording_step == 0) {
        const double mass_particles = this->compute_particle_mass();
        const double mass_fluid = this->compute_fluid_mass();
        const double mass_total = mass_particles + mass_fluid;
        const double mass_added = this->compute_total_added_mass();
        const double correct_total = mass_added + this->initial_mass_fluid;

        // Write values to file
        if (rank == 0) {
          nprint(step, ",",
                 abs(correct_total - mass_total) / abs(correct_total), ",",
                 mass_particles, ",", mass_fluid, ",");
          fh << step << ","
             << abs(correct_total - mass_total) / abs(correct_total) << ","
             << mass_particles << "," << mass_fluid << "\n";
        }
      }
    }
  };
};
} // namespace NESO::Solvers::H3LAPD

#endif
