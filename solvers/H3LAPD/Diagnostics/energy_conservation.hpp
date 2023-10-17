#ifndef __H3LAPD_ENERGY_CONSERVATION_H_
#define __H3LAPD_ENERGY_CONSERVATION_H_

#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>
using namespace NESO;
using namespace NESO::Particles;

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
using namespace Nektar;

#include "../ParticleSystems/neutral_particles.hpp"

#include <fstream>
#include <iostream>

template <typename T> class EnergyRecording {
protected:
  const LibUtilities::SessionReaderSharedPtr session;
  std::shared_ptr<NeutralParticleSystem> particle_sys;
  std::shared_ptr<T> E;

  SYCLTargetSharedPtr sycl_target;
  BufferDeviceHost<double> dh_particle_total_weight;
  bool initial_energy_computed;
  double initial_energy_fluid;
  int energy_recording_step;
  int rank;
  ofstream fh;

public:
  EnergyRecording(const LibUtilities::SessionReaderSharedPtr session,
                  std::shared_ptr<NeutralParticleSystem> particle_sys,
                  std::shared_ptr<T> E)
      : session(session), particle_sys(particle_sys), E(E),
        sycl_target(particle_sys->sycl_target),
        dh_particle_total_weight(sycl_target, 1),
        initial_energy_computed(false) {

    session->LoadParameter("energy_recording_step", energy_recording_step, 0);
    rank = sycl_target->comm_pair.rank_parent;
    if ((rank == 0) && (energy_recording_step > 0)) {
      fh.open("energy_recording.csv");
      fh << "step,relative_error,energy_particles,energy_fluid\n";
    }
  };

  ~EnergyRecording() {
    if ((rank == 0) && (energy_recording_step > 0)) {
      fh.close();
    }
  }

  inline double compute_particle_energy() {
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

  inline double compute_fluid_energy() {
    return this->E->Integral(this->E->GetPhys());
  }

  inline double compute_total_added_energy() {
    const uint64_t num_particles_added =
        this->particle_sys->total_num_particles_added;
    uint64_t global_num_added;
    MPICHK(MPI_Allreduce(&num_particles_added, &global_num_added, 1,
                         MPI_UINT64_T, MPI_SUM,
                         sycl_target->comm_pair.comm_parent));

    double added_energy =
        ((double)global_num_added) * this->particle_sys->particle_init_weight;

    return added_energy;
  }

  inline void compute_initial_fluid_energy() {
    if (!this->initial_energy_computed) {
      this->initial_energy_fluid = this->compute_fluid_energy();
      this->initial_energy_computed = true;
    }
  }

  inline double get_initial_energy() {
    NESOASSERT(this->initial_energy_computed == true,
               "initial energy not computed");
    return this->initial_energy_fluid;
  }

  inline void compute(int step) {
    if (energy_recording_step > 0) {
      if (step % energy_recording_step == 0) {
        const double energy_particles = this->compute_particle_energy();
        const double energy_fluid = this->compute_fluid_energy();
        const double energy_total = energy_particles + energy_fluid;
        const double energy_added = this->compute_total_added_energy();
        const double correct_total = energy_added + this->initial_energy_fluid;

        // Write values to file
        if (rank == 0) {
          nprint(step, ",",
                 abs(correct_total - energy_total) / abs(correct_total), ",",
                 energy_particles, ",", energy_fluid, ",");
          fh << step << ","
             << abs(correct_total - energy_total) / abs(correct_total) << ","
             << energy_particles << "," << energy_fluid << "\n";
        }
      }
    }
  };
};

#endif
