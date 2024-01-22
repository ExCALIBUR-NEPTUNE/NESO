#ifndef __SIMPLESOL_MOMENTUM_CONSERVATION_H_
#define __SIMPLESOL_MOMENTUM_CONSERVATION_H_

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

template <typename T> class MomentumRecording {
protected:
  const LibUtilities::SessionReaderSharedPtr session;
  std::shared_ptr<NeutralParticleSystem> particle_sys;
  std::shared_ptr<T> momentum_0_ions;
  std::shared_ptr<T> momentum_1_ions;
    
  SYCLTargetSharedPtr sycl_target;
  BufferDeviceHost<double> dh_particle_total_momentum_0;
  BufferDeviceHost<double> dh_particle_total_momentum_1;
  bool initial_fluid_0_momentum_computed;
  bool initial_fluid_1_momentum_computed;
  bool initial_particle_0_momentum_computed;
  bool initial_particle_1_momentum_computed;
  
  double initial_0_momentum_fluid;
  double initial_1_momentum_fluid;
  
  double initial_0_momentum_particles;
  double initial_1_momentum_particles;
  int momentum_recording_step;
  int rank;
  ofstream fh;

public:
  MomentumRecording(const LibUtilities::SessionReaderSharedPtr session,
                std::shared_ptr<NeutralParticleSystem> particle_sys,
                std::shared_ptr<T> momentum_0_ions,std::shared_ptr<T> momentum_1_ions)
      : session(session), particle_sys(particle_sys), momentum_0_ions(momentum_0_ions), momentum_1_ions(momentum_1_ions), 
        sycl_target(particle_sys->sycl_target),
        dh_particle_total_momentum_0(sycl_target, 1), dh_particle_total_momentum_1(sycl_target, 1), initial_fluid_0_momentum_computed(false), initial_fluid_1_momentum_computed(false),
        initial_particle_0_momentum_computed(false), initial_particle_1_momentum_computed(false){

    session->LoadParameter("momentum_recording_step", momentum_recording_step, 0);
    rank = sycl_target->comm_pair.rank_parent;
    if ((rank == 0) && (momentum_recording_step > 0)) {
      fh.open("momentum_recording.csv");
      fh << "step,relative_error,momentum_particles,momentum_fluid\n";
    }
  };

  ~MomentumRecording() {
    if ((rank == 0) && (momentum_recording_step > 0)) {
      fh.close();
    }
  }
    
    inline double compute_particle_momentum_0() {
    auto particle_group = this->particle_sys->particle_group;
    auto k_ND = (*particle_group)[Sym<REAL>("NEUTRAL_DENSITY")]
                  ->cell_dat.device_ptr();
    auto k_V = (*particle_group)[Sym<REAL>("VELOCITY")]
                        ->cell_dat.device_ptr();
                    
    this->dh_particle_total_momentum_0.h_buffer.ptr[0] = 0.0;
    this->dh_particle_total_momentum_0.host_to_device();
    auto k_particle_momentum_0 = this->dh_particle_total_momentum_0.d_buffer.ptr;

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

                const double contrib = k_V[cellx][0][layerx]*k_ND[cellx][0][layerx]/(this->particle_sys->n_to_SI);
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    energy_atomic(k_particle_momentum_0[0]);
                energy_atomic.fetch_add(contrib);

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();

    this->dh_particle_total_momentum_0.device_to_host();
    const double tmp_particle_momentum_0 = this->dh_particle_total_momentum_0.h_buffer.ptr[0];
    double total_particle_momentum_0;
    MPICHK(MPI_Allreduce(&tmp_particle_momentum_0, &total_particle_momentum_0, 1, MPI_DOUBLE,
                         MPI_SUM, sycl_target->comm_pair.comm_parent));

    return total_particle_momentum_0;
  }
  
   inline double compute_total_transferred_momentum_0() {
    const double particle_momentum_transferred_0 =
        this->particle_sys->total_particle_momentum_transferred[0];
    double global_particle_momentum_transferred_0;
    MPICHK(MPI_Allreduce(&particle_momentum_transferred_0, &global_particle_momentum_transferred_0, 1,
                         MPI_DOUBLE, MPI_SUM,
                         sycl_target->comm_pair.comm_parent));


    return global_particle_momentum_transferred_0;
  } 
  
   inline double compute_total_added_momentum_0() {
    const double particle_momentum_added_0 =
        this->particle_sys->total_particle_momentum_added[0];
    double global_particle_momentum_added_0;
    MPICHK(MPI_Allreduce(&particle_momentum_added_0, &global_particle_momentum_added_0, 1,
                         MPI_DOUBLE, MPI_SUM,
                         sycl_target->comm_pair.comm_parent));


    return global_particle_momentum_added_0;
  } 
  
  inline double compute_fluid_0_momentum() {
    return this->momentum_0_ions->Integral(this->momentum_0_ions->GetPhys()) ;
  }

  inline void compute_initial_fluid_0_momentum() {
    if (!this->initial_fluid_0_momentum_computed) {
      this->initial_0_momentum_fluid = this->compute_fluid_0_momentum();
      this->initial_fluid_0_momentum_computed = true;
    }
  }
  
  inline void compute_initial_particle_0_momentum() {
    if (!this->initial_particle_0_momentum_computed) {
      this->initial_0_momentum_particles = this->compute_particle_momentum_0();
      this->initial_particle_0_momentum_computed = true;
    }
  }   

  inline double get_initial_fluid_0_momentum() {
    NESOASSERT(this->initial_fluid_0_momentum_computed == true,
               "initial x momentum not computed");
    return this->initial_0_momentum_fluid;
  }
  
  inline double get_initial_particle_0_momentum() {
    NESOASSERT(this->initial_particle_0_momentum_computed == true,
               "initial x momentum not computed");
    return this->initial_0_momentum_particles;
  }

  inline void compute(int step) {
    if (momentum_recording_step > 0) {
      if (step % momentum_recording_step == 0) {
        const double momentum_particles_0 = this->compute_particle_momentum_0();
        const double momentum_added_0 = this->compute_total_added_momentum_0();        
        const double momentum_fluid_0 = this->compute_fluid_0_momentum();
        const double momentum_transferred_0 = this->compute_total_transferred_momentum_0();
        const double momentum_total_0 = momentum_particles_0 + momentum_fluid_0 +momentum_transferred_0;
        const double correct_total_0 = this->initial_0_momentum_fluid + this->initial_0_momentum_particles;

        // Write values to file
        if (rank == 0) {
        //  nprint(step, ",",
        //         abs(correct_total_0 - momentum_total_0) / abs(correct_total_0), ",",
        //         momentum_particles_0, ",", momentum_fluid_0, ",");
        //  fh << step << ","
        //     << abs(correct_total_0 - momentum_total_0) / abs(correct_total_0) << ","
         //    << momentum_particles_0 << "," << momentum_fluid_0 << "\n";

          nprint(step, ", ",
                 this->initial_0_momentum_fluid + this->initial_0_momentum_particles + momentum_added_0
                 , ", ", 
                 momentum_particles_0 + momentum_fluid_0, ",", momentum_particles_0, ",",momentum_fluid_0 , "," , momentum_added_0);
          fh << step << ","
             << this->initial_0_momentum_fluid + this->initial_0_momentum_particles + momentum_added_0 << ","
             << momentum_particles_0 + momentum_fluid_0 << "," << momentum_particles_0 << "," << momentum_fluid_0 << "," << momentum_added_0 << "\n";
        }
      }
    }
  };
};

#endif
