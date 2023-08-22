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
  std::shared_ptr<T> momentum_x_ions;
  std::shared_ptr<T> momentum_y_ions;
    
  SYCLTargetSharedPtr sycl_target;
  BufferDeviceHost<double> dh_particle_total_weight;
  bool initial_x_momentum_computed;
  bool initial_y_momentum_computed;
  double initial_x_momentum_fluid;
  double initial_y_momentum_fluid;
  int momentum_recording_step;
  int rank;
  ofstream fh;

public:
  MomentumRecording(const LibUtilities::SessionReaderSharedPtr session,
                std::shared_ptr<NeutralParticleSystem> particle_sys,
                std::shared_ptr<T> momentum_x_ions,std::shared_ptr<T> momentum_y_ions)
      : session(session), particle_sys(particle_sys), momentum_x_ions(momentum_x_ions), momentum_y_ions(momentum_y_ions), 
        sycl_target(particle_sys->sycl_target),
        dh_particle_total_weight(sycl_target, 1), initial_x_momentum_computed(false), initial_y_momentum_computed(false){

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


  inline double compute_fluid_x_momentum() {
    return this->momentum_x_ions->Integral(this->momentum_x_ions->GetPhys()) *
           this->particle_sys->n_to_SI;
  }
  
  inline double compute_fluid_y_momentum() {
    return this->momentum_y_ions->Integral(this->momentum_y_ions->GetPhys()) *
           this->particle_sys->n_to_SI;
  }

  inline void compute_initial_fluid_x_momentum() {
    if (!this->initial_x_momentum_computed) {
      this->initial_x_momentum_fluid = this->compute_fluid_x_momentum();
      this->initial_x_momentum_computed = true;
    }
  }
  
  inline void compute_initial_fluid_y_momentum() {
    if (!this->initial_y_momentum_computed) {
      this->initial_y_momentum_fluid = this->compute_fluid_y_momentum();
      this->initial_y_momentum_computed = true;
    }
  }

  inline double get_initial_x_momentum() {
    NESOASSERT(this->initial_x_momentum_computed == true,
               "initial x momentum not computed");
    return this->initial_x_momentum_fluid;
  }
  
  inline double get_initial_y_momentum() {
    NESOASSERT(this->initial_y_momentum_computed == true,
               "initial y momentum not computed");
    return this->initial_y_momentum_fluid;
  }

  inline void compute(int step) {
    if (momentum_recording_step > 0) {
      if (step % momentum_recording_step == 0) {
        const double momentum_particles = 0.0;
        const double x_momentum_fluid = this->compute_fluid_x_momentum();
        const double y_momentum_fluid = this->compute_fluid_y_momentum();
        const double momentum_total_x = momentum_particles + x_momentum_fluid;
        const double momentum_total_y = momentum_particles + y_momentum_fluid;
       // const double momentum_added = this->compute_total_added_momentum();
       // const double correct_total = momentum_added + this->initial_momentum_fluid;

        // Write values to file
  //      if (rank == 0) {
  //        nprint(step, ",",
  //               abs(correct_total - momentum_total) / abs(correct_total), ",",
  //               momentum_particles, ",", momentum_fluid, ",");
  //        fh << step << ","
  //           << abs(correct_total - momentum_total) / abs(correct_total) << ","
  //           << momentum_particles << "," << momentum_fluid << "\n";
  //      }
      }
    }
  };
};

#endif
