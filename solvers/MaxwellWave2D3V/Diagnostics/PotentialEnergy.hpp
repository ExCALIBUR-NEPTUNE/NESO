#ifndef __POTENTIAL_ENERGY_H_
#define __POTENTIAL_ENERGY_H_

#include <memory>
#include <mpi.h>

#include <nektar_interface/function_evaluation.hpp>
#include <neso_particles.hpp>
using namespace NESO;
using namespace NESO::Particles;

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
using namespace Nektar;

#include "FieldMean.hpp"

/**
 *  Class to compute and write to a HDF5 file the integral of a function
 *  squared.
 */
template <typename T> class PotentialEnergy {
private:
  BufferDeviceHost<double> dh_energy;
  std::shared_ptr<FieldEvaluate<T>> field_evaluate_phi;
  std::shared_ptr<FieldEvaluate<T>> field_evaluate_ax;
  std::shared_ptr<FieldEvaluate<T>> field_evaluate_ay;
  std::shared_ptr<FieldEvaluate<T>> field_evaluate_az;
  //  std::shared_ptr<FieldMean<T>> phi_field_mean;

public:
  /// The Nektar++ field of interest.
  std::shared_ptr<T> field;
  /// In use ParticleGroup
  ParticleGroupSharedPtr particle_group;
  /// The MPI communicator used by this instance.
  MPI_Comm comm;
  /// The last field energy that was computed on call to write.
  double energy;
  /*
   *  Create new instance.
   *
   *  @param phi_field Nektar++ phi field (DisContField, ContField) to use.
   *  @param ax_field Nektar++ ax field (DisContField, ContField) to use.
   *  @param ay_field Nektar++ ay field (DisContField, ContField) to use.
   *  @param az_field Nektar++ az field (DisContField, ContField) to use.
   *  @param particle_group ParticleGroup to use.
   *  @param cell_id_translation CellIDTranslation to use.
   *  @param filename Filename of HDF5 output file.
   *  @param comm MPI communicator (default MPI_COMM_WORLD).
   */
  PotentialEnergy(std::shared_ptr<T> phi_field, std::shared_ptr<T> ax_field,
                  std::shared_ptr<T> ay_field, std::shared_ptr<T> az_field,
                  ParticleGroupSharedPtr particle_group,
                  std::shared_ptr<CellIDTranslation> cell_id_translation,
                  MPI_Comm comm = MPI_COMM_WORLD)
      : particle_group(particle_group), comm(comm),
        dh_energy(particle_group->sycl_target, 1) {

    int flag;
    MPICHK(MPI_Initialized(&flag));
    ASSERTL1(flag, "MPI is not initialised");

    // this->particle_group->add_particle_dat(
    //     ParticleDat(this->particle_group->sycl_target,
    //                 ParticleProp(Sym<REAL>("phi"), 1),
    //                 this->particle_group->domain->mesh->get_cell_count()));
    // for (int i = 0; i < 3; i++) {
    //   this->particle_group->add_particle_dat(
    //       ParticleDat(this->particle_group->sycl_target,
    //                   ParticleProp(Sym<REAL>("A"), i),
    //                   this->particle_group->domain->mesh->get_cell_count()));
    // }

    this->field_evaluate_phi = std::make_shared<FieldEvaluate<T>>(
        phi_field, this->particle_group, cell_id_translation, false);
    this->field_evaluate_ax = std::make_shared<FieldEvaluate<T>>(
        ax_field, this->particle_group, cell_id_translation, false);
    this->field_evaluate_ay = std::make_shared<FieldEvaluate<T>>(
        ay_field, this->particle_group, cell_id_translation, false);
    this->field_evaluate_az = std::make_shared<FieldEvaluate<T>>(
        az_field, this->particle_group, cell_id_translation, false);

    //    this->phi_field_mean = std::make_shared<FieldMean<T>>(phi_field);
  }

  /**
   *  Compute the current energy of the field.
   */
  inline double compute() {

    this->field_evaluate_phi->evaluate(Sym<REAL>("phi"));
    this->field_evaluate_ax->evaluate(Sym<REAL>("A"), 0);
    this->field_evaluate_ay->evaluate(Sym<REAL>("A"), 1);
    this->field_evaluate_az->evaluate(Sym<REAL>("A"), 2);

    auto t0 = profile_timestamp();
    auto sycl_target = this->particle_group->sycl_target;
    const auto k_WQ =
        (*this->particle_group)[Sym<REAL>("WQ")]->cell_dat.device_ptr();
    const auto k_V =
        (*this->particle_group)[Sym<REAL>("V")]->cell_dat.device_ptr();
    const auto k_phi =
        (*this->particle_group)[Sym<REAL>("phi")]->cell_dat.device_ptr();
    const auto k_A =
        (*this->particle_group)[Sym<REAL>("A")]->cell_dat.device_ptr();

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    this->dh_energy.h_buffer.ptr[0] = 0.0;
    this->dh_energy.host_to_device();

    auto k_energy = this->dh_energy.d_buffer.ptr;
    //    const double k_potential_shift = -this->phi_field_mean->get_mean();

    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          sycl::stream out(1024, 256, cgh);
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                const double phi = k_phi[cellx][0][layerx];
                const double Ax = k_A[cellx][0][layerx];
                const double Ay = k_A[cellx][1][layerx];
                const double Az = k_A[cellx][2][layerx];
                const double Vx = k_V[cellx][0][layerx];
                const double Vy = k_V[cellx][1][layerx];
                const double Vz = k_V[cellx][2][layerx];
                const double wq = k_WQ[cellx][0][layerx];
                const double vdotA = Vx * Ax + Vy * Ay + Vz * Az;
                const double u = wq * (phi - vdotA); //+ k_potential_shift);
                // out << "pe_i=" << u << ", phi=" << phi << ", Ax=" << Ax << ",
                // Ay=" << Ay << ", Az=" << Az << ", Vx=" << Vx << ", Vy=" << Vy
                // << ", Vz=" << Vz << cl::sycl::endl;
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    energy_atomic(k_energy[0]);
                energy_atomic.fetch_add(u);

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target->profile_map.inc("PotentialEnergy", "Execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));
    this->dh_energy.device_to_host();
    const double kernel_energy = this->dh_energy.h_buffer.ptr[0];
    this->energy = 0.0;

    MPICHK(MPI_Allreduce(&kernel_energy, &(this->energy), 1, MPI_DOUBLE,
                         MPI_SUM, this->comm));

    return this->energy;
  }
};

#endif
