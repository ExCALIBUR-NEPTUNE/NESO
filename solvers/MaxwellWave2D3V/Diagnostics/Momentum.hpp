#ifndef __MOMENTUM_H_
#define __MOMENTUM_H_

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
template <typename T> class Momentum {
private:
  BufferDeviceHost<double> dh_momentum_x;
  BufferDeviceHost<double> dh_momentum_y;
  BufferDeviceHost<double> dh_momentum_z;
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
  /// The last field momentum that was computed on call to write.
  double momentum;
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
  Momentum(std::shared_ptr<T> phi_field, std::shared_ptr<T> ax_field,
                  std::shared_ptr<T> ay_field, std::shared_ptr<T> az_field,
                  ParticleGroupSharedPtr particle_group,
                  std::shared_ptr<CellIDTranslation> cell_id_translation,
                  MPI_Comm comm = MPI_COMM_WORLD)
      : particle_group(particle_group), comm(comm),
        dh_momentum_x(particle_group->sycl_target, 1),
        dh_momentum_y(particle_group->sycl_target, 1),
        dh_momentum_z(particle_group->sycl_target, 1) {

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

    this->field_evaluate_ax = std::make_shared<FieldEvaluate<T>>(
        ax_field, this->particle_group, cell_id_translation, false);
    this->field_evaluate_ay = std::make_shared<FieldEvaluate<T>>(
        ay_field, this->particle_group, cell_id_translation, false);
    this->field_evaluate_az = std::make_shared<FieldEvaluate<T>>(
        az_field, this->particle_group, cell_id_translation, false);

    //    this->phi_field_mean = std::make_shared<FieldMean<T>>(phi_field);
  }

  /**
   *  Compute the current momentum of the field.
   */
  inline double compute() {

    this->field_evaluate_ax->evaluate(Sym<REAL>("A"), 0);
    this->field_evaluate_ay->evaluate(Sym<REAL>("A"), 1);
    this->field_evaluate_az->evaluate(Sym<REAL>("A"), 2);

    auto t0 = profile_timestamp();
    auto sycl_target = this->particle_group->sycl_target;
    const auto k_W =
        (*this->particle_group)[Sym<REAL>("W")]->cell_dat.device_ptr();
    const auto k_M =
        (*this->particle_group)[Sym<REAL>("M")]->cell_dat.device_ptr();
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

    this->dh_momentum_x.h_buffer.ptr[0] = 0.0;
    this->dh_momentum_y.h_buffer.ptr[0] = 0.0;
    this->dh_momentum_z.h_buffer.ptr[0] = 0.0;
    this->dh_momentum_x.host_to_device();
    this->dh_momentum_y.host_to_device();
    this->dh_momentum_z.host_to_device();

    auto k_momentum_x = this->dh_momentum_x.d_buffer.ptr;
    auto k_momentum_y = this->dh_momentum_y.d_buffer.ptr;
    auto k_momentum_z = this->dh_momentum_z.d_buffer.ptr;
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
                const double Q = k_Q[cellx][0][layerx];
                const double W = k_W[cellx][0][layerx];
                const double M = k_M[cellx][0][layerx];
                const double Px = W * (M * Vx + Q * Ax);
                const double Py = W * (M * Vy + Q * Ay);
                const double Pz = W * (M * Vz + Q * Az);
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    momentum_atomic_x(k_momentum_x[0]);
                momentum_atomic_x.fetch_add(Px);
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    momentum_atomic_y(k_momentum_y[0]);
                momentum_atomic_y.fetch_add(Py);
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    momentum_atomic_z(k_momentum[2]);
                momentum_atomic_z.fetch_add(Pz);

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target->profile_map.inc("Momentum", "Execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));
    this->dh_momentum_x.device_to_host();
    this->dh_momentum_y.device_to_host();
    this->dh_momentum_z.device_to_host();
    const double kernel_momentum_x = this->dh_momentum_x.h_buffer.ptr[0];
    const double kernel_momentum_y = this->dh_momentum_y.h_buffer.ptr[0];
    const double kernel_momentum_z = this->dh_momentum_z.h_buffer.ptr[0];
    this->momentum_x = 0.0;
    this->momentum_y = 0.0;
    this->momentum_z = 0.0;

    MPICHK(MPI_Allreduce(&kernel_momentum_x, &(this->momentum_x), 1, MPI_DOUBLE,
                         MPI_SUM, this->comm));
    MPICHK(MPI_Allreduce(&kernel_momentum_y, &(this->momentum_y), 1, MPI_DOUBLE,
                         MPI_SUM, this->comm));
    MPICHK(MPI_Allreduce(&kernel_momentum_z, &(this->momentum_z), 1, MPI_DOUBLE,
                         MPI_SUM, this->comm));

    return std::tuple<double, double, double>(
        this->momentum_x, this->momentum_y, this->momentum_z);
  }
};

#endif
