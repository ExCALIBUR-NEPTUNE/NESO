#ifndef __POTENTIAL_ENERGY_H_
#define __POTENTIAL_ENERGY_H_

#include <hdf5.h>
#include <memory>
#include <mpi.h>

#include <nektar_interface/function_evaluation.hpp>
#include <neso_particles.hpp>
using namespace NESO;
using namespace NESO::Particles;

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
using namespace Nektar;

#include "field_normalisation.hpp"

/**
 *  Class to compute and write to a HDF5 file the integral of a function
 *  squared.
 */
template <typename T> class PotentialEnergy {
private:
  int rank;
  hid_t file;
  Array<OneD, NekDouble> phys_values;
  int num_quad_points;
  int step;

  inline void fe_H5CHK(const bool flag) { ASSERTL1((cmd) >= 0, "HDF5 ERROR"); }

  BufferDeviceHost<double> dh_energy;
  std::shared_ptr<FieldEvaluate<T>> field_evaluate;
  std::shared_ptr<FieldNormalisation<T>> field_normalisation;

public:
  /// The Nektar++ field of interest.
  std::shared_ptr<T> field;
  /// In use ParticleGroup
  ParticleGroupSharedPtr particle_group;
  /// The output HDF5 filename.
  std::string filename;
  /// The MPI communicator used by this instance.
  MPI_Comm comm;
  /// The last field energy that was computed on call to write.
  double energy;
  /*
   *  Create new instance.
   *
   *  @param field Nektar++ field (DisContField, ContField) to use.
   *  @param particle_group ParticleGroup to use.
   *  @param cell_id_translation CellIDTranslation to use.
   *  @param filename Filename of HDF5 output file.
   *  @param comm MPI communicator (default MPI_COMM_WORLD).
   */
  PotentialEnergy(std::shared_ptr<T> field,
                  ParticleGroupSharedPtr particle_group,
                  std::shared_ptr<CellIDTranslation> cell_id_translation,
                  std::string filename, MPI_Comm comm = MPI_COMM_WORLD)
      : field(field), particle_group(particle_group), filename(filename),
        comm(comm), step(0), dh_energy(particle_group->sycl_target, 1) {

    int flag;
    int err;
    err = MPI_Initialized(&flag);
    ASSERTL1(err == MPI_SUCCESS, "MPI_Initialised error.");
    ASSERTL1(flag, "MPI is not initialised");

    err = MPI_Comm_rank(this->comm, &(this->rank));
    ASSERTL1(err == MPI_SUCCESS, "Error getting MPI rank.");

    // only rank 0 interfaces with hdf5 as nektar reduces the integrals
    if (this->rank == 0) {
      this->file = H5Fcreate(this->filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                             H5P_DEFAULT);
      ASSERTL1(this->file != H5I_INVALID_HID, "Invalid HDF5 file identifier");
    }

    // create space to store u^2
    this->num_quad_points = this->field->GetNpoints();
    this->phys_values = Array<OneD, NekDouble>(num_quad_points);

    this->particle_group->add_particle_dat(
        ParticleDat(this->particle_group->sycl_target,
                    ParticleProp(Sym<REAL>("ELEC_PIC_PE"), 1),
                    this->particle_group->domain->mesh->get_cell_count()));

    this->field_evaluate = std::make_shared<FieldEvaluate<T>>(
        this->field, this->particle_group, cell_id_translation, false);

    this->field_normalisation =
        std::make_shared<FieldNormalisation<T>>(this->field);
  }

  /**
   *  Compute the current energy of the field and write to the HDF5 file.
   *
   *  @param step_in Optional integer to set the iteration step.
   */
  inline void write(int step_in = -1) {

    if (step_in > -1) {
      this->step = step_in;
    }

    this->field_evaluate->evaluate(Sym<REAL>("ELEC_PIC_PE"));

    auto t0 = profile_timestamp();
    auto sycl_target = this->particle_group->sycl_target;
    const auto k_Q =
        (*this->particle_group)[Sym<REAL>("Q")]->cell_dat.device_ptr();
    const auto k_PHI = (*this->particle_group)[Sym<REAL>("ELEC_PIC_PE")]
                           ->cell_dat.device_ptr();

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    this->dh_energy.h_buffer.ptr[0] = 0.0;
    this->dh_energy.host_to_device();

    auto k_energy = this->dh_energy.d_buffer.ptr;
    const double k_potential_shift = this->field_normalisation->get_shift();

    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                const double phi = k_PHI[cellx][0][layerx];
                const double q = k_Q[cellx][0][layerx];
                const double tmp_contrib = q * (phi + k_potential_shift);

                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    energy_atomic(k_energy[0]);
                energy_atomic.fetch_add(tmp_contrib);

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target->profile_map.inc("PotentialEnergy", "Execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));
    this->dh_energy.device_to_host();
    const double kernel_energy = this->dh_energy.h_buffer.ptr[0];

    MPICHK(MPI_Allreduce(&kernel_energy, &(this->energy), 1, MPI_DOUBLE,
                         MPI_SUM, this->comm));

    if (this->rank == 0) {
      ASSERTL1(this->file != H5I_INVALID_HID,
               "Invalid file identifier on write.");

      // write the value to the HDF5 file.
      // Create the group for this time step.
      std::string step_name = "Step#";
      step_name += std::to_string(this->step++);
      hid_t group_step = H5Gcreate(this->file, step_name.c_str(), H5P_DEFAULT,
                                   H5P_DEFAULT, H5P_DEFAULT);

      const hsize_t dims[1] = {1};
      auto dataspace = H5Screate_simple(1, dims, NULL);
      auto dataset =
          H5Dcreate2(group_step, "potential_energy", H5T_NATIVE_DOUBLE,
                     dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      fe_H5CHK(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, &(this->energy)));

      fe_H5CHK(H5Dclose(dataset));
      fe_H5CHK(H5Sclose(dataspace));
      fe_H5CHK(H5Gclose(group_step));
    }
  }

  /**
   * Close the HDF5 file. Close must be called before the instance is freed.
   */
  inline void close() {
    if (this->rank == 0) {
      fe_H5CHK(H5Fclose(this->file));
      this->file = H5I_INVALID_HID;
    }
  }
};

#endif
