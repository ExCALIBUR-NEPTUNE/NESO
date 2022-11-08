#ifndef __KINETIC_ENERGY_H_
#define __KINETIC_ENERGY_H_

#include <hdf5.h>
#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>

using namespace NESO::Particles;

/**
 * Compute the kinetic energy of particles in a ParticleGroup.
 */
class KineticEnergy {
private:
  int rank;
  hid_t file;
  int step;
  BufferDeviceHost<double> dh_kinetic_energy;

public:
  /// ParticleGroup of interest.
  ParticleGroupSharedPtr particle_group;
  /// The output HDF5 filename.
  std::string filename;
  /// The MPI communicator used by this instance.
  MPI_Comm comm;
  /// The last kinetic energy that was computed on call to write.
  double kinetic_energy;
  /// The mass of the particles.
  const double particle_mass;

  /*
   *  Create new instance.
   *
   *  @parm particle_group ParticleGroup to compute kinetic energy of.
   *  @param filename Filename of HDF5 output file.
   *  @param particle_mass Mass of each particle.
   *  @param comm MPI communicator (default MPI_COMM_WORLD).
   */
  KineticEnergy(ParticleGroupSharedPtr particle_group, std::string filename,
                const double particle_mass, MPI_Comm comm = MPI_COMM_WORLD)
      : particle_group(particle_group), filename(filename),
        particle_mass(particle_mass), comm(comm), step(0),
        dh_kinetic_energy(particle_group->sycl_target, 1) {

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
  }

  /**
   *  Compute the current kinetic energy of the ParticleGroup and write to the
   *  HDF5 file.
   *
   *  @param step_in Optional integer to set the iteration step.
   */
  inline void write(int step_in = -1) {

    if (step_in > -1) {
      this->step = step_in;
    }

    auto t0 = profile_timestamp();
    auto sycl_target = this->particle_group->sycl_target;
    const double k_half_particle_mass = 0.5 * this->particle_mass;
    auto k_V = (*this->particle_group)[Sym<REAL>("V")]->cell_dat.device_ptr();

    const auto pl_iter_range =
        this->particle_group->mpi_rank_dat->get_particle_loop_iter_range();
    const auto pl_stride =
        this->particle_group->mpi_rank_dat->get_particle_loop_cell_stride();
    const auto pl_npart_cell =
        this->particle_group->mpi_rank_dat->get_particle_loop_npart_cell();

    this->dh_kinetic_energy.h_buffer.ptr[0] = 0.0;
    this->dh_kinetic_energy.host_to_device();

    auto k_kinetic_energy = this->dh_kinetic_energy.d_buffer.ptr;

    sycl_target->queue
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for<>(
              sycl::range<1>(pl_iter_range), [=](sycl::id<1> idx) {
                NESO_PARTICLES_KERNEL_START
                const INT cellx = NESO_PARTICLES_KERNEL_CELL;
                const INT layerx = NESO_PARTICLES_KERNEL_LAYER;

                const double V0 = k_V[cellx][0][layerx];
                const double V1 = k_V[cellx][1][layerx];
                const double half_mvv =
                    k_half_particle_mass * (V0 * V0 + V1 * V1);

                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                 sycl::memory_scope::device>
                    kinetic_energy_atomic(k_kinetic_energy[0]);
                kinetic_energy_atomic.fetch_add(half_mvv);

                NESO_PARTICLES_KERNEL_END
              });
        })
        .wait_and_throw();
    sycl_target->profile_map.inc("KineticEnergy", "Execute", 1,
                                 profile_elapsed(t0, profile_timestamp()));
    this->dh_kinetic_energy.device_to_host();
    const double kernel_kinetic_energy =
        this->dh_kinetic_energy.h_buffer.ptr[0];

    MPICHK(MPI_Allreduce(&kernel_kinetic_energy, &(this->kinetic_energy), 1,
                         MPI_DOUBLE, MPI_SUM, this->comm));

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
          H5Dcreate2(group_step, "kinetic_energy", H5T_NATIVE_DOUBLE, dataspace,
                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

      H5CHK(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     &(this->kinetic_energy)));

      H5CHK(H5Dclose(dataset));
      H5CHK(H5Sclose(dataspace));
      H5CHK(H5Gclose(group_step));
    }
  }

  /**
   * Close the HDF5 file. Close must be called before the instance is freed.
   */
  inline void close() {
    if (this->rank == 0) {
      H5CHK(H5Fclose(this->file));
      this->file = H5I_INVALID_HID;
    }
  }
};

#endif
