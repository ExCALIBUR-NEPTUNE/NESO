#ifndef __FIELD_ENERGY_H_
#define __FIELD_ENERGY_H_

#include <hdf5.h>
#include <memory>
#include <mpi.h>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
using namespace Nektar;


template <typename T> class FieldEnergy {
private:
  int rank;
  hid_t file;
  Array<OneD, NekDouble> phys_values;
  int num_quad_points;
  int step;

  static inline void H5CHK(const bool flag){
    ASSERTL1((cmd) >= 0, "HDF5 ERROR");
  }

public:
  std::shared_ptr<T> field;
  std::string filename;
  MPI_Comm comm;

  FieldEnergy(std::shared_ptr<T> field, std::string filename,
              MPI_Comm comm = MPI_COMM_WORLD)
      : field(field), filename(filename), comm(comm) {
    ASSERTL1(MPI_Comm_rank(this->comm, &this->rank) == MPI_SUCCESS,
             "Error getting MPI rank.");

    // only rank 0 interfaces with hdf5 as nektar reduces the integrals
    if (this->rank == 0) {
      ASSERTL1((this->file = H5Fcreate(this->filename.c_str(), H5ACC_TRUNC,
                                       H5P_DEFAULT, H5P_DEFAULT)) !=
                   H5I_INVALID_HID,
               "Invalid HDF5 file identifier");
    }
    
    // create space to store u^2
    this->num_quad_points = this->field->GetNpoints();
    this->phys_values = Array<OneD, NekDouble>(num_quad_points);
    this->step = 0;
  }


  inline void write() {

    // compute u^2 at the quadrature points
    auto field_phys_values = this->field->GetPhys();
    for(int pointx=0 ; pointx<num_quad_points ; pointx++){
      const NekDouble point_value = field_phys_values[pointx];
      this->phys_values[pointx] = point_value * point_value;
    }
    
    // nektar reduces this value accross MPI ranks
    const double u2_integral = this->field->Integral(this->phys_values);

    // write the value to the HDF5 file.
    // Create the group for this time step.
    std::string step_name = "Step#";
    step_name += std::to_string(this->step++);
    hid_t group_step = H5Gcreate(this->file, step_name.c_str(), H5P_DEFAULT,
                                 H5P_DEFAULT, H5P_DEFAULT);
    
    const hsize_t dims[1] = {1};
    auto dataspace = H5Screate_simple(1, dims, NULL);
    auto dataset = H5Dcreate2(group_step, "field_energy", H5T_NATIVE_DOUBLE,
                              dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);





    H5CHK(H5Dclose(dataset));
    H5CHK(H5Sclose(dataspace));
    H5CHK(H5Gclose(group_step));
  }

  inline void close() {
    if (this->rank == 0) {
      H5CHK(H5Fclose(this->file));
    }
  }
};

#endif
