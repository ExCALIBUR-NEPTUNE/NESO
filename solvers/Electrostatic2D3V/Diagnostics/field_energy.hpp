#ifndef __FIELD_ENERGY_H_
#define __FIELD_ENERGY_H_

#include <memory>
#include <mpi.h>
#include <neso_particles.hpp>

#include <LibUtilities/BasicUtils/ErrorUtil.hpp>
using namespace Nektar;
using namespace NESO::Particles;

#include "field_mean.hpp"

/**
 *  Class to compute and write to a HDF5 file the integral of a function
 *  squared.
 */
template <typename T> class FieldEnergy {
private:
  Array<OneD, NekDouble> phys_values;
  int num_quad_points;
  std::shared_ptr<FieldMean<T>> field_mean;

public:
  /// The Nektar++ field of interest.
  std::shared_ptr<T> field;
  /// The MPI communicator used by this instance.
  MPI_Comm comm;
  /// The last field energy that was computed on call to write.
  double energy;
  /*
   *  Create new instance.
   *
   *  @param field Nektar++ field (DisContField, ContField) to use.
   *  @param comm MPI communicator (default MPI_COMM_WORLD).
   */
  FieldEnergy(std::shared_ptr<T> field, MPI_Comm comm = MPI_COMM_WORLD)
      : field(field), comm(comm) {

    int flag;
    MPICHK(MPI_Initialized(&flag));
    ASSERTL1(flag, "MPI is not initialised");

    // create space to store u^2
    this->num_quad_points = this->field->GetNpoints();
    this->phys_values = Array<OneD, NekDouble>(num_quad_points);

    this->field_mean = std::make_shared<FieldMean<T>>(this->field);
  }

  /**
   *  Compute the current energy of the field.
   *
   *  @param step_in Optional integer to set the iteration step.
   */
  inline double compute() {

    const double potential_shift = -this->field_mean->get_mean();
    // compute u^2 at the quadrature points
    auto field_phys_values = this->field->GetPhys();
    for (int pointx = 0; pointx < num_quad_points; pointx++) {
      const NekDouble point_value = field_phys_values[pointx];
      const NekDouble shifted_point_value = point_value + potential_shift;
      this->phys_values[pointx] = shifted_point_value * shifted_point_value;
    }

    // nektar reduces this value across MPI ranks
    this->energy = this->field->Integral(this->phys_values);
    return this->energy;
  }
};

#endif
