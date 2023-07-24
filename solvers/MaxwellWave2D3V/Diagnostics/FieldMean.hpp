#ifndef __FIELD_MEAN_H_
#define __FIELD_MEAN_H_

#include <SolverUtils/Driver.h>
#include <memory>

using namespace Nektar;

/**
 * Helper class to compute the shift that is required to translate the given
 * field such that the integral is zero.
 */
template <typename T> class FieldMean {
private:
  std::shared_ptr<T> field;
  double volume;

public:
  /**
   *  Initialisation with field to compute shift for.
   *
   *  @param field Nektar++ field (ContField or DisContField).
   */
  FieldMean(std::shared_ptr<T> field) : field(field) {

    const int num_quad_points = this->field->GetTotPoints();
    auto phys = Array<OneD, NekDouble>(num_quad_points);
    for (int cx = 0; cx < num_quad_points; cx++) {
      phys[cx] = 1.0;
    }
    this->volume = this->field->Integral(phys);
  }

  /**
   *  Compute mean for current values at quadrature points on the field. This
   *  is the value to subtract from the field such that the integral over the
   *  domain of the field is zero.
   *
   *  @returns Field mean.
   */
  inline double get_mean() {
    // nektar reduces this value accross MPI ranks
    const double integral = this->field->Integral();
    const double shift = integral / this->volume;
    return shift;
  }
};

#endif
