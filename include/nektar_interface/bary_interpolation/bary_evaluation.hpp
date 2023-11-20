#ifndef __BARY_EVALUATION_H__
#define __BARY_EVALUATION_H__

#include <neso_particles.hpp>
using namespace NESO::Particles;

namespace NESO::Bary {

/**
 * For quadrature point r_i with weight bw_i compute bw_i / (r - r_i).
 *
 * @param num_phys The number of quadrature points for the element and
 * dimension for which this computation is performed.
 * @param[in] coord The evauation point in the dimension of interest.
 * @param[in] z_values A length num_phys array containing the quadrature
 * points.
 * @param[in] z_values A length num_phys array containing the quadrature
 * weights.
 * @param[in, out] div_values Array of length num_phys which will be
 * populated with the bw_i/(r - r_i) values.
 */
inline void preprocess_weights(const int num_phys, const REAL coord,
                               const REAL *const z_values,
                               const REAL *const bw_values, REAL *div_values) {
  for (int ix = 0; ix < num_phys; ix++) {
    div_values[ix] = 0.0;
  }
  for (int ix = 0; ix < num_phys; ix++) {
    const auto xdiff = z_values[ix] - coord;
    if (xdiff == 0.0) {
      div_values[ix] = 1.0;
      return;
    }
  }
  REAL denom = 0.0;
  for (int ix = 0; ix < num_phys; ix++) {
    const auto xdiff = z_values[ix] - coord;
    const auto bw_over_diff = bw_values[ix] / xdiff;
    div_values[ix] = bw_over_diff;
    denom += bw_over_diff;
  }
  const REAL factor = 1.0 / denom;
  for (int ix = 0; ix < num_phys; ix++) {
    div_values[ix] *= factor;
  }
}

/**
 * Perform Bary interpolation in the first dimension. This function is
 * intended to be called from a function that performs Bary interpolation
 * over the second dimension and first dimension.
 *
 * @param num_phys Number of quadrature points.
 * @param physvals Vector of length num_phys plus padding to multiple of the
 * vector length which contains the quadrature point values.
 * @returns Contribution to Bary interpolation from a dimension 0 evaluation.
 */
inline REAL compute_dir_0(const int num_phys, const REAL *const physvals,
                          const REAL *const div_space) {
  REAL numer = 0.0;
  for (int ix = 0; ix < num_phys; ix++) {
    const REAL pval = physvals[ix];
    const REAL tmp = div_space[ix];
    numer += tmp * pval;
  }
  const REAL eval0 = numer;
  return eval0;
}

/**
 * Computes Bary interpolation over two dimensions. The inner dimension is
 * computed with calls to compute_dir_0.
 *
 * @param num_phys0 Number of quadrature points in dimension 0.
 * @param num_phys1 Number of quadrature points in dimension 1.
 * @param physvals Array of function values at quadrature points.
 * @param div_space0 The output of preprocess_weights applied to dimension 0.
 * @param div_space1 The output of preprocess_weights applied to dimension 1.
 * @returns Bary evaluation of a function at a coordinate.
 */
inline REAL compute_dir_10(const int num_phys0, const int num_phys1,
                           const REAL *const physvals,
                           const REAL *const div_space0,
                           const REAL *const div_space1) {
  REAL pval1 = 0.0;
  for (int i1 = 0; i1 < num_phys1; i1++) {
    const REAL c1 = div_space1[i1];
    for (int i0 = 0; i0 < num_phys0; i0++) {
      pval1 += physvals[i1 * num_phys0 + i0] * div_space0[i0] * c1;
    }
  }
  return pval1;
}

/**
 * Computes Bary interpolation over three dimensions. The inner dimensions are
 * computed with calls to compute_dir_10.
 *
 * @param num_phys0 Number of quadrature points in dimension 0.
 * @param num_phys1 Number of quadrature points in dimension 1.
 * @param num_phys2 Number of quadrature points in dimension 2.
 * @param physvals Array of function values at quadrature points.
 * @param div_space0 The output of preprocess_weights applied to dimension 0.
 * @param div_space1 The output of preprocess_weights applied to dimension 1.
 * @param div_space2 The output of preprocess_weights applied to dimension 2.
 * @returns Bary evaluation of a function at a coordinate.
 */
inline REAL compute_dir_210(const int num_phys0, const int num_phys1,
                            const int num_phys2, const REAL *const physvals,
                            const REAL *const div_space0,
                            const REAL *const div_space1,
                            const REAL *const div_space2) {
  const int stride = num_phys0 * num_phys1;
  REAL pval2 = 0.0;
  for (int i2 = 0; i2 < num_phys2; i2++) {
    const REAL c2 = div_space2[i2];
    for (int i1 = 0; i1 < num_phys1; i1++) {
      const REAL c1 = c2 * div_space1[i1];
      for (int i0 = 0; i0 < num_phys0; i0++) {
        pval2 +=
            physvals[i2 * stride + i1 * num_phys0 + i0] * div_space0[i0] * c1;
      }
    }
  }
  return pval2;
}

/**
 * Compute a function evaluation at a point using the passed quadrature point
 * values, quadrature points and weights.
 *
 * @param coord0 Evaluation coordinate, x component.
 * @param coord1 Evaluation coordinate, y component.
 * @param num_phys0 Number of quadrature points in the x direction.
 * @param num_phys1 Number of quadrature points in the y direction.
 * @param physvals Function evaluations at quadrature points, x runs fastest
 * and y slowest.
 * @param div_space Space of size num_phys0 + num_phys1 + num_phys2 to use as
 * temporary space.
 * @param z0 Quadrature points in x direction.
 * @param z1 Quadrature points in y direction.
 * @param bw0 Weights for each quadrature point in x direction.
 * @param bw1 Weights for each quadrature point in y direction.
 * @returns Evaluation at passed point using Bary interpolation.
 * */
inline REAL evaluate_2d(const REAL coord0, const REAL coord1,
                        const int num_phys0, const int num_phys1,
                        const REAL *const physvals, REAL *div_space,
                        const REAL *const z0, const REAL *const z1,
                        const REAL *const bw0, const REAL *const bw1) {

  REAL *div_space0 = div_space;
  REAL *div_space1 = div_space0 + num_phys0;

  preprocess_weights(num_phys0, coord0, z0, bw0, div_space0);
  preprocess_weights(num_phys1, coord1, z1, bw1, div_space1);

  REAL eval =
      compute_dir_10(num_phys0, num_phys1, physvals, div_space0, div_space1);

  return eval;
}

/**
 * Compute a function evaluation at a point using the passed quadrature point
 * values, quadrature points and weights.
 *
 * @param coord0 Evaluation coordinate, x component.
 * @param coord1 Evaluation coordinate, y component.
 * @param coord2 Evaluation coordinate, z component.
 * @param num_phys0 Number of quadrature points in the x direction.
 * @param num_phys1 Number of quadrature points in the y direction.
 * @param num_phys2 Number of quadrature points in the z direction.
 * @param physvals Function evaluations at quadrature points, x runs fastest
 * and z slowest.
 * @param div_space Space of size num_phys0 + num_phys1 + num_phys2 to use as
 * temporary space.
 * @param z0 Quadrature points in x direction.
 * @param z1 Quadrature points in y direction.
 * @param z2 Quadrature points in z direction.
 * @param bw0 Weights for each quadrature point in x direction.
 * @param bw1 Weights for each quadrature point in y direction.
 * @param bw2 Weights for each quadrature point in z direction.
 * @returns Evaluation at passed point using Bary interpolation.
 */
inline REAL evaluate_3d(const REAL coord0, const REAL coord1, const REAL coord2,
                        const int num_phys0, const int num_phys1,
                        const int num_phys2, const REAL *const physvals,
                        REAL *div_space, const REAL *const z0,
                        const REAL *const z1, const REAL *const z2,
                        const REAL *const bw0, const REAL *const bw1,
                        const REAL *const bw2) {

  REAL *div_space0 = div_space;
  REAL *div_space1 = div_space0 + num_phys0;
  REAL *div_space2 = div_space1 + num_phys1;

  preprocess_weights(num_phys0, coord0, z0, bw0, div_space0);
  preprocess_weights(num_phys1, coord1, z1, bw1, div_space1);
  preprocess_weights(num_phys2, coord2, z2, bw2, div_space2);

  const REAL eval = compute_dir_210(num_phys0, num_phys1, num_phys2, physvals,
                                    div_space0, div_space1, div_space2);

  return eval;
}

} // namespace NESO::Bary

#endif
