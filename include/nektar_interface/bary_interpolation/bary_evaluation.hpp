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
 * @param[in] bw_values A length num_phys array containing the quadrature
 * weights.
 * @param[in, out] div_values Array of length num_phys * stride which will be
 * populated with the bw_i/(r - r_i) values.
 * @param[in] stride Stride to use for storing elements in div_values,
 * default 1.
 */
inline void preprocess_weights(const int num_phys, const REAL coord,
                               const REAL *const z_values,
                               const REAL *const bw_values, REAL *div_values,
                               const std::size_t stride = 1) {
  for (int ix = 0; ix < num_phys; ix++) {
    div_values[ix * stride] = 0.0;
  }
  for (int ix = 0; ix < num_phys; ix++) {
    const auto xdiff = z_values[ix] - coord;
    if (xdiff == 0.0) {
      div_values[ix * stride] = 1.0;
      return;
    }
  }
  REAL denom = 0.0;
  for (int ix = 0; ix < num_phys; ix++) {
    const auto xdiff = z_values[ix] - coord;
    const auto bw_over_diff = bw_values[ix] / xdiff;
    div_values[ix * stride] = bw_over_diff;
    denom += bw_over_diff;
  }
  const REAL factor = 1.0 / denom;
  for (int ix = 0; ix < num_phys; ix++) {
    div_values[ix * stride] *= factor;
  }
}

/**
 * For quadrature point r_i with weight bw_i compute bw_i / (r - r_i).
 *
 * @param num_phys The number of quadrature points for the element and
 * dimension for which this computation is performed.
 * @param[in] coord The evauation points, size N, in the dimension of interest.
 * @param[in] z_values A length num_phys array containing the quadrature
 * points.
 * @param[in] bw_values A length num_phys array containing the quadrature
 * weights.
 * @param[in, out] div_values Array of length num_phys * N which will be
 * populated with the bw_i/(r - r_i) values. Ordering is particle then
 * component.
 */
template <int N>
inline void
preprocess_weights_block(const int num_phys, const REAL *const coord,
                         const REAL *const z_values,
                         const REAL *const bw_values, REAL *div_values) {

  for (int ix = 0; ix < num_phys; ix++) {
    for (int blockx = 0; blockx < N; blockx++) {
      div_values[ix * N + blockx] = 0.0;
    }
  }
  bool on_point[N];
  for (int blockx = 0; blockx < N; blockx++) {
    on_point[blockx] = false;
  }

  for (int ix = 0; ix < num_phys; ix++) {
    for (int blockx = 0; blockx < N; blockx++) {
      const auto xdiff = z_values[ix] - coord[blockx];
      if (xdiff == 0.0) {
        div_values[ix * N + blockx] = 1.0;
        on_point[blockx] = true;
      }
    }
  }
  REAL denom[N];
  for (int blockx = 0; blockx < N; blockx++) {
    denom[blockx] = 0.0;
  }

  for (int ix = 0; ix < num_phys; ix++) {
    for (int blockx = 0; blockx < N; blockx++) {
      const auto xdiff = z_values[ix] - coord[blockx];
      const auto bw_over_diff = bw_values[ix] / xdiff;
      div_values[ix * N + blockx] =
          on_point[blockx] ? div_values[ix * N + blockx] : bw_over_diff;
      denom[blockx] += bw_over_diff;
    }
  }
  REAL factor[N];
  for (int blockx = 0; blockx < N; blockx++) {
    factor[blockx] = on_point[blockx] ? 1.0 : 1.0 / denom[blockx];
  }

  for (int ix = 0; ix < num_phys; ix++) {
    for (int blockx = 0; blockx < N; blockx++) {
      div_values[ix * N + blockx] *= factor[blockx];
    }
  }
}

/**
 * Perform Bary interpolation in the first dimension.
 *
 * @param num_phys Number of quadrature points.
 * @param physvals Vector of length num_phys plus padding to multiple of the
 * vector length which contains the quadrature point values.
 * @param stride Stride between elements in div_space, default 1.
 * @returns Contribution to Bary interpolation from a dimension 0 evaluation.
 */
inline REAL compute_dir_0(const int num_phys, const REAL *const physvals,
                          const REAL *const div_space,
                          const std::size_t stride = 1) {
  REAL numer = 0.0;
  for (int ix = 0; ix < num_phys; ix++) {
    const REAL pval = physvals[ix];
    const REAL tmp = div_space[ix * stride];
    numer += tmp * pval;
  }
  const REAL eval0 = numer;
  return eval0;
}

/**
 * Computes Bary interpolation over two dimensions.
 *
 * @param num_phys0 Number of quadrature points in dimension 0.
 * @param num_phys1 Number of quadrature points in dimension 1.
 * @param physvals Array of function values at quadrature points.
 * @param div_space0 The output of preprocess_weights applied to dimension 0.
 * @param div_space1 The output of preprocess_weights applied to dimension 1.
 * @param stride Stride between elements in div_space, default 1.
 * @returns Bary evaluation of a function at a coordinate.
 */
inline REAL compute_dir_10(const int num_phys0, const int num_phys1,
                           const REAL *const physvals,
                           const REAL *const div_space0,
                           const REAL *const div_space1,
                           const std::size_t stride = 1) {
  REAL pval1 = 0.0;
  for (int i1 = 0; i1 < num_phys1; i1++) {
    const REAL c1 = div_space1[i1 * stride];
    for (int i0 = 0; i0 < num_phys0; i0++) {
      pval1 += physvals[i1 * num_phys0 + i0] * div_space0[i0 * stride] * c1;
    }
  }
  return pval1;
}

/**
 * Computes Bary interpolation over two dimensions. Evaluates N functions
 * with interlaced quadrature point values.
 *
 * @param[in] num_phys0 Number of quadrature points in dimension 0.
 * @param[in] num_phys1 Number of quadrature points in dimension 1.
 * @param[in] physvals Array of function values at quadrature points interlaced
 * values for each function to evaluate.
 * @param[in] div_space0 The output of preprocess_weights applied to dimension
 * 0.
 * @param[in] div_space1 The output of preprocess_weights applied to
 * dimension 1.
 * @param[in, out] output Output function evaluations.
 * @param[in] stride Stride between elements in div_space, default 1.
 */
template <std::size_t N>
inline void compute_dir_10_interlaced(const int num_phys0, const int num_phys1,
                                      const REAL *const physvals,
                                      const REAL *const div_space0,
                                      const REAL *const div_space1,
                                      REAL *RESTRICT output,
                                      const std::size_t stride = 1) {
  REAL tmp[N];
  for (int ix = 0; ix < N; ix++) {
    tmp[ix] = 0.0;
  }
  for (int i1 = 0; i1 < num_phys1; i1++) {
    const REAL c1 = div_space1[i1 * stride];
    for (int i0 = 0; i0 < num_phys0; i0++) {
      const int inner_stride = (i1 * num_phys0 + i0) * N;
      const REAL inner_c = div_space0[i0 * stride] * c1;
      for (int ix = 0; ix < N; ix++) {
        tmp[ix] += physvals[inner_stride + ix] * inner_c;
      }
    }
  }
  for (int ix = 0; ix < N; ix++) {
    output[ix] = tmp[ix];
  }
}

/**
 * Computes Bary interpolation over two dimensions. Evaluates N functions
 * with interlaced quadrature point values.
 *
 * @param[in] num_functions Number of functions to evaluate.
 * @param[in] num_phys0 Number of quadrature points in dimension 0.
 * @param[in] num_phys1 Number of quadrature points in dimension 1.
 * @param[in] physvals Array of function values at quadrature points interlaced
 * values for each function to evaluate.
 * @param[in] div_space0 The output of preprocess_weights applied to dimension
 * 0.
 * @param[in] div_space1 The output of preprocess_weights applied to
 * dimension 1.
 * @param[in, out] output Output function evaluations.
 * @param[in] stride Stride between elements in div_space, default 1.
 * @param[in] stride_output Stride between elements in output, default 1.
 */
inline void compute_dir_10_interlaced(
    const int num_functions, const int num_phys0, const int num_phys1,
    const REAL *const physvals, const REAL *const div_space0,
    const REAL *const div_space1, REAL *RESTRICT output,
    const std::size_t stride = 1, const std::size_t stride_output = 1) {
  for (int ix = 0; ix < num_functions; ix++) {
    output[ix * stride_output] = 0.0;
  }
  for (int i1 = 0; i1 < num_phys1; i1++) {
    const REAL c1 = div_space1[i1 * stride];
    for (int i0 = 0; i0 < num_phys0; i0++) {
      const int inner_stride = (i1 * num_phys0 + i0) * num_functions;
      const REAL inner_c = div_space0[i0 * stride] * c1;
      for (int ix = 0; ix < num_functions; ix++) {
        output[ix * stride_output] += physvals[inner_stride + ix] * inner_c;
      }
    }
  }
}

/**
 * Computes Bary interpolation over three dimensions.
 *
 * @param num_phys0 Number of quadrature points in dimension 0.
 * @param num_phys1 Number of quadrature points in dimension 1.
 * @param num_phys2 Number of quadrature points in dimension 2.
 * @param physvals Array of function values at quadrature points.
 * @param div_space0 The output of preprocess_weights applied to dimension 0.
 * @param div_space1 The output of preprocess_weights applied to dimension 1.
 * @param div_space2 The output of preprocess_weights applied to dimension 2.
 * @param stride Stride between elements in div_space, default 1.
 * @returns Bary evaluation of a function at a coordinate.
 */
inline REAL compute_dir_210(const int num_phys0, const int num_phys1,
                            const int num_phys2, const REAL *const physvals,
                            const REAL *const div_space0,
                            const REAL *const div_space1,
                            const REAL *const div_space2,
                            const std::size_t stride = 1) {

  const int stride_phys = num_phys0 * num_phys1;
  REAL pval2 = 0.0;
  for (int i2 = 0; i2 < num_phys2; i2++) {
    const REAL c2 = div_space2[i2 * stride];
    for (int i1 = 0; i1 < num_phys1; i1++) {
      const REAL c1 = c2 * div_space1[i1 * stride];
      for (int i0 = 0; i0 < num_phys0; i0++) {
        pval2 += physvals[i2 * stride_phys + i1 * num_phys0 + i0] *
                 div_space0[i0 * stride] * c1;
      }
    }
  }
  return pval2;
}

/**
 * Computes Bary interpolation over three dimensions. Evaluates N functions
 * with interlaced quadrature point values.
 *
 * @param[in] num_phys0 Number of quadrature points in dimension 0.
 * @param[in] num_phys1 Number of quadrature points in dimension 1.
 * @param[in] num_phys2 Number of quadrature points in dimension 2.
 * @param[in] physvals Array of function values at quadrature points interlaced
 * values for each function to evaluate.
 * @param[in] div_space0 The output of preprocess_weights applied to dimension
 * 0.
 * @param[in] div_space1 The output of preprocess_weights applied to
 * dimension 1.
 * @param[in] div_space2 The output of preprocess_weights applied to
 * dimension 2.
 * @param[in, out] output Output function evaluations.
 * @param[in] stride Stride between elements in div_space, default 1.
 *
 */
template <std::size_t N>
inline void compute_dir_210_interlaced(
    const int num_phys0, const int num_phys1, const int num_phys2,
    const REAL *const physvals, const REAL *const div_space0,
    const REAL *const div_space1, const REAL *const div_space2,
    REAL *RESTRICT output, const std::size_t stride = 1) {
  REAL tmp[N];
  for (int ix = 0; ix < N; ix++) {
    tmp[ix] = 0.0;
  }
  const int stride_phys = num_phys0 * num_phys1;
  for (int i2 = 0; i2 < num_phys2; i2++) {
    const REAL c2 = div_space2[i2 * stride];
    for (int i1 = 0; i1 < num_phys1; i1++) {
      const REAL c1 = c2 * div_space1[i1 * stride];
      for (int i0 = 0; i0 < num_phys0; i0++) {
        const int inner_stride = (i2 * stride_phys + i1 * num_phys0 + i0) * N;
        const REAL inner_c = div_space0[i0 * stride] * c1;
        for (int ix = 0; ix < N; ix++) {
          tmp[ix] += physvals[inner_stride + ix] * inner_c;
        }
      }
    }
  }
  for (int ix = 0; ix < N; ix++) {
    output[ix] = tmp[ix];
  }
}

/**
 * Computes Bary interpolation over three dimensions. Evaluates N functions
 * with interlaced quadrature point values.
 *
 * @param[in] num_functions Number of functions to evaluate.
 * @param[in] num_phys0 Number of quadrature points in dimension 0.
 * @param[in] num_phys1 Number of quadrature points in dimension 1.
 * @param[in] num_phys2 Number of quadrature points in dimension 2.
 * @param[in] physvals Array of function values at quadrature points interlaced
 * values for each function to evaluate.
 * @param[in] div_space0 The output of preprocess_weights applied to dimension
 * 0.
 * @param[in] div_space1 The output of preprocess_weights applied to
 * dimension 1.
 * @param[in] div_space2 The output of preprocess_weights applied to
 * dimension 2.
 * @param[in, out] output Output function evaluations.
 * @param[in] stride Stride between elements in div_space, default 1.
 * @param[in] stride_output Stride between elements in output, default 1.
 */
inline void compute_dir_210_interlaced(
    const int num_functions, const int num_phys0, const int num_phys1,
    const int num_phys2, const REAL *RESTRICT const physvals,
    const REAL *RESTRICT const div_space0,
    const REAL *RESTRICT const div_space1,
    const REAL *RESTRICT const div_space2, REAL *RESTRICT output,
    const std::size_t stride = 1, const std::size_t stride_output = 1) {
  for (int ix = 0; ix < num_functions; ix++) {
    output[ix * stride_output] = 0.0;
  }

  const int stride_phys = num_phys0 * num_phys1;
  for (int i2 = 0; i2 < num_phys2; i2++) {
    const REAL c2 = div_space2[i2 * stride];
    for (int i1 = 0; i1 < num_phys1; i1++) {
      const REAL c1 = c2 * div_space1[i1 * stride];
      for (int i0 = 0; i0 < num_phys0; i0++) {
        const int inner_stride =
            (i2 * stride_phys + i1 * num_phys0 + i0) * num_functions;
        const REAL inner_c = div_space0[i0 * stride] * c1;
        for (int ix = 0; ix < num_functions; ix++) {
          output[ix * stride_output] += physvals[inner_stride + ix] * inner_c;
        }
      }
    }
  }
}

/**
 * Computes Bary interpolation over three dimensions. Evaluates N particles
 * with interlaced quadrature point values. See function
 * preprocess_weights_block.
 *
 * @param[in] num_functions Number of functions to evaluate.
 * @param[in] num_phys0 Number of quadrature points in dimension 0.
 * @param[in] num_phys1 Number of quadrature points in dimension 1.
 * @param[in] num_phys2 Number of quadrature points in dimension 2.
 * @param[in] physvals Array of function values at quadrature points interlaced
 * values for each function to evaluate.
 * @param[in] div_space0 The output of preprocess_weights applied to dimension
 * 0.
 * @param[in] div_space1 The output of preprocess_weights applied to
 * dimension 1.
 * @param[in] div_space2 The output of preprocess_weights applied to
 * dimension 2.
 * @param[in, out] output Output function evaluations.
 */
template <int N>
inline void compute_dir_210_interlaced_block(
    const int num_functions, const int num_phys0, const int num_phys1,
    const int num_phys2, const REAL *RESTRICT const physvals,
    const REAL *RESTRICT const div_space0,
    const REAL *RESTRICT const div_space1,
    const REAL *RESTRICT const div_space2, REAL *RESTRICT output) {

  for (int funcx = 0; funcx < num_functions; funcx++) {
    for (int blockx = 0; blockx < N; blockx++) {
      output[funcx * N + blockx] = 0.0;
    }
  }

  const int stride_phys = num_phys0 * num_phys1;

  for (int i2 = 0; i2 < num_phys2; i2++) {
    REAL b2[N];
    for (int blockx = 0; blockx < N; blockx++) {
      b2[blockx] = div_space2[i2 * N + blockx];
    }

    for (int i1 = 0; i1 < num_phys1; i1++) {
      REAL b1[N];
      for (int blockx = 0; blockx < N; blockx++) {
        b1[blockx] = div_space1[i1 * N + blockx] * b2[blockx];
      }

      for (int i0 = 0; i0 < num_phys0; i0++) {
        REAL basis_eval[N];
        for (int blockx = 0; blockx < N; blockx++) {
          const REAL b0 = div_space0[i0 * N + blockx];
          basis_eval[blockx] = b0 * b1[blockx];
        }

        for (int funcx = 0; funcx < num_functions; funcx++) {
          const int inner_stride =
              (i2 * stride_phys + i1 * num_phys0 + i0) * num_functions;
          const REAL func_coeff = physvals[inner_stride + funcx];
          for (int blockx = 0; blockx < N; blockx++) {
            output[funcx * N + blockx] += basis_eval[blockx] * func_coeff;
          }
        }
      }
    }
  }
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
 * @param div_space Space of size (num_phys0 + num_phys1 + num_phys2) * stride
 * to use as temporary space.
 * @param z0 Quadrature points in x direction.
 * @param z1 Quadrature points in y direction.
 * @param bw0 Weights for each quadrature point in x direction.
 * @param bw1 Weights for each quadrature point in y direction.
 * @param stride Stride between elements in div_space, default 1.
 * @returns Evaluation at passed point using Bary interpolation.
 * */
inline REAL evaluate_2d(const REAL coord0, const REAL coord1,
                        const int num_phys0, const int num_phys1,
                        const REAL *const physvals, REAL *div_space,
                        const REAL *const z0, const REAL *const z1,
                        const REAL *const bw0, const REAL *const bw1,
                        const std::size_t stride = 1) {

  REAL *div_space0 = div_space;
  REAL *div_space1 = div_space0 + num_phys0 * stride;

  preprocess_weights(num_phys0, coord0, z0, bw0, div_space0, stride);
  preprocess_weights(num_phys1, coord1, z1, bw1, div_space1, stride);

  REAL eval = compute_dir_10(num_phys0, num_phys1, physvals, div_space0,
                             div_space1, stride);

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
 * @param div_space Space of size (num_phys0 + num_phys1 + num_phys2) * stride
 * to use as temporary space.
 * @param z0 Quadrature points in x direction.
 * @param z1 Quadrature points in y direction.
 * @param z2 Quadrature points in z direction.
 * @param bw0 Weights for each quadrature point in x direction.
 * @param bw1 Weights for each quadrature point in y direction.
 * @param bw2 Weights for each quadrature point in z direction.
 * @param stride Stride between elements in div_space, default 1.
 * @returns Evaluation at passed point using Bary interpolation.
 */
inline REAL evaluate_3d(const REAL coord0, const REAL coord1, const REAL coord2,
                        const int num_phys0, const int num_phys1,
                        const int num_phys2, const REAL *const physvals,
                        REAL *div_space, const REAL *const z0,
                        const REAL *const z1, const REAL *const z2,
                        const REAL *const bw0, const REAL *const bw1,
                        const REAL *const bw2, const std::size_t stride = 1) {

  REAL *div_space0 = div_space;
  REAL *div_space1 = div_space0 + num_phys0 * stride;
  REAL *div_space2 = div_space1 + num_phys1 * stride;

  preprocess_weights(num_phys0, coord0, z0, bw0, div_space0, stride);
  preprocess_weights(num_phys1, coord1, z1, bw1, div_space1, stride);
  preprocess_weights(num_phys2, coord2, z2, bw2, div_space2, stride);

  const REAL eval = compute_dir_210(num_phys0, num_phys1, num_phys2, physvals,
                                    div_space0, div_space1, div_space2, stride);

  return eval;
}

} // namespace NESO::Bary

#endif
