#ifndef __JACOBI_EXPANSION_LOOPING_INTERFACE_H__
#define __JACOBI_EXPANSION_LOOPING_INTERFACE_H__

#include "../basis_evaluation.hpp"
#include "../coordinate_mapping.hpp"
#include <LibUtilities/BasicConst/NektarUnivConsts.hpp>
#include <SpatialDomains/MeshGraph.h>
#include <neso_particles.hpp>

using namespace Nektar::SpatialDomains;
using namespace NESO;
using namespace NESO::Particles;

namespace NESO::ExpansionLooping {

/**
 * Abstract base class for projection and evaluation implementations for each
 * element type. Assumes that the derived classes all require evaluation of
 * Jacobi polynomials (with beta=1). i.e. the methods will receive coefficients
 * such that
 *
 *  P_^{alpha, 1}_{n} =
 *      (coeffs_pnm10) * P_^{alpha, 1}_{n-1} * z
 *    + (coeffs_pnm11) * P_^{alpha, 1}_{n-1}
 *    + (coeffs_pnm2) * P_^{alpha, 1}_{n-2}
 *
 * See JacobiCoeffModBasis for further details relating to coefficient
 * computation.
 */
template <typename SPECIALISATION> struct JacobiExpansionLoopingInterface {

  /**
   * Compute the collapsed coordinate for a given input local coordinate.
   *
   * @param[in] xi0 Local coordinate, x component.
   * @param[in] xi1 Local coordinate, y component.
   * @param[in] xi2 Local coordinate, z component.
   * @param[out] eta0 Local collapsed coordinate, x component.
   * @param[out] eta1 Local collapsed coordinate, y component.
   * @param[out] eta2 Local collapsed coordinate, z component.
   */
  inline void loc_coord_to_loc_collapsed(const REAL xi0, const REAL xi1,
                                         const REAL xi2, REAL *eta0, REAL *eta1,
                                         REAL *eta2) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.loc_coord_to_loc_collapsed_v(xi0, xi1, xi2, eta0, eta1, eta2);
  }

  /**
   *  Evaluate the set of basis functions in the x direction of the reference
   * element.
   *
   *  @param[in] nummodes Number of modes in the expansion.
   *  @param[in] z Point to evaluate each of the basis functions at.
   *  @param[in] coeffs_stride Integer stride required to index into Jacobi
   *  coefficients.
   *  @param[in] coeffs_pnm10 First set of coefficients for Jacobi recursion.
   *  @param[in] coeffs_pnm11 Second set of coefficients for Jacobi recursion.
   *  @param[in] coeffs_pnm2 Third set of coefficients for Jacobi recursion.
   *  @param[out] output Output array with size at least the total number of
   *  modes for the expansion with nummodes.
   */
  inline void evaluate_basis_0(const int nummodes, const REAL z,
                               const int coeffs_stride,
                               const REAL *coeffs_pnm10,
                               const REAL *coeffs_pnm11,
                               const REAL *coeffs_pnm2, REAL *output) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.evaluate_basis_0_v(nummodes, z, coeffs_stride, coeffs_pnm10,
                                  coeffs_pnm11, coeffs_pnm2, output);
  }

  /**
   *  Evaluate the set of basis functions in the y direction of the reference
   * element.
   *
   *  @param[in] nummodes Number of modes in the expansion.
   *  @param[in] z Point to evaluate each of the basis functions at.
   *  @param[in] coeffs_stride Integer stride required to index into Jacobi
   *  coefficients.
   *  @param[in] coeffs_pnm10 First set of coefficients for Jacobi recursion.
   *  @param[in] coeffs_pnm11 Second set of coefficients for Jacobi recursion.
   *  @param[in] coeffs_pnm2 Third set of coefficients for Jacobi recursion.
   *  @param[out] output Output array with size at least the total number of
   *  modes for the expansion with nummodes.
   */
  inline void evaluate_basis_1(const int nummodes, const REAL z,
                               const int coeffs_stride,
                               const REAL *coeffs_pnm10,
                               const REAL *coeffs_pnm11,
                               const REAL *coeffs_pnm2, REAL *output) {

    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.evaluate_basis_1_v(nummodes, z, coeffs_stride, coeffs_pnm10,
                                  coeffs_pnm11, coeffs_pnm2, output);
  }

  /**
   *  Evaluate the set of basis functions in the z direction of the reference
   * element.
   *
   *  @param[in] nummodes Number of modes in the expansion.
   *  @param[in] z Point to evaluate each of the basis functions at.
   *  @param[in] coeffs_stride Integer stride required to index into Jacobi
   *  coefficients.
   *  @param[in] coeffs_pnm10 First set of coefficients for Jacobi recursion.
   *  @param[in] coeffs_pnm11 Second set of coefficients for Jacobi recursion.
   *  @param[in] coeffs_pnm2 Third set of coefficients for Jacobi recursion.
   *  @param[out] output Output array with size at least the total number of
   *  modes for the expansion with nummodes.
   */
  inline void evaluate_basis_2(const int nummodes, const REAL z,
                               const int coeffs_stride,
                               const REAL *coeffs_pnm10,
                               const REAL *coeffs_pnm11,
                               const REAL *coeffs_pnm2, REAL *output) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.evaluate_basis_2_v(nummodes, z, coeffs_stride, coeffs_pnm10,
                                  coeffs_pnm11, coeffs_pnm2, output);
  }

  /**
   * Construct each mode of the expansion over the element using the expansions
   * in each direction of the reference element. Multiply each of these modes
   * with the corresponding degree of freedom (coefficient) and sum the result.
   * Mathematically this method computes and returns
   *
   * \f[
   * \sum_{i} \phi_i(x) \alpha_i,
   * \f]
   *
   * where \f$\phi_i\f$ and \f$\alpha_i\f$ are the \f$i\f$-th basis function
   * and degree of freedom respectively.
   *
   * @param[in] nummodes Number of modes in the expansion.
   * @param[in] dofs Pointer to degrees of freedom (\f$\alpha_i\f$) to use when
   * evaluating the expansion.
   * @param[in] local_space_0 Output of `evaluate_basis_0`.
   * @param[in] local_space_1 Output of `evaluate_basis_1`.
   * @param[in] local_space_2 Output of `evaluate_basis_2`.
   * @param[output] output Output space for the evaluation (pointer to a single
   * REAL).
   */
  inline void loop_evaluate(const int nummodes, const REAL *dofs,
                            const REAL *local_space_0,
                            const REAL *local_space_1,
                            const REAL *local_space_2, REAL *output) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.loop_evaluate_v(nummodes, dofs, local_space_0, local_space_1,
                               local_space_2, output);
  }

  /**
   * Construct each mode of the expansion over the element using the expansions
   * in each direction of the reference element. For each basis function
   * \f$i\f$, atomically increment the i-th element in the RHS of
   *
   * \f[
   * M \vec{a} = \vec{\psi},
   * \f]
   *
   * where
   * \f[
   * \vec{\psi}_i = \sum{\text{particles}~j} \phi_i(\vec{r}_j) q_i,
   * \f]
   * \f$\vec{r}_j\f$ is the position of particle \f$j\f$, \f$q_i\f$ is the
   * quantity of interest on particle \f$i$\f and \f$M\f$ is the system mass
   * matrix.
   *
   * @param[in] nummodes Number of modes in the expansion.
   * @param[in] value Pointer to degrees of freedom (\f$\alpha_i\f$) to use when
   * evaluating the expansion.
   * @param[in] local_space_0 Output of `evaluate_basis_0`.
   * @param[in] local_space_1 Output of `evaluate_basis_1`.
   * @param[in] local_space_2 Output of `evaluate_basis_2`.
   * @param[output] output Output space for the evaluation of each basis
   * function times quantity of interest.
   */
  inline void loop_project(const int nummodes, const REAL value,
                           const REAL *local_space_0, const REAL *local_space_1,
                           const REAL *local_space_2, REAL *dofs) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.loop_project_v(nummodes, value, local_space_0, local_space_1,
                              local_space_2, dofs);
  }

  /**
   * Return the number of coordinate dimensions this expansion is based. i.e.
   * if there are only two dimensions then calls to the methods in this class
   * will pass dummy values for z components.
   *
   * @returns  Number of spatial dimensions.
   */
  inline int get_ndim() {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    return underlying.get_ndim_v();
  }

  /**
   * Return the Nektar++ enumeration for the element type the implementation
   * computes evaluation and projection for.
   *
   * @returns Nektar++ shape type enumeration
   */
  inline ShapeType get_shape_type() {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    return underlying.get_shape_type_v();
  }
};

} // namespace NESO::ExpansionLooping

#endif
