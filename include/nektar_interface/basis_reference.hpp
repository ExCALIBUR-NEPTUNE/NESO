#ifndef __BASIS_REFERENCE_H_
#define __BASIS_REFERENCE_H_
#include <cstdlib>
#include <map>
#include <memory>
#include <neso_particles.hpp>

#include <StdRegions/StdExpansion2D.h>

#include "geometry_transport/shape_mapping.hpp"
#include "special_functions.hpp"

using namespace NESO::Particles;
using namespace Nektar;
using namespace Nektar::LibUtilities;
using namespace Nektar::LocalRegions;
using namespace Nektar::StdRegions;

#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tgmath.h>
#include <tuple>

namespace NESO::BasisReference {

/**
 *  Reference implementation to compute eModified_A at an order p and point z.
 *
 *  @param p Polynomial order.
 *  @param z Point in [-1, 1] to evaluate at.
 *  @returns Basis function evaluated at point.
 */
double eval_modA_i(const int p, const double z);

/**
 *  Reference implementation to compute eModified_B at an order p,q and point z.
 *
 *  @param p First index for basis.
 *  @param q Second index for basis.
 *  @param z Point in [-1, 1] to evaluate at.
 *  @returns Basis function evaluated at point.
 */
double eval_modB_ij(const int p, const int q, const double z);

/**
 *  Reference implementation to compute eModified_C at an order p,q,r and point
 * z.
 *
 *  @param p First index for basis.
 *  @param q Second index for basis.
 *  @param r Third index for basis.
 *  @param z Point in [-1, 1] to evaluate at.
 *  @returns Basis function evaluated at point.
 */
double eval_modC_ijk(const int p, const int q, const int r, const double z);

/**
 *  Reference implementation to compute eModifiedPyr_C at an order p,q,r and
 * point z.
 *
 *  @param p First index for basis.
 *  @param q Second index for basis.
 *  @param r Third index for basis.
 *  @param z Point in [-1, 1] to evaluate at.
 *  @returns Basis function evaluated at point.
 */
double eval_modPyrC_ijk(const int p, const int q, const int r, const double z);

/**
 * Get the total number of modes in a given basis for a given number of input
 * modes. See Nektar GetTotNumModes.
 *
 * @param basis_type Basis type to query number of values for.
 * @param P Number of modes, i.e. Nektar GetNumModes();
 * @returns Total number of values required to represent the basis with the
 * given number of modes.
 */
int get_total_num_modes(const BasisType basis_type, const int P);

/**
 *  Reference implementation to compute eModified_A for order P-1 and point z.
 *
 *  @param[in] P Number of modes to compute.
 *  @param[in] z Point in [-1, 1] to evaluate at.
 *  @param[in, out] b Basis functions evaluated at z for each mode.
 */
void eval_modA(const int P, const double z, std::vector<double> &b);

/**
 *  Reference implementation to compute eModified_B for order P-1 and point z.
 *
 *  @param[in] P Number of modes to compute.
 *  @param[in] z Point in [-1, 1] to evaluate at.
 *  @param[in, out] b Basis functions evaluated at z for each mode.
 */
void eval_modB(const int P, const double z, std::vector<double> &b);

/**
 *  Reference implementation to compute eModified_C for order P-1 and point z.
 *
 *  @param[in] P Number of modes to compute.
 *  @param[in] z Point in [-1, 1] to evaluate at.
 *  @param[in, out] b Basis functions evaluated at z for each mode.
 */
void eval_modC(const int P, const double z, std::vector<double> &b);

/**
 *  Reference implementation to compute eModifiedPyr_C for order P-1 and point
 *  z.
 *
 *  @param[in] P Number of modes to compute.
 *  @param[in] z Point in [-1, 1] to evaluate at.
 *  @param[in, out] b Basis functions evaluated at z for each mode.
 */
void eval_modPyrC(const int P, const double z, std::vector<double> &b);

/**
 *  Reference implementation to compute a modified basis for order P-1 and
 *  point z.
 *
 *  @param[in] basis_type Basis type to compute.
 *  @param[in] P Number of modes to compute.
 *  @param[in] z Point in [-1, 1] to evaluate at.
 *  @param[in, out] b Basis functions evaluated at z for each mode.
 */
void eval_basis(const BasisType basis_type, const int P, const double z,
                std::vector<double> &b);

/**
 * Get the total number of modes in a given shape for a given number of input
 * modes.
 *
 * @param[in] shape_type Shape type to query number of values for.
 * @param[in] P Number of modes, i.e. Nektar GetNumModes(), in each dimension;
 * @param[in, out] max_n (optional) Get the maximum Jacobi polynomial order
 * required.
 * @param[in, out] max_alpha (optional) Get the maximum Jacobi alpha value
 * required.
 * @returns Total number of values required to represent the basis with the
 * given number of modes.
 */
int get_total_num_modes(const ShapeType shape_type, const int P,
                        int *max_n = nullptr, int *max_alpha = nullptr);

/**
 *  Evaluate all the basis function modes for a geometry object with P modes
 * in each coordinate direction using calls to eval_modA, ..., eval_modPyrC.
 *
 *  @param[in] shape_type Geometry shape type to compute modes for, e.g.
 * eHexahedron.
 *  @param[in] P Number of modes in each dimesion.
 *  @param[in] eta0 Evaluation point, first dimension.
 *  @param[in] eta1 Evaluation point, second dimension.
 *  @param[in] eta2 Evaluation point, third dimension.
 *  @param[in, out] b Output vector of mode evaluations.
 */
void eval_modes(const LibUtilities::ShapeType shape_type, const int P,
                const double eta0, const double eta1, const double eta2,
                std::vector<double> &b);

} // namespace NESO::BasisReference

#endif
