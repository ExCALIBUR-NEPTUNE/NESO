#include <nektar_interface/basis_reference.hpp>

namespace NESO::BasisReference {

/**
 *  Reference implementation to compute eModified_A at an order p and point z.
 *
 *  @param p Polynomial order.
 *  @param z Point in [-1, 1] to evaluate at.
 *  @returns Basis function evaluated at point.
 */
double eval_modA_i(const int p, const double z) {
  const double b0 = 0.5 * (1.0 - z);
  const double b1 = 0.5 * (1.0 + z);
  if (p == 0) {
    return b0;
  }
  if (p == 1) {
    return b1;
  }
  return b0 * b1 * jacobi(p - 2, z, 1, 1);
}

/**
 *  Reference implementation to compute eModified_B at an order p,q and point z.
 *
 *  @param p First index for basis.
 *  @param q Second index for basis.
 *  @param z Point in [-1, 1] to evaluate at.
 *  @returns Basis function evaluated at point.
 */
double eval_modB_ij(const int p, const int q, const double z) {

  double output;

  if (p == 0) {
    output = eval_modA_i(q, z);
  } else if (q == 0) {
    output = std::pow(0.5 * (1.0 - z), p);
  } else {
    output = std::pow(0.5 * (1.0 - z), p) * 0.5 * (1.0 + z) *
             jacobi(q - 1, z, 2 * p - 1, 1);
  }
  return output;
}

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
double eval_modC_ijk(const int p, const int q, const int r, const double z) {
  return eval_modB_ij(p + q, r, z);
}

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
double eval_modPyrC_ijk(const int p, const int q, const int r, const double z) {
  if (p == 0) {
    return eval_modB_ij(q, r, z);
  } else if (p == 1) {
    if (q == 0) {
      return eval_modB_ij(1, r, z);
    } else {
      return eval_modB_ij(q, r, z);
    }
  } else {
    if (q < 2) {
      return eval_modB_ij(p, r, z);
    } else {
      if (r == 0) {
        return std::pow(0.5 * (1.0 - z), p + q - 2);
      } else {
        return std::pow(0.5 * (1.0 - z), p + q - 2) * (0.5 * (1.0 + z)) *
               jacobi(r - 1, z, 2 * p + 2 * q - 3, 1);
      }
    }
  }
}

/**
 * Get the total number of modes in a given basis for a given number of input
 * modes. See Nektar GetTotNumModes.
 *
 * @param basis_type Basis type to query number of values for.
 * @param P Number of modes, i.e. Nektar GetNumModes();
 * @returns Total number of values required to represent the basis with the
 * given number of modes.
 */
int get_total_num_modes(const BasisType basis_type, const int P) {
  if (basis_type == eModified_A) {
    return P;
  } else if (basis_type == eModified_B) {
    int mode = 0;
    for (int p = 0; p < P; p++) {
      for (int q = 0; q < (P - p); q++) {
        mode++;
      }
    }
    return mode;
  } else if (basis_type == eModified_C) {
    int mode = 0;
    for (int p = 0; p < P; p++) {
      for (int q = 0; q < (P - p); q++) {
        for (int r = 0; r < (P - p - q); r++) {
          mode++;
        }
      }
    }
    return mode;
  } else if (basis_type == eModifiedPyr_C) {
    int mode = 0;
    for (int p = 0; p < P; p++) {
      for (int q = 0; q < P; q++) {
        for (int r = 0; r < (P - std::max(p, q)); r++) {
          mode++;
        }
      }
    }
    return mode;
  } else {
    NESOASSERT(false, "unknown basis type");
    return -1;
  }
}

/**
 *  Reference implementation to compute eModified_A for order P-1 and point z.
 *
 *  @param[in] P Number of modes to compute.
 *  @param[in] z Point in [-1, 1] to evaluate at.
 *  @param[in, out] b Basis functions evaluated at z for each mode.
 */
void eval_modA(const int P, const double z, std::vector<double> &b) {
  NESOASSERT(b.size() >= get_total_num_modes(eModified_A, P),
             "Output vector too small - see get_total_num_modes.");
  for (int p = 0; p < P; p++) {
    b.at(p) = eval_modA_i(p, z);
  }
}

/**
 *  Reference implementation to compute eModified_B for order P-1 and point z.
 *
 *  @param[in] P Number of modes to compute.
 *  @param[in] z Point in [-1, 1] to evaluate at.
 *  @param[in, out] b Basis functions evaluated at z for each mode.
 */
void eval_modB(const int P, const double z, std::vector<double> &b) {
  NESOASSERT(b.size() >= get_total_num_modes(eModified_B, P),
             "Output vector too small - see get_total_num_modes.");
  int mode = 0;
  for (int p = 0; p < P; p++) {
    for (int q = 0; q < (P - p); q++) {
      b.at(mode) = eval_modB_ij(p, q, z);
      mode++;
    }
  }
}

/**
 *  Reference implementation to compute eModified_C for order P-1 and point z.
 *
 *  @param[in] P Number of modes to compute.
 *  @param[in] z Point in [-1, 1] to evaluate at.
 *  @param[in, out] b Basis functions evaluated at z for each mode.
 */
void eval_modC(const int P, const double z, std::vector<double> &b) {
  NESOASSERT(b.size() >= get_total_num_modes(eModified_C, P),
             "Output vector too small - see get_total_num_modes.");
  int mode = 0;
  for (int p = 0; p < P; p++) {
    for (int q = 0; q < (P - p); q++) {
      for (int r = 0; r < (P - p - q); r++) {
        b.at(mode) = eval_modC_ijk(p, q, r, z);
        mode++;
      }
    }
  }
}

/**
 *  Reference implementation to compute eModifiedPyr_C for order P-1 and point
 *  z.
 *
 *  @param[in] P Number of modes to compute.
 *  @param[in] z Point in [-1, 1] to evaluate at.
 *  @param[in, out] b Basis functions evaluated at z for each mode.
 */
void eval_modPyrC(const int P, const double z, std::vector<double> &b) {
  NESOASSERT(b.size() >= get_total_num_modes(eModifiedPyr_C, P),
             "Output vector too small - see get_total_num_modes.");
  int mode = 0;
  for (int p = 0; p < P; p++) {
    for (int q = 0; q < P; q++) {
      for (int r = 0; r < (P - std::max(p, q)); r++) {
        b.at(mode) = eval_modPyrC_ijk(p, q, r, z);
        mode++;
      }
    }
  }
}

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
                std::vector<double> &b) {
  if (basis_type == eModified_A) {
    return eval_modA(P, z, b);
  } else if (basis_type == eModified_B) {
    return eval_modB(P, z, b);
  } else if (basis_type == eModified_C) {
    return eval_modC(P, z, b);
  } else if (basis_type == eModifiedPyr_C) {
    return eval_modPyrC(P, z, b);
  } else {
    NESOASSERT(false, "unknown basis type");
  }
}

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
int get_total_num_modes(const ShapeType shape_type, const int P, int *max_n,
                        int *max_alpha) {

  int n = 0;
  int alpha = 0;

  auto lambda_A = [&](const int p) {
    n = std::max(n, p - 2);
    alpha = std::max(alpha, 1);
  };
  auto lambda_B = [&](const int p, const int q) {
    n = std::max(n, q - 1);
    alpha = std::max(alpha, 2 * p - 1);
  };
  auto lambda_C = [&](const int p, const int q, const int r) {
    lambda_B(p + q, r);
  };
  auto lambda_PyrC = [&](const int p, const int q, const int r) {
    lambda_B(q, r);
    lambda_B(1, r);
    lambda_B(p, r);
    n = std::max(n, r - 1);
    alpha = std::max(alpha, 2 * p + 2 * q - 3);
  };

  int num_modes = -1;
  if (shape_type == eSegment) {
    num_modes = P;
    lambda_A(P - 1);
  } else if (shape_type == eTriangle) {
    int mode = 0;
    for (int p = 0; p < P; p++) {
      for (int q = 0; q < P - p; q++) {
        lambda_A(p);
        lambda_B(p, q);
        mode++;
      }
    }
    num_modes = mode;
  } else if (shape_type == eQuadrilateral) {
    num_modes = P * P;
    lambda_A(P - 1);
  } else if (shape_type == eHexahedron) {
    num_modes = P * P * P;
    lambda_A(P - 1);
  } else if (shape_type == ePyramid) {
    int mode = 0;
    for (int p = 0; p < P; ++p) {
      for (int q = 0; q < P; ++q) {
        int maxpq = std::max(p, q);
        for (int r = 0; r < P - maxpq; ++r) {
          lambda_A(p);
          lambda_A(q);
          lambda_PyrC(p, q, r);
          mode++;
        }
      }
    }
    num_modes = mode;
  } else if (shape_type == ePrism) {
    int mode = 0;
    for (int p = 0; p < P; p++) {
      for (int q = 0; q < P; q++) {
        for (int r = 0; r < (P - p); r++) {
          lambda_A(p);
          lambda_A(q);
          lambda_B(p, r);
          mode++;
        }
      }
    }
    num_modes = mode;
  } else if (shape_type == eTetrahedron) {
    int mode = 0;
    for (int p = 0; p < P; p++) {
      for (int q = 0; q < (P - p); q++) {
        for (int r = 0; r < (P - p - q); r++) {
          lambda_A(p);
          lambda_B(p, q);
          lambda_C(p, q, r);
          mode++;
        }
      }
    }
    num_modes = mode;
  } else {
    NESOASSERT(false, "unknown shape type.");
  }

  if (max_n != nullptr) {
    *max_n = n;
  }
  if (max_alpha != nullptr) {
    *max_alpha = alpha;
  }

  return num_modes;
}

/**
 *  Evaluate all the basis function modes for a geometry object with P modes in
 * each coordinate direction using calls to eval_modA, ..., eval_modPyrC.
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
                std::vector<double> &b) {

  NESOASSERT(b.size() >= get_total_num_modes(shape_type, P),
             "Output vector too small - see get_total_num_modes.");

  if (shape_type == eTriangle) {
    int mode = 0;
    for (int px = 0; px < P; px++) {
      for (int qx = 0; qx < P - px; qx++) {
        const REAL etmp0 = (mode == 1) ? 1.0 : eval_modA_i(px, eta0);
        const REAL etmp1 = eval_modB_ij(px, qx, eta1);
        b[mode] = etmp0 * etmp1;
        mode++;
      }
    }
  } else if (shape_type == eQuadrilateral) {
    int mode = 0;
    for (int my = 0; my < P; my++) {
      for (int mx = 0; mx < P; mx++) {
        b[mode] = eval_modA_i(mx, eta0) * eval_modA_i(my, eta1);
        mode++;
      }
    }
  } else if (shape_type == eHexahedron) {
    int mode = 0;
    for (int mz = 0; mz < P; mz++) {
      for (int my = 0; my < P; my++) {
        for (int mx = 0; mx < P; mx++) {
          b[mode] = eval_modA_i(mx, eta0) * eval_modA_i(my, eta1) *
                    eval_modA_i(mz, eta2);
          mode++;
        }
      }
    }
  } else if (shape_type == ePyramid) {
    int mode = 0;
    for (int p = 0; p < P; p++) {
      for (int q = 0; q < P; q++) {
        for (int r = 0; r < P - std::max(p, q); r++) {
          const double contrib_0 = eval_modA_i(p, eta0);
          const double contrib_1 = eval_modA_i(q, eta1);
          const double contrib_2 = eval_modPyrC_ijk(p, q, r, eta2);
          if (mode == 1) {
            b[mode] = contrib_2;
          } else {
            b[mode] = contrib_0 * contrib_1 * contrib_2;
          }
          mode++;
        }
      }
    }
  } else if (shape_type == ePrism) {
    int mode = 0;
    int mode_pr = 0;
    for (int p = 0; p < P; p++) {
      for (int q = 0; q < P; q++) {
        for (int r = 0; r < (P - p); r++) {

          const double contrib_0 = eval_modA_i(p, eta0);
          const double contrib_1 = eval_modA_i(q, eta1);
          const double contrib_2 = eval_modB_ij(p, r, eta2);

          if ((p == 0) && (r == 1)) {
            b[mode] = contrib_1 * contrib_2;
          } else {
            b[mode] = contrib_0 * contrib_1 * contrib_2;
          }
          mode++;
        }
      }
      mode_pr += P - p;
    }
  } else if (shape_type == eTetrahedron) {
    int mode = 0;
    for (int p = 0; p < P; p++) {
      for (int q = 0; q < (P - p); q++) {
        for (int r = 0; r < (P - p - q); r++) {
          const double contrib_0 = eval_modA_i(p, eta0);
          const double contrib_1 = eval_modB_ij(p, q, eta1);
          const double contrib_2 = eval_modC_ijk(p, q, r, eta2);

          if (mode == 1) {
            b[mode] = contrib_2;
          } else if (p == 0 && q == 1) {
            b[mode] = contrib_1 * contrib_2;
          } else {
            b[mode] = contrib_0 * contrib_1 * contrib_2;
          }
          mode++;
        }
      }
    }
  } else if (shape_type == eSegment) {
    int mode = 0;
    for (int mx = 0; mx < P; mx++) {
      b[mode] = eval_modA_i(mx, eta0);
      mode++;
    }
  } else {
    NESOASSERT(false, "unknown shape type.");
  }
}

} // namespace NESO::BasisReference
