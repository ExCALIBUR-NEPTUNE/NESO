#ifndef __EXPANSION_LOOPING_PYRAMID_H__
#define __EXPANSION_LOOPING_PYRAMID_H__

#include "jacobi_expansion_looping_interface.hpp"

namespace NESO::ExpansionLooping {

/**
 * Implements evaluation and projection for Pyramid elements with
 * eModified_A/A/PyrC basis functions.
 */
struct Pyramid : JacobiExpansionLoopingInterface<Pyramid> {

  inline void loc_coord_to_loc_collapsed_v(const REAL xi0, const REAL xi1,
                                           const REAL xi2, REAL *eta0,
                                           REAL *eta1, REAL *eta2) {
    GeometryInterface::Pyramid geom{};
    geom.loc_coord_to_loc_collapsed(xi0, xi1, xi2, eta0, eta1, eta2);
  }

  inline void evaluate_basis_0_v(const int nummodes, const REAL z,
                                 const int coeffs_stride,
                                 const REAL *coeffs_pnm10,
                                 const REAL *coeffs_pnm11,
                                 const REAL *coeffs_pnm2, REAL *output) {

    BasisJacobi::ModifiedA::evaluate(nummodes, z, coeffs_stride, coeffs_pnm10,
                                     coeffs_pnm11, coeffs_pnm2, output);
  }
  inline void evaluate_basis_1_v(const int nummodes, const REAL z,
                                 const int coeffs_stride,
                                 const REAL *coeffs_pnm10,
                                 const REAL *coeffs_pnm11,
                                 const REAL *coeffs_pnm2, REAL *output) {

    BasisJacobi::ModifiedA::evaluate(nummodes, z, coeffs_stride, coeffs_pnm10,
                                     coeffs_pnm11, coeffs_pnm2, output);
  }
  inline void evaluate_basis_2_v(const int nummodes, const REAL z,
                                 const int coeffs_stride,
                                 const REAL *coeffs_pnm10,
                                 const REAL *coeffs_pnm11,
                                 const REAL *coeffs_pnm2, REAL *output) {
    BasisJacobi::ModifiedPyrC::evaluate(nummodes, z, coeffs_stride,
                                        coeffs_pnm10, coeffs_pnm11, coeffs_pnm2,
                                        output);
  }

  inline void loop_evaluate_v(const int nummodes, const REAL *const dofs,
                              const REAL *const local_space_0,
                              const REAL *const local_space_1,
                              const REAL *const local_space_2, REAL *output) {
    REAL evaluation = 0.0;
    int mode = 0;
    for (int p = 0; p < nummodes; p++) {
      const REAL etmp0 = local_space_0[p];
      for (int q = 0; q < nummodes; q++) {
        const REAL etmp1 = local_space_1[q];
        const int l = std::max(p, q);
        for (int r = 0; r < nummodes - l; r++) {
          const REAL etmp2 = local_space_2[mode];
          const REAL coeff = dofs[mode];
          if (mode == 1) {
            evaluation += coeff * etmp2;
          } else {
            evaluation += coeff * etmp0 * etmp1 * etmp2;
          }
          mode++;
        }
      }
    }
    *output = evaluation;
  }

  inline void loop_project_v(const int nummodes, const REAL value,
                             const REAL *const local_space_0,
                             const REAL *const local_space_1,
                             const REAL *const local_space_2, REAL *dofs) {
    int mode = 0;
    for (int p = 0; p < nummodes; p++) {
      const REAL etmp0 = local_space_0[p];
      for (int q = 0; q < nummodes; q++) {
        const REAL etmp1 = local_space_1[q];
        const int l = std::max(p, q);
        for (int r = 0; r < nummodes - l; r++) {
          const REAL etmp2 = local_space_2[mode];
          REAL evaluation;
          if (mode == 1) {
            evaluation = value * etmp2;
          } else {
            evaluation = value * etmp0 * etmp1 * etmp2;
          }
          sycl::atomic_ref<REAL, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              coeff_atomic_ref(dofs[mode]);
          coeff_atomic_ref.fetch_add(evaluation);
          mode++;
        }
      }
    }
  }

  inline ShapeType get_shape_type_v() { return ePyramid; }

  inline int get_ndim_v() { return 3; }
};

} // namespace NESO::ExpansionLooping

#endif
