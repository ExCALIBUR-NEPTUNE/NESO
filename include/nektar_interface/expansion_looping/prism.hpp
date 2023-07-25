#ifndef __EXPANSION_LOOPING_PRISM_H__
#define __EXPANSION_LOOPING_PRISM_H__

#include "jacobi_expansion_looping_interface.hpp"

namespace NESO::ExpansionLooping {

/**
 * Implements evaluation and projection for Prism elements with eModified_A/B
 * basis functions.
 */
struct Prism : JacobiExpansionLoopingInterface<Prism> {

  inline void loc_coord_to_loc_collapsed_v(const REAL xi0, const REAL xi1,
                                           const REAL xi2, REAL *eta0,
                                           REAL *eta1, REAL *eta2) {
    GeometryInterface::Prism geom{};
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
    BasisJacobi::ModifiedB::evaluate(nummodes, z, coeffs_stride, coeffs_pnm10,
                                     coeffs_pnm11, coeffs_pnm2, output);
  }

  inline void loop_evaluate_v(const int nummodes, const REAL *const dofs,
                              const REAL *const local_space_0,
                              const REAL *const local_space_1,
                              const REAL *const local_space_2, REAL *output) {
    REAL evaluation = 0.0;
    int mode = 0;
    int mode_r = 0;
    for (int p = 0; p < nummodes; p++) {
      const REAL etmp0 = local_space_0[p];
      for (int q = 0; q < nummodes; q++) {
        const REAL etmp1 = local_space_1[q];
        for (int r = 0; r < nummodes - p; r++) {
          const REAL etmp2 = local_space_2[mode_r + r];
          REAL tmp;
          if ((p == 0) && (r == 1)) {
            tmp = etmp1 * etmp2;
          } else {
            tmp = etmp0 * etmp1 * etmp2;
          }
          const REAL coeff = dofs[mode];
          evaluation += coeff * tmp;
          mode++;
        }
      }
      mode_r += nummodes - p;
    }
    *output = evaluation;
  }

  inline void loop_project_v(const int nummodes, const REAL value,
                             const REAL *const local_space_0,
                             const REAL *const local_space_1,
                             const REAL *const local_space_2, REAL *dofs) {
    int mode = 0;
    int mode_r = 0;
    for (int p = 0; p < nummodes; p++) {
      const REAL etmp0 = local_space_0[p];
      for (int q = 0; q < nummodes; q++) {
        const REAL etmp1 = local_space_1[q];
        for (int r = 0; r < nummodes - p; r++) {
          const REAL etmp2 = local_space_2[mode_r + r];
          REAL evaluation;
          if ((p == 0) && (r == 1)) {
            evaluation = value * etmp1 * etmp2;
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
      mode_r += nummodes - p;
    }
  }

  inline ShapeType get_shape_type_v() { return ePrism; }

  inline int get_ndim_v() { return 3; }
};

} // namespace NESO::ExpansionLooping

#endif
