#ifndef __EXPANSION_LOOPING_TETRAHEDRON_H__
#define __EXPANSION_LOOPING_TETRAHEDRON_H__

#include "expansion_looping_interface.hpp"

namespace NESO::ExpansionLooping {

/**
 * TODO
 */
struct Tetrahedron : ExpansionLoopingInterface<Tetrahedron> {

  inline void loc_coord_to_loc_collapsed_v(const REAL xi0, const REAL xi1,
                                           const REAL xi2, REAL *eta0,
                                           REAL *eta1, REAL *eta2) {
    GeometryInterface::Tetrahedron geom{};
    geom.loc_coord_to_loc_collapsed(xi0, xi1, xi2, eta0, eta1, eta2);
  }

  inline void evaluate_basis_0_v(const int numnodes, const REAL z,
                                 const int data_int, const REAL *data_real0,
                                 const REAL *data_real1, const REAL *data_real2,
                                 REAL *output) {

    BasisJacobi::ModifiedA::evaluate(numnodes, z, data_int, data_real0,
                                     data_real1, data_real2, output);
  }
  inline void evaluate_basis_1_v(const int numnodes, const REAL z,
                                 const int data_int, const REAL *data_real0,
                                 const REAL *data_real1, const REAL *data_real2,
                                 REAL *output) {

    BasisJacobi::ModifiedB::evaluate(numnodes, z, data_int, data_real0,
                                     data_real1, data_real2, output);
  }
  inline void evaluate_basis_2_v(const int numnodes, const REAL z,
                                 const int data_int, const REAL *data_real0,
                                 const REAL *data_real1, const REAL *data_real2,
                                 REAL *output) {
    BasisJacobi::ModifiedC::evaluate(numnodes, z, data_int, data_real0,
                                     data_real1, data_real2, output);
  }

  inline void loop_evaluate_v(const int nummodes, const REAL *const dofs,
                              const REAL *const local_space_0,
                              const REAL *const local_space_1,
                              const REAL *const local_space_2, REAL *output) {
    REAL evaluation = 0.0;
    int mode = 0;
    int mode_q = 0;
    for (int p = 0; p < nummodes; p++) {
      const REAL etmp0 = local_space_0[p];
      for (int q = 0; q < (nummodes - p); q++) {
        const REAL etmp1 = local_space_1[mode_q];
        mode_q++;
        for (int r = 0; r < nummodes - p - q; r++) {
          const REAL etmp2 = local_space_2[mode];
          const REAL coeff = dofs[mode];
          if (mode == 1) {
            evaluation += coeff * etmp2;
          } else if (p == 0 && q == 1) {
            evaluation += coeff * etmp1 * etmp2;
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
    int mode_q = 0;
    for (int p = 0; p < nummodes; p++) {
      const REAL etmp0 = local_space_0[p];
      for (int q = 0; q < (nummodes - p); q++) {
        const REAL etmp1 = local_space_1[mode_q];
        mode_q++;
        for (int r = 0; r < nummodes - p - q; r++) {
          const REAL etmp2 = local_space_2[mode];
          REAL evaluation;
          if (mode == 1) {
            evaluation = value * etmp2;
          } else if (p == 0 && q == 1) {
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
    }
  }

  inline ShapeType get_shape_type_v() { return eTetrahedron; }

  inline int get_ndim_v() { return 3; }
};

} // namespace NESO::ExpansionLooping

#endif
