#ifndef __EXPANSION_LOOPING_QUADRILATERAL_H__
#define __EXPANSION_LOOPING_QUADRILATERAL_H__

#include "expansion_looping_interface.hpp"

namespace NESO::ExpansionLooping {

/**
 * TODO
 */
struct Quadrilateral : ExpansionLoopingInterface<Quadrilateral> {

  inline void loc_coord_to_loc_collapsed_v(const REAL xi0, const REAL xi1,
                                           const REAL xi2, REAL *eta0,
                                           REAL *eta1, REAL *eta2) {
    GeometryInterface::Quadrilateral geom{};
    geom.loc_coord_to_loc_collapsed(xi0, xi1, eta0, eta1);
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

    BasisJacobi::ModifiedA::evaluate(numnodes, z, data_int, data_real0,
                                     data_real1, data_real2, output);
  }
  inline void evaluate_basis_2_v(const int numnodes, const REAL z,
                                 const int data_int, const REAL *data_real0,
                                 const REAL *data_real1, const REAL *data_real2,
                                 REAL *output) {}

  inline void loop_evaluate_v(const int nummodes, const REAL *const dofs,
                              const REAL *const local_space_0,
                              const REAL *const local_space_1,
                              const REAL *const local_space_2, REAL *output) {
    int mode = 0;
    REAL evaluation = 0.0;
    for (int qx = 0; qx < nummodes; qx++) {
      for (int px = 0; px < nummodes; px++) {
        const int mode = qx * nummodes + px;
        const REAL coeff = dofs[mode];
        const REAL etmp0 = local_space_0[px];
        const REAL etmp1 = local_space_1[qx];
        evaluation += coeff * etmp0 * etmp1;
      }
    }
    *output = evaluation;
  }

  inline void loop_project_v(const int nummodes, const REAL value,
                             const REAL *const local_space_0,
                             const REAL *const local_space_1,
                             const REAL *const local_space_2, REAL *dofs) {

    for (int qx = 0; qx < nummodes; qx++) {
      for (int px = 0; px < nummodes; px++) {
        const int mode = qx * nummodes + px;
        const REAL etmp0 = local_space_0[px];
        const REAL etmp1 = local_space_1[qx];

        const REAL evaluation = value * etmp0 * etmp1;
        sycl::atomic_ref<REAL, sycl::memory_order::relaxed,
                         sycl::memory_scope::device>
            coeff_atomic_ref(dofs[mode]);
        coeff_atomic_ref.fetch_add(evaluation);
      }
    }
  }

  inline ShapeType get_shape_type_v() { return eQuadrilateral; }

  inline int get_ndim_v() { return 2; }
};

} // namespace NESO::ExpansionLooping

#endif
