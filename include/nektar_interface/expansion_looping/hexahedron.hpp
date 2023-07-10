#ifndef __EXPANSION_LOOPING_HEXAHEDRON_H__
#define __EXPANSION_LOOPING_HEXAHEDRON_H__

#include "expansion_looping_interface.hpp"

namespace NESO::ExpansionLooping {

/**
 * TODO
 */
struct Hexahedron : ExpansionLoopingInterface<Hexahedron> {

  inline void loc_coord_to_loc_collapsed_v(const REAL xi0, const REAL xi1,
                                           const REAL xi2, REAL *eta0,
                                           REAL *eta1, REAL *eta2) {
    GeometryInterface::Hexahedron geom{};
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

    BasisJacobi::ModifiedA::evaluate(numnodes, z, data_int, data_real0,
                                     data_real1, data_real2, output);
  }
  inline void evaluate_basis_2_v(const int numnodes, const REAL z,
                                 const int data_int, const REAL *data_real0,
                                 const REAL *data_real1, const REAL *data_real2,
                                 REAL *output) {
    BasisJacobi::ModifiedA::evaluate(numnodes, z, data_int, data_real0,
                                     data_real1, data_real2, output);
  }

  inline void loop_evaluate_v(const int nummodes, const REAL *const dofs,
                              const REAL *const local_space_0,
                              const REAL *const local_space_1,
                              const REAL *const local_space_2, REAL *output) {
    REAL evaluation = 0.0;
    for (int rx = 0; rx < nummodes; rx++) {
      const int mode_r = rx * nummodes * nummodes;
      const REAL etmp2 = local_space_2[rx];
      for (int qx = 0; qx < nummodes; qx++) {
        const int mode_q = qx * nummodes + mode_r;
        const REAL etmp1 = local_space_1[qx] * etmp2;
        for (int px = 0; px < nummodes; px++) {
          const int mode = px + mode_q;
          const REAL coeff = dofs[mode];
          const REAL etmp0 = local_space_0[px];
          evaluation += coeff * etmp0 * etmp1;
        }
      }
    }
    *output = evaluation;
  }

  inline void loop_project_v(const int nummodes, const REAL value,
                             const REAL *const local_space_0,
                             const REAL *const local_space_1,
                             const REAL *const local_space_2, REAL *dofs) {

    for (int rx = 0; rx < nummodes; rx++) {
      const int mode_r = rx * nummodes * nummodes;
      const REAL etmp2 = local_space_2[rx] * value;
      for (int qx = 0; qx < nummodes; qx++) {
        const int mode_q = qx * nummodes + mode_r;
        const REAL etmp1 = local_space_1[qx] * etmp2;
        for (int px = 0; px < nummodes; px++) {
          const int mode = px + mode_q;
          const REAL evaluation = local_space_0[px] * etmp1;
          sycl::atomic_ref<REAL, sycl::memory_order::relaxed,
                           sycl::memory_scope::device>
              coeff_atomic_ref(dofs[mode]);
          coeff_atomic_ref.fetch_add(evaluation);
        }
      }
    }
  }

  inline ShapeType get_shape_type_v() { return eHexahedron; }

  inline int get_ndim_v() { return 3; }
};

} // namespace NESO::ExpansionLooping

#endif
