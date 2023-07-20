#ifndef __EXPANSION_LOOPING_INTERFACE_H__
#define __EXPANSION_LOOPING_INTERFACE_H__

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
 * TODO
 */
template <typename SPECIALISATION> struct ExpansionLoopingInterface {

  /**
   * TODO
   */
  inline void loc_coord_to_loc_collapsed(const REAL xi0, const REAL xi1,
                                         const REAL xi2, REAL *eta0, REAL *eta1,
                                         REAL *eta2) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.loc_coord_to_loc_collapsed_v(xi0, xi1, xi2, eta0, eta1, eta2);
  }

  /**
   *  TODO
   */
  inline void evaluate_basis_0(const int numnodes, const REAL z,
                               const int data_int, const REAL *data_real0,
                               const REAL *data_real1, const REAL *data_real2,
                               REAL *output) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.evaluate_basis_0_v(numnodes, z, data_int, data_real0, data_real1,
                                  data_real2, output);
  }

  /**
   *  TODO
   */
  inline void evaluate_basis_1(const int numnodes, const REAL z,
                               const int data_int, const REAL *data_real0,
                               const REAL *data_real1, const REAL *data_real2,
                               REAL *output) {

    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.evaluate_basis_1_v(numnodes, z, data_int, data_real0, data_real1,
                                  data_real2, output);
  }

  /**
   *  TODO
   */
  inline void evaluate_basis_2(const int numnodes, const REAL z,
                               const int data_int, const REAL *data_real0,
                               const REAL *data_real1, const REAL *data_real2,
                               REAL *output) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.evaluate_basis_2_v(numnodes, z, data_int, data_real0, data_real1,
                                  data_real2, output);
  }

  /**
   * TODO
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
   * TODO
   */
  inline void loop_project(const int nummodes, const REAL value,
                           const REAL *local_space_0, const REAL *local_space_1,
                           const REAL *local_space_2, REAL *dofs) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.loop_project_v(nummodes, value, local_space_0, local_space_1,
                              local_space_2, dofs);
  }

  /**
   * TODO
   */
  inline int get_ndim() {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    return underlying.get_ndim_v();
  }

  /**
   * TODO
   */
  inline ShapeType get_shape_type() {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    return underlying.get_shape_type_v();
  }
};

} // namespace NESO::ExpansionLooping

#endif
