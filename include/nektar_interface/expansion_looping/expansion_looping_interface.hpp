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

namespace NESO {
namespace ExpansionLooping {

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
    underlying.template evaluate_basis_0_v(numnodes, z, data_int, data_real0,
                                           data_real1, data_real2, output);
  }

  /**
   *  TODO
   */
  inline void evaluate_basis_1(const int numnodes, const REAL z,
                               const int data_int, const REAL *data_real0,
                               const REAL *data_real1, const REAL *data_real2,
                               REAL *output) {

    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.template evaluate_basis_1_v(numnodes, z, data_int, data_real0,
                                           data_real1, data_real2, output);
  }

  /**
   *  TODO
   */
  inline void evaluate_basis_2(const int numnodes, const REAL z,
                               const int data_int, const REAL *data_real0,
                               const REAL *data_real1, const REAL *data_real2,
                               REAL *output) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.template evaluate_basis_2_v(numnodes, z, data_int, data_real0,
                                           data_real1, data_real2, output);
  }

  /**
   * TODO
   */
  inline void loop_evaluate(const int nummodes, const REAL *dofs,
                            const REAL *local_space_0,
                            const REAL *local_space_1,
                            const REAL *local_space_2, REAL *output) {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    underlying.template loop_evaluate_v(nummodes, dofs, local_space_0,
                                        local_space_1, local_space_2, output);
  }

  /**
   * TODO
   */
  inline int get_ndim() {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    return underlying.template get_ndim_v();
  }

  /**
   * TODO
   */
  inline ShapeType get_shape_type() {
    auto &underlying = static_cast<SPECIALISATION &>(*this);
    return underlying.template get_shape_type_v();
  }
};

/**
 * TODO
 */
struct Triangle : ExpansionLoopingInterface<Triangle> {

  inline void loc_coord_to_loc_collapsed_v(const REAL xi0, const REAL xi1,
                                           const REAL xi2, REAL *eta0,
                                           REAL *eta1, REAL *eta2) {
    GeometryInterface::Triangle geom{};
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

    BasisJacobi::ModifiedB::evaluate(numnodes, z, data_int, data_real0,
                                     data_real1, data_real2, output);
  }
  inline void evaluate_basis_2_v(const int numnodes, const REAL z,
                                 const int data_int, const REAL *data_real0,
                                 const REAL *data_real1, const REAL *data_real2,
                                 REAL *output) {}

  inline void loop_evaluate_v(const int nummodes, const REAL *dofs,
                              const REAL *local_space_0,
                              const REAL *local_space_1,
                              const REAL *local_space_2, REAL *output) {
    int mode = 0;
    REAL evaluation = 0.0;
    for (int px = 0; px < nummodes; px++) {
      for (int qx = 0; qx < nummodes - px; qx++) {
        const REAL coeff = dofs[mode];
        // There exists a correction for mode == 1 in the Nektar++
        // definition of this 2D basis which we apply here.
        const REAL etmp0 = (mode == 1) ? 1.0 : local_space_0[px];
        const REAL etmp1 = local_space_1[mode];
        evaluation += coeff * etmp0 * etmp1;
        mode++;
      }
    }
    *output = evaluation;
  }

  inline ShapeType get_shape_type_v() { return eTriangle; }

  inline int get_ndim_v() { return 2; }
};

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

  inline void loop_evaluate_v(const int nummodes, const REAL *dofs,
                              const REAL *local_space_0,
                              const REAL *local_space_1,
                              const REAL *local_space_2, REAL *output) {
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

  inline ShapeType get_shape_type_v() { return eQuadrilateral; }

  inline int get_ndim_v() { return 2; }
};

} // namespace ExpansionLooping
} // namespace NESO

#endif
