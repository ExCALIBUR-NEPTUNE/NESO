#ifndef __FUNCTION_COUPLING_BASE_H_
#define __FUNCTION_COUPLING_BASE_H_
#include "particle_interface.hpp"
#include <map>
#include <memory>
#include <neso_particles.hpp>

#include <LocalRegions/QuadExp.h>
#include <LocalRegions/TriExp.h>
#include <StdRegions/StdExpansion2D.h>

using namespace NESO::Particles;
using namespace Nektar::LocalRegions;
using namespace Nektar::StdRegions;

namespace NESO {

/**
 *  Class to provide a common method that builds the map from geometry ids to
 *  expansion ids.
 */
class GeomToExpansionBuilder {

protected:
  /**
   *  Build the map from geometry ids to expansion ids for the expansion.
   *  Nektar++ Expansions hold a reference to the geometry element they are
   *  defined over. This function assumes that map is injective and builds the
   *  inverse map from geometry id to expansion id.
   *
   *  @param field Nektar++ Expansion, e.g. ContField or DisContField.
   *  @param geom_to_exp Output map to build.
   */
  template <typename T>
  static inline void
  build_geom_to_expansion_map(std::shared_ptr<T> field,
                              std::map<int, int> &geom_to_exp) {
    // build the map from geometry ids to expansion ids
    auto expansions = field->GetExp();
    const int num_expansions = (*expansions).size();
    for (int ex = 0; ex < num_expansions; ex++) {
      auto exp = (*expansions)[ex];
      // The indexing in Nektar++ source suggests that ex is the important
      // index if these do not match in future.
      NESOASSERT(ex == exp->GetElmtId(),
                 "expected expansion id to match element id?");
      int geom_gid = exp->GetGeom()->GetGlobalID();
      geom_to_exp[geom_gid] = ex;
    }
  };
};

namespace Coordinate {

namespace Mapping {

/**
 * TODO
 */
template <typename T>
static inline void xi_to_eta(const T xi0, const T xi1, T *eta0, T *eta1) {
  const NekDouble d1_original = 1.0 - xi1;
  const bool mask_small_cond = (fabs(d1_original) < NekConstants::kNekZeroTol);
  NekDouble d1 = d1_original;

  d1 =
      (mask_small_cond && (d1 >= 0.0))
          ? NekConstants::kNekZeroTol
          : ((mask_small_cond && (d1 < 0.0)) ? -NekConstants::kNekZeroTol : d1);
  *eta0 = 2. * (1. + xi0) / d1 - 1.0;
  *eta1 = xi1;
}

/**
 * TODO
 */
template <typename T>
static inline void identity(const T xi0, const T xi1, T *eta0, T *eta1) {
  *eta0 = xi0;
  *eta1 = xi1;
}

/**
 * TODO
 */
template <typename T>
static inline void conditional_xi_to_eta(const int expansion_type,
                                         const int convert_type, const T xi0,
                                         const T xi1, T *eta0, T *eta1) {
  T eta_inner0;
  xi_to_eta(xi0, xi1, &eta_inner0, eta1);
  *eta0 = (expansion_type == convert_type) ? eta_inner0 : xi0;
}

/**
 * TODO
 */
template <typename SPECIALISATION> struct Map2D {
  template <typename T>
  static inline void map(const T xi0, const T xi1, T *eta0, T *eta1) {
    SPECIALISATION::map(xi0, xi1, eta0, eta1);
  }
};

/**
 * TODO
 */
struct MapIdentity2D : Map2D<MapIdentity2D> {
  template <typename T>
  static inline void map(const T xi0, const T xi1, T *eta0, T *eta1) {
    identity(xi0, xi1, eta0, eta1);
  }
};

/**
 * TODO
 */
struct MapXiToEta : Map2D<MapXiToEta> {
  template <typename T>
  static inline void map(const T xi0, const T xi1, T *eta0, T *eta1) {
    xi_to_eta(xi0, xi1, eta0, eta1);
  }
};

}; // namespace Mapping

} // namespace Coordinate

} // namespace NESO

#endif
