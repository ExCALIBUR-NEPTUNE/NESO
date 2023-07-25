#ifndef __FUNCTION_COUPLING_BASE_H_
#define __FUNCTION_COUPLING_BASE_H_
#include "nektar_interface/basis_reference.hpp"
#include "nektar_interface/expansion_looping/jacobi_coeff_mod_basis.hpp"
#include "nektar_interface/particle_interface.hpp"
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

} // namespace NESO

#endif
