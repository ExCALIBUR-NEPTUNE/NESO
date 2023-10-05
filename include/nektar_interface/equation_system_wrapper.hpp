#ifndef __EQUATION_SYSTEM_WRAPPER_HPP_
#define __EQUATION_SYSTEM_WRAPPER_HPP_

#include <SolverUtils/EquationSystem.h>
using namespace Nektar;
#include "utilities.hpp"

namespace NESO {

/**
 *  Helper class to create Nektar++ fields from a session and graph. Also
 *  provides FLD output functionality.
 */
class EquationSystemWrapper : public SolverUtils::EquationSystem {

public:
  /// Map from field string name to integer index.
  NektarFieldIndexMap field_to_index;

  /**
   * Create and initialise new wrapper.
   *
   * @param session Session instance to use.
   * @param graph MeshGraph instance to use.
   */
  EquationSystemWrapper(const LibUtilities::SessionReaderSharedPtr &session,
                        const SpatialDomains::MeshGraphSharedPtr &graph)
      : EquationSystem(session, graph),
        field_to_index(session->GetVariables()) {
    this->InitObject();
  };

  /**
   *  Replace the field stored in this index with a different field (ExpList).
   *
   *  @param field_name Name of field to replace. Must exist in this instance.
   *  @param field ExpList to replace the field with.
   */
  template <typename T>
  inline void set_field(std::string field_name, std::shared_ptr<T> field) {
    const int idx = this->field_to_index.get_idx(field_name);
    this->m_fields[idx] = std::dynamic_pointer_cast<ExpList>(field);
  }
};

} // namespace NESO

#endif
