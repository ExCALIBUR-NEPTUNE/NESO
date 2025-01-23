#ifndef __TIME_EVOLVED_EQNSYS_BASE_H_
#define __TIME_EVOLVED_EQNSYS_BASE_H_

#include <SolverUtils/UnsteadySystem.h>

#include "eqnsys_base.hpp"

#include <type_traits>

namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers {

/**
 * @brief Base class for time-evolving Nektar++ equation systems, based on
 * Nektar::SolverUtils::UnsteadySystem, coupled to a NESO-Particles
 * particle system derived from NESO::Solvers::PartSysBase.
 */
template <typename NEKEQNSYS, typename PARTSYS>
class TimeEvoEqnSysBase : public EqnSysBase<NEKEQNSYS, PARTSYS> {
  /// Template param must derive from Nektar's UnsteadySystem class
  static_assert(std::is_base_of<SU::UnsteadySystem, NEKEQNSYS>(),
                "Template arg to TimeEvoEqnSysBase must derive from "
                "Nektar::SolverUtils::UnsteadySystem");

protected:
  TimeEvoEqnSysBase(const LU::SessionReaderSharedPtr &session,
                    const SD::MeshGraphSharedPtr &graph)
      : EqnSysBase<NEKEQNSYS, PARTSYS>(session, graph),
        int_fld_names(std::vector<std::string>()) {}

  /// Names of fields that will be time integrated
  std::vector<std::string> int_fld_names;

  /**
   * @brief Load common parameters associated with all time evolved equation
   * systems.
   */
  virtual void load_params() override {
    EqnSysBase<NEKEQNSYS, PARTSYS>::load_params();

    // No additional params yet
  };

  /** @brief Check that the names of fields identified as time-evolving are
   * valid.
   *
   */
  virtual void v_InitObject(bool create_fields) override {
    EqnSysBase<NEKEQNSYS, PARTSYS>::v_InitObject(create_fields);
    // Tell UnsteadySystem to only integrate a subset of fields in time
    this->m_intVariables.resize(this->int_fld_names.size());
    for (auto ii = 0; ii < this->int_fld_names.size(); ii++) {
      int var_idx = this->field_to_index.get_idx(this->int_fld_names[ii]);
      NESOASSERT(var_idx >= 0,
                 "Setting time integration vars - GetIntFieldNames() "
                 "returned an invalid field name.");
      this->m_intVariables[ii] = var_idx;
    }
  }
};

} // namespace NESO::Solvers
#endif