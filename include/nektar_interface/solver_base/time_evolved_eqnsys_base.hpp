#ifndef __TIME_EVOLVD_EQNSYS_BASE_H_
#define __TIME_EVOLVD_EQNSYS_BASE_H_

#include <SolverUtils/UnsteadySystem.h>

#include "eqnsys_base.hpp"

#include <type_traits>

namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers {

template <typename NEKEQNSYS, typename PARTSYS>
class TimeEvoEqnSysBase : public EqnSysBase<NEKEQNSYS, PARTSYS> {
  // Template param must derive from Nektar's UnsteadySystem class
  static_assert(std::is_base_of<SU::UnsteadySystem, NEKEQNSYS>(),
                "Template arg to TimeEvoEqnSysBase must derive from "
                "Nektar::SolverUtils::UnsteadySystem");

public:
protected:
  TimeEvoEqnSysBase(const LU::SessionReaderSharedPtr &session,
                    const SD::MeshGraphSharedPtr &graph)
      : EqnSysBase<NEKEQNSYS, PARTSYS>(session, graph),
        int_fld_names(std::vector<std::string>()) {}
  /// Names of fields that will be time integrated
  std::vector<std::string> int_fld_names;

  virtual void load_params() override {
    EqnSysBase<NEKEQNSYS, PARTSYS>::load_params();
  };

  virtual void v_InitObject(bool create_fields) {
    EqnSysBase<NEKEQNSYS, PARTSYS>::v_InitObject(create_fields);
    // Tell UnsteadySystem to only integrate a subset of fields in time
    // (Ignore fields that don't have a time derivative)
    this->m_intVariables.resize(this->int_fld_names.size());
    for (auto ii = 0; ii < this->int_fld_names.size(); ii++) {
      int var_idx = this->field_to_index.get_idx(this->int_fld_names[ii]);
      ASSERTL0(var_idx >= 0,
               "Setting time integration vars - GetIntFieldNames() "
               "returned an invalid field name.");
      this->m_intVariables[ii] = var_idx;
    }
  }
};

} // namespace NESO::Solvers
#endif