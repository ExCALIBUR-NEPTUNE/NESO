#ifndef __TIME_EVOLVD_EQNSYS_BASE_H_
#define __TIME_EVOLVD_EQNSYS_BASE_H_

#include <SolverUtils/UnsteadySystem.h>

#include "eqnsys_base.hpp"

#include <type_traits>

namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers {

template <typename NEKEQNSYS>
class TimeEvoEqnSysBase : public EqnSysBase<NEKEQNSYS> {
  // Template param must derive from Nektar's UnsteadySystem class
  static_assert(std::is_base_of<SU::UnsteadySystem, NEKEQNSYS>(),
                "Template arg to TimeEvoEqnSysBase must derive from "
                "Nektar::SolverUtils::UnsteadySystem");

public:
protected:
  TimeEvoEqnSysBase(const LU::SessionReaderSharedPtr &session,
                    const SD::MeshGraphSharedPtr &graph)
      : EqnSysBase<NEKEQNSYS>(session, graph),
        m_int_fld_names(std::vector<std::string>()) {}
  /// Names of fields that will be time integrated
  std::vector<std::string> m_int_fld_names;

  virtual void load_params() override { EqnSysBase<NEKEQNSYS>::load_params(); };

private:
};

} // namespace NESO::Solvers
#endif