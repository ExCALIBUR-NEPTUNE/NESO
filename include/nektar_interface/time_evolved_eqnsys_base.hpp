#ifndef __TIME_EVOLVD_EQNSYS_BASE_H_
#define __TIME_EVOLVD_EQNSYS_BASE_H_

#include <SolverUtils/UnsteadySystem.h>

#include "eqnsys_base.hpp"

#include <type_traits>

namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers {

template <typename NEKEQNSYS>
class TimeEvoEqnSysBase : virtual public EqnSysBase<NEKEQNSYS> {
  // Template param must derive from Nektar's UnsteadySystem class
  static_assert(std::is_base_of<SU::UnsteadySystem, NEKEQNSYS>(),
                "Template arg to TimeEvoEqnSysBase must derive from "
                "Nektar::SolverUtils::UnsteadySystem");

public:
  virtual void pub_func() = 0;

protected:
  virtual void prot_func() {
    std::cout << "Protected function call." << std::endl;
  }

private:
  virtual void priv_func() {
    std::cout << "Private function call." << std::endl;
  }
};

} // namespace NESO::Solvers
#endif