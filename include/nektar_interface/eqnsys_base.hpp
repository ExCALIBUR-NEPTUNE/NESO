#ifndef __EQNSYS_BASE_H_
#define __EQNSYS_BASE_H_

#include <SolverUtils/EquationSystem.h>

#include <type_traits>

namespace SU = Nektar::SolverUtils;

namespace NESO::Solvers {

// Fwd declare NEKEQNSYS as a class
class NEKEQNSYS;

template <typename NEKEQNSYS> class EqnSysBase : virtual public NEKEQNSYS {
  // Template param must derive from Nektar's EquationSystem base class
  static_assert(std::is_base_of<SU::EquationSystem, NEKEQNSYS>(),
                "Template arg to EqnSysBase must derive from "
                "Nektar::SolverUtils::EquationSystem");

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