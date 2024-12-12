///////////////////////////////////////////////////////////////////////////////
//
// File: Diffusion.cpp
//
//
// Description: Driver for Diffusion.
//
///////////////////////////////////////////////////////////////////////////////

#include "Diffusion.hpp"
#include "solvers/solver_runner.hpp"

namespace NESO::Solvers {
int run_Diffusion(int argc, char *argv[]) {
  SolverRunner solver_runner(argc, argv);
  solver_runner.execute();
  solver_runner.finalise();
  return 0;
}

} // namespace NESO::Solvers
