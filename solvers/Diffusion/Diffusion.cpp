///////////////////////////////////////////////////////////////////////////////
//
// File: Diffusion.cpp
//
//
// Description: Initialise and run the Diffusion solver.
//
///////////////////////////////////////////////////////////////////////////////

#include "Diffusion.hpp"
#include "solvers/solver_runner.hpp"

namespace NESO::Solvers::Diffusion {
int run(int argc, char *argv[]) {
  SolverRunner solver_runner(argc, argv);
  solver_runner.execute();
  solver_runner.finalise();
  return 0;
}

} // namespace NESO::Solvers::Diffusion
