///////////////////////////////////////////////////////////////////////////////
//
// File: H3LAPD.cpp
//
//
// Description: Runner function for H3LAPD.
//
///////////////////////////////////////////////////////////////////////////////

#include "H3LAPD.hpp"
#include "solvers/solver_runner.hpp"

namespace NESO::Solvers::H3LAPD {

int run_H3LAPD(int argc, char *argv[]) {
  try {
    SolverRunner solver_runner(argc, argv);
    solver_runner.execute();
    solver_runner.finalise();
    return 0;
  }

} // namespace NESO::Solvers::H3LAPD
