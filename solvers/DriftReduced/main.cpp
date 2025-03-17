///////////////////////////////////////////////////////////////////////////////
//
// File: main.cpp
//
//
// Description: Entrypoint for the DriftReduced solver.
//
///////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <mpi.h>

#include "solvers/solver_runner.hpp"

int main(int argc, char *argv[]) {

  // MPI is initialised/finalised here to ensure that Nektar++ does not
  // initialise/finalise MPI when we were not expecting it to.
  int provided_thread_level;
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED,
                      &provided_thread_level) != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Init != MPI_SUCCESS" << std::endl;
    return -1;
  }
  SolverRunner solver_runner(argc, argv);
  solver_runner.execute();
  solver_runner.finalise();
  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Finalize != MPI_SUCCESS" << std::endl;
    return -1;
  }

  return 0;
}
