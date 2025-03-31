///////////////////////////////////////////////////////////////////////////////
//
// File: main.cpp
//
//
// Description: Common entrypoint for solvers.
//
///////////////////////////////////////////////////////////////////////////////
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

  int solver_ret_code = run_solver(argc, argv);

  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Finalize != MPI_SUCCESS" << std::endl;
    return -1;
  }

  return solver_ret_code;
}
