///////////////////////////////////////////////////////////////////////////////
//
// File: main.cpp
//
//
// Description: Entrypoint for the Diffusion solver.
//
///////////////////////////////////////////////////////////////////////////////
#include "Diffusion.hpp"
#include <iostream>
#include <mpi.h>

int main(int argc, char *argv[]) {

  // MPI is initialised/finalised here to ensure that Nektar++ does not
  // initialise/finalise MPI when we were not expecting it to.
  int provided_thread_level;
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED,
                      &provided_thread_level) != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Init != MPI_SUCCESS" << std::endl;
    return -1;
  }
  int err = NESO::Solvers::run_Diffusion(argc, argv);
  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Finalize != MPI_SUCCESS" << std::endl;
    return -1;
  }

  return err;
}
