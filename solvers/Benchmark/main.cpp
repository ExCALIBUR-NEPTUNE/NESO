///////////////////////////////////////////////////////////////////////////////
//
// File: main.cpp
//
//
// Description: Entrypoint for the Benchmark solver.
//
///////////////////////////////////////////////////////////////////////////////
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <iostream>
#include <map>
#include <mpi.h>
#include <string>
using namespace Nektar;
#include "main_evaluation.hpp"
#include "main_projection.hpp"

int main(int argc, char *argv[]) {

  // MPI is initialised/finalised here to ensure that Nektar++ does not
  // initialise/finalise MPI when we were not expecting it to.
  int provided_thread_level;
  if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED,
                      &provided_thread_level) != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Init != MPI_SUCCESS" << std::endl;
    return -1;
  }
  int err = 0;

  LibUtilities::SessionReaderSharedPtr session;
  session = LibUtilities::SessionReader::CreateInstance(argc, argv);
  MeshGraphSharedPtr graph;
  graph = SpatialDomains::MeshGraph::Read(session);

  int benchmark_id = 0;
  if (session->DefinesParameter("benchmark_id")) {
    session->LoadParameter("benchmark_id", benchmark_id);
  }

  switch (benchmark_id) {
  // Evaluation benchmark
  case 0:
    err = main_evaluation(argc, argv, session, graph);
    break;
  // Projection benchmark
  case 1:
    err = main_projection(argc, argv, session, graph);
    break;

  default:
    err = -2;
    break;
  };

  if (MPI_Finalize() != MPI_SUCCESS) {
    std::cout << "ERROR: MPI_Finalize != MPI_SUCCESS" << std::endl;
    return -1;
  }

  return err;
}
