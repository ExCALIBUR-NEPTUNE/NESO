#ifndef __NESO_SOLVERS_SOLVERRUNNER_HPP__
#define __NESO_SOLVERS_SOLVERRUNNER_HPP__

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>
#include <SpatialDomains/MeshGraph.h>
#include <SpatialDomains/MeshGraphIO.h>

namespace LU = Nektar::LibUtilities;
namespace SD = Nektar::SpatialDomains;
namespace SU = Nektar::SolverUtils;

/**
 * Class to abstract setting up sessions and drivers for Nektar++ solvers.
 */
class SolverRunner {
public:
  /// Nektar++ session object.
  LU::SessionReaderSharedPtr session;
  /// MeshGraph instance for solver.
  SD::MeshGraphSharedPtr graph;
  /// The Driver created for the solver.
  SU::DriverSharedPtr driver;

  /**
   *  Create session, graph and driver from files.
   *
   *  @param argc Number of arguments (like for main).
   *  @param argv Array of char* filenames (like for main).
   */
  SolverRunner(int argc, char **argv) {
    // Create session reader.
    this->session = LU::SessionReader::CreateInstance(argc, argv);
    // Read the mesh and create a MeshGraph object.
    this->graph = SD::MeshGraphIO::Read(this->session);
    // Create driver.
    std::string driverName;
    session->LoadSolverInfo("Driver", driverName, "Standard");
    this->driver =
        SU::GetDriverFactory().CreateInstance(driverName, session, graph);
  }

  /**
   *  Calls Execute on the underlying driver object to run the solver.
   */
  inline void execute() { this->driver->Execute(); }

  /**
   *  Calls Finalise on the underlying session object.
   */
  inline void finalise() { this->session->Finalise(); }
};

/**
 * @brief Run a solver.
 *
 * @param argc number of arguments
 * @param argv argument array
 * @return int non-zero on error
 */
inline int run_solver(int argc, char *argv[]) {

  // Construct a runner instance
  SolverRunner solver_runner(argc, argv);
  // Try-catch and return err code here?

  // Execute the driver
  solver_runner.execute();

  // Finalise MPI, etc.
  solver_runner.finalise();

  return 0;
}

#endif // __NESO_SOLVERS_SOLVERRUNNER_HPP__
