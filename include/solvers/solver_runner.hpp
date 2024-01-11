#ifndef __SOLVER_RUNNER_H_
#define __SOLVER_RUNNER_H_

#include <LibUtilities/BasicUtils/SessionReader.h>
#include <SolverUtils/Driver.h>
#include <SpatialDomains/MeshGraph.h>

using namespace Nektar;

/**
 * Class to abstract setting up sessions and drivers for Nektar++ solvers.
 */
class SolverRunner {
public:
  /// Nektar++ session object.
  LibUtilities::SessionReaderSharedPtr session;
  /// MeshGraph instance for solver.
  SpatialDomains::MeshGraphSharedPtr graph;
  /// The Driver created for the solver.
  SolverUtils::DriverSharedPtr driver;

  /**
   *  Create session, graph and driver from files.
   *
   *  @param argc Number of arguments (like for main).
   *  @param argv Array of char* filenames (like for main).
   */
  SolverRunner(int argc, char **argv) {
    // Create session reader.
    this->session = LibUtilities::SessionReader::CreateInstance(argc, argv);
    // Read the mesh and create a MeshGraph object.
    this->graph = SpatialDomains::MeshGraph::Read(this->session);
    // Create driver.
    std::string driverName;
    session->LoadSolverInfo("Driver", driverName, "Standard");
    this->driver = SolverUtils::GetDriverFactory().CreateInstance(
        driverName, session, graph);
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

#endif
