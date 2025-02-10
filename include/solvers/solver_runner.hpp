#ifndef __SOLVER_RUNNER_H_
#define __SOLVER_RUNNER_H_

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

#endif
